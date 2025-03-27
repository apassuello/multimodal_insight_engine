# src/optimization/mixed_precision.py
import torch
import torch.nn as nn
from typing import Dict, List, Optional, Union, Any, Callable
import logging

class MixedPrecisionConverter:
    """
    Converts models to use mixed precision formats for training and inference.
    
    This class provides utilities for using FP16 or BF16 mixed precision, with
    special handling for Apple Silicon MPS acceleration.
    """
    
    def __init__(
        self,
        model: nn.Module,
        dtype: torch.dtype = torch.float16,
        use_auto_cast: bool = True,
    ):
        """
        Initialize the mixed precision converter.
        
        Args:
            model: The model to convert
            dtype: Target data type (torch.float16 or torch.bfloat16)
            use_auto_cast: Whether to use automatic mixed precision via autocast
        """
        self.model = model
        self.original_model = type(model)()
        self.original_model.load_state_dict(model.state_dict())
        
        self.dtype = dtype
        self.use_auto_cast = use_auto_cast
        
        # Check if the hardware supports the requested dtype
        self._check_hardware_support()
    
    def _check_hardware_support(self):
        """Check if the current hardware supports the requested dtype."""
        device = next(self.model.parameters()).device
        
        if self.dtype == torch.bfloat16:
            if device.type == "cuda" and not torch.cuda.is_bf16_supported():
                logging.warning("BFloat16 not supported on this GPU. Falling back to FP16.")
                self.dtype = torch.float16
            elif device.type == "mps":
                logging.warning("BFloat16 not fully supported on MPS. Falling back to FP16.")
                self.dtype = torch.float16
        
        # For Apple Silicon, ensure we use the most efficient format
        if device.type == "mps" and self.dtype == torch.float16:
            logging.info("Using FP16 with MPS acceleration on Apple Silicon.")
    
    def convert_to_mixed_precision(self) -> nn.Module:
        """
        Convert the model to use mixed precision.
        
        Returns:
            Model with mixed precision
        """
        # Create a copy of the model with mixed precision weights
        mp_model = type(self.model)()
        
        # Convert the state dict to the target dtype where appropriate
        state_dict = self.model.state_dict()
        converted_state_dict = {}
        
        for name, param in state_dict.items():
            # Only convert floating point parameters
            if param.dtype in [torch.float32, torch.float64]:
                # LayerNorm and embedding parameters often work better in FP32
                if any(layer_type in name for layer_type in ["layer_norm", "LayerNorm", "embedding"]):
                    converted_state_dict[name] = param
                else:
                    converted_state_dict[name] = param.to(self.dtype)
            else:
                converted_state_dict[name] = param
        
        # Load the converted state dict
        mp_model.load_state_dict(converted_state_dict)
        
        # Create a wrapped model with autocast if requested
        if self.use_auto_cast:
            return MixedPrecisionWrapper(mp_model, self.dtype)
        else:
            return mp_model
    
    def restore_original_precision(self):
        """Restore the model to its original precision."""
        self.model.load_state_dict(self.original_model.state_dict())


class MixedPrecisionWrapper(nn.Module):
    """
    Wrapper for mixed precision inference with autocast.
    
    This wrapper automatically applies torch.autocast around the forward method
    to ensure mixed precision is used during inference.
    """
    
    def __init__(self, model: nn.Module, dtype: torch.dtype = torch.float16):
        """
        Initialize the mixed precision wrapper.
        
        Args:
            model: The model to wrap
            dtype: Data type to use for mixed precision
        """
        super().__init__()
        self.model = model
        self.dtype = dtype
        
        # Determine device type for autocast
        self.device_type = next(model.parameters()).device.type
        if self.device_type == "mps":
            # MPS support for autocast is relatively new
            logging.info("Using MPS-enabled autocast for mixed precision.")
    
    def forward(self, *args, **kwargs):
        """
        Forward pass with automatic mixed precision.
        
        Args:
            *args, **kwargs: Arguments to pass to the wrapped model
            
        Returns:
            Model outputs
        """
        with torch.autocast(device_type=self.device_type, dtype=self.dtype):
            return self.model(*args, **kwargs)
    
    def __getattr__(self, name):
        """Delegate attribute access to the wrapped model."""
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.model, name)