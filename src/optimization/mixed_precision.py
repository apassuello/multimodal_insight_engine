# src/optimization/mixed_precision.py

"""MODULE: mixed_precision.py
PURPOSE: Converts models to use mixed precision formats for training and inference.
KEY COMPONENTS:
- MixedPrecisionConverter: Converts models to use mixed precision formats.
- MixedPrecisionWrapper: Wrapper for mixed precision inference with autocast.
DEPENDENCIES: torch, typing, logging
SPECIAL NOTES: Supports FP16 and BF16 mixed precision, with special handling for Apple Silicon MPS acceleration."""

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Union, Any, Callable
import logging
import os

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
        # Store the original model state dict instead of creating a new instance
        self.original_state_dict = {k: v.clone() for k, v in model.state_dict().items()}
        
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
        # Convert the state dict to the target dtype where appropriate
        state_dict = self.model.state_dict()
        converted_state_dict = {}
        
        # Define lists of layer types to keep in FP32 for numerical stability
        fp32_layer_types = [
            "layer_norm", "LayerNorm", "norm", 
            "embedding", "Embedding", "embed",
            "bias",  # Bias terms often better in FP32
            "positional"  # Positional encoding/embeddings
        ]
        
        for name, param in state_dict.items():
            # Only convert floating point parameters
            if param.dtype in [torch.float32, torch.float64]:
                # Check if this parameter should remain in FP32
                keep_fp32 = any(layer_type in name.lower() for layer_type in fp32_layer_types)
                
                if keep_fp32:
                    converted_state_dict[name] = param
                else:
                    converted_state_dict[name] = param.to(self.dtype)
            else:
                converted_state_dict[name] = param
        
        # Load the converted state dict back into the original model
        self.model.load_state_dict(converted_state_dict)
        
        # Create a wrapped model with autocast if requested
        if self.use_auto_cast:
            return MixedPrecisionWrapper(self.model, self.dtype)
        else:
            return self.model
    
    def restore_original_precision(self):
        """Restore the model to its original precision."""
        self.model.load_state_dict(self.original_state_dict)


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


def extract_file_metadata(file_path: str = __file__):
    """
    Extract structured metadata about this module.

    Args:
        file_path: Path to the source file (defaults to current file)

    Returns:
        dict: Structured metadata about the module's purpose and components
    """
    return {
        "filename": os.path.basename(file_path),
        "module_purpose": "Converts models to use mixed precision formats for training and inference.",
        "key_classes": [
            {
                "name": "MixedPrecisionConverter",
                "purpose": "Converts models to use mixed precision formats.",
                "key_methods": [
                    {
                        "name": "__init__",
                        "signature": "(self, model: nn.Module, dtype: torch.dtype = torch.float16, use_auto_cast: bool = True)",
                        "brief_description": "Initialize the mixed precision converter."
                    },
                    {
                        "name": "convert_to_mixed_precision",
                        "signature": "(self) -> nn.Module",
                        "brief_description": "Convert the model to use mixed precision."
                    },
                    {
                        "name": "restore_original_precision",
                        "signature": "(self)",
                        "brief_description": "Restore the model to its original precision."
                    }
                ],
                "inheritance": "",
                "dependencies": ["torch", "typing", "logging"]
            },
            {
                "name": "MixedPrecisionWrapper",
                "purpose": "Wrapper for mixed precision inference with autocast.",
                "key_methods": [
                    {
                        "name": "__init__",
                        "signature": "(self, model: nn.Module, dtype: torch.dtype = torch.float16)",
                        "brief_description": "Initialize the mixed precision wrapper."
                    },
                    {
                        "name": "forward",
                        "signature": "(self, *args, **kwargs)",
                        "brief_description": "Forward pass with automatic mixed precision."
                    },
                    {
                        "name": "__getattr__",
                        "signature": "(self, name)",
                        "brief_description": "Delegate attribute access to the wrapped model."
                    }
                ],
                "inheritance": "nn.Module",
                "dependencies": ["torch", "typing", "logging"]
            }
        ],
        "external_dependencies": ["torch", "typing", "logging"],
        "complexity_score": 5,
    }