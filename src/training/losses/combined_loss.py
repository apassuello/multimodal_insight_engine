# src/training/loss/combined_loss.py

"""
Combined loss functions for multimodal training.

This module provides adapter classes for combining multiple loss functions
with flexible weighting and compatibility with the MultimodalTrainer.
"""

import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Optional, Callable, Union, Any


class CombinedLoss(nn.Module):
    """
    Wrapper class for combining multiple loss functions with weights.
    
    This adapter enables combining primary and secondary loss functions
    with configurable weights for advanced training strategies.
    """
    
    def __init__(
        self,
        primary_loss: nn.Module,
        secondary_loss: Optional[nn.Module] = None,
        secondary_loss_weight: float = 0.5,
        tertiary_loss: Optional[nn.Module] = None,
        tertiary_loss_weight: float = 0.3,
    ):
        """
        Initialize combined loss function.
        
        Args:
            primary_loss: Main loss function
            secondary_loss: Optional secondary loss function
            secondary_loss_weight: Weight for secondary loss (0-1)
            tertiary_loss: Optional tertiary loss function
            tertiary_loss_weight: Weight for tertiary loss (0-1)
        """
        super().__init__()
        self.primary_loss = primary_loss
        self.secondary_loss = secondary_loss
        self.secondary_loss_weight = secondary_loss_weight
        self.tertiary_loss = tertiary_loss
        self.tertiary_loss_weight = tertiary_loss_weight
        
    def forward(self, *args, **kwargs) -> Dict[str, torch.Tensor]:
        """
        Compute combined loss.
        
        Args:
            *args, **kwargs: Arguments to pass to loss functions
            
        Returns:
            Dictionary with combined loss and individual loss components
        """
        # Compute primary loss
        primary_output = self.primary_loss(*args, **kwargs)
        
        # Extract primary loss value
        if isinstance(primary_output, dict):
            primary_loss_value = primary_output["loss"]
        else:
            primary_loss_value = primary_output
            primary_output = {"loss": primary_loss_value}
            
        # Initialize combined loss with primary loss
        combined_loss = primary_loss_value
        combined_output = {
            **primary_output,
            "primary_loss": primary_loss_value.item() if isinstance(primary_loss_value, torch.Tensor) else primary_loss_value,
        }
        
        # Add secondary loss if provided
        if self.secondary_loss is not None:
            secondary_output = self.secondary_loss(*args, **kwargs)
            
            # Extract secondary loss value
            if isinstance(secondary_output, dict):
                secondary_loss_value = secondary_output["loss"]
            else:
                secondary_loss_value = secondary_output
                secondary_output = {"loss": secondary_loss_value}
            
            # Add to combined loss
            combined_loss = combined_loss + self.secondary_loss_weight * secondary_loss_value
            
            # Add secondary loss components to output
            combined_output["secondary_loss"] = secondary_loss_value.item() if isinstance(secondary_loss_value, torch.Tensor) else secondary_loss_value
            combined_output["secondary_weight"] = self.secondary_loss_weight
            
            # Add other secondary metrics to output with prefix
            for key, value in secondary_output.items():
                if key != "loss":
                    combined_output[f"secondary_{key}"] = value
        
        # Add tertiary loss if provided
        if self.tertiary_loss is not None:
            tertiary_output = self.tertiary_loss(*args, **kwargs)
            
            # Extract tertiary loss value
            if isinstance(tertiary_output, dict):
                tertiary_loss_value = tertiary_output["loss"]
            else:
                tertiary_loss_value = tertiary_output
                tertiary_output = {"loss": tertiary_loss_value}
            
            # Add to combined loss
            combined_loss = combined_loss + self.tertiary_loss_weight * tertiary_loss_value
            
            # Add tertiary loss components to output
            combined_output["tertiary_loss"] = tertiary_loss_value.item() if isinstance(tertiary_loss_value, torch.Tensor) else tertiary_loss_value
            combined_output["tertiary_weight"] = self.tertiary_loss_weight
            
            # Add other tertiary metrics to output with prefix
            for key, value in tertiary_output.items():
                if key != "loss":
                    combined_output[f"tertiary_{key}"] = value
        
        # Update combined loss in output
        combined_output["loss"] = combined_loss
        
        return combined_output
    
    def train(self, mode: bool = True):
        """
        Set the module in training mode.
        
        Args:
            mode: Whether to set training mode (True) or evaluation mode (False)
            
        Returns:
            self
        """
        # Set training mode for all loss functions
        super().train(mode)
        if hasattr(self.primary_loss, "train"):
            self.primary_loss.train(mode)
        if self.secondary_loss is not None and hasattr(self.secondary_loss, "train"):
            self.secondary_loss.train(mode)
        if self.tertiary_loss is not None and hasattr(self.tertiary_loss, "train"):
            self.tertiary_loss.train(mode)
        return self
    
    def eval(self):
        """
        Set the module in evaluation mode.
        
        Returns:
            self
        """
        return self.train(False)