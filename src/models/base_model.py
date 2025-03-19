import os
import torch
import torch.nn as nn
from typing import Dict, Any, Union, Optional

class BaseModel(nn.Module):
    """
    Base class for all models in the MultiModal Insight Engine.
    
    This class extends PyTorch's nn.Module with additional functionality for
    saving/loading models, parameter counting, and other utilities that will be
    common across all models in the project.
    """
    
    def __init__(self):
        """Initialize the base model."""
        super().__init__()
        self.model_type = self.__class__.__name__
    
    def forward(self, x):
        """
        Forward pass of the model.
        
        Args:
            x: Input tensor or dictionary of tensors
            
        Returns:
            Model output
        """
        raise NotImplementedError("Subclasses must implement forward method")
    
    def save(self, path: str, optimizer: Optional[torch.optim.Optimizer] = None,
             epoch: Optional[int] = None, loss: Optional[float] = None,
             additional_info: Optional[Dict[str, Any]] = None):
        """
        Save model weights and training state to a file.
        
        Args:
            path: Path to save the model
            optimizer: Optional optimizer to save state
            epoch: Optional current epoch number
            loss: Optional current loss value
            additional_info: Optional dictionary with additional info to save
        """
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        # Prepare the state dictionary
        state_dict = {
            'model_type': self.model_type,
            'model_state_dict': self.state_dict(),
        }
        
        # Add optional information
        if optimizer is not None:
            state_dict['optimizer_state_dict'] = optimizer.state_dict()
        if epoch is not None:
            state_dict['epoch'] = epoch
        if loss is not None:
            state_dict['loss'] = loss
        if additional_info is not None:
            state_dict.update(additional_info)
        
        # Save the state dictionary
        torch.save(state_dict, path)
        print(f"Model saved to {path}")
    
    def load(self, path: str, map_location: Optional[str] = None):
        """
        Load model weights from a file.
        
        Args:
            path: Path to the saved model
            map_location: Optional device mapping (e.g., 'cpu', 'cuda')
            
        Returns:
            Dictionary containing loaded information besides model weights
        """
        # Load the state dictionary with weights_only=True for better security
        checkpoint = torch.load(path, map_location=map_location, weights_only=True)
        
        # Check if the model type matches
        saved_model_type = checkpoint.get('model_type')
        if saved_model_type != self.model_type:
            print(f"Warning: Loading weights from {saved_model_type} into {self.model_type}")
        
        # Load the model weights
        self.load_state_dict(checkpoint['model_state_dict'])
        print(f"Model loaded from {path}")
        
        # Remove model-related keys and return the rest
        checkpoint.pop('model_type', None)
        checkpoint.pop('model_state_dict', None)
        return checkpoint
    
    def count_parameters(self):
        """
        Count the number of trainable parameters.
        
        Returns:
            int: Number of trainable parameters
        """
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def get_device(self):
        """
        Get the device where the model is currently located.
        
        Returns:
            torch.device: Device of the first parameter
        """
        return next(self.parameters()).device