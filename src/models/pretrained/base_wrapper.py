# src/models/pretrained/base_wrapper.py
import torch
import torch.nn as nn
from typing import Dict, Any, Optional

class PretrainedModelWrapper(nn.Module):
    """Base wrapper for pretrained models providing a consistent interface."""
    
    def __init__(self, model_name: str = None, model: nn.Module = None):
        super().__init__()
        self.model_name = model_name
        self.pretrained_model = model
        self.config = {}
        
    def load_model(self, model_name: str) -> None:
        """Load a pretrained model - to be implemented by subclasses."""
        raise NotImplementedError
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass - to be implemented by subclasses."""
        raise NotImplementedError
    
    def save(self, path: str) -> None:
        """Save wrapper configuration and model weights."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        # Prepare state dict
        state_dict = {
            'model_name': self.model_name,
            'config': self.config,
            'model_state_dict': self.pretrained_model.state_dict(),
        }
        
        torch.save(state_dict, path)
        
    def load(self, path: str, map_location: Optional[str] = None) -> Dict[str, Any]:
        """Load wrapper configuration and model weights."""
        checkpoint = torch.load(path, map_location=map_location)
        
        self.model_name = checkpoint.get('model_name', self.model_name)
        self.config = checkpoint.get('config', {})
        
        # If model hasn't been loaded yet, load it
        if self.pretrained_model is None:
            self.load_model(self.model_name)
            
        # Load weights
        self.pretrained_model.load_state_dict(checkpoint['model_state_dict'])
        
        return checkpoint