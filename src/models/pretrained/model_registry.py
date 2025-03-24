# src/models/model_registry.py
from typing import Dict, Type, Any, Optional
import torch.nn as nn

# Import your model wrappers
from .pretrained.vision_transformer import VisionTransformerWrapper
from .pretrained.clip_model import CLIPModelWrapper
# Add more model imports as you expand

class ModelRegistry:
    """Registry pattern for accessing and instantiating models."""
    
    _models = {
        # Homemade models
        'transformer': None,  # Your current implementation
        
        # Pretrained model wrappers
        'vit': VisionTransformerWrapper,
        'clip': CLIPModelWrapper,
        # Add more models as you implement them
    }
    
    @classmethod
    def get_model(cls, model_type: str, **kwargs) -> nn.Module:
        """
        Get a model instance by type.
        
        Args:
            model_type: Type of model to instantiate
            **kwargs: Arguments to pass to the model constructor
            
        Returns:
            Model instance
        """
        if model_type not in cls._models:
            raise ValueError(f"Unknown model type: {model_type}")
            
        model_class = cls._models[model_type]
        return model_class(**kwargs)
    
    @classmethod
    def register_model(cls, model_type: str, model_class: Type[nn.Module]) -> None:
        """
        Register a new model type.
        
        Args:
            model_type: Type name for the model
            model_class: Model class to instantiate
        """
        cls._models[model_type] = model_class