# src/models/model_registry.py
import os
from typing import Type

import torch.nn as nn

from .clip_model import CLIPModelWrapper

# Import your model wrappers
from .vision_transformer import VisionTransformerWrapper

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

def extract_file_metadata(file_path=__file__):
    """
    Extract structured metadata about this module.
    
    Args:
        file_path: Path to the source file (defaults to current file)
        
    Returns:
        dict: Structured metadata about the module's purpose and components
    """
    return {
        "filename": os.path.basename(file_path),
        "module_purpose": "Implements a registry pattern for accessing and instantiating pretrained models",
        "key_classes": [
            {
                "name": "ModelRegistry",
                "purpose": "Registry that provides centralized access to all available models",
                "key_methods": [
                    {
                        "name": "get_model",
                        "signature": "get_model(cls, model_type: str, **kwargs) -> nn.Module",
                        "brief_description": "Instantiate a model by type name with optional configuration"
                    },
                    {
                        "name": "register_model",
                        "signature": "register_model(cls, model_type: str, model_class: Type[nn.Module]) -> None",
                        "brief_description": "Register a new model type in the registry"
                    }
                ],
                "inheritance": "object",
                "dependencies": ["torch.nn", ".vision_transformer", ".clip_model"]
            }
        ],
        "external_dependencies": ["torch"],
        "complexity_score": 3  # Moderate complexity for a registry pattern
    }
