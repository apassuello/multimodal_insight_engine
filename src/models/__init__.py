# src/models/__init__.py
"""
Model implementations for the MultiModal Insight Engine.

This package contains modules for creating, loading, and managing 
various neural network model architectures.
"""

from .model_factory import create_multimodal_model

# Import components
from .base_model import BaseModel
from .transformer import Transformer
from .feed_forward import FeedForwardNN, MultiLayerPerceptron
from .attention import MultiHeadAttention
from .embeddings import TokenEmbedding

# Import pretrained model wrappers
from .pretrained import (
    HuggingFaceTextModelWrapper,
    VisionTransformerWrapper,
    DimensionMatchingWrapper
)

__all__ = [
    "create_multimodal_model",
    "BaseModel",
    "Transformer",
    "FeedForwardNN", 
    "MultiLayerPerceptron",
    "MultiHeadAttention",
    "TokenEmbedding",
    "HuggingFaceTextModelWrapper",
    "VisionTransformerWrapper",
    "DimensionMatchingWrapper"
]