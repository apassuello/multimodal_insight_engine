# src/models/__init__.py
"""
Model implementations for the MultiModal Insight Engine.

This package contains modules for creating, loading, and managing 
various neural network model architectures.
"""

from .attention import MultiHeadAttention

# Import components
from .base_model import BaseModel
from .embeddings import TokenEmbedding
from .feed_forward import FeedForwardNN, MultiLayerPerceptron
from .model_factory import create_multimodal_model

# Import pretrained model wrappers
from .pretrained import (
    DimensionMatchingWrapper,
    HuggingFaceTextModelWrapper,
    VisionTransformerWrapper,
)
from .transformer import Transformer

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
