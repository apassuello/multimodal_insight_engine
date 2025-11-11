"""Multimodal loss functions.

This package contains loss implementations for multimodal learning
that leverage the base classes to eliminate code duplication.
"""

from .mixed_loss import MixedMultimodalLoss

__all__ = [
    "MixedMultimodalLoss",
]
