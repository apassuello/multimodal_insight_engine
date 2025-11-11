"""Contrastive learning losses.

This package contains contrastive loss implementations that leverage
the base classes to eliminate code duplication.
"""

from .clip_loss import CLIPLoss

__all__ = [
    "CLIPLoss",
]
