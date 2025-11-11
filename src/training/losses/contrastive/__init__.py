"""Contrastive learning losses.

This package contains contrastive loss implementations that leverage
the base classes to eliminate code duplication.
"""

from .clip_loss import CLIPLoss
from .simclr_loss import SimCLRLoss
from .moco_loss import MoCoLoss
from .hard_negative_loss import HardNegativeLoss

__all__ = [
    "CLIPLoss",
    "SimCLRLoss",
    "MoCoLoss",
    "HardNegativeLoss",
]
