"""Contrastive learning losses.

This package contains contrastive loss implementations that leverage
the base classes to eliminate code duplication.
"""

from .clip_loss import CLIPLoss
from .decoupled_loss import DecoupledLoss
from .dynamic_temperature_loss import DynamicTemperatureLoss
from .hard_negative_loss import HardNegativeLoss
from .moco_loss import MoCoLoss
from .simclr_loss import SimCLRLoss

__all__ = [
    "CLIPLoss",
    "SimCLRLoss",
    "MoCoLoss",
    "HardNegativeLoss",
    "DynamicTemperatureLoss",
    "DecoupledLoss",
]
