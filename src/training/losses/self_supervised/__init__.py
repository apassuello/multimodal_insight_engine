"""Self-supervised learning losses.

This package contains self-supervised loss implementations including VICReg,
Barlow Twins, and other methods that don't require labels.
"""

from .vicreg_loss import VICRegLoss
from .barlow_twins_loss import BarlowTwinsLoss

__all__ = [
    "VICRegLoss",
    "BarlowTwinsLoss",
]
