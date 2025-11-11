"""Wrapper losses for combining multiple loss functions.

This package contains wrapper losses that combine other loss functions with
configurable weighting strategies.
"""

from .combined_loss import CombinedLoss
from .multitask_loss import MultitaskLoss

__all__ = [
    "CombinedLoss",
    "MultitaskLoss",
]
