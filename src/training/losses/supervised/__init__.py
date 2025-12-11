"""Supervised losses.

This package contains supervised loss implementations that use explicit
labels for training, leveraging BaseSupervisedLoss to eliminate duplication.
"""

from .supervised_loss import SupervisedContrastiveLoss

__all__ = [
    "SupervisedContrastiveLoss",
]
