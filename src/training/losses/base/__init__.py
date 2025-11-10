"""Base classes and mixins for loss functions.

This module provides shared functionality for all loss implementations,
reducing code duplication and establishing consistent patterns.
"""

from .mixins import (
    TemperatureScalingMixin,
    NormalizationMixin,
    ProjectionMixin,
    HardNegativeMiningMixin,
)
from .base_contrastive import BaseContrastiveLoss
from .base_supervised import BaseSupervisedLoss

__all__ = [
    # Mixins
    "TemperatureScalingMixin",
    "NormalizationMixin",
    "ProjectionMixin",
    "HardNegativeMiningMixin",
    # Base classes
    "BaseContrastiveLoss",
    "BaseSupervisedLoss",
]
