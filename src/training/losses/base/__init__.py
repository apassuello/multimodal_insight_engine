"""Base classes and mixins for loss functions.

This module provides shared functionality for all loss implementations,
reducing code duplication and establishing consistent patterns.
"""

from .base_contrastive import BaseContrastiveLoss
from .base_supervised import BaseSupervisedLoss
from .mixins import (
    HardNegativeMiningMixin,
    NormalizationMixin,
    ProjectionMixin,
    TemperatureScalingMixin,
)


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
