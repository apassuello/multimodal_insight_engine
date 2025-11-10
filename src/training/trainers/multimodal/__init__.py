"""Multimodal trainer components.

This package contains the decomposed multimodal trainer implementation,
split into focused modules following the Single Responsibility Principle.
"""

from .checkpoint_manager import CheckpointManager
from .metrics_collector import MetricsCollector
from .training_loop import TrainingLoop

__all__ = [
    "CheckpointManager",
    "MetricsCollector",
    "TrainingLoop",
]
