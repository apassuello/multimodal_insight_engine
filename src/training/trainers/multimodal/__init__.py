"""Multimodal trainer components.

This package contains the decomposed multimodal trainer implementation,
split into focused modules following the Single Responsibility Principle.
"""

from .checkpoint_manager import CheckpointManager
from .metrics_collector import MetricsCollector

__all__ = [
    "CheckpointManager",
    "MetricsCollector",
]
