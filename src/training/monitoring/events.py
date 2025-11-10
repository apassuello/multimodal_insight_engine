"""MODULE: events.py
PURPOSE: Training event definitions for monitoring system
KEY COMPONENTS:
- TrainingPhase: Enum of training lifecycle phases
- TrainingEvent: Immutable event data passed to callbacks
DEPENDENCIES: dataclasses, enum, typing, time
SPECIAL NOTES: Uses frozen dataclasses with slots for memory efficiency
"""

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Dict
import time


class TrainingPhase(Enum):
    """
    Training lifecycle phases for event tracking.

    Events are emitted at these key points during training to allow
    callbacks to observe and react to training progress.
    """
    INITIALIZATION = auto()
    TRAINING_START = auto()
    EPOCH_START = auto()
    ITERATION_START = auto()
    RESPONSE_GENERATED = auto()
    REWARD_COMPUTED = auto()
    POLICY_UPDATE = auto()
    VALUE_UPDATE = auto()
    ITERATION_END = auto()
    EPOCH_END = auto()
    TRAINING_END = auto()


@dataclass(frozen=True, slots=True)
class TrainingEvent:
    """
    Immutable training event passed to callbacks.

    Design: Frozen dataclass with slots for memory efficiency and immutability.
    Zero-copy sharing between callbacks.

    Attributes:
        phase: Current training phase
        iteration: Current iteration number (0-indexed)
        epoch: Current epoch number (0-indexed, -1 if N/A)
        metrics: Numerical metrics (loss, reward, etc.)
        metadata: Additional context (responses, prompts, etc.)
        timestamp: Event creation time (seconds since epoch)
    """
    phase: TrainingPhase
    iteration: int
    metrics: Dict[str, float] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    epoch: int = -1
    timestamp: float = field(default_factory=time.time)

    def get_metric(self, name: str, default: float = 0.0) -> float:
        """Get a metric value with optional default."""
        return self.metrics.get(name, default)

    def has_metric(self, name: str) -> bool:
        """Check if a metric exists."""
        return name in self.metrics

    def get_metadata(self, key: str, default: Any = None) -> Any:
        """Get metadata value with optional default."""
        return self.metadata.get(key, default)
