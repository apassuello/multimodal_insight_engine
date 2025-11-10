"""Multimodal trainer components.

This package contains the decomposed multimodal trainer implementation,
split into focused modules following the Single Responsibility Principle.

Main Components:
- MultimodalTrainer: Main orchestrator for end-to-end training
- CheckpointManager: Handles model checkpointing and state persistence
- MetricsCollector: Tracks and visualizes training metrics
- TrainingLoop: Executes the core training loop
- Evaluator: Computes evaluation metrics
- DataHandler: Manages data preprocessing and device placement
- ModalityBalancingScheduler: Balances learning rates between modalities
"""

from .checkpoint_manager import CheckpointManager
from .metrics_collector import MetricsCollector
from .training_loop import TrainingLoop
from .evaluation import Evaluator
from .data_handler import DataHandler
from .trainer import MultimodalTrainer, ModalityBalancingScheduler

__all__ = [
    "MultimodalTrainer",
    "ModalityBalancingScheduler",
    "CheckpointManager",
    "MetricsCollector",
    "TrainingLoop",
    "Evaluator",
    "DataHandler",
]
