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
from .data_handler import DataHandler
from .evaluation import Evaluator
from .metrics_collector import MetricsCollector
from .trainer import ModalityBalancingScheduler, MultimodalTrainer
from .training_loop import TrainingLoop


__all__ = [
    "MultimodalTrainer",
    "ModalityBalancingScheduler",
    "CheckpointManager",
    "MetricsCollector",
    "TrainingLoop",
    "Evaluator",
    "DataHandler",
]
