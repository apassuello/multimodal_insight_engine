# src/training/__init__.py
"""
Training modules for the MultiModal Insight Engine.

This package contains trainer implementations, loss functions, optimizers,
metrics, and other training utilities.
"""

# Import trainers
from .trainers.trainer import train_model
from .trainers.multimodal_trainer import MultimodalTrainer
from .trainers.transformer_trainer import TransformerTrainer

# Import loss functions
from .losses import (
    ContrastiveLoss,
    MemoryQueueContrastiveLoss,
    DynamicTemperatureContrastiveLoss,
    HardNegativeMiningContrastiveLoss,
    MultiModalMixedContrastiveLoss,
    DecoupledContrastiveLoss,
)
from .losses.loss_factory import create_loss_function

# Import optimizers
from .optimizers import (
    AdamW,
    OneCycleLR,
    CosineAnnealingLR,
    LinearWarmupLR,
    GradientClipper,
)

__all__ = [
    "train_model",
    "MultimodalTrainer",
    "TransformerTrainer",
    "ContrastiveLoss",
    "MemoryQueueContrastiveLoss",
    "DynamicTemperatureContrastiveLoss",
    "HardNegativeMiningContrastiveLoss",
    "MultiModalMixedContrastiveLoss",
    "DecoupledContrastiveLoss",
    "create_loss_function",
    "AdamW",
    "OneCycleLR",
    "CosineAnnealingLR",
    "LinearWarmupLR",
    "GradientClipper",
]
