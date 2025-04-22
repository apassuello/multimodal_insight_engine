# src/training/__init__.py
"""
Training modules for the MultiModal Insight Engine.

This package contains trainer implementations, loss functions, optimizers,
metrics, and other training utilities.
"""

# Import trainers
from .trainer import train_model
from .multimodal_trainer import MultimodalTrainer
from .transformer_trainer import TransformerTrainer

# Import loss functions
from .loss import (
    ContrastiveLoss,
    MemoryQueueContrastiveLoss,
    DynamicTemperatureContrastiveLoss,
    HardNegativeMiningContrastiveLoss,
    MultiModalMixedContrastiveLoss,
    DecoupledContrastiveLoss,
)
from .loss.loss_factory import create_loss_function

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
