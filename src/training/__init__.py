# src/training/__init__.py
"""
Training modules for the MultiModal Insight Engine.

This package contains trainer implementations, loss functions, optimizers,
metrics, and other training utilities.
"""

from .trainers.multimodal_trainer import MultimodalTrainer

# Import trainers
from .trainers.trainer import train_model
from .trainers.transformer_trainer import TransformerTrainer

# Optional Constitutional AI trainer
try:
    from .trainers.constitutional_trainer import ConstitutionalTrainer
    CONSTITUTIONAL_TRAINER_AVAILABLE = True
except ImportError:
    CONSTITUTIONAL_TRAINER_AVAILABLE = False
    ConstitutionalTrainer = None

# Import loss functions
from .losses import (
    ContrastiveLoss,
    DecoupledContrastiveLoss,
    DynamicTemperatureContrastiveLoss,
    HardNegativeMiningContrastiveLoss,
    MemoryQueueContrastiveLoss,
    MultiModalMixedContrastiveLoss,
)
from .losses.loss_factory import create_loss_function

# Import optimizers
from .optimizers import (
    AdamW,
    CosineAnnealingLR,
    GradientClipper,
    LinearWarmupLR,
    OneCycleLR,
)

__all__ = [
    "train_model",
    "MultimodalTrainer",
    "TransformerTrainer",
    "ConstitutionalTrainer",
    "CONSTITUTIONAL_TRAINER_AVAILABLE",
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
