# src/training/__init__.py
"""
Training modules for the MultiModal Insight Engine.

This package contains trainer implementations, loss functions, optimizers,
metrics, and other training utilities.
"""

# Import trainers
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
from .optimizers import AdamW, CosineAnnealingLR, GradientClipper, LinearWarmupLR, OneCycleLR
from .trainers.multimodal import MultimodalTrainer
from .trainers.trainer import train_model
from .trainers.transformer_trainer import TransformerTrainer


# NOTE: Constitutional AI trainer has been extracted to standalone repository
# Location: extracted/constitutional-ai/
# Standalone repo: /Users/apa/ml_projects/constitutional-ai
# For Constitutional AI training functionality, use the standalone repository
CONSTITUTIONAL_TRAINER_AVAILABLE = False
ConstitutionalTrainer = None


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
