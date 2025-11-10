# src/training/__init__.py
"""
Training modules for the MultiModal Insight Engine.

This package contains trainer implementations, loss functions, optimizers,
metrics, and other training utilities.
"""

# Lazy imports to avoid pulling in heavy dependencies during testing
_lazy_imports = {
    "MultimodalTrainer": ".trainers.multimodal_trainer",
    "train_model": ".trainers.trainer",
    "TransformerTrainer": ".trainers.transformer_trainer",
    "ConstitutionalTrainer": ".trainers.constitutional_trainer",
    "ContrastiveLoss": ".losses",
    "DecoupledContrastiveLoss": ".losses",
    "DynamicTemperatureContrastiveLoss": ".losses",
    "HardNegativeMiningContrastiveLoss": ".losses",
    "MemoryQueueContrastiveLoss": ".losses",
    "MultiModalMixedContrastiveLoss": ".losses",
    "create_loss_function": ".losses.loss_factory",
    "AdamW": ".optimizers",
    "CosineAnnealingLR": ".optimizers",
    "GradientClipper": ".optimizers",
    "LinearWarmupLR": ".optimizers",
    "OneCycleLR": ".optimizers",
}

CONSTITUTIONAL_TRAINER_AVAILABLE = True  # Set optimistically, will check on access

def __getattr__(name):
    """Lazy import on first access."""
    if name in _lazy_imports:
        import importlib
        module_path = _lazy_imports[name]
        try:
            module = importlib.import_module(module_path, __package__)
            obj = getattr(module, name)
            globals()[name] = obj
            return obj
        except (ImportError, AttributeError):
            if name == "ConstitutionalTrainer":
                globals()["CONSTITUTIONAL_TRAINER_AVAILABLE"] = False
                return None
            raise
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

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
