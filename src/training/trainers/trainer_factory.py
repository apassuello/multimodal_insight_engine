# src/training/trainers/trainer_factory.py

import torch
import torch.nn as nn
from typing import Dict, Optional, Any, Union, List, Type
import logging
import os

# Import trainers
from src.training.trainers.multimodal import MultimodalTrainer
from src.training.trainers.multistage_trainer import MultistageTrainer

# Import strategies
from src.training.strategies.training_strategy import TrainingStrategy
from src.training.strategies.single_modality_strategy import SingleModalityStrategy
from src.training.strategies.cross_modal_strategy import CrossModalStrategy
from src.training.strategies.end_to_end_strategy import EndToEndStrategy

# Import utilities
from src.utils.learningrate_scheduler import (
    WarmupCosineScheduler,
    LinearWarmupScheduler,
)

logger = logging.getLogger(__name__)

"""
MODULE: trainer_factory.py
PURPOSE: Factory for creating trainers based on configuration with appropriate strategies
KEY COMPONENTS:
- TrainerFactory: Static factory class for creating trainer instances with appropriate configuration
DEPENDENCIES: torch, torch.nn, typing, logging, os, MultimodalTrainer, MultistageTrainer, training strategies
SPECIAL NOTES: Creates trainers with appropriate configuration for different training approaches
"""


class TrainerFactory:
    """
    Factory for creating trainers based on configuration.

    This factory:
    1. Creates appropriate trainer instances based on configuration
    2. Configures training strategies for multistage training
    3. Sets up training parameters like learning rates, batch sizes, etc.
    4. Provides reasonable defaults for common configurations
    """

    @staticmethod
    def create_trainer(
        model: nn.Module,
        train_dataloader: torch.utils.data.DataLoader,
        val_dataloader: Optional[torch.utils.data.DataLoader] = None,
        test_dataloader: Optional[torch.utils.data.DataLoader] = None,
        config: Optional[Dict[str, Any]] = None,
        device: Optional[torch.device] = None,
        **kwargs
    ) -> Union[MultimodalTrainer, MultistageTrainer]:
        """
        Create a trainer based on configuration.

        Args:
            model: The model to train
            train_dataloader: DataLoader for training data
            val_dataloader: DataLoader for validation data
            test_dataloader: DataLoader for test data
            config: Configuration dictionary
            device: Device to train on
            **kwargs: Additional keyword arguments

        Returns:
            An appropriate trainer instance
        """
        # Merge config with kwargs
        if config is None:
            config = {}

        full_config = {**config, **kwargs}

        # Determine trainer type
        trainer_type = full_config.get("trainer_type", "multimodal").lower()

        # Create appropriate trainer
        if trainer_type == "multistage":
            return TrainerFactory._create_multistage_trainer(
                model,
                train_dataloader,
                val_dataloader,
                test_dataloader,
                full_config,
                device,
            )
        else:
            # Default to standard multimodal trainer
            return TrainerFactory._create_multimodal_trainer(
                model,
                train_dataloader,
                val_dataloader,
                test_dataloader,
                full_config,
                device,
            )

    @staticmethod
    def _create_multimodal_trainer(
        model: nn.Module,
        train_dataloader: torch.utils.data.DataLoader,
        val_dataloader: Optional[torch.utils.data.DataLoader],
        test_dataloader: Optional[torch.utils.data.DataLoader],
        config: Dict[str, Any],
        device: Optional[torch.device],
    ) -> MultimodalTrainer:
        """
        Create a standard multimodal trainer.

        Args:
            model: The model to train
            train_dataloader: DataLoader for training data
            val_dataloader: DataLoader for validation data
            test_dataloader: DataLoader for test data
            config: Configuration dictionary
            device: Device to train on

        Returns:
            Configured MultimodalTrainer instance
        """
        # Extract configuration with defaults
        num_epochs = config.get("num_epochs", 20)
        learning_rate = config.get("learning_rate", 1e-4)
        weight_decay = config.get("weight_decay", 0.01)
        warmup_steps = config.get("warmup_steps", 500)
        checkpoint_dir = config.get("checkpoint_dir", "checkpoints")
        log_dir = config.get("log_dir", "logs")
        mixed_precision = config.get("mixed_precision", False)
        accumulation_steps = config.get("accumulation_steps", 1)
        evaluation_steps = config.get("evaluation_steps", 0)
        log_steps = config.get("log_steps", 50)
        early_stopping_patience = config.get("early_stopping_patience", None)
        clip_grad_norm = config.get("clip_grad_norm", None)
        balance_modality_gradients = config.get("balance_modality_gradients", False)

        # Create directories
        os.makedirs(checkpoint_dir, exist_ok=True)
        os.makedirs(log_dir, exist_ok=True)

        # Create optimizer if needed
        optimizer = None
        if "optimizer" not in config:
            # Extract optimizer configuration
            optimizer_type = config.get("optimizer_type", "adamw").lower()

            # Create parameter groups based on configuration
            param_groups = TrainerFactory._create_parameter_groups(model, config)

            # Create optimizer
            if optimizer_type == "adamw":
                optimizer = torch.optim.AdamW(
                    param_groups,
                    lr=learning_rate,
                    weight_decay=weight_decay,
                    betas=(0.9, 0.98),
                    eps=1e-6,
                )
            elif optimizer_type == "adam":
                optimizer = torch.optim.Adam(
                    param_groups,
                    lr=learning_rate,
                    weight_decay=weight_decay,
                    betas=(0.9, 0.999),
                    eps=1e-8,
                )
            elif optimizer_type == "sgd":
                optimizer = torch.optim.SGD(
                    param_groups,
                    lr=learning_rate,
                    momentum=config.get("momentum", 0.9),
                    weight_decay=weight_decay,
                )
        else:
            optimizer = config["optimizer"]

        # Create scheduler if needed
        scheduler = None
        if "scheduler" not in config and optimizer is not None:
            # Extract scheduler configuration
            scheduler_type = config.get("scheduler_type", "warmup_cosine").lower()

            # Calculate total steps if not provided
            if "total_steps" not in config:
                # Estimate based on epochs, dataset size, and batch size
                steps_per_epoch = len(train_dataloader)
                total_steps = steps_per_epoch * num_epochs
            else:
                total_steps = config["total_steps"]

            # Create scheduler
            if scheduler_type == "warmup_cosine":
                scheduler = WarmupCosineScheduler(
                    optimizer=optimizer,
                    warmup_steps=warmup_steps,
                    total_steps=total_steps,
                    min_lr=learning_rate * 0.01,
                )
            elif scheduler_type == "linear_warmup":
                scheduler = LinearWarmupScheduler(
                    optimizer=optimizer,
                    warmup_epochs=config.get("warmup_epochs", 5),
                    total_epochs=num_epochs,
                    init_lr=learning_rate * 0.1,
                    final_lr=learning_rate * 0.01,
                )
            elif scheduler_type == "step":
                scheduler = torch.optim.lr_scheduler.StepLR(
                    optimizer,
                    step_size=config.get("step_size", 10),
                    gamma=config.get("gamma", 0.1),
                )
        else:
            scheduler = config.get("scheduler", None)

        # Create loss function if needed
        loss_fn = config.get("loss_fn", None)

        # Create trainer
        trainer = MultimodalTrainer(
            model=model,
            train_dataloader=train_dataloader,
            val_dataloader=val_dataloader,
            test_dataloader=test_dataloader,
            optimizer=optimizer,
            scheduler=scheduler,
            loss_fn=loss_fn,
            num_epochs=num_epochs,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            warmup_steps=warmup_steps,
            checkpoint_dir=checkpoint_dir,
            log_dir=log_dir,
            device=device,
            mixed_precision=mixed_precision,
            accumulation_steps=accumulation_steps,
            evaluation_steps=evaluation_steps,
            log_steps=log_steps,
            early_stopping_patience=early_stopping_patience,
            clip_grad_norm=clip_grad_norm,
            balance_modality_gradients=balance_modality_gradients,
            args=config,
        )

        return trainer

    @staticmethod
    def _create_multistage_trainer(
        model: nn.Module,
        train_dataloader: torch.utils.data.DataLoader,
        val_dataloader: Optional[torch.utils.data.DataLoader],
        test_dataloader: Optional[torch.utils.data.DataLoader],
        config: Dict[str, Any],
        device: Optional[torch.device],
    ) -> MultistageTrainer:
        """
        Create a multistage trainer with appropriate strategies.

        Args:
            model: The model to train
            train_dataloader: DataLoader for training data
            val_dataloader: DataLoader for validation data
            test_dataloader: DataLoader for test data
            config: Configuration dictionary
            device: Device to train on

        Returns:
            Configured MultistageTrainer instance
        """
        # Extract configuration with defaults
        num_epochs = config.get("num_epochs", 20)
        checkpoint_dir = config.get("checkpoint_dir", "checkpoints")
        log_dir = config.get("log_dir", "logs")

        # Create directories
        os.makedirs(checkpoint_dir, exist_ok=True)
        os.makedirs(log_dir, exist_ok=True)

        # Create strategies for each stage
        strategies = {}

        # Get stage-specific settings
        stage_configs = config.get("stages", {})

        # Configure stage 1: Single Modality
        stage1_config = stage_configs.get("stage1", {})
        stage1_epochs = stage1_config.get("epochs", num_epochs // 3)
        stage1_strategy = SingleModalityStrategy(
            model=model, device=device, **stage1_config
        )
        strategies["stage1"] = {"strategy": stage1_strategy, "epochs": stage1_epochs}

        # Configure stage 2: Cross-Modal Fusion
        stage2_config = stage_configs.get("stage2", {})
        stage2_epochs = stage2_config.get("epochs", num_epochs // 3)
        stage2_strategy = CrossModalStrategy(
            model=model, device=device, **stage2_config
        )
        strategies["stage2"] = {"strategy": stage2_strategy, "epochs": stage2_epochs}

        # Configure stage 3: End-to-End Fine-tuning
        stage3_config = stage_configs.get("stage3", {})
        stage3_epochs = stage3_config.get("epochs", num_epochs // 3)
        stage3_strategy = EndToEndStrategy(model=model, device=device, **stage3_config)
        strategies["stage3"] = {"strategy": stage3_strategy, "epochs": stage3_epochs}

        # Create MultistageTrainer
        trainer = MultistageTrainer(
            model=model,
            train_dataloader=train_dataloader,
            val_dataloader=val_dataloader,
            test_dataloader=test_dataloader,
            strategies=strategies,
            checkpoint_dir=checkpoint_dir,
            log_dir=log_dir,
            device=device,
            **config
        )

        return trainer

    @staticmethod
    def _create_parameter_groups(
        model: nn.Module, config: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """
        Create parameter groups with different learning rates.

        Args:
            model: The model to train
            config: Configuration dictionary

        Returns:
            List of parameter groups for optimizer
        """
        # Extract configuration
        base_lr = config.get("learning_rate", 1e-4)
        layer_specific_lrs = config.get("layer_specific_lrs", True)

        if not layer_specific_lrs:
            # Return all parameters with the same learning rate
            return [{"params": model.parameters(), "lr": base_lr}]

        # Create layer-wise parameter groups
        param_groups = [
            # Vision base model: very low learning rate
            {
                "params": [
                    p
                    for n, p in model.named_parameters()
                    if "vision_model" in n and p.requires_grad
                ],
                "lr": base_lr * config.get("vision_model_lr_factor", 0.01),
                "name": "vision_model",
            },
            # Text base model: very low learning rate
            {
                "params": [
                    p
                    for n, p in model.named_parameters()
                    if "text_model" in n and p.requires_grad
                ],
                "lr": base_lr * config.get("text_model_lr_factor", 0.01),
                "name": "text_model",
            },
            # Fusion components: medium learning rate
            {
                "params": [
                    p
                    for n, p in model.named_parameters()
                    if any(x in n for x in ["fusion", "cross_attention"])
                    and p.requires_grad
                ],
                "lr": base_lr * config.get("fusion_lr_factor", 0.1),
                "name": "fusion_components",
            },
            # Projection layers: full learning rate
            {
                "params": [
                    p
                    for n, p in model.named_parameters()
                    if any(x in n for x in ["projection", "adapter"])
                    and p.requires_grad
                ],
                "lr": base_lr,
                "name": "projection_layers",
            },
            # Other parameters: default learning rate
            {
                "params": [
                    p
                    for n, p in model.named_parameters()
                    if not any(
                        x in n
                        for x in [
                            "vision_model",
                            "text_model",
                            "fusion",
                            "cross_attention",
                            "projection",
                            "adapter",
                        ]
                    )
                    and p.requires_grad
                ],
                "lr": base_lr * config.get("other_lr_factor", 0.1),
                "name": "other",
            },
        ]

        # Remove empty groups
        param_groups = [g for g in param_groups if len(g["params"]) > 0]

        return param_groups


def extract_file_metadata(file_path=__file__):
    """
    Extract structured metadata about this module.

    Args:
        file_path: Path to the source file (defaults to current file)

    Returns:
        dict: Structured metadata about the module's purpose and components
    """
    return {
        "filename": os.path.basename(file_path),
        "module_purpose": "Factory for creating trainers based on configuration with appropriate strategies",
        "key_classes": [
            {
                "name": "TrainerFactory",
                "purpose": "Factory for creating trainers based on configuration",
                "key_methods": [
                    {
                        "name": "create_trainer",
                        "signature": "create_trainer(model: nn.Module, train_dataloader: torch.utils.data.DataLoader, val_dataloader: Optional[torch.utils.data.DataLoader] = None, test_dataloader: Optional[torch.utils.data.DataLoader] = None, config: Optional[Dict[str, Any]] = None, device: Optional[torch.device] = None, **kwargs) -> Union[MultimodalTrainer, MultistageTrainer]",
                        "brief_description": "Create a trainer based on configuration",
                    },
                    {
                        "name": "_create_multimodal_trainer",
                        "signature": "_create_multimodal_trainer(model: nn.Module, train_dataloader: torch.utils.data.DataLoader, val_dataloader: Optional[torch.utils.data.DataLoader], test_dataloader: Optional[torch.utils.data.DataLoader], config: Dict[str, Any], device: Optional[torch.device]) -> MultimodalTrainer",
                        "brief_description": "Create a standard multimodal trainer with configuration",
                    },
                    {
                        "name": "_create_multistage_trainer",
                        "signature": "_create_multistage_trainer(model: nn.Module, train_dataloader: torch.utils.data.DataLoader, val_dataloader: Optional[torch.utils.data.DataLoader], test_dataloader: Optional[torch.utils.data.DataLoader], config: Dict[str, Any], device: Optional[torch.device]) -> MultistageTrainer",
                        "brief_description": "Create a multistage trainer with appropriate strategies",
                    },
                    {
                        "name": "_create_parameter_groups",
                        "signature": "_create_parameter_groups(model: nn.Module, config: Dict[str, Any]) -> List[Dict[str, Any]]",
                        "brief_description": "Create parameter groups with different learning rates",
                    },
                ],
                "inheritance": "",
                "dependencies": [
                    "torch",
                    "torch.nn",
                    "MultimodalTrainer",
                    "MultistageTrainer",
                    "TrainingStrategy",
                    "SingleModalityStrategy",
                    "CrossModalStrategy",
                    "EndToEndStrategy",
                    "WarmupCosineScheduler",
                    "LinearWarmupScheduler",
                ],
            }
        ],
        "external_dependencies": ["torch", "os"],
        "complexity_score": 7,  # Complex factory pattern with multiple trainer creation methods and configuration
    }
