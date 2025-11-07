# src/utils/learningrate_scheduler.py

import os
import torch
from torch.optim.optimizer import Optimizer
from torch.optim.sgd import SGD
from torch.optim.lr_scheduler import _LRScheduler
import math
from typing import Dict, List, Optional, Any, Callable, Union
import logging

logger = logging.getLogger(__name__)

"""
MODULE: learningrate_scheduler.py
PURPOSE: Implements custom learning rate schedulers for transformer and multimodal training
KEY COMPONENTS:
- WarmupCosineScheduler: Scheduler with linear warmup and cosine annealing
- LinearWarmupScheduler: Scheduler with linear warmup and linear decay 
- LayerwiseLRScheduler: Utility for managing different schedulers for different model components
DEPENDENCIES: torch, torch.optim.lr_scheduler, math, typing, logging
SPECIAL NOTES: Provides specialized schedulers for staged training with different warmup behaviors
"""


class WarmupCosineScheduler(_LRScheduler):
    """
    Scheduler that combines a warmup phase with cosine annealing.

    This scheduler:
    1. Linearly increases learning rate from 0 to base_lr over warmup_steps
    2. Uses cosine annealing to decrease learning rate from base_lr to min_lr over remaining steps
    """

    def __init__(
        self,
        optimizer: Optimizer,
        warmup_steps: int,
        total_steps: int,
        min_lr: float = 0.0,
        warmup_start_factor: float = 0.1,
        last_epoch: int = -1,
        verbose: bool = False,
    ):
        """
        Initialize scheduler.

        Args:
            optimizer: Optimizer to schedule
            warmup_steps: Number of warmup steps
            total_steps: Total number of training steps
            min_lr: Minimum learning rate at the end of scheduling
            warmup_start_factor: Initial warmup factor (0.0 means start from 0, 1.0 means no warmup)
            last_epoch: The index of the last epoch
            verbose: If True, prints a message to stdout for each update
        """
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.min_lr = min_lr
        self.warmup_start_factor = warmup_start_factor

        # Store base LRs for scaling
        self.base_lrs = [group["lr"] for group in optimizer.param_groups]

        super().__init__(optimizer, last_epoch, "verbose")

    def get_lr(self) -> List[float]:
        """
        Calculate current learning rate based on step.

        Returns:
            List of learning rates, one for each parameter group
        """
        if not self._get_lr_called_within_step:
            logger.warning(
                "To get the last learning rate computed by the scheduler, "
                "please use `get_last_lr()`."
            )

        # Calculate current step for scheduling
        current_step = self.last_epoch + 1

        # Calculate new learning rates for each parameter group
        lrs = []
        for base_lr in self.base_lrs:
            # Warmup phase
            if current_step < self.warmup_steps:
                # Linear warmup from warmup_start_factor to 100%
                warmup_factor = self.warmup_start_factor + (
                    1 - self.warmup_start_factor
                ) * (current_step / self.warmup_steps)
                lr = base_lr * warmup_factor
            else:
                # Cosine annealing phase
                progress = (current_step - self.warmup_steps) / max(
                    1, (self.total_steps - self.warmup_steps)
                )
                cos_output = 0.5 * (1.0 + math.cos(math.pi * progress))
                lr = self.min_lr + (base_lr - self.min_lr) * cos_output

            lrs.append(lr)

        return lrs


class LinearWarmupScheduler(_LRScheduler):
    """
    Scheduler with linear warmup followed by linear decay.

    This scheduler:
    1. Linearly increases learning rate from init_lr to base_lr over warmup_epochs
    2. Linearly decreases learning rate from base_lr to final_lr over remaining epochs
    """

    def __init__(
        self,
        optimizer: Optimizer,
        warmup_epochs: int,
        total_epochs: int,
        init_lr: float = 0.0,
        final_lr: float = 0.0,
        last_epoch: int = -1,
        verbose: bool = False,
    ):
        """
        Initialize scheduler.

        Args:
            optimizer: Optimizer to schedule
            warmup_epochs: Number of warmup epochs
            total_epochs: Total number of training epochs
            init_lr: Initial learning rate at the beginning of warmup
            final_lr: Final learning rate at the end of training
            last_epoch: The index of the last epoch
            verbose: If True, prints a message to stdout for each update
        """
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        self.init_lr = init_lr
        self.final_lr = final_lr

        # Store base LRs for scaling
        self.base_lrs = [group["lr"] for group in optimizer.param_groups]

        super().__init__(optimizer, last_epoch, "verbose")

    def get_lr(self) -> List[float]:
        """
        Calculate current learning rate based on epoch.

        Returns:
            List of learning rates, one for each parameter group
        """
        if not self._get_lr_called_within_step:
            logger.warning(
                "To get the last learning rate computed by the scheduler, "
                "please use `get_last_lr()`."
            )

        # Get current epoch
        current_epoch = self.last_epoch + 1

        # Calculate new learning rates for each parameter group
        lrs = []
        for base_lr in self.base_lrs:
            # Warmup phase
            if current_epoch < self.warmup_epochs:
                # Linear warmup from init_lr to base_lr
                warmup_factor = self.init_lr + (base_lr - self.init_lr) * (
                    current_epoch / self.warmup_epochs
                )
                lr = warmup_factor
            else:
                # Linear decay phase
                progress = (current_epoch - self.warmup_epochs) / max(
                    1, (self.total_epochs - self.warmup_epochs)
                )
                lr = base_lr + (self.final_lr - base_lr) * progress

            lrs.append(lr)

        return lrs


class LayerwiseLRScheduler:
    """
    A utility for creating layer-wise learning rates with different schedules.

    This enables differential learning rates across model components with:
    - Different initial learning rates for different components
    - Different warmup schedules
    - Different decay patterns

    Used for fine-grained control in multistage training.
    """

    def __init__(self, optimizer: Optimizer):
        """
        Initialize the layerwise scheduler manager.

        Args:
            optimizer: Optimizer with parameter groups
        """
        self.optimizer = optimizer
        self.schedulers = {}

        # Extract group names for reference
        self.param_groups = {}
        for i, group in enumerate(optimizer.param_groups):
            name = group.get("name", f"group_{i}")
            self.param_groups[name] = i

    def add_scheduler(
        self, group_name: str, scheduler_type: str = "warmup_cosine", **scheduler_kwargs
    ) -> None:
        """
        Add a scheduler for a specific parameter group.

        Args:
            group_name: Name of the parameter group
            scheduler_type: Type of scheduler ('warmup_cosine', 'linear', etc.)
            **scheduler_kwargs: Arguments for the specific scheduler
        """
        if group_name not in self.param_groups:
            logger.warning(f"Parameter group '{group_name}' not found in optimizer")
            return

        # Create a filtered optimizer with only the specific group
        filtered_optimizer = SGD(
            [self.optimizer.param_groups[self.param_groups[group_name]]],
            lr=self.optimizer.param_groups[self.param_groups[group_name]]["lr"],
        )

        # Create the appropriate scheduler
        if scheduler_type == "warmup_cosine":
            scheduler = WarmupCosineScheduler(filtered_optimizer, **scheduler_kwargs)
        elif scheduler_type == "linear_warmup":
            scheduler = LinearWarmupScheduler(filtered_optimizer, **scheduler_kwargs)
        else:
            raise ValueError(f"Unsupported scheduler type: {scheduler_type}")

        self.schedulers[group_name] = (scheduler, self.param_groups[group_name])

    def step(self) -> None:
        """
        Update learning rates for all parameter groups with schedulers.
        """
        for name, (scheduler, group_idx) in self.schedulers.items():
            # Get the new learning rate from scheduler
            scheduler.step()
            new_lr = scheduler.get_last_lr()[0]

            # Update the actual optimizer's parameter group
            self.optimizer.param_groups[group_idx]["lr"] = new_lr

    def get_last_lrs(self) -> Dict[str, float]:
        """
        Get the last computed learning rate for each parameter group.

        Returns:
            Dictionary mapping group names to current learning rates
        """
        lrs = {}

        # Add LRs for groups with schedulers
        for name, (scheduler, _) in self.schedulers.items():
            lrs[name] = scheduler.get_last_lr()[0]

        # Add LRs for groups without schedulers
        for i, group in enumerate(self.optimizer.param_groups):
            name = group.get("name", f"group_{i}")
            if name not in lrs:
                lrs[name] = group["lr"]

        return lrs


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
        "module_purpose": "Implements custom learning rate schedulers for transformer and multimodal training",
        "key_classes": [
            {
                "name": "WarmupCosineScheduler",
                "purpose": "Scheduler that combines a warmup phase with cosine annealing decay",
                "key_methods": [
                    {
                        "name": "__init__",
                        "signature": "__init__(self, optimizer: torch.optim.Optimizer, warmup_steps: int, total_steps: int, min_lr: float = 0.0, warmup_start_factor: float = 0.1, last_epoch: int = -1, verbose: bool = False)",
                        "brief_description": "Initialize scheduler with warmup and cosine decay parameters",
                    },
                    {
                        "name": "get_lr",
                        "signature": "get_lr(self) -> List[float]",
                        "brief_description": "Calculate learning rates based on current step and schedule parameters",
                    },
                ],
                "inheritance": "_LRScheduler",
                "dependencies": ["torch", "torch.optim.lr_scheduler", "math"],
            },
            {
                "name": "LinearWarmupScheduler",
                "purpose": "Scheduler with linear warmup followed by linear decay",
                "key_methods": [
                    {
                        "name": "__init__",
                        "signature": "__init__(self, optimizer: torch.optim.Optimizer, warmup_epochs: int, total_epochs: int, init_lr: float = 0.0, final_lr: float = 0.0, last_epoch: int = -1, verbose: bool = False)",
                        "brief_description": "Initialize scheduler with linear warmup and decay parameters",
                    },
                    {
                        "name": "get_lr",
                        "signature": "get_lr(self) -> List[float]",
                        "brief_description": "Calculate learning rates using linear scheduling",
                    },
                ],
                "inheritance": "_LRScheduler",
                "dependencies": ["torch", "torch.optim.lr_scheduler"],
            },
            {
                "name": "LayerwiseLRScheduler",
                "purpose": "A utility for creating layer-wise learning rates with different schedules",
                "key_methods": [
                    {
                        "name": "add_scheduler",
                        "signature": "add_scheduler(self, group_name: str, scheduler_type: str = 'warmup_cosine', **scheduler_kwargs)",
                        "brief_description": "Add a scheduler for a specific parameter group",
                    },
                    {
                        "name": "step",
                        "signature": "step(self) -> None",
                        "brief_description": "Update learning rates for all parameter groups with schedulers",
                    },
                    {
                        "name": "get_last_lrs",
                        "signature": "get_last_lrs(self) -> Dict[str, float]",
                        "brief_description": "Get the last computed learning rate for each parameter group",
                    },
                ],
                "inheritance": "object",
                "dependencies": ["torch", "typing"],
            },
        ],
        "external_dependencies": ["torch", "math"],
        "complexity_score": 7,  # Multiple scheduler implementations with mathematically involved calculations
    }
