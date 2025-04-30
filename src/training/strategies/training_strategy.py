# src/training/strategies/training_strategy.py

from abc import ABC, abstractmethod
import torch
import torch.nn as nn
from torch.optim.optimizer import Optimizer
from typing import Dict, List, Optional, Any, Callable, Union
import logging

logger = logging.getLogger(__name__)


class TrainingStrategy(ABC):
    """
    Abstract base class for training strategies used in the multistage training framework.

    This class defines the contract that all concrete strategies must implement.
    Each strategy handles a specific stage of training with its own:
    - Parameter freezing/unfreezing logic
    - Learning rate configurations
    - Loss function setup
    - Optimizer configuration
    - Training and evaluation behavior
    """

    def __init__(
        self,
        model: nn.Module,
        optimizer: Optional[Optimizer] = None,
        scheduler: Optional[Any] = None,
        loss_fn: Optional[nn.Module] = None,
        device: Optional[torch.device] = None,
        **kwargs,
    ):
        """
        Initialize the training strategy.

        Args:
            model: The model to train
            optimizer: The optimizer (may be created by the strategy if None)
            scheduler: The learning rate scheduler (may be created by the strategy if None)
            loss_fn: The loss function (may be created by the strategy if None)
            device: The device to train on
            **kwargs: Additional strategy-specific parameters
        """
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.loss_fn = loss_fn

        # Set device with priority: provided > CUDA > MPS > CPU
        if device is None:
            if torch.cuda.is_available():
                self.device = torch.device("cuda")
            elif (
                hasattr(torch, "backends")
                and hasattr(torch.backends, "mps")
                and torch.backends.mps.is_available()
            ):
                self.device = torch.device("mps")
            else:
                self.device = torch.device("cpu")
        else:
            self.device = device

        # Strategy configuration parameters
        self.config = kwargs

        # Initialize strategy
        self.initialize_strategy()

    @abstractmethod
    def initialize_strategy(self) -> None:
        """
        Initialize the strategy, including:
        - Setting up parameter groups
        - Freezing/unfreezing parameters as needed
        - Creating optimizer and scheduler if needed
        - Setting up loss function if needed

        This method must be implemented by concrete strategies.
        """
        pass

    @abstractmethod
    def prepare_batch(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        """
        Prepare a batch of data for the model.
        May include custom preprocessing specific to the strategy.

        Args:
            batch: The batch of data from dataloader

        Returns:
            Processed batch ready for the model
        """
        pass

    @abstractmethod
    def training_step(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        """
        Perform a single training step.

        Args:
            batch: The batch of data

        Returns:
            Dictionary containing the loss and other metrics
        """
        pass

    @abstractmethod
    def validation_step(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        """
        Perform a single validation step.

        Args:
            batch: The batch of data

        Returns:
            Dictionary containing validation metrics
        """
        pass

    @abstractmethod
    def configure_optimizers(self) -> tuple:
        """
        Configure optimizers and schedulers.

        Returns:
            Tuple of (optimizer, scheduler)
        """
        pass

    def freeze_parameters(self, model_components: List[str]) -> None:
        """
        Freeze parameters of specific model components.

        Args:
            model_components: List of component names to freeze
        """
        for name, param in self.model.named_parameters():
            for component in model_components:
                if component in name:
                    param.requires_grad = False
                    break

    def unfreeze_parameters(self, model_components: List[str]) -> None:
        """
        Unfreeze parameters of specific model components.

        Args:
            model_components: List of component names to unfreeze
        """
        for name, param in self.model.named_parameters():
            for component in model_components:
                if component in name:
                    param.requires_grad = True
                    break

    def log_parameter_status(self) -> Dict[str, int]:
        """
        Log the trainable status of model parameters for debugging.

        Returns:
            Dictionary with components and their parameter counts
        """
        component_stats = {}

        # Identify main components from parameter names
        all_components = set()
        for name, _ in self.model.named_parameters():
            # Extract component from parameter name (e.g., vision_model, text_model)
            component = name.split(".")[0] if "." in name else "other"
            all_components.add(component)

        # Calculate statistics for each component
        for component in all_components:
            trainable_params = 0
            frozen_params = 0

            for name, param in self.model.named_parameters():
                if component in name:
                    if param.requires_grad:
                        trainable_params += param.numel()
                    else:
                        frozen_params += param.numel()

            total_params = trainable_params + frozen_params
            if total_params > 0:
                trainable_percentage = 100 * trainable_params / total_params
                component_stats[component] = {
                    "trainable": trainable_params,
                    "frozen": frozen_params,
                    "total": total_params,
                    "trainable_pct": f"{trainable_percentage:.1f}%",
                }

        # Log parameter status
        logger.info("Parameter trainable status:")
        for component, stats in component_stats.items():
            logger.info(
                f"  {component}: {stats['trainable_pct']} trainable "
                f"({stats['trainable']:,} / {stats['total']:,} parameters)"
            )

        return component_stats

    def get_learning_rates(self) -> Dict[str, float]:
        """
        Get current learning rates for different parameter groups.

        Returns:
            Dictionary mapping parameter group names to learning rates
        """
        if self.optimizer is None:
            return {}

        lrs = {}
        for i, group in enumerate(self.optimizer.param_groups):
            group_name = group.get("name", f"group_{i}")
            lrs[group_name] = group["lr"]

        return lrs

    def on_epoch_start(self, epoch: int) -> None:
        """
        Perform actions at the start of each epoch.

        Args:
            epoch: Current epoch number
        """
        # Default implementation does nothing
        pass

    def on_epoch_end(self, epoch: int) -> None:
        """
        Perform actions at the end of each epoch.

        Args:
            epoch: Current epoch number
        """
        # Default implementation does nothing
        pass
