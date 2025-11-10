"""MODULE: checkpoint_manager.py
PURPOSE: Manages model checkpointing including saving, loading, and resuming training state.

KEY COMPONENTS:
- CheckpointManager: Handles checkpoint save/load operations and training state persistence

DEPENDENCIES:
- PyTorch (torch)
- Python standard library (os, logging)

SPECIAL NOTES:
- Supports saving/loading of model, optimizer, scheduler states
- Tracks training progress (epoch, step, metrics, patience counter)
- Creates checkpoint directories automatically
"""

import os
import logging
from typing import Dict, Optional, Any
from collections import defaultdict

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class CheckpointManager:
    """Manages model checkpointing and training state persistence."""

    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        checkpoint_dir: str,
        scheduler: Optional[Any] = None,
        device: Optional[torch.device] = None,
    ):
        """
        Initialize the checkpoint manager.

        Args:
            model: The model to checkpoint
            optimizer: The optimizer to checkpoint
            checkpoint_dir: Directory to save checkpoints
            scheduler: Optional learning rate scheduler
            device: Device to load checkpoints to (default: CPU)
        """
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.checkpoint_dir = checkpoint_dir
        self.device = device if device is not None else torch.device("cpu")

        # Create checkpoint directory
        os.makedirs(checkpoint_dir, exist_ok=True)

        # Training state (to be updated by trainer)
        self.current_epoch = 0
        self.global_step = 0
        self.best_val_metric = 0.0
        self.patience_counter = 0
        self.history = defaultdict(list)

    def save_checkpoint(
        self,
        path: str,
        current_epoch: Optional[int] = None,
        global_step: Optional[int] = None,
        best_val_metric: Optional[float] = None,
        patience_counter: Optional[int] = None,
        history: Optional[Dict[str, list]] = None,
    ) -> None:
        """
        Save a checkpoint.

        Args:
            path: Path to save checkpoint
            current_epoch: Current training epoch (uses internal state if None)
            global_step: Global training step (uses internal state if None)
            best_val_metric: Best validation metric so far (uses internal state if None)
            patience_counter: Early stopping patience counter (uses internal state if None)
            history: Training history dictionary (uses internal state if None)
        """
        # Create directory if needed
        os.makedirs(os.path.dirname(path), exist_ok=True)

        # Use provided values or fall back to internal state
        checkpoint = {
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "current_epoch": current_epoch if current_epoch is not None else self.current_epoch,
            "global_step": global_step if global_step is not None else self.global_step,
            "best_val_metric": best_val_metric if best_val_metric is not None else self.best_val_metric,
            "patience_counter": patience_counter if patience_counter is not None else self.patience_counter,
            "history": dict(history if history is not None else self.history),
        }

        # Add scheduler state if available
        if self.scheduler is not None:
            checkpoint["scheduler_state_dict"] = self.scheduler.state_dict()

        # Save checkpoint
        torch.save(checkpoint, path)
        logger.info(f"Checkpoint saved to {path}")

    def load_checkpoint(self, path: str) -> Dict[str, Any]:
        """
        Load a checkpoint.

        Args:
            path: Path to checkpoint

        Returns:
            Dictionary containing loaded training state
        """
        if not os.path.exists(path):
            logger.warning(f"Checkpoint not found: {path}")
            return {}

        checkpoint = torch.load(path, map_location=self.device, weights_only=True)

        # Load model weights
        self.model.load_state_dict(checkpoint["model_state_dict"])

        # Load optimizer state
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        # Load scheduler state if available
        if self.scheduler is not None and "scheduler_state_dict" in checkpoint:
            self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

        # Load training state
        self.current_epoch = checkpoint["current_epoch"] + 1  # Resume from next epoch
        self.global_step = checkpoint["global_step"]
        self.best_val_metric = checkpoint["best_val_metric"]
        self.patience_counter = checkpoint["patience_counter"]

        # Load history
        if "history" in checkpoint:
            self.history = defaultdict(list, checkpoint["history"])

        logger.info(f"Checkpoint loaded from {path}")
        logger.info(f"Resuming from epoch {self.current_epoch}")

        return {
            "current_epoch": self.current_epoch,
            "global_step": self.global_step,
            "best_val_metric": self.best_val_metric,
            "patience_counter": self.patience_counter,
            "history": dict(self.history),
        }

    def get_latest_checkpoint(self) -> Optional[str]:
        """
        Find the most recent checkpoint in the checkpoint directory.

        Returns:
            Path to the latest checkpoint, or None if no checkpoints found
        """
        if not os.path.exists(self.checkpoint_dir):
            return None

        checkpoints = [
            f for f in os.listdir(self.checkpoint_dir)
            if f.endswith(".pt") or f.endswith(".pth")
        ]

        if not checkpoints:
            return None

        # Sort by modification time
        checkpoints.sort(
            key=lambda x: os.path.getmtime(os.path.join(self.checkpoint_dir, x)),
            reverse=True
        )

        latest = os.path.join(self.checkpoint_dir, checkpoints[0])
        logger.info(f"Found latest checkpoint: {latest}")
        return latest

    def get_checkpoint_path(self, epoch: int, metric_value: Optional[float] = None) -> str:
        """
        Generate a checkpoint path for a given epoch.

        Args:
            epoch: Epoch number
            metric_value: Optional metric value to include in filename

        Returns:
            Full path to checkpoint file
        """
        if metric_value is not None:
            filename = f"checkpoint_epoch_{epoch}_metric_{metric_value:.4f}.pt"
        else:
            filename = f"checkpoint_epoch_{epoch}.pt"

        return os.path.join(self.checkpoint_dir, filename)

    def save_best_checkpoint(
        self,
        metric_value: float,
        current_epoch: int,
        global_step: int,
        history: Dict[str, list],
    ) -> None:
        """
        Save a checkpoint as the best model so far.

        Args:
            metric_value: Current metric value
            current_epoch: Current epoch number
            global_step: Current global step
            history: Training history
        """
        path = os.path.join(self.checkpoint_dir, "best_model.pt")
        self.save_checkpoint(
            path=path,
            current_epoch=current_epoch,
            global_step=global_step,
            best_val_metric=metric_value,
            patience_counter=self.patience_counter,
            history=history,
        )
        logger.info(f"Best model saved with metric: {metric_value:.4f}")

    def update_state(
        self,
        current_epoch: Optional[int] = None,
        global_step: Optional[int] = None,
        best_val_metric: Optional[float] = None,
        patience_counter: Optional[int] = None,
        history: Optional[Dict[str, list]] = None,
    ) -> None:
        """
        Update internal training state.

        Args:
            current_epoch: Current training epoch
            global_step: Global training step
            best_val_metric: Best validation metric so far
            patience_counter: Early stopping patience counter
            history: Training history dictionary
        """
        if current_epoch is not None:
            self.current_epoch = current_epoch
        if global_step is not None:
            self.global_step = global_step
        if best_val_metric is not None:
            self.best_val_metric = best_val_metric
        if patience_counter is not None:
            self.patience_counter = patience_counter
        if history is not None:
            self.history = defaultdict(list, history)
