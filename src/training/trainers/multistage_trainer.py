# src/training/trainers/multistage_trainer.py

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Dict, List, Optional, Any, Callable, Union
import logging
import os
import time
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np

from src.training.strategies.training_strategy import TrainingStrategy
from src.utils.metrics_tracker import MetricsTracker

logger = logging.getLogger(__name__)


class MultistageTrainer:
    """
    Trainer that implements a multistage training approach for multimodal models.

    This trainer:
    1. Manages multiple training strategies across different stages
    2. Handles transitions between stages appropriately
    3. Provides stage-specific logging and visualization
    4. Implements checkpoint saving and loading for each stage
    5. Offers comprehensive training monitoring and diagnostics
    """

    def __init__(
        self,
        model: nn.Module,
        train_dataloader: DataLoader,
        val_dataloader: Optional[DataLoader] = None,
        test_dataloader: Optional[DataLoader] = None,
        strategies: Dict[str, Dict] = None,
        checkpoint_dir: str = "checkpoints",
        log_dir: str = "logs",
        device: Optional[torch.device] = None,
        **kwargs,
    ):
        """
        Initialize the multistage trainer.

        Args:
            model: The model to train
            train_dataloader: DataLoader for training data
            val_dataloader: Optional DataLoader for validation data
            test_dataloader: Optional DataLoader for test data
            strategies: Dictionary of stage strategies with configurations
                        Format: {'stage_name': {'strategy': TrainingStrategy, 'epochs': int}}
            checkpoint_dir: Directory to save checkpoints
            log_dir: Directory to save logs
            device: Device to train on
            **kwargs: Additional keyword arguments
        """
        self.model = model
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.test_dataloader = test_dataloader
        self.strategies = strategies or {}
        self.checkpoint_dir = checkpoint_dir
        self.log_dir = log_dir

        # Set device
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

        # Move model to device
        self.model = self.model.to(self.device)

        # Create directories
        os.makedirs(checkpoint_dir, exist_ok=True)
        os.makedirs(log_dir, exist_ok=True)

        # Initialize stage tracking
        self.current_stage = None
        self.current_strategy = None
        self.current_epoch = 0
        self.global_step = 0

        # Store additional configuration
        self.config = kwargs

        # Create metrics tracker
        self.metrics_tracker = MetricsTracker(
            log_dir=os.path.join(log_dir, "metrics"),
            early_stopping=self.config.get("early_stopping", False),
            patience=self.config.get("early_stopping_patience", 5),
            monitor=self.config.get("monitor_metric", "val_loss"),
            mode=self.config.get("monitor_mode", "min"),
            visualization_frequency=1,
            issue_detection=True,
        )

        # Validate strategies
        if not self.strategies:
            logger.warning(
                "No training strategies provided. Training will not be effective."
            )
        else:
            logger.info(
                f"Initialized MultistageTrainer with {len(self.strategies)} stages"
            )
            for stage_name, stage_info in self.strategies.items():
                logger.info(
                    f"  {stage_name}: {stage_info['epochs']} epochs with {type(stage_info['strategy']).__name__}"
                )

    def train(self) -> Dict[str, Any]:
        """
        Train the model through all stages.

        Returns:
            Dictionary with training history
        """
        # Record overall training start time
        start_time = time.time()
        logger.info("Starting multistage training")

        # Track total epochs across all stages
        total_epochs = sum(
            stage_info["epochs"] for stage_info in self.strategies.values()
        )
        current_global_epoch = 0

        # Train through each stage
        for stage_name, stage_info in self.strategies.items():
            # Set current stage and strategy
            self.current_stage = stage_name
            self.current_strategy = stage_info["strategy"]
            stage_epochs = stage_info["epochs"]

            logger.info(f"Starting {stage_name} with {stage_epochs} epochs")

            # Create stage-specific directories
            stage_checkpoint_dir = os.path.join(self.checkpoint_dir, stage_name)
            stage_log_dir = os.path.join(self.log_dir, stage_name)
            os.makedirs(stage_checkpoint_dir, exist_ok=True)
            os.makedirs(stage_log_dir, exist_ok=True)

            # Train for this stage
            stage_metrics = self._train_stage(
                strategy=self.current_strategy,
                epochs=stage_epochs,
                checkpoint_dir=stage_checkpoint_dir,
                log_dir=stage_log_dir,
                global_epoch_offset=current_global_epoch,
            )

            # Update global epoch counter
            current_global_epoch += stage_epochs

            # Save stage completion checkpoint
            checkpoint_path = os.path.join(
                stage_checkpoint_dir, f"{stage_name}_complete.pt"
            )
            self._save_checkpoint(checkpoint_path, stage_metrics)

            # Evaluate after stage if requested
            if self.config.get("evaluate_after_stage", True) and self.val_dataloader:
                logger.info(f"Evaluating after {stage_name}")
                val_metrics = self._evaluate(self.current_strategy, self.val_dataloader)

                # Log metrics
                logger.info(f"Validation metrics after {stage_name}:")
                for metric_name, metric_value in val_metrics.items():
                    logger.info(f"  {metric_name}: {metric_value}")

        # Record overall training time
        total_time = time.time() - start_time
        logger.info(f"Multistage training completed in {total_time:.2f} seconds")

        # Final evaluation if test data is available
        if self.test_dataloader:
            logger.info("Performing final evaluation on test data")
            test_metrics = self._evaluate(self.current_strategy, self.test_dataloader)

            # Log metrics
            logger.info("Final test metrics:")
            for metric_name, metric_value in test_metrics.items():
                logger.info(f"  {metric_name}: {metric_value}")

        # Return metrics from the metrics tracker
        return self.metrics_tracker.get_latest_metrics()

    def _train_stage(
        self,
        strategy: TrainingStrategy,
        epochs: int,
        checkpoint_dir: str,
        log_dir: str,
        global_epoch_offset: int = 0,
    ) -> Dict[str, Any]:
        """
        Train the model for a specific stage.

        Args:
            strategy: The training strategy to use
            epochs: Number of epochs to train
            checkpoint_dir: Directory to save checkpoints
            log_dir: Directory to save logs
            global_epoch_offset: Offset for global epoch counter

        Returns:
            Dictionary with stage training metrics
        """
        # Initialize epoch metrics
        epoch_metrics = {}

        # Train for specified epochs
        for epoch in range(epochs):
            # Update current epoch
            self.current_epoch = epoch
            global_epoch = global_epoch_offset + epoch

            # Signal epoch start to metrics tracker
            self.metrics_tracker.start_epoch(global_epoch)

            # Call strategy's epoch start hook
            strategy.on_epoch_start(epoch)

            # Train for one epoch
            train_metrics = self._train_epoch(strategy)

            # Update metrics tracker
            self.metrics_tracker.update_epoch_metrics(train_metrics, group="train")

            # Validate if validation data is available
            if self.val_dataloader:
                val_metrics = self._evaluate(strategy, self.val_dataloader)
                self.metrics_tracker.update_epoch_metrics(val_metrics, group="val")

            # Call strategy's epoch end hook
            strategy.on_epoch_end(epoch)

            # Signal epoch end to metrics tracker
            epoch_summary = self.metrics_tracker.end_epoch()

            # Save checkpoint
            if (epoch + 1) % self.config.get("checkpoint_frequency", 1) == 0:
                checkpoint_path = os.path.join(checkpoint_dir, f"epoch_{epoch + 1}.pt")
                self._save_checkpoint(checkpoint_path, train_metrics)

            # Check for early stopping
            if self.metrics_tracker.stopped_early:
                logger.info(f"Early stopping triggered at epoch {epoch + 1}")
                break

        return epoch_metrics

    def _train_epoch(self, strategy: TrainingStrategy) -> Dict[str, float]:
        """
        Train the model for one epoch using the provided strategy.

        Args:
            strategy: The training strategy to use

        Returns:
            Dictionary with training metrics
        """
        # Set model to train mode
        self.model.train()

        # Initialize metrics
        epoch_metrics = {}
        running_metrics = {}
        step_count = 0

        # Create progress bar
        pbar = tqdm(
            self.train_dataloader,
            desc=f"Epoch {self.current_epoch + 1}/{strategy.config.get('epochs', '?')} ({self.current_stage})",
        )

        # Iterate through batches
        for batch_idx, batch in enumerate(pbar):
            # Zero gradients
            strategy.optimizer.zero_grad()

            # Perform training step
            batch_metrics = strategy.training_step(batch)

            # Update step metrics in tracker
            self.metrics_tracker.update_step_metrics(batch_metrics, group="train")

            # Update running metrics for progress bar
            for k, v in batch_metrics.items():
                if k not in running_metrics:
                    running_metrics[k] = 0.0
                running_metrics[k] += v

            # Update progress bar
            avg_metrics = {k: v / (batch_idx + 1) for k, v in running_metrics.items()}
            pbar.set_postfix(avg_metrics)

            # Step optimizer
            strategy.optimizer.step()

            # Step scheduler if it's step-based
            if (
                strategy.scheduler is not None
                and getattr(strategy.scheduler, "step_batch", None) is not None
            ):
                strategy.scheduler.step_batch()

            # Increment step counters
            step_count += 1
            self.global_step += 1

        # Step scheduler if it's epoch-based
        if (
            strategy.scheduler is not None
            and getattr(strategy.scheduler, "step_batch", None) is None
        ):
            strategy.scheduler.step()

        # Calculate epoch metrics (averages)
        epoch_metrics = {k: v / step_count for k, v in running_metrics.items()}

        # Add learning rates to metrics
        epoch_metrics["learning_rate"] = strategy.optimizer.param_groups[0]["lr"]

        return epoch_metrics

    def _evaluate(
        self, strategy: TrainingStrategy, dataloader: DataLoader
    ) -> Dict[str, float]:
        """
        Evaluate the model using the provided strategy and dataloader.

        Args:
            strategy: The training strategy to use
            dataloader: DataLoader for evaluation

        Returns:
            Dictionary with evaluation metrics
        """
        # Set model to eval mode
        self.model.eval()

        # Initialize metrics
        all_metrics = {}
        batch_metrics = []

        # Create progress bar
        pbar = tqdm(dataloader, desc=f"Evaluating ({self.current_stage})")

        # Iterate through batches
        for batch in pbar:
            # Perform validation step
            metrics = strategy.validation_step(batch)
            batch_metrics.append(metrics)

        # Calculate average metrics across batches
        for k in batch_metrics[0].keys():
            values = [m[k] for m in batch_metrics if k in m]
            if values:
                all_metrics[k] = sum(values) / len(values)

        return all_metrics

    def _save_checkpoint(
        self, path: str, metrics: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Save a checkpoint of the current training state.

        Args:
            path: Path to save the checkpoint
            metrics: Optional metrics to include in the checkpoint
        """
        # Create checkpoint
        checkpoint = {
            "model_state_dict": self.model.state_dict(),
            "current_stage": self.current_stage,
            "current_epoch": self.current_epoch,
            "global_step": self.global_step,
            "metrics": metrics or {},
        }

        # Include optimizer and scheduler if available
        if self.current_strategy and hasattr(self.current_strategy, "optimizer"):
            checkpoint["optimizer_state_dict"] = (
                self.current_strategy.optimizer.state_dict()
            )

        if self.current_strategy and hasattr(self.current_strategy, "scheduler"):
            checkpoint["scheduler_state_dict"] = (
                self.current_strategy.scheduler.state_dict()
            )

        # Save checkpoint
        torch.save(checkpoint, path)
        logger.info(f"Checkpoint saved to {path}")

    def load_checkpoint(self, path: str) -> None:
        """
        Load a checkpoint to resume training.

        Args:
            path: Path to the checkpoint
        """
        # Check if file exists
        if not os.path.exists(path):
            logger.error(f"Checkpoint not found: {path}")
            return

        # Load checkpoint
        checkpoint = torch.load(path, map_location=self.device)

        # Load model state
        self.model.load_state_dict(checkpoint["model_state_dict"])

        # Restore training state
        self.current_stage = checkpoint["current_stage"]
        self.current_epoch = checkpoint["current_epoch"]
        self.global_step = checkpoint["global_step"]

        # Find the corresponding strategy
        if self.current_stage in self.strategies:
            self.current_strategy = self.strategies[self.current_stage]["strategy"]

            # Restore optimizer state if available
            if "optimizer_state_dict" in checkpoint and hasattr(
                self.current_strategy, "optimizer"
            ):
                self.current_strategy.optimizer.load_state_dict(
                    checkpoint["optimizer_state_dict"]
                )

            # Restore scheduler state if available
            if "scheduler_state_dict" in checkpoint and hasattr(
                self.current_strategy, "scheduler"
            ):
                self.current_strategy.scheduler.load_state_dict(
                    checkpoint["scheduler_state_dict"]
                )

        logger.info(f"Checkpoint loaded from {path}")
        logger.info(
            f"Resuming from stage {self.current_stage}, epoch {self.current_epoch}"
        )

    def create_visualizations(self) -> None:
        """
        Create visualizations of training progress.
        """
        # Use the metrics tracker to create visualizations
        self.metrics_tracker.create_visualizations()
