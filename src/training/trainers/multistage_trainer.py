# src/training/trainers/multistage_trainer.py

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Dict, List, Optional, Any, Callable, Union, Tuple
import logging
import os
import time
import copy
import json
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np

from src.training.strategies.training_strategy import TrainingStrategy
from src.training.strategies.single_modality_strategy import SingleModalityStrategy
from src.training.strategies.cross_modal_strategy import CrossModalStrategy
from src.training.strategies.end_to_end_strategy import EndToEndStrategy
from src.utils.metrics_tracker import MetricsTracker
from src.configs.training_config import TrainingConfig, StageConfig

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
        config: Optional[TrainingConfig] = None,
        strategies: Optional[Dict[str, Dict]] = None,
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
            config: Training configuration with stage settings (preferred over strategies)
            strategies: Dictionary of stage strategies with configurations (legacy support)
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

        # Configuration handling - prefer TrainingConfig if provided
        self.config = config
        self.strategies = strategies or {}

        if self.config:
            # Override directories if in config
            checkpoint_dir = self.config.output_dir or checkpoint_dir
            log_dir = os.path.join(checkpoint_dir, "logs")

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
        self.current_stage_idx = 0
        self.current_stage = None
        self.current_strategy = None
        self.current_epoch = 0
        self.global_step = 0
        self.stage_results = {}

        # Store additional configuration
        self.extra_config = kwargs

        # Create dict of metrics trackers (one per stage)
        self.metrics_trackers = {}

        # If using TrainingConfig, convert to strategies
        if self.config and not self.strategies:
            self._setup_strategies_from_config()

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

    def _setup_strategies_from_config(self) -> None:
        """
        Set up strategies based on the provided TrainingConfig.
        """
        if not self.config or not self.config.stages:
            logger.warning("No stages found in training config")
            return

        for idx, stage_config in enumerate(self.config.stages):
            # Create appropriate strategy based on stage name
            if stage_config.name == "modality_specific_learning":
                strategy_class = SingleModalityStrategy
            elif stage_config.name == "cross_modal_fusion":
                strategy_class = CrossModalStrategy
            elif stage_config.name == "end_to_end_fine_tuning":
                strategy_class = EndToEndStrategy
            else:
                logger.warning(
                    f"Unknown stage type: {stage_config.name}, using default strategy"
                )
                strategy_class = TrainingStrategy

            # Create the strategy
            strategy = strategy_class(
                model=self.model,
                train_dataloader=self.train_dataloader,
                val_dataloader=self.val_dataloader,
                config=stage_config,
                device=self.device,
            )

            # Apply component freezing specified in config
            self._apply_component_freezing(stage_config)

            # Add to strategies dictionary
            self.strategies[stage_config.name] = {
                "strategy": strategy,
                "epochs": stage_config.epochs,
            }

            # Create metrics tracker for this stage
            stage_log_dir = os.path.join(self.log_dir, f"{idx+1}_{stage_config.name}")
            os.makedirs(stage_log_dir, exist_ok=True)

            self.metrics_trackers[stage_config.name] = MetricsTracker(
                log_dir=stage_log_dir,
                early_stopping=stage_config.early_stopping,
                patience=stage_config.patience,
                monitor=stage_config.monitor_metric,
                mode=stage_config.monitor_mode,
                visualization_frequency=1,
                issue_detection=True,
            )

    def _apply_component_freezing(self, stage_config: StageConfig) -> None:
        """
        Apply component freezing based on stage configuration.

        Args:
            stage_config: Configuration for the current stage
        """
        if not hasattr(stage_config, "components"):
            return

        # Map component names to actual model components
        component_map = {
            "vision_model": getattr(self.model, "vision_model", None),
            "text_model": getattr(self.model, "text_model", None),
            "vision_projection": getattr(self.model, "vision_proj", None),
            "text_projection": getattr(self.model, "text_proj", None),
            "cross_attention": getattr(self.model, "cross_attention", None),
        }

        # Add any additional components that might exist in the specific model
        if hasattr(self.model, "fusion_layer"):
            component_map["fusion_layer"] = self.model.fusion_layer

        if hasattr(self.model, "bidirectional_cross_attention"):
            component_map["cross_attention"] = self.model.bidirectional_cross_attention

        # Apply freezing/unfreezing
        for comp_config in stage_config.components:
            component = component_map.get(comp_config.name)

            if component is None:
                logger.warning(
                    f"Component {comp_config.name} not found in model, skipping"
                )
                continue

            if comp_config.freeze:
                logger.info(f"Freezing component: {comp_config.name}")
                for param in component.parameters():
                    param.requires_grad = False
            else:
                logger.info(f"Unfreezing component: {comp_config.name}")
                for param in component.parameters():
                    param.requires_grad = True

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
        stage_idx = 0
        for stage_name, stage_info in self.strategies.items():
            # Set current stage and strategy
            self.current_stage_idx = stage_idx
            self.current_stage = stage_name
            self.current_strategy = stage_info["strategy"]
            stage_epochs = stage_info["epochs"]

            logger.info(f"Starting {stage_name} with {stage_epochs} epochs")

            # Create stage-specific directories
            stage_checkpoint_dir = os.path.join(
                self.checkpoint_dir, f"{stage_idx+1}_{stage_name}"
            )
            stage_log_dir = os.path.join(self.log_dir, f"{stage_idx+1}_{stage_name}")
            os.makedirs(stage_checkpoint_dir, exist_ok=True)
            os.makedirs(stage_log_dir, exist_ok=True)

            # Get metrics tracker for this stage
            metrics_tracker = self.metrics_trackers.get(
                stage_name,
                MetricsTracker(
                    log_dir=stage_log_dir,
                    early_stopping=self.extra_config.get("early_stopping", False),
                    patience=self.extra_config.get("early_stopping_patience", 5),
                    monitor=self.extra_config.get("monitor_metric", "val_loss"),
                    mode=self.extra_config.get("monitor_mode", "min"),
                    visualization_frequency=1,
                    issue_detection=True,
                ),
            )

            # Train for this stage
            stage_metrics = self._train_stage(
                strategy=self.current_strategy,
                epochs=stage_epochs,
                checkpoint_dir=stage_checkpoint_dir,
                log_dir=stage_log_dir,
                metrics_tracker=metrics_tracker,
                global_epoch_offset=current_global_epoch,
            )

            # Store stage results
            self.stage_results[stage_idx] = {
                "stage_name": stage_name,
                "metrics": stage_metrics,
                "best_metric": metrics_tracker.best_value,
            }

            # Update global epoch counter
            current_global_epoch += stage_epochs

            # Save stage completion checkpoint
            checkpoint_path = os.path.join(
                stage_checkpoint_dir, f"{stage_name}_complete.pt"
            )
            self._save_checkpoint(checkpoint_path, stage_metrics)

            # Evaluate after stage if requested
            if (
                self.extra_config.get("evaluate_after_stage", True)
                and self.val_dataloader
            ):
                logger.info(f"Evaluating after {stage_name}")
                val_metrics = self._evaluate(self.current_strategy, self.val_dataloader)

                # Log metrics
                logger.info(f"Validation metrics after {stage_name}:")
                for metric_name, metric_value in val_metrics.items():
                    logger.info(f"  {metric_name}: {metric_value}")

            # Increment stage index
            stage_idx += 1

        # Record overall training time
        total_time = time.time() - start_time
        logger.info(f"Multistage training completed in {total_time:.2f} seconds")

        # Final evaluation if test data is available
        final_metrics = {}
        if self.test_dataloader:
            logger.info("Performing final evaluation on test data")
            test_metrics = self._evaluate(self.current_strategy, self.test_dataloader)

            # Log metrics
            logger.info("Final test metrics:")
            for metric_name, metric_value in test_metrics.items():
                logger.info(f"  {metric_name}: {metric_value}")
                final_metrics[f"test_{metric_name}"] = metric_value

        # Return all stage results plus final metrics
        return {
            "stages": self.stage_results,
            "final_metrics": final_metrics,
            "total_training_time": total_time,
        }

    def _train_stage(
        self,
        strategy: TrainingStrategy,
        epochs: int,
        checkpoint_dir: str,
        log_dir: str,
        metrics_tracker: MetricsTracker,
        global_epoch_offset: int = 0,
    ) -> Dict[str, Any]:
        """
        Train the model for a specific stage.

        Args:
            strategy: The training strategy to use
            epochs: Number of epochs to train
            checkpoint_dir: Directory to save checkpoints
            log_dir: Directory to save logs
            metrics_tracker: Metrics tracker to use for this stage
            global_epoch_offset: Offset for global epoch counter

        Returns:
            Dictionary with stage training metrics
        """
        # Initialize metrics and best model state
        epoch_metrics = {}
        best_model_state = None
        best_metric_value = (
            float("inf") if metrics_tracker.mode == "min" else float("-inf")
        )

        # Train for specified epochs
        for epoch in range(epochs):
            # Update current epoch
            self.current_epoch = epoch
            global_epoch = global_epoch_offset + epoch

            # Signal epoch start to metrics tracker
            metrics_tracker.start_epoch(global_epoch)

            # Call strategy's epoch start hook if available
            if hasattr(strategy, "on_epoch_start"):
                strategy.on_epoch_start(epoch)

            # Train for one epoch
            train_metrics = self._train_epoch(strategy)

            # Update metrics tracker
            metrics_tracker.update_epoch_metrics(train_metrics, group="train")

            # Validate if validation data is available
            if self.val_dataloader:
                val_metrics = self._evaluate(strategy, self.val_dataloader)
                metrics_tracker.update_epoch_metrics(val_metrics, group="val")

                # Check if we should save a new best model
                current_metric = val_metrics.get(metrics_tracker.monitor, None)
                if current_metric is not None:
                    is_better = False

                    if metrics_tracker.mode == "min":
                        is_better = current_metric < best_metric_value
                    else:
                        is_better = current_metric > best_metric_value

                    if is_better:
                        best_metric_value = current_metric
                        best_model_state = copy.deepcopy(self.model.state_dict())
                        # Save best model checkpoint
                        checkpoint_path = os.path.join(checkpoint_dir, "best_model.pt")
                        self._save_checkpoint(checkpoint_path, val_metrics)

            # Call strategy's epoch end hook if available
            if hasattr(strategy, "on_epoch_end"):
                strategy.on_epoch_end(epoch)

            # Signal epoch end to metrics tracker
            epoch_summary = metrics_tracker.end_epoch()

            # Create visualizations if needed
            if epoch % self.extra_config.get("visualize_every", 1) == 0:
                metrics_tracker.create_visualizations()

            # Save regular checkpoint
            if (epoch + 1) % self.extra_config.get("checkpoint_frequency", 1) == 0:
                checkpoint_path = os.path.join(checkpoint_dir, f"epoch_{epoch + 1}.pt")
                self._save_checkpoint(checkpoint_path, train_metrics)

            # Check for early stopping
            if metrics_tracker.check_early_stopping():
                logger.info(f"Early stopping triggered at epoch {epoch + 1}")
                break

            # Check for training issues
            issues = metrics_tracker.check_for_issues()
            if issues:
                for issue in issues:
                    logger.warning(f"Training issue detected: {issue}")

        # Restore best model if we found one
        if best_model_state is not None:
            logger.info(
                f"Restoring best model state with {metrics_tracker.monitor} = {best_metric_value}"
            )
            self.model.load_state_dict(best_model_state)

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
            # Perform training step (each strategy handles its own optimization)
            batch_metrics = strategy.training_step(batch)

            # Update step metrics in tracker if available
            if self.current_stage in self.metrics_trackers:
                self.metrics_trackers[self.current_stage].update_step_metrics(
                    batch_metrics, group="train"
                )

            # Update running metrics for progress bar
            for k, v in batch_metrics.items():
                if isinstance(v, torch.Tensor):
                    v = v.item()
                if k not in running_metrics:
                    running_metrics[k] = 0.0
                running_metrics[k] += v

            # Update progress bar
            avg_metrics = {k: v / (batch_idx + 1) for k, v in running_metrics.items()}
            pbar.set_postfix(avg_metrics)

            # Increment step counters
            step_count += 1
            self.global_step += 1

        # Calculate epoch metrics (averages)
        epoch_metrics = {k: v / step_count for k, v in running_metrics.items()}

        # Add learning rates to metrics if optimizer is available
        if hasattr(strategy, "optimizer"):
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
        with torch.no_grad():
            for batch in pbar:
                # Perform validation step
                metrics = strategy.validation_step(batch)
                batch_metrics.append(metrics)

        # Calculate average metrics across batches
        for k in batch_metrics[0].keys():
            values = [m[k] for m in batch_metrics if k in m]
            if values:
                if isinstance(values[0], torch.Tensor):
                    values = [v.item() for v in values]
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
            "current_stage_idx": self.current_stage_idx,
            "current_epoch": self.current_epoch,
            "global_step": self.global_step,
            "metrics": metrics or {},
            "stage_results": self.stage_results,
        }

        # Include strategy-specific state
        if self.current_strategy:
            # Include optimizer state if available
            if hasattr(self.current_strategy, "optimizer"):
                checkpoint["optimizer_state_dict"] = (
                    self.current_strategy.optimizer.state_dict()
                )

            # Include scheduler state if available
            if hasattr(self.current_strategy, "scheduler"):
                checkpoint["scheduler_state_dict"] = (
                    self.current_strategy.scheduler.state_dict()
                )

        # Include configuration if available
        if self.config:
            checkpoint["config"] = (
                self.config.to_dict()
                if hasattr(self.config, "to_dict")
                else vars(self.config)
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
        self.current_stage_idx = checkpoint.get("current_stage_idx", 0)
        self.current_epoch = checkpoint["current_epoch"]
        self.global_step = checkpoint["global_step"]

        # Restore stage results if available
        if "stage_results" in checkpoint:
            self.stage_results = checkpoint["stage_results"]

        # Find the corresponding strategy
        if self.strategies and self.current_stage in self.strategies:
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
            f"Resuming from stage {self.current_stage} ({self.current_stage_idx+1}), epoch {self.current_epoch+1}"
        )

    def train_stage(self, stage_idx: int) -> Dict[str, Any]:
        """
        Train a specific stage.

        Args:
            stage_idx: Index of the stage to train

        Returns:
            Dictionary with training results
        """
        # Validate stage index
        stage_names = list(self.strategies.keys())
        if stage_idx >= len(stage_names):
            raise ValueError(
                f"Stage index {stage_idx} out of range (total stages: {len(stage_names)})"
            )

        # Set current stage info
        self.current_stage_idx = stage_idx
        self.current_stage = stage_names[stage_idx]
        self.current_strategy = self.strategies[self.current_stage]["strategy"]
        stage_epochs = self.strategies[self.current_stage]["epochs"]

        logger.info(
            f"Starting training for stage {stage_idx+1}/{len(stage_names)}: {self.current_stage}"
        )

        # Create stage-specific directories
        stage_checkpoint_dir = os.path.join(
            self.checkpoint_dir, f"{stage_idx+1}_{self.current_stage}"
        )
        stage_log_dir = os.path.join(
            self.log_dir, f"{stage_idx+1}_{self.current_stage}"
        )
        os.makedirs(stage_checkpoint_dir, exist_ok=True)
        os.makedirs(stage_log_dir, exist_ok=True)

        # Get or create metrics tracker for this stage
        metrics_tracker = self.metrics_trackers.get(
            self.current_stage,
            MetricsTracker(
                log_dir=stage_log_dir,
                early_stopping=self.extra_config.get("early_stopping", False),
                patience=self.extra_config.get("early_stopping_patience", 5),
                monitor=self.extra_config.get("monitor_metric", "val_loss"),
                mode=self.extra_config.get("monitor_mode", "min"),
                visualization_frequency=1,
                issue_detection=True,
            ),
        )

        # Initialize strategy if method exists
        if hasattr(self.current_strategy, "initialize_strategy"):
            self.current_strategy.initialize_strategy()

        # Train stage
        stage_metrics = self._train_stage(
            strategy=self.current_strategy,
            epochs=stage_epochs,
            checkpoint_dir=stage_checkpoint_dir,
            log_dir=stage_log_dir,
            metrics_tracker=metrics_tracker,
            global_epoch_offset=self.current_epoch,
        )

        # Store stage results
        self.stage_results[stage_idx] = {
            "stage_name": self.current_stage,
            "metrics": stage_metrics,
            "best_metric": metrics_tracker.best_value,
        }

        # Save stage completion checkpoint
        checkpoint_path = os.path.join(
            stage_checkpoint_dir, f"{self.current_stage}_complete.pt"
        )
        self._save_checkpoint(checkpoint_path, stage_metrics)

        return self.stage_results[stage_idx]

    def evaluate(self, dataloader: Optional[DataLoader] = None) -> Dict[str, float]:
        """
        Evaluate the model on a dataset.

        Args:
            dataloader: DataLoader to use (defaults to test_dataloader if not provided)

        Returns:
            Dictionary with evaluation metrics
        """
        eval_dataloader = dataloader or self.test_dataloader

        if eval_dataloader is None:
            logger.warning("No test dataloader provided, skipping evaluation")
            return {}

        logger.info("Evaluating model")

        # Use the current strategy for evaluation
        if self.current_strategy is None:
            logger.error("No current strategy available for evaluation")
            return {}

        return self._evaluate(self.current_strategy, eval_dataloader)

    def create_visualizations(self) -> None:
        """
        Create visualizations of training progress for all stages.
        """
        for stage_name, metrics_tracker in self.metrics_trackers.items():
            logger.info(f"Creating visualizations for stage: {stage_name}")
            metrics_tracker.create_visualizations()

    def save_model(self, path: str) -> None:
        """
        Save the complete model with metadata.

        Args:
            path: Path to save the model
        """
        # Create model directory if needed
        os.makedirs(os.path.dirname(path), exist_ok=True)

        # Create save dict
        model_dict = {
            "model_state_dict": self.model.state_dict(),
            "stage_results": self.stage_results,
        }

        # Add config if available
        if self.config:
            model_dict["config"] = (
                self.config.to_dict()
                if hasattr(self.config, "to_dict")
                else vars(self.config)
            )

        # Save the model
        torch.save(model_dict, path)
        logger.info(f"Model saved to {path}")
