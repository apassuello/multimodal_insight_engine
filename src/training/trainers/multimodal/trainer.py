"""MODULE: trainer.py
PURPOSE: Main coordinator for multimodal model training using modular components.

This is the refactored trainer that delegates responsibilities to specialized modules:
- CheckpointManager: Handles model checkpointing and state persistence
- MetricsCollector: Tracks and visualizes training metrics
- TrainingLoop: Executes the core training loop with mixed precision
- Evaluator: Computes evaluation metrics on validation/test sets
- DataHandler: Manages data preprocessing and device placement

KEY COMPONENTS:
- ModalityBalancingScheduler: Balances learning rates between vision and text modalities
- MultimodalTrainer: Main orchestrator class that coordinates all modules

DEPENDENCIES:
- All 5 extracted modules from .multimodal package
- PyTorch for model training
- Logging for progress tracking
"""

# Standard library imports
import os
import time
import logging
from collections import defaultdict
from typing import Dict, List, Optional, Any

# Third-party imports
import torch
import torch.nn as nn
from torch.optim.adamw import AdamW
from torch.utils.data import DataLoader

# Local imports
from .checkpoint_manager import CheckpointManager
from .metrics_collector import MetricsCollector
from .training_loop import TrainingLoop
from .evaluation import Evaluator
from .data_handler import DataHandler
from src.training.losses import (
    MultiModalMixedContrastiveLoss,
)
from src.data.tokenization.tokenizer_metrics import log_tokenizer_evaluation

logger = logging.getLogger(__name__)


class ModalityBalancingScheduler:
    """
    Dynamically balances learning rates between vision and text modalities
    based on gradient magnitudes to prevent one modality from dominating.
    """

    def __init__(self, optimizer, target_ratio=1.0, check_interval=10):
        """
        Initialize the modality balancing scheduler.

        Args:
            optimizer: The optimizer to adjust
            target_ratio: Target text/vision gradient ratio (default: 1.0 for balanced)
            check_interval: Steps between gradient checks and adjustments
        """
        self.optimizer = optimizer
        self.target_ratio = target_ratio
        self.check_interval = check_interval
        self.vision_grad_history = []
        self.text_grad_history = []
        self.step_count = 0

    def collect_gradient_stats(self, model):
        """
        Collect gradient statistics for vision and text components.

        Args:
            model: The model to analyze

        Returns:
            Tuple of (avg_vision_grad, avg_text_grad)
        """
        vision_grad_norm = 0.0
        vision_count = 0
        text_grad_norm = 0.0
        text_count = 0

        for name, param in model.named_parameters():
            if param.grad is not None:
                if "vision_model" in name:
                    vision_grad_norm += param.grad.norm().item()
                    vision_count += 1
                elif "text_model" in name:
                    text_grad_norm += param.grad.norm().item()
                    text_count += 1

        # Calculate average gradient norms
        avg_vision_grad = vision_grad_norm / max(1, vision_count)
        avg_text_grad = text_grad_norm / max(1, text_count)

        self.vision_grad_history.append(avg_vision_grad)
        self.text_grad_history.append(avg_text_grad)

        # Keep history limited to recent gradients
        if len(self.vision_grad_history) > 100:
            self.vision_grad_history = self.vision_grad_history[-100:]
            self.text_grad_history = self.text_grad_history[-100:]

        return avg_vision_grad, avg_text_grad

    def step(self, model):
        """
        Adjust learning rates based on gradient ratio.

        Args:
            model: The model being trained
        """
        self.step_count += 1

        # Only adjust every check_interval steps
        if self.step_count % self.check_interval != 0:
            return

        # Get gradient statistics
        avg_vision_grad, avg_text_grad = self.collect_gradient_stats(model)

        # Calculate current ratio
        if avg_vision_grad > 0:
            current_ratio = avg_text_grad / avg_vision_grad
        else:
            current_ratio = self.target_ratio

        # Only adjust if ratio is far from target
        if abs(current_ratio - self.target_ratio) > 0.5:
            # Calculate adjustment factor
            adjustment = self.target_ratio / max(current_ratio, 0.1)

            # Limit adjustment to avoid extreme changes
            adjustment = max(0.5, min(2.0, adjustment))

            # Adjust learning rates
            for param_group in self.optimizer.param_groups:
                if "vision_model" in str(param_group.get("name", "")):
                    param_group["lr"] *= adjustment
                elif "text_model" in str(param_group.get("name", "")):
                    param_group["lr"] /= adjustment

            logger.info(
                f"Adjusted learning rates - ratio: {current_ratio:.2f}, "
                f"target: {self.target_ratio:.2f}"
            )


class MultimodalTrainer:
    """
    Main trainer coordinator for multimodal models.

    This class orchestrates training by delegating to specialized modules:
    - CheckpointManager for saving/loading state
    - MetricsCollector for tracking metrics
    - TrainingLoop for executing training steps
    - Evaluator for computing evaluation metrics
    - DataHandler for data preprocessing
    """

    def __init__(
        self,
        model: nn.Module,
        train_dataloader: DataLoader,
        val_dataloader: Optional[DataLoader] = None,
        test_dataloader: Optional[DataLoader] = None,
        optimizer: Optional[Any] = None,
        scheduler: Optional[Any] = None,
        loss_fn: Optional[nn.Module] = None,
        num_epochs: int = 20,
        learning_rate: float = 1e-4,
        weight_decay: float = 0.01,
        warmup_steps: int = 0,
        checkpoint_dir: str = "checkpoints",
        log_dir: str = "logs",
        device: Optional[torch.device] = None,
        mixed_precision: bool = False,
        accumulation_steps: int = 1,
        evaluation_steps: int = 0,
        log_steps: int = 50,
        early_stopping_patience: Optional[int] = None,
        clip_grad_norm: Optional[float] = None,
        balance_modality_gradients: bool = False,
        args: Optional[Any] = None,
    ):
        """
        Initialize the multimodal trainer.

        Args:
            model: The multimodal model to train
            train_dataloader: DataLoader for training data
            val_dataloader: Optional DataLoader for validation
            test_dataloader: Optional DataLoader for test
            optimizer: Optional optimizer (created if not provided)
            scheduler: Optional learning rate scheduler
            loss_fn: Optional loss function (defaults to MultiModalMixedContrastiveLoss)
            num_epochs: Number of epochs to train
            learning_rate: Learning rate for optimizer
            weight_decay: Weight decay for optimizer
            warmup_steps: Number of warmup steps for learning rate
            checkpoint_dir: Directory to save checkpoints
            log_dir: Directory to save logs
            device: Device to train on (auto-detected if not provided)
            mixed_precision: Whether to use mixed precision training
            accumulation_steps: Number of steps to accumulate gradients over
            evaluation_steps: Steps between evaluations during training (0 = end of epoch only)
            log_steps: Steps between logging during training
            early_stopping_patience: Evaluations with no improvement to trigger early stopping
            clip_grad_norm: Maximum norm for gradient clipping
            balance_modality_gradients: Whether to balance gradients between modalities
            args: Original argument namespace for additional configuration
        """
        self.model = model
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.test_dataloader = test_dataloader
        self.scheduler = scheduler
        self.num_epochs = num_epochs
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.warmup_steps = warmup_steps
        self.checkpoint_dir = checkpoint_dir
        self.log_dir = log_dir
        self.evaluation_steps = evaluation_steps
        self.log_steps = log_steps
        self.early_stopping_patience = early_stopping_patience
        self.args = args

        # Create directories
        os.makedirs(checkpoint_dir, exist_ok=True)
        os.makedirs(log_dir, exist_ok=True)

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

        logger.info(f"Initialized MultimodalTrainer with device: {self.device}")

        # Initialize DataHandler
        self.data_handler = DataHandler(model, self.device)
        self.data_handler.ensure_model_on_device()

        # Initialize optimizer if not provided
        if optimizer is None:
            optimizer = AdamW(
                self.model.parameters(),
                lr=self.learning_rate,
                weight_decay=self.weight_decay,
            )
        self.optimizer = optimizer

        # Initialize modality balancing scheduler if requested
        self.grad_scheduler = None
        if balance_modality_gradients:
            self.grad_scheduler = ModalityBalancingScheduler(
                self.optimizer, target_ratio=1.0
            )

        # Initialize loss function if not provided
        if loss_fn is None:
            logger.info("Creating MultiModalMixedContrastiveLoss with default configuration")
            loss_fn = MultiModalMixedContrastiveLoss(
                temperature=0.2,
                loss_weights=None,  # Use default weights
                add_projection=False,  # Model already has projections
            )
        self.loss_fn = loss_fn

        # Initialize all 5 modules
        self.checkpoint_manager = CheckpointManager(
            model=self.model,
            optimizer=self.optimizer,
            checkpoint_dir=self.checkpoint_dir,
            scheduler=self.scheduler,
        )

        self.metrics_collector = MetricsCollector()

        self.training_loop = TrainingLoop(
            model=self.model,
            loss_fn=self.loss_fn,
            optimizer=self.optimizer,
            device=self.device,
            mixed_precision=mixed_precision,
            accumulation_steps=accumulation_steps,
            clip_grad_norm=clip_grad_norm,
            log_steps=log_steps,
        )

        self.evaluator = Evaluator(
            model=self.model,
            device=self.device,
        )

        # Initialize counters
        self.start_time = time.time()
        self.best_val_metric = 0.0
        self.patience_counter = 0

        logger.info(f"Mixed precision: {mixed_precision}")
        if accumulation_steps > 1:
            logger.info(f"Accumulation steps: {accumulation_steps}")

    def _detect_model_dimension(self) -> int:
        """
        Detect the appropriate dimension for loss functions.

        Returns:
            Model dimension for loss function projections
        """
        # Try fusion dimension first
        if hasattr(self.model, "fusion_module") and hasattr(
            self.model.fusion_module, "fusion_dim"
        ):
            return self.model.fusion_module.fusion_dim

        # Try vision model dimensions
        if hasattr(self.model, "vision_model"):
            if hasattr(self.model.vision_model, "num_features"):
                return self.model.vision_model.num_features
            if hasattr(self.model.vision_model, "embed_dim"):
                return self.model.vision_model.embed_dim

        # Try text model dimensions
        if hasattr(self.model, "text_model"):
            if (
                hasattr(self.model.text_model, "encoder")
                and hasattr(self.model.text_model.encoder, "config")
                and hasattr(self.model.text_model.encoder.config, "hidden_size")
            ):
                return self.model.text_model.encoder.config.hidden_size
            if hasattr(self.model.text_model, "d_model"):
                return self.model.text_model.d_model

        # Default fallback
        logger.warning("Could not detect model dimension, using default 768")
        return 768

    def train(self) -> Dict[str, List[float]]:
        """
        Train the model for the specified number of epochs.

        Returns:
            Dictionary with training history
        """
        logger.info(f"Starting training for {self.num_epochs} epochs")

        # Training loop
        for epoch in range(self.num_epochs):
            # Train for one epoch
            train_metrics = self.training_loop.train_epoch(
                dataloader=self.train_dataloader,
                epoch=epoch,
                num_epochs=self.num_epochs,
                prepare_model_inputs_fn=self.data_handler.prepare_model_inputs,
                prepare_loss_inputs_fn=self.data_handler.prepare_loss_inputs,
                to_device_fn=self.data_handler.to_device,
                evaluation_fn=(
                    lambda: self.evaluator.evaluate(
                        self.val_dataloader,
                        self.data_handler.prepare_model_inputs,
                        self.data_handler.to_device,
                    )
                    if self.val_dataloader
                    else None
                ),
                evaluation_steps=self.evaluation_steps,
            )

            # Log and store training metrics
            self.metrics_collector.log_metrics(train_metrics, prefix="train")

            # Evaluate on validation set if available
            if self.val_dataloader:
                val_metrics = self.evaluator.evaluate(
                    self.val_dataloader,
                    self.data_handler.prepare_model_inputs,
                    self.data_handler.to_device,
                )
                self.metrics_collector.log_metrics(val_metrics, prefix="val")

                # Check for early stopping
                if self._check_early_stopping(val_metrics, epoch):
                    break

            # Save checkpoint for this epoch
            self.checkpoint_manager.save_checkpoint(
                os.path.join(self.checkpoint_dir, f"checkpoint_epoch_{epoch + 1}.pt"),
                current_epoch=epoch,
                global_step=self.training_loop.global_step,
                best_val_metric=self.best_val_metric,
                patience_counter=self.patience_counter,
                history=self.metrics_collector.to_dict(),
            )

            # Run diagnostics after first epoch
            if epoch > 0:
                logger.info("Running training diagnostics...")
                diagnostic_report = self.metrics_collector.diagnose_training_issues()
                logger.info(f"\n{diagnostic_report}")

                # Evaluate tokenizer quality if applicable
                self._evaluate_tokenizer_quality(epoch)

                # Plot metrics
                self.metrics_collector.plot_history(save_dir=self.log_dir)

            # Apply modality balancing if enabled
            if self.grad_scheduler is not None:
                self.grad_scheduler.step(self.model)

        # Training completed
        logger.info(
            f"Training completed in {time.time() - self.start_time:.2f} seconds"
        )

        return self.metrics_collector.to_dict()

    def _check_early_stopping(self, val_metrics: Dict[str, float], epoch: int) -> bool:
        """
        Check if early stopping should be triggered.

        Args:
            val_metrics: Validation metrics
            epoch: Current epoch number

        Returns:
            True if training should stop, False otherwise
        """
        if self.early_stopping_patience is None:
            return False

        validation_metric = val_metrics.get("loss", float("inf"))

        if validation_metric < self.best_val_metric or epoch == 0:
            self.best_val_metric = validation_metric
            self.patience_counter = 0
            # Save best model
            self.checkpoint_manager.save_best_checkpoint(
                metric_value=validation_metric,
                current_epoch=epoch,
                global_step=self.training_loop.global_step,
                history=self.metrics_collector.to_dict(),
            )
            logger.info(f"New best validation metric: {validation_metric:.4f}")
        else:
            self.patience_counter += 1
            logger.info(
                f"Validation metric did not improve, "
                f"counter: {self.patience_counter}/{self.early_stopping_patience}"
            )
            if self.patience_counter >= self.early_stopping_patience:
                logger.info(f"Early stopping triggered after {epoch + 1} epochs")
                return True

        return False

    def _evaluate_tokenizer_quality(self, epoch: int):
        """
        Evaluate tokenizer quality if the model has a text tokenizer.

        Args:
            epoch: Current epoch number
        """
        if not hasattr(self.model, "text_model") or not hasattr(
            self.model.text_model, "tokenizer"
        ):
            return

        try:
            tokenizer = self.model.text_model.tokenizer
            log_tokenizer_evaluation(
                tokenizer,
                epoch=epoch,
                log_dir=self.log_dir,
            )
        except Exception as e:
            logger.warning(f"Could not evaluate tokenizer: {str(e)}")

    def evaluate_test(self) -> Dict[str, float]:
        """
        Evaluate the model on the test set.

        Returns:
            Dictionary of test metrics
        """
        if self.test_dataloader is None:
            logger.warning("No test dataloader provided")
            return {}

        logger.info("Evaluating on test set...")
        test_metrics = self.evaluator.evaluate(
            self.test_dataloader,
            self.data_handler.prepare_model_inputs,
            self.data_handler.to_device,
        )
        self.metrics_collector.log_metrics(test_metrics, prefix="test")
        return test_metrics

    def load_checkpoint(self, checkpoint_path: str):
        """
        Load a checkpoint and restore training state.

        Args:
            checkpoint_path: Path to checkpoint file
        """
        state = self.checkpoint_manager.load_checkpoint(checkpoint_path)

        # Restore trainer state
        if "current_epoch" in state:
            self.training_loop.current_epoch = state["current_epoch"]
        if "global_step" in state:
            self.training_loop.global_step = state["global_step"]
        if "best_val_metric" in state:
            self.best_val_metric = state["best_val_metric"]
        if "patience_counter" in state:
            self.patience_counter = state["patience_counter"]
        if "history" in state:
            self.metrics_collector.from_dict(state["history"])

        logger.info(f"Loaded checkpoint from {checkpoint_path}")
