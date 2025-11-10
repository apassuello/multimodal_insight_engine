"""MODULE: training_loop.py
PURPOSE: Handles core training loop execution with support for advanced training techniques.

KEY COMPONENTS:
- TrainingLoop: Manages epoch and batch-level training with comprehensive diagnostics
- Support for mixed precision, gradient accumulation, and gradient clipping
- Feature collapse detection and gradient diagnostics
- Integration with loss functions that have curriculum/phase tracking

DEPENDENCIES:
- PyTorch (torch, torch.nn)
- tqdm for progress tracking
- Python standard library (logging, collections)

SPECIAL NOTES:
- Supports both standard and mixed precision training
- Provides detailed gradient diagnostics for debugging
- Handles multiple loss function types (contrastive, VICReg, hybrid)
- Integrates with checkpoint and metrics systems
"""

import logging
from collections import defaultdict
from typing import Dict, Optional, Any, Callable

import torch
import torch.nn as nn
from tqdm import tqdm

logger = logging.getLogger(__name__)


class TrainingLoop:
    """Manages the core training loop execution with advanced features."""

    def __init__(
        self,
        model: nn.Module,
        loss_fn: nn.Module,
        optimizer: torch.optim.Optimizer,
        device: torch.device,
        mixed_precision: bool = False,
        accumulation_steps: int = 1,
        clip_grad_norm: Optional[float] = None,
        log_steps: int = 50,
        scheduler: Optional[Any] = None,
        enable_diagnostics: bool = True,
        check_feature_collapse: bool = True,
        grad_scheduler: Optional[Any] = None,
    ):
        """
        Initialize the training loop.

        Args:
            model: The model to train
            loss_fn: Loss function
            optimizer: Optimizer for training
            device: Device to train on
            mixed_precision: Whether to use mixed precision training
            accumulation_steps: Number of gradient accumulation steps
            clip_grad_norm: Maximum norm for gradient clipping (None = no clipping)
            log_steps: Number of steps between logging
            scheduler: Optional learning rate scheduler
            enable_diagnostics: Whether to enable gradient diagnostics
            check_feature_collapse: Whether to check for feature collapse
            grad_scheduler: Optional gradient scheduler for modality balancing
        """
        self.model = model
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.device = device
        self.mixed_precision = mixed_precision
        self.accumulation_steps = accumulation_steps
        self.clip_grad_norm = clip_grad_norm
        self.log_steps = log_steps
        self.scheduler = scheduler
        self.enable_diagnostics = enable_diagnostics
        self.check_feature_collapse = check_feature_collapse
        self.grad_scheduler = grad_scheduler

        # Initialize mixed precision scaler
        self.scaler = None
        if self.mixed_precision:
            self.scaler = torch.cuda.amp.GradScaler()

        # Tracking
        self.global_step = 0
        self.current_epoch = 0

    def train_epoch(
        self,
        dataloader: torch.utils.data.DataLoader,
        epoch: int,
        num_epochs: int,
        prepare_model_inputs_fn: Callable,
        prepare_loss_inputs_fn: Callable,
        to_device_fn: Callable,
        evaluation_fn: Optional[Callable] = None,
        evaluation_steps: int = 0,
    ) -> Dict[str, float]:
        """
        Train for one epoch.

        Args:
            dataloader: Training dataloader
            epoch: Current epoch number
            num_epochs: Total number of epochs
            prepare_model_inputs_fn: Function to prepare model inputs from batch
            prepare_loss_inputs_fn: Function to prepare loss inputs
            to_device_fn: Function to move batch to device
            evaluation_fn: Optional function to call for periodic evaluation
            evaluation_steps: Steps between evaluations (0 = no periodic eval)

        Returns:
            Dictionary with epoch metrics
        """
        self.model.train()
        self.current_epoch = epoch

        epoch_metrics = defaultdict(float)
        nested_metrics = {}
        num_batches = 0
        total_loss = 0.0

        # Update loss function epoch/step if supported
        self._update_loss_curriculum(epoch, num_epochs, len(dataloader))

        # Progress bar
        pbar = tqdm(dataloader, desc=f"Epoch {epoch + 1}/{num_epochs}")

        # Reset gradients
        self.optimizer.zero_grad()

        for batch_idx, batch in enumerate(pbar):
            # Move batch to device
            batch = to_device_fn(batch)

            # Prepare inputs
            model_inputs = prepare_model_inputs_fn(batch)

            # Forward pass
            loss, loss_dict, outputs = self._forward_pass(
                model_inputs, batch, prepare_loss_inputs_fn
            )

            # Scale loss by accumulation steps
            loss = loss / self.accumulation_steps

            # Backward pass
            self._backward_pass(loss)

            # Gradient scheduler step (for modality balancing)
            if self.grad_scheduler is not None and batch_idx % 10 == 0:
                self.grad_scheduler.step(self.model)

            # Track metrics
            loss_value = loss.item() * self.accumulation_steps
            total_loss += loss_value
            num_batches += 1

            # Check for problematic loss values
            self._check_loss_anomalies(loss, loss_dict, loss_value, batch_idx)

            # Feature collapse detection
            if self.check_feature_collapse and batch_idx % 5 == 0:
                self._check_feature_collapse(prepare_loss_inputs_fn(batch, outputs))

            # Gradient diagnostics
            if self.enable_diagnostics and (batch_idx + 1) % self.accumulation_steps == 0:
                self._analyze_gradients()

            # Update metrics
            self._update_metrics(loss_dict, pbar, epoch_metrics, nested_metrics)

            # Optimizer step
            if (batch_idx + 1) % self.accumulation_steps == 0 or batch_idx == len(dataloader) - 1:
                self._optimizer_step()

            # Periodic logging
            if self.global_step % self.log_steps == 0:
                lr = self.optimizer.param_groups[0]["lr"]
                logger.debug(f"Step {self.global_step}: loss={loss_dict['loss'].item():.4f}, lr={lr:.6f}")

            # Periodic evaluation
            if evaluation_fn and evaluation_steps > 0 and self.global_step % evaluation_steps == 0:
                val_metrics = evaluation_fn()
                self.model.train()  # Return to training mode

            self.global_step += 1

        # Calculate final epoch metrics
        epoch_metrics["loss"] = total_loss / max(1, num_batches)

        # Merge nested metrics
        result = dict(epoch_metrics)
        for k, v in nested_metrics.items():
            result[k] = dict(v)

        return result

    def _forward_pass(
        self,
        model_inputs: Dict[str, torch.Tensor],
        batch: Dict[str, Any],
        prepare_loss_inputs_fn: Callable,
    ) -> tuple:
        """
        Execute forward pass with optional mixed precision.

        Args:
            model_inputs: Inputs for the model
            batch: Original batch data
            prepare_loss_inputs_fn: Function to prepare loss inputs

        Returns:
            Tuple of (loss, loss_dict, outputs)
        """
        if self.mixed_precision and self.scaler is not None:
            with torch.cuda.amp.autocast():
                outputs = self.model(**model_inputs)
                loss_inputs = prepare_loss_inputs_fn(batch, outputs)
                loss_dict = self.loss_fn(**loss_inputs)
                loss = loss_dict["loss"]
        else:
            outputs = self.model(**model_inputs)
            loss_inputs = prepare_loss_inputs_fn(batch, outputs)
            loss_dict = self.loss_fn(**loss_inputs)
            loss = loss_dict["loss"]

        return loss, loss_dict, outputs

    def _backward_pass(self, loss: torch.Tensor) -> None:
        """
        Execute backward pass with optional mixed precision.

        Args:
            loss: Loss tensor
        """
        if self.mixed_precision and self.scaler is not None:
            self.scaler.scale(loss).backward()
        else:
            loss.backward()

    def _optimizer_step(self) -> None:
        """Execute optimizer step with gradient clipping and scheduler."""
        # Gradient clipping
        if self.clip_grad_norm is not None:
            if self.mixed_precision and self.scaler is not None:
                self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip_grad_norm)

        # Optimizer step
        if self.mixed_precision and self.scaler is not None:
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            self.optimizer.step()

        # Scheduler step
        if self.scheduler is not None:
            self.scheduler.step()

        # Update loss function step counter if supported
        if hasattr(self.loss_fn, "update_step"):
            # Note: total_steps should be passed from trainer
            self.loss_fn.update_step(self.global_step, None)

        # Reset gradients
        self.optimizer.zero_grad()

    def _update_loss_curriculum(self, epoch: int, num_epochs: int, dataloader_len: int) -> None:
        """
        Update loss function curriculum/phase if supported.

        Args:
            epoch: Current epoch
            num_epochs: Total epochs
            dataloader_len: Length of dataloader
        """
        if hasattr(self.loss_fn, "update_epoch"):
            self.loss_fn.update_epoch(epoch)

            # Set total steps for better warm-up calculation
            total_steps = dataloader_len * num_epochs
            if hasattr(self.loss_fn, "update_step"):
                self.loss_fn.update_step(self.global_step, total_steps)

            # Log phase information
            if hasattr(self.loss_fn, "current_phase"):
                phase = getattr(self.loss_fn, "current_phase", "unknown")
                logger.info(f"Training curriculum: phase={phase}, epoch={epoch}, step={self.global_step}")
            else:
                logger.info(f"Training curriculum: epoch={epoch}, step={self.global_step}")

    def _check_loss_anomalies(
        self,
        loss: torch.Tensor,
        loss_dict: Dict[str, Any],
        loss_value: float,
        batch_idx: int,
    ) -> None:
        """
        Check for problematic loss values.

        Args:
            loss: Loss tensor
            loss_dict: Dictionary of loss components
            loss_value: Scalar loss value
            batch_idx: Current batch index
        """
        if torch.isnan(loss).any().item():
            logger.error(f"NaN LOSS DETECTED at batch {batch_idx}!")
            logger.error(f"Loss breakdown: {self._format_loss_dict(loss_dict)}")
        elif torch.isinf(loss).any().item():
            logger.error(f"Infinite LOSS DETECTED at batch {batch_idx}!")
            logger.error(f"Loss breakdown: {self._format_loss_dict(loss_dict)}")
        elif loss_value > 8.0:
            logger.warning(f"Unusually HIGH LOSS: {loss_value:.4f} at batch {batch_idx}")
            for k, v in loss_dict.items():
                if k != "loss" and not isinstance(v, dict):
                    logger.info(f"  {k}: {v.item() if isinstance(v, torch.Tensor) else v:.4f}")

            # Check for near-zero accuracy in contrastive loss
            if "accuracy" in loss_dict and loss_dict["accuracy"] < 0.01:
                logger.error("NEAR-ZERO ACCURACY DETECTED - model is likely not learning!")
        elif loss_value < 1e-6 and loss_value > 0:
            logger.warning(f"Unusually LOW LOSS: {loss_value:.8f} at batch {batch_idx}")

    def _check_feature_collapse(self, loss_inputs: Dict[str, torch.Tensor]) -> None:
        """
        Check for feature collapse in multimodal embeddings.

        Args:
            loss_inputs: Loss inputs containing features
        """
        vision_features = loss_inputs.get("vision_features")
        text_features = loss_inputs.get("text_features")

        if vision_features is not None and text_features is not None:
            # Check feature variance
            vision_var = torch.var(vision_features, dim=0).mean().item()
            text_var = torch.var(text_features, dim=0).mean().item()
            logger.info(f"Feature variance - Vision: {vision_var:.6f}, Text: {text_var:.6f}")

            # Check similarity distribution
            similarity = torch.matmul(vision_features, text_features.T)
            sim_mean = similarity.mean().item()
            sim_std = similarity.std().item()
            logger.info(f"Similarity stats - Mean: {sim_mean:.4f}, Std: {sim_std:.4f}")

            # Feature collapse warning
            if vision_var < 1e-4 or text_var < 1e-4:
                logger.warning(f"FEATURE COLLAPSE DETECTED! Vision var: {vision_var:.6f}, Text var: {text_var:.6f}")

    def _analyze_gradients(self) -> None:
        """Analyze gradients for debugging purposes."""
        total_grad_norm = 0.0
        param_count = 0
        component_grads = {}

        # Track key model components
        component_prefixes = [
            "vision_model",
            "text_model",
            "fusion_module",
            "vision_projection",
            "text_projection",
            "cross_attention",
        ]

        for prefix in component_prefixes:
            component_grads[prefix] = {
                "count": 0,
                "total_norm": 0.0,
                "max_norm": 0.0,
                "min_norm": float("inf"),
                "has_zero": False,
                "has_nan": False,
            }

        # Analyze gradients by component
        for name, param in self.model.named_parameters():
            if param.grad is not None:
                param_norm = param.grad.norm().item()
                total_grad_norm += param_norm
                param_count += 1

                # Check for problematic gradients
                has_nan = torch.isnan(param.grad).any().item()
                has_inf = torch.isinf(param.grad).any().item()
                is_zero = (param.grad == 0).all().item()

                # Categorize by component
                for prefix in component_prefixes:
                    if prefix in name:
                        component_grads[prefix]["count"] += 1
                        component_grads[prefix]["total_norm"] += param_norm
                        component_grads[prefix]["max_norm"] = max(component_grads[prefix]["max_norm"], param_norm)

                        if param_norm > 0:
                            component_grads[prefix]["min_norm"] = min(component_grads[prefix]["min_norm"], param_norm)

                        if is_zero:
                            component_grads[prefix]["has_zero"] = True
                        if has_nan or has_inf:
                            component_grads[prefix]["has_nan"] = True
                        break

                # Log critical issues
                if has_nan or has_inf:
                    logger.error(f"CRITICAL: NaN/Inf detected in gradients for {name}")
                elif param_norm > 10.0:
                    logger.warning(f"Unusually LARGE gradient norm for {name}: {param_norm:.4f}")

        # Log component summary
        if param_count > 0:
            avg_grad_norm = total_grad_norm / param_count
            logger.info(f"Gradient Analysis - Overall avg: {avg_grad_norm:.4f}, {param_count} parameters")

            component_summary = []
            for prefix, stats in component_grads.items():
                if stats["count"] > 0:
                    avg_norm = stats["total_norm"] / stats["count"]
                    status = "✓"

                    if stats["has_nan"]:
                        status = "❌ NaN"
                    elif stats["has_zero"]:
                        status = "⚠️ ZeroGrad"
                    elif avg_norm < 1e-5:
                        status = "⚠️ VerySmall"
                    elif avg_norm > 5.0:
                        status = "⚠️ VeryLarge"

                    component_summary.append((prefix, avg_norm, stats["count"], status))

            component_summary.sort(key=lambda x: x[1], reverse=True)

            for prefix, avg_norm, count, status in component_summary:
                logger.info(f"  {prefix}: avg={avg_norm:.4f}, params={count} {status}")

            # Check for gradient imbalance
            if len(component_summary) > 1:
                highest_avg = component_summary[0][1]
                lowest_avg = component_summary[-1][1]
                if lowest_avg > 0:
                    gradient_ratio = highest_avg / lowest_avg
                    logger.info(f"  Gradient ratio (highest/lowest): {gradient_ratio:.1f}x")

                    if gradient_ratio > 1000:
                        logger.warning(f"SEVERE GRADIENT IMBALANCE: {gradient_ratio:.1f}x")
                    elif gradient_ratio > 100:
                        logger.warning(f"GRADIENT IMBALANCE: {gradient_ratio:.1f}x")

    def _update_metrics(
        self,
        loss_dict: Dict[str, Any],
        pbar: tqdm,
        epoch_metrics: Dict[str, float],
        nested_metrics: Dict[str, Dict[str, float]],
    ) -> None:
        """
        Update metrics and progress bar.

        Args:
            loss_dict: Loss dictionary
            pbar: Progress bar
            epoch_metrics: Epoch metrics accumulator
            nested_metrics: Nested metrics accumulator
        """
        # Update metrics
        for k, v in loss_dict.items():
            if k != "loss":
                if isinstance(v, dict):
                    if k not in nested_metrics:
                        nested_metrics[k] = defaultdict(float)
                    for sub_k, sub_v in v.items():
                        nested_metrics[k][sub_k] += sub_v / len(pbar)
                else:
                    epoch_metrics[k] += v / len(pbar)

        # Update progress bar based on loss type
        postfix = {"loss": loss_dict["loss"].item()}

        if "accuracy" in loss_dict:
            postfix["acc"] = loss_dict.get("accuracy", 0.0)

        if "current_phase" in loss_dict:
            postfix["phase"] = loss_dict["current_phase"]

        if "warmup_factor" in loss_dict:
            postfix["warmup"] = loss_dict.get("warmup_factor", 0.0)

        pbar.set_postfix(postfix)

    def _format_loss_dict(self, loss_dict: Dict[str, Any]) -> str:
        """Format loss dictionary for logging."""
        parts = []
        for k, v in loss_dict.items():
            if k != "loss" and not isinstance(v, dict):
                val = v.item() if isinstance(v, torch.Tensor) else v
                parts.append(f"{k}={val:.4f}")
        return ", ".join(parts)

    def set_global_step(self, step: int) -> None:
        """Set the global step counter."""
        self.global_step = step

    def set_current_epoch(self, epoch: int) -> None:
        """Set the current epoch."""
        self.current_epoch = epoch
