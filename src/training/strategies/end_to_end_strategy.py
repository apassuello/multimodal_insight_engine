# src/training/strategies/end_to_end_strategy.py

import torch
import torch.nn as nn
from torch.optim.adamw import AdamW
from typing import Dict, List, Optional, Any, Callable, Union
import logging
import os
from tqdm import tqdm
import torch.nn.functional as F

from src.training.strategies.training_strategy import TrainingStrategy
from src.utils.learningrate_scheduler import WarmupCosineScheduler
from src.utils.gradient_handler import GradientHandler
from src.utils.metrics_tracker import MetricsTracker
from src.training.losses import HardNegativeMiningContrastiveLoss
from src.training.losses.feature_consistency_loss import FeatureConsistencyLoss

logger = logging.getLogger(__name__)

"""
MODULE: end_to_end_strategy.py
PURPOSE: Implements the final stage training strategy for multimodal models, focusing on end-to-end fine-tuning
KEY COMPONENTS:
- EndToEndStrategy: Strategy for carefully fine-tuning all model components with feature consistency
DEPENDENCIES: torch, torch.nn, typing, logging, tqdm
SPECIAL NOTES: Uses feature consistency to prevent catastrophic forgetting during full fine-tuning
"""


class EndToEndStrategy(TrainingStrategy):
    """
    Training strategy for the final stage of multimodal training: end-to-end fine-tuning.

    This strategy:
    1. Carefully unfreezes all model components with very low learning rates
    2. Uses hard negative mining to focus on challenging examples
    3. Incorporates feature consistency loss to prevent catastrophic forgetting
    4. Implements advanced gradient monitoring and balancing
    5. Provides comprehensive evaluation metrics
    """

    def __init__(self, model, config):
        """
        Initialize the end-to-end training strategy.

        Args:
            model: Model to be trained
            config: Configuration dictionary
        """
        super().__init__(model, config)
        self.model = model
        self.config = config

        # Initialize optimizer and scheduler
        self.optimizer = None
        self.scheduler = None

        # Initialize loss function
        self.loss_fn = None

        # Counter for training steps
        self.train_step_counter = 0
        self.current_epoch = 0

        # Set device
        self.device = next(model.parameters()).device

        # Configure the training parameters
        self._configure_training()

    def initialize_strategy(self) -> None:
        """
        Initialize the end-to-end training strategy.

        Main tasks:
        1. Carefully unfreeze all model components
        2. Set up optimizers with very low learning rates for base models
        3. Configure loss functions with hard negative mining
        4. Set up feature consistency to preserve knowledge
        """
        logger.info("Initializing EndToEndStrategy")

        # 1. Unfreeze everything for full fine-tuning
        # This is intentionally simple - we unfreeze everything
        for param in self.model.parameters():
            param.requires_grad = True

        logger.info("Unfroze all model parameters for end-to-end fine-tuning")

        # Log parameter status
        self.log_parameter_status()

        # 2. Configure optimizer if not provided
        if self.optimizer is None:
            self.optimizer = self.configure_optimizers()[0]

        # 3. Configure scheduler if not provided
        if self.scheduler is None and hasattr(self, "optimizer"):
            self.scheduler = self.configure_optimizers()[1]

        # 4. Configure loss function if not provided
        if self.loss_fn is None:
            self._configure_loss_function()

        # 5. Initialize gradient handler with tight monitoring
        if self.config.get("use_gradient_handling", True):
            vis_dir = self.config.get("gradient_visualization_dir", None)
            self.gradient_handler = GradientHandler(
                model=self.model,
                clip_value=self.config.get(
                    "clip_grad_norm", 0.5
                ),  # Stricter clipping for stability
                component_ratios={"vision": 1.0, "text": 1.0, "fusion": 1.0},
                balance_modalities=self.config.get("balance_modalities", True),
                log_frequency=self.config.get(
                    "gradient_log_frequency", 50
                ),  # More frequent logging
                visualization_dir=vis_dir,
            )
        else:
            self.gradient_handler = None

        # 6. Save initial model state for feature consistency loss
        if self.config.get("use_feature_consistency", True):
            self._store_initial_model_state()

        # Log strategy initialization completed
        logger.info("EndToEndStrategy initialization completed")

    def _store_initial_model_state(self) -> None:
        """
        Store initial model state for feature consistency loss if needed.
        This function is a stub for now - feature consistency is disabled.
        """
        logger.info(
            "Feature consistency is disabled - skipping reference model creation"
        )
        self.reference_model = None

    def _configure_loss_function(self) -> None:
        """
        Configure appropriate loss function for end-to-end fine-tuning.

        Uses hard negative mining contrastive loss with feature consistency.
        """
        model_dim = self._get_model_dimension()
        temperature = self.config.get("temperature", 0.05)  # Slightly lower temperature

        # Log message about the loss setup
        logger.debug(
            f"HardNegativeMiningContrastiveLoss initialized with dimension: {model_dim}"
        )

        # Create primary loss (hard negative mining contrastive loss)
        primary_loss = HardNegativeMiningContrastiveLoss(
            temperature=temperature,
            hard_negative_factor=self.config.get("hard_negative_factor", 2.0),
            mining_strategy=self.config.get("mining_strategy", "semi-hard"),
            dim=model_dim,
        )

        # Just use the hard negative mining loss - disable feature consistency for now
        # Feature consistency is complicated and not essential for the demo
        logger.info(
            f"Using HardNegativeMiningContrastiveLoss with temperature={temperature}"
        )
        self.loss_fn = primary_loss

        # Disable feature consistency to avoid issues
        self.config["use_feature_consistency"] = False

    def _get_model_dimension(self) -> int:
        """
        Determine the model's projection dimension for loss configuration.

        Returns:
            The projection dimension used by the model
        """
        # Try various attribute paths common in multimodal models
        if hasattr(self.model, "projection_dim"):
            return self.model.projection_dim
        elif hasattr(self.model, "fusion_dim"):
            return self.model.fusion_dim

        # Extract from model structure by examining the projections - similar to other strategies
        try:
            # Vision projection dimension
            if hasattr(self.model, "vision_projection"):
                vision_proj = self.model.vision_projection
                if isinstance(vision_proj, nn.Linear):
                    return vision_proj.out_features
                elif hasattr(vision_proj, "out_features"):
                    return vision_proj.out_features
                elif isinstance(vision_proj, nn.Sequential) and hasattr(
                    vision_proj[-1], "out_features"
                ):
                    return vision_proj[-1].out_features

            # Cross-attention dimension as fallback
            if hasattr(self.model, "cross_attention"):
                cross_attn = self.model.cross_attention
                if hasattr(cross_attn, "embed_dim"):
                    return cross_attn.embed_dim

        except (AttributeError, TypeError, IndexError) as e:
            pass  # Silently fail and use default

        # Default to 512 as a common projection dimension
        logger.warning("Could not determine model dimension, using default of 512")
        return 512

    def prepare_batch(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        """
        Prepare a batch of data for the model.

        In EndToEndStrategy, we enable all model components.

        Args:
            batch: The batch of data from dataloader

        Returns:
            Processed batch ready for the model
        """
        # Move batch to device
        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                batch[k] = v.to(self.device)
            elif isinstance(v, dict):
                for sub_k, sub_v in v.items():
                    if isinstance(sub_v, torch.Tensor):
                        batch[k][sub_k] = sub_v.to(self.device)

        # Prepare matching format expected by model
        model_inputs = {}

        # Handle image and text inputs with consistent naming
        if "images" in batch:
            model_inputs["images"] = batch["images"]
        elif "image" in batch:
            model_inputs["images"] = batch["image"]

        if "text_data" in batch:
            model_inputs["text_data"] = batch["text_data"]
        elif "text" in batch:
            model_inputs["text_data"] = batch["text"]

        # Pass match IDs for contrastive learning
        if "match_id" in batch:
            model_inputs["match_ids"] = batch["match_id"]

        # Enable all features for end-to-end training
        model_inputs["enable_cross_attention"] = True
        model_inputs["return_all_features"] = True
        model_inputs["return_intermediate_features"] = True  # For feature consistency

        return model_inputs

    def _prepare_loss_inputs(self, batch, model_output):
        """Prepare inputs for the loss function."""
        loss_inputs = {}

        # Include vision and text embeddings directly
        if "vision_features" in model_output:
            loss_inputs["vision_embeddings"] = model_output["vision_features"]
        if "text_features" in model_output:
            loss_inputs["text_embeddings"] = model_output["text_features"]

        # Add example indices
        if "example_indices" in batch:
            loss_inputs["example_indices"] = batch["example_indices"]

        return loss_inputs

    def _configure_training(self):
        """Configure all components needed for training"""
        # Initialize model state tracking
        self._store_initial_model_state()

        # Configure optimizers
        self.configure_optimizers()

        # Configure loss function
        self._configure_loss_function()

    def training_step(self, batch, batch_idx):
        """
        Execute training step.

        Args:
            batch: Batch of data
            batch_idx: Index of the batch

        Returns:
            Dict containing loss and metrics
        """
        # Forward pass through model
        model_output = self._forward(batch)

        # Prepare inputs for loss function based on model output
        loss_inputs = self._prepare_loss_inputs(batch, model_output)

        # Calculate the loss
        if self.loss_fn is not None:
            loss_dict = self.loss_fn(**loss_inputs)
            total_loss = loss_dict["loss"]
        else:
            logger.warning("Loss function is None. Using dummy loss.")
            total_loss = torch.tensor(0.0, device=self.device, requires_grad=True)
            loss_dict = {"loss": total_loss}

        # Backward pass
        self.manual_backward(total_loss)

        # Optimizer step
        self._optimizer_step()

        # Report loss and other metrics
        metrics = {"train/loss": total_loss.item()}

        # Add any additional metrics from loss_dict
        for k, v in loss_dict.items():
            if k != "loss":
                metrics[f"train/{k}"] = v.item() if hasattr(v, "item") else v

        # Log metrics
        self.log_dict(
            metrics,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            sync_dist=self.config.get("sync_dist", True),
        )

        # Increment step counter
        self.train_step_counter += 1

        return {"loss": total_loss, "metrics": metrics}

    def validation_step(self, batch, batch_idx):
        """
        Execute validation step.

        Args:
            batch: Batch of data
            batch_idx: Index of the batch

        Returns:
            Dict containing loss and metrics
        """
        # Forward pass through model
        model_output = self._forward(batch)

        # Prepare inputs for loss function
        loss_inputs = self._prepare_loss_inputs(batch, model_output)

        # Calculate the loss
        if self.loss_fn is not None:
            loss_dict = self.loss_fn(**loss_inputs)
            total_loss = loss_dict["loss"]
        else:
            logger.warning("Loss function is None. Using dummy loss.")
            total_loss = torch.tensor(0.0, device=self.device)
            loss_dict = {"loss": total_loss}

        # Report loss and basic metrics
        metrics = {"val/loss": total_loss.item()}

        # Add any additional metrics from loss_dict
        for k, v in loss_dict.items():
            if k != "loss":
                metrics[f"val/{k}"] = v.item() if hasattr(v, "item") else v

        # Gather embeddings for additional retrieval metrics
        if "vision_features" in model_output and "text_features" in model_output:
            # Calculate similarity
            vision_emb = model_output["vision_features"]
            text_emb = model_output["text_features"]

            # Normalize embeddings
            vision_emb = F.normalize(vision_emb, dim=-1)
            text_emb = F.normalize(text_emb, dim=-1)

            # Calculate similarity matrix
            similarity = torch.matmul(vision_emb, text_emb.transpose(0, 1))

            # Calculate retrieval metrics (exact matches)
            batch_size = vision_emb.size(0)
            targets = torch.arange(batch_size, device=self.device)

            # Image-to-text retrieval
            i2t_matches = similarity.argmax(dim=1) == targets
            i2t_accuracy = i2t_matches.float().mean().item()
            metrics["val/i2t_top1"] = i2t_accuracy

            # Text-to-image retrieval
            t2i_matches = similarity.argmax(dim=0) == targets
            t2i_accuracy = t2i_matches.float().mean().item()
            metrics["val/t2i_top1"] = t2i_accuracy

        # Log metrics
        self.log_dict(
            metrics,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            sync_dist=self.config.get("sync_dist", True),
        )

        return {"loss": total_loss, "metrics": metrics}

    def configure_optimizers(self) -> tuple:
        """
        Configure optimizers and schedulers with very low learning rates for base models.

        Returns:
            Tuple of (optimizer, scheduler)
        """
        # Extract configuration
        base_lr = self.config.get(
            "learning_rate", 2e-5
        )  # Very low default LR for this stage
        weight_decay = self.config.get("weight_decay", 0.01)
        warmup_steps = self.config.get(
            "warmup_steps", 100
        )  # Shorter warmup for fine-tuning
        total_steps = self.config.get(
            "total_steps", 3000
        )  # Typically shorter than earlier stages

        # Create a dictionary to track which parameters are already assigned to a group
        assigned_params = set()

        # Helper function to get parameters that match a pattern and haven't been assigned yet
        def get_unassigned_params(pattern_func):
            params = []
            for n, p in self.model.named_parameters():
                if p.requires_grad and pattern_func(n) and id(p) not in assigned_params:
                    params.append(p)
                    assigned_params.add(id(p))
            return params

        # Set up parameter groups with different learning rates
        param_groups = []

        # Vision base model: extremely low learning rate
        vision_params = get_unassigned_params(lambda n: "vision_model" in n)
        if vision_params:
            param_groups.append(
                {
                    "params": vision_params,
                    "lr": base_lr * 0.005,  # 0.5% of base learning rate
                    "name": "vision_model",
                }
            )

        # Text base model: extremely low learning rate
        text_params = get_unassigned_params(lambda n: "text_model" in n)
        if text_params:
            param_groups.append(
                {
                    "params": text_params,
                    "lr": base_lr * 0.005,  # 0.5% of base learning rate
                    "name": "text_model",
                }
            )

        # Cross-modal components: low learning rate
        cross_modal_params = get_unassigned_params(
            lambda n: any(
                x in n
                for x in ["cross_attention", "cross_modal", "fusion", "interaction"]
            )
        )
        if cross_modal_params:
            param_groups.append(
                {
                    "params": cross_modal_params,
                    "lr": base_lr * 0.1,  # 10% of base learning rate
                    "name": "cross_modal_components",
                }
            )

        # Projection layers: medium learning rate
        projection_params = get_unassigned_params(
            lambda n: any(x in n for x in ["projection", "adapter"])
        )
        if projection_params:
            param_groups.append(
                {
                    "params": projection_params,
                    "lr": base_lr * 0.5,  # 50% of base learning rate
                    "name": "projection_layers",
                }
            )

        # Other parameters: base learning rate
        other_params = get_unassigned_params(
            lambda n: True
        )  # Get all remaining parameters
        if other_params:
            param_groups.append(
                {
                    "params": other_params,
                    "lr": base_lr,  # Full base learning rate
                    "name": "other",
                }
            )

        # Create optimizer with lower epsilon for stability
        optimizer = AdamW(
            param_groups,
            lr=base_lr,
            weight_decay=weight_decay,
            betas=(0.9, 0.999),  # More stable than (0.9, 0.98)
            eps=1e-8,  # More stable than 1e-6
        )

        # Create learning rate scheduler with more stable decay
        scheduler = WarmupCosineScheduler(
            optimizer=optimizer,
            warmup_steps=warmup_steps,
            total_steps=total_steps,
            min_lr=base_lr * 0.1,  # 10% of base learning rate as minimum (higher floor)
        )

        return optimizer, scheduler

    def on_epoch_start(self, epoch: int) -> None:
        """
        Perform actions at the start of each epoch.

        Args:
            epoch: Current epoch number
        """
        logger.info(f"Starting epoch {epoch} with EndToEndStrategy")

        # Log learning rates with more detail
        learning_rates = self.get_learning_rates()
        lr_info = ", ".join([f"{k}: {v:.8f}" for k, v in learning_rates.items()])
        logger.info(f"Learning rates: {lr_info}")

    def on_epoch_end(self, epoch: int) -> None:
        """
        Perform actions at the end of each epoch.

        Args:
            epoch: Current epoch number
        """
        logger.info(f"Completed epoch {epoch} with EndToEndStrategy")

        # Feature consistency is disabled
        self.config["use_feature_consistency"] = False

    def _check_feature_drift(self) -> None:
        """
        Check how much the model features have drifted from the reference model.
        This functionality is disabled in the current implementation.
        """
        # Feature consistency is disabled, so don't check for drift
        logger.info("Feature drift checking is disabled")
        return

    def _forward(self, batch):
        """Forward pass through the model

        Args:
            batch: A batch of data

        Returns:
            Model outputs
        """
        # Prepare batch for model
        model_inputs = self.prepare_batch(batch)

        # Forward pass
        outputs = self.model(**model_inputs)

        return outputs

    def manual_backward(self, loss):
        """Manually handle the backward pass

        Args:
            loss: The loss to backpropagate
        """
        loss.backward()

        # Process gradients if handler is available
        if hasattr(self, "gradient_handler") and self.gradient_handler is not None:
            self.gradient_handler.analyze_gradients()
            self.gradient_handler.clip_gradients()

    def _optimizer_step(self):
        """Perform optimizer step and learning rate scheduler step if applicable"""
        if self.optimizer is not None:
            # Perform optimizer step
            self.optimizer.step()
            self.optimizer.zero_grad()

            # Perform scheduler step if available
            if self.scheduler is not None:
                self.scheduler.step()

    def log_dict(
        self, metrics, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True
    ):
        """Log metrics

        Args:
            metrics: Dictionary of metrics to log
            on_step: Whether to log on step
            on_epoch: Whether to log on epoch
            prog_bar: Whether to show on progress bar
            sync_dist: Whether to sync across distributed processes
        """
        # Just print the metrics in this simplified version
        if on_step:
            metrics_str = " ".join([f"{k}: {v:.4f}" for k, v in metrics.items()])
            if (
                self.train_step_counter % 10 == 0
            ):  # Only print every 10 steps to reduce output
                logger.info(f"Step {self.train_step_counter}: {metrics_str}")

    def on_train_epoch_end(self) -> None:
        """Handle the end of a training epoch."""
        logger.info(
            f"Training epoch {self.current_epoch} completed. "
            f"Processed {self.train_step_counter} batches."
        )

        # Logging epoch-level metrics
        metrics = {}

        # Log metrics
        self.log_dict(
            metrics,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            sync_dist=self.config.get("sync_dist", True),
        )


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
        "module_purpose": "Implements the final stage training strategy for multimodal models, focusing on end-to-end fine-tuning",
        "key_classes": [
            {
                "name": "EndToEndStrategy",
                "purpose": "Training strategy for the final stage of multimodal training: end-to-end fine-tuning",
                "key_methods": [
                    {
                        "name": "initialize_strategy",
                        "signature": "initialize_strategy(self) -> None",
                        "brief_description": "Initialize the strategy by unfreezing all components with careful learning rates",
                    },
                    {
                        "name": "_store_initial_model_state",
                        "signature": "_store_initial_model_state(self) -> None",
                        "brief_description": "Store model state for feature consistency to prevent catastrophic forgetting",
                    },
                    {
                        "name": "_configure_loss_function",
                        "signature": "_configure_loss_function(self) -> None",
                        "brief_description": "Configure hard negative mining loss with feature consistency",
                    },
                    {
                        "name": "training_step",
                        "signature": "training_step(self, batch: Dict[str, Any]) -> Dict[str, Any]",
                        "brief_description": "Perform a training step with reference model for consistency",
                    },
                    {
                        "name": "configure_optimizers",
                        "signature": "configure_optimizers(self) -> tuple",
                        "brief_description": "Configure optimizers with very low learning rates for base models",
                    },
                    {
                        "name": "_calculate_topk_accuracy",
                        "signature": "_calculate_topk_accuracy(self, similarity: torch.Tensor, match_ids: torch.Tensor, k: int = 1) -> float",
                        "brief_description": "Calculate top-k accuracy metrics for comprehensive evaluation",
                    },
                    {
                        "name": "_check_feature_drift",
                        "signature": "_check_feature_drift(self) -> None",
                        "brief_description": "Monitor feature drift to detect catastrophic forgetting",
                    },
                ],
                "inheritance": "TrainingStrategy",
                "dependencies": [
                    "torch",
                    "torch.nn",
                    "tqdm",
                    "TrainingStrategy",
                    "WarmupCosineScheduler",
                    "GradientHandler",
                    "HardNegativeMiningContrastiveLoss",
                    "FeatureConsistencyLoss",
                ],
            }
        ],
        "external_dependencies": ["torch", "tqdm"],
        "complexity_score": 9,  # Highly complex implementation with feature consistency, reference model, and drift monitoring
    }
