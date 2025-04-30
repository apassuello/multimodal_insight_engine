# src/training/strategies/end_to_end_strategy.py

import torch
import torch.nn as nn
from torch.optim.adamw import AdamW
from typing import Dict, List, Optional, Any, Callable, Union
import logging
import os
from tqdm import tqdm

from src.training.strategies.training_strategy import TrainingStrategy
from src.utils.learningrate_scheduler import WarmupCosineScheduler
from src.utils.gradient_handler import GradientHandler
from src.utils.metrics_tracker import MetricsTracker
from src.training.losses.hard_negative_mining_contrastive_loss import (
    HardNegativeMiningContrastiveLoss,
)
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
        Store initial model state for feature consistency loss.

        This helps prevent catastrophic forgetting by maintaining
        consistency with the model's behavior at the start of this stage.
        """
        # Create a copy of the model's state dict
        self.initial_state_dict = {
            k: v.clone().detach() for k, v in self.model.state_dict().items()
        }

        # Create a reference model if needed for feature consistency
        if self.config.get("reference_model_for_consistency", True):
            logger.info("Creating reference model for feature consistency")

            # Import the same model class
            model_class = self.model.__class__

            # Create a new instance with the same configuration
            self.reference_model = model_class(**self.config.get("model_config", {}))

            # Load the initial state dict
            self.reference_model.load_state_dict(self.initial_state_dict)

            # Move to the same device
            self.reference_model = self.reference_model.to(self.device)

            # Set to eval mode
            self.reference_model.eval()

            # Freeze all parameters
            for param in self.reference_model.parameters():
                param.requires_grad = False

        logger.info("Initial model state stored for feature consistency")

    def _configure_loss_function(self) -> None:
        """
        Configure appropriate loss function for end-to-end fine-tuning.

        Uses hard negative mining contrastive loss with feature consistency.
        """
        model_dim = self._get_model_dimension()
        temperature = self.config.get("temperature", 0.05)  # Slightly lower temperature

        # Create primary loss (hard negative mining contrastive loss)
        primary_loss = HardNegativeMiningContrastiveLoss(
            temperature=temperature,
            hard_negative_factor=self.config.get("hard_negative_factor", 2.0),
            mining_strategy=self.config.get("mining_strategy", "semi-hard"),
            dim=model_dim,
        )

        # If feature consistency is enabled, create a combined loss
        if self.config.get("use_feature_consistency", True) and hasattr(
            self, "reference_model"
        ):
            consistency_weight = self.config.get("consistency_weight", 0.5)

            # Create feature consistency loss
            consistency_loss = FeatureConsistencyLoss(
                reference_model=self.reference_model,
                weight=consistency_weight,
                consistency_layers=self.config.get(
                    "consistency_layers", ["vision_features", "text_features"]
                ),
            )

            # Combine losses
            logger.info(
                f"Using combined HardNegativeMiningContrastiveLoss with FeatureConsistencyLoss "
                f"(consistency_weight={consistency_weight})"
            )

            # The feature consistency loss will handle calling the primary loss
            self.loss_fn = consistency_loss
            self.loss_fn.set_primary_loss(primary_loss)
        else:
            # Just use the hard negative mining loss
            logger.info(
                f"Using HardNegativeMiningContrastiveLoss with temperature={temperature}"
            )
            self.loss_fn = primary_loss

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

        except:
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

    def training_step(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        """
        Perform a single training step for end-to-end fine-tuning.

        Args:
            batch: The batch of data

        Returns:
            Dictionary containing the loss and other metrics
        """
        # Prepare batch for model
        model_inputs = self.prepare_batch(batch)

        # Forward pass
        outputs = self.model(**model_inputs)

        # Prepare loss inputs - special handling for feature consistency
        if self.config.get("use_feature_consistency", True) and hasattr(
            self, "reference_model"
        ):
            # First get reference model outputs
            with torch.no_grad():
                reference_outputs = self.reference_model(**model_inputs)

            # Combine outputs with reference outputs
            loss_inputs = self._prepare_loss_inputs(batch, outputs, reference_outputs)
        else:
            loss_inputs = self._prepare_loss_inputs(batch, outputs)

        # Compute loss
        loss_outputs = self.loss_fn(**loss_inputs)

        # Extract loss and metrics
        metrics = {}
        if isinstance(loss_outputs, dict):
            loss = loss_outputs.get("loss", 0.0)
            # Copy all other metrics
            for k, v in loss_outputs.items():
                if k != "loss" and not isinstance(v, dict):
                    metrics[k] = v
        else:
            loss = loss_outputs

        metrics["loss"] = loss.item()

        # Backward pass
        loss.backward()

        # Process gradients if handler is available
        if self.gradient_handler is not None:
            self.gradient_handler.analyze_gradients()
            self.gradient_handler.clip_gradients()

            # Balance gradients if requested
            if self.config.get("balance_modalities", True):
                self.gradient_handler.balance_component_gradients(self.optimizer)

        return metrics

    def validation_step(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        """
        Perform a single validation step for end-to-end evaluation.

        Args:
            batch: The batch of data

        Returns:
            Dictionary containing validation metrics
        """
        # Similar to CrossModalStrategy validation, but with more comprehensive metrics
        # and handling of feature consistency if enabled

        # Prepare batch for model
        model_inputs = self.prepare_batch(batch)

        # Forward pass in no_grad context
        with torch.no_grad():
            outputs = self.model(**model_inputs)

            # Prepare loss inputs
            if self.config.get("use_feature_consistency", True) and hasattr(
                self, "reference_model"
            ):
                # Get reference model outputs
                reference_outputs = self.reference_model(**model_inputs)
                loss_inputs = self._prepare_loss_inputs(
                    batch, outputs, reference_outputs
                )
            else:
                loss_inputs = self._prepare_loss_inputs(batch, outputs)

            # Compute loss and metrics without gradients
            loss_outputs = self.loss_fn(**loss_inputs)

            # Extract loss and metrics
            metrics = {}
            if isinstance(loss_outputs, dict):
                loss = loss_outputs.get("loss", 0.0)
                # Copy all other metrics
                for k, v in loss_outputs.items():
                    if k != "loss" and not isinstance(v, dict):
                        metrics[k] = v
            else:
                loss = loss_outputs

            metrics["loss"] = loss.item()

            # Calculate additional similarity metrics for multimodal alignment
            if "vision_features" in loss_inputs and "text_features" in loss_inputs:
                vision_features = loss_inputs["vision_features"]
                text_features = loss_inputs["text_features"]

                # Normalize features for cosine similarity
                if vision_features.dim() > 1 and text_features.dim() > 1:
                    vision_features = torch.nn.functional.normalize(
                        vision_features, p=2, dim=1
                    )
                    text_features = torch.nn.functional.normalize(
                        text_features, p=2, dim=1
                    )

                    # Compute similarity matrix
                    similarity = torch.matmul(vision_features, text_features.T)
                    diag_sim = torch.diagonal(similarity).mean().item()
                    mean_sim = similarity.mean().item()
                    std_sim = similarity.std().item()

                    # Add to metrics
                    metrics["diag_similarity"] = diag_sim
                    metrics["mean_similarity"] = mean_sim
                    metrics["alignment_gap"] = diag_sim - mean_sim

                    # Add additional retrieval metrics if match_ids available
                    if "match_ids" in loss_inputs:
                        # Calculate top-k accuracy for k=1,3,5
                        for k in [1, 3, 5]:
                            metrics[f"top{k}_accuracy"] = self._calculate_topk_accuracy(
                                similarity, loss_inputs["match_ids"], k=k
                            )

                        # Calculate recall@K for K=1,5,10
                        for k in [1, 5, 10]:
                            metrics[f"recall@{k}"] = self._calculate_recall_at_k(
                                similarity, loss_inputs["match_ids"], k=k
                            )

        return metrics

    def _prepare_loss_inputs(
        self,
        batch: Dict[str, Any],
        outputs: Dict[str, Any],
        reference_outputs: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Prepare inputs for the loss function from model outputs.

        Args:
            batch: Original batch data
            outputs: Outputs from model forward pass
            reference_outputs: Optional outputs from reference model

        Returns:
            Dictionary with inputs for loss function
        """
        loss_inputs = {}

        # Extract features from outputs with multiple fallback options
        # For vision features
        if "vision_features_enhanced" in outputs:
            loss_inputs["vision_features"] = outputs["vision_features_enhanced"]
        elif "vision_features" in outputs:
            loss_inputs["vision_features"] = outputs["vision_features"]
        elif "image_features" in outputs:
            loss_inputs["vision_features"] = outputs["image_features"]

        # For text features
        if "text_features_enhanced" in outputs:
            loss_inputs["text_features"] = outputs["text_features_enhanced"]
        elif "text_features" in outputs:
            loss_inputs["text_features"] = outputs["text_features"]
        elif "text_embeddings" in outputs:
            loss_inputs["text_features"] = outputs["text_embeddings"]

        # Include match IDs for contrastive learning
        if "match_ids" in batch:
            loss_inputs["match_ids"] = batch["match_ids"]
        elif "match_id" in batch:
            loss_inputs["match_ids"] = batch["match_id"]

        # Add reference features for feature consistency loss if available
        if reference_outputs is not None:
            loss_inputs["reference_outputs"] = reference_outputs

        # Add any intermediate features for feature consistency
        if "intermediate_features" in outputs:
            loss_inputs["intermediate_features"] = outputs["intermediate_features"]

        return loss_inputs

    def _calculate_topk_accuracy(
        self, similarity: torch.Tensor, match_ids: torch.Tensor, k: int = 1
    ) -> float:
        """
        Calculate top-k retrieval accuracy using match IDs.

        Args:
            similarity: Similarity matrix between vision and text features
            match_ids: Tensor of match IDs for the batch
            k: Number of top results to consider

        Returns:
            Top-k retrieval accuracy
        """
        batch_size = similarity.size(0)

        # Create matching matrix based on match IDs
        matching_matrix = torch.zeros_like(similarity, dtype=torch.bool)
        for i in range(batch_size):
            for j in range(batch_size):
                matching_matrix[i, j] = match_ids[i] == match_ids[j]

        # Get top-k indices for each query
        k_adjusted = min(k, batch_size)
        _, indices = similarity.topk(k_adjusted, dim=1)

        # Check if any of the top-k retrievals match
        accuracy_flags = torch.zeros(batch_size, dtype=torch.bool, device=self.device)

        for i in range(batch_size):
            for rank in range(k_adjusted):
                j = indices[i, rank]
                if matching_matrix[i, j]:
                    accuracy_flags[i] = True
                    break

        # Calculate top-k accuracy
        accuracy = accuracy_flags.float().mean().item()

        return accuracy

    def _calculate_recall_at_k(
        self, similarity: torch.Tensor, match_ids: torch.Tensor, k: int = 5
    ) -> float:
        """
        Calculate recall@K for image-to-text retrieval.

        Args:
            similarity: Similarity matrix between vision and text features
            match_ids: Tensor of match IDs for the batch
            k: Number of top results to consider

        Returns:
            Recall@K value
        """
        batch_size = similarity.size(0)

        # Create matching matrix based on match IDs
        matching_matrix = torch.zeros_like(similarity, dtype=torch.bool)
        for i in range(batch_size):
            for j in range(batch_size):
                matching_matrix[i, j] = match_ids[i] == match_ids[j]

        # Get top-k indices for each query
        k_adjusted = min(k, batch_size)
        _, indices = similarity.topk(k_adjusted, dim=1)

        # Check if any of the top-k retrievals match
        recall_flags = torch.zeros(batch_size, dtype=torch.bool, device=self.device)

        for i in range(batch_size):
            for rank in range(k_adjusted):
                j = indices[i, rank]
                if matching_matrix[i, j]:
                    recall_flags[i] = True
                    break

        # Calculate recall@k
        recall = recall_flags.float().mean().item()

        return recall

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

        # Set up parameter groups with different learning rates
        param_groups = [
            # Vision base model: extremely low learning rate
            {
                "params": [
                    p
                    for n, p in self.model.named_parameters()
                    if "vision_model" in n and p.requires_grad
                ],
                "lr": base_lr * 0.005,  # 0.5% of base learning rate
                "name": "vision_model",
            },
            # Text base model: extremely low learning rate
            {
                "params": [
                    p
                    for n, p in self.model.named_parameters()
                    if "text_model" in n and p.requires_grad
                ],
                "lr": base_lr * 0.005,  # 0.5% of base learning rate
                "name": "text_model",
            },
            # Cross-modal components: low learning rate
            {
                "params": [
                    p
                    for n, p in self.model.named_parameters()
                    if any(
                        x in n
                        for x in [
                            "cross_attention",
                            "cross_modal",
                            "fusion",
                            "interaction",
                        ]
                    )
                    and p.requires_grad
                ],
                "lr": base_lr * 0.1,  # 10% of base learning rate
                "name": "cross_modal_components",
            },
            # Projection layers: medium learning rate
            {
                "params": [
                    p
                    for n, p in self.model.named_parameters()
                    if any(x in n for x in ["projection", "adapter"])
                    and p.requires_grad
                ],
                "lr": base_lr * 0.5,  # 50% of base learning rate
                "name": "projection_layers",
            },
            # Other parameters: base learning rate
            {
                "params": [
                    p
                    for n, p in self.model.named_parameters()
                    if not any(
                        x in n
                        for x in [
                            "vision_model",
                            "text_model",
                            "cross_attention",
                            "cross_modal",
                            "fusion",
                            "interaction",
                            "projection",
                            "adapter",
                        ]
                    )
                    and p.requires_grad
                ],
                "lr": base_lr,  # Full base learning rate
                "name": "other",
            },
        ]

        # Remove empty groups
        param_groups = [g for g in param_groups if len(g["params"]) > 0]

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

        # Check for feature drift if using feature consistency
        if self.config.get("use_feature_consistency", True) and hasattr(
            self, "reference_model"
        ):
            self._check_feature_drift()

    def _check_feature_drift(self) -> None:
        """
        Check how much the model features have drifted from the reference model.

        This helps monitor catastrophic forgetting during fine-tuning.
        """
        if not hasattr(self.loss_fn, "get_consistency_metrics"):
            return

        # Get consistency metrics from the loss function
        consistency_metrics = self.loss_fn.get_consistency_metrics()

        if consistency_metrics:
            metrics_str = ", ".join(
                [f"{k}: {v:.4f}" for k, v in consistency_metrics.items()]
            )
            logger.info(f"Feature consistency metrics: {metrics_str}")

            # Check for significant drift
            if any(v > 0.5 for k, v in consistency_metrics.items() if "dist" in k):
                logger.warning(
                    "Significant feature drift detected - possible catastrophic forgetting"
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
