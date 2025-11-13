# src/training/strategies/cross_modal_strategy.py

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
from src.training.losses import MemoryQueueContrastiveLoss

logger = logging.getLogger(__name__)

"""
MODULE: cross_modal_strategy.py
PURPOSE: Implements the second stage training strategy for multimodal models, focusing on cross-modal integration
KEY COMPONENTS:
- CrossModalStrategy: Strategy for training cross-modal components while freezing base encoder models
DEPENDENCIES: torch, torch.nn, typing, logging, tqdm
SPECIAL NOTES: Uses memory queue contrastive loss for efficient cross-modal alignment
"""


class CrossModalStrategy(TrainingStrategy):
    """
    Training strategy for the second stage of multimodal training: cross-modal integration.

    This strategy:
    1. Freezes base encoder models to preserve their representations
    2. Focuses on training cross-modal components like fusion layers
    3. Uses memory queue contrastive loss for efficient cross-modal alignment
    4. Implements careful learning rate scheduling for stable training
    """

    def initialize_strategy(self) -> None:
        """
        Initialize the cross-modal training strategy.

        Main tasks:
        1. Freeze base encoder models
        2. Unfreeze cross-modal components
        3. Configure optimizer with appropriate learning rates
        4. Set up cross-modal contrastive loss
        """
        logger.info("Initializing CrossModalStrategy")

        # 1. Freeze base encoder models to preserve representations
        encoder_components = ["vision_model", "text_model"]
        self.freeze_parameters(encoder_components)
        logger.info(f"Froze base encoder models: {', '.join(encoder_components)}")

        # 2. Selectively unfreeze top encoder layers if specified
        if self.config.get("unfreeze_top_encoder_layers", False):
            unfreeze_layer_count = self.config.get("unfreeze_layer_count", 2)

            # Unfreeze top layers of vision model
            vision_patterns = [
                f"vision_model.layer{i}" for i in range(12 - unfreeze_layer_count, 12)
            ]

            # Unfreeze top layers of text model
            text_patterns = [
                f"text_model.encoder.layer.{i}"
                for i in range(12 - unfreeze_layer_count, 12)
            ]

            self.unfreeze_parameters(vision_patterns + text_patterns)
            logger.info(
                f"Selectively unfroze top {unfreeze_layer_count} layers of encoder models"
            )

        # 3. Unfreeze cross-modal components for focused training
        cross_modal_components = [
            "cross_attention",
            "cross_modal",
            "fusion",
            "interaction",
        ]
        self.unfreeze_parameters(cross_modal_components)
        logger.info(
            f"Unfroze cross-modal components: {', '.join(cross_modal_components)}"
        )

        # Always unfreeze projection layers for adaptation
        projection_patterns = ["projection", "projector", "adapter"]
        self.unfreeze_parameters(projection_patterns)
        logger.info("Unfroze projection/adapter layers for training")

        # Log parameter status
        self.log_parameter_status()

        # 4. Configure optimizer if not provided
        if self.optimizer is None:
            self.optimizer = self.configure_optimizers()[0]

        # 5. Configure scheduler if not provided
        if self.scheduler is None and hasattr(self, "optimizer"):
            self.scheduler = self.configure_optimizers()[1]

        # 6. Configure loss function if not provided
        if self.loss_fn is None:
            self._configure_loss_function()

        # 7. Initialize gradient handler if enabled
        if self.config.get("use_gradient_handling", True):
            vis_dir = self.config.get("gradient_visualization_dir", None)
            self.gradient_handler = GradientHandler(
                model=self.model,
                clip_value=self.config.get("clip_grad_norm", 1.0),
                # Balanced toward fusion components
                component_ratios={"vision": 0.5, "text": 0.5, "fusion": 1.0},
                balance_modalities=self.config.get("balance_modalities", True),
                log_frequency=self.config.get("gradient_log_frequency", 100),
                visualization_dir=vis_dir,
            )
        else:
            self.gradient_handler = None

        # Log strategy initialization completed
        logger.info("CrossModalStrategy initialization completed")

    def _configure_loss_function(self) -> None:
        """
        Configure appropriate loss function for cross-modal training.

        Uses memory queue contrastive loss for efficient training with
        larger effective batch sizes.
        """
        model_dim = self._get_model_dimension()
        temperature = self.config.get("temperature", 0.07)
        queue_size = self.config.get("queue_size", 8192)

        # Use memory queue contrastive loss for larger effective batch sizes
        self.loss_fn = MemoryQueueContrastiveLoss(
            temperature=temperature,
            queue_size=queue_size,
            dim=model_dim,
        )

        logger.info(
            f"Using MemoryQueueContrastiveLoss with temperature={temperature}, "
            f"queue_size={queue_size} for cross-modal training"
        )

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

        # Extract from model structure by examining the projections
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

        In CrossModalStrategy, we focus on cross-modal interactions.

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

        # Enable cross-attention layers specifically for this strategy
        model_inputs["enable_cross_attention"] = True
        model_inputs["return_all_features"] = True

        return model_inputs

    def training_step(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        """
        Perform a single training step focusing on cross-modal integration.

        Args:
            batch: The batch of data

        Returns:
            Dictionary containing the loss and other metrics
        """
        # Prepare batch for model
        model_inputs = self.prepare_batch(batch)

        # Forward pass
        outputs = self.model(**model_inputs)

        # Prepare loss inputs
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
        Perform a single validation step focused on cross-modal integration.

        Args:
            batch: The batch of data

        Returns:
            Dictionary containing validation metrics
        """
        # Prepare batch for model
        model_inputs = self.prepare_batch(batch)

        # Forward pass in no_grad context
        with torch.no_grad():
            outputs = self.model(**model_inputs)

            # Prepare loss inputs
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
                        top1_accuracy = self._calculate_retrieval_accuracy(
                            similarity, loss_inputs["match_ids"]
                        )
                        metrics["top1_accuracy"] = top1_accuracy

                        # Calculate recall@K for K=1,5,10
                        for k in [1, 5, 10]:
                            metrics[f"recall@{k}"] = self._calculate_recall_at_k(
                                similarity, loss_inputs["match_ids"], k=k
                            )

        return metrics

    def _prepare_loss_inputs(
        self, batch: Dict[str, Any], outputs: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Prepare inputs for the memory queue contrastive loss function from model outputs.

        Args:
            batch: Original batch data
            outputs: Outputs from model forward pass

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

        return loss_inputs

    def _calculate_retrieval_accuracy(
        self, similarity: torch.Tensor, match_ids: torch.Tensor
    ) -> float:
        """
        Calculate top-1 retrieval accuracy using match IDs.

        Args:
            similarity: Similarity matrix between vision and text features
            match_ids: Tensor of match IDs for the batch

        Returns:
            Top-1 retrieval accuracy
        """
        # Convert match IDs to a matching matrix
        batch_size = similarity.size(0)
        matching_matrix = torch.zeros_like(similarity, dtype=torch.bool)

        # Fill matching matrix based on match IDs
        for i in range(batch_size):
            for j in range(batch_size):
                matching_matrix[i, j] = match_ids[i] == match_ids[j]

        # Calculate accuracy
        # For each image, find the index of the top text match
        _, indices = similarity.topk(1, dim=1)
        top1_matches = torch.zeros(batch_size, dtype=torch.bool, device=self.device)

        for i in range(batch_size):
            j = indices[i, 0]
            top1_matches[i] = matching_matrix[i, j]

        return top1_matches.float().mean().item()

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
        Configure optimizers and schedulers with focus on cross-modal components.

        Returns:
            Tuple of (optimizer, scheduler)
        """
        # Extract configuration
        base_lr = self.config.get(
            "learning_rate", 5e-5
        )  # Lower default LR for this stage
        weight_decay = self.config.get("weight_decay", 0.01)
        warmup_steps = self.config.get("warmup_steps", 200)
        total_steps = self.config.get("total_steps", 5000)

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

        # Vision base model: low learning rate
        vision_params = get_unassigned_params(lambda n: "vision_model" in n)
        if vision_params:
            param_groups.append(
                {
                    "params": vision_params,
                    "lr": base_lr * 0.01,  # 1% of base learning rate
                    "name": "vision_model",
                }
            )

        # Text base model: low learning rate
        text_params = get_unassigned_params(lambda n: "text_model" in n)
        if text_params:
            param_groups.append(
                {
                    "params": text_params,
                    "lr": base_lr * 0.01,  # 1% of base learning rate
                    "name": "text_model",
                }
            )

        # Cross-modal components: full learning rate (main focus)
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
                    "lr": base_lr,  # Full learning rate
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

        # Other parameters: medium learning rate
        other_params = get_unassigned_params(
            lambda n: True
        )  # Get all remaining parameters
        if other_params:
            param_groups.append(
                {
                    "params": other_params,
                    "lr": base_lr * 0.1,  # 10% of base learning rate
                    "name": "other",
                }
            )

        # Create optimizer
        optimizer = AdamW(
            param_groups,
            lr=base_lr,
            weight_decay=weight_decay,
            betas=(0.9, 0.98),
            eps=1e-6,
        )

        # Create learning rate scheduler
        scheduler = WarmupCosineScheduler(
            optimizer=optimizer,
            warmup_steps=warmup_steps,
            total_steps=total_steps,
            min_lr=base_lr * 0.01,  # 1% of base learning rate as minimum
        )

        return optimizer, scheduler

    def on_epoch_start(self, epoch: int) -> None:
        """
        Perform actions at the start of each epoch.

        Args:
            epoch: Current epoch number
        """
        logger.info(f"Starting epoch {epoch} with CrossModalStrategy")

        # Log learning rates
        learning_rates = self.get_learning_rates()
        lr_info = ", ".join([f"{k}: {v:.6f}" for k, v in learning_rates.items()])
        logger.info(f"Learning rates: {lr_info}")

    def on_epoch_end(self, epoch: int) -> None:
        """
        Perform actions at the end of each epoch.

        Args:
            epoch: Current epoch number
        """
        logger.info(f"Completed epoch {epoch} with CrossModalStrategy")

        # Log memory queue statistics if available
        if hasattr(self.loss_fn, "get_queue_stats"):
            queue_stats = self.loss_fn.get_queue_stats()
            if queue_stats:
                stats_info = ", ".join(
                    [f"{k}: {v:.4f}" for k, v in queue_stats.items()]
                )
                logger.info(f"Memory queue stats: {stats_info}")


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
        "module_purpose": "Implements the second stage training strategy for multimodal models, focusing on cross-modal integration",
        "key_classes": [
            {
                "name": "CrossModalStrategy",
                "purpose": "Training strategy for the second stage of multimodal training: cross-modal integration",
                "key_methods": [
                    {
                        "name": "initialize_strategy",
                        "signature": "initialize_strategy(self) -> None",
                        "brief_description": "Initialize the strategy by freezing base encoders and configuring cross-modal training",
                    },
                    {
                        "name": "_configure_loss_function",
                        "signature": "_configure_loss_function(self) -> None",
                        "brief_description": "Configure memory queue contrastive loss for cross-modal training",
                    },
                    {
                        "name": "prepare_batch",
                        "signature": "prepare_batch(self, batch: Dict[str, Any]) -> Dict[str, Any]",
                        "brief_description": "Prepare a batch with focus on cross-modal interactions",
                    },
                    {
                        "name": "training_step",
                        "signature": "training_step(self, batch: Dict[str, Any]) -> Dict[str, Any]",
                        "brief_description": "Perform a training step focusing on cross-modal integration",
                    },
                    {
                        "name": "configure_optimizers",
                        "signature": "configure_optimizers(self) -> tuple",
                        "brief_description": "Configure optimizers with focus on cross-modal components",
                    },
                    {
                        "name": "_calculate_recall_at_k",
                        "signature": "_calculate_recall_at_k(self, similarity: torch.Tensor, match_ids: torch.Tensor, k: int = 5) -> float",
                        "brief_description": "Calculate recall@K metrics for cross-modal retrieval evaluation",
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
                    "MemoryQueueContrastiveLoss",
                ],
            }
        ],
        "external_dependencies": ["torch", "tqdm"],
        "complexity_score": 7,  # Complex implementation with memory queue, gradient handling, and cross-modal focus
    }
