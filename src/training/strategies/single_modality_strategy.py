# src/training/strategies/single_modality_strategy.py

import torch
import torch.nn as nn
from torch.optim.adamw import AdamW
from typing import Dict, List, Optional, Any, Callable, Union
import logging
from tqdm import tqdm

from src.training.strategies.training_strategy import TrainingStrategy
from src.utils.learningrate_scheduler import WarmupCosineScheduler
from src.utils.gradient_handler import GradientHandler
from src.utils.metrics_tracker import MetricsTracker
from src.training.losses.contrastive import SimCLRLoss as ContrastiveLoss  # Use new implementation
from src.training.losses.self_supervised import VICRegLoss

logger = logging.getLogger(__name__)

"""
MODULE: single_modality_strategy.py
PURPOSE: Implements the first stage training strategy for multimodal models, focusing on modality-specific learning
KEY COMPONENTS:
- SingleModalityStrategy: Strategy for training modality-specific components while freezing cross-modal integration
DEPENDENCIES: torch, torch.nn, typing, logging, tqdm
SPECIAL NOTES: Freezes cross-modal components and focuses on unimodal representation learning
"""


class SingleModalityStrategy(TrainingStrategy):
    """
    Training strategy for the first stage of multimodal training: modality-specific learning.

    This strategy:
    1. Freezes all cross-modal components (cross-attention, fusion layers)
    2. Focuses on training each modality-specific component separately
    3. Uses separate losses for vision and text without cross-modal objectives
    4. Keeps base pretrained models mostly frozen to preserve pretrained knowledge
    5. Focuses training on projection layers to adapt pretrained models
    """

    def initialize_strategy(self) -> None:
        """
        Initialize the single modality training strategy.

        Main tasks:
        1. Freeze cross-modal components
        2. Set up distinct projection layers for each modality
        3. Configure optimizers with layer-wise learning rates
        4. Set up separate losses for each modality
        """
        logger.info("Initializing SingleModalityStrategy")

        # 1. Freeze cross-modal components
        components_to_freeze = [
            "cross_attention",
            "cross_modal",
            "fusion",
            "interaction",
        ]
        self.freeze_parameters(components_to_freeze)
        logger.info(f"Froze cross-modal components: {', '.join(components_to_freeze)}")

        # 2. Selectively freeze base model components
        # Initially freeze all base models entirely
        base_components_to_freeze = ["vision_model", "text_model"]
        self.freeze_parameters(base_components_to_freeze)

        # If specified in config, selectively unfreeze only top layers
        if self.config.get("unfreeze_top_layers", False):
            unfreeze_layer_count = self.config.get("unfreeze_layer_count", 2)

            # For vision model, unfreeze top layers and classifier
            vision_patterns = [
                f"vision_model.layer{i}" for i in range(12 - unfreeze_layer_count, 12)
            ] + ["vision_model.classifier", "vision_model.head"]

            # For text model, unfreeze top layers and pooler
            text_patterns = [
                f"text_model.encoder.layer.{i}"
                for i in range(12 - unfreeze_layer_count, 12)
            ] + ["text_model.pooler"]

            self.unfreeze_parameters(vision_patterns + text_patterns)
            logger.info(
                f"Selectively unfroze top {unfreeze_layer_count} layers of base models"
            )

        # Always unfreeze projection layers - these are critical for adaptation
        projection_patterns = ["projection", "projector", "adapter"]
        self.unfreeze_parameters(projection_patterns)
        logger.info("Unfroze projection/adapter layers for training")

        # Log parameter status
        self.log_parameter_status()

        # 3. Configure optimizer if not provided
        if self.optimizer is None:
            self.optimizer = self.configure_optimizers()[0]

        # 4. Configure scheduler if not provided
        if self.scheduler is None and hasattr(self, "optimizer"):
            self.scheduler = self.configure_optimizers()[1]

        # 5. Configure loss function if not provided
        if self.loss_fn is None:
            self._configure_loss_function()

        # 6. Initialize gradient handler if enabled
        if self.config.get("use_gradient_handling", True):
            vis_dir = self.config.get("gradient_visualization_dir", None)
            self.gradient_handler = GradientHandler(
                model=self.model,
                clip_value=self.config.get("clip_grad_norm", 1.0),
                component_ratios={"vision": 1.0, "text": 1.0},
                balance_modalities=self.config.get("balance_modalities", True),
                log_frequency=self.config.get("gradient_log_frequency", 100),
                visualization_dir=vis_dir,
            )
        else:
            self.gradient_handler = None

        # Log strategy initialization completed
        logger.info("SingleModalityStrategy initialization completed")

    def _configure_loss_function(self) -> None:
        """
        Configure appropriate loss function for single modality training.

        Options:
        1. VICRegLoss for better feature space learning
        2. ContrastiveLoss for standard contrastive learning
        """
        loss_type = self.config.get("loss_type", "contrastive").lower()

        if loss_type == "vicreg":
            # VICRegLoss is effective for preventing feature collapse and creating better representations
            self.loss_fn = VICRegLoss(
                sim_coeff=self.config.get("sim_coefficient", 25.0),
                var_coeff=self.config.get("var_coefficient", 25.0),
                cov_coeff=self.config.get("cov_coefficient", 1.0),
            )
            logger.info("Using VICRegLoss for single modality training")
        else:
            # Default to standard contrastive loss
            model_dim = self._get_model_dimension()
            temperature = self.config.get("temperature", 0.07)

            self.loss_fn = ContrastiveLoss(
                temperature=temperature,
                # Add projection if requested (default: False since we already have projection layers)
                add_projection=self.config.get("add_loss_projection", False),
                input_dim=model_dim,
            )
            logger.info(
                f"Using ContrastiveLoss with temperature={temperature} for single modality training"
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

            # Text projection dimension as fallback
            if hasattr(self.model, "text_projection"):
                text_proj = self.model.text_projection
                if isinstance(text_proj, nn.Linear):
                    return text_proj.out_features
                elif hasattr(text_proj, "out_features"):
                    return text_proj.out_features
                elif isinstance(text_proj, nn.Sequential) and hasattr(
                    text_proj[-1], "out_features"
                ):
                    return text_proj[-1].out_features
        except:
            pass  # Silently fail and use default

        # Default to 512 as a common projection dimension
        logger.warning("Could not determine model dimension, using default of 512")
        return 512

    def prepare_batch(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        """
        Prepare a batch of data for the model.

        In SingleModalityStrategy, we focus on separate processing
        of modalities (not cross-modal processing).

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

        # Standardize the model interface for this strategy
        model_inputs["return_all_features"] = (
            True  # Ensure we get back features for loss calculation
        )

        return model_inputs

    def training_step(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        """
        Perform a single training step.

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
        Perform a single validation step.

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

        return metrics

    def _prepare_loss_inputs(
        self, batch: Dict[str, Any], outputs: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Prepare inputs for the loss function from model outputs.

        Args:
            batch: Original batch data
            outputs: Outputs from model forward pass

        Returns:
            Dictionary with inputs for loss function
        """
        loss_inputs = {}

        # For VICRegLoss
        if isinstance(self.loss_fn, VICRegLoss):
            # VICRegLoss expects z_a and z_b (two views of the same data)
            if "vision_features" in outputs:
                loss_inputs["z_a"] = outputs["vision_features"]

            if "text_features" in outputs:
                loss_inputs["z_b"] = outputs["text_features"]

            # If we're missing either representation, try to extract from other outputs
            if "z_a" not in loss_inputs and "image_features" in outputs:
                loss_inputs["z_a"] = outputs["image_features"]

            if "z_b" not in loss_inputs and "text_embeddings" in outputs:
                loss_inputs["z_b"] = outputs["text_embeddings"]

        # For ContrastiveLoss and other similar losses
        else:
            # Copy features from outputs
            if "vision_features" in outputs:
                loss_inputs["vision_features"] = outputs["vision_features"]
            elif "image_features" in outputs:
                loss_inputs["vision_features"] = outputs["image_features"]

            if "text_features" in outputs:
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

    def configure_optimizers(self) -> tuple:
        """
        Configure optimizers and schedulers with layer-wise learning rates.

        Returns:
            Tuple of (optimizer, scheduler)
        """
        # Extract configuration
        base_lr = self.config.get("learning_rate", 1e-4)
        weight_decay = self.config.get("weight_decay", 0.01)
        warmup_steps = self.config.get("warmup_steps", 500)
        total_steps = self.config.get("total_steps", 10000)

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

        # Vision base model: very low learning rate to preserve pretrained knowledge
        vision_params = get_unassigned_params(lambda n: "vision_model" in n)
        if vision_params:
            param_groups.append(
                {
                    "params": vision_params,
                    "lr": base_lr * 0.01,  # 1% of base learning rate
                    "name": "vision_model",
                }
            )

        # Text base model: very low learning rate to preserve pretrained knowledge
        text_params = get_unassigned_params(lambda n: "text_model" in n)
        if text_params:
            param_groups.append(
                {
                    "params": text_params,
                    "lr": base_lr * 0.01,  # 1% of base learning rate
                    "name": "text_model",
                }
            )

        # Projection layers: full learning rate for adaptation
        projection_params = get_unassigned_params(
            lambda n: any(x in n for x in ["projection", "adapter"])
        )
        if projection_params:
            param_groups.append(
                {
                    "params": projection_params,
                    "lr": base_lr,  # Full learning rate
                    "name": "projection_layers",
                }
            )

        # Other parameters: default learning rate
        other_params = get_unassigned_params(
            lambda n: True  # Get all remaining parameters
        )
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
        logger.info(f"Starting epoch {epoch} with SingleModalityStrategy")

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
        logger.info(f"Completed epoch {epoch} with SingleModalityStrategy")

        # Implement any strategy-specific logic for epoch end
        # For example, gradually unfreeze more layers of base models as training progresses
        if self.config.get("progressive_unfreezing", False) and epoch > 0:
            if epoch % self.config.get("unfreezing_interval", 5) == 0:
                # Determine layers to unfreeze based on current epoch
                layers_to_unfreeze = min(
                    3, (epoch // self.config.get("unfreezing_interval", 5))
                )

                # Unfreeze additional layers
                vision_patterns = [
                    f"vision_model.layer{i}" for i in range(12 - layers_to_unfreeze, 12)
                ]
                text_patterns = [
                    f"text_model.encoder.layer.{i}"
                    for i in range(12 - layers_to_unfreeze, 12)
                ]

                self.unfreeze_parameters(vision_patterns + text_patterns)
                logger.info(
                    f"Progressive unfreezing: unfroze additional layers ({12 - layers_to_unfreeze}-12) of base models"
                )

                # Log updated parameter status
                self.log_parameter_status()


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
        "module_purpose": "Implements the first stage training strategy for multimodal models, focusing on modality-specific learning",
        "key_classes": [
            {
                "name": "SingleModalityStrategy",
                "purpose": "Training strategy for the first stage of multimodal training: modality-specific learning",
                "key_methods": [
                    {
                        "name": "initialize_strategy",
                        "signature": "initialize_strategy(self) -> None",
                        "brief_description": "Initialize the strategy by freezing cross-modal components and configuring modality-specific training",
                    },
                    {
                        "name": "_configure_loss_function",
                        "signature": "_configure_loss_function(self) -> None",
                        "brief_description": "Configure appropriate loss function for single modality training",
                    },
                    {
                        "name": "prepare_batch",
                        "signature": "prepare_batch(self, batch: Dict[str, Any]) -> Dict[str, Any]",
                        "brief_description": "Prepare a batch with focus on separate modality processing",
                    },
                    {
                        "name": "training_step",
                        "signature": "training_step(self, batch: Dict[str, Any]) -> Dict[str, Any]",
                        "brief_description": "Perform a training step with modality-specific focus",
                    },
                    {
                        "name": "configure_optimizers",
                        "signature": "configure_optimizers(self) -> tuple",
                        "brief_description": "Configure optimizers with layer-wise learning rates for modality components",
                    },
                    {
                        "name": "on_epoch_end",
                        "signature": "on_epoch_end(self, epoch: int) -> None",
                        "brief_description": "Perform end-of-epoch actions including progressive unfreezing",
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
                    "ContrastiveLoss",
                    "VICRegLoss",
                ],
            }
        ],
        "external_dependencies": ["torch", "tqdm"],
        "complexity_score": 8,  # Complex implementation with parameter freezing, gradient handling, and progressive unfreezing
    }
