"""Supervised contrastive loss using labels.

Refactored from supervised_contrastive_loss.py (434 lines) to eliminate
code duplication. Reduces to ~220 lines by leveraging BaseSupervisedLoss.

Uses explicit class labels to form positive pairs, creating tighter semantic
clusters compared to unsupervised contrastive learning.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Any
import logging

from ..base import BaseSupervisedLoss

logger = logging.getLogger(__name__)


class SupervisedContrastiveLoss(BaseSupervisedLoss):
    """
    Supervised contrastive learning for multimodal models.

    Unlike unsupervised contrastive learning where only paired examples are
    positives, this leverages explicit labels to form positive pairs from
    samples of the same class, creating tighter semantic clusters.

    Supports:
    - Single-label and multi-label classification
    - Intra-modal (within vision or text)
    - Cross-modal (vision-to-text)
    - Class weighting for imbalanced datasets

    This refactored version uses BaseSupervisedLoss to eliminate duplication.
    """

    def __init__(
        self,
        temperature: float = 0.07,
        contrast_mode: str = "all",  # "all", "cross", "intra"
        base_temperature: float = 0.07,
        similarity_threshold: float = 0.5,
        use_class_weights: bool = False,
        reduction: str = "mean",
        **kwargs
    ):
        """
        Initialize supervised contrastive loss.

        Args:
            temperature: Temperature for similarity scaling
            contrast_mode: "all" (both), "cross" (cross-modal only), "intra" (within-modal only)
            base_temperature: Base temperature for normalization
            similarity_threshold: Threshold for continuous similarity scores
            use_class_weights: Whether to weight by inverse class frequency
            reduction: Loss reduction method
        """
        super().__init__(
            normalize_features=True,
            temperature=temperature,
            reduction=reduction,
            **kwargs
        )

        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature
        self.similarity_threshold = similarity_threshold
        self.use_class_weights = use_class_weights

    def forward(
        self,
        vision_features: torch.Tensor,
        text_features: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        similarity_scores: Optional[torch.Tensor] = None,
        class_weights: Optional[torch.Tensor] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Compute supervised contrastive loss using labels or similarity scores.

        Args:
            vision_features: Vision embeddings [batch_size, dim]
            text_features: Text embeddings [batch_size, dim]
            labels: Class labels [batch_size] or [batch_size, num_classes] for multi-label
            similarity_scores: Optional pairwise similarity matrix
            class_weights: Optional weights per class [num_classes]
            **kwargs: Additional arguments

        Returns:
            Dictionary with loss components and metrics
        """
        device = vision_features.device
        batch_size = vision_features.shape[0]

        # Normalize features (uses base class mixin)
        vision_features = self.normalize(vision_features)
        text_features = self.normalize(text_features)

        # Create positive pairs mask from labels or similarity scores
        mask = self._create_positive_mask(
            labels, similarity_scores, batch_size, device
        )

        # Remove self-contrast (diagonal)
        logits_mask = torch.ones_like(mask) - torch.eye(batch_size, device=device)
        mask = mask * logits_mask

        # Count positives per sample
        pos_per_sample = mask.sum(1)
        valid_samples = pos_per_sample > 0

        if not valid_samples.any():
            logger.warning("No samples with positive pairs, returning zero loss")
            return {
                "loss": torch.tensor(0.0, device=device),
                "intra_vision_loss": torch.tensor(0.0, device=device),
                "intra_text_loss": torch.tensor(0.0, device=device),
                "cross_modal_loss": torch.tensor(0.0, device=device),
            }

        # Compute loss components based on contrast mode
        intra_vision_loss = torch.tensor(0.0, device=device)
        intra_text_loss = torch.tensor(0.0, device=device)
        cross_modal_loss = torch.tensor(0.0, device=device)

        if self.contrast_mode in ["all", "intra"]:
            intra_vision_loss = self._compute_intra_modal_loss(
                vision_features, mask, logits_mask, pos_per_sample,
                valid_samples, labels, class_weights
            )
            intra_text_loss = self._compute_intra_modal_loss(
                text_features, mask, logits_mask, pos_per_sample,
                valid_samples, labels, class_weights
            )

        if self.contrast_mode in ["all", "cross"]:
            cross_modal_loss = self._compute_cross_modal_loss(
                vision_features, text_features, mask, logits_mask,
                pos_per_sample, valid_samples, labels, class_weights
            )

        # Compute total loss
        if self.contrast_mode == "all":
            total_loss = (intra_vision_loss + intra_text_loss + cross_modal_loss) / 3
        elif self.contrast_mode == "intra":
            total_loss = (intra_vision_loss + intra_text_loss) / 2
        else:  # cross
            total_loss = cross_modal_loss

        return {
            "loss": total_loss,
            "intra_vision_loss": intra_vision_loss,
            "intra_text_loss": intra_text_loss,
            "cross_modal_loss": cross_modal_loss,
        }

    def _create_positive_mask(
        self,
        labels: Optional[torch.Tensor],
        similarity_scores: Optional[torch.Tensor],
        batch_size: int,
        device: torch.device
    ) -> torch.Tensor:
        """Create mask indicating which pairs are positives."""
        if labels is not None:
            # Multi-label case
            if len(labels.shape) > 1 and labels.shape[1] > 1:
                # Two samples are similar if they share any label
                similarity = torch.matmul(labels.float(), labels.float().T)
                mask = (similarity > 0).float()
            else:
                # Single-label case
                labels = labels.contiguous().view(-1, 1)
                mask = torch.eq(labels, labels.T).float()

        elif similarity_scores is not None:
            # Use provided similarity scores with threshold
            mask = (similarity_scores >= self.similarity_threshold).float()

        else:
            # Fallback to identity (only self-similarity)
            logger.warning("No labels or similarity scores, using identity mask")
            mask = torch.eye(batch_size, device=device)

        return mask

    def _compute_intra_modal_loss(
        self,
        features: torch.Tensor,
        mask: torch.Tensor,
        logits_mask: torch.Tensor,
        pos_per_sample: torch.Tensor,
        valid_samples: torch.Tensor,
        labels: Optional[torch.Tensor],
        class_weights: Optional[torch.Tensor]
    ) -> torch.Tensor:
        """Compute supervised contrastive loss within a modality."""
        # Compute similarity
        sim = torch.matmul(features, features.T) / self.temperature

        # Numerical stability
        sim_max, _ = torch.max(sim, dim=1, keepdim=True)
        sim = sim - sim_max.detach()

        # Log probabilities
        exp_sim = torch.exp(sim) * logits_mask
        log_prob = sim - torch.log(exp_sim.sum(1, keepdim=True) + 1e-8)

        # Supervised contrastive loss
        mean_log_prob = (mask * log_prob).sum(1) / (pos_per_sample + 1e-8)

        # Apply class weights if requested
        if self.use_class_weights and class_weights is not None and labels is not None:
            sample_weights = self._get_sample_weights(labels, class_weights)
            mean_log_prob = mean_log_prob * sample_weights

        # Scale and aggregate
        loss = -(self.temperature / self.base_temperature) * mean_log_prob

        if self.reduction == "mean":
            return loss[valid_samples].mean()
        elif self.reduction == "sum":
            return loss[valid_samples].sum()
        return loss[valid_samples]

    def _compute_cross_modal_loss(
        self,
        vision_features: torch.Tensor,
        text_features: torch.Tensor,
        mask: torch.Tensor,
        logits_mask: torch.Tensor,
        pos_per_sample: torch.Tensor,
        valid_samples: torch.Tensor,
        labels: Optional[torch.Tensor],
        class_weights: Optional[torch.Tensor]
    ) -> torch.Tensor:
        """Compute supervised contrastive loss across modalities."""
        # Compute cross-modal similarity
        sim = torch.matmul(vision_features, text_features.T) / self.temperature

        # Numerical stability
        sim_max, _ = torch.max(sim, dim=1, keepdim=True)
        sim = sim - sim_max.detach()

        # Log probabilities
        exp_sim = torch.exp(sim) * logits_mask
        log_prob = sim - torch.log(exp_sim.sum(1, keepdim=True) + 1e-8)

        # Supervised contrastive loss
        mean_log_prob = (mask * log_prob).sum(1) / (pos_per_sample + 1e-8)

        # Apply class weights if requested
        if self.use_class_weights and class_weights is not None and labels is not None:
            sample_weights = self._get_sample_weights(labels, class_weights)
            mean_log_prob = mean_log_prob * sample_weights

        # Scale and aggregate
        loss = -(self.temperature / self.base_temperature) * mean_log_prob

        if self.reduction == "mean":
            return loss[valid_samples].mean()
        elif self.reduction == "sum":
            return loss[valid_samples].sum()
        return loss[valid_samples]

    def _get_sample_weights(
        self,
        labels: torch.Tensor,
        class_weights: torch.Tensor
    ) -> torch.Tensor:
        """Get per-sample weights from class weights."""
        if len(labels.shape) > 1 and labels.shape[1] > 1:
            # Multi-label: average weights across labels
            sample_weights = (labels.float() * class_weights.unsqueeze(0)).sum(1)
            sample_weights = sample_weights / (labels.float().sum(1) + 1e-8)
        else:
            # Single-label: use class weight directly
            sample_weights = class_weights[labels.view(-1)]
        return sample_weights
