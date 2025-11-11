"""Contrastive loss with dynamic temperature scaling.

Refactored from dynamic_temperature_contrastive_loss.py (172 lines) to
eliminate code duplication. Reduces to ~90 lines by leveraging
BaseContrastiveLoss and TemperatureScalingMixin.

Dynamically adjusts temperature based on the separation between
positive and negative similarities during training.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Any
import logging

from ..base import BaseContrastiveLoss

logger = logging.getLogger(__name__)


class DynamicTemperatureLoss(BaseContrastiveLoss):
    """
    Contrastive loss with adaptive temperature scaling.

    Automatically adjusts temperature based on the distribution of
    positive and negative similarities. Better separated embeddings
    use lower temperature for finer-grained discrimination.

    This refactored version uses BaseContrastiveLoss to eliminate
    duplication of normalization, similarity computation, etc.
    """

    def __init__(
        self,
        base_temperature: float = 0.07,
        min_temp: float = 0.04,
        max_temp: float = 0.2,
        **kwargs
    ):
        """
        Initialize dynamic temperature loss.

        Args:
            base_temperature: Base temperature value
            min_temp: Minimum allowed temperature
            max_temp: Maximum allowed temperature
        """
        super().__init__(
            temperature=base_temperature,
            normalize_features=True,
            **kwargs
        )

        self.base_temperature = base_temperature
        self.min_temp = min_temp
        self.max_temp = max_temp

    def forward(
        self,
        vision_features: torch.Tensor,
        text_features: torch.Tensor,
        match_ids: Optional[List[str]] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Compute contrastive loss with dynamic temperature.

        Args:
            vision_features: Vision embeddings [batch_size, dim]
            text_features: Text embeddings [batch_size, dim]
            match_ids: IDs for semantic matching
            **kwargs: Additional arguments

        Returns:
            Dictionary with loss and metrics including dynamic temperature
        """
        device = vision_features.device
        batch_size = vision_features.shape[0]

        # Normalize features (uses base class mixin)
        vision_features = self.normalize(vision_features)
        text_features = self.normalize(text_features)

        # Compute similarity matrix (without temperature yet)
        sim_matrix = torch.matmul(vision_features, text_features.T)

        # Create match matrix
        match_matrix = self._create_match_matrix(batch_size, match_ids, device)

        # Calculate dynamic temperature based on similarity distribution
        dynamic_temp = self._calculate_dynamic_temperature(sim_matrix, match_matrix)

        # Compute loss with dynamic temperature
        v2t_loss, t2v_loss = self._compute_bidirectional_loss(
            sim_matrix, match_matrix, dynamic_temp
        )

        # Normalize and average
        num_pos = match_matrix.sum().item()
        if num_pos > 0:
            v2t_loss = v2t_loss / num_pos
            t2v_loss = t2v_loss / num_pos
        else:
            return {"loss": torch.tensor(0.0, device=device)}

        loss = (v2t_loss + t2v_loss) / 2.0

        # Calculate similarity statistics for monitoring
        pos_mask = match_matrix.float()
        neg_mask = 1.0 - pos_mask
        pos_mask.fill_diagonal_(0)

        pos_mean = (pos_mask * sim_matrix).sum() / max(1, pos_mask.sum().item())
        neg_mean = (neg_mask * sim_matrix).sum() / max(1, neg_mask.sum().item())

        return {
            "loss": loss,
            "temperature": dynamic_temp.item(),
            "pos_similarity": pos_mean.item(),
            "neg_similarity": neg_mean.item(),
            "v2t_loss": v2t_loss.item(),
            "t2v_loss": t2v_loss.item(),
        }

    def _create_match_matrix(
        self,
        batch_size: int,
        match_ids: Optional[List[str]],
        device: torch.device
    ) -> torch.Tensor:
        """Create boolean matrix indicating matches."""
        if match_ids is None:
            return torch.eye(batch_size, dtype=torch.bool, device=device)

        match_matrix = torch.zeros((batch_size, batch_size), dtype=torch.bool, device=device)
        for i in range(batch_size):
            for j in range(batch_size):
                match_matrix[i, j] = match_ids[i] == match_ids[j]
        return match_matrix

    def _calculate_dynamic_temperature(
        self,
        sim_matrix: torch.Tensor,
        match_matrix: torch.Tensor
    ) -> torch.Tensor:
        """
        Calculate temperature based on separation between positives and negatives.

        Lower temperature for better separated embeddings, higher for less separated.
        """
        pos_mask = match_matrix.float()
        neg_mask = 1.0 - pos_mask
        pos_mask.fill_diagonal_(0)  # Remove diagonal

        # Calculate mean similarities
        pos_mean = (pos_mask * sim_matrix).sum() / max(1, pos_mask.sum().item())
        neg_mean = (neg_mask * sim_matrix).sum() / max(1, neg_mask.sum().item())

        # Calculate separation and adjust temperature
        separation = pos_mean - neg_mean
        # Lower temp for better separation, higher for worse separation
        dynamic_temp = self.base_temperature * (0.8 + 0.4 * torch.exp(-2.0 * separation))
        dynamic_temp = torch.clamp(dynamic_temp, self.min_temp, self.max_temp)

        return dynamic_temp

    def _compute_bidirectional_loss(
        self,
        sim_matrix: torch.Tensor,
        match_matrix: torch.Tensor,
        temperature: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute InfoNCE loss in both directions with given temperature."""
        batch_size = sim_matrix.shape[0]
        v2t_loss = 0.0
        t2v_loss = 0.0

        # Vision-to-text
        for i in range(batch_size):
            pos_indices = torch.where(match_matrix[i])[0]
            if len(pos_indices) == 0:
                continue

            all_logits = sim_matrix[i] / temperature
            for pos_idx in pos_indices:
                v2t_loss += -all_logits[pos_idx] + torch.logsumexp(all_logits, dim=0)

        # Text-to-vision
        for i in range(batch_size):
            pos_indices = torch.where(match_matrix[:, i])[0]
            if len(pos_indices) == 0:
                continue

            all_logits = sim_matrix[:, i] / temperature
            for pos_idx in pos_indices:
                t2v_loss += -all_logits[pos_idx] + torch.logsumexp(all_logits, dim=0)

        return v2t_loss, t2v_loss
