"""Contrastive loss with hard negative mining.

Refactored from hard_negative_mining_contrastive_loss.py (237 lines) to
eliminate code duplication. Reduces to ~100 lines by leveraging
BaseContrastiveLoss and HardNegativeMiningMixin.

Supports:
- Hard mining: Select top-k hardest negatives
- Semi-hard mining: Select negatives in specific similarity range
- Configurable weighting for mined negatives
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Any
import logging

from ..base import BaseContrastiveLoss

logger = logging.getLogger(__name__)


class HardNegativeLoss(BaseContrastiveLoss):
    """
    Contrastive loss with hard negative mining for more effective training.

    Mines hard negatives online during training and weights them more heavily
    in the loss computation. This refactored version uses BaseContrastiveLoss
    and HardNegativeMiningMixin to eliminate code duplication.
    """

    def __init__(
        self,
        temperature: float = 0.07,
        hard_negative_factor: float = 2.0,
        mining_strategy: str = "semi-hard",
        hard_negative_percentile: float = 90.0,
        **kwargs
    ):
        """
        Initialize hard negative mining loss.

        Args:
            temperature: Temperature for similarity scaling
            hard_negative_factor: Weight multiplier for hard negatives
            mining_strategy: "hard" (top-k) or "semi-hard" (similarity range)
            hard_negative_percentile: Percentile threshold for hard negative mining
        """
        super().__init__(
            temperature=temperature,
            normalize_features=True,
            hard_negative_percentile=hard_negative_percentile,
            **kwargs
        )

        self.hard_negative_factor = hard_negative_factor
        self.mining_strategy = mining_strategy

    def forward(
        self,
        vision_features: torch.Tensor,
        text_features: torch.Tensor,
        match_ids: Optional[List[str]] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Compute contrastive loss with hard negative mining.

        Args:
            vision_features: Vision embeddings [batch_size, dim]
            text_features: Text embeddings [batch_size, dim]
            match_ids: IDs for semantic matching
            **kwargs: Additional arguments

        Returns:
            Dictionary with loss and metrics
        """
        device = vision_features.device
        batch_size = vision_features.shape[0]

        # Normalize features (uses base class mixin)
        vision_features = self.normalize(vision_features)
        text_features = self.normalize(text_features)

        # Compute similarity matrix
        sim_matrix = torch.matmul(vision_features, text_features.T)

        # Create match matrix
        match_matrix = self._create_match_matrix(batch_size, match_ids, device)

        # Compute loss in both directions with hard negative mining
        v2t_loss = self._compute_direction_loss_with_mining(
            sim_matrix, match_matrix, direction="v2t"
        )
        t2v_loss = self._compute_direction_loss_with_mining(
            sim_matrix.T, match_matrix.T, direction="t2v"
        )

        # Normalize and average
        num_pos = match_matrix.sum().item()
        if num_pos > 0:
            v2t_loss = v2t_loss / num_pos
            t2v_loss = t2v_loss / num_pos
        else:
            return {"loss": torch.tensor(0.0, device=device)}

        loss = (v2t_loss + t2v_loss) / 2.0

        return {
            "loss": loss,
            "v2t_loss": v2t_loss.item(),
            "t2v_loss": t2v_loss.item(),
            "hard_negative_factor": self.hard_negative_factor,
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

    def _compute_direction_loss_with_mining(
        self,
        sim_matrix: torch.Tensor,
        match_matrix: torch.Tensor,
        direction: str
    ) -> torch.Tensor:
        """Compute InfoNCE loss with hard negative mining for one direction."""
        batch_size = sim_matrix.shape[0]
        total_loss = 0.0

        for i in range(batch_size):
            # Get positive and negative indices
            pos_indices = torch.where(match_matrix[i])[0]
            neg_indices = torch.where(~match_matrix[i])[0]

            if len(pos_indices) == 0 or len(neg_indices) == 0:
                continue

            # Positive and negative similarities
            pos_sims = sim_matrix[i, pos_indices]
            neg_sims = sim_matrix[i, neg_indices]

            # Mine hard negatives and get weights
            hard_weights = self._mine_and_weight_negatives(
                pos_sims, neg_sims
            )

            # Compute weighted InfoNCE loss
            pos_exp = torch.exp(pos_sims / self.temp)
            weighted_neg_exp = torch.exp(neg_sims / self.temp) * hard_weights

            # Loss for each positive
            for pos_idx in range(len(pos_sims)):
                pos_term = pos_exp[pos_idx]
                neg_term = weighted_neg_exp.sum()
                total_loss += -torch.log(pos_term / (pos_term + neg_term))

        return total_loss

    def _mine_and_weight_negatives(
        self,
        pos_sims: torch.Tensor,
        neg_sims: torch.Tensor
    ) -> torch.Tensor:
        """Mine hard negatives and return weights."""
        mean_pos_sim = pos_sims.mean()
        weights = torch.ones_like(neg_sims)

        if self.mining_strategy == "hard":
            # Select top 10% hardest negatives (highest similarity)
            num_hard = max(1, int(len(neg_sims) * 0.1))
            hard_indices = torch.topk(neg_sims, num_hard)[1]
            weights[hard_indices] = self.hard_negative_factor

        elif self.mining_strategy == "semi-hard":
            # Semi-hard: negatives closer than mean positive but within threshold
            semi_hard_mask = (neg_sims < mean_pos_sim) & (neg_sims > mean_pos_sim - 0.2)
            if semi_hard_mask.sum() > 0:
                weights[semi_hard_mask] = self.hard_negative_factor

        return weights
