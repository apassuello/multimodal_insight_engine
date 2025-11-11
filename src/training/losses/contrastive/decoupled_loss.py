"""Decoupled contrastive loss with instance discrimination.

Refactored from decoupled_contrastive_loss.py (359 lines) to eliminate
code duplication. Reduces to ~180 lines by leveraging BaseContrastiveLoss.

Separates vision-to-text and text-to-vision learning objectives, and adds
instance discrimination within each modality for better representation learning.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Any
import logging

from ..base import BaseContrastiveLoss

logger = logging.getLogger(__name__)


class DecoupledLoss(BaseContrastiveLoss):
    """
    Decoupled contrastive learning with instance discrimination.

    Separates the vision-to-text and text-to-vision learning objectives,
    and adds instance discrimination within each modality. This allows for:
    1. More flexible training with separate weighting per modality
    2. Better intra-modality representations via instance discrimination
    3. Fine-grained control over cross-modal vs intra-modal learning

    This refactored version uses BaseContrastiveLoss to eliminate duplication.
    """

    def __init__(
        self,
        temperature: float = 0.07,
        lambda_v: float = 0.5,  # Weight for vision instance discrimination
        lambda_t: float = 0.5,  # Weight for text instance discrimination
        reduction: str = "mean",
        **kwargs
    ):
        """
        Initialize decoupled contrastive loss.

        Args:
            temperature: Temperature for similarity scaling
            lambda_v: Weight for vision instance discrimination loss
            lambda_t: Weight for text instance discrimination loss
            reduction: Loss reduction method
        """
        super().__init__(
            temperature=temperature,
            normalize_features=True,
            reduction=reduction,
            **kwargs
        )

        self.lambda_v = lambda_v
        self.lambda_t = lambda_t

    def forward(
        self,
        vision_features: torch.Tensor,
        text_features: torch.Tensor,
        match_ids: List[str],
        **kwargs
    ) -> Dict[str, Any]:
        """
        Compute decoupled contrastive loss.

        Args:
            vision_features: Vision embeddings [batch_size, dim]
            text_features: Text embeddings [batch_size, dim]
            match_ids: IDs for semantic matching
            **kwargs: Additional arguments

        Returns:
            Dictionary with loss components and metrics
        """
        device = vision_features.device
        batch_size = vision_features.shape[0]

        # Safety checks
        if batch_size == 0:
            return {"loss": torch.tensor(0.0, device=device)}
        if batch_size != text_features.shape[0]:
            raise ValueError(f"Batch size mismatch: {vision_features.shape[0]} vs {text_features.shape[0]}")

        # Normalize features (uses base class mixin)
        vision_features = self.normalize(vision_features)
        text_features = self.normalize(text_features)

        # Create match matrix
        match_matrix = self._create_match_matrix(batch_size, match_ids, device)

        # Compute cross-modal losses (vision-to-text and text-to-vision)
        v2t_loss, t2v_loss = self._compute_cross_modal_loss(
            vision_features, text_features, match_matrix
        )

        # Compute instance discrimination losses (within modality)
        vision_inst_loss = self._compute_instance_loss(vision_features)
        text_inst_loss = self._compute_instance_loss(text_features)

        # Normalize losses
        num_pos = match_matrix.sum().item()
        if num_pos > 0:
            v2t_loss = v2t_loss / num_pos
            t2v_loss = t2v_loss / num_pos
            cross_modal_loss = (v2t_loss + t2v_loss) / 2
        else:
            return {"loss": torch.tensor(0.0, device=device)}

        vision_inst_loss = vision_inst_loss / batch_size
        text_inst_loss = text_inst_loss / batch_size
        instance_loss = (vision_inst_loss + text_inst_loss) / 2

        # Combine losses with weights
        total_loss = (
            cross_modal_loss
            + self.lambda_v * vision_inst_loss
            + self.lambda_t * text_inst_loss
        )

        # Compute metrics
        metrics = self._compute_metrics(
            vision_features, text_features, match_matrix
        )

        return {
            "loss": total_loss,
            "cross_modal_loss": cross_modal_loss,
            "instance_loss": instance_loss,
            "v2t_loss": v2t_loss,
            "t2v_loss": t2v_loss,
            "vision_inst_loss": vision_inst_loss,
            "text_inst_loss": text_inst_loss,
            **metrics
        }

    def _create_match_matrix(
        self,
        batch_size: int,
        match_ids: List[str],
        device: torch.device
    ) -> torch.Tensor:
        """Create boolean matrix indicating matches."""
        match_matrix = torch.zeros((batch_size, batch_size), dtype=torch.bool, device=device)
        for i in range(batch_size):
            for j in range(batch_size):
                match_matrix[i, j] = match_ids[i] == match_ids[j]
        return match_matrix

    def _compute_cross_modal_loss(
        self,
        vision_features: torch.Tensor,
        text_features: torch.Tensor,
        match_matrix: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute cross-modal contrastive losses."""
        batch_size = vision_features.shape[0]

        # Compute similarities (uses base class method with temperature)
        v2t_similarity = self.compute_similarity(vision_features, text_features, normalize=False)
        t2v_similarity = self.compute_similarity(text_features, vision_features, normalize=False)

        v2t_loss = 0.0
        t2v_loss = 0.0

        # Vision-to-text
        for i in range(batch_size):
            pos_indices = torch.where(match_matrix[i])[0]
            neg_indices = torch.where(~match_matrix[i])[0]
            if len(pos_indices) == 0 or len(neg_indices) == 0:
                continue

            pos_logits = v2t_similarity[i, pos_indices]
            neg_logits = v2t_similarity[i, neg_indices]

            pos_exp = torch.exp(pos_logits)
            neg_exp = torch.exp(neg_logits)
            v2t_loss += -torch.log(pos_exp.sum() / (pos_exp.sum() + neg_exp.sum()))

        # Text-to-vision
        for j in range(batch_size):
            pos_indices = torch.where(match_matrix[:, j])[0]
            neg_indices = torch.where(~match_matrix[:, j])[0]
            if len(pos_indices) == 0 or len(neg_indices) == 0:
                continue

            pos_logits = t2v_similarity[j, pos_indices]
            neg_logits = t2v_similarity[j, neg_indices]

            pos_exp = torch.exp(pos_logits)
            neg_exp = torch.exp(neg_logits)
            t2v_loss += -torch.log(pos_exp.sum() / (pos_exp.sum() + neg_exp.sum()))

        return v2t_loss, t2v_loss

    def _compute_instance_loss(
        self,
        features: torch.Tensor
    ) -> torch.Tensor:
        """Compute instance discrimination loss within a modality."""
        batch_size = features.shape[0]

        # Self-similarity with temperature
        similarity = torch.matmul(features, features.T) / self.temperature

        inst_loss = 0.0
        for i in range(batch_size):
            # Positive: self (index i)
            pos_indices = torch.tensor([i], device=features.device)

            # Negatives: all others
            neg_indices = torch.cat([
                torch.arange(0, i, device=features.device),
                torch.arange(i + 1, batch_size, device=features.device)
            ])

            if len(neg_indices) == 0:
                continue

            pos_logits = similarity[i, pos_indices]
            neg_logits = similarity[i, neg_indices]

            pos_exp = torch.exp(pos_logits)
            neg_exp = torch.exp(neg_logits)
            inst_loss += -torch.log(pos_exp.sum() / (pos_exp.sum() + neg_exp.sum()))

        return inst_loss

    def _compute_metrics(
        self,
        vision_features: torch.Tensor,
        text_features: torch.Tensor,
        match_matrix: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """Compute accuracy metrics."""
        with torch.no_grad():
            batch_size = vision_features.shape[0]

            # Compute similarities
            v2t_similarity = self.compute_similarity(vision_features, text_features, normalize=False)
            t2v_similarity = self.compute_similarity(text_features, vision_features, normalize=False)

            v2t_pred = torch.argmax(v2t_similarity, dim=1)
            t2v_pred = torch.argmax(t2v_similarity, dim=1)

            # Create target tensors
            v2t_targets = torch.zeros(batch_size, dtype=torch.long, device=vision_features.device)
            t2v_targets = torch.zeros(batch_size, dtype=torch.long, device=vision_features.device)

            for i in range(batch_size):
                # Find first matching text for each vision
                matches = torch.where(match_matrix[i])[0]
                v2t_targets[i] = matches[0] if len(matches) > 0 else i

                # Find first matching vision for each text
                matches = torch.where(match_matrix[:, i])[0]
                t2v_targets[i] = matches[0] if len(matches) > 0 else i

            v2t_accuracy = (v2t_pred == v2t_targets).float().mean()
            t2v_accuracy = (t2v_pred == t2v_targets).float().mean()

            return {
                "v2t_accuracy": v2t_accuracy,
                "t2v_accuracy": t2v_accuracy,
                "accuracy": (v2t_accuracy + t2v_accuracy) / 2
            }
