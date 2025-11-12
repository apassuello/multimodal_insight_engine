"""Mixed multimodal contrastive loss using base classes.

Refactored from multimodal_mixed_contrastive_loss.py (560 lines) to eliminate
code duplication. Reduces to ~180 lines by leveraging BaseContrastiveLoss.

Combines multiple contrastive objectives:
- InfoNCE loss (CLIP-style)
- NT-Xent loss (SimCLR-style)
- Supervised contrastive loss (with class labels)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Any
import logging

from ..base import BaseContrastiveLoss

logger = logging.getLogger(__name__)


class MixedMultimodalLoss(BaseContrastiveLoss):
    """
    Mixed contrastive loss combining multiple objectives with configurable weights.

    Supports:
    - InfoNCE: Standard contrastive loss (CLIP-style)
    - NT-Xent: Normalized temperature-scaled cross entropy (SimCLR)
    - Supervised: Uses class labels to form positives

    This refactored version uses BaseContrastiveLoss to eliminate duplication
    of normalization, temperature scaling, similarity computation, and projection.
    """

    def __init__(
        self,
        temperature: float = 0.07,
        loss_weights: Optional[Dict[str, float]] = None,
        reduction: str = "mean",
        input_dim: Optional[int] = None,
        projection_dim: int = 256,
        add_projection: bool = False,
        **kwargs
    ):
        """
        Initialize mixed contrastive loss.

        Args:
            temperature: Temperature for similarity scaling
            loss_weights: Dict mapping loss types to weights
                         (e.g., {"infonce": 0.4, "nt_xent": 0.3, "supervised": 0.3})
            reduction: Loss reduction method
            input_dim: Input feature dimension
            projection_dim: Projection space dimension
            add_projection: Whether to add MLP projection
        """
        # Initialize base class with all mixins
        super().__init__(
            temperature=temperature,
            normalize_features=True,
            use_projection=add_projection,
            projection_input_dim=input_dim,
            projection_hidden_dim=input_dim if input_dim else projection_dim,
            projection_output_dim=projection_dim,
            reduction=reduction,
            **kwargs
        )

        # Set default loss weights if not provided
        if loss_weights is None:
            loss_weights = {
                "infonce": 0.4,
                "nt_xent": 0.3,
                "supervised": 0.3,
            }
        self.loss_weights = loss_weights

    def forward(
        self,
        vision_features: torch.Tensor,
        text_features: torch.Tensor,
        match_ids: Optional[List[str]] = None,
        labels: Optional[torch.Tensor] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Compute mixed contrastive loss.

        Args:
            vision_features: Vision embeddings [batch_size, dim]
            text_features: Text embeddings [batch_size, dim]
            match_ids: IDs for semantic matching
            labels: Class labels for supervised loss
            **kwargs: Additional arguments

        Returns:
            Dictionary with loss components and metrics
        """
        batch_size = vision_features.shape[0]

        # Safety checks
        if batch_size == 0:
            return {"loss": torch.tensor(0.0, device=vision_features.device)}

        if batch_size != text_features.shape[0]:
            raise ValueError(
                f"Batch size mismatch: vision={vision_features.shape[0]}, "
                f"text={text_features.shape[0]}"
            )

        # Project and normalize features (uses mixins from base class)
        vision_features = self.project(vision_features)
        text_features = self.project(text_features)
        vision_features = self.normalize(vision_features)
        text_features = self.normalize(text_features)

        # Create targets based on match_ids
        v2t_targets, t2i_targets = self._create_targets(
            batch_size, match_ids, vision_features.device
        )

        # Compute similarity (uses base class method with temperature)
        similarity = self.compute_similarity(vision_features, text_features, normalize=False)

        # Add noise for small batches
        if batch_size < 16 and self.training:
            noise_scale = max(0.005, 0.02 * (16 - batch_size) / 16)
            similarity = similarity + torch.randn_like(similarity) * noise_scale

        # Compute InfoNCE loss
        loss_v2t = F.cross_entropy(similarity, v2t_targets, reduction=self.reduction)
        loss_t2v = F.cross_entropy(similarity.T, t2i_targets, reduction=self.reduction)
        loss_infonce = (loss_v2t + loss_t2v) / 2

        # Compute NT-Xent loss (uses base class method)
        loss_nt_xent = self.nt_xent_loss(
            vision_features,
            text_features,
            temperature=self.temp,
            reduction=self.reduction
        )

        # Compute supervised contrastive loss if labels provided
        if labels is not None:
            loss_supervised = self._supervised_contrastive_loss(
                vision_features, text_features, labels
            )
        else:
            loss_supervised = torch.tensor(0.0, device=vision_features.device)

        # Combine losses according to weights
        total_loss = (
            self.loss_weights["infonce"] * loss_infonce
            + self.loss_weights["nt_xent"] * loss_nt_xent
            + self.loss_weights["supervised"] * loss_supervised
        )

        # Compute metrics
        metrics = self._compute_metrics(
            similarity, v2t_targets, t2i_targets, vision_features, text_features
        )

        return {
            "loss": total_loss,
            "loss_infonce": loss_infonce,
            "loss_nt_xent": loss_nt_xent,
            "loss_supervised": loss_supervised,
            **metrics
        }

    def _create_targets(
        self,
        batch_size: int,
        match_ids: Optional[List[str]],
        device: torch.device
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Create target indices based on semantic matching IDs.

        Args:
            batch_size: Batch size
            match_ids: Optional semantic matching IDs
            device: Device for tensors

        Returns:
            Tuple of (v2t_targets, t2i_targets)
        """
        if match_ids is None:
            # Diagonal matching (position-based)
            targets = torch.arange(batch_size, device=device)
            return targets, targets

        # Convert to strings for consistent comparison
        string_match_ids = [str(mid) for mid in match_ids]

        # Create match matrix
        match_matrix = torch.zeros(
            (batch_size, batch_size),
            dtype=torch.bool,
            device=device
        )

        for i in range(batch_size):
            for j in range(batch_size):
                match_matrix[i, j] = string_match_ids[i] == string_match_ids[j]

        # Create v2t targets
        v2t_targets = torch.zeros(batch_size, dtype=torch.long, device=device)
        for i in range(batch_size):
            matches = torch.where(match_matrix[i])[0]
            if len(matches) > 0:
                idx = torch.randint(0, len(matches), (1,), device=device)[0]
                v2t_targets[i] = matches[idx]
            else:
                v2t_targets[i] = i

        # Create t2i targets
        t2i_targets = torch.zeros(batch_size, dtype=torch.long, device=device)
        for j in range(batch_size):
            matches = torch.where(match_matrix[:, j])[0]
            if len(matches) > 0:
                idx = torch.randint(0, len(matches), (1,), device=device)[0]
                t2i_targets[j] = matches[idx]
            else:
                t2i_targets[j] = j

        return v2t_targets, t2i_targets

    def _supervised_contrastive_loss(
        self,
        vision_features: torch.Tensor,
        text_features: torch.Tensor,
        labels: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute supervised contrastive loss using class labels.

        Args:
            vision_features: Vision embeddings
            text_features: Text embeddings
            labels: Class labels

        Returns:
            Supervised contrastive loss
        """
        batch_size = vision_features.shape[0]

        # Concatenate features and labels
        features = torch.cat([vision_features, text_features], dim=0)
        labels_expanded = torch.cat([labels, labels], dim=0)

        # Create mask where 1 indicates same class
        labels_expanded = labels_expanded.contiguous().view(-1, 1)
        mask = torch.eq(labels_expanded, labels_expanded.T).float()

        # Remove self-similarities
        identity_mask = torch.eye(2 * batch_size, device=vision_features.device)
        mask = mask - identity_mask

        # Compute similarity (with temperature)
        similarity = torch.matmul(features, features.T) / self.temp

        # Mask for non-self samples
        non_self_mask = 1 - identity_mask

        # Compute positive and negative masks
        pos_mask = mask * non_self_mask

        # Compute positive term
        pos_term = torch.sum(similarity * pos_mask, dim=1)
        pos_count = torch.clamp(torch.sum(pos_mask, dim=1), min=1)
        pos_term = pos_term / pos_count

        # Compute denominator with log-sum-exp
        neg_exp = torch.exp(similarity) * non_self_mask
        per_sample_denom = torch.log(torch.sum(neg_exp, dim=1))

        # Final loss
        loss = -pos_term + per_sample_denom

        # Apply reduction
        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        else:
            return loss

    def _compute_metrics(
        self,
        similarity: torch.Tensor,
        v2t_targets: torch.Tensor,
        t2i_targets: torch.Tensor,
        vision_features: torch.Tensor,
        text_features: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """Compute accuracy and retrieval metrics."""
        with torch.no_grad():
            batch_size = similarity.shape[0]

            # Accuracy
            v2t_pred = torch.argmax(similarity, dim=1)
            v2t_accuracy = (v2t_pred == v2t_targets).float().mean()

            t2v_pred = torch.argmax(similarity.T, dim=1)
            t2v_accuracy = (t2v_pred == t2i_targets).float().mean()

            accuracy = (v2t_accuracy + t2v_accuracy) / 2.0

            # Recall@K
            recalls = {}
            for k in [1, 5, 10]:
                if k <= batch_size:
                    v2t_topk = torch.topk(similarity, k, dim=1)[1]
                    v2t_recall = (v2t_topk == v2t_targets.unsqueeze(1)).any(dim=1).float().mean()

                    t2v_topk = torch.topk(similarity.T, k, dim=1)[1]
                    t2v_recall = (t2v_topk == t2i_targets.unsqueeze(1)).any(dim=1).float().mean()

                    recalls[f"v2t_recall@{k}"] = v2t_recall
                    recalls[f"t2i_recall@{k}"] = t2v_recall
                    recalls[f"avg_recall@{k}"] = (v2t_recall + t2v_recall) / 2.0

            return {
                "v2t_accuracy": v2t_accuracy,
                "t2v_accuracy": t2v_accuracy,
                "accuracy": accuracy,
                **{f"recalls.{k}": v for k, v in recalls.items()}
            }
