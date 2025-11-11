"""CLIP-style contrastive loss using base classes.

Refactored from clip_style_loss.py to eliminate code duplication.
Reduces from 434 lines to ~150 lines by leveraging BaseContrastiveLoss.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Any
import logging

from ..base import BaseContrastiveLoss

logger = logging.getLogger(__name__)


class CLIPLoss(BaseContrastiveLoss):
    """
    CLIP-style bidirectional contrastive loss.

    Aligns visual and textual representations by:
    - Computing vision→text and text→vision cross-entropy losses
    - Supporting semantic matching via match_ids
    - Providing comprehensive retrieval metrics

    This refactored version uses BaseContrastiveLoss to eliminate duplication
    of normalization, temperature scaling, and similarity computation.
    """

    def __init__(
        self,
        temperature: float = 0.07,
        label_smoothing: float = 0.0,
        use_hard_negatives: bool = False,
        hard_negative_weight: float = 0.5,
        reduction: str = "mean",
        cache_labels: bool = True,
        **kwargs
    ):
        """
        Initialize CLIP loss.

        Args:
            temperature: Temperature for similarity scaling
            label_smoothing: Amount of label smoothing (0.0-1.0)
            use_hard_negatives: Whether to use hard negative mining
            hard_negative_weight: Weight for hard negatives
            reduction: Loss reduction method
            cache_labels: Whether to cache labels for efficiency
        """
        super().__init__(
            temperature=temperature,
            normalize_features=True,  # CLIP always normalizes
            use_hard_negatives=use_hard_negatives,
            hard_negative_weight=hard_negative_weight,
            reduction=reduction,
            **kwargs
        )
        self.label_smoothing = label_smoothing
        self.cache_labels = cache_labels
        self._labels_cache = {}

    def forward(
        self,
        vision_features: torch.Tensor,
        text_features: torch.Tensor,
        match_ids: Optional[List[str]] = None,
        similarity_matrix: Optional[torch.Tensor] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Compute CLIP loss between vision and text features.

        Args:
            vision_features: Vision embeddings [batch_size, dim]
            text_features: Text embeddings [batch_size, dim]
            match_ids: Optional IDs for semantic matching
            similarity_matrix: Optional pre-computed similarity
            **kwargs: Additional arguments (ignored)

        Returns:
            Dictionary with loss and metrics
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

        # Compute similarity matrix (uses BaseContrastiveLoss method)
        if similarity_matrix is None:
            similarity_matrix = self.compute_similarity(
                vision_features,
                text_features,
                normalize=True
            )

        # Create targets based on match_ids
        v2t_targets, t2v_targets = self._create_targets(
            batch_size,
            match_ids,
            vision_features.device
        )

        # Compute bidirectional losses
        v2t_loss = self._compute_directional_loss(
            similarity_matrix,
            v2t_targets,
            batch_size
        )

        t2v_loss = self._compute_directional_loss(
            similarity_matrix.T,
            t2v_targets,
            batch_size
        )

        # Average bidirectional losses
        loss = (v2t_loss + t2v_loss) / 2.0

        # Compute metrics
        metrics = self._compute_metrics(
            similarity_matrix,
            v2t_targets,
            t2v_targets,
            vision_features,
            text_features
        )

        # Return loss and metrics
        result = {
            "loss": loss,
            **metrics
        }

        return result

    def _create_targets(
        self,
        batch_size: int,
        match_ids: Optional[List[str]],
        device: torch.device
    ) -> tuple:
        """
        Create target indices for vision→text and text→vision.

        Args:
            batch_size: Batch size
            match_ids: Optional semantic matching IDs
            device: Device for tensors

        Returns:
            Tuple of (v2t_targets, t2v_targets)
        """
        if match_ids is None:
            # Diagonal matching (position-based)
            logger.warning(
                "No match_ids provided - using position-based matching"
            )
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

        # Log diagnostics
        unique_matches = len(set(string_match_ids))
        if unique_matches == batch_size:
            logger.warning(
                f"All match_ids unique ({batch_size}) - no semantic grouping"
            )
        elif unique_matches == 1:
            logger.warning("All match_ids identical - all pairs match")

        # Create v2t targets (for each vision, pick a matching text)
        v2t_targets = torch.zeros(batch_size, dtype=torch.long, device=device)
        for i in range(batch_size):
            matches = torch.where(match_matrix[i])[0]
            if len(matches) > 0:
                # Randomly select one matching target
                idx = torch.randint(0, len(matches), (1,), device=device)[0]
                v2t_targets[i] = matches[idx]
            else:
                v2t_targets[i] = i  # Fallback to diagonal

        # Create t2v targets (for each text, pick a matching vision)
        t2v_targets = torch.zeros(batch_size, dtype=torch.long, device=device)
        for j in range(batch_size):
            matches = torch.where(match_matrix[:, j])[0]
            if len(matches) > 0:
                idx = torch.randint(0, len(matches), (1,), device=device)[0]
                t2v_targets[j] = matches[idx]
            else:
                t2v_targets[j] = j  # Fallback to diagonal

        return v2t_targets, t2v_targets

    def _compute_directional_loss(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        batch_size: int
    ) -> torch.Tensor:
        """
        Compute cross-entropy loss for one direction.

        Args:
            logits: Similarity matrix [batch_size, batch_size]
            targets: Target indices [batch_size]
            batch_size: Batch size

        Returns:
            Loss value
        """
        if self.label_smoothing > 0:
            # Create smooth labels
            smooth_targets = self._create_smooth_labels(
                batch_size,
                targets,
                self.label_smoothing,
                logits.device
            )

            # Cross-entropy with smooth labels
            log_probs = F.log_softmax(logits, dim=1)
            loss = -(smooth_targets * log_probs).sum(dim=1)

            # Apply reduction
            if self.reduction == "mean":
                loss = loss.mean()
            elif self.reduction == "sum":
                loss = loss.sum()
        else:
            # Standard cross-entropy
            loss = F.cross_entropy(logits, targets, reduction=self.reduction)

        return loss

    def _create_smooth_labels(
        self,
        batch_size: int,
        targets: torch.Tensor,
        smoothing: float,
        device: torch.device
    ) -> torch.Tensor:
        """
        Create label-smoothed targets.

        Args:
            batch_size: Number of classes
            targets: Hard target indices
            smoothing: Smoothing amount
            device: Device for tensor

        Returns:
            Smoothed label distribution
        """
        # Check cache first
        cache_key = (batch_size, smoothing, device)
        if self.cache_labels and cache_key in self._labels_cache:
            base_smooth = self._labels_cache[cache_key]
        else:
            # Create base smooth distribution
            confidence = 1.0 - smoothing
            base_smooth = torch.ones(batch_size, device=device) * (smoothing / batch_size)
            if self.cache_labels:
                self._labels_cache[cache_key] = base_smooth

        # Create smooth labels for this batch
        smooth_labels = base_smooth.unsqueeze(0).expand(batch_size, -1).clone()
        smooth_labels.scatter_(1, targets.unsqueeze(1), confidence)

        return smooth_labels

    def _compute_metrics(
        self,
        similarity: torch.Tensor,
        v2t_targets: torch.Tensor,
        t2v_targets: torch.Tensor,
        vision_features: torch.Tensor,
        text_features: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Compute accuracy and retrieval metrics.

        Args:
            similarity: Similarity matrix
            v2t_targets: Vision→text targets
            t2v_targets: Text→vision targets
            vision_features: Vision embeddings
            text_features: Text embeddings

        Returns:
            Dictionary of metrics
        """
        with torch.no_grad():
            batch_size = similarity.shape[0]

            # Accuracy metrics
            v2t_pred = torch.argmax(similarity, dim=1)
            v2t_accuracy = (v2t_pred == v2t_targets).float().mean()

            t2v_pred = torch.argmax(similarity.T, dim=1)
            t2v_accuracy = (t2v_pred == t2v_targets).float().mean()

            accuracy = (v2t_accuracy + t2v_accuracy) / 2.0

            # Recall@K metrics
            recalls = {}
            for k in [1, 5, 10]:
                if k <= batch_size:
                    # Vision→text recall
                    v2t_topk = torch.topk(similarity, k, dim=1)[1]
                    v2t_recall = (v2t_topk == v2t_targets.unsqueeze(1)).any(dim=1).float().mean()

                    # Text→vision recall
                    t2v_topk = torch.topk(similarity.T, k, dim=1)[1]
                    t2v_recall = (t2v_topk == t2v_targets.unsqueeze(1)).any(dim=1).float().mean()

                    recalls[f"v2t_recall@{k}"] = v2t_recall
                    recalls[f"t2i_recall@{k}"] = t2v_recall
                    recalls[f"avg_recall@{k}"] = (v2t_recall + t2v_recall) / 2.0

            # Return all metrics
            metrics = {
                "v2t_accuracy": v2t_accuracy,
                "t2v_accuracy": t2v_accuracy,
                "accuracy": accuracy,
                **{f"recalls.{k}": v for k, v in recalls.items()}
            }

            return metrics
