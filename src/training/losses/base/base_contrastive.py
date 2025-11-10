"""Base class for contrastive learning losses.

This module provides the foundation for all contrastive loss implementations,
including shared logic for similarity computation, positive/negative pair handling,
and common contrastive loss patterns.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Union, List
from abc import ABC, abstractmethod

from .mixins import (
    TemperatureScalingMixin,
    NormalizationMixin,
    ProjectionMixin,
    HardNegativeMiningMixin,
)


class BaseContrastiveLoss(
    TemperatureScalingMixin,
    NormalizationMixin,
    ProjectionMixin,
    HardNegativeMiningMixin,
    nn.Module,
    ABC
):
    """
    Base class for all contrastive learning losses.

    This class provides common functionality used across different contrastive
    loss variants (InfoNCE, NT-Xent, SimCLR, CLIP, etc.) including:
    - Feature normalization
    - Temperature scaling
    - Projection heads
    - Similarity computation
    - Hard negative mining
    - Positive pair identification

    Subclasses should implement the forward() method to define their specific
    contrastive loss computation.
    """

    def __init__(
        self,
        temperature: float = 0.07,
        normalize_features: bool = True,
        learnable_temperature: bool = False,
        use_projection: bool = False,
        input_dim: Optional[int] = None,
        projection_dim: int = 256,
        use_hard_negatives: bool = False,
        hard_negative_weight: float = 1.0,
        reduction: str = "mean",
        **kwargs
    ):
        """
        Initialize base contrastive loss.

        Args:
            temperature: Temperature parameter for scaling similarities
            normalize_features: Whether to L2-normalize features
            learnable_temperature: Whether temperature should be learnable
            use_projection: Whether to use projection heads
            input_dim: Input dimension for projection heads
            projection_dim: Output dimension for projection heads
            use_hard_negatives: Whether to use hard negative mining
            hard_negative_weight: Weight for hard negatives
            reduction: How to reduce loss ("mean", "sum", or "none")
        """
        super().__init__(
            temperature=temperature,
            normalize_features=normalize_features,
            learnable_temperature=learnable_temperature,
            use_projection=use_projection,
            input_dim=input_dim,
            projection_dim=projection_dim,
            use_hard_negatives=use_hard_negatives,
            hard_negative_weight=hard_negative_weight,
        )
        assert reduction in ["mean", "sum", "none"], \
            f"reduction must be 'mean', 'sum', or 'none', got {reduction}"
        self.reduction = reduction

    def compute_similarity(
        self,
        features1: torch.Tensor,
        features2: torch.Tensor,
        normalize: bool = True
    ) -> torch.Tensor:
        """
        Compute pairwise similarity between two sets of features.

        Args:
            features1: First set of features [batch_size, dim]
            features2: Second set of features [batch_size, dim]
            normalize: Whether to normalize features before computing similarity

        Returns:
            Similarity matrix [batch_size, batch_size]
        """
        if normalize:
            features1 = self.normalize(features1)
            features2 = self.normalize(features2)

        # Compute cosine similarity
        similarity = torch.matmul(features1, features2.T)

        # Apply temperature scaling
        similarity = self.scale_by_temperature(similarity)

        return similarity

    def create_positive_mask(
        self,
        batch_size: int,
        match_ids: Optional[List[str]] = None,
        device: Optional[torch.device] = None
    ) -> torch.Tensor:
        """
        Create boolean mask indicating positive pairs.

        Args:
            batch_size: Size of the batch
            match_ids: Optional list of IDs to determine which samples match
            device: Device to create mask on

        Returns:
            Boolean mask [batch_size, batch_size] where True indicates positive pairs
        """
        if device is None:
            device = torch.device('cpu')

        if match_ids is None:
            # Default: diagonal elements are positives (self-matching)
            mask = torch.eye(batch_size, dtype=torch.bool, device=device)
        else:
            # Create mask based on matching IDs
            mask = torch.zeros(batch_size, batch_size, dtype=torch.bool, device=device)
            for i, id_i in enumerate(match_ids):
                for j, id_j in enumerate(match_ids):
                    if id_i == id_j:
                        mask[i, j] = True

        return mask

    def info_nce_loss(
        self,
        similarity: torch.Tensor,
        positive_mask: torch.Tensor,
        negative_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Compute InfoNCE (Normalized Temperature-scaled Cross Entropy) loss.

        This is the core loss used in many contrastive learning methods including
        SimCLR, MoCo, and CLIP.

        Args:
            similarity: Similarity matrix [batch_size, batch_size]
            positive_mask: Boolean mask indicating positive pairs
            negative_mask: Optional boolean mask indicating valid negatives

        Returns:
            InfoNCE loss value
        """
        batch_size = similarity.shape[0]

        if negative_mask is None:
            # All non-positive pairs are negatives
            negative_mask = ~positive_mask

        # For each anchor, compute loss
        losses = []
        for i in range(batch_size):
            # Get similarities for this anchor
            anchor_similarities = similarity[i]

            # Positive examples for this anchor
            positives = anchor_similarities[positive_mask[i]]

            # Negative examples for this anchor
            negatives = anchor_similarities[negative_mask[i]]

            if len(positives) == 0:
                continue  # Skip if no positives

            # InfoNCE: log( exp(pos) / (exp(pos) + sum(exp(neg))) )
            # Equivalently: pos - log(exp(pos) + sum(exp(neg)))
            # Use logsumexp for numerical stability
            pos_exp = torch.exp(positives)
            neg_exp = torch.exp(negatives)

            # For each positive, compute loss
            for pos in positives:
                denominator = torch.logsumexp(
                    torch.cat([pos.unsqueeze(0), negatives]),
                    dim=0
                )
                loss = -pos + denominator
                losses.append(loss)

        if len(losses) == 0:
            return torch.tensor(0.0, device=similarity.device)

        losses = torch.stack(losses)

        if self.reduction == "mean":
            return losses.mean()
        elif self.reduction == "sum":
            return losses.sum()
        else:  # none
            return losses

    def nt_xent_loss(
        self,
        features: torch.Tensor,
        labels: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Compute NT-Xent (Normalized Temperature-scaled Cross Entropy) loss.

        This is the loss used in SimCLR, operating on augmented pairs.

        Args:
            features: Concatenated features from two views [2*batch_size, dim]
            labels: Optional labels for supervised variant

        Returns:
            NT-Xent loss value
        """
        # Assume features contains [view1; view2] concatenated
        batch_size = features.shape[0] // 2

        # Normalize features
        features = self.normalize(features)

        # Compute similarity matrix
        similarity = torch.matmul(features, features.T)
        similarity = self.scale_by_temperature(similarity)

        # Create positive pair mask
        # In SimCLR, sample i and sample i+batch_size are augmented versions
        positive_mask = torch.zeros(2 * batch_size, 2 * batch_size, dtype=torch.bool, device=features.device)
        for i in range(batch_size):
            positive_mask[i, i + batch_size] = True
            positive_mask[i + batch_size, i] = True

        # Remove self-similarity from consideration
        mask_self = torch.eye(2 * batch_size, dtype=torch.bool, device=features.device)
        negative_mask = ~(positive_mask | mask_self)

        return self.info_nce_loss(similarity, positive_mask, negative_mask)

    @abstractmethod
    def forward(
        self,
        *args,
        **kwargs
    ) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Compute the loss.

        Subclasses must implement this method to define their specific loss computation.

        Returns:
            Either a single loss tensor or a dictionary containing multiple loss components
        """
        raise NotImplementedError("Subclasses must implement forward()")

    def reduce_loss(
        self,
        loss: torch.Tensor,
        reduction: Optional[str] = None
    ) -> torch.Tensor:
        """
        Apply reduction to loss values.

        Args:
            loss: Loss tensor
            reduction: Reduction method (if None, uses self.reduction)

        Returns:
            Reduced loss
        """
        if reduction is None:
            reduction = self.reduction

        if reduction == "mean":
            return loss.mean()
        elif reduction == "sum":
            return loss.sum()
        else:  # none
            return loss
