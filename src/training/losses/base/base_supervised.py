"""Base class for supervised learning losses.

This module provides the foundation for supervised loss implementations,
including label handling, class weighting, and common supervised patterns.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Union
from abc import ABC, abstractmethod

from .mixins import NormalizationMixin, ProjectionMixin


class BaseSupervisedLoss(
    NormalizationMixin,
    ProjectionMixin,
    nn.Module,
    ABC
):
    """
    Base class for supervised learning losses.

    This class provides common functionality for supervised losses including:
    - Class weighting
    - Label smoothing
    - Projection heads (for representation learning)
    - Feature normalization

    Subclasses should implement the forward() method to define their specific
    loss computation.
    """

    def __init__(
        self,
        num_classes: Optional[int] = None,
        class_weights: Optional[torch.Tensor] = None,
        label_smoothing: float = 0.0,
        normalize_features: bool = False,
        use_projection: bool = False,
        input_dim: Optional[int] = None,
        projection_dim: Optional[int] = None,
        reduction: str = "mean",
        **kwargs
    ):
        """
        Initialize base supervised loss.

        Args:
            num_classes: Number of classes (required for some losses)
            class_weights: Optional weights for each class
            label_smoothing: Label smoothing factor (0.0 = no smoothing)
            normalize_features: Whether to L2-normalize features
            use_projection: Whether to use projection heads
            input_dim: Input dimension for projection heads
            projection_dim: Output dimension for projection heads
            reduction: How to reduce loss ("mean", "sum", or "none")
        """
        super().__init__(
            normalize_features=normalize_features,
            use_projection=use_projection,
            input_dim=input_dim,
            projection_dim=projection_dim or num_classes,
        )

        self.num_classes = num_classes
        self.label_smoothing = label_smoothing

        assert reduction in ["mean", "sum", "none"], \
            f"reduction must be 'mean', 'sum', or 'none', got {reduction}"
        self.reduction = reduction

        # Register class weights as buffer (not a parameter)
        if class_weights is not None:
            if num_classes is not None:
                assert len(class_weights) == num_classes, \
                    f"class_weights length ({len(class_weights)}) must match num_classes ({num_classes})"
            self.register_buffer('class_weights', class_weights)
        else:
            self.class_weights = None

    def apply_label_smoothing(
        self,
        labels: torch.Tensor,
        num_classes: Optional[int] = None
    ) -> torch.Tensor:
        """
        Apply label smoothing to hard labels.

        Converts hard labels to soft labels using:
        y_smooth = (1 - smoothing) * y_hard + smoothing / num_classes

        Args:
            labels: Hard labels [batch_size] or one-hot [batch_size, num_classes]
            num_classes: Number of classes (uses self.num_classes if not provided)

        Returns:
            Smoothed labels [batch_size, num_classes]
        """
        if self.label_smoothing == 0.0:
            if labels.dim() == 1:
                # Convert to one-hot
                if num_classes is None:
                    num_classes = self.num_classes
                assert num_classes is not None, "num_classes required for label conversion"
                return F.one_hot(labels, num_classes).float()
            return labels.float()

        if num_classes is None:
            if labels.dim() == 2:
                num_classes = labels.shape[1]
            else:
                num_classes = self.num_classes

        assert num_classes is not None, "num_classes required for label smoothing"

        # Convert to one-hot if needed
        if labels.dim() == 1:
            labels = F.one_hot(labels, num_classes).float()

        # Apply smoothing
        smoothed = labels * (1 - self.label_smoothing)
        smoothed += self.label_smoothing / num_classes

        return smoothed

    def weighted_cross_entropy(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        weights: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Compute weighted cross-entropy loss.

        Args:
            logits: Predicted logits [batch_size, num_classes]
            labels: True labels [batch_size] or [batch_size, num_classes]
            weights: Optional per-sample weights [batch_size]

        Returns:
            Cross-entropy loss
        """
        # Apply label smoothing
        if labels.dim() == 1:
            # Hard labels - apply smoothing and convert to probs
            target_probs = self.apply_label_smoothing(labels)
        else:
            # Already soft labels
            target_probs = labels

        # Compute log probabilities
        log_probs = F.log_softmax(logits, dim=-1)

        # Compute cross-entropy
        loss = -(target_probs * log_probs).sum(dim=-1)

        # Apply class weights if provided
        if self.class_weights is not None and labels.dim() == 1:
            class_weights = self.class_weights[labels]
            loss = loss * class_weights

        # Apply per-sample weights if provided
        if weights is not None:
            loss = loss * weights

        # Apply reduction
        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        else:  # none
            return loss

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
