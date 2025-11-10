"""Mixins for shared loss function functionality.

This module provides mixins that can be composed with base classes
to add specific capabilities like temperature scaling, normalization,
projection heads, and hard negative mining.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


class TemperatureScalingMixin:
    """Mixin for temperature-scaled losses.

    Adds temperature parameter that can be either fixed or learnable.
    Common in contrastive learning to control the concentration of the distribution.
    """

    def __init__(
        self,
        *args,
        temperature: float = 0.07,
        learnable_temperature: bool = False,
        **kwargs
    ):
        """
        Initialize temperature scaling.

        Args:
            temperature: Initial temperature value
            learnable_temperature: Whether temperature should be a learnable parameter
        """
        super().__init__(*args, **kwargs)
        if learnable_temperature:
            self.temperature = nn.Parameter(torch.tensor(temperature))
            self._is_learnable_temp = True
        else:
            self.register_buffer('_temperature', torch.tensor(temperature))
            self._is_learnable_temp = False

    @property
    def temp(self) -> torch.Tensor:
        """Get current temperature value."""
        if self._is_learnable_temp:
            return self.temperature
        return self._temperature

    def scale_by_temperature(self, logits: torch.Tensor) -> torch.Tensor:
        """Apply temperature scaling to logits."""
        return logits / self.temp


class NormalizationMixin:
    """Mixin for feature normalization.

    Adds L2 normalization capability commonly used in contrastive learning
    to compute cosine similarities.
    """

    def __init__(self, *args, normalize_features: bool = True, **kwargs):
        """
        Initialize normalization.

        Args:
            normalize_features: Whether to L2-normalize features
        """
        super().__init__(*args, **kwargs)
        self.normalize_features = normalize_features

    def normalize(
        self,
        features: torch.Tensor,
        dim: int = -1,
        eps: float = 1e-8
    ) -> torch.Tensor:
        """
        L2-normalize features along specified dimension.

        Args:
            features: Input features
            dim: Dimension to normalize along
            eps: Small constant for numerical stability

        Returns:
            Normalized features
        """
        if self.normalize_features:
            return F.normalize(features, p=2, dim=dim, eps=eps)
        return features


class ProjectionMixin:
    """Mixin for learnable projection heads.

    Adds MLP projection heads to transform features before computing loss.
    Common in contrastive learning (SimCLR, MoCo, etc.).
    """

    def __init__(
        self,
        *args,
        use_projection: bool = False,
        input_dim: Optional[int] = None,
        projection_dim: int = 256,
        projection_hidden_dim: Optional[int] = None,
        num_projection_layers: int = 2,
        **kwargs
    ):
        """
        Initialize projection head.

        Args:
            use_projection: Whether to use projection heads
            input_dim: Input feature dimension (required if use_projection=True)
            projection_dim: Output projection dimension
            projection_hidden_dim: Hidden layer dimension (default: same as projection_dim)
            num_projection_layers: Number of layers in projection head (2 or 3)
        """
        super().__init__(*args, **kwargs)
        self.use_projection = use_projection

        if use_projection:
            assert input_dim is not None, "input_dim required when use_projection=True"

            if projection_hidden_dim is None:
                projection_hidden_dim = projection_dim

            if num_projection_layers == 2:
                self.projection = nn.Sequential(
                    nn.Linear(input_dim, projection_hidden_dim),
                    nn.ReLU(),
                    nn.Linear(projection_hidden_dim, projection_dim)
                )
            elif num_projection_layers == 3:
                self.projection = nn.Sequential(
                    nn.Linear(input_dim, projection_hidden_dim),
                    nn.BatchNorm1d(projection_hidden_dim),
                    nn.ReLU(),
                    nn.Linear(projection_hidden_dim, projection_hidden_dim),
                    nn.BatchNorm1d(projection_hidden_dim),
                    nn.ReLU(),
                    nn.Linear(projection_hidden_dim, projection_dim)
                )
            else:
                raise ValueError(f"num_projection_layers must be 2 or 3, got {num_projection_layers}")
        else:
            self.projection = nn.Identity()

    def project(self, features: torch.Tensor) -> torch.Tensor:
        """
        Apply projection head to features.

        Args:
            features: Input features

        Returns:
            Projected features
        """
        return self.projection(features)


class HardNegativeMiningMixin:
    """Mixin for hard negative mining in contrastive losses.

    Adds functionality to identify and weight hard negative examples
    that are close to the anchor in feature space.
    """

    def __init__(
        self,
        *args,
        use_hard_negatives: bool = False,
        hard_negative_weight: float = 1.0,
        hard_negative_percentile: float = 0.5,
        **kwargs
    ):
        """
        Initialize hard negative mining.

        Args:
            use_hard_negatives: Whether to use hard negative mining
            hard_negative_weight: Weight for hard negatives vs all negatives
            hard_negative_percentile: Percentile threshold for "hard" (0.5 = median)
        """
        super().__init__(*args, **kwargs)
        self.use_hard_negatives = use_hard_negatives
        self.hard_negative_weight = hard_negative_weight
        self.hard_negative_percentile = hard_negative_percentile

    def mine_hard_negatives(
        self,
        similarities: torch.Tensor,
        positive_mask: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Identify hard negative examples based on similarity scores.

        Args:
            similarities: Similarity matrix [batch_size, batch_size]
            positive_mask: Boolean mask indicating positive pairs

        Returns:
            Tuple of (hard_negative_mask, weights)
        """
        if not self.use_hard_negatives:
            # Return uniform weights for all negatives
            negative_mask = ~positive_mask
            weights = torch.ones_like(similarities)
            return negative_mask, weights

        # Get negative similarities
        negative_mask = ~positive_mask
        negative_similarities = similarities.masked_fill(~negative_mask, float('-inf'))

        # Find threshold for hard negatives (high similarity = hard)
        valid_negatives = negative_similarities[negative_mask]
        if len(valid_negatives) == 0:
            weights = torch.ones_like(similarities)
            return negative_mask, weights

        threshold = torch.quantile(
            valid_negatives,
            self.hard_negative_percentile
        )

        # Create weight matrix: higher weight for hard negatives
        hard_negative_mask = (similarities > threshold) & negative_mask
        weights = torch.ones_like(similarities)
        weights[hard_negative_mask] = self.hard_negative_weight

        return negative_mask, weights
