# src/training/losses/decorrelation_loss.py

import logging
from typing import Any, Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


"""MODULE: decorrelation_loss.py
PURPOSE: Implements decorrelation regularization to prevent feature collapse in multimodal models.

KEY COMPONENTS:
- DecorrelationLoss: Loss that penalizes correlation between feature dimensions
  - Prevents feature collapse by encouraging diversity in representations
  - Works with vision features, text features, or both simultaneously
  - Tracks feature statistics to detect and warn about potential collapse
  - Configurable regularization strength and normalization options

DEPENDENCIES:
- PyTorch (torch, torch.nn, torch.nn.functional)
- Python standard library (logging, math)

SPECIAL NOTES:
- Particularly useful in Stage 1 and Stage 3 of the progressive training approach
- Directly combats feature collapse, a common issue in multimodal learning
- Works well in combination with other losses like contrastive or VICReg
- Provides early warnings for feature collapse through variance monitoring
"""


class DecorrelationLoss(nn.Module):
    """
    Implements decorrelation regularization to prevent feature collapse.

    Feature collapse occurs when a model learns to encode different inputs into
    very similar feature vectors, effectively collapsing the representation space.
    This loss explicitly penalizes correlation between feature dimensions, encouraging
    the model to use the full embedding space and maintain representational diversity.

    Key features:
    - Off-diagonal covariance regularization to reduce feature correlation
    - Support for regularizing both vision and text features independently
    - Configurable regularization strength for fine-tuning
    - Tracking of low-variance dimensions for early warning of collapse
    - Can be combined with other losses like contrastive or VICReg loss
    """

    def __init__(
        self,
        coef: float = 1.0,
        epsilon: float = 1e-5,
        normalize_embeddings: bool = True,
        track_feature_stats: bool = True,
        reduction: str = "mean",
    ):
        """
        Initialize the decorrelation loss.

        Args:
            coef: Coefficient controlling the strength of the decorrelation regularization
            epsilon: Small constant for numerical stability
            normalize_embeddings: Whether to normalize embeddings before computing covariance
            track_feature_stats: Whether to track feature statistics (variance, etc.)
            reduction: How to reduce the loss ("mean", "sum", or "none")
        """
        super().__init__()
        self.coef = coef
        self.epsilon = epsilon
        self.normalize_embeddings = normalize_embeddings
        self.track_feature_stats = track_feature_stats
        self.reduction = reduction

        # For tracking minimum variance across dimensions
        self.min_variance_threshold = 1e-4

    def _compute_covariance(self, features: torch.Tensor) -> torch.Tensor:
        """
        Compute the feature covariance matrix.

        Args:
            features: Feature tensor [batch_size, feature_dim]

        Returns:
            Covariance matrix [feature_dim, feature_dim]
        """
        batch_size, feature_dim = features.shape

        # Center the features (subtract mean)
        features = features - features.mean(dim=0, keepdim=True)

        # Compute sample covariance
        # Formula: cov = (X^T X) / (N-1)
        if batch_size > 1:
            cov = torch.matmul(features.T, features) / (batch_size - 1)
        else:
            # Handle edge case of batch size 1
            cov = torch.zeros(feature_dim, feature_dim, device=features.device)

        return cov

    def _compute_off_diagonal_loss(self, cov: torch.Tensor) -> torch.Tensor:
        """
        Compute the sum of squared off-diagonal elements of the covariance matrix.

        Args:
            cov: Covariance matrix [feature_dim, feature_dim]

        Returns:
            Sum of squared off-diagonal elements
        """
        # Create a mask for off-diagonal elements
        feature_dim = cov.shape[0]
        mask = 1.0 - torch.eye(feature_dim, device=cov.device)

        # Compute and sum the squared off-diagonal elements
        off_diag = cov * mask
        off_diag_sq = torch.pow(off_diag, 2)

        # Apply reduction
        if self.reduction == "mean":
            return off_diag_sq.sum() / (feature_dim * (feature_dim - 1))
        elif self.reduction == "sum":
            return off_diag_sq.sum()
        else:  # "none"
            return off_diag_sq

    def forward(
        self,
        vision_features: Optional[torch.Tensor] = None,
        text_features: Optional[torch.Tensor] = None,
        features: Optional[
            torch.Tensor
        ] = None,  # Generic features (if only one modality)
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Compute decorrelation loss to prevent feature collapse.

        The loss can be applied to vision features, text features, or both.

        Args:
            vision_features: Vision features [batch_size, vision_dim]
            text_features: Text features [batch_size, text_dim]
            features: Generic features if only one modality is used
            **kwargs: Additional arguments (ignored)

        Returns:
            Dictionary with loss values and additional metrics
        """
        # Track which modalities are being regularized
        regularize_vision = vision_features is not None
        regularize_text = text_features is not None
        regularize_generic = features is not None

        device = None
        if regularize_vision:
            device = vision_features.device
        elif regularize_text:
            device = text_features.device
        elif regularize_generic:
            device = features.device
        else:
            return {"loss": torch.tensor(0.0)}  # Nothing to regularize

        # Loss components
        vision_decor_loss = 0.0
        text_decor_loss = 0.0
        generic_decor_loss = 0.0
        feature_stats = {}

        # Process vision features if provided
        if regularize_vision:
            batch_size, vision_dim = vision_features.shape

            # Skip if batch size is too small
            if batch_size <= 1:
                logger.warning("Batch size too small for decorrelation loss (vision)")
                vision_decor_loss = torch.tensor(0.0, device=device)
            else:
                # Normalize vision features if requested
                if self.normalize_embeddings:
                    vision_features = F.normalize(vision_features, p=2, dim=1)

                # Compute vision covariance matrix
                vision_cov = self._compute_covariance(vision_features)

                # Compute decorrelation loss
                vision_decor_loss = self._compute_off_diagonal_loss(vision_cov)

                # Track feature statistics (for diagnosing feature collapse)
                if self.track_feature_stats:
                    # Compute variance of each dimension
                    vision_var = torch.var(vision_features, dim=0)
                    vision_min_var = vision_var.min().item()
                    vision_max_var = vision_var.max().item()
                    vision_mean_var = vision_var.mean().item()

                    # Check for low-variance dimensions (potential feature collapse)
                    vision_low_var_dims = (
                        (vision_var < self.min_variance_threshold).sum().item()
                    )
                    vision_low_var_pct = vision_low_var_dims / vision_dim

                    # Store statistics
                    feature_stats.update(
                        {
                            "vision_min_var": vision_min_var,
                            "vision_max_var": vision_max_var,
                            "vision_mean_var": vision_mean_var,
                            "vision_low_var_dims": vision_low_var_dims,
                            "vision_low_var_pct": vision_low_var_pct,
                        }
                    )

                    # Warning for potential feature collapse
                    if vision_low_var_pct > 0.5:
                        logger.warning(
                            f"VISION FEATURE COLLAPSE DETECTED! {vision_low_var_dims}/{vision_dim} "
                            f"dimensions have variance < {self.min_variance_threshold}"
                        )

        # Process text features if provided
        if regularize_text:
            batch_size, text_dim = text_features.shape

            # Skip if batch size is too small
            if batch_size <= 1:
                logger.warning("Batch size too small for decorrelation loss (text)")
                text_decor_loss = torch.tensor(0.0, device=device)
            else:
                # Normalize text features if requested
                if self.normalize_embeddings:
                    text_features = F.normalize(text_features, p=2, dim=1)

                # Compute text covariance matrix
                text_cov = self._compute_covariance(text_features)

                # Compute decorrelation loss
                text_decor_loss = self._compute_off_diagonal_loss(text_cov)

                # Track feature statistics
                if self.track_feature_stats:
                    # Compute variance of each dimension
                    text_var = torch.var(text_features, dim=0)
                    text_min_var = text_var.min().item()
                    text_max_var = text_var.max().item()
                    text_mean_var = text_var.mean().item()

                    # Check for low-variance dimensions
                    text_low_var_dims = (
                        (text_var < self.min_variance_threshold).sum().item()
                    )
                    text_low_var_pct = text_low_var_dims / text_dim

                    # Store statistics
                    feature_stats.update(
                        {
                            "text_min_var": text_min_var,
                            "text_max_var": text_max_var,
                            "text_mean_var": text_mean_var,
                            "text_low_var_dims": text_low_var_dims,
                            "text_low_var_pct": text_low_var_pct,
                        }
                    )

                    # Warning for potential feature collapse
                    if text_low_var_pct > 0.5:
                        logger.warning(
                            f"TEXT FEATURE COLLAPSE DETECTED! {text_low_var_dims}/{text_dim} "
                            f"dimensions have variance < {self.min_variance_threshold}"
                        )

        # Process generic features if provided
        if regularize_generic:
            batch_size, feature_dim = features.shape

            # Skip if batch size is too small
            if batch_size <= 1:
                logger.warning("Batch size too small for decorrelation loss (generic)")
                generic_decor_loss = torch.tensor(0.0, device=device)
            else:
                # Normalize features if requested
                if self.normalize_embeddings:
                    features = F.normalize(features, p=2, dim=1)

                # Compute covariance matrix
                features_cov = self._compute_covariance(features)

                # Compute decorrelation loss
                generic_decor_loss = self._compute_off_diagonal_loss(features_cov)

                # Track feature statistics
                if self.track_feature_stats:
                    # Compute variance of each dimension
                    feature_var = torch.var(features, dim=0)
                    feature_min_var = feature_var.min().item()
                    feature_max_var = feature_var.max().item()
                    feature_mean_var = feature_var.mean().item()

                    # Check for low-variance dimensions
                    feature_low_var_dims = (
                        (feature_var < self.min_variance_threshold).sum().item()
                    )
                    feature_low_var_pct = feature_low_var_dims / feature_dim

                    # Store statistics
                    feature_stats.update(
                        {
                            "feature_min_var": feature_min_var,
                            "feature_max_var": feature_max_var,
                            "feature_mean_var": feature_mean_var,
                            "feature_low_var_dims": feature_low_var_dims,
                            "feature_low_var_pct": feature_low_var_pct,
                        }
                    )

                    # Warning for potential feature collapse
                    if feature_low_var_pct > 0.5:
                        logger.warning(
                            f"FEATURE COLLAPSE DETECTED! {feature_low_var_dims}/{feature_dim} "
                            f"dimensions have variance < {self.min_variance_threshold}"
                        )

        # Combine losses based on which modalities are present
        total_loss = 0.0
        num_modalities = sum([regularize_vision, regularize_text, regularize_generic])

        if regularize_vision:
            total_loss = total_loss + vision_decor_loss
        if regularize_text:
            total_loss = total_loss + text_decor_loss
        if regularize_generic:
            total_loss = total_loss + generic_decor_loss

        # Average across modalities if more than one is present
        if num_modalities > 1:
            total_loss = total_loss / num_modalities

        # Apply coefficient
        total_loss = self.coef * total_loss

        # Return loss and metrics
        result = {
            "loss": total_loss,
            "decorrelation_loss": total_loss.item(),
        }

        # Add detailed components and feature stats
        if regularize_vision:
            result["vision_decorrelation_loss"] = (
                vision_decor_loss.item()
                if isinstance(vision_decor_loss, torch.Tensor)
                else vision_decor_loss
            )
        if regularize_text:
            result["text_decorrelation_loss"] = (
                text_decor_loss.item()
                if isinstance(text_decor_loss, torch.Tensor)
                else text_decor_loss
            )
        if regularize_generic:
            result["generic_decorrelation_loss"] = (
                generic_decor_loss.item()
                if isinstance(generic_decor_loss, torch.Tensor)
                else generic_decor_loss
            )

        # Add feature statistics if tracking is enabled
        if self.track_feature_stats:
            result["feature_stats"] = feature_stats

        return result


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
        "module_purpose": "Implements decorrelation regularization to prevent feature collapse in multimodal models",
        "key_classes": [
            {
                "name": "DecorrelationLoss",
                "purpose": "Prevents feature collapse by penalizing correlation between feature dimensions",
                "key_methods": [
                    {
                        "name": "forward",
                        "signature": "forward(self, vision_features: Optional[torch.Tensor] = None, text_features: Optional[torch.Tensor] = None, features: Optional[torch.Tensor] = None, **kwargs) -> Dict[str, Any]",
                        "brief_description": "Computes decorrelation loss for vision, text, or generic features",
                    },
                    {
                        "name": "_compute_covariance",
                        "signature": "_compute_covariance(self, features: torch.Tensor) -> torch.Tensor",
                        "brief_description": "Computes feature covariance matrix for regularization",
                    },
                    {
                        "name": "_compute_off_diagonal_loss",
                        "signature": "_compute_off_diagonal_loss(self, cov: torch.Tensor) -> torch.Tensor",
                        "brief_description": "Calculates loss from off-diagonal covariance elements",
                    },
                ],
                "inheritance": "nn.Module",
                "dependencies": [
                    "torch",
                    "torch.nn",
                    "torch.nn.functional",
                    "logging",
                    "math",
                ],
            }
        ],
        "external_dependencies": ["torch", "logging", "math"],
        "complexity_score": 6,  # Medium complexity due to covariance computation and feature statistics tracking
    }
