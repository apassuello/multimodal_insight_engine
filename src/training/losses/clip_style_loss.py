# src/training/losses/clip_style_loss.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Any, Union
import logging

logger = logging.getLogger(__name__)

"""MODULE: clip_style_loss.py
PURPOSE: Implements a CLIP-style contrastive loss for aligning visual and textual representations in a shared embedding space.

KEY COMPONENTS:
- CLIPStyleLoss: Main class implementing the bidirectional contrastive loss
  - Supports both in-batch negatives and pre-computed similarity matrices
  - Handles semantic matching through match_ids
  - Provides comprehensive metrics for evaluation
  - Includes label smoothing option
  - Calculates recall@K metrics for retrieval evaluation

DEPENDENCIES:
- PyTorch (torch, torch.nn, torch.nn.functional)
- Python standard library (logging)

SPECIAL NOTES:
- Designed for the second stage of progressive multimodal training
- Handles both position-based and semantic-based matching
- Includes extensive diagnostics to detect and report feature collapse
- Compatible with both small and large batch sizes
"""


class CLIPStyleLoss(nn.Module):
    """
    Implements a CLIP-style contrastive loss for multimodal learning.

    This loss aligns visual and textual representations in a shared embedding space
    by maximizing similarity between matching pairs while minimizing similarity
    between non-matching pairs.

    Key features:
    - Bidirectional loss computation (image→text, text→image)
    - Temperature parameter to scale similarity distributions
    - Support for both in-batch negatives and pre-computed similarity matrices
    - Extensive logging and diagnostics to help identify training issues
    - Metrics for tracking alignment quality and retrieval performance
    """

    def __init__(
        self,
        temperature: float = 0.07,
        use_hard_negatives: bool = False,
        hard_negative_weight: float = 0.5,
        reduction: str = "mean",
        label_smoothing: float = 0.0,
        cache_labels: bool = True,
    ):
        """
        Initialize the CLIP-style contrastive loss.

        Args:
            temperature: Scaling factor for similarity scores, controls distribution sharpness
            use_hard_negatives: Whether to apply hard negative mining during training
            hard_negative_weight: Weight for hard negative samples when enabled
            reduction: How to reduce the loss ("mean", "sum", "none")
            label_smoothing: Amount of label smoothing to apply (0.0 to 1.0)
            cache_labels: Whether to cache labels for large batch efficiency
        """
        super().__init__()
        self.temperature = temperature
        self.use_hard_negatives = use_hard_negatives
        self.hard_negative_weight = hard_negative_weight
        self.reduction = reduction
        self.label_smoothing = label_smoothing
        self.cache_labels = cache_labels

        # Cache for labels to avoid recomputing for the same batch size
        self._labels_cache = {}

    def forward(
        self,
        vision_features: torch.Tensor,
        text_features: torch.Tensor,
        match_ids: Optional[List[str]] = None,
        similarity_matrix: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Compute CLIP-style contrastive loss between vision and text features.

        Args:
            vision_features: Vision features [batch_size, embed_dim]
            text_features: Text features [batch_size, embed_dim]
            match_ids: IDs that determine which items should match
            similarity_matrix: Pre-computed similarity matrix (optional)
            **kwargs: Additional arguments (ignored)

        Returns:
            Dictionary with loss values and additional metrics
        """
        batch_size = vision_features.shape[0]

        # Safety check for empty batch
        if batch_size == 0:
            return {"loss": torch.tensor(0.0, device=vision_features.device)}

        # Safety check for dimension mismatch between modalities
        if batch_size != text_features.shape[0]:
            raise ValueError(
                f"Batch size mismatch: vision={vision_features.shape}, text={text_features.shape}"
            )

        # Normalize features for cosine similarity
        vision_features = F.normalize(vision_features, p=2, dim=1)
        text_features = F.normalize(text_features, p=2, dim=1)

        # Compute similarity matrix if not provided
        if similarity_matrix is None:
            # Formula: (vision_features @ text_features.T) / temperature
            similarity_matrix = (
                torch.matmul(vision_features, text_features.T) / self.temperature
            )

        # Create match matrix based on match_ids if provided
        # This determines which pairs are positive matches based on semantic IDs
        if match_ids is not None:
            # Convert match_ids to strings for consistent comparison
            if not isinstance(match_ids[0], str):
                string_match_ids = [str(mid) for mid in match_ids]
            else:
                string_match_ids = match_ids

            # Create match matrix - True where items should match semantically
            match_matrix = torch.zeros(
                (batch_size, batch_size),
                dtype=torch.bool,
                device=vision_features.device,
            )

            # Fill match matrix based on semantic match IDs
            for i in range(batch_size):
                for j in range(batch_size):
                    match_matrix[i, j] = string_match_ids[i] == string_match_ids[j]

            # Log match matrix statistics for diagnostics
            unique_matches = len(set(string_match_ids))
            positive_pairs = match_matrix.sum().item()

            # Warn if match_ids don't provide meaningful grouping
            if unique_matches == batch_size and batch_size > 1:
                logger.warning(
                    f"All match_ids are unique ({batch_size} IDs) - no semantic grouping possible!"
                )
            elif unique_matches == 1:
                logger.warning(
                    f"All match_ids are identical - treating all pairs as matches!"
                )
            else:
                # Calculate average semantic group size for logging
                avg_group_size = batch_size / max(1, unique_matches)
                logger.debug(
                    f"Found {unique_matches} semantic groups, avg size: {avg_group_size:.1f}, "
                    f"positive pairs: {positive_pairs}/{batch_size*batch_size}"
                )

            # Create targets for vision-to-text direction
            v2t_targets = torch.zeros(
                batch_size, dtype=torch.long, device=vision_features.device
            )

            # For each vision feature, assign a target text feature with matching match_id
            for i in range(batch_size):
                matches = torch.where(match_matrix[i])[0]
                if len(matches) > 0:
                    # If multiple matches, randomly select one
                    match_idx = torch.randint(
                        0, len(matches), (1,), device=matches.device
                    )[0]
                    v2t_targets[i] = matches[match_idx]
                else:
                    # Fallback if no matches (shouldn't happen with valid match_ids)
                    v2t_targets[i] = i

            # Create targets for text-to-vision direction (similar process)
            t2v_targets = torch.zeros(
                batch_size, dtype=torch.long, device=vision_features.device
            )

            for j in range(batch_size):
                matches = torch.where(match_matrix[:, j])[0]
                if len(matches) > 0:
                    match_idx = torch.randint(
                        0, len(matches), (1,), device=matches.device
                    )[0]
                    t2v_targets[j] = matches[match_idx]
                else:
                    t2v_targets[j] = j
        else:
            # Fallback to diagonal matching if no match_ids provided
            logger.warning(
                "No match_ids provided - using diagonal matching (index-based pairing)"
            )
            v2t_targets = torch.arange(batch_size, device=vision_features.device)
            t2v_targets = torch.arange(batch_size, device=vision_features.device)

        # Compute vision-to-text loss
        if self.label_smoothing > 0:
            # Create smooth labels for cross-entropy (when using label smoothing)
            v2t_smooth_targets = self._create_smooth_labels(
                batch_size, v2t_targets, self.label_smoothing
            )

            # Cross-entropy with smooth labels
            v2t_loss = -(
                v2t_smooth_targets * F.log_softmax(similarity_matrix, dim=1)
            ).sum(dim=1)
            if self.reduction == "mean":
                v2t_loss = v2t_loss.mean()
            elif self.reduction == "sum":
                v2t_loss = v2t_loss.sum()
        else:
            # Standard cross-entropy without label smoothing
            v2t_loss = F.cross_entropy(
                similarity_matrix, v2t_targets, reduction=self.reduction
            )

        # Compute text-to-vision loss (transposed similarity matrix)
        if self.label_smoothing > 0:
            t2v_smooth_targets = self._create_smooth_labels(
                batch_size, t2v_targets, self.label_smoothing
            )

            t2v_loss = -(
                t2v_smooth_targets * F.log_softmax(similarity_matrix.T, dim=1)
            ).sum(dim=1)
            if self.reduction == "mean":
                t2v_loss = t2v_loss.mean()
            elif self.reduction == "sum":
                t2v_loss = t2v_loss.sum()
        else:
            t2v_loss = F.cross_entropy(
                similarity_matrix.T, t2v_targets, reduction=self.reduction
            )

        # Average the bidirectional losses
        loss = (v2t_loss + t2v_loss) / 2.0

        # Calculate accuracy metrics
        with torch.no_grad():
            # Vision-to-text accuracy
            v2t_pred = torch.argmax(similarity_matrix, dim=1)
            v2t_accuracy = (v2t_pred == v2t_targets).float().mean()

            # Text-to-vision accuracy
            t2v_pred = torch.argmax(similarity_matrix.T, dim=1)
            t2v_accuracy = (t2v_pred == t2v_targets).float().mean()

            # Overall accuracy (average of both directions)
            accuracy = (v2t_accuracy + t2v_accuracy) / 2.0

            # Calculate similarity statistics for monitoring alignment quality
            # Mean similarity across all pairs (background level)
            mean_sim = similarity_matrix.mean().item()

            # Mean similarity for matched pairs (signal level)
            matched_indices = torch.stack(
                [torch.arange(batch_size, device=vision_features.device), v2t_targets]
            )
            matched_sim = (
                similarity_matrix[matched_indices[0], matched_indices[1]].mean().item()
            )

            # Signal-to-noise ratio (how much matched pairs stand out)
            sim_gap = matched_sim - mean_sim

            # Compute recall@K metrics
            recalls = self._compute_recall_at_k(
                similarity_matrix,
                K=[1, 5, 10],
                v2t_targets=v2t_targets,
                t2v_targets=t2v_targets,
            )

        # Return comprehensive metrics dictionary
        return {
            "loss": loss,
            "v2t_loss": v2t_loss.item() if torch.is_tensor(v2t_loss) else v2t_loss,
            "t2v_loss": t2v_loss.item() if torch.is_tensor(t2v_loss) else t2v_loss,
            "v2t_accuracy": v2t_accuracy.item(),
            "t2v_accuracy": t2v_accuracy.item(),
            "accuracy": accuracy.item(),
            "mean_similarity": mean_sim,
            "matched_similarity": matched_sim,
            "similarity_gap": sim_gap,
            "recalls": recalls,
        }

    def _create_smooth_labels(
        self, batch_size: int, targets: torch.Tensor, label_smoothing: float
    ) -> torch.Tensor:
        """
        Create smooth labels for cross-entropy loss with label smoothing.

        Args:
            batch_size: Size of the batch
            targets: Target indices [batch_size]
            label_smoothing: Label smoothing factor (0.0 to 1.0)

        Returns:
            Smooth label distribution [batch_size, batch_size]
        """
        # Use cache for efficiency if enabled and this batch size has been seen before
        if self.cache_labels and batch_size in self._labels_cache:
            device = targets.device
            return self._labels_cache[batch_size].to(device)

        # Create one-hot encoding
        smooth_targets = torch.zeros(batch_size, batch_size, device=targets.device)
        smooth_targets.scatter_(1, targets.unsqueeze(1), 1.0)

        # Apply label smoothing
        smooth_targets = (
            smooth_targets * (1.0 - label_smoothing) + label_smoothing / batch_size
        )

        # Cache for future use
        if self.cache_labels:
            self._labels_cache[batch_size] = smooth_targets.detach().cpu()

        return smooth_targets

    def _compute_recall_at_k(
        self,
        similarity: torch.Tensor,
        K: List[int] = [1, 5, 10],
        v2t_targets: Optional[torch.Tensor] = None,
        t2v_targets: Optional[torch.Tensor] = None,
    ) -> Dict[str, float]:
        """
        Compute recall@K metrics for image-text retrieval.

        Args:
            similarity: Similarity matrix [batch_size, batch_size]
            K: List of K values to compute recall for
            v2t_targets: Target indices for vision-to-text direction
            t2v_targets: Target indices for text-to-vision direction

        Returns:
            Dictionary with recall metrics
        """
        batch_size = similarity.shape[0]

        # Default targets (diagonal matching) if not provided
        if v2t_targets is None:
            v2t_targets = torch.arange(batch_size, device=similarity.device)
        if t2v_targets is None:
            t2v_targets = torch.arange(batch_size, device=similarity.device)

        # Compute recalls
        results = {}

        for k in K:
            k_adjusted = min(k, batch_size)

            # Image-to-text retrieval (using provided targets)
            top_k_v2t = torch.topk(similarity, k_adjusted, dim=1)[1]
            matches_v2t = torch.zeros(
                batch_size, dtype=torch.bool, device=similarity.device
            )
            for i in range(batch_size):
                matches_v2t[i] = (top_k_v2t[i] == v2t_targets[i]).any()
            recall_v2t = matches_v2t.float().mean().item()

            # Text-to-image retrieval (using provided targets)
            top_k_t2v = torch.topk(similarity.T, k_adjusted, dim=1)[1]
            matches_t2v = torch.zeros(
                batch_size, dtype=torch.bool, device=similarity.device
            )
            for j in range(batch_size):
                matches_t2v[j] = (top_k_t2v[j] == t2v_targets[j]).any()
            recall_t2v = matches_t2v.float().mean().item()

            # Average recall
            recall_avg = (recall_v2t + recall_t2v) / 2

            results[f"v2t_recall@{k}"] = recall_v2t
            results[f"t2v_recall@{k}"] = recall_t2v
            results[f"avg_recall@{k}"] = recall_avg

        return results


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
        "module_purpose": "Implements a CLIP-style contrastive loss for multimodal learning with bidirectional alignment",
        "key_classes": [
            {
                "name": "CLIPStyleLoss",
                "purpose": "Aligns visual and textual representations in a shared embedding space using bidirectional contrastive learning",
                "key_methods": [
                    {
                        "name": "forward",
                        "signature": "forward(self, vision_features: torch.Tensor, text_features: torch.Tensor, match_ids: Optional[List[str]] = None, similarity_matrix: Optional[torch.Tensor] = None, **kwargs) -> Dict[str, Any]",
                        "brief_description": "Computes bidirectional contrastive loss with comprehensive metrics",
                    },
                    {
                        "name": "_create_smooth_labels",
                        "signature": "_create_smooth_labels(self, batch_size: int, targets: torch.Tensor, label_smoothing: float) -> torch.Tensor",
                        "brief_description": "Creates smoothed label distributions for more stable training",
                    },
                    {
                        "name": "_compute_recall_at_k",
                        "signature": "_compute_recall_at_k(self, similarity: torch.Tensor, K: List[int] = [1, 5, 10], v2t_targets: Optional[torch.Tensor] = None, t2v_targets: Optional[torch.Tensor] = None) -> Dict[str, float]",
                        "brief_description": "Calculates recall@K metrics for retrieval evaluation",
                    },
                ],
                "inheritance": "nn.Module",
                "dependencies": ["torch", "torch.nn", "torch.nn.functional", "logging"],
            }
        ],
        "external_dependencies": ["torch", "logging"],
        "complexity_score": 7,  # Medium-high complexity due to comprehensive metrics and match ID handling
    }
