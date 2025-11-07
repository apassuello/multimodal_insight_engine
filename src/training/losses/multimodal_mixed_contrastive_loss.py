"""MODULE: multimodal_mixed_contrastive_loss.py
PURPOSE: Implements a mixed contrastive loss combining multiple contrastive learning objectives for multimodal data.

KEY COMPONENTS:
- MultiModalMixedContrastiveLoss: Main class implementing mixed contrastive loss
- Support for NT-Xent and supervised contrastive losses
- Optional projection layers for feature transformation
- Configurable loss weights for different components
- Performance metrics tracking (recall@K)

DEPENDENCIES:
- torch
- torch.nn
- typing
"""

import logging
import os
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


class MultiModalMixedContrastiveLoss(nn.Module):
    """
    Implements a mixed contrastive loss that combines multiple contrastive loss formulations
    for multimodal learning. This allows for more flexible and potentially more effective
    training by leveraging different loss formulations simultaneously.
    """

    def __init__(
        self,
        temperature: float = 0.07,
        loss_weights: Optional[Dict[str, float]] = None,
        reduction: str = "mean",
        add_projection: bool = False,
        projection_dim: int = 256,
        input_dim: Optional[int] = None,
    ):
        """
        Initialize the mixed contrastive loss module.

        Args:
            temperature: Temperature parameter controlling the sharpness of the distribution
            loss_weights: Dictionary mapping loss types to their weights
                         (e.g., {"infonce": 0.5, "nt_xent": 0.3, "supervised": 0.2})
            reduction: How to reduce the loss ("mean", "sum", or "none")
            add_projection: Whether to add MLP projection heads for embeddings
            projection_dim: Dimension of projection space (if add_projection is True)
            input_dim: Input dimension for projection heads (required if add_projection is True)
        """
        super().__init__()
        self.temperature = temperature
        self.reduction = reduction
        self.training = True  # Explicitly set training mode

        # Set default loss weights if not provided
        if loss_weights is None:
            loss_weights = {
                "infonce": 0.4,
                "nt_xent": 0.3,
                "supervised": 0.3,
            }
        self.loss_weights = loss_weights

        # Create projection heads if specified
        self.add_projection = add_projection
        if add_projection:
            assert (
                input_dim is not None
            ), "input_dim must be specified when add_projection=True"
            # Create projection heads and explicitly move to same device as buffers
            self.vision_projection = nn.Sequential(
                nn.Linear(input_dim, input_dim),
                nn.ReLU(),
                nn.Linear(input_dim, projection_dim),
            )
            self.text_projection = nn.Sequential(
                nn.Linear(input_dim, input_dim),
                nn.ReLU(),
                nn.Linear(input_dim, projection_dim),
            )

            # Ensure projection heads are on the same device as the rest of the model
            # This prevents device mismatch errors when using MPS or CUDA
            device = next(self.parameters()).device if self.parameters() else None
            if device:
                self.vision_projection = self.vision_projection.to(device)
                self.text_projection = self.text_projection.to(device)

    def project(
        self, vision_features: torch.Tensor, text_features: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply projection heads to features if enabled.

        Args:
            vision_features: Vision features [batch_size, vision_dim]
            text_features: Text features [batch_size, text_dim]

        Returns:
            Tuple of (projected_vision_features, projected_text_features)
        """
        if self.add_projection:
            # Make sure projection heads are on the same device as the input tensors
            device = vision_features.device
            if next(self.vision_projection.parameters()).device != device:
                self.vision_projection = self.vision_projection.to(device)
                self.text_projection = self.text_projection.to(device)

            # Apply projections
            vision_features = self.vision_projection(vision_features)
            text_features = self.text_projection(text_features)

        # Normalize features
        vision_features = F.normalize(vision_features, p=2, dim=1)
        text_features = F.normalize(text_features, p=2, dim=1)

        return vision_features, text_features

    def forward(
        self,
        vision_features: torch.Tensor,
        text_features: torch.Tensor,
        match_ids: Optional[List[str]] = None,
        labels: Optional[torch.Tensor] = None,  # For supervised contrastive loss
    ) -> Dict[str, Union[torch.Tensor, Dict[str, float]]]:
        """
        Compute mixed contrastive loss between vision and text features.

        Args:
            vision_features: Vision features [batch_size, vision_dim]
            text_features: Text features [batch_size, text_dim]
            match_ids: IDs that determine which items should match
            labels: Optional class labels for supervised contrastive loss

        Returns:
            Dictionary with loss values and additional metrics
        """
        # Get batch size and validate inputs
        batch_size = vision_features.shape[0]

        # Safety check for empty batch or mismatched dimensions
        if batch_size == 0:
            return {"loss": torch.tensor(0.0, device=vision_features.device)}

        if batch_size != text_features.shape[0]:
            raise ValueError(
                f"Batch size mismatch: vision={vision_features.shape}, text={text_features.shape}"
            )

        # Log feature shapes at debug level only
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(
                f"Vision features shape: {vision_features.shape}, device: {vision_features.device}"
            )
            logger.debug(
                f"Text features shape: {text_features.shape}, device: {text_features.device}"
            )
            logger.debug(f"Projection input_dim: {getattr(self, 'input_dim', None)}")

        # Apply projection and normalization
        vision_features, text_features = self.project(vision_features, text_features)

        # CRITICAL FIX: Properly handle match_ids for semantic matching
        # This part is essential for breaking positional correlations and ensuring meaningful learning
        if match_ids is not None:
            # Convert all match_ids to string format for consistent comparison
            if not isinstance(match_ids[0], str):
                string_match_ids = [str(mid) for mid in match_ids]
            else:
                string_match_ids = match_ids

            # Create match matrix - True where items should match semantically
            # match_matrix[i,j] is True if item i matches with item j
            match_matrix = torch.zeros(
                (batch_size, batch_size),
                dtype=torch.bool,
                device=vision_features.device,
            )

            # Fill match matrix based on match_ids (using string comparison)
            # This ensures that match IDs with the same semantic meaning are properly matched
            for i in range(batch_size):
                for j in range(batch_size):
                    match_matrix[i, j] = string_match_ids[i] == string_match_ids[j]

            # Enhanced debugging to troubleshoot match ID issues
            unique_matches = len(set(string_match_ids))
            match_id_counts = {}
            for mid in string_match_ids:
                match_id_counts[mid] = match_id_counts.get(mid, 0) + 1

            # Find match IDs that appear multiple times (these form semantic groups)
            semantic_groups = {
                mid: count for mid, count in match_id_counts.items() if count > 1
            }

            if unique_matches == batch_size and batch_size > 1:
                # If every item has a unique match ID, it means no semantic grouping
                # This would revert to diagonal matching which is problematic
                print(
                    "WARNING: All match_ids are unique - no semantic grouping possible!"
                )
                print("This will likely lead to poor training performance.")
                # Print first few match IDs to aid debugging
                print(f"Sample match_ids: {string_match_ids[:5]}")
            elif unique_matches == 1:
                # If all items have the same match ID, it's also problematic
                print(
                    "WARNING: All match_ids are identical - treating all pairs as matches!"
                )
                print("This will likely lead to poor training performance.")
            else:
                # Some semantic grouping exists - print statistics
                # print(f"Found {len(semantic_groups)} semantic groups with sizes: {list(semantic_groups.values())}")
                # Print match matrix statistics
                positives = match_matrix.sum().item()
                total = match_matrix.numel()
                # print(f"Match matrix has {positives}/{total} positive pairs ({positives/total*100:.2f}%)")

            # For each row (image), identify all valid matching columns (texts)
            v2t_targets = []
            for i in range(batch_size):
                # Get matching text indices for this image
                matches = torch.where(match_matrix[i])[0]

                if len(matches) == 0:
                    # Fallback if no matches (shouldn't happen)
                    v2t_targets.append(i)  # Default to same position
                else:
                    # Randomly select one of the matching texts
                    match_idx = torch.randint(
                        0, len(matches), (1,), device=matches.device
                    )[0]
                    v2t_targets.append(matches[match_idx].item())

            # Convert to tensor
            v2t_targets = torch.tensor(v2t_targets, device=vision_features.device)

            # For each column (text), identify all valid matching rows (images)
            t2i_targets = []
            for j in range(batch_size):
                # Get matching image indices for this text
                matches = torch.where(match_matrix[:, j])[0]

                if len(matches) == 0:
                    # Fallback if no matches (shouldn't happen)
                    t2i_targets.append(j)  # Default to same position
                else:
                    # Randomly select one of the matching images
                    match_idx = torch.randint(
                        0, len(matches), (1,), device=matches.device
                    )[0]
                    t2i_targets.append(matches[match_idx].item())

            # Convert to tensor
            t2i_targets = torch.tensor(t2i_targets, device=vision_features.device)

        else:
            # Fallback to traditional diagonal matching (position-based)
            print("WARNING: Using position-based matching in contrastive loss.")
            print("This is not recommended for meaningful multimodal learning.")
            v2t_targets = torch.arange(batch_size, device=vision_features.device)
            t2i_targets = torch.arange(batch_size, device=vision_features.device)

        # Compute similarity matrix
        similarity = torch.matmul(vision_features, text_features.T) / self.temperature

        # Add noise to the similarity matrix when batch size is small (< 16)
        if batch_size < 16 and self.training:
            # Scale noise based on batch size - smaller batches get more noise
            noise_scale = max(0.005, 0.02 * (16 - batch_size) / 16)
            similarity_noise = torch.randn_like(similarity) * noise_scale
            similarity = similarity + similarity_noise

        # Compute InfoNCE loss
        loss_v2t = F.cross_entropy(similarity, v2t_targets, reduction=self.reduction)
        loss_t2v = F.cross_entropy(similarity.T, t2i_targets, reduction=self.reduction)
        loss_infonce = (loss_v2t + loss_t2v) / 2

        # Compute NT-Xent loss
        loss_nt_xent = self._nt_xent_loss(vision_features, text_features)

        # Compute supervised contrastive loss if labels are provided
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

        # Calculate accuracy metrics
        with torch.no_grad():
            v2t_pred = torch.argmax(similarity, dim=1)
            t2v_pred = torch.argmax(similarity, dim=0)
            v2t_accuracy = (v2t_pred == v2t_targets).float().mean()
            t2v_accuracy = (t2v_pred == t2i_targets).float().mean()
            accuracy = (v2t_accuracy + t2v_accuracy) / 2

            # Calculate recall@K metrics
            recalls = self._compute_recall_at_k(
                similarity,
                K=[1, 5, 10],
                v2t_targets=v2t_targets,
                t2i_targets=t2i_targets,
            )

        return {
            "loss": total_loss,
            "loss_infonce": loss_infonce,
            "loss_nt_xent": loss_nt_xent,
            "loss_supervised": loss_supervised,
            "v2t_accuracy": v2t_accuracy,
            "t2v_accuracy": t2v_accuracy,
            "accuracy": accuracy,
            "recalls": recalls,
        }

    def _nt_xent_loss(
        self, vision_features: torch.Tensor, text_features: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute NT-Xent loss (Normalized Temperature-scaled Cross Entropy).

        Args:
            vision_features: Vision features [batch_size, vision_dim]
            text_features: Text features [batch_size, text_dim]

        Returns:
            NT-Xent loss
        """
        batch_size = vision_features.shape[0]

        # Concatenate features along the batch dimension to get 2*batch_size samples
        features = torch.cat([vision_features, text_features], dim=0)

        # Create labels where [0, 1, 2, ..., batch_size-1] maps to [batch_size, batch_size+1, ..., 2*batch_size-1]
        labels = torch.arange(batch_size, device=vision_features.device)
        labels = torch.cat([labels + batch_size, labels], dim=0)

        # Compute similarity matrix
        similarity = torch.matmul(features, features.T) / self.temperature

        # Mask out self-similarity
        mask = torch.eye(2 * batch_size, device=vision_features.device)
        mask = 1 - mask  # Invert to get non-diagonal elements

        # Apply mask (set self-similarities to large negative value)
        similarity = similarity * mask - 1e9 * (1 - mask)

        # Compute NT-Xent loss (each row contains one positive pair)
        loss = F.cross_entropy(similarity, labels, reduction=self.reduction)

        return loss

    def _supervised_contrastive_loss(
        self,
        vision_features: torch.Tensor,
        text_features: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute supervised contrastive loss using class labels.

        Args:
            vision_features: Vision features [batch_size, vision_dim]
            text_features: Text features [batch_size, text_dim]
            labels: Class labels [batch_size]

        Returns:
            Supervised contrastive loss
        """
        batch_size = vision_features.shape[0]

        # Concatenate features and labels
        features = torch.cat([vision_features, text_features], dim=0)
        labels_expanded = torch.cat([labels, labels], dim=0)

        # Create mask where 1 indicates samples from the same class
        labels_expanded = labels_expanded.contiguous().view(-1, 1)
        mask = torch.eq(labels_expanded, labels_expanded.T).float()

        # Remove self-similarities from mask
        identity_mask = torch.eye(2 * batch_size, device=vision_features.device)
        mask = mask - identity_mask

        # Compute similarity matrix
        similarity = torch.matmul(features, features.T) / self.temperature

        # For each anchor, compute loss against positive samples only
        exp_similarity = torch.exp(similarity)

        # Mask for denominators (all samples except self)
        non_self_mask = 1 - identity_mask

        # For numerical stability, compute log sum exp directly
        # Lower triangle matrix for row-wise / column-wise calculations
        pos_mask = mask * non_self_mask
        neg_mask = (1 - mask) * non_self_mask

        # Compute positive term (numerator) and full term (denominator)
        pos_term = torch.sum(similarity * pos_mask, dim=1)
        pos_count = torch.sum(pos_mask, dim=1)
        pos_count = torch.clamp(pos_count, min=1)  # Avoid division by zero
        pos_term = pos_term / pos_count

        # Compute log sum exp for denominator
        neg_exp = torch.exp(similarity) * non_self_mask
        per_sample_denom = torch.log(torch.sum(neg_exp, dim=1))

        # Compute final loss
        loss = -pos_term + per_sample_denom

        # Apply reduction
        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        else:
            return loss

    def _compute_recall_at_k(
        self,
        similarity: torch.Tensor,
        K: List[int] = [1, 5, 10],
        v2t_targets: Optional[torch.Tensor] = None,
        t2i_targets: Optional[torch.Tensor] = None,
    ) -> Dict[str, float]:
        """
        Compute recall@K metrics for image-text retrieval.

        Args:
            similarity: Similarity matrix [batch_size, batch_size]
            K: List of K values to compute recall for
            v2t_targets: Target indices for vision-to-text direction
            t2i_targets: Target indices for text-to-vision direction

        Returns:
            Dictionary with recall metrics
        """
        batch_size = similarity.shape[0]

        # Default targets (diagonal matching) if not provided
        if v2t_targets is None:
            v2t_targets = torch.arange(batch_size, device=similarity.device)
        if t2i_targets is None:
            t2i_targets = torch.arange(batch_size, device=similarity.device)

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
            top_k_t2i = torch.topk(similarity.t(), k_adjusted, dim=1)[1]
            matches_t2i = torch.zeros(
                batch_size, dtype=torch.bool, device=similarity.device
            )
            for j in range(batch_size):
                matches_t2i[j] = (top_k_t2i[j] == t2i_targets[j]).any()
            recall_t2i = matches_t2i.float().mean().item()

            # Average recall
            recall_avg = (recall_v2t + recall_t2i) / 2

            results[f"v2t_recall@{k}"] = recall_v2t
            results[f"t2i_recall@{k}"] = recall_t2i
            results[f"avg_recall@{k}"] = recall_avg

        return results

    def train(self, mode: bool = True):
        """
        Set the module in training mode.

        Args:
            mode: Whether to set training mode (True) or evaluation mode (False)

        Returns:
            self
        """
        self.training = mode
        if self.add_projection:
            self.vision_projection.train(mode)
            self.text_projection.train(mode)
        return self

    def eval(self):
        """
        Set the module in evaluation mode.

        Returns:
            self
        """
        return self.train(False)


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
        "module_purpose": "Implements a mixed contrastive loss combining multiple contrastive learning objectives for multimodal data",
        "key_classes": [
            {
                "name": "MultiModalMixedContrastiveLoss",
                "purpose": "Combines multiple contrastive learning objectives with configurable weights",
                "key_methods": [
                    {
                        "name": "__init__",
                        "signature": "__init__(self, temperature: float = 0.07, loss_weights: Optional[Dict[str, float]] = None)",
                        "brief_description": "Initialize mixed contrastive loss with temperature and weights",
                    },
                    {
                        "name": "forward",
                        "signature": "forward(self, vision_features: torch.Tensor, text_features: torch.Tensor, match_ids: Optional[List[str]] = None) -> Dict[str, torch.Tensor]",
                        "brief_description": "Compute mixed contrastive loss components",
                    },
                    {
                        "name": "_nt_xent_loss",
                        "signature": "_nt_xent_loss(self, vision_features: torch.Tensor, text_features: torch.Tensor) -> torch.Tensor",
                        "brief_description": "Compute NT-Xent contrastive loss",
                    },
                ],
                "inheritance": "nn.Module",
                "dependencies": ["torch", "torch.nn"],
            }
        ],
        "external_dependencies": ["torch", "typing"],
        "complexity_score": 8,
    }
