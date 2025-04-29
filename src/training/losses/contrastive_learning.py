# src/training/contrastive_learning.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Any, Union
import numpy as np
import logging

logger = logging.getLogger(__name__)


def nt_xent_loss(
    vision_features: torch.Tensor,
    text_features: torch.Tensor,
    temperature: float = 0.07,
    reduction: str = "mean",
) -> torch.Tensor:
    """
    Compute NT-Xent loss (Normalized Temperature-scaled Cross Entropy).

    This is an alternative contrastive loss formulation used in SimCLR and similar frameworks.

    Args:
        vision_features: Vision features [batch_size, vision_dim]
        text_features: Text features [batch_size, text_dim]
        temperature: Temperature parameter
        reduction: Reduction method ("mean", "sum", or "none")

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
    similarity = torch.matmul(features, features.T) / temperature

    # Mask out self-similarity
    mask = torch.eye(2 * batch_size, device=vision_features.device)
    mask = 1 - mask  # Invert to get non-diagonal elements

    # Apply mask (set self-similarities to large negative value)
    similarity = similarity * mask - 1e9 * (1 - mask)

    # Compute NT-Xent loss (each row contains one positive pair)
    loss = F.cross_entropy(similarity, labels, reduction=reduction)

    return loss


def supervised_contrastive_loss(
    vision_features: torch.Tensor,
    text_features: torch.Tensor,
    labels: torch.Tensor,
    temperature: float = 0.07,
    reduction: str = "mean",
) -> torch.Tensor:
    """
    Compute supervised contrastive loss using class labels.

    This extends contrastive learning by using class labels to form positive pairs
    (samples from the same class are considered positives).

    Args:
        vision_features: Vision features [batch_size, vision_dim]
        text_features: Text features [batch_size, text_dim]
        labels: Class labels [batch_size]
        temperature: Temperature parameter
        reduction: Reduction method ("mean", "sum", or "none")

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
    similarity = torch.matmul(features, features.T) / temperature

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
    if reduction == "mean":
        return loss.mean()
    elif reduction == "sum":
        return loss.sum()
    else:
        return loss


def compute_recall_at_k(
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


class MultiModalMixedContrastiveLoss(nn.Module):
    """
    Advanced contrastive loss that combines multiple objectives.

    This module provides a flexible framework for combining contrastive loss
    with other objectives like classification and regression.
    """

    def __init__(
        self,
        contrastive_weight: float = 1.0,
        classification_weight: float = 0.0,
        multimodal_matching_weight: float = 0.0,
        temperature: float = 0.07,
        use_hard_negatives: bool = False,
        hard_negative_weight: float = 0.5,
        dim: int = 768,  # Default to modern vision transformer dimension
    ):
        """
        Initialize mixed contrastive loss.

        Args:
            contrastive_weight: Weight for the contrastive loss component
            classification_weight: Weight for the classification loss component
            multimodal_matching_weight: Weight for the multimodal matching component
            temperature: Temperature parameter for contrastive loss
            use_hard_negatives: Whether to use hard negatives mining
            hard_negative_weight: Weight for hard negatives (if used)
            dim: Feature dimension for the models (default: 768 for ViT-base)
        """
        super().__init__()
        self.contrastive_weight = contrastive_weight
        self.classification_weight = classification_weight
        self.multimodal_matching_weight = multimodal_matching_weight
        self.temperature = temperature
        self.use_hard_negatives = use_hard_negatives
        self.hard_negative_weight = hard_negative_weight
        self.dim = dim

        # Store dimension for debugging
        print(f"MultiModalMixedContrastiveLoss initialized with dimension: {dim}")

        # Base contrastive loss with correct input dimension
        self.contrastive_loss = ContrastiveLoss(
            temperature=temperature,
            add_projection=False,
            input_dim=dim,  # Pass the correct dimension
            projection_dim=min(
                512, dim // 2
            ),  # Smaller projection size for better generalization
        )

        # Classification loss
        self.classification_loss = nn.CrossEntropyLoss()

        # Multimodal matching loss (binary classification for match/non-match)
        self.matching_loss = nn.BCEWithLogitsLoss()

    def forward(
        self,
        vision_features: torch.Tensor,
        text_features: torch.Tensor,
        match_ids: Optional[List[str]] = None,
        class_logits: Optional[torch.Tensor] = None,
        class_labels: Optional[torch.Tensor] = None,
        matching_logits: Optional[torch.Tensor] = None,
        matching_labels: Optional[torch.Tensor] = None,
        hard_negatives: Optional[Dict[str, torch.Tensor]] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute mixed contrastive loss.

        Args:
            vision_features: Vision features [batch_size, vision_dim]
            text_features: Text features [batch_size, text_dim]
            match_ids: IDs that determine which items should match
            class_logits: Optional classification logits [batch_size, num_classes]
            class_labels: Optional class labels [batch_size]
            matching_logits: Optional matching logits [batch_size, 1]
            matching_labels: Optional matching labels [batch_size, 1]
            hard_negatives: Optional hard negatives features

        Returns:
            Dictionary with loss values and additional metrics
        """
        results = {}
        total_loss = 0.0

        # Contrastive loss
        if self.contrastive_weight > 0:
            contrastive_args = {
                "vision_features": vision_features,
                "text_features": text_features,
            }
            if match_ids is not None:
                contrastive_args["match_ids"] = match_ids

            contrastive_results = self.contrastive_loss(**contrastive_args)
            contrastive_loss = contrastive_results["loss"]

            # Add hard negatives if specified
            if self.use_hard_negatives and hard_negatives is not None:
                hard_vision_neg = hard_negatives.get("vision", None)
                hard_text_neg = hard_negatives.get("text", None)

                if hard_vision_neg is not None and hard_text_neg is not None:
                    # Create hard negative pairs
                    hard_results = self._compute_hard_negatives_loss(
                        vision_features, text_features, hard_vision_neg, hard_text_neg
                    )
                    contrastive_loss = (
                        1 - self.hard_negative_weight
                    ) * contrastive_loss + self.hard_negative_weight * hard_results[
                        "loss"
                    ]
                    results.update({f"hard_{k}": v for k, v in hard_results.items()})

            weighted_contrastive_loss = self.contrastive_weight * contrastive_loss
            total_loss += weighted_contrastive_loss

            results.update(contrastive_results)
            results["contrastive_loss"] = contrastive_loss.item()
            results["weighted_contrastive_loss"] = weighted_contrastive_loss.item()

        # Classification loss
        if (
            self.classification_weight > 0
            and class_logits is not None
            and class_labels is not None
        ):
            cls_loss = self.classification_loss(class_logits, class_labels)
            weighted_cls_loss = self.classification_weight * cls_loss
            total_loss += weighted_cls_loss

            # Compute accuracy
            with torch.no_grad():
                pred = torch.argmax(class_logits, dim=1)
                accuracy = (pred == class_labels).float().mean().item()

            results["classification_loss"] = cls_loss.item()
            results["weighted_classification_loss"] = weighted_cls_loss.item()
            results["classification_accuracy"] = accuracy

        # Multimodal matching loss
        if (
            self.multimodal_matching_weight > 0
            and matching_logits is not None
            and matching_labels is not None
        ):
            match_loss = self.matching_loss(matching_logits, matching_labels.float())
            weighted_match_loss = self.multimodal_matching_weight * match_loss
            total_loss += weighted_match_loss

            # Compute accuracy
            with torch.no_grad():
                pred = (torch.sigmoid(matching_logits) > 0.5).float()
                accuracy = (pred == matching_labels).float().mean().item()

            results["matching_loss"] = match_loss.item()
            results["weighted_matching_loss"] = weighted_match_loss.item()
            results["matching_accuracy"] = accuracy

        results["total_loss"] = total_loss

        return results

    def _compute_hard_negatives_loss(
        self,
        vision_features: torch.Tensor,
        text_features: torch.Tensor,
        hard_vision_neg: torch.Tensor,
        hard_text_neg: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute contrastive loss with hard negatives.

        Args:
            vision_features: Vision features [batch_size, vision_dim]
            text_features: Text features [batch_size, text_dim]
            hard_vision_neg: Hard negative vision features [batch_size, vision_dim]
            hard_text_neg: Hard negative text features [batch_size, text_dim]

        Returns:
            Dictionary with loss values and metrics for hard negatives
        """
        batch_size = vision_features.shape[0]

        # Normalize features
        vision_features = F.normalize(vision_features, p=2, dim=1)
        text_features = F.normalize(text_features, p=2, dim=1)
        hard_vision_neg = F.normalize(hard_vision_neg, p=2, dim=1)
        hard_text_neg = F.normalize(hard_text_neg, p=2, dim=1)

        # Create labels
        pos_labels = torch.arange(batch_size, device=vision_features.device)

        # Positive pairs similarity
        pos_similarity = (
            torch.matmul(vision_features, text_features.T) / self.temperature
        )

        # Hard negative similarities
        v2t_hard_sim = torch.matmul(vision_features, hard_text_neg.T) / self.temperature
        t2v_hard_sim = torch.matmul(text_features, hard_vision_neg.T) / self.temperature

        # Combined similarity matrices with hard negatives
        v2t_combined = torch.cat([pos_similarity, v2t_hard_sim], dim=1)
        t2v_combined = torch.cat([pos_similarity.T, t2v_hard_sim], dim=1)

        # Compute losses
        v2t_loss = F.cross_entropy(v2t_combined, pos_labels)
        t2v_loss = F.cross_entropy(t2v_combined, pos_labels)
        combined_loss = (v2t_loss + t2v_loss) / 2

        # Compute accuracy metrics
        with torch.no_grad():
            v2t_pred = torch.argmax(v2t_combined, dim=1)
            t2v_pred = torch.argmax(t2v_combined, dim=1)
            v2t_accuracy = (v2t_pred == pos_labels).float().mean()
            t2v_accuracy = (t2v_pred == pos_labels).float().mean()
            accuracy = (v2t_accuracy + t2v_accuracy) / 2

        return {
            "loss": combined_loss,
            "v2t_loss": v2t_loss.item(),
            "t2v_loss": t2v_loss.item(),
            "v2t_accuracy": v2t_accuracy.item(),
            "t2v_accuracy": t2v_accuracy.item(),
            "accuracy": accuracy.item(),
        }


class DecoupledContrastiveLoss(nn.Module):
    """
    Decoupled contrastive loss that separates instance discrimination from cross-modal matching.

    This loss function helps models learn both modality-specific features and cross-modal alignment.
    """

    def __init__(
        self,
        temperature: float = 0.07,
        lambda_v: float = 0.5,  # Weight for vision instance discrimination
        lambda_t: float = 0.5,  # Weight for text instance discrimination
        reduction: str = "mean",
    ):
        """
        Initialize the decoupled contrastive loss.

        Args:
            temperature: Temperature parameter for scaling similarity
            lambda_v: Weight for vision instance discrimination loss
            lambda_t: Weight for text instance discrimination loss
            reduction: Reduction method ('mean', 'sum', or 'none')
        """
        super().__init__()
        self.temperature = temperature
        self.lambda_v = lambda_v
        self.lambda_t = lambda_t
        self.reduction = reduction

    def forward(
        self,
        vision_features: torch.Tensor,
        text_features: torch.Tensor,
        match_ids: List[str],
    ) -> Dict[str, Any]:
        """
        Compute decoupled contrastive loss.

        Args:
            vision_features: Vision features [batch_size, vision_dim]
            text_features: Text features [batch_size, text_dim]
            match_ids: IDs that determine which items should match

        Returns:
            Dictionary with loss values and additional metrics
        """
        device = vision_features.device
        batch_size = vision_features.shape[0]

        # Normalize features
        vision_features = F.normalize(vision_features, p=2, dim=1)
        text_features = F.normalize(text_features, p=2, dim=1)

        # Create match matrix based on match_ids
        match_matrix = torch.zeros(
            (batch_size, batch_size), dtype=torch.bool, device=device
        )
        for i in range(batch_size):
            for j in range(batch_size):
                match_matrix[i, j] = match_ids[i] == match_ids[j]

        # Create mask to exclude self-matching for instance discrimination
        self_mask = torch.eye(batch_size, dtype=torch.bool, device=device)
        diag_mask = ~self_mask

        # ==== Cross-Modal Contrastive Loss ====
        # Compute all-pairs similarity
        similarity = torch.matmul(vision_features, text_features.T) / self.temperature

        # Vision-to-text direction
        v2t_loss = 0.0
        v2t_correct = 0

        for i in range(batch_size):
            # Find positive pairs for this vision feature
            pos_indices = torch.where(match_matrix[i])[0]
            if len(pos_indices) == 0:
                continue

            # Compute loss for each positive pair
            pos_logits = similarity[i, pos_indices]
            all_logits = similarity[i]

            # For each positive, compute InfoNCE loss
            for pos_idx in pos_indices:
                v2t_loss += -all_logits[pos_idx] + torch.logsumexp(all_logits, dim=0)

            # Track accuracy for metrics
            pred_idx = torch.argmax(similarity[i])
            if match_matrix[i, pred_idx]:
                v2t_correct += 1

        # Text-to-vision direction
        t2v_loss = 0.0
        t2v_correct = 0

        for i in range(batch_size):
            # Find positive pairs for this text feature
            pos_indices = torch.where(match_matrix[:, i])[0]
            if len(pos_indices) == 0:
                continue

            # Compute loss for each positive pair
            pos_logits = similarity[pos_indices, i]
            all_logits = similarity[:, i]

            # For each positive, compute InfoNCE loss
            for pos_idx in pos_indices:
                t2v_loss += -all_logits[pos_idx] + torch.logsumexp(all_logits, dim=0)

            # Track accuracy for metrics
            pred_idx = torch.argmax(similarity[:, i])
            if match_matrix[pred_idx, i]:
                t2v_correct += 1

        # ==== Instance Discrimination Losses ====
        # Vision-to-vision similarity
        vision_sim = torch.matmul(vision_features, vision_features.T) / self.temperature
        # Apply mask to exclude self-similarity
        vision_sim_masked = vision_sim.masked_fill(self_mask, -float("inf"))

        # Text-to-text similarity
        text_sim = torch.matmul(text_features, text_features.T) / self.temperature
        # Apply mask to exclude self-similarity
        text_sim_masked = text_sim.masked_fill(self_mask, -float("inf"))

        # Instance discrimination loss for vision
        vision_inst_loss = 0.0
        for i in range(batch_size):
            # Find positive examples (same match_id) but not self
            pos_indices = torch.where(match_matrix[i] & diag_mask[i])[0]
            if len(pos_indices) == 0:
                continue

            # Compute loss for each positive pair
            for pos_idx in pos_indices:
                vision_inst_loss += -vision_sim_masked[i, pos_idx] + torch.logsumexp(
                    vision_sim_masked[i], dim=0
                )

        # Instance discrimination loss for text
        text_inst_loss = 0.0
        for i in range(batch_size):
            # Find positive examples (same match_id) but not self
            pos_indices = torch.where(match_matrix[i] & diag_mask[i])[0]
            if len(pos_indices) == 0:
                continue

            # Compute loss for each positive pair
            for pos_idx in pos_indices:
                text_inst_loss += -text_sim_masked[i, pos_idx] + torch.logsumexp(
                    text_sim_masked[i], dim=0
                )

        # ==== Calculate Final Loss ====
        # Count number of positive pairs (excluding self-matches)
        num_pos_pairs = (match_matrix & diag_mask).sum().item()

        # Normalize losses
        if num_pos_pairs > 0:
            v2t_loss = v2t_loss / max(1, num_pos_pairs)
            t2v_loss = t2v_loss / max(1, num_pos_pairs)
            vision_inst_loss = vision_inst_loss / max(1, num_pos_pairs)
            text_inst_loss = text_inst_loss / max(1, num_pos_pairs)
        else:
            v2t_loss = torch.tensor(0.0, device=device)
            t2v_loss = torch.tensor(0.0, device=device)
            vision_inst_loss = torch.tensor(0.0, device=device)
            text_inst_loss = torch.tensor(0.0, device=device)

        # Calculate accuracies
        v2t_accuracy = v2t_correct / batch_size if batch_size > 0 else 0.0
        t2v_accuracy = t2v_correct / batch_size if batch_size > 0 else 0.0

        # Combine losses
        cross_modal_loss = (v2t_loss + t2v_loss) / 2
        instance_loss = (
            self.lambda_v * vision_inst_loss + self.lambda_t * text_inst_loss
        )

        total_loss = cross_modal_loss + instance_loss

        return {
            "loss": total_loss,
            "cross_modal_loss": cross_modal_loss.item(),
            "instance_loss": instance_loss.item(),
            "v2t_loss": v2t_loss.item(),
            "t2v_loss": t2v_loss.item(),
            "vision_inst_loss": vision_inst_loss.item(),
            "text_inst_loss": text_inst_loss.item(),
            "v2t_accuracy": v2t_accuracy,
            "t2v_accuracy": t2v_accuracy,
            "accuracy": (v2t_accuracy + t2v_accuracy) / 2,
        }
