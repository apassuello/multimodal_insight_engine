# src/training/losses/supervised_contrastive_loss.py

import logging
from typing import Any, Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)

"""MODULE: supervised_contrastive_loss.py
PURPOSE: Implements supervised contrastive learning for multimodal models using explicit labels.

KEY COMPONENTS:
- SupervisedContrastiveLoss: Contrastive loss that leverages class labels for semantic grouping
  - Uses explicit labels to identify semantically related samples
  - Supports both classification labels and continuous similarity scores
  - Can operate in cross-modal, intra-modal, or combined modes
  - Handles class imbalance through sample weighting
  - Compatible with both single-label and multi-label classification tasks

DEPENDENCIES:
- PyTorch (torch, torch.nn, torch.nn.functional)
- Python standard library (logging)

SPECIAL NOTES:
- Most beneficial in Stage 3 of the progressive training approach
- More sample-efficient than unsupervised contrastive approaches
- Creates tighter, more coherent semantic clusters
- Provides additional metrics for interpretability and evaluation
"""


class SupervisedContrastiveLoss(nn.Module):
    """
    Implements supervised contrastive learning for multimodal models.

    Unlike traditional contrastive learning where only paired examples are considered
    positives, supervised contrastive learning leverages explicit labels to form
    positive pairs from samples of the same class. This creates tighter clusters of
    semantically similar examples and improves sample efficiency.

    Key features:
    - Uses explicit label information to identify semantically related samples
    - Supports both classification labels and continuous similarity scores
    - Can be applied within modalities, across modalities, or both
    - Configurable temperature parameter and margin for similarity thresholding
    - More sample-efficient than unsupervised contrastive approaches
    """

    def __init__(
        self,
        temperature: float = 0.07,
        contrast_mode: str = "all",  # "all", "cross", "intra"
        base_temperature: float = 0.07,
        similarity_threshold: float = 0.5,  # For continuous similarity scores
        use_class_weights: bool = False,
        reduction: str = "mean",
    ):
        """
        Initialize the supervised contrastive loss.

        Args:
            temperature: Temperature parameter for similarity scaling
            contrast_mode: How to apply contrast ("all", "cross", "intra")
                - "all": Both cross-modal and intra-modal contrasting
                - "cross": Only cross-modal contrasting
                - "intra": Only intra-modal contrasting
            base_temperature: Base temperature for normalization
            similarity_threshold: Threshold for binning continuous similarity scores
            use_class_weights: Whether to weight samples by inverse class frequency
            reduction: How to reduce the loss ("mean", "sum", or "none")
        """
        super().__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature
        self.similarity_threshold = similarity_threshold
        self.use_class_weights = use_class_weights
        self.reduction = reduction

    def forward(
        self,
        vision_features: torch.Tensor,
        text_features: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        similarity_scores: Optional[torch.Tensor] = None,
        class_weights: Optional[torch.Tensor] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Compute supervised contrastive loss based on class labels or similarity scores.

        Args:
            vision_features: Vision features [batch_size, vision_dim]
            text_features: Text features [batch_size, text_dim]
            labels: Class labels [batch_size] or [batch_size, num_classes] for multi-label
            similarity_scores: Pairwise similarity matrix [batch_size, batch_size]
            class_weights: Optional weights for each class [num_classes]
            **kwargs: Additional arguments (ignored)

        Returns:
            Dictionary with loss values and additional metrics
        """
        # Get batch size and device
        batch_size = vision_features.shape[0]
        device = vision_features.device

        # Normalize features for cosine similarity
        vision_features = F.normalize(vision_features, p=2, dim=1)
        text_features = F.normalize(text_features, p=2, dim=1)

        # Determine how to construct the mask for positive pairs
        if labels is not None:
            # Using class labels to determine positives

            # Handle multi-label case (one-hot encoded)
            if len(labels.shape) > 1 and labels.shape[1] > 1:
                # Convert multi-label format to pairwise similarity scores
                # Two samples are similar if they share any label
                similarity_scores = torch.matmul(labels.float(), labels.float().T)
                similarity_scores = (similarity_scores > 0).float()
            else:
                # Single-label case
                # Reshape to handle both [batch_size] and [batch_size, 1] formats
                labels = labels.contiguous().view(-1, 1)

                # Samples with the same label are positives
                mask = torch.eq(labels, labels.T).float()

        elif similarity_scores is not None:
            # Using provided similarity scores to determine positives
            # Apply threshold to convert to binary mask
            mask = (similarity_scores >= self.similarity_threshold).float()
        else:
            # Fallback to identity mask (only self-similarity)
            logger.warning(
                "No labels or similarity scores provided, "
                "using identity mask (diagonal positives only)"
            )
            mask = torch.eye(batch_size, device=device)

        # Remove self-contrast cases (diagonal)
        logits_mask = torch.ones_like(mask) - torch.eye(batch_size, device=device)
        mask = mask * logits_mask

        # Count positive examples for each anchor
        pos_per_sample = mask.sum(1)

        # Skip samples with no positive examples
        valid_samples = pos_per_sample > 0
        if not valid_samples.any():
            logger.warning("No samples with positive pairs found, returning zero loss")
            return {
                "loss": torch.tensor(0.0, device=device),
                "intra_vision_loss": torch.tensor(0.0, device=device),
                "intra_text_loss": torch.tensor(0.0, device=device),
                "cross_modal_loss": torch.tensor(0.0, device=device),
            }

        # Initialize loss components
        intra_vision_loss = torch.tensor(0.0, device=device)
        intra_text_loss = torch.tensor(0.0, device=device)
        cross_modal_loss = torch.tensor(0.0, device=device)

        # Compute intra-modal loss for vision features if needed
        if self.contrast_mode in ["all", "intra"]:
            # Compute similarity matrix within vision features
            vision_sim = (
                torch.matmul(vision_features, vision_features.T) / self.temperature
            )

            # For numerical stability, subtract max logit
            vision_logits_max, _ = torch.max(vision_sim, dim=1, keepdim=True)
            vision_sim = vision_sim - vision_logits_max.detach()

            # Compute log probabilities
            exp_vision_sim = torch.exp(vision_sim) * logits_mask
            log_prob_vision = vision_sim - torch.log(
                exp_vision_sim.sum(1, keepdim=True) + 1e-8
            )

            # Compute supervised contrastive loss
            # Formula: -sum(mask * log_prob) / sum(mask)
            mean_log_prob_vision = (mask * log_prob_vision).sum(1) / (
                pos_per_sample + 1e-8
            )

            # Apply class weights if requested
            if self.use_class_weights and class_weights is not None:
                if len(labels.shape) > 1 and labels.shape[1] > 1:
                    # For multi-label, average class weights across labels
                    sample_weights = (labels.float() * class_weights.unsqueeze(0)).sum(
                        1
                    ) / (labels.float().sum(1) + 1e-8)
                else:
                    # For single-label, use class weight directly
                    sample_weights = class_weights[labels.view(-1)]

                # Apply weights to loss
                mean_log_prob_vision = mean_log_prob_vision * sample_weights

            # Scale by temperature ratio and aggregate
            intra_vision_loss = (
                -(self.temperature / self.base_temperature) * mean_log_prob_vision
            )

            # Apply reduction
            if self.reduction == "mean":
                intra_vision_loss = intra_vision_loss[valid_samples].mean()
            elif self.reduction == "sum":
                intra_vision_loss = intra_vision_loss[valid_samples].sum()

        # Compute intra-modal loss for text features if needed
        if self.contrast_mode in ["all", "intra"]:
            # Compute similarity matrix within text features
            text_sim = torch.matmul(text_features, text_features.T) / self.temperature

            # For numerical stability, subtract max logit
            text_logits_max, _ = torch.max(text_sim, dim=1, keepdim=True)
            text_sim = text_sim - text_logits_max.detach()

            # Compute log probabilities
            exp_text_sim = torch.exp(text_sim) * logits_mask
            log_prob_text = text_sim - torch.log(
                exp_text_sim.sum(1, keepdim=True) + 1e-8
            )

            # Compute supervised contrastive loss
            mean_log_prob_text = (mask * log_prob_text).sum(1) / (pos_per_sample + 1e-8)

            # Apply class weights if requested
            if self.use_class_weights and class_weights is not None:
                if len(labels.shape) > 1 and labels.shape[1] > 1:
                    # For multi-label, average class weights across labels
                    sample_weights = (labels.float() * class_weights.unsqueeze(0)).sum(
                        1
                    ) / (labels.float().sum(1) + 1e-8)
                else:
                    # For single-label, use class weight directly
                    sample_weights = class_weights[labels.view(-1)]

                # Apply weights to loss
                mean_log_prob_text = mean_log_prob_text * sample_weights

            # Scale by temperature ratio and aggregate
            intra_text_loss = (
                -(self.temperature / self.base_temperature) * mean_log_prob_text
            )

            # Apply reduction
            if self.reduction == "mean":
                intra_text_loss = intra_text_loss[valid_samples].mean()
            elif self.reduction == "sum":
                intra_text_loss = intra_text_loss[valid_samples].sum()

        # Compute cross-modal loss if needed
        if self.contrast_mode in ["all", "cross"]:
            # Compute cross-modal similarity matrix
            cross_sim = (
                torch.matmul(vision_features, text_features.T) / self.temperature
            )

            # Compute vision-to-text loss
            # For numerical stability, subtract max logit
            v2t_logits_max, _ = torch.max(cross_sim, dim=1, keepdim=True)
            v2t_sim = cross_sim - v2t_logits_max.detach()

            # Compute log probabilities
            exp_v2t_sim = torch.exp(v2t_sim)
            log_prob_v2t = v2t_sim - torch.log(exp_v2t_sim.sum(1, keepdim=True) + 1e-8)

            # Compute supervised contrastive loss
            mean_log_prob_v2t = (mask * log_prob_v2t).sum(1) / (pos_per_sample + 1e-8)

            # Apply class weights if requested
            if self.use_class_weights and class_weights is not None:
                if len(labels.shape) > 1 and labels.shape[1] > 1:
                    sample_weights = (labels.float() * class_weights.unsqueeze(0)).sum(
                        1
                    ) / (labels.float().sum(1) + 1e-8)
                else:
                    sample_weights = class_weights[labels.view(-1)]

                mean_log_prob_v2t = mean_log_prob_v2t * sample_weights

            # Scale and negate
            v2t_loss = -(self.temperature / self.base_temperature) * mean_log_prob_v2t

            # Compute text-to-vision loss (similar process)
            t2v_sim = cross_sim.T  # Transpose to get text-to-vision perspective

            # For numerical stability, subtract max logit
            t2v_logits_max, _ = torch.max(t2v_sim, dim=1, keepdim=True)
            t2v_sim = t2v_sim - t2v_logits_max.detach()

            # Compute log probabilities
            exp_t2v_sim = torch.exp(t2v_sim)
            log_prob_t2v = t2v_sim - torch.log(exp_t2v_sim.sum(1, keepdim=True) + 1e-8)

            # Compute supervised contrastive loss
            mean_log_prob_t2v = (mask.T * log_prob_t2v).sum(1) / (pos_per_sample + 1e-8)

            # Apply class weights if requested
            if self.use_class_weights and class_weights is not None:
                if len(labels.shape) > 1 and labels.shape[1] > 1:
                    sample_weights = (labels.float() * class_weights.unsqueeze(0)).sum(
                        1
                    ) / (labels.float().sum(1) + 1e-8)
                else:
                    sample_weights = class_weights[labels.view(-1)]

                mean_log_prob_t2v = mean_log_prob_t2v * sample_weights

            # Scale and negate
            t2v_loss = -(self.temperature / self.base_temperature) * mean_log_prob_t2v

            # Apply reduction to both directions
            if self.reduction == "mean":
                v2t_loss = v2t_loss[valid_samples].mean()
                t2v_loss = t2v_loss[valid_samples].mean()
            elif self.reduction == "sum":
                v2t_loss = v2t_loss[valid_samples].sum()
                t2v_loss = t2v_loss[valid_samples].sum()

            # Average bidirectional losses
            cross_modal_loss = (v2t_loss + t2v_loss) / 2

        # Combine all loss components
        loss = 0.0
        num_components = 0

        if self.contrast_mode in ["all", "intra"]:
            loss += intra_vision_loss + intra_text_loss
            num_components += 2

        if self.contrast_mode in ["all", "cross"]:
            loss += cross_modal_loss
            num_components += 1

        # Average across components
        if num_components > 0:
            loss = loss / num_components

        # Compute contrastive accuracy (proportion of positive pairs with higher similarity than negatives)
        with torch.no_grad():
            # Vision-to-text accuracy
            v2t_sim = torch.matmul(vision_features, text_features.T)
            if labels is not None:
                pos_mask = torch.eq(labels, labels.T).float()
                neg_mask = 1.0 - pos_mask
            else:
                pos_mask = mask
                neg_mask = 1.0 - mask - torch.eye(batch_size, device=device)

            v2t_pos_sim = v2t_sim * pos_mask
            v2t_neg_sim = v2t_sim * neg_mask

            # For each anchor, compare its similarity with positives vs. negatives
            total_comparisons = 0
            total_correct = 0

            for i in range(batch_size):
                pos_indices = torch.nonzero(pos_mask[i], as_tuple=True)[0]
                neg_indices = torch.nonzero(neg_mask[i], as_tuple=True)[0]

                if len(pos_indices) > 0 and len(neg_indices) > 0:
                    pos_sims = v2t_sim[i, pos_indices]
                    neg_sims = v2t_sim[i, neg_indices]

                    # Count number of positive pairs with higher similarity than negatives
                    for pos_sim in pos_sims:
                        correct = (pos_sim > neg_sims).float().sum().item()
                        total_correct += correct
                        total_comparisons += len(neg_sims)

            if total_comparisons > 0:
                contrastive_accuracy = total_correct / total_comparisons
            else:
                contrastive_accuracy = 0.0

        # Return loss and metrics
        return {
            "loss": loss,
            "intra_vision_loss": (
                intra_vision_loss.item()
                if isinstance(intra_vision_loss, torch.Tensor)
                else intra_vision_loss
            ),
            "intra_text_loss": (
                intra_text_loss.item()
                if isinstance(intra_text_loss, torch.Tensor)
                else intra_text_loss
            ),
            "cross_modal_loss": (
                cross_modal_loss.item()
                if isinstance(cross_modal_loss, torch.Tensor)
                else cross_modal_loss
            ),
            "contrastive_accuracy": contrastive_accuracy,
        }


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
        "module_purpose": "Implements supervised contrastive learning for multimodal models using explicit labels",
        "key_classes": [
            {
                "name": "SupervisedContrastiveLoss",
                "purpose": "Improves contrastive learning by leveraging class labels to identify semantically related samples",
                "key_methods": [
                    {
                        "name": "forward",
                        "signature": "forward(self, vision_features: torch.Tensor, text_features: torch.Tensor, labels: Optional[torch.Tensor] = None, similarity_scores: Optional[torch.Tensor] = None, class_weights: Optional[torch.Tensor] = None, **kwargs) -> Dict[str, Any]",
                        "brief_description": "Computes supervised contrastive loss with support for class labels or similarity scores",
                    }
                ],
                "inheritance": "nn.Module",
                "dependencies": ["torch", "torch.nn", "torch.nn.functional", "logging"],
            }
        ],
        "external_dependencies": ["torch", "logging"],
        "complexity_score": 7,  # Medium-high complexity due to multiple contrast modes and class weighting
    }
