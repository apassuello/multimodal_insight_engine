"""MODULE: decoupled_contrastive_loss.py
PURPOSE: Implements decoupled contrastive learning for multimodal models by separating vision-to-text and text-to-vision learning objectives.

KEY COMPONENTS:
- DecoupledContrastiveLoss: Main loss class that decouples vision-to-text and text-to-vision learning
- Instance discrimination within and across modalities
- Flexible weighting of different loss components

DEPENDENCIES:
- PyTorch (torch, torch.nn)
- Logging

SPECIAL NOTES:
- Provides better separation of concerns for vision and text modalities
- Implements fine-grained control over cross-modal and intra-modal learning
- Includes comprehensive metrics and similarity calculations
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Any, Union
import numpy as np
import logging
import os

logger = logging.getLogger(__name__)


class DecoupledContrastiveLoss(nn.Module):
    """
    Implements decoupled contrastive learning for multimodal learning.

    This approach separates the vision-to-text and text-to-vision learning objectives,
    allowing for more flexible and potentially more effective training by decoupling
    the two directions of contrastive learning.
    """

    def __init__(
        self,
        temperature: float = 0.07,
        lambda_v: float = 0.5,  # Weight for vision instance discrimination
        lambda_t: float = 0.5,  # Weight for text instance discrimination
        reduction: str = "mean",
    ):
        """
        Initialize the decoupled contrastive loss module.

        Args:
            temperature: Temperature parameter controlling the sharpness of the distribution
            lambda_v: Weight for vision instance discrimination loss
            lambda_t: Weight for text instance discrimination loss
            reduction: How to reduce the loss ("mean", "sum", or "none")
        """
        super().__init__()
        self.temperature = temperature
        self.lambda_v = lambda_v
        self.lambda_t = lambda_t
        self.reduction = reduction
        self.training = True  # Explicitly set training mode

    def forward(
        self,
        vision_features: torch.Tensor,
        text_features: torch.Tensor,
        match_ids: List[str],
    ) -> Dict[str, Any]:
        """
        Compute decoupled contrastive loss between vision and text features.

        Args:
            vision_features: Vision features [batch_size, vision_dim]
            text_features: Text features [batch_size, text_dim]
            match_ids: IDs that determine which items should match

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

        # Normalize features
        vision_features = F.normalize(vision_features, p=2, dim=1)
        text_features = F.normalize(text_features, p=2, dim=1)

        # Create match matrix based on match_ids
        match_matrix = torch.zeros(
            (batch_size, batch_size), dtype=torch.bool, device=vision_features.device
        )
        for i in range(batch_size):
            for j in range(batch_size):
                match_matrix[i, j] = match_ids[i] == match_ids[j]

        # Compute similarity matrices
        v2t_similarity = (
            torch.matmul(vision_features, text_features.T) / self.temperature
        )
        t2v_similarity = (
            torch.matmul(text_features, vision_features.T) / self.temperature
        )

        # Compute cross-modal loss (vision-to-text and text-to-vision)
        cross_modal_loss = 0.0
        v2t_loss = 0.0
        t2v_loss = 0.0

        # Vision-to-text direction
        for i in range(batch_size):
            # Get positive indices for this vision feature
            pos_indices = torch.where(match_matrix[i])[0]
            if len(pos_indices) == 0:
                continue

            # Get negative indices
            neg_indices = torch.where(~match_matrix[i])[0]
            if len(neg_indices) == 0:
                continue

            # Compute positive and negative logits
            pos_logits = v2t_similarity[i, pos_indices]
            neg_logits = v2t_similarity[i, neg_indices]

            # Compute InfoNCE loss for this vision feature
            pos_exp = torch.exp(pos_logits)
            neg_exp = torch.exp(neg_logits)
            v2t_loss += -torch.log(pos_exp.sum() / (pos_exp.sum() + neg_exp.sum()))

        # Text-to-vision direction
        for j in range(batch_size):
            # Get positive indices for this text feature
            pos_indices = torch.where(match_matrix[:, j])[0]
            if len(pos_indices) == 0:
                continue

            # Get negative indices
            neg_indices = torch.where(~match_matrix[:, j])[0]
            if len(neg_indices) == 0:
                continue

            # Compute positive and negative logits
            pos_logits = t2v_similarity[j, pos_indices]
            neg_logits = t2v_similarity[j, neg_indices]

            # Compute InfoNCE loss for this text feature
            pos_exp = torch.exp(pos_logits)
            neg_exp = torch.exp(neg_logits)
            t2v_loss += -torch.log(pos_exp.sum() / (pos_exp.sum() + neg_exp.sum()))

        # Normalize by number of positive pairs
        num_pos_pairs = match_matrix.sum().item()
        if num_pos_pairs > 0:
            v2t_loss = v2t_loss / num_pos_pairs
            t2v_loss = t2v_loss / num_pos_pairs
            cross_modal_loss = (v2t_loss + t2v_loss) / 2
        else:
            return {"loss": torch.tensor(0.0, device=vision_features.device)}

        # Compute instance discrimination loss (within modality)
        # Vision instance discrimination
        vision_similarity = (
            torch.matmul(vision_features, vision_features.T) / self.temperature
        )
        vision_inst_loss = 0.0

        for i in range(batch_size):
            # Get positive indices (same vision feature)
            pos_indices = torch.tensor([i], device=vision_features.device)

            # Get negative indices (different vision features)
            neg_indices = torch.cat(
                [
                    torch.arange(0, i, device=vision_features.device),
                    torch.arange(i + 1, batch_size, device=vision_features.device),
                ]
            )

            if len(neg_indices) == 0:
                continue

            # Compute positive and negative logits
            pos_logits = vision_similarity[i, pos_indices]
            neg_logits = vision_similarity[i, neg_indices]

            # Compute InfoNCE loss for this vision feature
            pos_exp = torch.exp(pos_logits)
            neg_exp = torch.exp(neg_logits)
            vision_inst_loss += -torch.log(
                pos_exp.sum() / (pos_exp.sum() + neg_exp.sum())
            )

        # Text instance discrimination
        text_similarity = (
            torch.matmul(text_features, text_features.T) / self.temperature
        )
        text_inst_loss = 0.0

        for j in range(batch_size):
            # Get positive indices (same text feature)
            pos_indices = torch.tensor([j], device=text_features.device)

            # Get negative indices (different text features)
            neg_indices = torch.cat(
                [
                    torch.arange(0, j, device=text_features.device),
                    torch.arange(j + 1, batch_size, device=text_features.device),
                ]
            )

            if len(neg_indices) == 0:
                continue

            # Compute positive and negative logits
            pos_logits = text_similarity[j, pos_indices]
            neg_logits = text_similarity[j, neg_indices]

            # Compute InfoNCE loss for this text feature
            pos_exp = torch.exp(pos_logits)
            neg_exp = torch.exp(neg_logits)
            text_inst_loss += -torch.log(
                pos_exp.sum() / (pos_exp.sum() + neg_exp.sum())
            )

        # Normalize instance discrimination losses
        if batch_size > 0:
            vision_inst_loss = vision_inst_loss / batch_size
            text_inst_loss = text_inst_loss / batch_size
            instance_loss = (vision_inst_loss + text_inst_loss) / 2
        else:
            instance_loss = torch.tensor(0.0, device=vision_features.device)

        # Combine losses with weights
        total_loss = (
            cross_modal_loss
            + self.lambda_v * vision_inst_loss
            + self.lambda_t * text_inst_loss
        )

        # Calculate accuracy metrics
        with torch.no_grad():
            v2t_pred = torch.argmax(v2t_similarity, dim=1)
            t2v_pred = torch.argmax(t2v_similarity, dim=1)

            # Create target tensors for accuracy calculation
            v2t_targets = torch.zeros(
                batch_size, dtype=torch.long, device=vision_features.device
            )
            t2v_targets = torch.zeros(
                batch_size, dtype=torch.long, device=vision_features.device
            )

            for i in range(batch_size):
                # For each vision feature, find the first matching text
                matches = torch.where(match_matrix[i])[0]
                if len(matches) > 0:
                    v2t_targets[i] = matches[0]
                else:
                    v2t_targets[i] = i  # Default to same position if no match

                # For each text feature, find the first matching vision
                matches = torch.where(match_matrix[:, i])[0]
                if len(matches) > 0:
                    t2v_targets[i] = matches[0]
                else:
                    t2v_targets[i] = i  # Default to same position if no match

            v2t_accuracy = (v2t_pred == v2t_targets).float().mean()
            t2v_accuracy = (t2v_pred == t2v_targets).float().mean()

        return {
            "loss": total_loss,
            "cross_modal_loss": cross_modal_loss,
            "instance_loss": instance_loss,
            "v2t_loss": v2t_loss,
            "t2v_loss": t2v_loss,
            "vision_inst_loss": vision_inst_loss,
            "text_inst_loss": text_inst_loss,
            "v2t_accuracy": v2t_accuracy,
            "t2v_accuracy": t2v_accuracy,
            "accuracy": (v2t_accuracy + t2v_accuracy) / 2,
        }

    def train(self, mode: bool = True):
        """
        Set the module in training mode.

        Args:
            mode: Whether to set training mode (True) or evaluation mode (False)

        Returns:
            self
        """
        self.training = mode
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
        "module_purpose": "Implements decoupled contrastive learning for multimodal models by separating vision-to-text and text-to-vision learning objectives",
        "key_classes": [
            {
                "name": "DecoupledContrastiveLoss",
                "purpose": "Implements a decoupled approach to contrastive learning with separated vision-to-text and text-to-vision objectives",
                "key_methods": [
                    {
                        "name": "__init__",
                        "signature": "__init__(self, temperature: float = 0.07, lambda_v: float = 0.5, lambda_t: float = 0.5, reduction: str = 'mean')",
                        "brief_description": "Initialize with temperature scaling and loss component weights",
                    },
                    {
                        "name": "forward",
                        "signature": "forward(self, vision_features: torch.Tensor, text_features: torch.Tensor, match_ids: List[str]) -> Dict[str, Any]",
                        "brief_description": "Compute decoupled contrastive loss with separated vision-to-text and text-to-vision components",
                    },
                    {
                        "name": "train",
                        "signature": "train(self, mode: bool = True)",
                        "brief_description": "Set the module to training mode",
                    },
                    {
                        "name": "eval",
                        "signature": "eval(self)",
                        "brief_description": "Set the module to evaluation mode",
                    },
                ],
                "inheritance": "nn.Module",
                "dependencies": ["torch", "torch.nn", "torch.nn.functional"],
            }
        ],
        "external_dependencies": ["torch", "numpy", "logging"],
        "complexity_score": 8,  # High complexity due to separate similarity matrices and bidirectional loss computation
    }
