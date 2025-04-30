"""MODULE: dynamic_temperature_contrastive_loss.py
PURPOSE: Implements contrastive loss with dynamic temperature scaling based on training progress.

KEY COMPONENTS:
- DynamicTemperatureContrastiveLoss: Main class for dynamic temperature contrastive loss
- Adaptive temperature scheduling
- Training progress tracking
- Performance monitoring
- Configurable temperature bounds

DEPENDENCIES:
- torch
- torch.nn
- typing
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Any, Union
import numpy as np
import logging

logger = logging.getLogger(__name__)


class DynamicTemperatureContrastiveLoss(nn.Module):
    def __init__(self, base_temperature=0.07, min_temp=0.04, max_temp=0.2, dim=768):
        super().__init__()
        self.base_temperature = base_temperature
        self.min_temp = min_temp
        self.max_temp = max_temp
        self.dim = dim

        # Store dimension for debugging
        print(f"DynamicTemperatureContrastiveLoss initialized with dimension: {dim}")

    def forward(self, vision_features, text_features, match_ids):
        device = vision_features.device

        # Normalize features
        vision_features = F.normalize(vision_features, dim=1)
        text_features = F.normalize(text_features, dim=1)

        # Compute similarity matrix
        sim_matrix = torch.matmul(vision_features, text_features.T)

        # Create match matrix based on match_ids
        batch_size = len(match_ids)
        match_matrix = torch.zeros(
            (batch_size, batch_size), dtype=torch.bool, device=device
        )
        for i in range(batch_size):
            for j in range(batch_size):
                match_matrix[i, j] = match_ids[i] == match_ids[j]

        # CRITICAL ENHANCEMENT: Dynamic temperature calculation
        # Get positive and negative similarities
        pos_mask = match_matrix.float()
        neg_mask = 1.0 - pos_mask
        pos_mask.fill_diagonal_(0)  # Remove diagonal from positives

        # Get distribution statistics of similarities
        pos_mean = (pos_mask * sim_matrix).sum() / max(1, pos_mask.sum().item())
        neg_mean = (neg_mask * sim_matrix).sum() / max(1, neg_mask.sum().item())

        # Calculate optimal temperature based on separation between positives and negatives
        separation = pos_mean - neg_mean
        # Lower temperature for better separated embeddings
        dynamic_temp = self.base_temperature * (
            0.8 + 0.4 * torch.exp(-2.0 * separation)
        )
        dynamic_temp = torch.clamp(dynamic_temp, self.min_temp, self.max_temp)

        # Use dynamic temperature for loss calculation
        # Vision-to-text loss
        v2t_loss = 0.0
        # Text-to-vision loss
        t2v_loss = 0.0

        # For each vision feature
        for i in range(batch_size):
            pos_indices = torch.where(match_matrix[i])[0]
            if len(pos_indices) == 0:
                continue

            # Compute positive logits
            pos_logits = sim_matrix[i, pos_indices] / dynamic_temp

            # Compute all logits (for normalization)
            all_logits = sim_matrix[i] / dynamic_temp

            # InfoNCE loss for each positive
            for pos_idx in pos_indices:
                v2t_loss += -all_logits[pos_idx] + torch.logsumexp(all_logits, dim=0)

        # Now explicitly implement the text-to-vision direction (no longer omitted)
        for i in range(batch_size):
            pos_indices = torch.where(match_matrix[:, i])[0]
            if len(pos_indices) == 0:
                continue

            # Compute all logits for this text (against all images)
            t2v_logits = sim_matrix[:, i] / dynamic_temp

            # For each positive pair, compute InfoNCE loss
            for pos_idx in pos_indices:
                t2v_loss += -t2v_logits[pos_idx] + torch.logsumexp(t2v_logits, dim=0)

        # Normalize by number of positive pairs
        num_pos_pairs = match_matrix.sum().item()
        if num_pos_pairs > 0:
            v2t_loss = v2t_loss / num_pos_pairs
            t2v_loss = t2v_loss / num_pos_pairs
        else:
            return {"loss": torch.tensor(0.0, device=device)}

        # Average bidirectional loss
        loss = (v2t_loss + t2v_loss) / 2

        return {
            "loss": loss,
            "temperature": dynamic_temp.item(),
            "pos_similarity": pos_mean.item(),
            "neg_similarity": neg_mean.item(),
            "v2t_loss": v2t_loss.item(),
            "t2v_loss": t2v_loss.item(),
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
        "module_purpose": "Implements contrastive loss with dynamic temperature scaling based on training progress",
        "key_classes": [
            {
                "name": "DynamicTemperatureContrastiveLoss",
                "purpose": "Contrastive loss with temperature that adapts during training",
                "key_methods": [
                    {
                        "name": "__init__",
                        "signature": "__init__(self, initial_temp: float = 0.07, min_temp: float = 0.01, max_temp: float = 0.2)",
                        "brief_description": "Initialize loss with temperature bounds",
                    },
                    {
                        "name": "forward",
                        "signature": "forward(self, features: torch.Tensor, labels: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]",
                        "brief_description": "Compute loss with current temperature",
                    },
                    {
                        "name": "update_temperature",
                        "signature": "update_temperature(self, metrics: Dict[str, float]) -> None",
                        "brief_description": "Update temperature based on training metrics",
                    },
                ],
                "inheritance": "nn.Module",
                "dependencies": ["torch", "torch.nn"],
            }
        ],
        "external_dependencies": ["torch", "typing"],
        "complexity_score": 7,
    }
