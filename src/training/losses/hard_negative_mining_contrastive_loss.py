"""MODULE: hard_negative_mining_contrastive_loss.py
PURPOSE: Implements contrastive loss with hard negative mining for more effective training.

KEY COMPONENTS:
- HardNegativeMiningContrastiveLoss: Main class for hard negative mining
- Online hard negative mining strategies
- Support for various mining criteria
- Efficient batch processing
- Mining statistics tracking

DEPENDENCIES:
- torch
- torch.nn
- typing
"""

import logging
import os

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


class HardNegativeMiningContrastiveLoss(nn.Module):
    def __init__(
        self,
        temperature=0.07,
        hard_negative_factor=2.0,
        mining_strategy="semi-hard",
        dim=768,
    ):
        super().__init__()
        self.temperature = temperature
        self.hard_negative_factor = hard_negative_factor  # Weight for hard negatives
        self.mining_strategy = mining_strategy  # "hard" or "semi-hard"
        self.dim = dim

        # Store dimension for debugging
        print(f"HardNegativeMiningContrastiveLoss initialized with dimension: {dim}")

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

        # Vision-to-text direction
        v2t_loss = 0.0
        for i in range(batch_size):
            # Identify positives
            pos_indices = torch.where(match_matrix[i])[0]
            if len(pos_indices) == 0:
                continue

            # Get positive similarities
            pos_sims = sim_matrix[i, pos_indices]
            # Calculate mean positive similarity
            mean_pos_sim = pos_sims.mean()

            # Get negative similarities
            neg_indices = torch.where(~match_matrix[i])[0]
            if len(neg_indices) == 0:
                continue

            neg_sims = sim_matrix[i, neg_indices]

            # CRITICAL ENHANCEMENT: Hard negative mining
            if self.mining_strategy == "hard":
                # Select the hardest negatives (highest similarity)
                num_hard = max(1, int(len(neg_indices) * 0.1))  # Use top 10% as hard
                hard_indices = torch.topk(neg_sims, num_hard)[1]
                hard_neg_sims = neg_sims[hard_indices]

                # Give more weight to hard negatives
                hard_weights = torch.ones_like(neg_sims)
                hard_weights[hard_indices] = self.hard_negative_factor

            elif self.mining_strategy == "semi-hard":
                # Semi-hard negative: closer than the positive but not too hard
                # Find negatives that are closer than the mean positive similarity
                semi_hard_mask = (neg_sims < mean_pos_sim) & (
                    neg_sims > mean_pos_sim - 0.2
                )

                if semi_hard_mask.sum() > 0:
                    hard_weights = torch.ones_like(neg_sims)
                    hard_weights[semi_hard_mask] = self.hard_negative_factor
                else:
                    # Fallback to regular weighting if no semi-hard negatives found
                    hard_weights = torch.ones_like(neg_sims)
            else:
                # No mining, equal weights
                hard_weights = torch.ones_like(neg_sims)

            # Weighted InfoNCE loss calculation
            pos_exp = torch.exp(pos_sims / self.temperature)
            weighted_neg_exp = torch.exp(neg_sims / self.temperature) * hard_weights

            # For each positive, compute loss
            for pos_idx in range(len(pos_sims)):
                pos_term = pos_exp[pos_idx]
                neg_term = weighted_neg_exp.sum()
                v2t_loss += -torch.log(pos_term / (pos_term + neg_term))

        # Implement text-to-vision direction explicitly
        t2v_loss = 0.0
        for i in range(batch_size):
            # Find positive images for this text
            pos_indices = torch.where(match_matrix[:, i])[0]
            if len(pos_indices) == 0:
                continue

            # Get similarities for this text with all images
            text_sims = sim_matrix[:, i]

            # Get positive and negative similarities
            pos_sims = text_sims[pos_indices]
            mean_pos_sim = pos_sims.mean()

            # Get negative indices and similarities
            neg_indices = torch.where(~match_matrix[:, i])[0]
            if len(neg_indices) == 0:
                continue

            neg_sims = text_sims[neg_indices]

            # Apply the same mining strategy as for vision-to-text
            if self.mining_strategy == "hard":
                # Get hard negatives (highest similarity non-matches)
                num_hard = max(1, int(len(neg_indices) * 0.1))
                hard_indices = torch.topk(neg_sims, num_hard)[1]

                # Weight hard negatives more heavily
                hard_weights = torch.ones_like(neg_sims)
                hard_weights[hard_indices] = self.hard_negative_factor

            elif self.mining_strategy == "semi-hard":
                # Get semi-hard negatives (close to but below positives)
                semi_hard_mask = (neg_sims < mean_pos_sim) & (
                    neg_sims > mean_pos_sim - 0.2
                )

                if semi_hard_mask.sum() > 0:
                    hard_weights = torch.ones_like(neg_sims)
                    hard_weights[semi_hard_mask] = self.hard_negative_factor
                else:
                    hard_weights = torch.ones_like(neg_sims)
            else:
                hard_weights = torch.ones_like(neg_sims)

            # Compute weighted loss
            pos_exp = torch.exp(pos_sims / self.temperature)
            weighted_neg_exp = torch.exp(neg_sims / self.temperature) * hard_weights

            # Calculate loss for each positive pair
            for pos_idx in range(len(pos_sims)):
                pos_term = pos_exp[pos_idx]
                neg_term = weighted_neg_exp.sum()
                t2v_loss += -torch.log(pos_term / (pos_term + neg_term))

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
            "v2t_loss": v2t_loss.item(),
            "t2v_loss": t2v_loss.item(),
            "hard_negative_factor": self.hard_negative_factor,
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
        "module_purpose": "Implements contrastive loss with hard negative mining for more effective training",
        "key_classes": [
            {
                "name": "HardNegativeMiningContrastiveLoss",
                "purpose": "Contrastive loss that mines hard negatives during training",
                "key_methods": [
                    {
                        "name": "__init__",
                        "signature": "__init__(self, temperature: float = 0.07, mining_ratio: float = 0.5)",
                        "brief_description": "Initialize loss with temperature and mining parameters",
                    },
                    {
                        "name": "forward",
                        "signature": "forward(self, features: torch.Tensor, labels: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]",
                        "brief_description": "Compute loss with mined hard negatives",
                    },
                    {
                        "name": "mine_hard_negatives",
                        "signature": "mine_hard_negatives(self, features: torch.Tensor, labels: torch.Tensor) -> torch.Tensor",
                        "brief_description": "Find hard negative examples in batch",
                    },
                ],
                "inheritance": "nn.Module",
                "dependencies": ["torch", "torch.nn"],
            }
        ],
        "external_dependencies": ["torch", "typing"],
        "complexity_score": 8,
    }
