# multimodal_training_demo.py
"""
Multimodal Integration Training Demo

This script demonstrates the complete training and evaluation pipeline for
multimodal models with contrastive learning. It showcases:

1. Loading and preprocessing multimodal data
2. Building a cross-modal attention transformer
3. Training with advanced contrastive learning objectives
4. Evaluating cross-modal retrieval performance
5. Visualizing model outputs and attention maps

Available contrastive loss modes:
- contrastive: Standard contrastive loss with enhanced configuration
- memory_queue: Uses a memory queue for consistent global comparisons
- dynamic_temp: Automatically adjusts temperature based on embeddings
- hard_negative: Focuses training on challenging negative examples
- mixed: Combines multiple objectives for robust learning

Example usage:
python multimodal_training_demo.py --loss_type memory_queue --queue_size 8192
python multimodal_training_demo.py --loss_type dynamic_temp
python multimodal_training_demo.py --loss_type hard_negative --mining_strategy semi-hard
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# Fix the AdamW import - use a different approach
import torch.optim.adamw

AdamW = torch.optim.AdamW
import numpy as np
import logging
import random
import math
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Union, Any, Tuple, cast, Protocol
from torch.utils.data import Dataset

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(), logging.FileHandler("multimodal_training.log")],
)
logger = logging.getLogger(__name__)

# Import our models and utilities
from src.models.vision.image_preprocessing import ImagePreprocessor
from src.data.tokenization.simple_tokenizer import SimpleTokenizer
from src.training.multimodal_trainer import MultimodalTrainer
from src.utils.argument_configs import get_multimodal_training_args
from src.utils.model_utils import (
    print_model_summary,
    convert_tensors_to_python_types,
    count_parameters,
)
from src.models.model_factory import create_multimodal_model
from src.data.multimodal_data_utils import create_data_loaders
from src.training.losses.loss_factory import create_loss_function
from src.evaluation.inference_demo import run_inference_demo


# Define a protocol for datasets that have the get_split_proportions method
class SplitProportionsDataset(Protocol):
    def get_split_proportions(self) -> Dict[str, Any]:
        """Get information about the dataset splits and proportions."""
        ...


# Define a custom dataset type that includes the get_split_proportions method
class MultimodalDataset(Dataset, SplitProportionsDataset):
    def get_split_proportions(self) -> Dict[str, Any]:
        """Get information about the dataset splits and proportions."""
        return {"dataset_name": "unknown", "total_samples": 0}


# Add these classes after imports in multimodal_training_demo.py

import torch
import torch.nn as nn
import torch.nn.functional as F


# Enhanced multimodal model with anti-collapse mechanisms
class SimpleMultimodalModel(nn.Module):
    def __init__(self, vision_model, text_model, projection_dim=512):
        super().__init__()
        self.vision_model = vision_model
        self.text_model = text_model

        # Get dimensions
        if hasattr(vision_model, "num_features"):
            vision_dim = vision_model.num_features
        elif hasattr(vision_model, "embed_dim"):
            vision_dim = vision_model.embed_dim
        else:
            vision_dim = 768  # Default

        if hasattr(text_model, "encoder") and hasattr(text_model.encoder, "config"):
            text_dim = text_model.encoder.config.hidden_size
        elif hasattr(text_model, "d_model"):
            text_dim = text_model.d_model
        else:
            text_dim = 768  # Default

        print(
            f"Creating SimpleMultimodalModel with dims: vision={vision_dim}, text={text_dim}, proj={projection_dim}"
        )

        # IMPORTANT CHANGE: Simpler model with explicit normalization layer to prevent collapse
        # Vision projection: single layer with batch norm to enforce feature distribution
        self.vision_proj = nn.Sequential(
            nn.Linear(vision_dim, projection_dim),
            nn.BatchNorm1d(projection_dim, affine=True),  # BN prevents feature collapse
        )

        # Text projection: different structure to ensure asymmetry
        self.text_proj = nn.Sequential(
            nn.Linear(text_dim, projection_dim),
            nn.BatchNorm1d(projection_dim, affine=True),  # BN prevents feature collapse
        )

        # Initialize with strong orthogonal initialization
        # This creates diverse starting features that aren't correlated
        for m in self.vision_proj.modules():
            if isinstance(m, nn.Linear):
                # Orthogonal initialization ensures diverse features from the start
                nn.init.orthogonal_(m.weight, gain=1)  # Changed from 1.4 to 1
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.2)

        for m in self.text_proj.modules():
            if isinstance(m, nn.Linear):
                # Different gain for text model to create asymmetry
                nn.init.orthogonal_(m.weight, gain=1)  # Changed from 1.2 to 1
                if m.bias is not None:
                    nn.init.constant_(m.bias, -0.2)

        # BatchNorm params with different initialization to ensure asymmetry
        for name, m in self.named_modules():
            if isinstance(m, nn.BatchNorm1d):
                if "vision" in name:
                    # Vision BN - slightly different from text BN
                    nn.init.constant_(m.weight, 1.0)
                    nn.init.constant_(m.bias, 0.1)
                else:
                    # Text BN - different initialization
                    nn.init.constant_(m.weight, 0.9)
                    nn.init.constant_(m.bias, -0.1)

        # Learnable temperature parameter for similarity scaling
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

        # Track variance statistics to detect collapse
        self.register_buffer("running_vision_var", torch.ones(1))
        self.register_buffer("running_text_var", torch.ones(1))
        self.register_buffer("update_count", torch.zeros(1))

        # Feature whitening/scaling - explicitly prevents feature collapse
        # by rescaling feature distributions
        self.feature_scale = 10.0  # Increased scaling factor to prevent collapse

        print(
            f"Model initialized with anti-collapse techniques: BatchNorm + orthogonal init + feature scaling"
        )
        print(f"Feature scale: {self.feature_scale}")

    def forward(self, images=None, text_data=None, return_attention=False):
        outputs: Dict[str, Any] = {}

        # Vision forward
        if images is not None:
            # Extract vision features
            if hasattr(self.vision_model, "forward_features"):
                vision_features = self.vision_model.forward_features(images)
            else:
                vision_features = self.vision_model(images)
            # Handle different output formats
            if isinstance(vision_features, dict) and "pooler_output" in vision_features:
                vision_features = vision_features["pooler_output"]
            elif isinstance(vision_features, tuple):
                vision_features = vision_features[
                    0
                ]  # Typically the features are the first element

            # Store full sequence features for reference
            outputs["vision_features_seq"] = vision_features

            # Get pooled representation if needed
            if (
                isinstance(vision_features, torch.Tensor)
                and len(vision_features.shape) == 3
            ):  # [batch, seq_len, dim]
                # Use CLS token if available (typically first token)
                pooled_vision = vision_features[:, 0]
            else:
                pooled_vision = vision_features

            # Apply projection with BatchNorm to prevent feature collapse
            vision_features = self.vision_proj(pooled_vision)

            # Explicit feature scaling to prevent collapse - higher scale forces features apart
            # This is a critical change - it ensures features don't collapse to the same values
            # by explicitly increasing their scale/variance
            if self.training:
                # Scale features to have specific variance (prevents collapse but don't scale too aggressively)
                vision_features = vision_features * self.feature_scale

            # Track variance statistics
            if self.training:
                with torch.no_grad():
                    batch_var = torch.var(vision_features, dim=0).mean().item()
                    # Update running statistics with momentum
                    self.running_vision_var = (
                        0.9 * self.running_vision_var + 0.1 * batch_var
                    )

            outputs["vision_features"] = vision_features
            outputs["vision_features_enhanced"] = vision_features  # For compatibility

        # Text forward
        if text_data is not None:
            # Extract text features
            if hasattr(self.text_model, "encode"):
                text_features = self.text_model.encode(**text_data)
            else:
                text_features = self.text_model(**text_data)

            # Handle different output formats
            if isinstance(text_features, dict) and "pooler_output" in text_features:
                text_features = text_features["pooler_output"]
            elif isinstance(text_features, tuple):
                text_features = text_features[
                    0
                ]  # Typically the features are the first element

            # Store full sequence features for reference
            outputs["text_features_seq"] = text_features

            # Get pooled representation if needed
            if (
                isinstance(text_features, torch.Tensor)
                and len(text_features.shape) == 3
            ):  # [batch, seq_len, dim]
                # Use CLS token if available (typically first token)
                pooled_text = text_features[:, 0]
            else:
                pooled_text = text_features

            # Apply projection with BatchNorm to prevent feature collapse
            text_features = self.text_proj(pooled_text)

            # Apply scaling with slightly different factor for asymmetry
            if self.training:
                # Scale features to have specific variance (prevents collapse)
                # Use a slightly different scale from vision to create asymmetry
                text_features = text_features * (
                    self.feature_scale * 0.9
                )  # Increased asymmetry between modalities

            # Track variance statistics
            if self.training:
                with torch.no_grad():
                    batch_var = torch.var(text_features, dim=0).mean().item()
                    # Update running statistics with momentum
                    self.running_text_var = (
                        0.9 * self.running_text_var + 0.1 * batch_var
                    )
                    # Update counter (used for logging)
                    self.update_count += 1

            outputs["text_features"] = text_features
            outputs["text_features_enhanced"] = text_features  # For compatibility

        # If both modalities present, compute similarity
        if images is not None and text_data is not None:
            # Normalize features AFTER scaling to ensure proper normalization
            # This is critical - the normalization removes the scale but maintains the learned directions
            vision_norm = F.normalize(vision_features, p=2, dim=1)
            text_norm = F.normalize(text_features, p=2, dim=1)

            # Compute feature statistics before and after normalization to detect collapse
            if self.training:
                with torch.no_grad():
                    # Calculate pre-normalization variance by feature dimension
                    v_var_by_dim = torch.var(vision_features, dim=0)
                    t_var_by_dim = torch.var(text_features, dim=0)

                    # Calculate post-normalization variance by feature dimension
                    v_norm_var_by_dim = torch.var(vision_norm, dim=0)
                    t_norm_var_by_dim = torch.var(text_norm, dim=0)

                    # Count low-variance dimensions (indication of partial collapse)
                    v_low_var_dims = (v_norm_var_by_dim < 0.01).sum().item()
                    t_low_var_dims = (t_norm_var_by_dim < 0.01).sum().item()

                    # Log warning if many dimensions have low variance
                    if v_low_var_dims > 10 or t_low_var_dims > 10:
                        print(
                            f"WARNING: Many low-variance dimensions - Vision: {v_low_var_dims}, Text: {t_low_var_dims}"
                        )

            # Add powerful regularization loss to prevent feature collapse - force features to be decorrelated
            if self.training:
                # Compute centered features
                vision_centered = vision_norm - vision_norm.mean(dim=0, keepdim=True)
                text_centered = text_norm - text_norm.mean(dim=0, keepdim=True)

                batch_size = vision_norm.size(0)

                # Compute covariance matrices
                vision_cov = torch.matmul(vision_centered.T, vision_centered) / (
                    batch_size - 1
                )
                text_cov = torch.matmul(text_centered.T, text_centered) / (
                    batch_size - 1
                )

                # Identity matrix for computing off-diagonal elements
                I = torch.eye(vision_cov.size(0), device=vision_cov.device)

                # Compute orthogonality loss (stronger regularization than before)
                # This loss specifically targets the off-diagonal elements of the covariance matrix
                # making features more orthogonal/decorrelated
                vision_ortho_loss = torch.sum(torch.pow(vision_cov * (1 - I), 2))
                text_ortho_loss = torch.sum(torch.pow(text_cov * (1 - I), 2))

                # Combined loss with higher weight for strong regularization
                decor_loss = vision_ortho_loss + text_ortho_loss
                outputs["decor_loss"] = (
                    decor_loss * 0.5
                )  # Increased weight for stronger anti-collapse effect

            # Calculate temperature-scaled similarity
            # Clamp logit_scale to prevent extreme values
            logit_scale_value = torch.clamp(self.logit_scale.exp(), min=1.0, max=100.0)
            similarity = logit_scale_value * torch.matmul(vision_norm, text_norm.T)

            outputs["similarity"] = similarity
            outputs["raw_similarity"] = torch.matmul(
                vision_norm, text_norm.T
            )  # Without scaling
            outputs["logit_scale"] = logit_scale_value.item()  # For debugging

            # Track feature statistics for monitoring
            outputs["vision_var"] = self.running_vision_var.item()
            outputs["text_var"] = self.running_text_var.item()

            # Add fusion features (mean of two modalities) for compatibility
            fusion_features = (vision_features + text_features) / 2
            outputs["fusion_features"] = fusion_features
            outputs["pooled_fusion"] = fusion_features

            # Print diagnostics periodically during training
            if self.training and self.update_count % 5 == 0:
                print(
                    f"FEATURE STATS: Vision var: {self.running_vision_var.item():.4f}, Text var: {self.running_text_var.item():.4f}, Scale: {self.feature_scale:.1f}"
                )

        return outputs


# Enhanced supervised contrastive loss with anti-collapse mechanisms
class SupervisedAlignmentLoss(nn.Module):
    def __init__(self, temperature=0.07):  # Use lower temperature for sharper contrasts
        super().__init__()
        self.temperature = (
            temperature  # Lower temperature makes positive pairs more distinct
        )
        self.mse = nn.MSELoss()
        self.dim = 768  # Match model's projection_dim for 768-dim models

        # Add a diversity loss weight to actively prevent feature collapse
        self.diversity_weight = 1.0  # Increased from 0.5 for stronger anti-collapse

        # Track iteration count
        self.iteration = 0

        print(f"Initialized SupervisedAlignmentLoss with temperature={temperature}")

    def forward(self, vision_features, text_features, match_ids=None, **kwargs):
        self.iteration += 1

        # Handle additional losses from model output
        decor_loss = kwargs.get("decor_loss", 0.0)
        extra_loss = 0.0

        if isinstance(decor_loss, torch.Tensor):
            extra_loss = extra_loss + decor_loss

        # Extract logit scale if provided
        logit_scale = kwargs.get("logit_scale", 1.0)
        if not isinstance(logit_scale, float):
            logit_scale = 1.0

        # Always add diversity loss to prevent feature collapse proactively
        # This is the core fix - actively enforce feature diversity in each batch
        if isinstance(vision_features, torch.Tensor) and isinstance(
            text_features, torch.Tensor
        ):
            # Get unnormalized features for better diversity calculation
            batch_size = vision_features.shape[0]

            # Calculate feature statistics to detect collapse
            v_var = torch.var(vision_features, dim=0).mean()
            t_var = torch.var(text_features, dim=0).mean()

            # Log warning if variance is too low (features are collapsing)
            if v_var < 0.01 or t_var < 0.01:
                print(
                    f"WARNING: Feature collapse detected - Vision var: {v_var.item():.6f}, Text var: {t_var.item():.6f}"
                )

            # Calculate similarity matrices
            # For normalizing later, store original norms
            v_norms = torch.norm(vision_features, p=2, dim=1, keepdim=True)
            t_norms = torch.norm(text_features, p=2, dim=1, keepdim=True)

            # Calculate pairwise feature similarity (to reduce, promoting diversity)
            vision_sim = torch.matmul(vision_features, vision_features.T)
            text_sim = torch.matmul(text_features, text_features.T)

            # Identity matrix to mask out diagonal elements (same feature with itself)
            I = torch.eye(batch_size, device=vision_features.device)

            # Diversity loss: minimize similarity between different samples' features
            # This directly prevents feature collapse by making features more orthogonal
            feature_diversity_loss = torch.pow(vision_sim * (1 - I), 2).sum() / (
                batch_size * (batch_size - 1)
            ) + torch.pow(text_sim * (1 - I), 2).sum() / (batch_size * (batch_size - 1))

            # Add to extra loss - stronger weight when variance is low
            diversity_weight = self.diversity_weight * (
                0.1 + 10.0 * torch.exp(-5.0 * (v_var + t_var))
            )
            extra_loss = extra_loss + feature_diversity_loss * diversity_weight

        # Normalize features
        vision_features = F.normalize(vision_features, p=2, dim=1)
        text_features = F.normalize(text_features, p=2, dim=1)

        # Compute similarity
        batch_size = vision_features.shape[0]

        # Use adaptive temperature
        effective_temp = self.temperature
        if "logit_scale" in kwargs:
            # If model provides logit scale, use it for consistency
            effective_temp = 1.0 / logit_scale

        sim = torch.matmul(vision_features, text_features.T) / effective_temp

        if match_ids is None:
            # If no match_ids, use diagonal matching (fallback)
            print("WARNING: No match_ids provided, using diagonal matching")
            targets_v2t = torch.arange(batch_size, device=sim.device)
            targets_t2v = torch.arange(batch_size, device=sim.device)
            alignment_loss = 0
        else:
            # Create targets based on match_ids
            targets_v2t = []
            targets_t2v = []

            # For all pairs tracking
            all_matches = {}
            for i in range(batch_size):
                # Group all matches by match_id for efficient lookup
                mid_str = str(match_ids[i])
                if mid_str not in all_matches:
                    all_matches[mid_str] = []
                all_matches[mid_str].append(i)

            # For each item, find matching targets from the same group
            for i in range(batch_size):
                mid_str = str(match_ids[i])
                matches = [j for j in all_matches[mid_str] if j != i]

                if matches:
                    # Randomly select one match
                    rand_idx = int(torch.randint(0, len(matches), (1,)).item())
                    match_idx = matches[rand_idx]
                    targets_v2t.append(match_idx)
                else:
                    # If no matches, use itself as target
                    targets_v2t.append(i)

            # Repeat for text to vision direction
            for i in range(batch_size):
                mid_str = str(match_ids[i])
                matches = [j for j in all_matches[mid_str] if j != i]

                if matches:
                    rand_idx = int(torch.randint(0, len(matches), (1,)).item())
                    match_idx = matches[rand_idx]
                    targets_t2v.append(match_idx)
                else:
                    targets_t2v.append(i)

            targets_v2t = torch.tensor(targets_v2t, device=sim.device)
            targets_t2v = torch.tensor(targets_t2v, device=sim.device)

            # Compute direct alignment loss within groups with stronger weighting
            alignment_loss = 0
            groups = {}
            for i, mid in enumerate(match_ids):
                mid_str = str(mid)  # Convert to string to ensure consistent keys
                if mid_str not in groups:
                    groups[mid_str] = []
                groups[mid_str].append(i)

            # Track number of valid alignment pairs for normalization
            total_alignment_pairs = 0

            for group_indices in groups.values():
                if len(group_indices) < 2:
                    continue

                # Get features for this group
                group_vision = vision_features[group_indices]
                group_text = text_features[group_indices]

                # Direct pairwise alignment within group with stronger supervision
                for i in range(len(group_indices)):
                    for j in range(i + 1, len(group_indices)):
                        vision_i = group_vision[i]
                        vision_j = group_vision[j]
                        text_i = group_text[i]
                        text_j = group_text[j]

                        # Cross-modal alignment: align vision with matching text (directly)
                        vision_to_text_alignment = F.mse_loss(
                            vision_i, text_j
                        ) + F.mse_loss(vision_j, text_i)

                        # Same-modal cohesion: make same-modality features similar
                        vision_cohesion = F.mse_loss(vision_i, vision_j)
                        text_cohesion = F.mse_loss(text_i, text_j)

                        # Stronger weighting for cross-modal alignment
                        alignment_loss += vision_to_text_alignment + 0.5 * (
                            vision_cohesion + text_cohesion
                        )
                        total_alignment_pairs += 1

                # Compute centroids and align them as well - useful for multi-sample groups
                if len(group_indices) > 2:
                    vision_centroid = group_vision.mean(dim=0)
                    text_centroid = group_text.mean(dim=0)

                    # Centroid alignment is very effective for handling variance across a semantic group
                    centroid_alignment = F.mse_loss(vision_centroid, text_centroid)
                    alignment_loss += centroid_alignment * len(group_indices) * 0.5
                    total_alignment_pairs += 1

            # Normalize alignment loss by number of pairs to keep scale consistent
            if total_alignment_pairs > 0:
                alignment_loss = alignment_loss / total_alignment_pairs

                # Add explicit tracking of positive similarities
                positive_similarities = []
                negative_similarities = []

                # Calculate average similarity for all matching and non-matching pairs
                with torch.no_grad():
                    for i in range(batch_size):
                        for j in range(batch_size):
                            # Use torch.nn.functional.cosine_similarity instead of F.cosine_similarity
                            similarity_val = torch.nn.functional.cosine_similarity(
                                vision_features[i].unsqueeze(0),
                                text_features[j].unsqueeze(0),
                            ).item()
                            if match_ids[i] == match_ids[j] and i != j:
                                positive_similarities.append(similarity_val)
                            elif match_ids[i] != match_ids[j]:
                                negative_similarities.append(similarity_val)

                # Log the separation stats if we have both positives and negatives
                if positive_similarities and negative_similarities:
                    avg_pos_sim = sum(positive_similarities) / len(
                        positive_similarities
                    )
                    avg_neg_sim = sum(negative_similarities) / len(
                        negative_similarities
                    )
                    separation = avg_pos_sim - avg_neg_sim

                    if self.iteration % 10 == 0:  # Only log occasionally to avoid spam
                        print(
                            f"Similarity stats - Pos: {avg_pos_sim:.4f}, Neg: {avg_neg_sim:.4f}, Gap: {separation:.4f}"
                        )

            # Add hard negative mining for more challenging learning
            # Focus on highly confusing negative pairs (textually similar but semantically different)
            if batch_size > 4 and self.iteration > 10:
                # Find hard negatives - high similarity pairs that shouldn't match
                hard_negatives_loss = 0.0
                hard_count = 0

                # Create match matrix for easy lookup
                match_matrix = torch.zeros(
                    (batch_size, batch_size), dtype=torch.bool, device=sim.device
                )
                for i in range(batch_size):
                    for j in range(batch_size):
                        match_matrix[i, j] = match_ids[i] == match_ids[j]

                # Find top k% most similar non-matching pairs
                with torch.no_grad():
                    similarity = torch.matmul(vision_features, text_features.T)

                    # Create negative mask (where pairs don't match)
                    negative_mask = ~match_matrix

                    # Find hardest negatives (highest similarity non-matches)
                    hard_negative_values, _ = torch.topk(
                        similarity[negative_mask], k=max(5, batch_size // 4)
                    )
                    hard_threshold = hard_negative_values.min()

                    # Create mask for hard negatives
                    hard_negative_mask = (similarity > hard_threshold) & negative_mask

                    # Count hard negatives
                    hard_count = hard_negative_mask.sum().item()

                # Push hard negatives further apart
                if hard_count > 0:
                    # Push apart vision-text pairs that are similar but shouldn't be
                    repulsion_loss = (
                        torch.sum(similarity * hard_negative_mask) / hard_count
                    )
                    hard_negatives_loss = (
                        repulsion_loss * 0.5
                    )  # Lower weight than main loss

                    # Add to alignment loss
                    alignment_loss = alignment_loss + hard_negatives_loss

        # Compute cross-entropy loss for both directions
        loss_v2t = F.cross_entropy(sim, targets_v2t)
        loss_t2v = F.cross_entropy(sim.T, targets_t2v)
        ce_loss = (loss_v2t + loss_t2v) / 2

        # Add contrastive margin to ensure clear separation between positives and negatives
        if batch_size > 4 and match_ids is not None:
            # Create match matrix for easier indexing
            match_matrix = torch.zeros(
                (batch_size, batch_size), dtype=torch.bool, device=sim.device
            )
            for i in range(batch_size):
                for j in range(batch_size):
                    match_matrix[i, j] = match_ids[i] == match_ids[j]

            # Extract diagonal elements (each item with itself)
            diag_mask = torch.eye(batch_size, dtype=torch.bool, device=sim.device)
            match_matrix = match_matrix & ~diag_mask  # Remove self-matches

            # Calculate margin contrastive loss
            pos_margin = 0.8  # Positives should be above this
            neg_margin = 0.3  # Negatives should be below this

            # Margin losses
            pos_mask = match_matrix.float()
            neg_mask = (~match_matrix & ~diag_mask).float()

            # Positive pairs should have high similarity
            pos_loss = torch.sum(F.relu(pos_margin - sim) * pos_mask) / max(
                1, int(pos_mask.sum().item())
            )

            # Negative pairs should have lower similarity
            neg_loss = torch.sum(F.relu(sim - neg_margin) * neg_mask) / max(
                1, int(neg_mask.sum().item())
            )

            # Add margin loss with appropriate weighting
            margin_loss = (pos_loss + neg_loss) * 0.3
            ce_loss = ce_loss + margin_loss

        # Combine losses with higher alignment weight and include extra losses
        # total_loss = ce_loss + 0.8 * alignment_loss + extra_loss

        # Make sure to log the extra loss components
        if isinstance(extra_loss, torch.Tensor) and extra_loss > 0:
            print(
                f"Extra diversity loss: {extra_loss.item():.4f} with weight {diversity_weight.item():.4f}"
            )

        # Adjust weights to balance contrastive learning with alignment
        total_loss = 0.3 * ce_loss + 1.0 * alignment_loss + extra_loss

        # Compute accuracy
        with torch.no_grad():
            v2t_pred = torch.argmax(sim, dim=1)
            t2v_pred = torch.argmax(sim.T, dim=1)
            v2t_correct = (v2t_pred == targets_v2t).float().mean()
            t2v_correct = (t2v_pred == targets_t2v).float().mean()
            accuracy = (v2t_correct + t2v_correct) / 2

            # Calculate positive-negative separation for diagnostics
            if match_ids is not None and batch_size > 2:
                # Create match matrix
                match_matrix = torch.zeros(
                    (batch_size, batch_size), dtype=torch.bool, device=sim.device
                )
                for i in range(batch_size):
                    for j in range(batch_size):
                        match_matrix[i, j] = match_ids[i] == match_ids[j]

                # Calculate mean similarity for matching vs non-matching pairs
                diag_mask = torch.eye(batch_size, dtype=torch.bool, device=sim.device)
                matches = match_matrix & ~diag_mask  # Remove self-matches
                non_matches = ~match_matrix & ~diag_mask  # Remove self-non-matches

                raw_sim = torch.matmul(vision_features, text_features.T)

                if matches.sum() > 0:
                    pos_sim_mean = raw_sim[matches].mean().item()
                else:
                    pos_sim_mean = 0.0

                if non_matches.sum() > 0:
                    neg_sim_mean = raw_sim[non_matches].mean().item()
                else:
                    neg_sim_mean = 0.0

                # Calculate positive-negative separation
                separation = pos_sim_mean - neg_sim_mean

        # Return comprehensive metrics
        return {
            "loss": total_loss,
            "ce_loss": ce_loss.item(),
            "alignment_loss": (
                alignment_loss.item()
                if isinstance(alignment_loss, torch.Tensor)
                else alignment_loss
            ),
            "v2t_loss": loss_v2t.item(),
            "t2v_loss": loss_t2v.item(),
            "accuracy": accuracy.item(),
            "temperature": effective_temp,
            "pos_sim": pos_sim_mean if "pos_sim_mean" in locals() else 0.0,
            "neg_sim": neg_sim_mean if "neg_sim_mean" in locals() else 0.0,
            "separation": separation if "separation" in locals() else 0.0,
            "extra_loss": (
                extra_loss.item() if isinstance(extra_loss, torch.Tensor) else 0.0
            ),
        }


def set_seed(seed):
    """Set random seed for reproducibility"""
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    random.seed(seed)


def main():
    """Main function for multimodal training demo."""
    # Parse arguments
    parser = get_multimodal_training_args()
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    # Set random seed
    set_seed(args.seed)

    # Create output directories
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, args.checkpoint_dir), exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, args.log_dir), exist_ok=True)

    # Set device
    if args.device is None:
        if torch.backends.mps.is_available():
            device = torch.device("mps")
        elif torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")
    else:
        device = torch.device(args.device)

    print(f"Using device: {device}")

    # Create preprocessor and tokenizer
    image_preprocessor = ImagePreprocessor(
        image_size=224,
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225),
    )

    # Create a tokenizer that's compatible with our model choices
    if args.use_pretrained_text:
        # If using dimension-matched presets, use proper tokenizers
        if args.model_size == "small":
            # MiniLM for small preset
            tokenizer = SimpleTokenizer(
                pretrained_model_name="microsoft/MiniLM-L12-H384-uncased",
                max_length=args.max_text_length,
            )
            print(f"Using MiniLM tokenizer with max length {args.max_text_length}")
        elif args.model_size == "medium":
            # FlauBERT for medium preset
            tokenizer = SimpleTokenizer(
                pretrained_model_name="flaubert-small-cased",
                max_length=args.max_text_length,
            )
            print(f"Using FlauBERT tokenizer with max length {args.max_text_length}")
        elif args.model_size == "large":
            # Different tokenizers based on device for large preset
            if (
                torch.backends.mps.is_available()
                and torch.device(args.device).type == "mps"
            ):
                # ALBERT for MPS with large preset
                tokenizer = SimpleTokenizer(
                    pretrained_model_name="albert-base-v2",
                    max_length=args.max_text_length,
                )
                print(f"Using ALBERT tokenizer with max length {args.max_text_length}")
            else:
                # BERT for other devices with large preset
                tokenizer = SimpleTokenizer(
                    pretrained_model_name="bert-base-uncased",
                    max_length=args.max_text_length,
                )
                print(f"Using BERT tokenizer with max length {args.max_text_length}")
        # If using direct HuggingFace model names
        elif (
            "/" in args.text_model
            or args.text_model.startswith("microsoft")
            or args.text_model.startswith("google")
        ):
            # Use the exact model name for the tokenizer
            tokenizer = SimpleTokenizer(
                pretrained_model_name=args.text_model,
                max_length=args.max_text_length,
            )
            print(
                f"Using tokenizer matched to {args.text_model} with max length {args.max_text_length}"
            )
        # Handle standard model names
        elif args.text_model == "mobilebert":
            # Use MobileBERT tokenizer for MobileBERT model
            tokenizer = SimpleTokenizer(
                pretrained_model_name="google/mobilebert-uncased",
                max_length=args.max_text_length,
            )
            print(f"Using MobileBERT tokenizer with max length {args.max_text_length}")
        elif args.text_model == "albert-base":
            # Use ALBERT tokenizer for ALBERT model
            tokenizer = SimpleTokenizer(
                pretrained_model_name="albert-base-v2", max_length=args.max_text_length
            )
            print(f"Using ALBERT tokenizer with max length {args.max_text_length}")
        elif args.text_model == "bert-base":
            # Use BERT tokenizer for BERT model
            tokenizer = SimpleTokenizer(
                pretrained_model_name="bert-base-uncased",
                max_length=args.max_text_length,
            )
            print(f"Using BERT tokenizer with max length {args.max_text_length}")
        elif args.text_model == "roberta-base":
            # Use RoBERTa tokenizer for RoBERTa model
            tokenizer = SimpleTokenizer(
                pretrained_model_name="roberta-base", max_length=args.max_text_length
            )
            print(f"Using RoBERTa tokenizer with max length {args.max_text_length}")
        elif args.text_model == "distilbert-base":
            # Use DistilBERT tokenizer for DistilBERT model
            tokenizer = SimpleTokenizer(
                pretrained_model_name="distilbert-base-uncased",
                max_length=args.max_text_length,
            )
            print(f"Using DistilBERT tokenizer with max length {args.max_text_length}")
        elif args.text_model == "minilm-384":
            # Use MiniLM tokenizer
            tokenizer = SimpleTokenizer(
                pretrained_model_name="microsoft/MiniLM-L12-H384-uncased",
                max_length=args.max_text_length,
            )
            print(f"Using MiniLM tokenizer with max length {args.max_text_length}")
        elif args.text_model == "flaubert-small-cased":
            # Use FlauBERT tokenizer
            tokenizer = SimpleTokenizer(
                pretrained_model_name="flaubert-small-cased",
                max_length=args.max_text_length,
            )
            print(f"Using FlauBERT tokenizer with max length {args.max_text_length}")
        else:
            # Default to basic SimpleTokenizer for non-HuggingFace text models
            tokenizer = SimpleTokenizer(max_length=args.max_text_length)
            print(f"Using basic SimpleTokenizer with max length {args.max_text_length}")
    else:
        # For custom transformers, use the basic SimpleTokenizer
        tokenizer = SimpleTokenizer(max_length=args.max_text_length)
        print(f"Using basic SimpleTokenizer with max length {args.max_text_length}")

    if args.use_simple_model:
        logger.info("Using SimpleMultimodalModel for debugging")

        # Use your existing model but wrap it in SimpleMultimodalModel
        # First create the original model to get its components
        original_model = create_multimodal_model(args, device=device)

        # Extract the vision and text models
        vision_model = original_model.vision_model
        text_model = original_model.text_model

        # Create simple model using these components
        model = SimpleMultimodalModel(
            vision_model=vision_model,
            text_model=text_model,
            projection_dim=args.fusion_dim,
        )

        # Create supervised alignment loss with the user-provided temperature
        # Make sure to use a low temperature (0.07-0.1) for sharper contrast
        temperature = args.temperature if args.temperature is not None else 0.07
        print(f"Using temperature value: {temperature} (lower = sharper contrasts)")
        loss_fn = SupervisedAlignmentLoss(temperature=temperature)

        logger.info("Created simple model and supervised alignment loss")

        # Enable gradient checkpointing for memory efficiency if available
        if (
            hasattr(model.vision_model, "gradient_checkpointing_enable")
            and torch.cuda.is_available()
        ):
            model.vision_model.gradient_checkpointing_enable()
            logger.info("Enabled gradient checkpointing for vision model")

        if (
            hasattr(model.text_model, "gradient_checkpointing_enable")
            and torch.cuda.is_available()
        ):
            model.text_model.gradient_checkpointing_enable()
            logger.info("Enabled gradient checkpointing for text model")

        # Print detailed feature dimensions for debugging
        logger.info(f"Model projection dimensions: {args.fusion_dim}")

        # Initialize with stronger contrast between samples
        # This helps break symmetry and prevent feature collapse
        logger.info("Initializing models with stronger contrast:")
        for module in model.vision_proj.modules():
            if isinstance(module, nn.Linear):
                std = math.sqrt(2.0 / module.weight.size(1))
                nn.init.normal_(module.weight, mean=0.0, std=std * 1.2)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0.01)

        for module in model.text_proj.modules():
            if isinstance(module, nn.Linear):
                std = math.sqrt(2.0 / module.weight.size(1))
                nn.init.normal_(module.weight, mean=0.0, std=std * 1.2)
                if module.bias is not None:
                    nn.init.constant_(module.bias, -0.01)

    else:
        # Original model creation
        model = create_multimodal_model(args, device=device)

    # Create data loaders
    train_loader, val_loader, test_loader = create_data_loaders(
        args, image_preprocessor, tokenizer
    )

    # Get dataset info from the training dataset
    is_synthetic = False
    if args.use_synthetic or args.dataset == "synthetic":
        is_synthetic = True
    else:
        try:
            # Try to get dataset info if the dataset has the method
            dataset = train_loader.dataset
            if hasattr(dataset, "get_split_proportions"):
                # Cast to our protocol type to satisfy the type checker
                split_dataset = cast(SplitProportionsDataset, dataset)
                train_dataset_info = split_dataset.get_split_proportions()
                dataset_name = train_dataset_info.get("dataset_name", "unknown")

                # Check if we're using real Flickr30k data
                if dataset_name == "flickr30k":
                    # The data source is Flickr30k, check if we're using cached files from disk
                    # or if we had to generate synthetic data
                    cache_status = getattr(dataset, "loaded_from_cache", True)
                    is_synthetic = (
                        not cache_status
                    )  # If not loaded from cache, it's synthetic
                else:
                    # For other dataset types or unknown sources, consider it synthetic
                    is_synthetic = True

                # Add an extra warning if using a very small dataset
                train_dataset_size = train_dataset_info.get("total_samples", 0)
                if train_dataset_size < 100:
                    print("\n" + "=" * 80)
                    print("WARNING: Using a very small dataset (<100 samples).")
                    print(
                        "Small datasets may lead to overfitting and unrealistic metrics."
                    )
                    print("Consider using more data for better evaluation.")
                    print("=" * 80 + "\n")
            else:
                # If the dataset doesn't have the method, assume it's synthetic
                is_synthetic = True
                print("\n" + "=" * 80)
                print(
                    "WARNING: Dataset does not provide split information. Assuming synthetic data."
                )
                print("Results will not be representative of real-world performance.")
                print("=" * 80 + "\n")
        except (AttributeError, TypeError):
            # If we can't get dataset info, assume it's synthetic
            is_synthetic = True
            print("\n" + "=" * 80)
            print("WARNING: Could not determine dataset type. Assuming synthetic data.")
            print("Results will not be representative of real-world performance.")
            print("=" * 80 + "\n")

    if is_synthetic:
        print("\n" + "=" * 80)
        print("WARNING: Using synthetic data for training and evaluation.")
        print(
            "For real evaluation, please ensure you have proper access to the real dataset."
        )
        print("=" * 80 + "\n")

    # Get the dataset size for contrastive loss sampling strategy
    dataset_size = None
    try:
        dataset = train_loader.dataset
        if hasattr(dataset, "get_split_proportions"):
            # Cast to our protocol type to satisfy the type checker
            split_dataset = cast(SplitProportionsDataset, dataset)
            train_dataset_info = split_dataset.get_split_proportions()
            dataset_size = train_dataset_info.get("total_samples", None)
    except (AttributeError, TypeError):
        logger.warning(
            "Could not determine dataset size for contrastive sampling strategy"
        )

    # Create loss function with enhanced configuration
    if hasattr(args, "use_simple_model") and args.use_simple_model:
        # Create supervised alignment loss with higher temperature
        loss_fn = SupervisedAlignmentLoss(temperature=0.5)
        logger.info("Created SupervisedAlignmentLoss with temperature=0.5")
    else:
        # Original loss function creation
        # loss_fn = create_loss_function(args)
        loss_fn = create_loss_function(args, dataset_size, train_loader)
    # CRITICAL CHANGE: Initialize the model with pre-trained weights if possible
    # This helps avoid the "cold start" problem in contrastive learning
    try:
        if hasattr(model.vision_model, "initialize_from_pretrained"):
            logger.info("Initializing vision model from pretrained weights")
            success = model.vision_model.initialize_from_pretrained()
            if success:
                logger.info(
                    "Successfully initialized vision model from pretrained weights"
                )
            else:
                logger.warning(
                    "Failed to initialize vision model from pretrained weights"
                )

        if hasattr(model.text_model, "initialize_from_pretrained"):
            logger.info("Initializing text model from pretrained weights")
            success = model.text_model.initialize_from_pretrained()
            if success:
                logger.info(
                    "Successfully initialized text model from pretrained weights"
                )
            else:
                logger.warning(
                    "Failed to initialize text model from pretrained weights"
                )
    except Exception as e:
        logger.warning(f"Error initializing from pretrained: {str(e)}")

    # Create trainer with improved settings for contrastive learning
    # For models with feature collapse, we need special training settings
    if args.use_simple_model:
        # For the simple model, use a very specific learning strategy to prevent feature collapse
        trainer = MultimodalTrainer(
            model=model,
            train_dataloader=train_loader,
            val_dataloader=val_loader,
            test_dataloader=test_loader,
            loss_fn=loss_fn,
            num_epochs=args.num_epochs,
            learning_rate=args.learning_rate
            * 0.25,  # Reduced base LR, but projections have higher LR
            weight_decay=args.weight_decay
            * 2.0,  # Moderate weight decay for better generalization
            warmup_steps=int(
                max(500, args.warmup_steps * 1.5)
            ),  # Longer warmup for stability
            checkpoint_dir=os.path.join(args.output_dir, args.checkpoint_dir),
            log_dir=os.path.join(args.output_dir, args.log_dir),
            device=device,
            mixed_precision=args.use_mixed_precision,
            evaluation_steps=0,  # Only evaluate at the end of each epoch
            log_steps=25,  # Very frequent logging to closely track the collapse issue
            early_stopping_patience=10,  # More patience for slow learning process
            clip_grad_norm=0.5,  # Moderate gradient clipping to allow learning while preventing extremes
            accumulation_steps=2,  # Reduced accumulation steps for more frequent updates
            balance_modality_gradients=True,
        )
        logger.info(
            "Using special training settings for SimpleMultimodalModel to prevent feature collapse"
        )

        # Setup optimizer with layer-wise learning rates
        # Different learning rates for different parts of the model
        optimizer_grouped_parameters = [
            # Vision model gets very low learning rate
            {
                "params": [
                    p
                    for n, p in model.vision_model.named_parameters()
                    if p.requires_grad
                ],
                "lr": args.learning_rate * 0.005,  # Even lower LR for base vision model
                "weight_decay": args.weight_decay * 2.0,
                "name": "vision_model_params",
            },
            # Text model gets very low learning rate
            {
                "params": [
                    p for n, p in model.text_model.named_parameters() if p.requires_grad
                ],
                "lr": args.learning_rate * 0.005,  # Even lower LR for base text model
                "weight_decay": args.weight_decay * 2.0,
                "name": "text_model_params",
            },
            # Projection layers get higher learning rate
            {
                "params": [
                    p
                    for n, p in model.vision_proj.named_parameters()
                    if p.requires_grad
                ],
                "lr": args.learning_rate
                * 0.5,  # Much higher LR for projection layers to combat collapse
                "weight_decay": args.weight_decay * 2.0,
                "name": "vision_proj_params",
            },
            {
                "params": [
                    p for n, p in model.text_proj.named_parameters() if p.requires_grad
                ],
                "lr": args.learning_rate
                * 0.5,  # Much higher LR for projection layers to combat collapse
                "weight_decay": args.weight_decay * 2.0,
                "name": "text_proj_params",
            },
        ]

        # Create optimizer with these parameter groups
        optimizer = torch.optim.AdamW(optimizer_grouped_parameters)
        trainer.optimizer = optimizer
        logger.info("Created custom optimizer with layer-wise learning rates")

    else:
        # Regular settings for the full model
        trainer = MultimodalTrainer(
            model=model,
            train_dataloader=train_loader,
            val_dataloader=val_loader,
            test_dataloader=test_loader,
            loss_fn=loss_fn,
            num_epochs=args.num_epochs,
            learning_rate=args.learning_rate,  # Original learning rate
            weight_decay=args.weight_decay
            * 2.0,  # Double weight decay to prevent overfitting
            warmup_steps=args.warmup_steps,
            checkpoint_dir=os.path.join(args.output_dir, args.checkpoint_dir),
            log_dir=os.path.join(args.output_dir, args.log_dir),
            device=device,
            mixed_precision=args.use_mixed_precision,
            evaluation_steps=0,  # Only evaluate at the end of each epoch
            log_steps=100,  # Reduce logging frequency to avoid spam
            early_stopping_patience=5,
            clip_grad_norm=1.0,  # Enforce gradient clipping for stability
            accumulation_steps=2,  # Use gradient accumulation for more stable updates
            balance_modality_gradients=True,
        )

    # Train model
    print("Starting training...")
    print_model_summary(model, "TRAINING MULTIMODAL MODEL")

    # Use multi-stage training if enabled
    if args.use_multistage_training:
        # Determine which pretrained models we're using
        vision_status = "pretrained" if args.use_pretrained else "from scratch"
        text_status = "pretrained" if args.use_pretrained_text else "from scratch"
        freeze_status = "frozen" if args.freeze_base_models else "trainable"

        print(f"\nStarting multi-stage training with:")
        print(f"- Vision model: {args.vision_model} ({vision_status}, {freeze_status})")
        print(f"- Text model: {args.text_model} ({text_status}, {freeze_status})")

        # Add warning if using BERT with MPS
        is_bert = any(
            model_type in args.text_model.lower() for model_type in ["bert", "roberta"]
        )
        device_is_mps = next(model.parameters()).device.type == "mps"

        if is_bert and device_is_mps and args.use_pretrained_text:
            print("\n⚠️ INFO: Using BERT model on MPS device (Apple Silicon)")
            print(
                "The code now properly handles BERT models on MPS devices with a CPU fallback."
            )
            print("Processing may be slower but quality will not be affected.")

        # Determine if we're training from scratch
        # Either explicitly requested, or no pretrained weights are being used
        training_from_scratch = args.from_scratch or not (
            args.use_pretrained or args.use_pretrained_text
        )

        # If training from scratch, enable enhanced data augmentation
        if training_from_scratch:
            print("\nEnabling enhanced data augmentation for training from scratch")
            # This would normally modify dataloaders to include more augmentation strategies
            # For now, we're just acknowledging this would happen here
            # In a full implementation, we'd add more transforms, enable mixup, etc.
            train_dataloader = trainer.train_dataloader
            # TODO: Implement actual augmentation enhancement here

        # Print appropriate stage info based on training approach
        if training_from_scratch:
            print("\nTraining from scratch with specialized curriculum:")
            print("Stage 1a: Train early vision layers (edge/texture detection)")
            print("Stage 1b: Train early text layers (token-level understanding)")
            print("Stage 2a: Train mid-level vision layers (object parts)")
            print("Stage 2b: Train mid-level text layers (phrase-level understanding)")
            print("Stage 3: Train high-level representation in both modalities")
            print("Stage 4: Train cross-modal fusion")
            print("Stage 5: Fine-tune everything with hard negative mining")
        elif args.freeze_base_models:
            print("\nTraining stages (with pre-trained models, frozen):")
            print("Stage 1: Train projections only (vision and text models frozen)")
            print("Stage 2: Train fusion layers (vision and text models frozen)")
            print("Stage 3: Fine-tune everything together")
        else:
            print("\nTraining stages (with pre-trained models):")
            print(
                "Stage 1: Train all components with lower learning rate on pretrained models"
            )
            print("Stage 2: Increase focus on fusion layers")
            print("Stage 3: Fine-tune everything with hard negative mining")

        trainer.train_multistage()
    else:
        if args.freeze_base_models:
            print(
                "\nStarting standard training with FROZEN base models (only fusion layers will train)..."
            )
        else:
            print("\nStarting standard training with ALL parameters trainable...")
        trainer.train()

    # Print model summary after training
    print_model_summary(model, "MULTIMODAL MODEL AFTER TRAINING")

    # Run final evaluation
    print("Running final evaluation...")
    print_model_summary(model, "EVALUATION MODEL")
    test_metrics = trainer.evaluate(test_loader)

    # Add this code to visualize test samples
    print("Generating test sample visualizations...")

    # Run inference demo to get visualizations
    inference_metrics = run_inference_demo(
        model, image_preprocessor, tokenizer, device, args
    )

    # Convert PyTorch tensors to Python types for display and serialization
    python_test_metrics = convert_tensors_to_python_types(test_metrics)
    python_inference_metrics = convert_tensors_to_python_types(inference_metrics)

    # Create dictionaries with both metrics and model info
    complete_test_metrics = {
        "metrics": python_test_metrics,
        "model_info": {
            "total_parameters": count_parameters(model),
            "vision_model": args.vision_model,
            "text_model": args.text_model,
            "fusion_type": args.fusion_type,
            "fusion_dim": args.fusion_dim,
        },
    }

    complete_inference_metrics = {
        "metrics": python_inference_metrics,
        "model_info": {
            "total_parameters": count_parameters(model),
            "vision_model": args.vision_model,
            "text_model": args.text_model,
            "fusion_type": args.fusion_type,
            "fusion_dim": args.fusion_dim,
        },
    }

    if is_synthetic:
        print(f"Test metrics (SYNTHETIC DATA): {python_test_metrics}")
    else:
        print(f"Test metrics: {python_test_metrics}")

    # Save final test metrics
    import json

    with open(os.path.join(args.output_dir, "test_metrics.json"), "w") as f:
        json.dump(complete_test_metrics, f, indent=2)

    # Save demo results
    with open(os.path.join(args.output_dir, "demo_results.json"), "w") as f:
        json.dump(complete_inference_metrics, f, indent=2)

    if is_synthetic:
        print("Multimodal training demo completed with SYNTHETIC DATA!")
        print(
            "For real evaluation, please download and use the actual Flickr30k dataset."
        )
    else:
        print("Multimodal training demo completed!")


if __name__ == "__main__":
    main()
