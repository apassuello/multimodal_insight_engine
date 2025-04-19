# src/training/contrastive_learning.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Any, Union
import numpy as np


class MemoryQueueContrastiveLoss(nn.Module):
    def __init__(self, dim=512, queue_size=8192, temperature=0.07):
        super().__init__()
        self.temperature = temperature
        self.queue_size = queue_size
        self.dim = dim

        # We'll initialize the queues on first forward pass to get correct dimensions
        self.register_buffer("vision_queue", None)
        self.register_buffer("text_queue", None)

        # Queue pointers
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))
        
        # Add a fill level tracker for smoother stage transitions
        self.register_buffer("queue_fill_level", torch.zeros(1, dtype=torch.long))

        # Flag to track device and initialization
        self.initialized = False

        # For gradient computation safety, we'll use two separate buffers:
        # - One for storing previous iterations' embeddings (non-gradient)
        # - One for the current computation (with gradient)
        self.use_two_buffers = True
        
        # Store initial temperature for adaptive adjustment
        self.initial_temperature = temperature
        self.max_temperature = temperature * 1.3  # Higher temperature for early training

    def initialize_queue(self, vision_features, text_features):
        """
        Initialize the queue with the provided features.
        This is useful for pre-filling the queue when transitioning between training stages.
        
        Args:
            vision_features: Vision features [batch_size, dim]
            text_features: Text features [batch_size, dim]
        """
        if not isinstance(vision_features, torch.Tensor) or not isinstance(text_features, torch.Tensor):
            print("Warning: initialize_queue requires tensor inputs")
            return
            
        # Get device and feature dimensions
        device = vision_features.device
        feature_dim = vision_features.shape[1]
        batch_size = vision_features.shape[0]
        
        # Initialize queues if not initialized yet
        if not self.initialized:
            # Create and normalize queues with the correct feature dimension
            self.register_buffer(
                "vision_queue",
                F.normalize(
                    torch.randn(feature_dim, self.queue_size, device=device), dim=0
                ).detach(),
            )
            self.register_buffer(
                "text_queue",
                F.normalize(
                    torch.randn(feature_dim, self.queue_size, device=device), dim=0
                ).detach(),
            )
            self.queue_ptr = self.queue_ptr.to(device)
            self.queue_fill_level = self.queue_fill_level.to(device)
            self.initialized = True
            print(f"Initialized memory queues with dimension {feature_dim}x{self.queue_size}")
            
        # Normalize the features if they're not already normalized
        vision_features = F.normalize(vision_features, dim=1)
        text_features = F.normalize(text_features, dim=1)
            
        # Call update queue to add these features
        self._update_queue(vision_features, text_features)
            
        # Print initialization info
        fill_level = int(self.queue_fill_level.item())
        print(f"Queue initialized with {fill_level}/{self.queue_size} entries ({fill_level/self.queue_size:.1%} full)")

    def forward(self, vision_features, text_features, match_ids):
        # Get device and feature dimensions
        device = vision_features.device
        feature_dim = vision_features.shape[1]

        # Initialize queues on first forward pass with correct dimensions
        if not self.initialized:
            # Create and normalize queues with the correct feature dimension
            # Use .data to avoid tracking in autograd
            self.register_buffer(
                "vision_queue",
                F.normalize(
                    torch.randn(feature_dim, self.queue_size, device=device), dim=0
                ).detach(),
            )
            self.register_buffer(
                "text_queue",
                F.normalize(
                    torch.randn(feature_dim, self.queue_size, device=device), dim=0
                ).detach(),
            )
            self.queue_ptr = self.queue_ptr.to(device)
            self.queue_fill_level = torch.zeros(1, dtype=torch.long, device=device)
            self.initialized = True
            print(
                f"Initialized memory queues with dimension {feature_dim}x{self.queue_size}"
            )
        elif self.vision_queue.shape[0] != feature_dim:
            # Handle case where feature dimensions have changed
            print(
                f"Feature dimension changed from {self.vision_queue.shape[0]} to {feature_dim}, reinitializing queues"
            )
            self.register_buffer(
                "vision_queue",
                F.normalize(
                    torch.randn(feature_dim, self.queue_size, device=device), dim=0
                ).detach(),
            )
            self.register_buffer(
                "text_queue",
                F.normalize(
                    torch.randn(feature_dim, self.queue_size, device=device), dim=0
                ).detach(),
            )
            # Reset fill level when reinitializing
            self.queue_fill_level = torch.zeros(1, dtype=torch.long, device=device)
        elif self.vision_queue.device != device:
            # Move queues to the correct device if needed
            self.vision_queue = self.vision_queue.to(device).detach()
            self.text_queue = self.text_queue.to(device).detach()
            self.queue_ptr = self.queue_ptr.to(device)
            self.queue_fill_level = self.queue_fill_level.to(device)

        # Get batch size and normalize features
        batch_size = vision_features.shape[0]
        vision_features = F.normalize(vision_features, dim=1)
        text_features = F.normalize(text_features, dim=1)

        # Create match matrix based on match_ids
        match_matrix = torch.zeros(
            (batch_size, batch_size), dtype=torch.bool, device=vision_features.device
        )
        for i in range(batch_size):
            for j in range(batch_size):
                match_matrix[i, j] = match_ids[i] == match_ids[j]

        # Get current fill level for queue weighting and adaptive temperature
        fill_level = min(int(self.queue_fill_level.item()), self.queue_size)
        fill_ratio = fill_level / self.queue_size
        
        # Adjust temperature based on fill level - higher at beginning, lower as queue fills
        # This makes learning smoother when transitioning between stages
        effective_temperature = self.max_temperature - (self.max_temperature - self.initial_temperature) * fill_ratio
        if fill_ratio >= 0.95:  # Once queue is nearly full, use base temperature
            effective_temperature = self.initial_temperature
        
        # Apply temperature to similarities
        # CRITICAL ENHANCEMENT: Compare with queue embeddings too
        # Current batch similarities
        batch_similarities = torch.matmul(vision_features, text_features.T) / effective_temperature

        # Make sure queues are detached from computation graph
        text_queue_detached = self.text_queue.detach()
        vision_queue_detached = self.vision_queue.detach()

        # Vision-to-queue text similarities
        v2q_similarities = torch.matmul(vision_features, text_queue_detached) / effective_temperature

        # Text-to-queue vision similarities
        t2q_similarities = torch.matmul(text_features, vision_queue_detached) / effective_temperature

        # Apply weighting to queue similarities based on fill level
        # For sparse/empty queues, reduce their influence to avoid noisy gradients
        queue_weight = min(1.0, fill_ratio * 1.5)  # Gradually increase queue weight as it fills
        if fill_ratio < 0.2:  # Very sparse queue
            queue_weight = fill_ratio * 0.5  # Significantly reduce influence
        
        # Apply weighting
        v2q_similarities = v2q_similarities * queue_weight
        t2q_similarities = t2q_similarities * queue_weight

        # For each vision query, compute InfoNCE loss against in-batch and queue texts
        v2t_loss = 0
        for i in range(batch_size):
            # Positives from batch (based on match_ids)
            pos_idx = torch.where(match_matrix[i])[0]
            if len(pos_idx) == 0:
                continue  # Skip if no positives

            # Get positive similarities
            pos_sims = batch_similarities[i, pos_idx]

            # Combine in-batch negatives and queue negatives
            neg_sims_batch = batch_similarities[i, ~match_matrix[i]]
            neg_sims_queue = v2q_similarities[i]

            # Concat all negatives
            all_neg_sims = torch.cat([neg_sims_batch, neg_sims_queue])

            # For each positive pair, compute InfoNCE loss
            for pos_sim in pos_sims:
                pos_exp = torch.exp(pos_sim)
                neg_exp_sum = torch.sum(torch.exp(all_neg_sims))

                # InfoNCE loss term: -log(pos_exp / (pos_exp + neg_exp_sum))
                v2t_loss += -torch.log(pos_exp / (pos_exp + neg_exp_sum))

        # Similar process for text-to-vision direction
        t2v_loss = 0
        for i in range(batch_size):
            # Find positives (images that match this text)
            pos_idx = torch.where(match_matrix[:, i])[0]
            if len(pos_idx) == 0:
                continue

            # Get positive similarities
            pos_sims = batch_similarities[pos_idx, i]

            # Combine in-batch negatives and queue negatives
            neg_mask = ~match_matrix[:, i]
            neg_sims_batch = batch_similarities[neg_mask, i]
            neg_sims_queue = t2q_similarities[i]

            # Concat all negatives
            all_neg_sims = torch.cat([neg_sims_batch, neg_sims_queue])

            # For each positive, compute loss
            for pos_sim in pos_sims:
                pos_exp = torch.exp(pos_sim)
                neg_exp_sum = torch.sum(torch.exp(all_neg_sims))
                t2v_loss += -torch.log(pos_exp / (pos_exp + neg_exp_sum))

        # Update the queue with current batch
        self._update_queue(vision_features, text_features)

        # Normalize losses by number of positive pairs
        num_pos_pairs = match_matrix.sum().item()
        if num_pos_pairs > 0:
            v2t_loss = v2t_loss / num_pos_pairs
            t2v_loss = t2v_loss / num_pos_pairs
        else:
            return {"loss": torch.tensor(0.0, device=vision_features.device)}

        # Average the two directions
        loss = (v2t_loss + t2v_loss) / 2

        # Periodically log queue status and temperature
        if torch.rand(1).item() < 0.01:  # ~1% chance of logging
            print(f"Memory queue: {fill_level}/{self.queue_size} filled ({fill_ratio:.2f}), " 
                  f"using temperature={effective_temperature:.3f}")

        return {
            "loss": loss, 
            "loss_v2t": v2t_loss.item(), 
            "loss_t2v": t2v_loss.item(),
            "temperature": effective_temperature,
            "queue_fill": fill_ratio
        }

    @torch.no_grad()
    def _update_queue(self, vision_features, text_features):
        device = vision_features.device
        batch_size = vision_features.shape[0]
        ptr = int(self.queue_ptr)

        # Check if queues are properly initialized
        if not self.initialized:
            return  # Skip update if not initialized yet

        # Create new queue tensors instead of in-place updates
        new_vision_queue = self.vision_queue.clone().detach()
        new_text_queue = self.text_queue.clone().detach()

        # Detach input features
        vision_features_detached = vision_features.detach()
        text_features_detached = text_features.detach()

        # Update vision queue
        if ptr + batch_size <= self.queue_size:
            new_vision_queue[:, ptr : ptr + batch_size] = vision_features_detached.T
        else:
            # Handle wrap-around
            new_vision_queue[:, ptr:] = vision_features_detached[
                : self.queue_size - ptr
            ].T
            new_vision_queue[:, : batch_size - (self.queue_size - ptr)] = (
                vision_features_detached[self.queue_size - ptr :].T
            )

        # Update text queue
        if ptr + batch_size <= self.queue_size:
            new_text_queue[:, ptr : ptr + batch_size] = text_features_detached.T
        else:
            # Handle wrap-around
            new_text_queue[:, ptr:] = text_features_detached[: self.queue_size - ptr].T
            new_text_queue[:, : batch_size - (self.queue_size - ptr)] = (
                text_features_detached[self.queue_size - ptr :].T
            )

        # Replace old queues with new ones
        self.register_buffer("vision_queue", new_vision_queue)
        self.register_buffer("text_queue", new_text_queue)

        # Update pointer - create new tensor to avoid in-place operation
        new_ptr = torch.tensor(
            [(ptr + batch_size) % self.queue_size], dtype=torch.long, device=device
        )
        self.register_buffer("queue_ptr", new_ptr)
        
        # Update fill level - will max out at queue_size
        current_fill = int(self.queue_fill_level.item())
        new_fill = min(current_fill + batch_size, self.queue_size)
        self.register_buffer("queue_fill_level", 
                             torch.tensor([new_fill], dtype=torch.long, device=device))


class DynamicTemperatureContrastiveLoss(nn.Module):
    def __init__(self, base_temperature=0.07, min_temp=0.04, max_temp=0.2):
        super().__init__()
        self.base_temperature = base_temperature
        self.min_temp = min_temp
        self.max_temp = max_temp

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
        pos_mean = (pos_mask * sim_matrix).sum() / max(1, pos_mask.sum())
        neg_mean = (neg_mask * sim_matrix).sum() / max(1, neg_mask.sum())

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


class HardNegativeMiningContrastiveLoss(nn.Module):
    def __init__(
        self, temperature=0.07, hard_negative_factor=2.0, mining_strategy="semi-hard"
    ):
        super().__init__()
        self.temperature = temperature
        self.hard_negative_factor = hard_negative_factor  # Weight for hard negatives
        self.mining_strategy = mining_strategy  # "hard" or "semi-hard"

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


class ContrastiveLoss(nn.Module):
    """
    Implements contrastive loss for multimodal learning.

    This module supports different contrastive loss formulations:
    - InfoNCE loss (used in CLIP)
    - NT-Xent loss (normalized temperature-scaled cross entropy)
    - Supervised contrastive loss

    And different negative sampling strategies:
    - in-batch: Use only the current batch for negatives (standard approach)
    - memory-bank: Maintain a memory bank of past embeddings for additional negatives
    - global: Use all examples in the dataset as potential negatives
    """

    def __init__(
        self,
        temperature: float = 0.07,
        loss_type: str = "infonce",
        reduction: str = "mean",
        add_projection: bool = False,
        projection_dim: int = 256,
        input_dim: Optional[int] = None,
        sampling_strategy: str = "auto",  # "in-batch", "memory-bank", "global", or "auto"
        memory_bank_size: int = 4096,
        dataset_size: Optional[int] = None,
    ):
        """
        Initialize the contrastive loss module.

        Args:
            temperature: Temperature parameter controlling the sharpness of the distribution
            loss_type: Type of contrastive loss ("infonce", "nt_xent", or "supervised")
            reduction: How to reduce the loss ("mean", "sum", or "none")
            add_projection: Whether to add MLP projection heads for embeddings
            projection_dim: Dimension of projection space (if add_projection is True)
            input_dim: Input dimension for projection heads (required if add_projection is True)
            sampling_strategy: Strategy for sampling negatives ("in-batch", "memory-bank", "global", or "auto")
            memory_bank_size: Size of memory bank (if using memory-bank strategy)
            dataset_size: Total size of the dataset (used for auto strategy selection)
        """
        super().__init__()
        self.temperature = temperature
        self.loss_type = loss_type
        self.reduction = reduction
        self.training = True  # Explicitly set training mode
        self.memory_bank_size = memory_bank_size

        # Determine sampling strategy
        if sampling_strategy == "auto" and dataset_size is not None:
            if dataset_size < 1000:
                self.sampling_strategy = "global"
            elif dataset_size < 10000:
                self.sampling_strategy = "memory-bank"
            else:
                self.sampling_strategy = "in-batch"
        else:
            self.sampling_strategy = sampling_strategy

        # Initialize memory banks if needed
        if self.sampling_strategy == "memory-bank":
            # Register memory banks as buffers so they're saved in state_dict
            self.register_buffer(
                "vision_bank",
                torch.zeros(
                    memory_bank_size,
                    projection_dim if add_projection else input_dim or 512,
                ),
            )
            self.register_buffer(
                "text_bank",
                torch.zeros(
                    memory_bank_size,
                    projection_dim if add_projection else input_dim or 512,
                ),
            )
            self.register_buffer("bank_ptr", torch.zeros(1, dtype=torch.long))
            self.register_buffer(
                "bank_size", torch.zeros(1, dtype=torch.long)
            )  # Track actual filled size

        # Register global embeddings buffer if using global strategy
        if self.sampling_strategy == "global":
            self.dataset_size = dataset_size or 1000  # Default to reasonable size
            self.register_buffer(
                "global_vision_embeddings",
                torch.zeros(
                    self.dataset_size,
                    projection_dim if add_projection else input_dim or 512,
                ),
            )
            self.register_buffer(
                "global_text_embeddings",
                torch.zeros(
                    self.dataset_size,
                    projection_dim if add_projection else input_dim or 512,
                ),
            )
            self.register_buffer(
                "global_indices", torch.zeros(self.dataset_size, dtype=torch.long)
            )
            self.register_buffer(
                "global_size", torch.zeros(1, dtype=torch.long)
            )  # Track actual filled size
            self.register_buffer(
                "global_initialized", torch.tensor(False)
            )  # Track if initialized

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

    def update_memory_bank(
        self, vision_features: torch.Tensor, text_features: torch.Tensor
    ):
        """
        Update memory bank with current batch embeddings.

        Args:
            vision_features: Vision embeddings [batch_size, embed_dim]
            text_features: Text embeddings [batch_size, embed_dim]
        """
        if self.sampling_strategy != "memory-bank":
            return

        with torch.no_grad():
            batch_size = vision_features.shape[0]
            ptr = int(self.bank_ptr.item())

            # Circular update of memory bank
            if ptr + batch_size >= self.memory_bank_size:
                # Fill to the end
                remaining = self.memory_bank_size - ptr
                self.vision_bank[ptr:] = vision_features[:remaining].detach()
                self.text_bank[ptr:] = text_features[:remaining].detach()

                # Wrap around to the beginning
                overflow = batch_size - remaining
                if overflow > 0:
                    self.vision_bank[:overflow] = vision_features[remaining:].detach()
                    self.text_bank[:overflow] = text_features[remaining:].detach()

                # Update pointer
                new_ptr = overflow
            else:
                # Simple update without wrapping
                self.vision_bank[ptr : ptr + batch_size] = vision_features.detach()
                self.text_bank[ptr : ptr + batch_size] = text_features.detach()
                new_ptr = ptr + batch_size

            # Update pointer and size
            self.bank_ptr[0] = new_ptr % self.memory_bank_size
            self.bank_size[0] = min(
                self.bank_size.item() + batch_size, self.memory_bank_size
            )

    def update_global_embeddings(
        self,
        vision_features: torch.Tensor,
        text_features: torch.Tensor,
        indices: torch.Tensor,
    ):
        """
        Update global embeddings buffer with current batch.

        Args:
            vision_features: Vision embeddings [batch_size, embed_dim]
            text_features: Text embeddings [batch_size, embed_dim]
            indices: Original indices of examples in the dataset [batch_size]
        """
        if self.sampling_strategy != "global":
            return

        with torch.no_grad():
            # Store embeddings at their corresponding indices
            for i, idx in enumerate(indices):
                if idx < self.dataset_size:
                    self.global_vision_embeddings[idx] = vision_features[i].detach()
                    self.global_text_embeddings[idx] = text_features[i].detach()
                    self.global_indices[idx] = idx

            # Update size and initialized flag
            self.global_size[0] = max(self.global_size.item(), indices.max().item() + 1)
            if not self.global_initialized and indices.numel() > 0:
                self.global_initialized.fill_(True)

    def forward(
        self,
        vision_features: torch.Tensor,
        text_features: torch.Tensor,
        match_ids: Optional[List[str]] = None,
        indices: Optional[torch.Tensor] = None,  # For backward compatibility
        labels: Optional[torch.Tensor] = None,  # For supervised contrastive loss
    ) -> Dict[str, torch.Tensor]:
        """
        Compute contrastive loss between vision and text features based on semantic matching.

        Args:
            vision_features: Vision features [batch_size, vision_dim]
            text_features: Text features [batch_size, text_dim]
            match_ids: IDs that determine which items should match
            indices: Original indices (for backward compatibility)
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

        # Different handling based on sampling strategy
        if self.sampling_strategy == "in-batch":
            # Standard in-batch contrastive loss
            similarity = (
                torch.matmul(vision_features, text_features.T) / self.temperature
            )

            # Add noise to the similarity matrix when batch size is small (< 16)
            if batch_size < 16 and self.training:
                # Scale noise based on batch size - smaller batches get more noise
                noise_scale = max(0.005, 0.02 * (16 - batch_size) / 16)
                similarity_noise = torch.randn_like(similarity) * noise_scale
                similarity = similarity + similarity_noise

            # Use content-based targets instead of diagonal
            loss_v2t = F.cross_entropy(
                similarity, v2t_targets, reduction=self.reduction
            )
            loss_t2v = F.cross_entropy(
                similarity.T, t2i_targets, reduction=self.reduction
            )

        elif self.sampling_strategy == "memory-bank" and self.bank_size.item() > 0:
            # Memory bank contrastive loss
            # Get actual bank size (may be less than max if not filled yet)
            actual_bank_size = int(self.bank_size.item())

            # Calculate similarity with current batch
            batch_similarity = (
                torch.matmul(vision_features, text_features.T) / self.temperature
            )

            # Calculate similarity with memory bank
            bank_vision = self.vision_bank[:actual_bank_size]
            bank_text = self.text_bank[:actual_bank_size]

            # Vision-to-text: current vision vs. memory text
            v2t_bank_sim = torch.matmul(vision_features, bank_text.T) / self.temperature
            # Text-to-vision: current text vs. memory vision
            t2v_bank_sim = torch.matmul(text_features, bank_vision.T) / self.temperature

            # Combine batch and memory bank similarities
            v2t_combined = torch.cat([batch_similarity, v2t_bank_sim], dim=1)
            t2v_combined = torch.cat([batch_similarity.T, t2v_bank_sim], dim=1)

            # Use content-based targets for in-batch part
            # For memory bank part, we don't have match_ids so use regular contrastive loss
            # Targets are the indices matching the current items
            loss_v2t = F.cross_entropy(
                v2t_combined, v2t_targets, reduction=self.reduction
            )
            loss_t2v = F.cross_entropy(
                t2v_combined, t2i_targets, reduction=self.reduction
            )

            # Use batch similarity for metrics
            similarity = batch_similarity

            # Update memory bank for next iteration
            if self.training:
                self.update_memory_bank(vision_features, text_features)

        elif (
            self.sampling_strategy == "global"
            and indices is not None
            and self.global_initialized.item()
        ):
            # Global contrastive loss using all available data
            # Get actual global size
            actual_global_size = int(self.global_size.item())

            # Get current global embeddings
            global_vision = self.global_vision_embeddings[:actual_global_size]
            global_text = self.global_text_embeddings[:actual_global_size]
            global_indices = self.global_indices[:actual_global_size]

            # Calculate global similarities
            v2t_global_sim = (
                torch.matmul(vision_features, global_text.T) / self.temperature
            )
            t2v_global_sim = (
                torch.matmul(text_features, global_vision.T) / self.temperature
            )

            # Create mapping from indices to match targets
            if match_ids is not None:
                # Use match_ids for determining correct matches
                # For each item, find its match in the global bank
                v2t_targets_global = []
                for i, match_id in enumerate(match_ids):
                    # Find all positions in global embeddings with same match_id
                    matching_positions = []
                    for j, idx in enumerate(global_indices):
                        # We need to know the match_id for each global index
                        # This is challenging without modifying the global strategy
                        # For now, use position-based matching as fallback
                        if idx == indices[i]:
                            matching_positions.append(j)

                    if matching_positions:
                        # Pick one matching position randomly
                        import random

                        match_idx = random.choice(matching_positions)
                        v2t_targets_global.append(match_idx)
                    else:
                        # Fallback - use position
                        v2t_targets_global.append(i % actual_global_size)

                # Similarly for t2i direction
                t2v_targets_global = []
                for i, match_id in enumerate(match_ids):
                    matching_positions = []
                    for j, idx in enumerate(global_indices):
                        if idx == indices[i]:
                            matching_positions.append(j)

                    if matching_positions:
                        import random

                        match_idx = random.choice(matching_positions)
                        t2v_targets_global.append(match_idx)
                    else:
                        t2v_targets_global.append(i % actual_global_size)
            else:
                # Use indices for determining correct matches (position-based)
                # Create target mapping: for each idx in the batch, find its position in global indices
                v2t_targets_global = []
                for i, idx in enumerate(indices):
                    matching_positions = (global_indices == idx).nonzero(as_tuple=True)[
                        0
                    ]
                    if matching_positions.numel() > 0:
                        v2t_targets_global.append(matching_positions[0].item())
                    else:
                        # Fallback - use position
                        v2t_targets_global.append(i % actual_global_size)

                # Similarly for t2i direction
                t2v_targets_global = []
                for i, idx in enumerate(indices):
                    matching_positions = (global_indices == idx).nonzero(as_tuple=True)[
                        0
                    ]
                    if matching_positions.numel() > 0:
                        t2v_targets_global.append(matching_positions[0].item())
                    else:
                        t2v_targets_global.append(i % actual_global_size)

            # Convert to tensors
            v2t_targets_global = torch.tensor(
                v2t_targets_global, device=vision_features.device
            )
            t2v_targets_global = torch.tensor(
                t2v_targets_global, device=vision_features.device
            )

            # Compute losses
            loss_v2t = F.cross_entropy(
                v2t_global_sim, v2t_targets_global, reduction=self.reduction
            )
            loss_t2v = F.cross_entropy(
                t2v_global_sim, t2v_targets_global, reduction=self.reduction
            )

            # For similarity metrics, use in-batch similarity
            similarity = (
                torch.matmul(vision_features, text_features.T) / self.temperature
            )

            # Update global embeddings for next iteration
            if self.training:
                self.update_global_embeddings(vision_features, text_features, indices)

        else:
            # Fallback to standard in-batch contrastive loss
            similarity = (
                torch.matmul(vision_features, text_features.T) / self.temperature
            )
            loss_v2t = F.cross_entropy(
                similarity, v2t_targets, reduction=self.reduction
            )
            loss_t2v = F.cross_entropy(
                similarity.T, t2i_targets, reduction=self.reduction
            )

        # Average the bi-directional losses
        loss_infonce = (loss_v2t + loss_t2v) / 2

        # Final loss depends on the specified type
        if self.loss_type == "infonce":
            loss = loss_infonce
        elif self.loss_type == "nt_xent":
            # NT-Xent loss formulation with symmetric loss
            loss = nt_xent_loss(
                vision_features, text_features, self.temperature, self.reduction
            )
        elif self.loss_type == "supervised" and labels is not None:
            # Supervised contrastive loss using class labels
            loss = supervised_contrastive_loss(
                vision_features, text_features, labels, self.temperature, self.reduction
            )
        else:
            loss = loss_infonce  # Default to InfoNCE

        # Calculate accuracy metrics
        with torch.no_grad():
            v2t_pred = torch.argmax(similarity, dim=1)
            t2v_pred = torch.argmax(similarity, dim=0)
            v2t_accuracy = (v2t_pred == v2t_targets).float().mean()
            t2v_accuracy = (t2v_pred == t2i_targets).float().mean()
            accuracy = (v2t_accuracy + t2v_accuracy) / 2

            # Calculate recall@K metrics
            recalls = compute_recall_at_k(
                similarity,
                K=[1, 5, 10],
                v2t_targets=v2t_targets,
                t2i_targets=t2i_targets,
            )

        return {
            "loss": loss,
            "loss_v2t": loss_v2t.item(),
            "loss_t2v": loss_t2v.item(),
            "v2t_accuracy": v2t_accuracy.item(),
            "t2v_accuracy": t2v_accuracy.item(),
            "accuracy": accuracy.item(),
            "recalls": recalls,
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
        """
        super().__init__()
        self.contrastive_weight = contrastive_weight
        self.classification_weight = classification_weight
        self.multimodal_matching_weight = multimodal_matching_weight
        self.temperature = temperature
        self.use_hard_negatives = use_hard_negatives
        self.hard_negative_weight = hard_negative_weight

        # Base contrastive loss
        self.contrastive_loss = ContrastiveLoss(temperature=temperature)

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
