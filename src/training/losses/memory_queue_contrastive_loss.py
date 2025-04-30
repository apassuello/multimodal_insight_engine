"""MODULE: memory_queue_contrastive_loss.py
PURPOSE: Implements contrastive loss with memory queue for maintaining large sets of negative examples.

KEY COMPONENTS:
- MemoryQueueContrastiveLoss: Main class for memory-based contrastive loss
- Efficient memory queue management
- Support for momentum encoders
- Dynamic queue updates
- Configurable queue size and momentum

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
        self.max_temperature = (
            temperature * 1.3
        )  # Higher temperature for early training

    def initialize_queue(self, vision_features, text_features):
        """
        Initialize the queue with the provided features.
        This is useful for pre-filling the queue when transitioning between training stages.

        Args:
            vision_features: Vision features [batch_size, dim]
            text_features: Text features [batch_size, dim]
        """
        if not isinstance(vision_features, torch.Tensor) or not isinstance(
            text_features, torch.Tensor
        ):
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
            print(
                f"Initialized memory queues with dimension {feature_dim}x{self.queue_size}"
            )

        # Normalize the features if they're not already normalized
        vision_features = F.normalize(vision_features, dim=1)
        text_features = F.normalize(text_features, dim=1)

        # Call update queue to add these features
        self._update_queue(vision_features, text_features)

        # Print initialization info
        fill_level = int(self.queue_fill_level.item())
        print(
            f"Queue initialized with {fill_level}/{self.queue_size} entries ({fill_level/self.queue_size:.1%} full)"
        )

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
        effective_temperature = (
            self.max_temperature
            - (self.max_temperature - self.initial_temperature) * fill_ratio
        )
        if fill_ratio >= 0.95:  # Once queue is nearly full, use base temperature
            effective_temperature = self.initial_temperature

        # Apply temperature to similarities
        # CRITICAL ENHANCEMENT: Compare with queue embeddings too
        # Current batch similarities
        batch_similarities = (
            torch.matmul(vision_features, text_features.T) / effective_temperature
        )

        # Make sure queues are detached from computation graph
        text_queue_detached = self.text_queue.detach()
        vision_queue_detached = self.vision_queue.detach()

        # Vision-to-queue text similarities
        v2q_similarities = (
            torch.matmul(vision_features, text_queue_detached) / effective_temperature
        )

        # Text-to-queue vision similarities
        t2q_similarities = (
            torch.matmul(text_features, vision_queue_detached) / effective_temperature
        )

        # Apply weighting to queue similarities based on fill level
        # For sparse/empty queues, reduce their influence to avoid noisy gradients
        queue_weight = min(
            1.0, fill_ratio * 1.5
        )  # Gradually increase queue weight as it fills
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
            print(
                f"Memory queue: {fill_level}/{self.queue_size} filled ({fill_ratio:.2f}), "
                f"using temperature={effective_temperature:.3f}"
            )

        return {
            "loss": loss,
            "loss_v2t": v2t_loss.item(),
            "loss_t2v": t2v_loss.item(),
            "temperature": effective_temperature,
            "queue_fill": fill_ratio,
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
        self.register_buffer(
            "queue_fill_level",
            torch.tensor([new_fill], dtype=torch.long, device=device),
        )


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
        "module_purpose": "Implements contrastive loss with memory queue for maintaining large sets of negative examples",
        "key_classes": [
            {
                "name": "MemoryQueueContrastiveLoss",
                "purpose": "Contrastive loss with memory queue for enhanced negative sampling",
                "key_methods": [
                    {
                        "name": "__init__",
                        "signature": "__init__(self, dim: int, K: int = 65536, m: float = 0.999, T: float = 0.07)",
                        "brief_description": "Initialize loss with queue parameters",
                    },
                    {
                        "name": "forward",
                        "signature": "forward(self, q: torch.Tensor, k: torch.Tensor) -> Dict[str, torch.Tensor]",
                        "brief_description": "Compute loss using memory queue",
                    },
                    {
                        "name": "_dequeue_and_enqueue",
                        "signature": "_dequeue_and_enqueue(self, keys: torch.Tensor) -> None",
                        "brief_description": "Update memory queue with new keys",
                    },
                ],
                "inheritance": "nn.Module",
                "dependencies": ["torch", "torch.nn"],
            }
        ],
        "external_dependencies": ["torch", "typing"],
        "complexity_score": 7,
    }
