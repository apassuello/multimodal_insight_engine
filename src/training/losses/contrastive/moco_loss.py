"""MoCo-style contrastive loss with memory queue.

Refactored from memory_queue_contrastive_loss.py (405 lines) to eliminate
code duplication. Reduces to ~200 lines by leveraging BaseContrastiveLoss.

Implements Momentum Contrast approach with:
- Memory queue for large-scale negative sampling
- Adaptive temperature based on queue fill level
- Efficient queue updates with circular buffer
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Any
import logging

from ..base import BaseContrastiveLoss

logger = logging.getLogger(__name__)


class MoCoLoss(BaseContrastiveLoss):
    """
    MoCo-style contrastive loss with memory queue for enhanced negative sampling.

    Maintains separate queues for vision and text features to provide
    a large set of consistent negatives across training batches.

    This refactored version uses BaseContrastiveLoss to eliminate duplication
    of normalization, temperature scaling, and similarity computation.
    """

    def __init__(
        self,
        dim: int = 512,
        queue_size: int = 8192,
        temperature: float = 0.07,
        adaptive_temperature: bool = True,
        max_temperature_factor: float = 1.3,
        **kwargs
    ):
        """
        Initialize MoCo loss with memory queue.

        Args:
            dim: Feature dimension
            queue_size: Size of memory queue
            temperature: Base temperature for scaling
            adaptive_temperature: Whether to adjust temperature based on queue fill
            max_temperature_factor: Maximum temperature multiplier when queue is empty
        """
        super().__init__(
            temperature=temperature,
            normalize_features=True,
            **kwargs
        )

        self.queue_size = queue_size
        self.dim = dim
        self.adaptive_temperature = adaptive_temperature
        self.initial_temperature = temperature
        self.max_temperature = temperature * max_temperature_factor

        # Initialize queues (will be properly initialized on first forward)
        self.register_buffer("vision_queue", None)
        self.register_buffer("text_queue", None)
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))
        self.register_buffer("queue_fill_level", torch.zeros(1, dtype=torch.long))

        self.initialized = False

    def initialize_queue(
        self,
        vision_features: torch.Tensor,
        text_features: torch.Tensor
    ):
        """
        Initialize or pre-fill the queue with provided features.

        Args:
            vision_features: Vision features [batch_size, dim]
            text_features: Text features [batch_size, dim]
        """
        if not isinstance(vision_features, torch.Tensor):
            logger.warning("initialize_queue requires tensor inputs")
            return

        device = vision_features.device
        feature_dim = vision_features.shape[1]

        # Initialize queues if needed
        if not self.initialized:
            self._init_queues(feature_dim, device)

        # Normalize and add features
        vision_features = self.normalize(vision_features)
        text_features = self.normalize(text_features)
        self._update_queue(vision_features, text_features)

        fill_level = int(self.queue_fill_level.item())
        logger.info(
            f"Queue initialized: {fill_level}/{self.queue_size} entries "
            f"({fill_level/self.queue_size:.1%} full)"
        )

    def forward(
        self,
        vision_features: torch.Tensor,
        text_features: torch.Tensor,
        match_ids: Optional[List[str]] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Compute MoCo loss using memory queue.

        Args:
            vision_features: Vision embeddings [batch_size, dim]
            text_features: Text embeddings [batch_size, dim]
            match_ids: IDs for semantic matching
            **kwargs: Additional arguments

        Returns:
            Dictionary with loss and metrics
        """
        device = vision_features.device
        feature_dim = vision_features.shape[1]
        batch_size = vision_features.shape[0]

        # Initialize queues on first forward pass
        if not self.initialized or self.vision_queue is None:
            self._init_queues(feature_dim, device)
        elif self.vision_queue.shape[0] != feature_dim:
            logger.info(f"Feature dim changed from {self.vision_queue.shape[0]} to {feature_dim}, reinitializing")
            self._init_queues(feature_dim, device)
        elif self.vision_queue.device != device:
            self._move_to_device(device)

        # Normalize features (uses base class mixin)
        vision_features = self.normalize(vision_features)
        text_features = self.normalize(text_features)

        # Create match matrix
        match_matrix = self._create_match_matrix(batch_size, match_ids, device)

        # Get queue fill level and adaptive temperature
        fill_level = min(int(self.queue_fill_level.item()), self.queue_size)
        fill_ratio = fill_level / self.queue_size
        effective_temp = self._compute_effective_temperature(fill_ratio)

        # Compute similarities
        batch_sim = torch.matmul(vision_features, text_features.T) / effective_temp

        # Queue similarities (detached)
        text_queue = self.text_queue.detach()
        vision_queue = self.vision_queue.detach()

        v2q_sim = torch.matmul(vision_features, text_queue) / effective_temp
        t2q_sim = torch.matmul(text_features, vision_queue) / effective_temp

        # Weight queue similarities based on fill level
        queue_weight = self._compute_queue_weight(fill_ratio)
        v2q_sim = v2q_sim * queue_weight
        t2q_sim = t2q_sim * queue_weight

        # Compute InfoNCE loss with queue
        v2t_loss = self._compute_direction_loss(
            batch_sim, v2q_sim, match_matrix, direction="v2t"
        )
        t2v_loss = self._compute_direction_loss(
            batch_sim.T, t2q_sim, match_matrix.T, direction="t2v"
        )

        # Update queue
        self._update_queue(vision_features, text_features)

        # Compute final loss
        num_pos = match_matrix.sum().item()
        if num_pos > 0:
            v2t_loss = v2t_loss / num_pos
            t2v_loss = t2v_loss / num_pos
        else:
            return {"loss": torch.tensor(0.0, device=device)}

        loss = (v2t_loss + t2v_loss) / 2.0

        return {
            "loss": loss,
            "loss_v2t": v2t_loss.item(),
            "loss_t2v": t2v_loss.item(),
            "temperature": effective_temp,
            "queue_fill": fill_ratio,
        }

    def _init_queues(self, feature_dim: int, device: torch.device):
        """Initialize memory queues with random normalized features."""
        self.register_buffer(
            "vision_queue",
            F.normalize(
                torch.randn(feature_dim, self.queue_size, device=device), dim=0
            ).detach()
        )
        self.register_buffer(
            "text_queue",
            F.normalize(
                torch.randn(feature_dim, self.queue_size, device=device), dim=0
            ).detach()
        )
        self.queue_ptr = self.queue_ptr.to(device)
        self.queue_fill_level = self.queue_fill_level.to(device)
        self.initialized = True
        logger.info(f"Initialized memory queues: {feature_dim}x{self.queue_size}")

    def _move_to_device(self, device: torch.device):
        """Move queues to specified device."""
        self.vision_queue = self.vision_queue.to(device).detach()
        self.text_queue = self.text_queue.to(device).detach()
        self.queue_ptr = self.queue_ptr.to(device)
        self.queue_fill_level = self.queue_fill_level.to(device)

    def _create_match_matrix(
        self,
        batch_size: int,
        match_ids: Optional[List[str]],
        device: torch.device
    ) -> torch.Tensor:
        """Create boolean matrix indicating which pairs should match."""
        if match_ids is None:
            return torch.eye(batch_size, dtype=torch.bool, device=device)

        match_matrix = torch.zeros((batch_size, batch_size), dtype=torch.bool, device=device)
        for i in range(batch_size):
            for j in range(batch_size):
                match_matrix[i, j] = match_ids[i] == match_ids[j]
        return match_matrix

    def _compute_effective_temperature(self, fill_ratio: float) -> float:
        """Compute temperature based on queue fill level."""
        if not self.adaptive_temperature or fill_ratio >= 0.95:
            return self.initial_temperature

        # Higher temperature when queue is empty, lower as it fills
        return self.max_temperature - (self.max_temperature - self.initial_temperature) * fill_ratio

    def _compute_queue_weight(self, fill_ratio: float) -> float:
        """Compute weighting for queue similarities based on fill level."""
        if fill_ratio < 0.2:
            # Very sparse queue - significantly reduce influence
            return fill_ratio * 0.5
        else:
            # Gradually increase weight as queue fills
            return min(1.0, fill_ratio * 1.5)

    def _compute_direction_loss(
        self,
        batch_sim: torch.Tensor,
        queue_sim: torch.Tensor,
        match_matrix: torch.Tensor,
        direction: str
    ) -> torch.Tensor:
        """Compute InfoNCE loss for one direction (v2t or t2v)."""
        batch_size = batch_sim.shape[0]
        total_loss = 0.0

        for i in range(batch_size):
            # Find positive pairs
            pos_idx = torch.where(match_matrix[i])[0]
            if len(pos_idx) == 0:
                continue

            # Positive similarities
            pos_sims = batch_sim[i, pos_idx]

            # Negative similarities (in-batch + queue)
            neg_sims_batch = batch_sim[i, ~match_matrix[i]]
            neg_sims_queue = queue_sim[i]
            all_neg_sims = torch.cat([neg_sims_batch, neg_sims_queue])

            # InfoNCE for each positive
            for pos_sim in pos_sims:
                pos_exp = torch.exp(pos_sim)
                neg_exp_sum = torch.sum(torch.exp(all_neg_sims))
                total_loss += -torch.log(pos_exp / (pos_exp + neg_exp_sum))

        return total_loss

    @torch.no_grad()
    def _update_queue(
        self,
        vision_features: torch.Tensor,
        text_features: torch.Tensor
    ):
        """Update memory queue with new features (circular buffer)."""
        if not self.initialized:
            return

        device = vision_features.device
        batch_size = vision_features.shape[0]
        ptr = int(self.queue_ptr.item())

        # Clone queues
        new_vision_queue = self.vision_queue.clone().detach()
        new_text_queue = self.text_queue.clone().detach()

        # Detach features
        vision_feat = vision_features.detach()
        text_feat = text_features.detach()

        # Update with wrap-around handling
        if ptr + batch_size <= self.queue_size:
            new_vision_queue[:, ptr:ptr + batch_size] = vision_feat.T
            new_text_queue[:, ptr:ptr + batch_size] = text_feat.T
        else:
            # Wrap around
            remaining = self.queue_size - ptr
            new_vision_queue[:, ptr:] = vision_feat[:remaining].T
            new_text_queue[:, ptr:] = text_feat[:remaining].T

            overflow = batch_size - remaining
            new_vision_queue[:, :overflow] = vision_feat[remaining:].T
            new_text_queue[:, :overflow] = text_feat[remaining:].T

        # Register updated buffers
        self.register_buffer("vision_queue", new_vision_queue)
        self.register_buffer("text_queue", new_text_queue)

        # Update pointer
        new_ptr = (ptr + batch_size) % self.queue_size
        self.register_buffer("queue_ptr", torch.tensor([new_ptr], dtype=torch.long, device=device))

        # Update fill level
        current_fill = int(self.queue_fill_level.item())
        new_fill = min(current_fill + batch_size, self.queue_size)
        self.register_buffer(
            "queue_fill_level",
            torch.tensor([new_fill], dtype=torch.long, device=device)
        )
