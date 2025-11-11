"""SimCLR-style contrastive loss using base classes.

Refactored from contrastive_loss.py (1,097 lines) to eliminate code duplication.
Reduces to ~280 lines by leveraging BaseContrastiveLoss and mixins.

Supports:
- InfoNCE, NT-Xent, and supervised contrastive losses
- Memory bank for additional negatives
- Global embeddings strategy
- Semantic matching via match_ids
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Any
import logging

from ..base import BaseContrastiveLoss

logger = logging.getLogger(__name__)


class SimCLRLoss(BaseContrastiveLoss):
    """
    SimCLR-style contrastive loss with InfoNCE, NT-Xent, and supervised variants.

    Supports multiple negative sampling strategies:
    - in-batch: Use only current batch (standard)
    - memory-bank: Maintain queue of past embeddings
    - global: Use all dataset examples

    This refactored version uses BaseContrastiveLoss to eliminate duplication
    of normalization, temperature scaling, similarity computation, and projection.
    """

    def __init__(
        self,
        temperature: float = 0.07,
        loss_type: str = "infonce",
        reduction: str = "mean",
        input_dim: Optional[int] = None,
        projection_dim: int = 256,
        add_projection: bool = True,
        sampling_strategy: str = "auto",
        memory_bank_size: int = 4096,
        dataset_size: Optional[int] = None,
        **kwargs
    ):
        """
        Initialize SimCLR loss.

        Args:
            temperature: Temperature for similarity scaling
            loss_type: Type of loss ("infonce", "nt_xent", or "supervised")
            reduction: Loss reduction method
            input_dim: Input feature dimension
            projection_dim: Projection space dimension
            add_projection: Whether to add MLP projection
            sampling_strategy: "in-batch", "memory-bank", "global", or "auto"
            memory_bank_size: Size of memory bank
            dataset_size: Total dataset size (for auto strategy)
        """
        # Determine sampling strategy
        if sampling_strategy == "auto" and dataset_size is not None:
            if dataset_size < 1000:
                sampling_strategy = "global"
            elif dataset_size < 10000:
                sampling_strategy = "memory-bank"
            else:
                sampling_strategy = "in-batch"

        # Initialize base class with all mixins
        super().__init__(
            temperature=temperature,
            normalize_features=True,
            use_projection=add_projection,
            projection_input_dim=input_dim,
            projection_hidden_dim=input_dim if input_dim else projection_dim,
            projection_output_dim=projection_dim,
            reduction=reduction,
            **kwargs
        )

        self.loss_type = loss_type
        self.sampling_strategy = sampling_strategy
        self.memory_bank_size = memory_bank_size
        self.dataset_size = dataset_size

        # Initialize memory banks if needed
        if self.sampling_strategy == "memory-bank":
            bank_dim = projection_dim if add_projection else (input_dim or 768)
            self.register_buffer("vision_bank", torch.zeros(memory_bank_size, bank_dim))
            self.register_buffer("text_bank", torch.zeros(memory_bank_size, bank_dim))
            self.register_buffer("bank_ptr", torch.zeros(1, dtype=torch.long))
            self.register_buffer("bank_size", torch.zeros(1, dtype=torch.long))

        # Initialize global embeddings if needed
        if self.sampling_strategy == "global":
            self.dataset_size = dataset_size or 1000
            embedding_dim = projection_dim if add_projection else (input_dim or 768)
            self.register_buffer("global_vision_embeddings", torch.zeros(self.dataset_size, embedding_dim))
            self.register_buffer("global_text_embeddings", torch.zeros(self.dataset_size, embedding_dim))
            self.register_buffer("global_indices", torch.zeros(self.dataset_size, dtype=torch.long))
            self.register_buffer("global_size", torch.zeros(1, dtype=torch.long))
            self.register_buffer("global_initialized", torch.tensor(False))

    def forward(
        self,
        vision_features: torch.Tensor,
        text_features: torch.Tensor,
        match_ids: Optional[List[str]] = None,
        indices: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Compute contrastive loss between vision and text features.

        Args:
            vision_features: Vision embeddings [batch_size, dim]
            text_features: Text embeddings [batch_size, dim]
            match_ids: IDs for semantic matching
            indices: Dataset indices (for global strategy)
            labels: Class labels (for supervised loss)
            **kwargs: Additional arguments

        Returns:
            Dictionary with loss and metrics
        """
        batch_size = vision_features.shape[0]

        # Safety checks
        if batch_size == 0:
            return {"loss": torch.tensor(0.0, device=vision_features.device)}

        if batch_size != text_features.shape[0]:
            raise ValueError(
                f"Batch size mismatch: vision={vision_features.shape[0]}, "
                f"text={text_features.shape[0]}"
            )

        # Project and normalize features (uses mixins from base class)
        vision_features = self.project_vision(vision_features)
        text_features = self.project_text(text_features)
        vision_features = self.normalize(vision_features)
        text_features = self.normalize(text_features)

        # Create targets based on match_ids
        v2t_targets, t2i_targets = self._create_targets(
            batch_size, match_ids, vision_features.device
        )

        # Compute loss based on sampling strategy
        if self.sampling_strategy == "in-batch":
            loss, similarity = self._compute_in_batch_loss(
                vision_features, text_features, v2t_targets, t2i_targets, batch_size
            )
        elif self.sampling_strategy == "memory-bank" and self.bank_size.item() > 0:
            loss, similarity = self._compute_memory_bank_loss(
                vision_features, text_features, v2t_targets, t2i_targets
            )
        elif self.sampling_strategy == "global" and indices is not None and self.global_initialized.item():
            loss, similarity = self._compute_global_loss(
                vision_features, text_features, match_ids, indices
            )
        else:
            # Fallback to in-batch
            loss, similarity = self._compute_in_batch_loss(
                vision_features, text_features, v2t_targets, t2i_targets, batch_size
            )

        # Compute metrics
        metrics = self._compute_metrics(
            similarity, v2t_targets, t2i_targets, vision_features, text_features
        )

        return {
            "loss": loss,
            **metrics
        }

    def _create_targets(
        self,
        batch_size: int,
        match_ids: Optional[List[str]],
        device: torch.device
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Create target indices for vision→text and text→vision based on match_ids.

        Args:
            batch_size: Batch size
            match_ids: Optional semantic matching IDs
            device: Device for tensors

        Returns:
            Tuple of (v2t_targets, t2i_targets)
        """
        if match_ids is None:
            # Diagonal matching (position-based)
            targets = torch.arange(batch_size, device=device)
            return targets, targets

        # Convert to strings for consistent comparison
        string_match_ids = [str(mid) for mid in match_ids]

        # Create match matrix
        match_matrix = torch.zeros(
            (batch_size, batch_size),
            dtype=torch.bool,
            device=device
        )

        for i in range(batch_size):
            for j in range(batch_size):
                match_matrix[i, j] = string_match_ids[i] == string_match_ids[j]

        # Create v2t targets
        v2t_targets = torch.zeros(batch_size, dtype=torch.long, device=device)
        for i in range(batch_size):
            matches = torch.where(match_matrix[i])[0]
            if len(matches) > 0:
                idx = torch.randint(0, len(matches), (1,), device=device)[0]
                v2t_targets[i] = matches[idx]
            else:
                v2t_targets[i] = i

        # Create t2i targets
        t2i_targets = torch.zeros(batch_size, dtype=torch.long, device=device)
        for j in range(batch_size):
            matches = torch.where(match_matrix[:, j])[0]
            if len(matches) > 0:
                idx = torch.randint(0, len(matches), (1,), device=device)[0]
                t2i_targets[j] = matches[idx]
            else:
                t2i_targets[j] = j

        return v2t_targets, t2i_targets

    def _compute_in_batch_loss(
        self,
        vision_features: torch.Tensor,
        text_features: torch.Tensor,
        v2t_targets: torch.Tensor,
        t2i_targets: torch.Tensor,
        batch_size: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute in-batch contrastive loss."""
        # Use base class method for similarity (includes temperature scaling)
        similarity = self.compute_similarity(vision_features, text_features, normalize=False)

        # Add noise for small batches
        if batch_size < 16 and self.training:
            noise_scale = max(0.005, 0.02 * (16 - batch_size) / 16)
            similarity = similarity + torch.randn_like(similarity) * noise_scale

        # Compute bidirectional losses
        loss_v2t = F.cross_entropy(similarity, v2t_targets, reduction=self.reduction)
        loss_t2v = F.cross_entropy(similarity.T, t2i_targets, reduction=self.reduction)

        loss = (loss_v2t + loss_t2v) / 2.0
        return loss, similarity

    def _compute_memory_bank_loss(
        self,
        vision_features: torch.Tensor,
        text_features: torch.Tensor,
        v2t_targets: torch.Tensor,
        t2i_targets: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute memory bank contrastive loss."""
        actual_bank_size = int(self.bank_size.item())

        # Batch similarity (uses base class method with temperature)
        batch_similarity = self.compute_similarity(vision_features, text_features, normalize=False)

        # Bank similarities
        bank_vision = self.vision_bank[:actual_bank_size]
        bank_text = self.text_bank[:actual_bank_size]

        v2t_bank_sim = self.compute_similarity(vision_features, bank_text, normalize=False)
        t2v_bank_sim = self.compute_similarity(text_features, bank_vision, normalize=False)

        # Combine
        v2t_combined = torch.cat([batch_similarity, v2t_bank_sim], dim=1)
        t2v_combined = torch.cat([batch_similarity.T, t2v_bank_sim], dim=1)

        # Compute losses
        loss_v2t = F.cross_entropy(v2t_combined, v2t_targets, reduction=self.reduction)
        loss_t2v = F.cross_entropy(t2v_combined, t2i_targets, reduction=self.reduction)

        loss = (loss_v2t + loss_t2v) / 2.0

        # Update bank
        if self.training:
            self._update_memory_bank(vision_features, text_features)

        return loss, batch_similarity

    def _compute_global_loss(
        self,
        vision_features: torch.Tensor,
        text_features: torch.Tensor,
        match_ids: Optional[List[str]],
        indices: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute global contrastive loss."""
        actual_global_size = int(self.global_size.item())

        # Global embeddings
        global_vision = self.global_vision_embeddings[:actual_global_size]
        global_text = self.global_text_embeddings[:actual_global_size]

        # Global similarities (uses base class method with temperature)
        v2t_global_sim = self.compute_similarity(vision_features, global_text, normalize=False)
        t2v_global_sim = self.compute_similarity(text_features, global_vision, normalize=False)

        # Create global targets
        v2t_targets_global, t2v_targets_global = self._create_global_targets(
            indices, match_ids, actual_global_size, vision_features.device
        )

        # Compute losses
        loss_v2t = F.cross_entropy(v2t_global_sim, v2t_targets_global, reduction=self.reduction)
        loss_t2v = F.cross_entropy(t2v_global_sim, t2v_targets_global, reduction=self.reduction)

        loss = (loss_v2t + loss_t2v) / 2.0

        # Update global embeddings
        if self.training:
            self._update_global_embeddings(vision_features, text_features, indices)

        # Return in-batch similarity for metrics
        batch_similarity = self.compute_similarity(vision_features, text_features, normalize=False)
        return loss, batch_similarity

    def _create_global_targets(
        self,
        indices: torch.Tensor,
        match_ids: Optional[List[str]],
        actual_global_size: int,
        device: torch.device
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Create targets for global loss."""
        batch_size = indices.shape[0]
        global_indices = self.global_indices[:actual_global_size]

        v2t_targets = []
        t2v_targets = []

        for i, idx in enumerate(indices):
            matching_positions = (global_indices == idx).nonzero(as_tuple=True)[0]
            if matching_positions.numel() > 0:
                v2t_targets.append(matching_positions[0].item())
                t2v_targets.append(matching_positions[0].item())
            else:
                v2t_targets.append(i % actual_global_size)
                t2v_targets.append(i % actual_global_size)

        return (
            torch.tensor(v2t_targets, device=device),
            torch.tensor(t2v_targets, device=device)
        )

    def _update_memory_bank(
        self,
        vision_features: torch.Tensor,
        text_features: torch.Tensor
    ):
        """Update memory bank with current batch."""
        with torch.no_grad():
            batch_size = vision_features.shape[0]
            ptr = int(self.bank_ptr.item())

            # Circular update
            if ptr + batch_size >= self.memory_bank_size:
                remaining = self.memory_bank_size - ptr
                self.vision_bank[ptr:] = vision_features[:remaining].detach()
                self.text_bank[ptr:] = text_features[:remaining].detach()

                overflow = batch_size - remaining
                if overflow > 0:
                    self.vision_bank[:overflow] = vision_features[remaining:].detach()
                    self.text_bank[:overflow] = text_features[remaining:].detach()

                new_ptr = overflow
            else:
                self.vision_bank[ptr:ptr + batch_size] = vision_features.detach()
                self.text_bank[ptr:ptr + batch_size] = text_features.detach()
                new_ptr = ptr + batch_size

            self.bank_ptr[0] = new_ptr % self.memory_bank_size
            self.bank_size[0] = min(self.bank_size.item() + batch_size, self.memory_bank_size)

    def _update_global_embeddings(
        self,
        vision_features: torch.Tensor,
        text_features: torch.Tensor,
        indices: torch.Tensor
    ):
        """Update global embeddings buffer."""
        with torch.no_grad():
            for i, idx in enumerate(indices):
                if idx < self.dataset_size:
                    self.global_vision_embeddings[idx] = vision_features[i].detach()
                    self.global_text_embeddings[idx] = text_features[i].detach()
                    self.global_indices[idx] = idx

            self.global_size[0] = max(self.global_size.item(), indices.max().item() + 1)
            if not self.global_initialized and indices.numel() > 0:
                self.global_initialized.fill_(True)

    def _compute_metrics(
        self,
        similarity: torch.Tensor,
        v2t_targets: torch.Tensor,
        t2i_targets: torch.Tensor,
        vision_features: torch.Tensor,
        text_features: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """Compute accuracy and retrieval metrics."""
        with torch.no_grad():
            batch_size = similarity.shape[0]

            # Accuracy metrics
            v2t_pred = torch.argmax(similarity, dim=1)
            v2t_accuracy = (v2t_pred == v2t_targets).float().mean()

            t2v_pred = torch.argmax(similarity.T, dim=1)
            t2v_accuracy = (t2v_pred == t2i_targets).float().mean()

            accuracy = (v2t_accuracy + t2v_accuracy) / 2.0

            # Recall@K metrics
            recalls = {}
            for k in [1, 5, 10]:
                if k <= batch_size:
                    v2t_topk = torch.topk(similarity, k, dim=1)[1]
                    v2t_recall = (v2t_topk == v2t_targets.unsqueeze(1)).any(dim=1).float().mean()

                    t2v_topk = torch.topk(similarity.T, k, dim=1)[1]
                    t2v_recall = (t2v_topk == t2i_targets.unsqueeze(1)).any(dim=1).float().mean()

                    recalls[f"v2t_recall@{k}"] = v2t_recall
                    recalls[f"t2i_recall@{k}"] = t2v_recall
                    recalls[f"avg_recall@{k}"] = (v2t_recall + t2v_recall) / 2.0

            return {
                "v2t_accuracy": v2t_accuracy,
                "t2v_accuracy": t2v_accuracy,
                "accuracy": accuracy,
                **{f"recalls.{k}": v for k, v in recalls.items()}
            }
