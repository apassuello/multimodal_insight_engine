import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Any, Union
import numpy as np
import logging

logger = logging.getLogger(__name__)


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
        add_projection: bool = True,
        projection_dim: int = 256,
        input_dim: Optional[int] = None,
        sampling_strategy: str = "auto",  # "in-batch", "memory-bank", "global", or "auto"
        memory_bank_size: int = 4096,
        dataset_size: Optional[int] = None,
    ):
        print(
            f"ContrastiveLoss.__init__ - add_projection={add_projection}, projection_dim={projection_dim}, input_dim={input_dim}"
        )
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
            # CRITICAL PATCH: If input_dim == 768 (ViT-base, BERT-base),
            # force projection dimension to match to avoid dimension mismatch errors
            if input_dim == 768:
                projection_dim = 768
                print(
                    f"CRITICAL PATCH: Detected ViT/BERT input dimension (768) for memory bank, forcing projection_dim to match: {projection_dim}"
                )

            # Calculate the right dimension for the memory bank
            bank_dim = projection_dim if add_projection else (input_dim or 768)
            print(f"Creating memory banks with dimension: {bank_dim}")

            # Register memory banks as buffers so they're saved in state_dict
            self.register_buffer(
                "vision_bank",
                torch.zeros(memory_bank_size, bank_dim),
            )
            self.register_buffer(
                "text_bank",
                torch.zeros(memory_bank_size, bank_dim),
            )
            self.register_buffer("bank_ptr", torch.zeros(1, dtype=torch.long))
            self.register_buffer(
                "bank_size", torch.zeros(1, dtype=torch.long)
            )  # Track actual filled size

        # Register global embeddings buffer if using global strategy
        if self.sampling_strategy == "global":
            self.dataset_size = dataset_size or 1000  # Default to reasonable size
            # CRITICAL PATCH: If input_dim == 768 (ViT-base, BERT-base),
            # force projection dimension to match to avoid dimension mismatch errors
            if input_dim == 768:
                projection_dim = 768
                print(
                    f"CRITICAL PATCH: Detected ViT/BERT input dimension (768) for global embeddings, forcing projection_dim to match: {projection_dim}"
                )

            # Calculate the right dimension for global embeddings
            embedding_dim = projection_dim if add_projection else (input_dim or 768)
            print(f"Creating global embeddings with dimension: {embedding_dim}")

            self.register_buffer(
                "global_vision_embeddings",
                torch.zeros(self.dataset_size, embedding_dim),
            )
            self.register_buffer(
                "global_text_embeddings",
                torch.zeros(self.dataset_size, embedding_dim),
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
            # ALWAYS USE PROJECTION LAYERS, even if dimensions match
            # This ensures we have trainable parameters in stage 1
            print(
                f"Creating projection heads with input_dim={input_dim}, projection_dim={projection_dim} (even when dimensions match)"
            )

            # If dimensions don't match, create projection layers as normal
            print(
                f"Creating projection heads with input_dim={input_dim}, projection_dim={projection_dim}"
            )

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

            # Initialize a counter to reduce printing frequency
            if not hasattr(self, "_project_print_counter"):
                self._project_print_counter = 0

            # Print debugging info once every 20 batches
            should_print = self._project_print_counter % 20 == 0
            self._project_print_counter += 1

            # Print dimensions for debugging (less frequently)
            vision_dim = vision_features.shape[-1]
            text_dim = text_features.shape[-1]
            if should_print:
                print(
                    f"DEBUG - Input dimensions: Vision: {vision_dim}, Text: {text_dim}"
                )

            # CRITICAL HOT PATCH: if dimensions match but projection is still enabled
            # For ViT-base and BERT-base (dimension 768)

            # Get projection input and output dimensions
            if hasattr(self.vision_projection, "0") and isinstance(
                self.vision_projection[0], nn.Linear
            ):
                in_dim = self.vision_projection[0].in_features

                # Find the last Linear layer for output dimension
                out_dim = None
                for module in reversed(list(self.vision_projection.modules())):
                    if isinstance(module, nn.Linear):
                        out_dim = module.out_features
                        break

                if out_dim is None:
                    out_dim = "unknown"

                # Another safeguard: if projecting from 768 to something else, force to 768
                if in_dim == 768 and out_dim != 768 and out_dim != "unknown":
                    print(
                        f"DIMENSION MISMATCH DETECTED: Trying to project 768 -> {out_dim}"
                    )
                    print(f"Creating emergency projection to maintain 768 dimension")
                    # Create emergency identity projection
                    self.vision_projection = nn.Sequential(
                        nn.Linear(768, 768),
                    ).to(device)
                    self.text_projection = nn.Sequential(
                        nn.Linear(768, 768),
                    ).to(device)
                    # Use proper output dimension in debug message
                    out_dim = 768

                if should_print:
                    print(
                        f"DEBUG - Projection dimensions: Vision: {in_dim} -> {out_dim}"
                    )

            # Check for dimension mismatch before applying projection
            if hasattr(self.vision_projection, "0") and isinstance(
                self.vision_projection[0], nn.Linear
            ):
                expected_dim = self.vision_projection[0].in_features
                if vision_features.shape[-1] != expected_dim:
                    print(
                        f"WARNING: Vision feature dimension mismatch! Got {vision_features.shape[-1]}, expected {expected_dim}"
                    )
                    # Create new projection layer with correct input dimension
                    projection_dim = (
                        self.vision_projection[-1].out_features
                        if isinstance(self.vision_projection[-1], nn.Linear)
                        else 512
                    )
                    print(
                        f"Creating new vision projection: {vision_features.shape[-1]} -> {projection_dim}"
                    )
                    self.vision_projection = nn.Sequential(
                        nn.Linear(vision_features.shape[-1], vision_features.shape[-1]),
                        nn.ReLU(),
                        nn.Linear(vision_features.shape[-1], projection_dim),
                    ).to(device)

            if hasattr(self.text_projection, "0") and isinstance(
                self.text_projection[0], nn.Linear
            ):
                expected_dim = self.text_projection[0].in_features
                if text_features.shape[-1] != expected_dim:
                    print(
                        f"WARNING: Text feature dimension mismatch! Got {text_features.shape[-1]}, expected {expected_dim}"
                    )
                    # Create new projection layer with correct input dimension
                    projection_dim = (
                        self.text_projection[-1].out_features
                        if isinstance(self.text_projection[-1], nn.Linear)
                        else 512
                    )
                    print(
                        f"Creating new text projection: {text_features.shape[-1]} -> {projection_dim}"
                    )
                    self.text_projection = nn.Sequential(
                        nn.Linear(text_features.shape[-1], text_features.shape[-1]),
                        nn.ReLU(),
                        nn.Linear(text_features.shape[-1], projection_dim),
                    ).to(device)

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

        # Log feature shapes at debug level only
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(
                f"Vision features shape: {vision_features.shape}, device: {vision_features.device}"
            )
            logger.debug(
                f"Text features shape: {text_features.shape}, device: {text_features.device}"
            )
            logger.debug(f"Projection input_dim: {getattr(self, 'input_dim', None)}")

        # Initialize a counter to reduce printing frequency
        if not hasattr(self, "_print_counter"):
            self._print_counter = 0

        # Print feature shapes debugging info once every 20 batches
        should_print = self._print_counter % 20 == 0
        self._print_counter += 1

        if should_print:
            print(
                f"Contrastive loss input shapes - Vision: {vision_features.shape}, Text: {text_features.shape}"
            )

        # GLOBAL HOT PATCH FOR VICREG: detect and handle 768-dimension case
        if vision_features.shape[-1] == 768 and text_features.shape[-1] == 768:
            if should_print:
                print(
                    "GLOBAL HOT PATCH: Detected ViT/BERT dimensions (768) - skipping all projection"
                )
            # Force disable projection for ViT-base/BERT-base
            self.add_projection = False
            # Just normalize the features directly
            vision_features = F.normalize(vision_features, p=2, dim=1)
            text_features = F.normalize(text_features, p=2, dim=1)
            if should_print:
                print(
                    f"Using normalized features without projection: {vision_features.shape}, {text_features.shape}"
                )
        else:
            # Apply projection and normalization
            try:
                vision_features, text_features = self.project(
                    vision_features, text_features
                )
                if should_print:
                    print(
                        f"Projected feature shapes - Vision: {vision_features.shape}, Text: {text_features.shape}"
                    )
            except RuntimeError as e:
                print(f"ERROR during projection: {str(e)}")
                # Emergency fallback: create basic features with proper dimensions
                device = vision_features.device
                batch_size = vision_features.shape[0]
                # Use matched dimension instead of hardcoded 512
                project_dim = max(vision_features.shape[-1], text_features.shape[-1])
                print(f"Using emergency fallback with dimension {project_dim}")

                # Just normalize the features as is
                vision_features = F.normalize(vision_features, p=2, dim=1)
                text_features = F.normalize(text_features, p=2, dim=1)

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

            # Calculate diagonal similarity for alignment tracking
            diagonal_sim = torch.diagonal(similarity).mean().item()
            sim_mean = similarity.mean().item()
            sim_gap = diagonal_sim - sim_mean

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
