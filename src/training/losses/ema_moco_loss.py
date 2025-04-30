# src/training/losses/ema_moco_loss.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Any, Union
import logging
import math
import copy

logger = logging.getLogger(__name__)

"""MODULE: ema_moco_loss.py
PURPOSE: Implements Momentum Contrast (MoCo) with Exponential Moving Average (EMA) for efficient multimodal contrastive learning.

KEY COMPONENTS:
- EMAMoCoLoss: Main class implementing MoCo with EMA updates
  - Maintains memory queues of previous batch embeddings for large effective batch size
  - Uses momentum-updated key encoders for feature stability
  - Supports both symmetric (bidirectional) and asymmetric contrasting
  - Handles device transfers and gradient disabling automatically
  - Provides comprehensive metrics for tracking training progress

DEPENDENCIES:
- PyTorch (torch, torch.nn, torch.nn.functional)
- Python standard library (logging, math, copy)

SPECIAL NOTES:
- Critical for Stage 2 of the progressive training approach
- Enables effective learning with limited batch sizes by using queue of negatives
- Improves training stability through slow-moving key encoders
- Requires copies of the encoders which may increase memory usage
"""


class EMAMoCoLoss(nn.Module):
    """
    Implements MoCo (Momentum Contrast) with EMA (Exponential Moving Average) for multimodal learning.

    This loss addresses two key challenges in contrastive learning:
    1. Limited batch size: Uses a memory queue to dramatically increase the number of negatives
    2. Consistency: Uses EMA-updated encoders to maintain consistency in the feature space

    These techniques allow training with much larger effective batch sizes without memory
    constraints, while also improving feature stability and downstream performance.

    Key features:
    - Momentum encoder updates for stable feature encoding
    - Memory queue for large number of negatives
    - Adjustable momentum coefficient and queue size
    - Support for both symmetric and asymmetric queue updates
    - Automatic handling of device transfers and gradient disabling
    """

    def __init__(
        self,
        vision_encoder: nn.Module,
        text_encoder: nn.Module,
        dim: int = 768,
        K: int = 65536,  # Queue size
        m: float = 0.999,  # Momentum coefficient
        T: float = 0.07,  # Temperature
        symmetric: bool = True,  # Whether to use both directions
        update_key_encoder: bool = True,  # Whether to update key encoder
    ):
        """
        Initialize the EMA-MoCo loss.

        Args:
            vision_encoder: Vision encoder model (will be copied for key encoder)
            text_encoder: Text encoder model (will be copied for key encoder)
            dim: Feature dimension
            K: Queue size (number of negative keys)
            m: Momentum coefficient for key encoder update (higher = slower updates)
            T: Temperature parameter for similarity scaling
            symmetric: Whether to use both vision->text and text->vision directions
            update_key_encoder: Whether to update the key encoder (set to False for debugging)
        """
        super().__init__()
        self.K = K
        self.m = m
        self.T = T
        self.symmetric = symmetric
        self.update_key_encoder = update_key_encoder
        self.dim = dim

        # Create the key encoders as copies of the query encoders
        # These will be updated using momentum and won't receive gradient updates
        self.vision_key_encoder = self._copy_and_detach(vision_encoder)
        self.text_key_encoder = self._copy_and_detach(text_encoder)

        # Register queues as buffers (so they're saved with state_dict)
        # Vision-to-text direction: vision queries, text keys
        self.register_buffer("text_queue", torch.randn(dim, K))
        self.text_queue = F.normalize(self.text_queue, dim=0)
        self.register_buffer("text_queue_ptr", torch.zeros(1, dtype=torch.long))

        # Text-to-vision direction: text queries, vision keys
        if self.symmetric:
            self.register_buffer("vision_queue", torch.randn(dim, K))
            self.vision_queue = F.normalize(self.vision_queue, dim=0)
            self.register_buffer("vision_queue_ptr", torch.zeros(1, dtype=torch.long))

    def _copy_and_detach(self, model: nn.Module) -> nn.Module:
        """
        Create a deep copy of a model and detach it from the computation graph.

        Args:
            model: The model to copy

        Returns:
            Detached copy of the model
        """
        model_copy = copy.deepcopy(model)

        # Detach all parameters to ensure no gradient tracking
        for param in model_copy.parameters():
            param.requires_grad = False

        return model_copy

    @torch.no_grad()
    def _momentum_update_key_encoder(self, q_encoder: nn.Module, k_encoder: nn.Module):
        """
        Update key encoder using momentum update.

        Args:
            q_encoder: Query encoder (being trained normally)
            k_encoder: Key encoder (updated via momentum)
        """
        if not self.update_key_encoder:
            return

        # For each parameter in the model, update with momentum
        for param_q, param_k in zip(q_encoder.parameters(), k_encoder.parameters()):
            # Update formula: param_k = m * param_k + (1 - m) * param_q
            param_k.data = param_k.data * self.m + param_q.data * (1.0 - self.m)

    @torch.no_grad()
    def _dequeue_and_enqueue(
        self, keys: torch.Tensor, queue: torch.Tensor, queue_ptr: torch.Tensor
    ):
        """
        Update the queue by dequeuing old keys and enqueuing new ones.

        Args:
            keys: New keys to enqueue [batch_size, dim]
            queue: Queue buffer [dim, K]
            queue_ptr: Queue pointer buffer [1]
        """
        batch_size = keys.shape[0]

        # Move queue and pointer to keys device if needed
        if queue.device != keys.device:
            queue = queue.to(keys.device)
        if queue_ptr.device != keys.device:
            queue_ptr = queue_ptr.to(keys.device)

        ptr = int(queue_ptr.item())

        # Replace the keys at ptr (dequeue and enqueue)
        if ptr + batch_size <= self.K:
            # Simple case: just replace the keys at ptr
            queue[:, ptr : ptr + batch_size] = keys.T
        else:
            # Handle wrap-around case
            remaining = self.K - ptr
            queue[:, ptr:] = keys[:remaining].T
            queue[:, : batch_size - remaining] = keys[remaining:].T

        # Move pointer
        ptr = (ptr + batch_size) % self.K
        queue_ptr[0] = ptr

    def forward(
        self,
        vision_queries: torch.Tensor,
        text_queries: torch.Tensor,
        vision_features: Optional[torch.Tensor] = None,
        text_features: Optional[torch.Tensor] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Compute MoCo loss with EMA-updated encoders and memory queue.

        Args:
            vision_queries: Raw vision inputs to query encoder
            text_queries: Raw text inputs to query encoder
            vision_features: Pre-computed vision features (optional, for efficiency)
            text_features: Pre-computed text features (optional, for efficiency)
            **kwargs: Additional arguments (ignored)

        Returns:
            Dictionary with loss values and additional metrics
        """
        batch_size = (
            vision_queries.shape[0]
            if isinstance(vision_queries, torch.Tensor)
            else vision_features.shape[0]
        )
        device = (
            vision_features.device
            if vision_features is not None
            else vision_queries.device
        )

        # Step 1: Compute query features using the query encoders (if not provided)
        if vision_features is None:
            vision_features = self.vision_encoder(vision_queries)
        if text_features is None:
            text_features = self.text_encoder(text_queries)

        # Normalize features
        vision_features = F.normalize(vision_features, dim=1)
        text_features = F.normalize(text_features, dim=1)

        # Step 2: Compute key features using the key encoders
        with torch.no_grad():  # No grad needed for key encoders
            # Clone queries to avoid modifications to inputs
            if isinstance(vision_queries, torch.Tensor):
                vision_query_k = vision_queries.clone().detach()
                vision_key_features = self.vision_key_encoder(vision_query_k)
                vision_key_features = F.normalize(vision_key_features, dim=1)
            else:
                # If raw queries not available, use the computed features with a warning
                logger.warning(
                    "Using pre-computed vision features for key encoder - "
                    "momentum contrast may be less effective"
                )
                vision_key_features = vision_features.clone().detach()

            if isinstance(text_queries, torch.Tensor):
                text_query_k = text_queries.clone().detach()
                text_key_features = self.text_key_encoder(text_query_k)
                text_key_features = F.normalize(text_key_features, dim=1)
            else:
                logger.warning(
                    "Using pre-computed text features for key encoder - "
                    "momentum contrast may be less effective"
                )
                text_key_features = text_features.clone().detach()

            # Update the queues with the new keys
            self._dequeue_and_enqueue(
                text_key_features, self.text_queue, self.text_queue_ptr
            )
            if self.symmetric:
                self._dequeue_and_enqueue(
                    vision_key_features, self.vision_queue, self.vision_queue_ptr
                )

        # Step 3: Compute vision-to-text logits
        # For each vision query (q), compute similarity with:
        # - The positive text key (k+) for this batch
        # - All negative text keys from the queue (k-)

        # Positives: compare with the paired samples in the current batch
        # Shape: [batch_size, batch_size]
        v2t_pos_logits = torch.matmul(vision_features, text_key_features.T) / self.T

        # Negatives: compare with all samples in the queue
        # Shape: [batch_size, K]
        v2t_neg_logits = (
            torch.matmul(vision_features, self.text_queue.to(device)) / self.T
        )

        # Combine positives and negatives
        # Shape: [batch_size, batch_size + K]
        v2t_logits = torch.cat([v2t_pos_logits, v2t_neg_logits], dim=1)

        # Create targets (positives are on the diagonal)
        # Shape: [batch_size]
        v2t_targets = torch.arange(batch_size, dtype=torch.long, device=device)

        # Compute vision-to-text loss
        v2t_loss = F.cross_entropy(v2t_logits, v2t_targets)

        # Step 4: Compute text-to-vision logits (if symmetric)
        if self.symmetric:
            # Positives: compare with the paired samples in the current batch
            # Shape: [batch_size, batch_size]
            t2v_pos_logits = torch.matmul(text_features, vision_key_features.T) / self.T

            # Negatives: compare with all samples in the queue
            # Shape: [batch_size, K]
            t2v_neg_logits = (
                torch.matmul(text_features, self.vision_queue.to(device)) / self.T
            )

            # Combine positives and negatives
            # Shape: [batch_size, batch_size + K]
            t2v_logits = torch.cat([t2v_pos_logits, t2v_neg_logits], dim=1)

            # Create targets (positives are on the diagonal)
            # Shape: [batch_size]
            t2v_targets = torch.arange(batch_size, dtype=torch.long, device=device)

            # Compute text-to-vision loss
            t2v_loss = F.cross_entropy(t2v_logits, t2v_targets)

            # Average the bidirectional losses
            loss = (v2t_loss + t2v_loss) / 2
        else:
            # Unidirectional loss (vision to text only)
            loss = v2t_loss
            t2v_loss = torch.tensor(0.0, device=device)

        # Step 5: Update key encoders with momentum
        self._momentum_update_key_encoder(self.vision_encoder, self.vision_key_encoder)
        self._momentum_update_key_encoder(self.text_encoder, self.text_key_encoder)

        # Step 6: Compute accuracy metrics
        with torch.no_grad():
            # Vision-to-text accuracy
            v2t_pred = torch.argmax(v2t_logits, dim=1)
            v2t_accuracy = (v2t_pred == v2t_targets).float().mean()

            # Text-to-vision accuracy (if symmetric)
            if self.symmetric:
                t2v_pred = torch.argmax(t2v_logits, dim=1)
                t2v_accuracy = (t2v_pred == t2v_targets).float().mean()
                accuracy = (v2t_accuracy + t2v_accuracy) / 2
            else:
                t2v_accuracy = torch.tensor(0.0, device=device)
                accuracy = v2t_accuracy

        # Return loss and metrics
        return {
            "loss": loss,
            "v2t_loss": v2t_loss.item(),
            "t2v_loss": t2v_loss.item() if self.symmetric else 0.0,
            "v2t_accuracy": v2t_accuracy.item(),
            "t2v_accuracy": t2v_accuracy.item() if self.symmetric else 0.0,
            "accuracy": accuracy.item(),
            "queue_size": self.K,
            "momentum": self.m,
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
        "module_purpose": "Implements MoCo (Momentum Contrast) with EMA updates for large-batch contrastive learning",
        "key_classes": [
            {
                "name": "EMAMoCoLoss",
                "purpose": "Enables efficient contrastive learning with memory queue and momentum-updated encoders",
                "key_methods": [
                    {
                        "name": "forward",
                        "signature": "forward(self, vision_queries: torch.Tensor, text_queries: torch.Tensor, vision_features: Optional[torch.Tensor] = None, text_features: Optional[torch.Tensor] = None, **kwargs) -> Dict[str, Any]",
                        "brief_description": "Computes MoCo loss with EMA-updated encoders and memory queue",
                    },
                    {
                        "name": "_momentum_update_key_encoder",
                        "signature": "_momentum_update_key_encoder(self, q_encoder: nn.Module, k_encoder: nn.Module)",
                        "brief_description": "Updates key encoder using momentum for stable feature space",
                    },
                    {
                        "name": "_dequeue_and_enqueue",
                        "signature": "_dequeue_and_enqueue(self, keys: torch.Tensor, queue: torch.Tensor, queue_ptr: torch.Tensor)",
                        "brief_description": "Updates queue by dequeuing old keys and enqueuing new ones",
                    },
                    {
                        "name": "_copy_and_detach",
                        "signature": "_copy_and_detach(self, model: nn.Module) -> nn.Module",
                        "brief_description": "Creates a deep copy of a model detached from computation graph",
                    },
                ],
                "inheritance": "nn.Module",
                "dependencies": [
                    "torch",
                    "torch.nn",
                    "torch.nn.functional",
                    "logging",
                    "math",
                    "copy",
                ],
            }
        ],
        "external_dependencies": ["torch", "logging", "math", "copy"],
        "complexity_score": 8,  # High complexity due to queue management and momentum updates
    }
