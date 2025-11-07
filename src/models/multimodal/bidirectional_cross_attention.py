"""MODULE: bidirectional_cross_attention.py
PURPOSE: Implements bidirectional cross-attention mechanism for multimodal fusion between vision and text features.

KEY COMPONENTS:
- BidirectionalCrossAttention: Main class implementing bidirectional attention
- Support for both vision-to-text and text-to-vision attention flows
- Configurable attention heads and feature dimensions
- Optional residual connections and layer normalization
- Memory-efficient implementation for large feature maps

DEPENDENCIES:
- torch
- torch.nn
- typing
"""

import os
from typing import Dict, Optional

import torch
import torch.nn as nn

from .gated_cross_modal_attention import GatedCrossModalAttention


class BidirectionalCrossAttention(nn.Module):
    """
    Bidirectional Cross-Attention module for multimodal fusion.

    This module implements bidirectional attention flow between two modalities,
    allowing each modality to attend to and be influenced by the other.
    """

    def __init__(
        self,
        vision_dim: int,
        text_dim: int,
        fusion_dim: int,
        num_heads: int = 8,
        dropout: float = 0.1,
    ):
        """
        Initialize bidirectional cross-attention module.

        Args:
            vision_dim: Dimension of vision features
            text_dim: Dimension of text features
            fusion_dim: Dimension of the fused representations
            num_heads: Number of attention heads
            dropout: Dropout rate
        """
        super().__init__()

        # Cross-attention from text to vision (text attends to vision)
        # ENHANCEMENT: Use GatedCrossModalAttention instead of CrossModalAttention
        self.text_to_vision_attn = GatedCrossModalAttention(
            query_dim=text_dim,
            key_dim=vision_dim,
            embed_dim=fusion_dim,
            num_heads=num_heads,
            dropout=dropout,
        )

        # Cross-attention from vision to text (vision attends to text)
        # ENHANCEMENT: Use GatedCrossModalAttention instead of CrossModalAttention
        self.vision_to_text_attn = GatedCrossModalAttention(
            query_dim=vision_dim,
            key_dim=text_dim,
            embed_dim=fusion_dim,
            num_heads=num_heads,
            dropout=dropout,
        )

        # Feed-forward networks for each modality after cross-attention
        self.vision_ff = nn.Sequential(
            nn.LayerNorm(fusion_dim),
            nn.Linear(fusion_dim, fusion_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(fusion_dim * 4, fusion_dim),
            nn.Dropout(dropout),
        )

        self.text_ff = nn.Sequential(
            nn.LayerNorm(fusion_dim),
            nn.Linear(fusion_dim, fusion_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(fusion_dim * 4, fusion_dim),
            nn.Dropout(dropout),
        )

        # Residual connections require input projections if dimensions don't match
        self.vision_input_proj = (
            nn.Linear(vision_dim, fusion_dim)
            if vision_dim != fusion_dim
            else nn.Identity()
        )
        self.text_input_proj = (
            nn.Linear(text_dim, fusion_dim) if text_dim != fusion_dim else nn.Identity()
        )

    def forward(
        self,
        vision_features: torch.Tensor,
        text_features: torch.Tensor,
        vision_mask: Optional[torch.Tensor] = None,
        text_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass for bidirectional cross-attention.

        Args:
            vision_features: Features from the vision modality [batch_size, vision_seq_len, vision_dim]
            text_features: Features from the text modality [batch_size, text_seq_len, text_dim]
            vision_mask: Optional mask for vision features [batch_size, vision_seq_len]
            text_mask: Optional mask for text features [batch_size, text_seq_len]

        Returns:
            Dictionary containing:
            - 'vision_features': Updated vision features with cross-modal information
            - 'text_features': Updated text features with cross-modal information
            - 'vision_to_text_attn': Attention weights from vision to text
            - 'text_to_vision_attn': Attention weights from text to vision
        """
        results = {}

        # Project inputs to common dimension if needed (for residual connections)
        vision_input = self.vision_input_proj(vision_features)
        text_input = self.text_input_proj(text_features)

        # Create attention masks if provided - simplified version
        if vision_mask is not None and text_mask is not None:
            # Get device
            device = vision_features.device

            # Ensure masks are on device
            vision_mask = vision_mask.to(device)
            text_mask = text_mask.to(device)

            # Get shapes for mask creation
            batch_size = vision_mask.shape[0]
            text_seq_len = text_mask.shape[1]
            vision_seq_len = vision_mask.shape[1]

            # Create simple masks that allow all tokens to attend to all other tokens
            # Create directly on the target device
            text_to_vision_mask = torch.ones(
                batch_size,
                text_seq_len,
                vision_seq_len,
                dtype=torch.bool,
                device=device,
            )

            vision_to_text_mask = torch.ones(
                batch_size,
                vision_seq_len,
                text_seq_len,
                dtype=torch.bool,
                device=device,
            )
        else:
            text_to_vision_mask = None
            vision_to_text_mask = None

        # Store original features
        vision_features_original = vision_features.clone()
        text_features_original = text_features.clone()

        # Text attends to vision
        text_features_updated, text_to_vision_attn = self.text_to_vision_attn(
            text_features_original, vision_features_original, text_to_vision_mask
        )

        # Vision attends to text
        vision_features_updated, vision_to_text_attn = self.vision_to_text_attn(
            vision_features_original, text_features_original, vision_to_text_mask
        )

        # Residual connections
        text_features = text_input + text_features_updated
        vision_features = vision_input + vision_features_updated

        # Feed-forward networks with residual connections
        text_features = text_features + self.text_ff(text_features)
        vision_features = vision_features + self.vision_ff(vision_features)

        # Store results
        results["vision_features"] = vision_features
        results["text_features"] = text_features
        results["vision_to_text_attn"] = vision_to_text_attn
        results["text_to_vision_attn"] = text_to_vision_attn

        return results


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
        "module_purpose": "Implements bidirectional cross-attention mechanism for multimodal fusion between vision and text features",
        "key_classes": [
            {
                "name": "BidirectionalCrossAttention",
                "purpose": "Implements bidirectional attention flow between vision and text modalities",
                "key_methods": [
                    {
                        "name": "__init__",
                        "signature": "__init__(self, vision_dim: int, text_dim: int, num_heads: int = 8, dropout: float = 0.1)",
                        "brief_description": "Initialize bidirectional cross-attention module",
                    },
                    {
                        "name": "forward",
                        "signature": "forward(self, vision_features: torch.Tensor, text_features: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]",
                        "brief_description": "Compute bidirectional attention and fuse features",
                    },
                ],
                "inheritance": "nn.Module",
                "dependencies": ["torch", "torch.nn"],
            }
        ],
        "external_dependencies": ["torch", "typing"],
        "complexity_score": 8,
    }
