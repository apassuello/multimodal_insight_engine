"""MODULE: gated_cross_modal_attention.py
PURPOSE: Implements gated cross-modal attention mechanism for controlled information flow between modalities.

KEY COMPONENTS:
- GatedCrossModalAttention: Main class implementing gated attention
- Learnable gating mechanism for attention control
- Support for selective feature fusion
- Configurable attention heads and dimensions
- Efficient parallel computation

DEPENDENCIES:
- torch
- torch.nn
- typing
"""

import os
import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple, Any

from .cross_modal_attention_base import CrossModalAttention


class GatedCrossModalAttention(nn.Module):
    """
    Enhanced cross-modal attention with gating mechanism.

    This module extends basic cross-attention with a gate that controls
    how much information from one modality flows into the other.
    """

    def __init__(
        self,
        query_dim: int,
        key_dim: int,
        embed_dim: int,
        num_heads: int = 8,
        dropout: float = 0.1,
    ):
        """
        Initialize the gated cross-modal attention module.

        Args:
            query_dim: Dimension of the query modality features
            key_dim: Dimension of the key/value modality features
            embed_dim: Dimension of the cross-attention output
            num_heads: Number of attention heads
            dropout: Dropout rate
        """
        super().__init__()

        # Base cross-attention mechanism
        self.cross_attention = CrossModalAttention(
            query_dim=query_dim,
            key_dim=key_dim,
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
        )

        # Gating mechanism
        # We'll add a projection layer to handle dimension mismatches
        self.query_dim = query_dim
        self.embed_dim = embed_dim

        # Add query projection if dimensions don't match
        self.query_proj_gate = (
            nn.Linear(query_dim, embed_dim) if query_dim != embed_dim else nn.Identity()
        )

        self.gate = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim),
            nn.Sigmoid(),
        )

    def forward(
        self,
        query_features: torch.Tensor,
        key_features: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:  # , torch.Tensor]:
        """
        Forward pass for gated cross-modal attention.

        Args:
            query_features: Features from the query modality [batch_size, query_seq_len, query_dim]
            key_features: Features from the key/value modality [batch_size, key_seq_len, key_dim]
            attention_mask: Optional mask for attention weights [batch_size, query_seq_len, key_seq_len]

        Returns:
            Tuple containing:
            - Updated query features with gated cross-modal information [batch_size, query_seq_len, embed_dim]
            - Attention weights [batch_size, num_heads, query_seq_len, key_seq_len]
            - Gate values [batch_size, query_seq_len, embed_dim]
        """
        # Apply cross-attention
        attended_features, attention_weights = self.cross_attention(
            query_features, key_features, attention_mask
        )

        # Project query features to match embedding dimension using the pre-defined projection
        query_features_projected = self.query_proj_gate(query_features)

        # Simplify the gating process - keep everything on the same device
        # Concatenate inputs on the original device
        gate_input = torch.cat([query_features_projected, attended_features], dim=-1)

        # Apply gate - all operations stay on the same device
        gate_values = self.gate(gate_input)

        # Apply gated residual connection
        output = query_features_projected + gate_values * attended_features

        return output, attention_weights  # , gate_values


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
        "module_purpose": "Implements gated cross-modal attention mechanism for controlled information flow between modalities",
        "key_classes": [
            {
                "name": "GatedCrossModalAttention",
                "purpose": "Implements gated attention mechanism for controlled cross-modal feature fusion",
                "key_methods": [
                    {
                        "name": "__init__",
                        "signature": "__init__(self, source_dim: int, target_dim: int, num_heads: int = 8, dropout: float = 0.1)",
                        "brief_description": "Initialize gated cross-modal attention module",
                    },
                    {
                        "name": "forward",
                        "signature": "forward(self, source_features: torch.Tensor, target_features: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]",
                        "brief_description": "Compute gated cross-modal attention and fuse features",
                    },
                ],
                "inheritance": "nn.Module",
                "dependencies": ["torch", "torch.nn"],
            }
        ],
        "external_dependencies": ["torch", "typing"],
        "complexity_score": 7,
    }
