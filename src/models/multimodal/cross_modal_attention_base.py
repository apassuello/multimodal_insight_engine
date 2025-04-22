import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple


class CrossModalAttention(nn.Module):
    """
    Cross-Modal Attention mechanism to enable interaction between different modalities.

    This module implements a cross-attention mechanism where one modality (e.g., text)
    attends to another modality (e.g., image), allowing information to flow between them.
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
        Initialize cross-modal attention module.

        Args:
            query_dim: Dimension of the query modality features
            key_dim: Dimension of the key/value modality features
            embed_dim: Dimension of the cross-attention output
            num_heads: Number of attention heads
            dropout: Dropout rate
        """
        super().__init__()

        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim**-0.5

        # Projection layers for queries, keys, and values
        self.query_proj = nn.Linear(query_dim, embed_dim)
        self.key_proj = nn.Linear(key_dim, embed_dim)
        self.value_proj = nn.Linear(key_dim, embed_dim)

        # Output projection
        self.output_proj = nn.Linear(embed_dim, embed_dim)

        # Dropout
        self.attn_dropout = nn.Dropout(dropout)
        self.proj_dropout = nn.Dropout(dropout)

        # Layer normalization for pre-norm architecture
        self.norm_query = nn.LayerNorm(query_dim)
        self.norm_key = nn.LayerNorm(key_dim)

    def forward(
        self,
        query_features: torch.Tensor,
        key_features: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass for cross-modal attention.

        Args:
            query_features: Features from the query modality [batch_size, query_seq_len, query_dim]
            key_features: Features from the key/value modality [batch_size, key_seq_len, key_dim]
            attention_mask: Optional mask for attention weights [batch_size, query_seq_len, key_seq_len]

        Returns:
            Tuple containing:
            - Updated query features with cross-modal information [batch_size, query_seq_len, embed_dim]
            - Attention weights [batch_size, num_heads, query_seq_len, key_seq_len]
        """
        batch_size, query_seq_len, _ = query_features.shape
        _, key_seq_len, _ = key_features.shape

        # Apply layer normalization (pre-norm architecture)
        query_features = self.norm_query(query_features)
        key_features = self.norm_key(key_features)

        # Project to multi-head queries, keys, and values
        # [batch_size, seq_len, embed_dim]
        queries = self.query_proj(query_features)
        keys = self.key_proj(key_features)
        values = self.value_proj(key_features)

        # Reshape for multi-head attention
        # [batch_size, seq_len, num_heads, head_dim]
        queries = queries.view(batch_size, query_seq_len, self.num_heads, self.head_dim)
        keys = keys.view(batch_size, key_seq_len, self.num_heads, self.head_dim)
        values = values.view(batch_size, key_seq_len, self.num_heads, self.head_dim)

        # Transpose to [batch_size, num_heads, seq_len, head_dim]
        queries = queries.transpose(1, 2)
        keys = keys.transpose(1, 2)
        values = values.transpose(1, 2)

        # Compute attention scores
        # [batch_size, num_heads, query_seq_len, key_seq_len]
        attention_scores = torch.matmul(queries, keys.transpose(-2, -1)) * self.scale

        # Apply attention mask if provided
        if attention_mask is not None:
            # For MPS compatibility, we'll take a simpler approach to masking
            # Get the dimensions we need to work with
            batch_size, num_heads, query_seq_len, key_seq_len = attention_scores.shape

            # For now, use a simplified approach - we'll just ensure the mask
            # is properly broadcast to the right shape

            # Convert to boolean if it's not already
            if attention_mask.dtype != torch.bool:
                attention_mask = attention_mask.bool()

            # Add the heads dimension if needed and broadcast
            if attention_mask.dim() == 3:  # [batch, query_seq, key_seq]
                # Add heads dimension
                attention_mask = attention_mask.unsqueeze(1)
                # Broadcast to all heads
                attention_mask = attention_mask.expand(-1, num_heads, -1, -1)

            # Apply mask by setting masked positions to -inf
            attention_scores = attention_scores.masked_fill(~attention_mask, -1e9)

        # Apply softmax to get attention weights
        attention_weights = F.softmax(attention_scores, dim=-1)
        attention_weights = self.attn_dropout(attention_weights)

        # Apply attention weights to values
        # [batch_size, num_heads, query_seq_len, head_dim]
        context = torch.matmul(attention_weights, values)

        # Transpose and reshape to [batch_size, query_seq_len, embed_dim]
        context = (
            context.transpose(1, 2).contiguous().view(batch_size, query_seq_len, -1)
        )

        # Apply output projection and dropout
        output = self.output_proj(context)
        output = self.proj_dropout(output)

        return output, attention_weights
