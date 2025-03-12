import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple


class ScaledDotProductAttention(nn.Module):
    """
    Scaled Dot-Product Attention mechanism.

    This is the core attention mechanism used in transformers:
    Attention(Q, K, V) = softmax(QK^T / sqrt(d_k)) * V
    """

    def __init__(self, dropout: float = 0.0):
        """
        Initialize the scaled dot-product attention.

        Args:
            dropout: Dropout probability (0.0 means no dropout)
        """
        super().__init__()
        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else None

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass of the attention mechanism.

        Args:
            query: Query tensor of shape [batch_size, num_queries, query_dim]
            key: Key tensor of shape [batch_size, num_keys, key_dim]
            value: Value tensor of shape [batch_size, num_values, value_dim]
            mask: Optional mask tensor of shape [batch_size, num_queries, num_keys]
                  or [num_queries, num_keys] to mask out certain positions

        Returns:
            Tuple containing:
            - Context vector after attention of shape [batch_size, num_queries, value_dim]
            - Attention weights of shape [batch_size, num_queries, num_keys]
        """
        # Get dimensions
        d_k = query.size(-1)

        # Compute attention scores
        # [batch_size, num_queries, query_dim] x [batch_size, key_dim, num_keys]
        # -> [batch_size, num_queries, num_keys]
        scores = torch.matmul(query, key.transpose(-2, -1))

        # Scale the scores
        scores = scores / math.sqrt(d_k)

        # Apply mask if provided
        if mask is not None:
            # Fill masked positions with large negative value
            scores = scores.masked_fill(mask == 0, -1e9)

        # Apply softmax to get attention weights
        attention_weights = F.softmax(scores, dim=-1)

        # Apply dropout if specified
        if self.dropout is not None:
            attention_weights = self.dropout(attention_weights)

        # Apply attention weights to values
        # [batch_size, num_queries, num_keys] x [batch_size, num_values, value_dim]
        # -> [batch_size, num_queries, value_dim]
        context = torch.matmul(attention_weights, value)

        return context, attention_weights


class SimpleAttention(nn.Module):
    """
    A simple attention mechanism based on scaled dot-product attention.

    This class adds projections for query, key, and value vectors.
    """

    def __init__(
        self, input_dim: int, attention_dim: Optional[int] = None, dropout: float = 0.0
    ):
        """
        Initialize the attention mechanism.

        Args:
            input_dim: Dimension of input features
            attention_dim: Dimension of attention space (defaults to input_dim if None)
            dropout: Dropout probability
        """
        super().__init__()

        # Set default attention dimension if not provided
        self.attention_dim = attention_dim if attention_dim is not None else input_dim

        # Linear projections for query, key, and value
        self.query_projection = nn.Linear(input_dim, self.attention_dim)
        self.key_projection = nn.Linear(input_dim, self.attention_dim)
        self.value_projection = nn.Linear(input_dim, self.attention_dim)

        # Scaled dot-product attention
        self.attention = ScaledDotProductAttention(dropout=dropout)

        # Output projection
        self.output_projection = nn.Linear(self.attention_dim, input_dim)

        # Initialize weights (similar to the approach in your existing layers)
        self._init_weights()

    def _init_weights(self):
        """Initialize the weights using Xavier uniform initialization."""
        for module in [
            self.query_projection,
            self.key_projection,
            self.value_projection,
            self.output_projection,
        ]:
            nn.init.xavier_uniform_(module.weight)
            nn.init.zeros_(module.bias)

    def forward(
        self,
        query: torch.Tensor,
        key: Optional[torch.Tensor] = None,
        value: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass of the attention mechanism.

        If key and value are not provided, self-attention is performed where
        query is used as key and value.

        Args:
            query: Query tensor of shape [batch_size, num_queries, input_dim]
            key: Optional key tensor of shape [batch_size, num_keys, input_dim]
            value: Optional value tensor of shape [batch_size, num_values, input_dim]
            mask: Optional mask tensor

        Returns:
            Tuple containing:
            - Output tensor after attention
            - Attention weights
        """
        # Handle self-attention case
        if key is None:
            key = query
        if value is None:
            value = query

        # Project inputs to attention space
        q = self.query_projection(query)
        k = self.key_projection(key)
        v = self.value_projection(value)

        # Apply scaled dot-product attention
        context, attention_weights = self.attention(q, k, v, mask)

        # Project back to input dimension
        output = self.output_projection(context)

        return output, attention_weights
