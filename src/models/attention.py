import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple
import inspect


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

        # Apply dropout if specified and in training mode
        if self.dropout is not None and self.training:
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


class MultiHeadAttention(nn.Module):
    """
    Multi-Head Attention mechanism.

    This splits the attention into multiple heads, allowing the model to
    jointly attend to information from different representation subspaces.
    """

    def __init__(self, input_dim: int, num_heads: int, dropout: float = 0.0):
        """
        Initialize the multi-head attention mechanism.

        Args:
            input_dim: Dimension of input features
            num_heads: Number of attention heads
            dropout: Dropout probability
        """
        super().__init__()

        # Ensure input_dim is divisible by num_heads
        assert (
            input_dim % num_heads == 0
        ), f"Input dimension ({input_dim}) must be divisible by the number of heads ({num_heads})"

        self.input_dim = input_dim
        self.num_heads = num_heads
        self.head_dim = input_dim // num_heads

        # Linear projections for query, key, and value
        self.query_projection = nn.Linear(input_dim, input_dim)
        self.key_projection = nn.Linear(input_dim, input_dim)
        self.value_projection = nn.Linear(input_dim, input_dim)

        # Scaled dot-product attention
        self.attention = ScaledDotProductAttention(dropout=dropout)

        # Output projection
        self.output_projection = nn.Linear(input_dim, input_dim)

        # Dropout for output
        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else None

        # Initialize weights
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

    def split_heads(self, x: torch.Tensor) -> torch.Tensor:
        """
        Split the last dimension into (num_heads, head_dim).

        Args:
            x: Tensor of shape [batch_size, seq_length, input_dim]

        Returns:
            Tensor of shape [batch_size, num_heads, seq_length, head_dim]
        """
        batch_size, seq_length, _ = x.size()

        # Reshape to [batch_size, seq_length, num_heads, head_dim]
        x = x.view(batch_size, seq_length, self.num_heads, self.head_dim)

        # Transpose to [batch_size, num_heads, seq_length, head_dim]
        return x.transpose(1, 2)

    def combine_heads(self, x: torch.Tensor) -> torch.Tensor:
        """
        Combine the heads back into original shape.

        Args:
            x: Tensor of shape [batch_size, num_heads, seq_length, head_dim]

        Returns:
            Tensor of shape [batch_size, seq_length, input_dim]
        """
        batch_size, _, seq_length, _ = x.size()

        # Transpose to [batch_size, seq_length, num_heads, head_dim]
        x = x.transpose(1, 2)

        # Reshape to [batch_size, seq_length, input_dim]
        return x.contiguous().view(batch_size, seq_length, self.input_dim)

    def forward(
        self,
        query: torch.Tensor,
        key: Optional[torch.Tensor] = None,
        value: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass of the multi-head attention mechanism.

        Args:
            query: Query tensor of shape [batch_size, num_queries, input_dim]
            key: Optional key tensor of shape [batch_size, num_keys, input_dim]
            value: Optional value tensor of shape [batch_size, num_values, input_dim]
            mask: Optional mask tensor

        Returns:
            Tuple containing:
            - Output tensor after attention
            - Attention weights (averaged over heads for visualization)
        """
        # Handle self-attention case
        if key is None:
            key = query
        if value is None:
            value = query

        batch_size = query.size(0)

        # Linear projections
        q = self.query_projection(query)
        k = self.key_projection(key)
        v = self.value_projection(value)

        # Split heads
        q = self.split_heads(q)  # [batch_size, num_heads, num_queries, head_dim]
        k = self.split_heads(k)  # [batch_size, num_heads, num_keys, head_dim]
        v = self.split_heads(v)  # [batch_size, num_heads, num_values, head_dim]

        # Adjust mask for multi-head attention if provided
        if mask is not None:
            # Add head dimension if mask doesn't have it
            if len(mask.shape) == 3:  # [batch_size, num_queries, num_keys]
                mask = mask.unsqueeze(1)  # [batch_size, 1, num_queries, num_keys]

        # Apply scaled dot-product attention to each head
        context, attention_weights = self.attention(q, k, v, mask)
        # context: [batch_size, num_heads, num_queries, head_dim]
        # attention_weights: [batch_size, num_heads, num_queries, num_keys]

        # Combine heads
        context = self.combine_heads(context)  # [batch_size, num_queries, input_dim]

        # Final linear projection
        output = self.output_projection(context)

        # Apply dropout if specified and in training mode
        if self.dropout is not None and self.training:
            attention_weights = self.dropout(attention_weights)

        # Average attention weights across heads for visualization purposes
        avg_attention_weights = attention_weights.mean(dim=1)

        return output, avg_attention_weights
