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

        # Ensure attention weights are normalized
        attention_weights = F.softmax(attention_weights, dim=-1)

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

    # src/models/attention.py (updates to MultiHeadAttention class)

    def forward(
        self,
        query: torch.Tensor,
        key: Optional[torch.Tensor] = None,
        value: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
        rotary_emb: Optional[nn.Module] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass of the multi-head attention mechanism.

        Args:
            query: Query tensor of shape [batch_size, num_queries, input_dim]
            key: Optional key tensor of shape [batch_size, num_keys, input_dim]
            value: Optional value tensor of shape [batch_size, num_values, input_dim]
            mask: Optional mask tensor
            rotary_emb: Optional rotary position embedding module

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
        q = self.query_projection(query)  # [batch_size, num_queries, input_dim]
        k = self.key_projection(key)      # [batch_size, num_keys, input_dim]
        v = self.value_projection(value)  # [batch_size, num_values, input_dim]

        # Split heads
        q = self.split_heads(q)  # [batch_size, num_heads, num_queries, head_dim]
        k = self.split_heads(k)  # [batch_size, num_heads, num_keys, head_dim]
        v = self.split_heads(v)  # [batch_size, num_heads, num_values, head_dim]

        # Apply rotary positional embeddings if provided
        if rotary_emb is not None:
            q, k = rotary_emb(q, k)

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
            output = self.dropout(output)

        # Average attention weights across heads for visualization purposes
        avg_attention_weights = attention_weights.mean(dim=1)

        return output, avg_attention_weights

class GroupedQueryAttention(nn.Module):
    """Implements Grouped-Query Attention (GQA) mechanism as described in papers like PaLM-2.
    
    This attention mechanism reduces computational and memory costs by sharing key-value heads
    across multiple query heads. Each key-value head serves a group of query heads, making it
    more efficient than standard multi-head attention while maintaining model quality.
    
    Args:
        d_model (int): Total dimension of the model (must be divisible by num_heads)
        num_heads (int): Total number of attention heads (must be divisible by num_key_value_heads)
        num_key_value_heads (int): Number of key/value heads to use (fewer than num_heads)
        
    Attributes:
        head_dim (int): Dimension of each attention head (d_model // num_heads)
        q_proj (nn.Linear): Query projection layer
        k_proj (nn.Linear): Key projection layer (shared across groups)
        v_proj (nn.Linear): Value projection layer (shared across groups)
        out_proj (nn.Linear): Output projection layer
        
    Shape:
        - Input: (batch_size, seq_len, d_model)
        - Output: (batch_size, seq_len, d_model)
        - mask (optional): (batch_size, seq_len, seq_len)
    """
    
    def __init__(self, d_model, num_heads, num_key_value_heads):
        super().__init__()
        assert d_model % num_heads == 0, f"d_model {d_model} must be divisible by num_heads {num_heads}"
        assert num_heads % num_key_value_heads == 0, f"num_heads {num_heads} must be divisible by num_key_value_heads {num_key_value_heads}"
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.num_kv_heads = num_key_value_heads
        self.head_dim = d_model // num_heads
        
        # Query projections (one per head)
        self.q_proj = nn.Linear(d_model, d_model)
        
        # Key/Value projections (shared across groups of heads)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        
        # Output projection
        self.out_proj = nn.Linear(d_model, d_model)
        
    def forward(self, x, mask=None):
        """Forward pass for the Grouped-Query Attention mechanism.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len, d_model)
            mask (torch.Tensor, optional): Attention mask of shape (batch_size, seq_len, seq_len).
                                         Positions with 0 are masked out. Defaults to None.
                                         
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, seq_len, d_model)
        """
        batch_size, seq_len, _ = x.shape
        
        # Project queries, keys, values
        q = self.q_proj(x)  # [batch_size, seq_len, d_model]
        k = self.k_proj(x)  # [batch_size, seq_len, d_model]
        v = self.v_proj(x)  # [batch_size, seq_len, d_model]
        
        # Reshape for multi-head attention
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim)
        k = k.view(batch_size, seq_len, self.num_kv_heads, self.head_dim)
        v = v.view(batch_size, seq_len, self.num_kv_heads, self.head_dim)
        
        # Transpose to [batch_size, num_heads/num_kv_heads, seq_len, head_dim]
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        
        # For each query head, determine which kv head to use
        heads_per_kv = self.num_heads // self.num_kv_heads
        
        # Repeat k and v for each query head in the group
        k = k.repeat_interleave(heads_per_kv, dim=1)  # [batch_size, num_heads, seq_len, head_dim]
        v = v.repeat_interleave(heads_per_kv, dim=1)  # [batch_size, num_heads, seq_len, head_dim]
        
        # Compute attention scores
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        # Apply softmax to get attention weights
        attn_weights = F.softmax(scores, dim=-1)
        
        # Apply attention weights to values
        context = torch.matmul(attn_weights, v)  # [batch_size, num_heads, seq_len, head_dim]
        
        # Reshape and apply output projection
        context = context.transpose(1, 2).contiguous()  # [batch_size, seq_len, num_heads, head_dim]
        context = context.view(batch_size, seq_len, self.d_model)  # [batch_size, seq_len, d_model]
        output = self.out_proj(context)
        
        return output


class ALiBiAttention(nn.Module):
    """Attention Layer with Linear Biases (ALiBi) for enhanced position encoding.
    
    ALiBi replaces traditional positional embeddings with linear biases added to attention scores.
    This approach has been shown to extrapolate better to longer sequences than learned or
    sinusoidal position embeddings. The bias term decreases linearly with distance between
    tokens, with different slopes for each attention head.
    
    Args:
        hidden_size (int): Size of the hidden/embedding dimension. Must be divisible by num_heads.
        num_heads (int): Number of attention heads.
        max_seq_length (int, optional): Maximum sequence length to pre-compute biases for.
            Defaults to 2048.
            
    Attributes:
        head_dim (int): Dimension of each attention head (hidden_size // num_heads)
        q_proj (nn.Linear): Query projection layer
        k_proj (nn.Linear): Key projection layer
        v_proj (nn.Linear): Value projection layer
        out_proj (nn.Linear): Output projection layer
        bias (torch.Tensor): Pre-computed ALiBi attention biases of shape 
            [1, num_heads, max_seq_length, max_seq_length]
            
    References:
        "Train Short, Test Long: Attention with Linear Biases Enables Input Length Extrapolation"
        https://arxiv.org/abs/2108.12409
        
    Shape:
        - Input x: (batch_size, seq_length, hidden_size)
        - Output: (batch_size, seq_length, hidden_size)
        - mask (optional): (batch_size, seq_length, seq_length)
    """
    
    def __init__(self, hidden_size, num_heads, max_seq_length=2048):
        super().__init__()
        
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        
        # Linear projections
        self.q_proj = nn.Linear(hidden_size, hidden_size)
        self.k_proj = nn.Linear(hidden_size, hidden_size)
        self.v_proj = nn.Linear(hidden_size, hidden_size)
        self.out_proj = nn.Linear(hidden_size, hidden_size)
        
        # Create slope matrix for ALiBi
        # Different slope for each head
        slopes = torch.Tensor(self._get_slopes(num_heads))
        
        # Create distance matrix [seq_len, seq_len]
        distances = torch.arange(max_seq_length).unsqueeze(0) - torch.arange(max_seq_length).unsqueeze(1)
        
        # Convert to bias matrix [1, num_heads, seq_len, seq_len]
        self.alibi_bias = slopes.unsqueeze(1).unsqueeze(1) * distances.unsqueeze(0)
        self.register_buffer("bias", self.alibi_bias)
        
    def _get_slopes(self, n):
        """Calculate attention head-specific slopes for ALiBi position biases.
        
        The slopes are calculated using a geometric sequence, where each head gets
        a different slope that decreases by a power of 2. For non-power-of-2 number
        of heads, the slopes are interpolated from the nearest power of 2.
        
        Args:
            n (int): Number of attention heads
            
        Returns:
            list: List of n slopes, one for each attention head
        """
        def get_slopes_power_of_2(n):
            start = 2**(-(2**-(math.log2(n)-3)))
            return [start * 2**(-i) for i in range(n)]
        
        if math.log2(n).is_integer():
            return get_slopes_power_of_2(n)
        else:
            closest_power_of_2 = 2**math.floor(math.log2(n))
            return get_slopes_power_of_2(closest_power_of_2) + \
                   get_slopes_power_of_2(2*closest_power_of_2)[0:n-closest_power_of_2]
    
    def forward(self, x, mask=None):
        """Forward pass of the ALiBi attention layer.
        
        Computes multi-head attention with linear biases added to the attention scores.
        The biases are pre-computed during initialization and depend on the relative
        positions of queries and keys.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_length, hidden_size)
            mask (torch.Tensor, optional): Attention mask of shape (batch_size, seq_length, seq_length).
                                         Values of 0 indicate positions to mask out. Defaults to None.
                                         
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, seq_length, hidden_size)
        """
        batch_size, seq_len, hidden_size = x.shape
        
        # Linear projections
        q = self.q_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim)
        k = self.k_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim)
        v = self.v_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim)
        
        # Transpose for attention [batch, heads, seq_len, head_dim]
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        
        # Calculate attention scores
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        
        # Add ALiBi bias (only need the part up to current sequence length)
        scores = scores + self.bias[:, :seq_len, :seq_len]
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        # Attention weights and context
        attn_weights = F.softmax(scores, dim=-1)
        context = torch.matmul(attn_weights, v)
        
        # Transpose back and reshape
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, hidden_size)
        output = self.out_proj(context)
        
        return output