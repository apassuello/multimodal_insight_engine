# src/models/transformer.py (new file)
import torch
import torch.nn as nn
from typing import Optional, Dict, Any

from .base_model import BaseModel
from .attention import MultiHeadAttention
from .layers import FeedForwardBlock
from .positional import PositionalEncoding, RotaryPositionEncoding

class TransformerEncoderLayer(nn.Module):
    """
    A single transformer encoder layer.
    
    This implements one layer of the transformer encoder as described in
    "Attention is All You Need" (Vaswani et al., 2017).
    Each layer consists of multi-head self-attention followed by a
    position-wise feed-forward network, with residual connections and layer normalization.
    """
    
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        dropout: float = 0.1,
        use_rotary_embeddings: bool = False,
    ):
        """
        Initialize the transformer encoder layer.
        
        Args:
            d_model: Dimension of model embeddings
            num_heads: Number of attention heads
            d_ff: Dimension of feed-forward network
            dropout: Dropout probability
            use_rotary_embeddings: Whether to use rotary positional embeddings
        """
        super().__init__()
        
        self.use_rotary_embeddings = use_rotary_embeddings
        
        # Ensure d_model is divisible by num_heads
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        
        # Multi-head attention layer
        self.self_attn = MultiHeadAttention(
            input_dim=d_model,
            num_heads=num_heads,
            dropout=dropout
        )
        
        # Rotary embeddings if specified (applied within attention)
        if use_rotary_embeddings:
            self.rotary_emb = RotaryPositionEncoding(
                d_model=d_model // num_heads,  # head_dim
                max_seq_length=5000
            )
        
        # Feed-forward block
        self.feed_forward = FeedForwardBlock(
            input_dim=d_model,
            hidden_dim=d_ff,
            output_dim=d_model,
            activation="relu",
            dropout=dropout,
            use_layer_norm=False,
        )
        
        # Layer normalization layers
        # Pre-norm architecture (applied before each block)
        # This is a variation from the original paper but has been shown to be more stable
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        # Dropout layers
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
    
    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass for the encoder layer.
        
        Args:
            x: Input tensor of shape [batch_size, seq_length, d_model]
            mask: Optional mask tensor of shape [batch_size, seq_length, seq_length]
            
        Returns:
            Output tensor of shape [batch_size, seq_length, d_model]
        """
        # Self-attention block with residual connection
        # Apply layer norm before attention (pre-norm architecture)
        norm_x = self.norm1(x)
        
        if self.use_rotary_embeddings and hasattr(self, 'rotary_emb'):
            # When using rotary embeddings, we need to apply them before multi-head attention
            # This requires modifying the inner workings of the attention mechanism
            # Here we'll assume the MultiHeadAttention module has been adapted to handle rotary embeddings
            # In practice, you'd need to modify the MultiHeadAttention class to support this
            attn_output, _ = self.self_attn(
                norm_x, norm_x, norm_x, mask=mask, rotary_emb=self.rotary_emb
            )
        else:
            attn_output, _ = self.self_attn(norm_x, norm_x, norm_x, mask=mask)
        
        # Apply dropout and residual connection
        x = x + self.dropout1(attn_output)
        
        # Feed-forward block with residual connection
        # Apply layer norm before feed-forward (pre-norm architecture)
        norm_x = self.norm2(x)
        ff_output = self.feed_forward(norm_x)
        
        # Apply dropout and residual connection
        x = x + self.dropout2(ff_output)
        
        return x

class TransformerEncoder(nn.Module):
    """
    Transformer encoder consisting of multiple encoder layers.
    
    This implements the encoder part of the transformer as described in
    "Attention is All You Need" (Vaswani et al., 2017).
    """
    
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        num_layers: int,
        d_ff: int,
        dropout: float = 0.1,
        max_seq_length: int = 5000,
        positional_encoding: str = "sinusoidal",
    ):
        """
        Initialize the transformer encoder.
        
        Args:
            d_model: Dimension of model embeddings
            num_heads: Number of attention heads
            num_layers: Number of encoder layers
            d_ff: Dimension of feed-forward network
            dropout: Dropout probability
            max_seq_length: Maximum sequence length
            positional_encoding: Type of positional encoding to use
                ("sinusoidal", "learned", or "rotary")
        """
        super().__init__()
        
        self.d_model = d_model
        self.positional_encoding_type = positional_encoding
        
        # Input embedding layer
        # Note: In practice, you'd have a separate embedding layer for converting tokens to vectors
        # We'll assume that's done before passing to this encoder
        
        # Positional encoding
        if positional_encoding == "rotary":
            self.use_rotary = True
            # No separate positional encoding layer for rotary embeddings
            # Instead, rotary embeddings are applied within each attention layer
        else:
            self.use_rotary = False
            self.positional_encoding = PositionalEncoding(
                d_model=d_model,
                max_seq_length=max_seq_length,
                dropout=dropout,
                encoding_type="sinusoidal" if positional_encoding == "sinusoidal" else "learned"
            )
        
        # Stack of encoder layers
        self.layers = nn.ModuleList([
            TransformerEncoderLayer(
                d_model=d_model,
                num_heads=num_heads,
                d_ff=d_ff,
                dropout=dropout,
                use_rotary_embeddings=(positional_encoding == "rotary")
            )
            for _ in range(num_layers)
        ])
        
        # Final layer normalization
        self.norm = nn.LayerNorm(d_model)
    
    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass for the encoder.
        
        Args:
            x: Input tensor of shape [batch_size, seq_length, d_model]
            mask: Optional mask tensor of shape [batch_size, seq_length, seq_length]
            
        Returns:
            Output tensor of shape [batch_size, seq_length, d_model]
        """
        # Apply positional encoding if not using rotary embeddings
        if not self.use_rotary:
            x = self.positional_encoding(x)
        
        # Apply each encoder layer
        for layer in self.layers:
            x = layer(x, mask=mask)
        
        # Apply final layer normalization
        x = self.norm(x)
        
        return x

class Transformer(BaseModel):
    """
    Complete transformer model with encoder only.
    
    This implements a transformer model for sequence processing tasks,
    following the architecture described in "Attention is All You Need"
    (Vaswani et al., 2017).
    """
    
    def __init__(
        self,
        d_model: int = 512,
        num_heads: int = 8,
        num_layers: int = 6,
        d_ff: int = 2048,
        dropout: float = 0.1,
        max_seq_length: int = 5000,
        positional_encoding: str = "sinusoidal",
        input_dim: int = None,
        output_dim: int = None,
    ):
        """
        Initialize the transformer model.
        
        Args:
            d_model: Dimension of model embeddings
            num_heads: Number of attention heads
            num_layers: Number of encoder layers
            d_ff: Dimension of feed-forward network
            dropout: Dropout probability
            max_seq_length: Maximum sequence length
            positional_encoding: Type of positional encoding to use
                ("sinusoidal", "learned", or "rotary")
            input_dim: Input dimension (if different from d_model)
            output_dim: Output dimension (if different from d_model)
        """
        super().__init__()
        
        self.d_model = d_model
        
        # Input projection (if needed)
        if input_dim is not None and input_dim != d_model:
            self.input_projection = nn.Linear(input_dim, d_model)
        else:
            self.input_projection = None
        
        # Encoder
        self.encoder = TransformerEncoder(
            d_model=d_model,
            num_heads=num_heads,
            num_layers=num_layers,
            d_ff=d_ff,
            dropout=dropout,
            max_seq_length=max_seq_length,
            positional_encoding=positional_encoding,
        )
        
        # Output projection (if needed)
        if output_dim is not None and output_dim != d_model:
            self.output_projection = nn.Linear(d_model, output_dim)
        else:
            self.output_projection = None
    
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass for the transformer.
        
        Args:
            x: Input tensor of shape [batch_size, seq_length, input_dim]
            mask: Optional mask tensor
            
        Returns:
            Output tensor of shape [batch_size, seq_length, output_dim]
        """
        # Apply input projection if needed
        if self.input_projection is not None:
            x = self.input_projection(x)
        
        # Apply encoder
        x = self.encoder(x, mask=mask)
        
        # Apply output projection if needed
        if self.output_projection is not None:
            x = self.output_projection(x)
        
        return x
    
    def configure_optimizers(self, lr: float = 0.0001) -> torch.optim.Optimizer:
        """
        Configure the optimizer for training.
        
        Args:
            lr: Learning rate
            
        Returns:
            Configured optimizer
        """
        return torch.optim.Adam(self.parameters(), lr=lr)