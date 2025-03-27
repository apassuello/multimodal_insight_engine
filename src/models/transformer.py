# src/models/transformer.py (new file)
import torch
import torch.nn as nn
from typing import Optional, Dict, Any

from .base_model import BaseModel
from .attention import MultiHeadAttention
from .layers import FeedForwardBlock
from .positional import PositionalEncoding, RotaryPositionEncoding
from .embeddings import TokenEmbedding

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
                head_dim=d_model // num_heads,  # head_dim
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
        # Move input to device
        x = x.to(next(self.parameters()).device)
        if mask is not None:
            mask = mask.to(next(self.parameters()).device)
        
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


# Updated TransformerEncoder class
class TransformerEncoder(nn.Module):
    """
    Transformer encoder consisting of multiple encoder layers.
    
    This implements the encoder part of the transformer as described in
    "Attention is All You Need" (Vaswani et al., 2017).
    """
    
    def __init__(
        self,
        vocab_size: Optional[int] = None,
        d_model: int = 512,
        num_heads: int = 8,
        num_layers: int = 6,
        d_ff: int = 2048,
        dropout: float = 0.1,
        max_seq_length: int = 5000,
        positional_encoding: str = "sinusoidal",
    ):
        """
        Initialize the transformer encoder.
        
        Args:
            vocab_size: Size of vocabulary (if None, token embedding is not created)
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
        self.vocab_size = vocab_size if vocab_size is not None else 0
        
        # Token embedding layer (optional)
        self.has_embeddings = vocab_size is not None
        if self.has_embeddings:
            self.token_embedding = TokenEmbedding(vocab_size, d_model)
        
        # Positional encoding
        if positional_encoding == "rotary":
            self.use_rotary = True
            # No separate positional encoding layer for rotary embeddings
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
        
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.d_ff = d_ff
        self.dropout = dropout
        self.max_seq_length = max_seq_length
        self.positional_encoding_type = positional_encoding
    
    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass for the encoder.
        
        Args:
            x: Either token indices [batch_size, seq_length] if self.has_embeddings is True,
               or already embedded tokens [batch_size, seq_length, d_model]
            mask: Optional mask tensor
            
        Returns:
            Output tensor of shape [batch_size, seq_length, d_model]
        """
        # Move input to device
        x = x.to(next(self.parameters()).device)
        if mask is not None:
            mask = mask.to(next(self.parameters()).device)
        
        # Apply token embeddings if needed
        if self.has_embeddings:
            x = self.token_embedding(x)
        
        # Apply positional encoding if not using rotary embeddings
        if not self.use_rotary:
            x = self.positional_encoding(x)
        
        # Apply each encoder layer
        attentions = []  # Store attention weights for visualization/interpretability
        for layer in self.layers:
            x = layer(x, mask=mask)
            
            # In practice, you'd collect attention weights from each layer
            # We'll assume this is done within the layer for now
        
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
        # Move input to device
        x = x.to(next(self.parameters()).device)
        if mask is not None:
            mask = mask.to(next(self.parameters()).device)
        
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


class TransformerDecoderLayer(nn.Module):
    """
    A single transformer decoder layer.
    
    This implements one layer of the transformer decoder as described in
    "Attention is All You Need" (Vaswani et al., 2017).
    Each layer consists of:
    1. Masked multi-head self-attention
    2. Multi-head cross-attention to encoder outputs
    3. Position-wise feed-forward network
    All with residual connections and layer normalization.
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
        Initialize the transformer decoder layer.
        
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
        
        # 1. Self-attention layer (masked to prevent looking at future tokens)
        self.self_attn = MultiHeadAttention(
            input_dim=d_model,
            num_heads=num_heads,
            dropout=dropout
        )
        
        # 2. Cross-attention layer (to attend to encoder outputs)
        self.cross_attn = MultiHeadAttention(
            input_dim=d_model,
            num_heads=num_heads,
            dropout=dropout
        )
        
        # Rotary embeddings if specified (applied within attention)
        if use_rotary_embeddings:
            self.rotary_emb = RotaryPositionEncoding(
                head_dim=d_model // num_heads,  # head_dim
                max_seq_length=5000
            )
        
        # 3. Feed-forward block
        self.feed_forward = FeedForwardBlock(
            input_dim=d_model,
            hidden_dim=d_ff,
            output_dim=d_model,
            activation="relu",
            dropout=dropout,
            use_layer_norm=False,
        )
        
        # Layer normalization layers (pre-norm architecture)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        
        # Dropout layers
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)
    
    def forward(
        self,
        x: torch.Tensor,
        memory: torch.Tensor,
        tgt_mask: Optional[torch.Tensor] = None,
        memory_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass for the decoder layer.
        
        Args:
            x: Input tensor from previous decoder layer [batch_size, tgt_len, d_model]
            memory: Output of the encoder [batch_size, src_len, d_model]
            tgt_mask: Mask for target sequence (prevents attending to future tokens)
            memory_mask: Mask for encoder outputs (usually for padding)
            
        Returns:
            Output tensor of shape [batch_size, tgt_len, d_model]
        """
        # Ensure memory is a tensor
        if isinstance(memory, list):
            memory = torch.tensor(memory, dtype=torch.float32)

        # Move inputs to device
        x = x.to(next(self.parameters()).device)
        memory = memory.to(next(self.parameters()).device)
        if tgt_mask is not None:
            tgt_mask = tgt_mask.to(next(self.parameters()).device)
        if memory_mask is not None:
            memory_mask = memory_mask.to(next(self.parameters()).device)
        
        # 1. Self-attention block with residual connection
        # Apply layer norm before attention (pre-norm architecture)
        norm_x = self.norm1(x)
        
        # Apply self-attention with rotary embeddings if specified
        if self.use_rotary_embeddings and hasattr(self, 'rotary_emb'):
            self_attn_output, _ = self.self_attn(
                norm_x, norm_x, norm_x, mask=tgt_mask, rotary_emb=self.rotary_emb
            )
        else:
            self_attn_output, _ = self.self_attn(norm_x, norm_x, norm_x, mask=tgt_mask)
        
        # Apply dropout and residual connection
        x = x + self.dropout1(self_attn_output)
        
        # 2. Cross-attention block with residual connection
        # Apply layer norm before attention
        norm_x = self.norm2(x)
        
        # Apply cross-attention to encoder outputs
        cross_attn_output, _ = self.cross_attn(norm_x, memory, memory, mask=memory_mask)
        
        # Apply dropout and residual connection
        x = x + self.dropout2(cross_attn_output)
        
        # 3. Feed-forward block with residual connection
        # Apply layer norm before feed-forward
        norm_x = self.norm3(x)
        ff_output = self.feed_forward(norm_x)
        
        # Apply dropout and residual connection
        x = x + self.dropout3(ff_output)
        
        return x


class TransformerDecoder(nn.Module):
    """
    Transformer decoder consisting of multiple decoder layers.
    
    This implements the decoder part of the transformer as described in
    "Attention is All You Need" (Vaswani et al., 2017).
    """
    
    def __init__(
        self,
        vocab_size: Optional[int] = None,
        d_model: int = 512,
        num_heads: int = 8,
        num_layers: int = 6,
        d_ff: int = 2048,
        dropout: float = 0.1,
        max_seq_length: int = 5000,
        positional_encoding: str = "sinusoidal",
    ):
        """
        Initialize the transformer decoder.
        
        Args:
            vocab_size: Size of vocabulary (if None, token embedding is not created)
            d_model: Dimension of model embeddings
            num_heads: Number of attention heads
            num_layers: Number of decoder layers
            d_ff: Dimension of feed-forward network
            dropout: Dropout probability
            max_seq_length: Maximum sequence length
            positional_encoding: Type of positional encoding to use
                ("sinusoidal", "learned", or "rotary")
        """
        super().__init__()
        
        self.d_model = d_model
        self.positional_encoding_type = positional_encoding
        
        # Token embedding layer (optional)
        self.has_embeddings = vocab_size is not None
        if self.has_embeddings:
            self.token_embedding = TokenEmbedding(vocab_size, d_model)
        
        # Positional encoding
        if positional_encoding == "rotary":
            self.use_rotary = True
            # No separate positional encoding layer for rotary embeddings
        else:
            self.use_rotary = False
            self.positional_encoding = PositionalEncoding(
                d_model=d_model,
                max_seq_length=max_seq_length,
                dropout=dropout,
                encoding_type="sinusoidal" if positional_encoding == "sinusoidal" else "learned"
            )
        
        # Stack of decoder layers
        self.layers = nn.ModuleList([
            TransformerDecoderLayer(
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
        
        # Output projection (if vocab_size is provided)
        if vocab_size is not None:
            self.output_projection = nn.Linear(d_model, vocab_size)
        else:
            self.output_projection = None
    
    def forward(
        self,
        x: torch.Tensor,
        memory: torch.Tensor,
        tgt_mask: Optional[torch.Tensor] = None,
        memory_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass for the decoder.
        
        Args:
            x: Either token indices [batch_size, tgt_len] if self.has_embeddings is True,
               or already embedded tokens [batch_size, tgt_len, d_model]
            memory: Output of the encoder [batch_size, src_len, d_model]
            tgt_mask: Mask for target sequence (prevents attending to future tokens)
            memory_mask: Mask for encoder outputs (usually for padding)
            
        Returns:
            Output tensor, either:
            - [batch_size, tgt_len, d_model] if output_projection is None
            - [batch_size, tgt_len, vocab_size] if output_projection is not None
        """
        # Ensure memory is a tensor
        if isinstance(memory, list):
            memory = torch.tensor(memory, dtype=torch.float32)

        # Move inputs to device
        x = x.to(next(self.parameters()).device)
        memory = memory.to(next(self.parameters()).device)
        if tgt_mask is not None:
            tgt_mask = tgt_mask.to(next(self.parameters()).device)
        if memory_mask is not None:
            memory_mask = memory_mask.to(next(self.parameters()).device)
        
        # Apply token embeddings if needed
        if self.has_embeddings:
            x = self.token_embedding(x)
        
        # Apply positional encoding if not using rotary embeddings
        if not self.use_rotary:
            x = self.positional_encoding(x)
        
        # Apply each decoder layer
        for layer in self.layers:
            x = layer(x, memory, tgt_mask=tgt_mask, memory_mask=memory_mask)
        
        # Apply final layer normalization
        x = self.norm(x)
        
        # Apply output projection if available
        if self.output_projection is not None:
            x = self.output_projection(x)
        
        return x


class EncoderDecoderTransformer(BaseModel):
    """
    Complete encoder-decoder transformer model.
    
    This implements the full transformer architecture as described in
    "Attention is All You Need" (Vaswani et al., 2017).
    """
    
    def __init__(
        self,
        src_vocab_size: int,
        tgt_vocab_size: int,
        d_model: int = 512,
        num_heads: int = 8,
        num_encoder_layers: int = 6,
        num_decoder_layers: int = 6,
        d_ff: int = 2048,
        dropout: float = 0.1,
        max_seq_length: int = 5000,
        positional_encoding: str = "sinusoidal",
        share_embeddings: bool = False,
    ):
        """
        Initialize the encoder-decoder transformer model.
        
        Args:
            src_vocab_size: Size of source vocabulary
            tgt_vocab_size: Size of target vocabulary
            d_model: Dimension of model embeddings
            num_heads: Number of attention heads
            num_encoder_layers: Number of encoder layers
            num_decoder_layers: Number of decoder layers
            d_ff: Dimension of feed-forward network
            dropout: Dropout probability
            max_seq_length: Maximum sequence length
            positional_encoding: Type of positional encoding to use
            share_embeddings: Whether to share embeddings between encoder and decoder
        """
        super().__init__()
        
        self.d_model = d_model
        self.share_embeddings = share_embeddings
        
        # Encoder
        self.encoder = TransformerEncoder(
            vocab_size=src_vocab_size,
            d_model=d_model,
            num_heads=num_heads,
            num_layers=num_encoder_layers,
            d_ff=d_ff,
            dropout=dropout,
            max_seq_length=max_seq_length,
            positional_encoding=positional_encoding,
        )
        
        # Decoder
        self.decoder = TransformerDecoder(
            vocab_size=tgt_vocab_size if not share_embeddings else None,
            d_model=d_model,
            num_heads=num_heads,
            num_layers=num_decoder_layers,
            d_ff=d_ff,
            dropout=dropout,
            max_seq_length=max_seq_length,
            positional_encoding=positional_encoding,
        )
        
        # If sharing embeddings, link decoder's embedding to encoder's
        if share_embeddings:
            if src_vocab_size != tgt_vocab_size:
                raise ValueError("Cannot share embeddings with different vocabulary sizes")
            self.decoder.token_embedding = self.encoder.token_embedding
        
        # Final output projection if not already in decoder
        if self.decoder.output_projection is None:
            self.output_projection = nn.Linear(d_model, tgt_vocab_size)
        else:
            self.output_projection = None
    
    def forward(
        self,
        src: torch.Tensor,
        tgt: torch.Tensor,
        src_mask: Optional[torch.Tensor] = None,
        tgt_mask: Optional[torch.Tensor] = None,
        memory_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass for the transformer.
        
        Args:
            src: Source token indices [batch_size, src_len]
            tgt: Target token indices [batch_size, tgt_len]
            src_mask: Mask for source sequence (usually for padding)
            tgt_mask: Mask for target sequence (prevents attending to future tokens)
            memory_mask: Mask for encoder outputs
            
        Returns:
            Output tensor of shape [batch_size, tgt_len, tgt_vocab_size]
        """
        # Move inputs to device
        src = src.to(next(self.parameters()).device)
        tgt = tgt.to(next(self.parameters()).device)
        if src_mask is not None:
            src_mask = src_mask.to(next(self.parameters()).device)
        if tgt_mask is not None:
            tgt_mask = tgt_mask.to(next(self.parameters()).device)
        if memory_mask is not None:
            memory_mask = memory_mask.to(next(self.parameters()).device)
        
        # Encode source sequence
        memory = self.encoder(src, mask=src_mask)
        
        # Decode target sequence using encoder memory
        output = self.decoder(tgt, memory, tgt_mask=tgt_mask, memory_mask=memory_mask)
        
        # Apply output projection if needed
        if self.output_projection is not None:
            output = self.output_projection(output)
        
        return output
    
    def encode(self, src: torch.Tensor, src_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Encode source sequence.
        
        Args:
            src: Source token indices [batch_size, src_len]
            src_mask: Mask for source sequence
            
        Returns:
            Encoder output of shape [batch_size, src_len, d_model]
        """
        return self.encoder(src, mask=src_mask)
    
    def decode(
        self,
        tgt: torch.Tensor,
        memory: torch.Tensor,
        tgt_mask: Optional[torch.Tensor] = None,
        memory_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Decode target sequence given encoder memory.
        
        Args:
            tgt: Target token indices [batch_size, tgt_len]
            memory: Encoder output [batch_size, src_len, d_model]
            tgt_mask: Mask for target sequence
            memory_mask: Mask for encoder outputs
            
        Returns:
            Decoder output of shape [batch_size, tgt_len, tgt_vocab_size]
        """
        output = self.decoder(tgt, memory, tgt_mask=tgt_mask, memory_mask=memory_mask)
        
        # Apply output projection if needed
        if self.output_projection is not None:
            output = self.output_projection(output)
        
        return output
    
    def generate(
        self,
        src: torch.Tensor,
        max_len: int,
        bos_token_id: int,
        eos_token_id: int,
        src_mask: Optional[torch.Tensor] = None,
        memory_mask: Optional[torch.Tensor] = None,
        temperature: float = 1.0,
    ) -> torch.Tensor:
        """
        Generate a sequence from a source sequence.
        
        Args:
            src: Source token indices [batch_size, src_len]
            max_len: Maximum length of generated sequence
            bos_token_id: Beginning of sequence token ID
            eos_token_id: End of sequence token ID
            src_mask: Mask for source sequence
            memory_mask: Mask for encoder outputs
            temperature: Sampling temperature (1.0 = greedy)
            
        Returns:
            Generated sequences of shape [batch_size, seq_len]
        """
        batch_size = src.size(0)
        device = src.device
        
        # Encode source sequence
        memory = self.encode(src, src_mask=src_mask)
        
        # Initialize target sequence with BOS token
        tgt = torch.full((batch_size, 1), bos_token_id, dtype=torch.long, device=device)
        
        # Track which sequences are complete
        completed = torch.zeros(batch_size, dtype=torch.bool, device=device)
        
        # Generate tokens auto-regressively
        for i in range(max_len - 1):
            # Create appropriate target mask (prevent attending to future tokens)
            tgt_len = tgt.size(1)
            tgt_mask = self.generate_square_subsequent_mask(tgt_len, device)
            
            # Get logits for next token
            logits = self.decode(tgt, memory, tgt_mask=tgt_mask, memory_mask=memory_mask)
            logits = logits[:, -1, :]  # Focus on last token prediction
            
            # Apply temperature
            if temperature != 1.0:
                logits = logits / temperature
            
            # Sample from logits
            probs = torch.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            
            # Concatenate new token to target sequence
            tgt = torch.cat([tgt, next_token], dim=1)
            
            # Mark sequences with EOS as completed
            completed = completed | (next_token.squeeze(-1) == eos_token_id)
            
            # Stop if all sequences are completed
            if completed.all():
                break
        
        return tgt
    
    def generate_square_subsequent_mask(self, size: int, device: torch.device) -> torch.Tensor:
        """
        Generate a square mask for the sequence.
        
        The mask ensures that the predictions for position i can only depend
        on known outputs at positions less than i.
        
        Args:
            size: Size of the square mask
            device: Device for the mask tensor
            
        Returns:
            Mask tensor of shape [size, size]
        """
        # Create a matrix where the upper triangle is masked
        mask = torch.triu(torch.ones(size, size, device=device), diagonal=1)
        return mask == 0  # Convert to boolean mask (True = keep, False = mask)
    
    def configure_optimizers(self, lr: float = 0.0001) -> dict:
        """
        Configure the optimizer and learning rate scheduler.
        
        Args:
            lr: Base learning rate
            
        Returns:
            Dictionary with optimizer and scheduler
        """
        # Create optimizer
        optimizer = torch.optim.Adam(self.parameters(), lr=lr, betas=(0.9, 0.98), eps=1e-9)
        
        # Create learning rate scheduler with warmup
        def lr_lambda(step):
            # Warmup for 4000 steps
            warmup_steps = 4000
            if step == 0:
                step = 1
            return min(step ** (-0.5), step * warmup_steps ** (-1.5)) * warmup_steps ** 0.5
        
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
        
        return {"optimizer": optimizer, "scheduler": scheduler}