import torch
import torch.nn as nn
import math
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import numpy as np
from typing import Optional, Literal
import os

"""MODULE: positional.py
PURPOSE: Implements various positional encoding schemes for transformer models to handle sequence order information
KEY COMPONENTS:
- PositionalEncoding: Supports both fixed sinusoidal and learned positional encodings
- RotaryPositionEncoding: Implements RoPE (Rotary Position Embedding) for enhanced position handling
DEPENDENCIES: torch, torch.nn, math, matplotlib.pyplot, numpy, typing
SPECIAL NOTES: Provides visualization tools for understanding encoding patterns"""


class PositionalEncoding(nn.Module):
    """
    Positional encoding for transformer models.

    This class provides positional information to transformer models, which
    otherwise have no notion of sequence order. It supports both fixed sinusoidal
    encodings (as in "Attention is All You Need") and learned position embeddings.
    """

    def __init__(
        self,
        d_model: int,
        max_seq_length: int = 5000,
        dropout: float = 0.1,
        encoding_type: Literal["sinusoidal", "learned"] = "sinusoidal",
    ):
        """
        Initialize the positional encoding module.

        Args:
            d_model: Dimension of the model embeddings (must be even for sinusoidal encoding)
            max_seq_length: Maximum sequence length to support
            dropout: Dropout probability
            encoding_type: Type of positional encoding to use

        Raises:
            ValueError: If d_model is odd when using sinusoidal encoding
        """
        super().__init__()

        if encoding_type == "sinusoidal" and d_model % 2 != 0:
            raise ValueError("d_model must be even for sinusoidal positional encoding")

        self.d_model = d_model
        self.max_seq_length = max_seq_length
        self.encoding_type = encoding_type

        # Dropout layer for regularization
        self.dropout = nn.Dropout(p=dropout)

        # Initialize the appropriate type of positional encoding
        if encoding_type == "sinusoidal":
            self._init_sinusoidal_encoding()
        elif encoding_type == "learned":
            self._init_learned_encoding()
        else:
            raise ValueError(f"Unknown encoding type: {encoding_type}")

    def _init_sinusoidal_encoding(self):
        """Initialize sinusoidal positional encodings."""
        pe = torch.zeros(1, self.max_seq_length, self.d_model)
        position = torch.arange(0, self.max_seq_length, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, self.d_model, 2).float() * (-math.log(10000.0) / self.d_model)
        )

        # Apply sinusoidal pattern
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)

        # Register as buffer (not a parameter)
        self.register_buffer("pe", pe)

    def _init_learned_encoding(self):
        """Initialize learned positional embeddings."""
        self.position_embeddings = nn.Parameter(
            torch.zeros(1, self.max_seq_length, self.d_model)
        )
        nn.init.xavier_normal_(self.position_embeddings)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Add positional encoding to the input embeddings.

        Args:
            x: Input embeddings of shape [batch_size, seq_length, d_model]
               or token indices of shape [batch_size, seq_length]

        Returns:
            Embeddings with positional information added
        """
        # Handle input shape
        if len(x.shape) == 2:
            batch_size, seq_length = x.shape
            # Expand to match expected dimensions
            x = x.unsqueeze(-1).expand(batch_size, seq_length, self.d_model)
        else:
            batch_size, seq_length, d_model = x.shape
            if d_model != self.d_model:
                raise ValueError(
                    f"Input dimension ({d_model}) does not match model dimension ({self.d_model})"
                )
        
        if seq_length > self.max_seq_length:
            raise ValueError(
                f"Sequence length ({seq_length}) exceeds maximum length ({self.max_seq_length})"
            )

        if self.encoding_type == "sinusoidal":
            # Expand pe to match batch size if needed
            pe = self.pe.expand(batch_size, -1, -1)
            x = x + pe[:, :seq_length, :]
        else:  # learned
            x = x + self.position_embeddings[:, :seq_length, :]

        return self.dropout(x)

    def visualize_encodings(self, seq_length: Optional[int] = None) -> Figure:
        """
        Visualize the positional encodings as a heatmap.

        Args:
            seq_length: Number of positions to visualize (defaults to all)

        Returns:
            Matplotlib figure with the visualization
        """
        # Explanation: "We include a visualization method to help understand the encoding patterns"
        if seq_length is None:
            seq_length = min(self.max_seq_length, 100)  # Limit to 100 for readability

        if self.encoding_type == "sinusoidal":
            # Get the pre-computed encodings
            encodings = self.pe[0, :seq_length, :].cpu().numpy()
        else:  # learned
            # Get the learned embeddings
            encodings = (
                self.position_embeddings[0, :seq_length, :].detach().cpu().numpy()
            )

        # Create a figure and axis
        fig, ax = plt.subplots(figsize=(10, 8))

        # Plot heatmap
        im = ax.imshow(encodings, cmap="viridis", aspect="auto")

        # Add color bar
        cbar = fig.colorbar(im, ax=ax)
        cbar.set_label("Encoding Value")

        # Set labels and title
        ax.set_xlabel("Embedding Dimension")
        ax.set_ylabel("Position in Sequence")
        ax.set_title(f"{self.encoding_type.capitalize()} Positional Encodings")

        # Set tick marks
        ax.set_xticks(np.arange(0, self.d_model, self.d_model // 10))
        ax.set_yticks(np.arange(0, seq_length, seq_length // 10))

        return fig

    def save_learned_embeddings(self, path: str) -> None:
        """
        Save learned positional embeddings to a file.

        Args:
            path: Path to save the embeddings

        Raises:
            ValueError: If called on sinusoidal encoding
        """
        if self.encoding_type != "learned":
            raise ValueError("Can only save embeddings for learned positional encoding")
        
        torch.save(self.position_embeddings, path)

    def load_learned_embeddings(self, path: str) -> None:
        """
        Load learned positional embeddings from a file.

        Args:
            path: Path to load the embeddings from

        Raises:
            ValueError: If called on sinusoidal encoding
        """
        if self.encoding_type != "learned":
            raise ValueError("Can only load embeddings for learned positional encoding")
        
        self.position_embeddings = torch.load(path)


class RotaryPositionEncoding(nn.Module):
    """
    Rotary Position Embedding (RoPE) for transformer models.
    
    RoPE performs position encoding by rotating word embeddings, which helps 
    models better handle relative positions and longer sequences.
    Based on the paper "RoFormer: Enhanced Transformer with Rotary Position Embedding"
    """
    
    def __init__(self, head_dim: int, max_seq_length: int = 5000, base: int = 10000):
        """
        Initialize the rotary position embeddings.
        
        Args:
            head_dim: Dimension of each attention head (must be divisible by 2)
            max_seq_length: Maximum sequence length to support
            base: Base for the frequency calculation
        """
        super().__init__()
        
        if head_dim % 2 != 0:
            raise ValueError(f"head_dim must be divisible by 2, got {head_dim}")
        
        self.head_dim = head_dim
        self.max_seq_length = max_seq_length
        self.base = base
        
        # Pre-compute frequency bands
        self._compute_freqs()
    
    def _compute_freqs(self):
        """Compute the frequency bands for rotary embeddings."""
        # Create frequency bands
        theta = 1.0 / (self.base ** (torch.arange(0, self.head_dim, 2).float() / self.head_dim))
        
        # Create position sequence
        seq_idx = torch.arange(self.max_seq_length).float()
        
        # Compute frequencies for each position and dimension
        # Shape: [seq_len, head_dim/2]
        freqs = torch.outer(seq_idx, theta)
        
        # Cache cos and sin values for efficiency
        # Each has shape [seq_len, head_dim/2]
        emb_cos = torch.cos(freqs)
        emb_sin = torch.sin(freqs)
        
        # Register buffers (not parameters)
        self.register_buffer("cos_cached", emb_cos)
        self.register_buffer("sin_cached", emb_sin)
    
    def forward(self, q: torch.Tensor, k: torch.Tensor, seq_len: Optional[int] = None) -> tuple:
        """
        Apply rotary position embeddings to query and key tensors.
        
        Args:
            q: Query tensor of shape [batch_size, seq_length, n_heads, head_dim]
            k: Key tensor of shape [batch_size, seq_length, n_heads, head_dim]
            seq_len: Sequence length (defaults to q's sequence length)
            
        Returns:
            Tuple of (rotated_q, rotated_k) with positional information
        """
        if seq_len is None:
            seq_len = q.size(1)
        
        if seq_len > self.max_seq_length:
            raise ValueError(
                f"Sequence length ({seq_len}) exceeds maximum length ({self.max_seq_length})"
            )
        
        # Validate input dimensions
        assert q.size(-1) == self.head_dim, f"Query dim {q.size(-1)} doesn't match head_dim {self.head_dim}"
        assert k.size(-1) == self.head_dim, f"Key dim {k.size(-1)} doesn't match head_dim {self.head_dim}"
        
        # Get the cos and sin values for the current sequence length
        cos = self.cos_cached[:seq_len]  # [seq_len, head_dim/2]
        sin = self.sin_cached[:seq_len]  # [seq_len, head_dim/2]
        
        # Reshape q and k for easier rotation
        q_reshape = q.reshape(*q.shape[:-1], -1, 2)  # [batch, seq_len, n_heads, head_dim/2, 2]
        k_reshape = k.reshape(*k.shape[:-1], -1, 2)  # [batch, seq_len, n_heads, head_dim/2, 2]
        
        # Separate real and imaginary parts
        q_real, q_imag = q_reshape[..., 0], q_reshape[..., 1]  # Each has shape [batch, seq_len, n_heads, head_dim/2]
        k_real, k_imag = k_reshape[..., 0], k_reshape[..., 1]  # Each has shape [batch, seq_len, n_heads, head_dim/2]
        
        # Reshape cos and sin for broadcasting
        # cos and sin have shape [seq_len, head_dim/2]
        # We need to add batch and head dimensions
        cos = cos.unsqueeze(0).unsqueeze(2)  # [1, seq_len, 1, head_dim/2]
        sin = sin.unsqueeze(0).unsqueeze(2)  # [1, seq_len, 1, head_dim/2]
        
        # Apply rotation
        # (a + bi) * (c + di) = (ac - bd) + (ad + bc)i
        q_real_new = q_real * cos - q_imag * sin  # [batch, seq_len, n_heads, head_dim/2]
        q_imag_new = q_real * sin + q_imag * cos  # [batch, seq_len, n_heads, head_dim/2]
        k_real_new = k_real * cos - k_imag * sin  # [batch, seq_len, n_heads, head_dim/2]
        k_imag_new = k_real * sin + k_imag * cos  # [batch, seq_len, n_heads, head_dim/2]
        
        # Reshape back to original shape
        q_new = torch.stack([q_real_new, q_imag_new], dim=-1)  # [batch, seq_len, n_heads, head_dim/2, 2]
        k_new = torch.stack([k_real_new, k_imag_new], dim=-1)  # [batch, seq_len, n_heads, head_dim/2, 2]
        
        q_new = q_new.reshape(*q.shape)  # [batch, seq_len, n_heads, head_dim]
        k_new = k_new.reshape(*k.shape)  # [batch, seq_len, n_heads, head_dim]
        
        return q_new, k_new
    
    def visualize_rotation(self, seq_length: int = 20) -> Figure:
        """
        Visualize the rotary position embeddings.

        Args:
            seq_length: Number of positions to visualize

        Returns:
            Matplotlib figure with the visualization
        """
        # Create a sample input where each position has the same embedding
        # Use a single head with the correct head_dim
        q = torch.ones(1, seq_length, 1, self.head_dim)
        k = torch.ones(1, seq_length, 1, self.head_dim)
        
        # Apply rotary embeddings
        q_rot, k_rot = self.forward(q, k, seq_length)
        
        # Create a figure with two subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 7))
        
        # Plot the first two dimensions of q at each position
        ax1.set_title("Rotation of Query Vector (first 2 dimensions)")
        for pos in range(seq_length):
            ax1.arrow(
                0, 0, 
                q_rot[0, pos, 0, 0].item(), q_rot[0, pos, 0, 1].item(),
                head_width=0.05, head_length=0.1, fc=f'C{pos}', ec=f'C{pos}',
                label=f"Position {pos}" if pos < 10 else None
            )
        
        ax1.set_xlim(-1.5, 1.5)
        ax1.set_ylim(-1.5, 1.5)
        ax1.axhline(y=0, color='k', linestyle='-', alpha=0.3)
        ax1.axvline(x=0, color='k', linestyle='-', alpha=0.3)
        ax1.set_aspect('equal')
        ax1.grid(True, alpha=0.3)
        
        # Only show legend for first 10 positions to avoid cluttering
        ax1.legend(loc='upper right')
        
        # Plot the first two dimensions of k at each position
        ax2.set_title("Rotation of Key Vector (first 2 dimensions)")
        for pos in range(seq_length):
            ax2.arrow(
                0, 0, 
                k_rot[0, pos, 0, 0].item(), k_rot[0, pos, 0, 1].item(),
                head_width=0.05, head_length=0.1, fc=f'C{pos}', ec=f'C{pos}',
                label=f"Position {pos}" if pos < 10 else None
            )
        
        ax2.set_xlim(-1.5, 1.5)
        ax2.set_ylim(-1.5, 1.5)
        ax2.axhline(y=0, color='k', linestyle='-', alpha=0.3)
        ax2.axvline(x=0, color='k', linestyle='-', alpha=0.3)
        ax2.set_aspect('equal')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig

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
        "module_purpose": "Implements various positional encoding schemes for transformer models",
        "key_classes": [
            {
                "name": "PositionalEncoding",
                "purpose": "Provides both fixed sinusoidal and learned positional encodings for transformer models",
                "key_methods": [
                    {
                        "name": "__init__",
                        "signature": "__init__(self, d_model: int, max_seq_length: int = 5000, dropout: float = 0.1, encoding_type: Literal['sinusoidal', 'learned'] = 'sinusoidal')",
                        "brief_description": "Initialize the positional encoding module with specified type"
                    },
                    {
                        "name": "forward",
                        "signature": "forward(self, x: torch.Tensor) -> torch.Tensor",
                        "brief_description": "Add positional encoding to input embeddings"
                    },
                    {
                        "name": "visualize_encodings",
                        "signature": "visualize_encodings(self, seq_length: Optional[int] = None) -> plt.Figure",
                        "brief_description": "Visualize the positional encodings as a heatmap"
                    }
                ],
                "inheritance": "nn.Module",
                "dependencies": ["torch", "torch.nn", "math", "matplotlib.pyplot", "numpy"]
            },
            {
                "name": "RotaryPositionEncoding",
                "purpose": "Implements RoPE (Rotary Position Embedding) for enhanced position handling in transformers",
                "key_methods": [
                    {
                        "name": "__init__",
                        "signature": "__init__(self, head_dim: int, max_seq_length: int = 5000, base: int = 10000)",
                        "brief_description": "Initialize the rotary position embeddings"
                    },
                    {
                        "name": "forward",
                        "signature": "forward(self, q: torch.Tensor, k: torch.Tensor, seq_len: Optional[int] = None) -> tuple",
                        "brief_description": "Apply rotary position embeddings to query and key tensors"
                    },
                    {
                        "name": "visualize_rotation",
                        "signature": "visualize_rotation(self, seq_length: int = 20) -> plt.Figure",
                        "brief_description": "Visualize the rotary position embeddings"
                    }
                ],
                "inheritance": "nn.Module",
                "dependencies": ["torch", "torch.nn", "math", "matplotlib.pyplot", "numpy"]
            }
        ],
        "external_dependencies": ["torch", "matplotlib", "numpy"],
        "complexity_score": 6,  # Moderate-high complexity due to multiple encoding types and visualization features
    }