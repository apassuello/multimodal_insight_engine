# src/models/vision/vision_transformer.py
"""
Vision Transformer (ViT) implementation with robust tensor operations.
Follows the architecture described in "An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale".
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, Union, List, Dict, Any
import math

from ..base_model import BaseModel


class PatchEmbed(nn.Module):
    """
    Split image into patches and embed them.

    This module handles the critical first step of converting an image into
    a sequence of tokens for transformer processing.
    """

    def __init__(
        self,
        img_size: int = 224,
        patch_size: int = 16,
        in_chans: int = 3,
        embed_dim: int = 768,
    ):
        """
        Initialize patch embedding layer.

        Args:
            img_size: Input image size (assumed square)
            patch_size: Patch size (assumed square)
            in_chans: Number of input channels
            embed_dim: Embedding dimension
        """
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.n_patches = (img_size // patch_size) ** 2

        # Patch projection using convolution
        # This is more efficient than manually splitting and embedding
        self.proj = nn.Conv2d(
            in_chans, embed_dim, kernel_size=patch_size, stride=patch_size
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass to extract and embed patches.

        Args:
            x: Input images of shape [B, C, H, W]

        Returns:
            Patch embeddings of shape [B, N, E] where N is number of patches
            and E is embedding dimension
        """
        B, C, H, W = x.shape

        # Check image size
        if H != self.img_size or W != self.img_size:
            raise ValueError(
                f"Input image size ({H}x{W}) doesn't match expected size ({self.img_size}x{self.img_size})"
            )

        # Apply convolution to extract patch embeddings
        # [B, C, H, W] -> [B, E, H//P, W//P] where P is patch size and E is embed dim
        x = self.proj(x)

        # Flatten the spatial dimensions
        # [B, E, H', W'] -> [B, E, N] where N is number of patches
        x = x.flatten(2)

        # Transpose to get [B, N, E] format expected by transformer
        x = x.contiguous().transpose(1, 2)

        return x


class Attention(nn.Module):
    """
    Multi-head attention module with improved tensor handling.
    """

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = False,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
    ):
        """
        Initialize attention module.

        Args:
            dim: Input dimension
            num_heads: Number of attention heads
            qkv_bias: Whether to include bias in the QKV projection
            attn_drop: Dropout rate for attention weights
            proj_drop: Dropout rate for output projection
        """
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim**-0.5

        # Combined QKV projection for efficiency
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor of shape [B, N, D] where N is sequence length

        Returns:
            Output tensor of shape [B, N, D]
        """
        B, N, C = x.shape

        # Project input to query, key, value
        # [B, N, C] -> [B, N, 3*C]
        qkv = self.qkv(x)

        # Reshape and permute to get separate Q, K, V tensors
        # [B, N, 3*C] -> [B, N, 3, num_heads, C//num_heads]
        qkv = qkv.contiguous().reshape(B, N, 3, self.num_heads, C // self.num_heads)

        # [B, N, 3, num_heads, C//num_heads] -> [3, B, num_heads, N, C//num_heads]
        qkv = qkv.contiguous().permute(2, 0, 3, 1, 4)

        # Unpack Q, K, V - ensure tensors are contiguous
        q, k, v = qkv[0], qkv[1], qkv[2]

        # Compute attention scores
        # [B, num_heads, N, C//num_heads] @ [B, num_heads, C//num_heads, N] -> [B, num_heads, N, N]
        attn = (q @ k.transpose(-2, -1)) * self.scale

        # Apply softmax to get attention weights
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        # Apply attention to values
        # [B, num_heads, N, N] @ [B, num_heads, N, C//num_heads] -> [B, num_heads, N, C//num_heads]
        x = attn @ v

        # Combine heads
        # [B, num_heads, N, C//num_heads] -> [B, N, num_heads, C//num_heads]
        x = x.contiguous().transpose(1, 2)

        # [B, N, num_heads, C//num_heads] -> [B, N, C]
        x = x.contiguous().reshape(B, N, C)

        # Linear projection and dropout
        x = self.proj(x)
        x = self.proj_drop(x)

        return x


class MLP(nn.Module):
    """
    MLP module used in Vision Transformer blocks.
    """

    def __init__(
        self,
        in_features: int,
        hidden_features: Optional[int] = None,
        out_features: Optional[int] = None,
        act_layer: nn.Module = nn.GELU,
        drop: float = 0.0,
    ):
        """
        Initialize MLP module.

        Args:
            in_features: Input feature dimension
            hidden_features: Hidden dimension
            out_features: Output feature dimension
            act_layer: Activation layer
            drop: Dropout rate
        """
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features

        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor

        Returns:
            Output tensor
        """
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Block(nn.Module):
    """
    Transformer block.
    """

    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = False,
        drop: float = 0.0,
        attn_drop: float = 0.0,
    ):
        """
        Initialize transformer block.

        Args:
            dim: Input dimension
            num_heads: Number of attention heads
            mlp_ratio: Ratio of MLP hidden dim to input dim
            qkv_bias: Whether to include bias in QKV projection
            drop: Dropout rate for MLP and projection
            attn_drop: Dropout rate for attention weights
        """
        super().__init__()

        # First norm and attention
        self.norm1 = nn.LayerNorm(dim)
        self.attn = Attention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            attn_drop=attn_drop,
            proj_drop=drop,
        )

        # Second norm and MLP
        self.norm2 = nn.LayerNorm(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = MLP(in_features=dim, hidden_features=mlp_hidden_dim, drop=drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with residual connections.

        Args:
            x: Input tensor

        Returns:
            Output tensor
        """
        # First block: attention with residual connection
        x = x + self.attn(self.norm1(x))

        # Second block: MLP with residual connection
        x = x + self.mlp(self.norm2(x))

        return x


class VisionTransformer(BaseModel):
    """
    Vision Transformer main model.
    """

    def __init__(
        self,
        image_size: int = 224,
        patch_size: int = 16,
        in_channels: int = 3,
        num_classes: int = 1000,
        embed_dim: int = 768,
        depth: int = 12,
        num_heads: int = 12,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        dropout: float = 0.0,
        attention_dropout: float = 0.0,
        pool: str = "cls",
        positional_encoding: str = "learned",
    ):
        """
        Initialize Vision Transformer.

        Args:
            image_size: Input image size
            patch_size: Patch size
            in_channels: Number of input channels
            num_classes: Number of output classes
            embed_dim: Embedding dimension
            depth: Number of transformer blocks
            num_heads: Number of attention heads
            mlp_ratio: Ratio of MLP hidden dim to embedding dim
            qkv_bias: Whether to include bias in QKV projection
            dropout: Dropout rate
            attention_dropout: Attention dropout rate
            pool: Type of pooling ('cls' or 'mean')
            positional_encoding: Type of positional encoding ('learned' or 'sinusoidal')
        """
        super().__init__()

        # Configuration
        self.image_size = image_size
        self.patch_size = patch_size
        self.num_classes = num_classes
        self.embed_dim = embed_dim
        self.pool = pool

        # Patch embedding
        self.patch_embed = PatchEmbed(
            img_size=image_size,
            patch_size=patch_size,
            in_chans=in_channels,
            embed_dim=embed_dim,
        )
        num_patches = self.patch_embed.n_patches

        # Class token
        if self.pool == "cls":
            self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
            num_tokens = num_patches + 1
        else:
            self.cls_token = None
            num_tokens = num_patches

        # Position embedding
        if positional_encoding == "learned":
            self.pos_embed = nn.Parameter(torch.zeros(1, num_tokens, embed_dim))
            nn.init.trunc_normal_(self.pos_embed, std=0.02)
        else:
            # Create sinusoidal position embeddings
            pos_embed = self._create_sinusoidal_embed(num_tokens, embed_dim)
            self.register_buffer("pos_embed", pos_embed, persistent=False)

        # Dropout after position embedding
        self.pos_drop = nn.Dropout(p=dropout)

        # Transformer blocks
        self.blocks = nn.ModuleList(
            [
                Block(
                    dim=embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    drop=dropout,
                    attn_drop=attention_dropout,
                )
                for _ in range(depth)
            ]
        )

        # Final normalization
        self.norm = nn.LayerNorm(embed_dim)

        # Classifier head
        self.head = nn.Linear(embed_dim, num_classes)

        # Initialize weights
        if self.pool == "cls" and self.cls_token is not None:
            nn.init.trunc_normal_(self.cls_token, std=0.02)
        self.apply(self._init_weights)

    def _create_sinusoidal_embed(self, num_tokens, embed_dim):
        """
        Create sinusoidal positional embeddings.

        Args:
            num_tokens: Number of tokens (patches + class token if used)
            embed_dim: Embedding dimension

        Returns:
            Positional embeddings of shape [1, num_tokens, embed_dim]
        """
        pe = torch.zeros(num_tokens, embed_dim)
        position = torch.arange(0, num_tokens, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, embed_dim, 2, dtype=torch.float)
            * (-math.log(10000.0) / embed_dim)
        )

        # Apply sine to even indices
        pe[:, 0::2] = torch.sin(position * div_term)

        # Apply cosine to odd indices
        pe[:, 1::2] = torch.cos(position * div_term)

        # Add batch dimension: [num_tokens, embed_dim] -> [1, num_tokens, embed_dim]
        pe = pe.unsqueeze(0)

        return pe

    def _init_weights(self, m):
        """
        Initialize weights.

        Args:
            m: Module to initialize
        """
        if isinstance(m, nn.Linear):
            # Use truncated normal distribution for linear layers
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.LayerNorm):
            # Initialize layer norm weights with ones and biases with zeros
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the Vision Transformer to get features.

        Args:
            x: Input images of shape [B, C, H, W]

        Returns:
            Features before classification head, shape depends on pooling method
        """
        # Get batch size
        B = x.shape[0]

        # Patch embedding: [B, C, H, W] -> [B, N, E]
        x = self.patch_embed(x)

        # Add class token if using cls pooling
        if self.pool == "cls" and self.cls_token is not None:
            # [1, 1, E] -> [B, 1, E]
            cls_tokens = self.cls_token.expand(B, -1, -1)
            # [B, N, E] + [B, 1, E] -> [B, N+1, E]
            x = torch.cat((cls_tokens, x), dim=1)

        # Add position embeddings and apply dropout
        x = x + self.pos_embed
        x = self.pos_drop(x)

        # Apply transformer blocks
        for blk in self.blocks:
            x = blk(x)

        # Apply final normalization
        x = self.norm(x)

        # Apply pooling
        if self.pool == "cls" and self.cls_token is not None:
            # Use class token for classification
            x = x[:, 0]
        else:
            # Use mean pooling
            x = x.mean(dim=1)

        return x

    def forward(
        self, x: torch.Tensor, return_features: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Forward pass.

        Args:
            x: Input images of shape [B, C, H, W]
            return_features: Whether to return features before classification head

        Returns:
            - If return_features=False: Class logits of shape [B, num_classes]
            - If return_features=True: Tuple of (logits, features)
        """
        # Extract features
        features = self.forward_features(x)

        # Apply classification head
        logits = self.head(features)

        if return_features:
            return logits, features
        else:
            return logits

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract features from the Vision Transformer before classification head.

        Args:
            x: Input images of shape [B, C, H, W]

        Returns:
            Features of shape [B, embed_dim]
        """
        return self.forward_features(x)

    def configure_optimizers(
        self,
        lr: float = 1e-3,
        weight_decay: float = 0.05,
        betas: Tuple[float, float] = (0.9, 0.999),
    ) -> torch.optim.Optimizer:
        """
        Configure the optimizer for training.

        Args:
            lr: Learning rate
            weight_decay: Weight decay
            betas: Adam betas

        Returns:
            Configured optimizer
        """
        # Following DeiT paper approach: no weight decay for biases and layer norm
        decay_params = []
        no_decay_params = []

        for name, param in self.named_parameters():
            if not param.requires_grad:
                continue
            if "bias" in name or "norm" in name:
                no_decay_params.append(param)
            else:
                decay_params.append(param)

        optimizer_groups = [
            {"params": decay_params, "weight_decay": weight_decay},
            {"params": no_decay_params, "weight_decay": 0.0},
        ]

        optimizer = torch.optim.AdamW(optimizer_groups, lr=lr, betas=betas)

        return optimizer
