# src/models/vision/patch_embedding.py
import torch
import torch.nn as nn
from typing import Tuple, Optional


class PatchEmbedding(nn.Module):
    """
    Converts image patches to embeddings.

    This combines patch extraction and linear projection in a single efficient module.
    """

    def __init__(
        self,
        image_size: int = 224,
        patch_size: int = 16,
        in_channels: int = 3,
        embed_dim: int = 768,
        dropout: float = 0.0,
        use_cls_token: bool = True,
        positional_encoding: str = "learned",
    ):
        """
        Initialize the patch embedding layer.

        Args:
            image_size: Size of the input images (assumed square)
            patch_size: Size of the patches to extract
            in_channels: Number of input channels (3 for RGB images)
            embed_dim: Dimension of the output embeddings
            dropout: Dropout probability
            use_cls_token: Whether to add a class token for classification
            positional_encoding: Type of positional encoding to use ('learned' or 'sinusoidal')
        """
        super().__init__()

        self.image_size = image_size
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.embed_dim = embed_dim
        self.use_cls_token = use_cls_token

        # Calculate number of patches (assuming square images)
        self.num_patches = (image_size // patch_size) ** 2

        # Patch extraction and embedding in one step using a convolution
        self.projection = nn.Conv2d(
            in_channels=in_channels,
            out_channels=embed_dim,
            kernel_size=patch_size,
            stride=patch_size,
        )

        # Class token (if used)
        if use_cls_token:
            self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
            self.num_tokens = self.num_patches + 1
        else:
            self.num_tokens = self.num_patches

        # Positional embedding
        if positional_encoding == "learned":
            self.position_embedding = nn.Parameter(
                torch.zeros(1, self.num_tokens, embed_dim)
            )
        else:
            # For sinusoidal, we'll create a non-learnable positional encoding
            position_ids = torch.arange(self.num_tokens).unsqueeze(0)
            self.register_buffer("position_ids", position_ids)
            self.position_embedding = create_sinusoidal_embeddings(
                self.num_tokens, embed_dim
            )

        # Dropout after adding position embeddings
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Convert image to patch embeddings.

        Args:
            x: Batch of images of shape [B, C, H, W]

        Returns:
            Embedded patches of shape [B, num_tokens, embed_dim]
        """
        B = x.shape[0]

        # [B, C, H, W] -> [B, embed_dim, num_patches_h, num_patches_w]
        x = self.projection(x)

        # [B, embed_dim, num_patches_h, num_patches_w] -> [B, embed_dim, num_patches]
        x = x.flatten(2)

        # [B, embed_dim, num_patches] -> [B, num_patches, embed_dim]
        x = x.transpose(1, 2)

        # Add class token if using
        if self.use_cls_token:
            # The cls tokens are broadcast to batch size
            cls_tokens = self.cls_token.expand(B, -1, -1)
            x = torch.cat((cls_tokens, x), dim=1)

        # Add position embeddings
        if isinstance(self.position_embedding, nn.Parameter):
            # Learned positional embeddings
            x = x + self.position_embedding
        else:
            # Sinusoidal positional embeddings
            x = x + self.position_embedding.to(x.device)

        # Apply dropout
        x = self.dropout(x)

        return x


def create_sinusoidal_embeddings(
    num_positions: int, embedding_dim: int
) -> torch.Tensor:
    """
    Create sinusoidal positional embeddings.

    Args:
        num_positions: Number of positions to embed
        embedding_dim: Dimension of each position embedding

    Returns:
        Positional embeddings of shape [1, num_positions, embedding_dim]
    """
    # Create position indices
    position = torch.arange(num_positions).unsqueeze(1)

    # Create dimension indices
    div_term = torch.exp(
        torch.arange(0, embedding_dim, 2)
        * (-torch.log(torch.tensor(10000.0)) / embedding_dim)
    )

    # Create positional embeddings
    pe = torch.zeros(1, num_positions, embedding_dim)
    pe[0, :, 0::2] = torch.sin(position * div_term)
    pe[0, :, 1::2] = torch.cos(position * div_term)

    return pe
