import torch
import torch.nn as nn
from typing import Dict, Optional, Tuple, Any

from .bidirectional_cross_attention import BidirectionalCrossAttention


class CoAttentionFusion(nn.Module):
    """
    Co-Attention Fusion module for deep multimodal integration.

    This module implements a co-attention mechanism that jointly models
    the interactions between two modalities, creating a fused representation.
    """

    def __init__(
        self,
        vision_dim: int,
        text_dim: int,
        fusion_dim: int,
        num_heads: int = 8,
        num_layers: int = 2,
        dropout: float = 0.1,
    ):
        """
        Initialize co-attention fusion module.

        Args:
            vision_dim: Dimension of vision features
            text_dim: Dimension of text features
            fusion_dim: Dimension of the fused representations
            num_heads: Number of attention heads
            num_layers: Number of co-attention layers
            dropout: Dropout rate
        """
        super().__init__()

        # Initial projections to fusion dimension
        # self.vision_proj = nn.Linear(vision_dim, fusion_dim)
        # self.text_proj = nn.Linear(text_dim, fusion_dim)

        # Increase feature scale in initial projections
        self.vision_proj = nn.Sequential(
            nn.Linear(vision_dim, fusion_dim),
            nn.LayerNorm(fusion_dim),
            nn.GELU(),  # Add activation for better non-linearity
            nn.Linear(fusion_dim, fusion_dim),  # Add second layer for more capacity
        )
        self.text_proj = nn.Sequential(
            nn.Linear(text_dim, fusion_dim),
            nn.LayerNorm(fusion_dim),
            nn.GELU(),  # Add activation for better non-linearity
            nn.Linear(fusion_dim, fusion_dim),  # Add second layer for more capacity
        )
        # Initialize the projections with higher scale
        for module in [self.vision_proj, self.text_proj]:
            for m in module.modules():
                if isinstance(m, nn.Linear):
                    nn.init.xavier_uniform_(m.weight, gain=2.0)  # Higher gain
        # Stack of bidirectional cross-attention layers
        self.fusion_layers = nn.ModuleList(
            [
                BidirectionalCrossAttention(
                    vision_dim=fusion_dim,
                    text_dim=fusion_dim,
                    fusion_dim=fusion_dim,
                    num_heads=num_heads,
                    dropout=dropout,
                )
                for _ in range(num_layers)
            ]
        )

        # Final fusion layer to combine modalities
        self.final_fusion = nn.Sequential(
            nn.LayerNorm(fusion_dim * 2),
            nn.Linear(fusion_dim * 2, fusion_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.LayerNorm(fusion_dim),
        )

        # Modality-specific output projections
        self.vision_output_proj = nn.Linear(fusion_dim, vision_dim)
        self.text_output_proj = nn.Linear(fusion_dim, text_dim)

        # Global fusion token for pooled representation
        self.fusion_token = nn.Parameter(torch.zeros(1, 1, fusion_dim))
        nn.init.normal_(self.fusion_token, mean=0.0, std=0.5)

    def forward(
        self,
        vision_features: torch.Tensor,
        text_features: torch.Tensor,
        vision_mask: Optional[torch.Tensor] = None,
        text_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, Any]:
        """
        Forward pass for co-attention fusion.

        Args:
            vision_features: Features from the vision modality [batch_size, vision_seq_len, vision_dim]
            text_features: Features from the text modality [batch_size, text_seq_len, text_dim]
            vision_mask: Optional mask for vision features [batch_size, vision_seq_len]
            text_mask: Optional mask for text features [batch_size, text_seq_len]

        Returns:
            Dictionary containing:
            - 'vision_features': Updated vision features
            - 'text_features': Updated text features
            - 'fusion_features': Fused multimodal representation
            - 'pooled_fusion': Global fusion representation for classification
            - 'attention_maps': Dictionary of attention maps from each layer
        """
        batch_size = vision_features.shape[0]

        # Project inputs to fusion dimension
        vision_features_proj = self.vision_proj(vision_features)
        text_features_proj = self.text_proj(text_features)

        # Add fusion token to vision features
        fusion_tokens = self.fusion_token.expand(batch_size, -1, -1)
        vision_features_with_fusion = torch.cat(
            [fusion_tokens, vision_features_proj], dim=1
        )

        # Update vision mask to account for fusion token
        if vision_mask is not None:
            fusion_token_mask = torch.ones(
                batch_size, 1, device=vision_mask.device, dtype=vision_mask.dtype
            )
            vision_mask_with_fusion = torch.cat([fusion_token_mask, vision_mask], dim=1)
        else:
            vision_mask_with_fusion = None

        # Store attention maps for interpretability
        attention_maps = {}

        # Apply bidirectional cross-attention layers
        current_vision = vision_features_with_fusion
        current_text = text_features_proj

        for i, layer in enumerate(self.fusion_layers):
            layer_outputs = layer(
                vision_features=current_vision,
                text_features=current_text,
                vision_mask=vision_mask_with_fusion,
                text_mask=text_mask,
            )

            current_vision = layer_outputs["vision_features"]
            current_text = layer_outputs["text_features"]

            # Store attention maps
            attention_maps[f"layer_{i}_vision_to_text"] = layer_outputs[
                "vision_to_text_attn"
            ]
            attention_maps[f"layer_{i}_text_to_vision"] = layer_outputs[
                "text_to_vision_attn"
            ]

        # Extract fusion token output
        fusion_token_output = current_vision[:, 0]

        # Remove fusion token from vision features
        current_vision = current_vision[:, 1:]

        # Compute pooled representations for each modality
        # For MPS compatibility, simplify the pooling approach and ensure device consistency
        device = current_vision.device

        # Simple average pooling for both modalities
        # This is more robust for MPS devices and avoids device inconsistencies
        vision_pooled = current_vision.mean(dim=1)
        text_pooled = current_text.mean(dim=1)

        # Alternative with explicit device handling if needed:
        # if vision_mask is not None:
        #     # Move mask to the right device and ensure float
        #     vision_mask = vision_mask.to(device).float()
        #     vision_mask_expanded = vision_mask.unsqueeze(-1)
        #     vision_pooled = (current_vision * vision_mask_expanded).sum(1) /
        #                     torch.clamp(vision_mask_expanded.sum(1), min=1.0)
        # else:
        #     vision_pooled = current_vision.mean(dim=1)
        #
        # if text_mask is not None:
        #     # Move mask to the right device and ensure float
        #     text_mask = text_mask.to(device).float()
        #     text_mask_expanded = text_mask.unsqueeze(-1)
        #     text_pooled = (current_text * text_mask_expanded).sum(1) /
        #                   torch.clamp(text_mask_expanded.sum(1), min=1.0)
        # else:
        #     text_pooled = current_text.mean(dim=1)

        # Concatenate and fuse the pooled representations
        pooled_concat = torch.cat([vision_pooled, text_pooled], dim=1)
        fused_representation = self.final_fusion(pooled_concat)

        # Project back to original dimensions
        vision_output = self.vision_output_proj(current_vision)
        text_output = self.text_output_proj(current_text)

        return {
            "vision_features": vision_output,
            "text_features": text_output,
            "fusion_features": fused_representation,
            "pooled_fusion": fusion_token_output,  # Alternative fusion representation
            "attention_maps": attention_maps,
        }
