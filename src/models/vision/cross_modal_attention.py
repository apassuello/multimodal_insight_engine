# src/models/vision/cross_modal_attention.py
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


class GatedCrossModalAttention(nn.Module):
    """
    Enhanced cross-modal attention with gating mechanism.

    This module extends basic cross-attention with a gate that controls
    how much information from one modality flows into the other.
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
        Initialize the gated cross-modal attention module.

        Args:
            query_dim: Dimension of the query modality features
            key_dim: Dimension of the key/value modality features
            embed_dim: Dimension of the cross-attention output
            num_heads: Number of attention heads
            dropout: Dropout rate
        """
        super().__init__()

        # Base cross-attention mechanism
        self.cross_attention = CrossModalAttention(
            query_dim=query_dim,
            key_dim=key_dim,
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
        )

        # Gating mechanism
        # We'll add a projection layer to handle dimension mismatches
        self.query_dim = query_dim
        self.embed_dim = embed_dim

        # Add query projection if dimensions don't match
        self.query_proj_gate = (
            nn.Linear(query_dim, embed_dim) if query_dim != embed_dim else nn.Identity()
        )

        self.gate = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim),
            nn.Sigmoid(),
        )

    def forward(
        self,
        query_features: torch.Tensor,
        key_features: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:  # , torch.Tensor]:
        """
        Forward pass for gated cross-modal attention.

        Args:
            query_features: Features from the query modality [batch_size, query_seq_len, query_dim]
            key_features: Features from the key/value modality [batch_size, key_seq_len, key_dim]
            attention_mask: Optional mask for attention weights [batch_size, query_seq_len, key_seq_len]

        Returns:
            Tuple containing:
            - Updated query features with gated cross-modal information [batch_size, query_seq_len, embed_dim]
            - Attention weights [batch_size, num_heads, query_seq_len, key_seq_len]
            - Gate values [batch_size, query_seq_len, embed_dim]
        """
        # Apply cross-attention
        attended_features, attention_weights = self.cross_attention(
            query_features, key_features, attention_mask
        )

        # Project query features to match embedding dimension using the pre-defined projection
        query_features_projected = self.query_proj_gate(query_features)

        # Simplify the gating process - keep everything on the same device
        # Concatenate inputs on the original device
        gate_input = torch.cat([query_features_projected, attended_features], dim=-1)

        # Apply gate - all operations stay on the same device
        gate_values = self.gate(gate_input)

        # Apply gated residual connection
        output = query_features_projected + gate_values * attended_features

        return output, attention_weights  # , gate_values


# Continuing in src/models/vision/cross_modal_attention.py
class BidirectionalCrossAttention(nn.Module):
    """
    Bidirectional Cross-Attention module for multimodal fusion.

    This module implements bidirectional attention flow between two modalities,
    allowing each modality to attend to and be influenced by the other.
    """

    def __init__(
        self,
        vision_dim: int,
        text_dim: int,
        fusion_dim: int,
        num_heads: int = 8,
        dropout: float = 0.1,
    ):
        """
        Initialize bidirectional cross-attention module.

        Args:
            vision_dim: Dimension of vision features
            text_dim: Dimension of text features
            fusion_dim: Dimension of the fused representations
            num_heads: Number of attention heads
            dropout: Dropout rate
        """
        super().__init__()

        # Cross-attention from text to vision (text attends to vision)
        # ENHANCEMENT: Use GatedCrossModalAttention instead of CrossModalAttention
        self.text_to_vision_attn = GatedCrossModalAttention(
            query_dim=text_dim,
            key_dim=vision_dim,
            embed_dim=fusion_dim,
            num_heads=num_heads,
            dropout=dropout,
        )

        # Cross-attention from vision to text (vision attends to text)
        # ENHANCEMENT: Use GatedCrossModalAttention instead of CrossModalAttention
        self.vision_to_text_attn = GatedCrossModalAttention(
            query_dim=vision_dim,
            key_dim=text_dim,
            embed_dim=fusion_dim,
            num_heads=num_heads,
            dropout=dropout,
        )

        # Feed-forward networks for each modality after cross-attention
        self.vision_ff = nn.Sequential(
            nn.LayerNorm(fusion_dim),
            nn.Linear(fusion_dim, fusion_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(fusion_dim * 4, fusion_dim),
            nn.Dropout(dropout),
        )

        self.text_ff = nn.Sequential(
            nn.LayerNorm(fusion_dim),
            nn.Linear(fusion_dim, fusion_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(fusion_dim * 4, fusion_dim),
            nn.Dropout(dropout),
        )

        # Residual connections require input projections if dimensions don't match
        self.vision_input_proj = (
            nn.Linear(vision_dim, fusion_dim)
            if vision_dim != fusion_dim
            else nn.Identity()
        )
        self.text_input_proj = (
            nn.Linear(text_dim, fusion_dim) if text_dim != fusion_dim else nn.Identity()
        )

    def forward(
        self,
        vision_features: torch.Tensor,
        text_features: torch.Tensor,
        vision_mask: Optional[torch.Tensor] = None,
        text_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass for bidirectional cross-attention.

        Args:
            vision_features: Features from the vision modality [batch_size, vision_seq_len, vision_dim]
            text_features: Features from the text modality [batch_size, text_seq_len, text_dim]
            vision_mask: Optional mask for vision features [batch_size, vision_seq_len]
            text_mask: Optional mask for text features [batch_size, text_seq_len]

        Returns:
            Dictionary containing:
            - 'vision_features': Updated vision features with cross-modal information
            - 'text_features': Updated text features with cross-modal information
            - 'vision_to_text_attn': Attention weights from vision to text
            - 'text_to_vision_attn': Attention weights from text to vision
        """
        results = {}

        # Project inputs to common dimension if needed (for residual connections)
        vision_input = self.vision_input_proj(vision_features)
        text_input = self.text_input_proj(text_features)

        # Create attention masks if provided - simplified version
        if vision_mask is not None and text_mask is not None:
            # Get device
            device = vision_features.device

            # Ensure masks are on device
            vision_mask = vision_mask.to(device)
            text_mask = text_mask.to(device)

            # Get shapes for mask creation
            batch_size = vision_mask.shape[0]
            text_seq_len = text_mask.shape[1]
            vision_seq_len = vision_mask.shape[1]

            # Create simple masks that allow all tokens to attend to all other tokens
            # Create directly on the target device
            text_to_vision_mask = torch.ones(
                batch_size,
                text_seq_len,
                vision_seq_len,
                dtype=torch.bool,
                device=device,
            )

            vision_to_text_mask = torch.ones(
                batch_size,
                vision_seq_len,
                text_seq_len,
                dtype=torch.bool,
                device=device,
            )
        else:
            text_to_vision_mask = None
            vision_to_text_mask = None

        # Store original features
        vision_features_original = vision_features.clone()
        text_features_original = text_features.clone()

        # Text attends to vision
        text_features_updated, text_to_vision_attn = self.text_to_vision_attn(
            text_features_original, vision_features_original, text_to_vision_mask
        )

        # Vision attends to text
        vision_features_updated, vision_to_text_attn = self.vision_to_text_attn(
            vision_features_original, text_features_original, vision_to_text_mask
        )

        # Residual connections
        text_features = text_input + text_features_updated
        vision_features = vision_input + vision_features_updated

        # Feed-forward networks with residual connections
        text_features = text_features + self.text_ff(text_features)
        vision_features = vision_features + self.vision_ff(vision_features)

        # Store results
        results["vision_features"] = vision_features
        results["text_features"] = text_features
        results["vision_to_text_attn"] = vision_to_text_attn
        results["text_to_vision_attn"] = text_to_vision_attn

        return results


# Continuing in src/models/vision/cross_modal_attention.py
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
    ) -> Dict[str, torch.Tensor]:
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
