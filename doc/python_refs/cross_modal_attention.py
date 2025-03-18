import torch
import torch.nn as nn

class CrossModalAttention(nn.Module):
    """
    Cross-modal attention for integrating text and image features.
    This mechanism allows text features to attend to image features
    and vice versa, enabling multimodal understanding.
    """
    def __init__(self, embed_dim, num_heads, dropout=0.1):
        super().__init__()
        
        # Multi-head attention for cross-modal attention
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=embed_dim, 
            num_heads=num_heads,
            dropout=dropout
        )
        
        # Layer normalization and residual connection components
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        
        # Feed-forward network
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, 4 * embed_dim),
            nn.GELU(),
            nn.Linear(4 * embed_dim, embed_dim)
        )
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, text_features, image_features):
        """
        Forward pass for cross-modal attention
        
        Args:
            text_features: Text embeddings [batch_size, seq_len, embed_dim]
            image_features: Image embeddings [batch_size, num_patches+1, embed_dim]
            
        Returns:
            Fused multimodal features
        """
        # Apply cross-attention: text attends to image
        # First normalize the text features
        norm_text = self.norm1(text_features)
        
        # Apply cross-attention (text as query, image as key/value)
        # Need to convert from [batch, seq, dim] to [seq, batch, dim] for nn.MultiheadAttention
        attn_output, _ = self.cross_attention(
            query=norm_text.transpose(0, 1),
            key=image_features.transpose(0, 1),
            value=image_features.transpose(0, 1)
        )
        
        # Convert back to [batch, seq, dim] and add residual connection
        attn_output = attn_output.transpose(0, 1)
        text_features = text_features + self.dropout(attn_output)
        
        # Apply feed-forward network with residual connection
        text_features = text_features + self.dropout(
            self.ffn(self.norm2(text_features))
        )
        
        return text_features