import torch
import torch.nn as nn

class ParallelTransformerBlock(nn.Module):
    """
    Advanced transformer block with parallel attention and feed-forward paths,
    similar to architectures used in modern language models like Claude.
    """
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        self.norm = RMSNorm(d_model)  # Assumes RMSNorm is imported
        self.attention = MultiHeadAttention(d_model, num_heads, dropout)  # Assumes MultiHeadAttention is imported
        self.feed_forward = SwiGLU(d_model, d_ff, d_model)  # Assumes SwiGLU is imported
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, mask=None):
        # Apply single normalization before splitting paths
        normalized = self.norm(x)
        
        # Process through attention and FFN in parallel
        attn_output, _ = self.attention(normalized, normalized, normalized, mask)
        ffn_output = self.feed_forward(normalized)
        
        # Combine results and apply residual connection
        return x + self.dropout(attn_output + ffn_output)