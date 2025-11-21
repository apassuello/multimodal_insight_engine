import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# Standard Transformer Attention
def scaled_dot_product_attention(query, key, value, mask=None):
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
    
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
        
    attention_weights = F.softmax(scores, dim=-1)
    return torch.matmul(attention_weights, value), attention_weights

# Enhanced Attention with Sparse Patterns and Linear Complexity
class EnhancedAttention(nn.Module):
    def __init__(self, d_model, num_heads, dropout=0.1, use_sparse_attention=True):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.use_sparse_attention = use_sparse_attention
        
        # Projections for Q, K, V
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        
        # Attention dropout
        self.dropout = nn.Dropout(dropout)
        
        # Optional sparse attention components
        if use_sparse_attention:
            self.sparse_block_size = 64
            self.global_tokens = 16  # Number of tokens that attend globally
    
    def forward(self, query, key, value, mask=None, attention_window=None):
        batch_size = query.size(0)
        
        # Project and reshape for multi-head attention
        q = self.q_proj(query).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(key).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(value).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        
        if self.use_sparse_attention and query.size(1) > self.sparse_block_size:
            # Use sparse attention for long sequences
            output, attention = self._sparse_attention(q, k, v, mask, attention_window)
        else:
            # Use standard attention for shorter sequences
            output, attention = self._standard_attention(q, k, v, mask)
        
        # Reshape and project back
        output = output.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        return self.out_proj(output), attention
        
    def _standard_attention(self, q, k, v, mask=None):
        # Regular scaled dot-product attention
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
            
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        return torch.matmul(attention_weights, v), attention_weights
        
    def _sparse_attention(self, q, k, v, mask=None, attention_window=None):
        # Implementation of sparse attention pattern
        # This could be local windowed attention + global tokens
        # or another efficient attention mechanism
        
        # Simplified implementation for illustration
        # In practice, you would use a more optimized implementation
        
        # Default window size if not specified
        window_size = attention_window or self.sparse_block_size
        seq_len = q.size(2)
        
        # Create a sparse attention mask that only allows attending to:
        # 1. Tokens within a local window
        # 2. Global tokens (e.g., first few tokens)
        sparse_mask = torch.zeros(seq_len, seq_len, device=q.device)
        
        # Allow local window attention
        for i in range(seq_len):
            window_start = max(0, i - window_size // 2)
            window_end = min(seq_len, i + window_size // 2 + 1)
            sparse_mask[i, window_start:window_end] = 1
            
        # Allow global tokens to attend to all positions
        sparse_mask[:self.global_tokens, :] = 1
        # Allow all positions to attend to global tokens
        sparse_mask[:, :self.global_tokens] = 1
        
        # Combine with the original attention mask if provided
        if mask is not None:
            combined_mask = mask * sparse_mask.unsqueeze(0).unsqueeze(0)
        else:
            combined_mask = sparse_mask.unsqueeze(0).unsqueeze(0)
        
        # Apply standard attention with the sparse mask
        return self._standard_attention(q, k, v, combined_mask)