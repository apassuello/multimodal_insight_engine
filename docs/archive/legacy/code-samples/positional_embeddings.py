import torch

def apply_rotary_embeddings(q, k, pos, theta=10000):
    """
    Apply Rotary Position Embeddings (RoPE) to queries and keys.
    
    RoPE encodes relative position information directly in self-attention and
    enables better generalization to sequence lengths not seen during training.
    
    Args:
        q: Query tensor [batch_size, heads, seq_len, head_dim]
        k: Key tensor [batch_size, heads, seq_len, head_dim]
        pos: Position indices [seq_len]
        theta: Base value for frequency (default: 10000)
    
    Returns:
        q_embed: Queries with positional encoding
        k_embed: Keys with positional encoding
    """
    device = q.device
    d = q.size(-1)
    
    # Create sinusoidal positions
    inv_freq = 1.0 / (theta ** (torch.arange(0, d, 2, device=device).float() / d))
    sinusoid_inp = torch.einsum("i,j->ij", pos.float(), inv_freq)
    sin = torch.sin(sinusoid_inp)
    cos = torch.cos(sinusoid_inp)
    
    # Apply rotations - each dimension is rotated by a different frequency
    q_embed = torch.cat([q[..., ::2] * cos - q[..., 1::2] * sin, 
                         q[..., 1::2] * cos + q[..., ::2] * sin], dim=-1)
    k_embed = torch.cat([k[..., ::2] * cos - k[..., 1::2] * sin,
                         k[..., 1::2] * cos + k[..., ::2] * sin], dim=-1)
    
    return q_embed, k_embed