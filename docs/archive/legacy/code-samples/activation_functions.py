import torch
import torch.nn as nn
import torch.nn.functional as F

class SwiGLU(nn.Module):
    """
    SwiGLU activation function (variant of GLU) used in advanced transformers
    that combines a sigmoid-weighted linear unit with residual connections.
    """
    def __init__(self, in_features, hidden_features, out_features):
        super().__init__()
        self.gate_proj = nn.Linear(in_features, hidden_features)
        self.up_proj = nn.Linear(in_features, hidden_features)
        self.down_proj = nn.Linear(hidden_features, out_features)
        
    def forward(self, x):
        gate = self.gate_proj(x)
        up = self.up_proj(x)
        return self.down_proj(F.silu(gate) * up)