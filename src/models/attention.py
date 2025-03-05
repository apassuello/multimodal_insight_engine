# Structure outline
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class SimpleAttention(nn.Module):
    """
    A simple attention mechanism.
    """
    
    def __init__(self, input_dim, attention_dim=None):
        """
        Initialize the attention mechanism.
        
        Args:
            input_dim (int): Dimension of input features
            attention_dim (int, optional): Dimension of attention space
        """
        # We'll implement this together
    
    def forward(self, query, key, value):
        """
        Forward pass of the attention mechanism.
        
        Args:
            query (torch.Tensor): Query tensor
            key (torch.Tensor): Key tensor
            value (torch.Tensor): Value tensor
            
        Returns:
            torch.Tensor: Context vector after attention
            torch.Tensor: Attention weights
        """
        # We'll implement this together