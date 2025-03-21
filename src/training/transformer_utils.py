# src/training/transformer_utils.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional, Any
import numpy as np

def create_padding_mask(seq: torch.Tensor, pad_idx: int) -> torch.Tensor:
    """
    Create a padding mask for attention.
    
    Args:
        seq: Tensor of sequence indices [batch_size, seq_len]
        pad_idx: Padding token index
        
    Returns:
        Boolean mask of shape [batch_size, 1, 1, seq_len]
        where True values are positions to attend to and False are padding
    """
    # Create mask for padding tokens (1 for padding, 0 for real tokens)
    mask = (seq == pad_idx).unsqueeze(1).unsqueeze(2)
    
    # Invert mask because in attention, 1 means "attend to" and 0 means "ignore"
    return ~mask  # True means "attend to", False means "ignore"

def create_causal_mask(seq_len: int, device: torch.device) -> torch.Tensor:
    """
    Create a causal mask to prevent attending to future tokens.
    
    Args:
        seq_len: Sequence length
        device: Device for the tensor
        
    Returns:
        Boolean mask of shape [1, 1, seq_len, seq_len]
        where True values are positions to attend to and False are future positions
    """
    # Create a mask for future positions (upper triangle)
    mask = torch.triu(torch.ones(seq_len, seq_len, device=device), diagonal=1)
    
    # Invert and reshape mask
    mask = (mask == 0).unsqueeze(0).unsqueeze(0)
    
    return mask  # True means "attend to", False means "ignore"

def create_combined_mask(seq: torch.Tensor, pad_idx: int) -> torch.Tensor:
    """
    Create a combined mask for causal attention and padding.
    
    Args:
        seq: Tensor of sequence indices [batch_size, seq_len]
        pad_idx: Padding token index
        
    Returns:
        Boolean mask of shape [batch_size, 1, seq_len, seq_len]
        where True values are positions to attend to and False are padding or future positions
    """
    # Get sequence length and device
    seq_len = seq.size(1)
    device = seq.device
    
    # Create padding mask [batch_size, 1, 1, seq_len]
    padding_mask = create_padding_mask(seq, pad_idx)
    
    # Create causal mask [1, 1, seq_len, seq_len]
    causal_mask = create_causal_mask(seq_len, device)
    
    # Combine masks: both must be True to attend
    # Broadcast padding_mask to match causal_mask dimensions
    combined_mask = padding_mask & causal_mask
    
    return combined_mask

def subsequent_mask(size: int, device: torch.device) -> torch.Tensor:
    """
    Create a mask for subsequent positions.
    
    Args:
        size: Size of the square mask
        device: Device for the mask tensor
        
    Returns:
        Mask tensor of shape [1, size, size]
    """
    # Create a matrix where the upper triangle is masked
    attn_shape = (1, size, size)
    mask = torch.triu(torch.ones(attn_shape, device=device), diagonal=1).bool()
    return ~mask  # Invert so True means "attend to"

class LabelSmoothing(nn.Module):
    """
    Label smoothing loss for transformer training.
    
    This implements label smoothing as described in "Rethinking the Inception
    Architecture for Computer Vision" (Szegedy et al., 2016).
    """
    
    def __init__(self, smoothing: float = 0.1, pad_idx: int = 0, reduction: str = "mean"):
        """
        Initialize the label smoothing loss.
        
        Args:
            smoothing: Smoothing factor (0 = no smoothing)
            pad_idx: Index of padding token (to ignore in loss)
            reduction: Loss reduction method ("mean", "sum", or "none")
        """
        super().__init__()
        self.smoothing = smoothing
        self.pad_idx = pad_idx
        self.reduction = reduction
        self.criterion = nn.KLDivLoss(reduction="none")
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Compute the smoothed loss.
        
        Args:
            pred: Predicted log probabilities [batch_size, seq_len, vocab_size]
            target: Target indices [batch_size, seq_len]
            
        Returns:
            Loss value
        """
        batch_size, seq_len, vocab_size = pred.size()
        
        # Create one-hot vectors for targets
        target_onehot = torch.zeros_like(pred).scatter_(2, target.unsqueeze(-1), 1.0)
        
        # Apply label smoothing
        target_smooth = target_onehot * (1.0 - self.smoothing)
        target_smooth += self.smoothing / vocab_size
        
        # Calculate KL divergence loss
        loss = self.criterion(pred.log_softmax(dim=-1), target_smooth)
        
        # Create a padding mask (1 for real tokens, 0 for padding)
        padding_mask = (target != self.pad_idx).float().unsqueeze(-1)
        
        # Apply padding mask to loss
        loss = loss * padding_mask
        
        # Apply reduction
        if self.reduction == "mean":
            return loss.sum() / padding_mask.sum()
        elif self.reduction == "sum":
            return loss.sum()
        else:  # 'none'
            return loss