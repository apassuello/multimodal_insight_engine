"""MODULE: losses.py
PURPOSE: Implements custom loss functions for model training, including cross-entropy with label smoothing and mean squared error with additional features.

KEY COMPONENTS:
- CrossEntropyLoss: Cross-entropy loss with label smoothing for classification tasks
- MeanSquaredError: Mean squared error loss with support for weighted samples and reduction options

DEPENDENCIES:
- PyTorch (torch, torch.nn, torch.nn.functional)

SPECIAL NOTES:
- All loss classes inherit from nn.Module for PyTorch integration
- Includes support for label smoothing to improve model generalization
- Provides flexible reduction options for different training scenarios
"""

import os
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class CrossEntropyLoss(nn.Module):
    """
    Cross-entropy loss with label smoothing for classification tasks.
    
    This loss function extends the standard cross-entropy loss with label smoothing,
    which helps prevent overfitting by softening the target distribution. It also
    supports weighted samples and different reduction modes.
    
    Args:
        smoothing (float, optional): Label smoothing factor. Defaults to 0.1.
        reduction (str, optional): Specifies the reduction to apply to the output:
            'none' | 'mean' | 'sum'. Defaults to 'mean'.
        weight (torch.Tensor, optional): Manual rescaling weight given to each class.
            Defaults to None.
    """

    def __init__(
        self,
        smoothing: float = 0.1,
        reduction: str = 'mean',
        weight: Optional[torch.Tensor] = None
    ):
        super().__init__()
        self.smoothing = smoothing
        self.reduction = reduction
        self.weight = weight

    def forward(
        self,
        input: torch.Tensor,
        target: torch.Tensor,
        sample_weight: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Compute the cross-entropy loss with label smoothing.
        
        Args:
            input: Predicted logits of shape (N, C) where C is the number of classes
            target: Target indices of shape (N,) where values are 0 ≤ targets[i] ≤ C-1
            sample_weight: Optional weights for each sample of shape (N,)
            
        Returns:
            torch.Tensor: The computed loss value
        """
        # Apply label smoothing
        if self.smoothing > 0:
            n_classes = input.size(-1)
            one_hot = F.one_hot(target, n_classes).float()
            smooth_one_hot = one_hot * (1 - self.smoothing) + (self.smoothing / n_classes)
            log_probs = F.log_softmax(input, dim=-1)
            loss = -(smooth_one_hot * log_probs).sum(dim=-1)
        else:
            loss = F.cross_entropy(input, target, weight=self.weight, reduction='none')

        # Apply sample weights if provided
        if sample_weight is not None:
            loss = loss * sample_weight

        # Apply reduction
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:  # 'none'
            return loss


class MeanSquaredError(nn.Module):
    """
    Mean squared error loss with additional functionality.
    
    This loss function extends the standard MSE loss with support for weighted
    samples, reduction options, and optional gradient clipping.
    
    Args:
        reduction (str, optional): Specifies the reduction to apply to the output:
            'none' | 'mean' | 'sum'. Defaults to 'mean'.
        clip_grad (float, optional): Maximum gradient value for clipping.
            Defaults to None.
    """

    def __init__(
        self,
        reduction: str = 'mean',
        clip_grad: Optional[float] = None
    ):
        super().__init__()
        self.reduction = reduction
        self.clip_grad = clip_grad

    def forward(
        self,
        input: torch.Tensor,
        target: torch.Tensor,
        sample_weight: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Compute the mean squared error loss.
        
        Args:
            input: Predicted values of shape (N, *) where * means any number of dimensions
            target: Target values of the same shape as input
            sample_weight: Optional weights for each sample of shape (N,)
            
        Returns:
            torch.Tensor: The computed loss value
        """
        # Compute squared error
        loss = (input - target) ** 2

        # Apply sample weights if provided
        if sample_weight is not None:
            loss = loss * sample_weight.unsqueeze(-1)

        # Apply reduction
        if self.reduction == 'mean':
            loss = loss.mean()
        elif self.reduction == 'sum':
            loss = loss.sum()

        # Clip gradients if specified
        if self.clip_grad is not None:
            loss = torch.clamp(loss, max=self.clip_grad)

        return loss


def extract_file_metadata(file_path=__file__):
    """
    Extract structured metadata about this module.
    
    Args:
        file_path: Path to the source file (defaults to current file)
        
    Returns:
        dict: Structured metadata about the module's purpose and components
    """
    return {
        "filename": os.path.basename(file_path),
        "module_purpose": "Implements custom loss functions for model training with support for label smoothing and weighted samples",
        "key_classes": [
            {
                "name": "CrossEntropyLoss",
                "purpose": "Cross-entropy loss with label smoothing for classification tasks",
                "key_methods": [
                    {
                        "name": "forward",
                        "signature": "forward(self, input: torch.Tensor, target: torch.Tensor, sample_weight: Optional[torch.Tensor] = None) -> torch.Tensor",
                        "brief_description": "Computes cross-entropy loss with label smoothing and optional sample weights"
                    }
                ],
                "inheritance": "nn.Module",
                "dependencies": ["torch", "torch.nn", "torch.nn.functional"]
            },
            {
                "name": "MeanSquaredError",
                "purpose": "Mean squared error loss with support for weighted samples and gradient clipping",
                "key_methods": [
                    {
                        "name": "forward",
                        "signature": "forward(self, input: torch.Tensor, target: torch.Tensor, sample_weight: Optional[torch.Tensor] = None) -> torch.Tensor",
                        "brief_description": "Computes MSE loss with optional sample weights and gradient clipping"
                    }
                ],
                "inheritance": "nn.Module",
                "dependencies": ["torch", "torch.nn"]
            }
        ],
        "external_dependencies": ["torch"],
        "complexity_score": 4,  # Medium-low complexity as it's focused on loss function implementations
    }
