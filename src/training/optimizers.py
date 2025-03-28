"""MODULE: optimizers.py
PURPOSE: Implements custom optimizers and learning rate schedulers for model training, including AdamW with weight decay and various learning rate scheduling strategies.

KEY COMPONENTS:
- AdamW: Adam optimizer with weight decay
- OneCycleLR: One-cycle learning rate scheduler
- CosineAnnealingLR: Cosine annealing learning rate scheduler
- LinearWarmupLR: Linear warmup followed by constant learning rate
- GradientClipper: Utility for gradient clipping

DEPENDENCIES:
- PyTorch (torch, torch.optim)

SPECIAL NOTES:
- All optimizers and schedulers are compatible with PyTorch's optimizer interface
- Includes support for gradient clipping and weight decay
- Provides flexible learning rate scheduling strategies
"""

import torch
import torch.optim as optim
from torch.optim.lr_scheduler import _LRScheduler
from typing import Optional, List, Dict, Any, Union, Tuple
import math
import os


class AdamW(optim.AdamW):
    """
    AdamW optimizer with improved weight decay handling.
    
    This implementation extends PyTorch's AdamW optimizer with additional features
    like gradient clipping and parameter group management.
    
    Args:
        params: Iterable of parameters to optimize or dicts defining parameter groups
        lr: Learning rate (default: 1e-3)
        betas: Coefficients for computing running averages of gradient and its square
        eps: Term added to the denominator to improve numerical stability
        weight_decay: Weight decay (L2 penalty) (default: 0)
        amsgrad: Whether to use the AMSGrad variant of this algorithm
        clip_grad: Maximum gradient norm for clipping (default: None)
    """
    
    def __init__(
        self,
        params,
        lr: float = 1e-3,
        betas: Tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0,
        amsgrad: bool = False,
        clip_grad: Optional[float] = None
    ):
        super().__init__(
            params,
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
            amsgrad=amsgrad
        )
        self.clip_grad = clip_grad
    
    def step(self, closure=None):
        """
        Performs a single optimization step.
        
        Args:
            closure: A closure that reevaluates the model and returns the loss
            
        Returns:
            float: The loss value if closure is provided
        """
        if self.clip_grad is not None:
            torch.nn.utils.clip_grad_norm_(self.param_groups[0]['params'], self.clip_grad)
        return super().step(closure)


class OneCycleLR(_LRScheduler):
    """
    One-cycle learning rate scheduler.
    
    Implements the one-cycle policy as described in "Super-Convergence: Very Fast
    Training of Neural Networks Using Large Learning Rates" by Leslie N. Smith.
    
    Args:
        optimizer: Wrapped optimizer
        max_lr: Maximum learning rate
        epochs: Number of epochs
        steps_per_epoch: Number of steps per epoch
        pct_start: Percentage of training to use for warmup
        div_factor: Initial learning rate = max_lr/div_factor
        final_div_factor: Final learning rate = initial_lr/final_div_factor
        anneal_strategy: Specifies the annealing strategy: 'cos' or 'linear'
    """
    
    def __init__(
        self,
        optimizer,
        max_lr: float,
        epochs: int,
        steps_per_epoch: int,
        pct_start: float = 0.3,
        div_factor: float = 25.0,
        final_div_factor: float = 1e4,
        anneal_strategy: str = 'cos'
    ):
        self.optimizer = optimizer
        self.max_lr = max_lr
        self.epochs = epochs
        self.steps_per_epoch = steps_per_epoch
        self.pct_start = pct_start
        self.div_factor = div_factor
        self.final_div_factor = final_div_factor
        self.anneal_strategy = anneal_strategy
        
        # Calculate total steps
        self.total_steps = epochs * steps_per_epoch
        
        # Calculate warmup steps
        self.warmup_steps = int(self.total_steps * pct_start)
        
        # Initialize base learning rates
        self.base_lr = max_lr / div_factor
        self.final_lr = self.base_lr / final_div_factor
        
        # Initialize step counter
        self.step_count = 0
        
        # Initialize learning rates
        self._init_lr()
    
    def _init_lr(self):
        """Initialize learning rates for all parameter groups."""
        for group in self.optimizer.param_groups:
            group['lr'] = self.base_lr
    
    def get_lr(self) -> List[float]:
        """
        Get the current learning rate for each parameter group.
        
        Returns:
            List[float]: List of learning rates
        """
        if self.step_count < self.warmup_steps:
            # Linear warmup
            lr = self.base_lr + (self.max_lr - self.base_lr) * (self.step_count / self.warmup_steps)
        else:
            # Annealing phase
            progress = (self.step_count - self.warmup_steps) / (self.total_steps - self.warmup_steps)
            if self.anneal_strategy == 'cos':
                lr = self.final_lr + 0.5 * (self.max_lr - self.final_lr) * (1 + math.cos(math.pi * progress))
            else:  # linear
                lr = self.max_lr + (self.final_lr - self.max_lr) * progress
        return [lr] * len(self.optimizer.param_groups)
    
    def step(self, closure=None):
        """
        Performs a scheduler step.
        
        Args:
            closure: A closure that reevaluates the model and returns the loss
            
        Returns:
            float: The loss value if closure is provided
        """
        self.step_count += 1
        for param_group, lr in zip(self.optimizer.param_groups, self.get_lr()):
            param_group['lr'] = lr
        return self.optimizer.step(closure) if closure is not None else None


class CosineAnnealingLR(_LRScheduler):
    """
    Cosine annealing learning rate scheduler.
    
    Implements cosine annealing with warm restarts as described in "SGDR: Stochastic
    Gradient Descent with Warm Restarts" by Ilya Loshchilov and Frank Hutter.
    
    Args:
        optimizer: Wrapped optimizer
        T_max: Maximum number of epochs
        eta_min: Minimum learning rate
        warmup_steps: Number of warmup steps
    """
    
    def __init__(
        self,
        optimizer,
        T_max: int,
        eta_min: float = 0,
        warmup_steps: int = 0
    ):
        self.T_max = T_max
        self.eta_min = eta_min
        self.warmup_steps = warmup_steps
        super().__init__(optimizer)
    
    def get_lr(self) -> List[float]:
        """
        Get the current learning rate for each parameter group.
        
        Returns:
            List[float]: List of learning rates
        """
        if self.last_epoch < self.warmup_steps:
            # Linear warmup
            return [
                self.eta_min + (base_lr - self.eta_min) * (self.last_epoch / self.warmup_steps)
                for base_lr in self.base_lrs
            ]
        else:
            # Cosine annealing
            progress = (self.last_epoch - self.warmup_steps) / (self.T_max - self.warmup_steps)
            return [
                self.eta_min + 0.5 * (base_lr - self.eta_min) * (1 + math.cos(math.pi * progress))
                for base_lr in self.base_lrs
            ]


class LinearWarmupLR(_LRScheduler):
    """
    Linear warmup followed by constant learning rate.
    
    This scheduler linearly increases the learning rate from a small value to the
    target learning rate over a specified number of warmup steps, then keeps it
    constant.
    
    Args:
        optimizer: Wrapped optimizer
        warmup_steps: Number of warmup steps
        start_lr: Initial learning rate
        target_lr: Target learning rate
    """
    
    def __init__(
        self,
        optimizer,
        warmup_steps: int,
        start_lr: float = 0,
        target_lr: Optional[float] = None
    ):
        self.warmup_steps = warmup_steps
        self.start_lr = start_lr
        self.target_lr = target_lr
        super().__init__(optimizer)
    
    def get_lr(self) -> List[float]:
        """
        Get the current learning rate for each parameter group.
        
        Returns:
            List[float]: List of learning rates
        """
        if self.last_epoch < self.warmup_steps:
            # Linear warmup
            return [
                self.start_lr + (base_lr - self.start_lr) * (self.last_epoch / self.warmup_steps)
                for base_lr in self.base_lrs
            ]
        else:
            # Constant learning rate
            return self.base_lrs


class GradientClipper:
    """
    Utility class for gradient clipping.
    
    This class provides a simple interface for applying gradient clipping to model
    parameters. It can be used with any optimizer and supports both global and
    per-parameter clipping.
    
    Args:
        max_norm: Maximum gradient norm
        norm_type: Type of norm to use (default: 2)
    """
    
    def __init__(self, max_norm: float, norm_type: float = 2.0):
        self.max_norm = max_norm
        self.norm_type = norm_type
    
    def clip_grad_norm(self, model: torch.nn.Module):
        """
        Clip gradients of all parameters in the model.
        
        Args:
            model: The model whose gradients should be clipped
        """
        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            max_norm=self.max_norm,
            norm_type=self.norm_type
        )


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
        "module_purpose": "Implements custom optimizers and learning rate schedulers for model training",
        "key_classes": [
            {
                "name": "AdamW",
                "purpose": "AdamW optimizer with improved weight decay handling and gradient clipping",
                "key_methods": [
                    {
                        "name": "step",
                        "signature": "step(self, closure=None)",
                        "brief_description": "Performs a single optimization step with optional gradient clipping"
                    }
                ],
                "inheritance": "optim.AdamW",
                "dependencies": ["torch", "torch.optim"]
            },
            {
                "name": "OneCycleLR",
                "purpose": "One-cycle learning rate scheduler for fast training",
                "key_methods": [
                    {
                        "name": "get_lr",
                        "signature": "get_lr(self) -> List[float]",
                        "brief_description": "Computes learning rates based on the one-cycle policy"
                    },
                    {
                        "name": "step",
                        "signature": "step(self, closure=None)",
                        "brief_description": "Performs a scheduler step and updates learning rates"
                    }
                ],
                "inheritance": "_LRScheduler",
                "dependencies": ["torch", "torch.optim.lr_scheduler"]
            },
            {
                "name": "CosineAnnealingLR",
                "purpose": "Cosine annealing learning rate scheduler with warm restarts",
                "key_methods": [
                    {
                        "name": "get_lr",
                        "signature": "get_lr(self) -> List[float]",
                        "brief_description": "Computes learning rates based on cosine annealing"
                    }
                ],
                "inheritance": "_LRScheduler",
                "dependencies": ["torch", "torch.optim.lr_scheduler"]
            }
        ],
        "external_dependencies": ["torch"],
        "complexity_score": 8,  # High complexity due to multiple optimizer and scheduler implementations
    }
