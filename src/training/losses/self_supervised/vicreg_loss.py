"""MODULE: vicreg_loss.py
PURPOSE: Implements VICReg (Variance-Invariance-Covariance Regularization) loss for representation learning.

KEY COMPONENTS:
- VICRegLoss: Main class implementing VICReg loss
- Variance regularization component
- Invariance regularization component
- Covariance regularization component
- Configurable component weights

DEPENDENCIES:
- torch
- torch.nn
- typing
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Optional, Union


class VICRegLoss(nn.Module):
    """
    VICReg loss as described in https://arxiv.org/abs/2105.04906

    Combines three terms:
    - Variance: encourages embeddings to be diverse (avoid collapse)
    - Invariance: aligns corresponding features
    - Covariance: decorrelates features within each embedding

    Enhanced with:
    - Coefficient warm-up for variance and covariance terms
    - Curriculum learning approach prioritizing invariance early in training
    """

    def __init__(
        self,
        sim_coeff: float = 10.0,  # Reduced from 50.0 to prevent instability
        var_coeff: float = 5.0,  # Keep variance coefficient
        cov_coeff: float = 1.0,
        epsilon: float = 1e-3,  # Increased from 1e-4 for better numerical stability
        warmup_epochs: int = 5,  # Epochs for warm-up
        curriculum: bool = True,  # Enable curriculum learning
        num_epochs: int = 30,  # Total epochs for training calculation
    ):
        """
        Initialize VICReg loss with curriculum learning support.

        Args:
            sim_coeff: Weight for invariance term (higher promotes feature alignment)
            var_coeff: Weight for variance term (lower prevents aggressive regularization)
            cov_coeff: Weight for covariance term
            epsilon: Small constant for numerical stability
            warmup_epochs: Number of epochs for coefficient warm-up
            curriculum: Whether to use curriculum learning
            num_epochs: Total epochs for training (helps with warmup calculation)
        """
        super().__init__()
        self.sim_coeff = sim_coeff
        self.var_coeff = var_coeff
        self.cov_coeff = cov_coeff
        self.epsilon = epsilon
        self.warmup_epochs = warmup_epochs
        self.curriculum = curriculum
        self.num_epochs = num_epochs

        # Track current epoch and step for warmup
        self.current_epoch = 0
        self.current_step = 0
        self.total_steps = 0

        # For logging
        self.effective_var_coeff = 0.0
        self.effective_cov_coeff = 0.0

        # For reducing verbosity
        self._print_counter = 0
        self._print_frequency = 200  # Print only every 200 steps

    def update_epoch(self, epoch: int) -> None:
        """Update the current epoch for curriculum learning.

        Args:
            epoch: Current training epoch
        """
        self.current_epoch = epoch

    def update_step(self, step: int, total_steps: int) -> None:
        """Update the current step for fine-grained warm-up.

        Args:
            step: Current global step
            total_steps: Total steps expected in training
        """
        self.current_step = step
        self.total_steps = total_steps

    def get_warmup_factor(self) -> float:
        """Calculate warm-up factor based on current epoch or step.

        Returns:
            Warm-up factor between 0 and 1
        """
        # Always have a minimum factor to avoid complete exclusion of regularization
        min_factor = 0.3  # Increased from 0.2 for better initial regularization

        # Fixed factor for early epochs
        if self.current_epoch == 0:
            # Use a very small factor in first epoch
            return 0.1
        elif self.current_epoch == 1:
            # Use a modest factor in second epoch
            return 0.3

        # Limit maximum factor to 0.5 for curriculum learning
        max_factor = 0.5

        # Attempt step-based warmup first
        if self.total_steps > 0 and self.current_step > 0:
            # Fine-grained warm-up based on steps if available
            steps_per_epoch = max(1, self.total_steps / max(self.num_epochs, 1))
            warmup_steps = self.warmup_epochs * steps_per_epoch

            # Use smoother growth curve (cubic) instead of square root
            progress = self.current_step / max(1, warmup_steps)
            raw_factor = min(
                max_factor, progress**0.33
            )  # Cubic root for even smoother growth
            return max(min_factor, raw_factor)

        # Fallback to epoch-based warmup with same smoothing
        progress = self.current_epoch / max(1, self.warmup_epochs)
        raw_factor = min(max_factor, progress**0.33)  # Cubic root for smoother growth
        return max(min_factor, raw_factor)

    def forward(
        self, z_a: torch.Tensor, z_b: torch.Tensor
    ) -> Dict[str, Union[torch.Tensor, float]]:
        """
        Compute VICReg loss between two sets of embeddings with curriculum learning.

        Args:
            z_a: First set of embeddings [batch_size, embedding_dim]
            z_b: Second set of embeddings [batch_size, embedding_dim]

        Returns:
            Dictionary with loss and component metrics
        """
        batch_size = z_a.size(0)

        # Control verbosity - only print every N steps
        should_print = self._print_counter % self._print_frequency == 0
        self._print_counter += 1

        if should_print:
            print(f"VICReg forward - z_a: {z_a.shape}, z_b: {z_b.shape}")

        # Apply curriculum learning if enabled
        if self.curriculum:
            warmup_factor = self.get_warmup_factor()

            # Start with very little variance/covariance regularization, then increase
            effective_var_coeff = self.var_coeff * warmup_factor
            effective_cov_coeff = self.cov_coeff * warmup_factor

            # For the first epoch, focus almost entirely on invariance (similarity)
            if self.current_epoch == 0:
                effective_var_coeff *= (
                    0.1  # Only 10% of standard variance regularization
                )
                effective_cov_coeff *= (
                    0.1  # Only 10% of standard covariance regularization
                )
        else:
            # No curriculum, use standard coefficients
            effective_var_coeff = self.var_coeff
            effective_cov_coeff = self.cov_coeff

        # Save effective coefficients for logging
        self.effective_var_coeff = effective_var_coeff
        self.effective_cov_coeff = effective_cov_coeff

        # Center the embeddings along the batch dimension
        z_a = z_a - z_a.mean(dim=0, keepdim=True)
        z_b = z_b - z_b.mean(dim=0, keepdim=True)

        # Invariance loss (MSE between embeddings)
        sim_loss = F.mse_loss(z_a, z_b)

        # Variance loss - ensures each dimension has unit variance
        std_z_a = torch.sqrt(z_a.var(dim=0) + self.epsilon)
        std_z_b = torch.sqrt(z_b.var(dim=0) + self.epsilon)
        std_loss = torch.mean(F.relu(1 - std_z_a)) + torch.mean(F.relu(1 - std_z_b))

        # Covariance loss - reduces correlation between dimensions
        cov_z_a = (z_a.T @ z_a) / (batch_size - 1)
        cov_z_b = (z_b.T @ z_b) / (batch_size - 1)

        # Zero out the diagonals
        mask = ~torch.eye(cov_z_a.shape[0], dtype=torch.bool, device=cov_z_a.device)
        cov_loss = (
            cov_z_a[mask].pow_(2).sum() / cov_z_a.shape[0]
            + cov_z_b[mask].pow_(2).sum() / cov_z_b.shape[0]
        )

        # Combine the losses with curriculum-adjusted coefficients
        loss = (
            self.sim_coeff * sim_loss
            + effective_var_coeff * std_loss
            + effective_cov_coeff * cov_loss
        )

        # Print loss components for debugging
        if should_print:
            print(
                f"VICReg loss components - Sim: {sim_loss.item():.4f}, Var: {std_loss.item():.4f}, Cov: {cov_loss.item():.4f}"
            )
            print(
                f"Using coefficients - Sim: {self.sim_coeff}, Var: {effective_var_coeff:.4f}, Cov: {effective_cov_coeff:.4f}"
            )

        # Return in the same format as other loss functions for compatibility with trainer
        return {
            "loss": loss,
            "invariance_loss": sim_loss.item(),
            "variance_loss": std_loss.item(),
            "covariance_loss": cov_loss.item(),
            "sim_weight": self.sim_coeff,
            "var_weight": effective_var_coeff,
            "cov_weight": effective_cov_coeff,
            "warmup_factor": warmup_factor if self.curriculum else 1.0,
        }


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
        "module_purpose": "Implements VICReg (Variance-Invariance-Covariance Regularization) loss for representation learning",
        "key_classes": [
            {
                "name": "VICRegLoss",
                "purpose": "Implements VICReg loss with variance, invariance, and covariance terms",
                "key_methods": [
                    {
                        "name": "__init__",
                        "signature": "__init__(self, sim_weight: float = 25.0, var_weight: float = 25.0, cov_weight: float = 1.0)",
                        "brief_description": "Initialize VICReg loss with component weights",
                    },
                    {
                        "name": "forward",
                        "signature": "forward(self, x: torch.Tensor, y: torch.Tensor) -> Dict[str, torch.Tensor]",
                        "brief_description": "Compute VICReg loss components",
                    },
                ],
                "inheritance": "nn.Module",
                "dependencies": ["torch", "torch.nn"],
            }
        ],
        "external_dependencies": ["torch", "typing"],
        "complexity_score": 7,
    }
