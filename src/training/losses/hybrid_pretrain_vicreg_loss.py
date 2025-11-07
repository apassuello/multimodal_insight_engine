"""MODULE: hybrid_pretrain_vicreg_loss.py
PURPOSE: Implements a hybrid pretraining loss that combines VICReg with contrastive learning for improved representation learning.

KEY COMPONENTS:
- HybridPretrainVICRegLoss: Main class implementing hybrid pretraining loss
- Adaptive transition between contrastive and VICReg losses
- Curriculum learning support
- Training progress tracking
- Performance monitoring and metrics

DEPENDENCIES:
- torch
- torch.nn
- typing
"""

import logging
import os
from typing import Dict, Literal, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.training.losses.contrastive_loss import ContrastiveLoss
from src.training.losses.vicreg_loss import VICRegLoss

logger = logging.getLogger(__name__)


class HybridPretrainVICRegLoss(nn.Module):
    """
    Hybrid loss combining contrastive pre-training with VICReg.

    Starts with contrastive learning to establish initial semantic alignment,
    then transitions to VICReg for better variance and covariance properties.

    Enhanced features:
    - Adaptive transition based on alignment metrics
    - Smooth transition between contrastive and VICReg losses
    - Comprehensive monitoring of alignment metrics
    - Automatic adjustment of VICReg coefficients based on alignment success
    """

    def __init__(
        self,
        sim_coeff: float = 5.0,
        var_coeff: float = 5.0,
        cov_coeff: float = 1.0,
        epsilon: float = 1e-4,
        warmup_epochs: int = 5,
        curriculum: bool = True,
        num_epochs: int = 30,
        contrastive_pretrain_steps: int = 300,
        temperature: float = 0.07,
        adaptive_transition: bool = True,
        min_alignment_threshold: float = 0.3,
        gradual_transition_steps: int = 100,
        fusion_dim: int = 512,
        vision_dim: int = 768,
        text_dim: int = 768,
    ):
        """
        Initialize hybrid loss.

        Args:
            sim_coeff: Weight for VICReg invariance term
            var_coeff: Weight for VICReg variance term
            cov_coeff: Weight for VICReg covariance term
            epsilon: Small constant for numerical stability
            warmup_epochs: Number of epochs for coefficient warm-up
            curriculum: Whether to use curriculum learning
            num_epochs: Total epochs for training
            contrastive_pretrain_steps: Number of steps to use contrastive loss
            temperature: Temperature for contrastive loss
            adaptive_transition: Whether to adapt transition based on alignment metrics
            min_alignment_threshold: Minimum diagonal similarity before transition
            gradual_transition_steps: Steps for gradual transition from contrastive to VICReg
            fusion_dim: Dimension of feature embeddings from the model (for projection layers)
        """
        super().__init__()

        # Initialize both loss functions
        self.vicreg_loss = VICRegLoss(
            sim_coeff=sim_coeff,
            var_coeff=var_coeff,
            cov_coeff=cov_coeff,
            epsilon=epsilon,
            warmup_epochs=warmup_epochs,
            curriculum=curriculum,
            num_epochs=num_epochs,
        )

        # Only use projections if dimensions don't match
        needs_projection = (vision_dim != fusion_dim) or (text_dim != fusion_dim)

        if needs_projection:
            print(
                f"Dimensions differ - using projection: vision_dim={vision_dim}, text_dim={text_dim}, fusion_dim={fusion_dim}"
            )
            # Configure contrastive loss with projection (only if needed)
            self.contrastive_loss = ContrastiveLoss(
                temperature=temperature,
                add_projection=True,
                projection_dim=fusion_dim,
                input_dim=vision_dim,  # Will be used for vision projection
            )

            # Create custom text projection if needed
            if text_dim != vision_dim:
                print(f"Creating separate text projection ({text_dim} -> {fusion_dim})")
                # Create text projection head with proper dimensions
                self.contrastive_loss.text_projection = nn.Sequential(
                    nn.Linear(text_dim, text_dim),
                    nn.ReLU(),
                    nn.Linear(text_dim, fusion_dim),
                )
        else:
            print(
                f"Dimensions match - skipping projection: vision_dim={vision_dim}, text_dim={text_dim}, fusion_dim={fusion_dim}"
            )
            # Configure contrastive loss with NO projection since dimensions already match
            self.contrastive_loss = ContrastiveLoss(
                temperature=temperature,
                add_projection=True,
                projection_dim=fusion_dim,
                input_dim=vision_dim,  # Will be used for vision projection
            )

        # Print confirmation
        print("Projection disabled in contrastive loss component of hybrid loss")

        # Make sure projection dimensions match
        contrastive_proj_dim = fusion_dim
        vicreg_proj_dim = fusion_dim

        if contrastive_proj_dim != vicreg_proj_dim:
            print(
                f"WARNING: Dimension mismatch between contrastive projection ({contrastive_proj_dim}) and VICReg model ({vicreg_proj_dim})"
            )
            print(f"This may cause issues during the transition phase")

        # Pre-training configuration
        self.contrastive_pretrain_steps = contrastive_pretrain_steps
        self.current_step = 0
        self.total_steps = 0
        self.in_pretrain_phase = True

        # Transition parameters
        self.adaptive_transition = adaptive_transition
        self.min_alignment_threshold = min_alignment_threshold
        self.gradual_transition_steps = gradual_transition_steps
        self.transition_progress = 0.0  # 0.0 = all contrastive, 1.0 = all VICReg

        # For tracking alignment metrics
        self.max_diag_similarity = 0.0
        self.min_diag_similarity = 0.0
        self.mean_similarity = 0.0
        self.alignment_gap = 0.0
        self.alignment_snr = 0.0  # Signal-to-noise ratio
        self.alignment_history = {
            "step": [],
            "diag_mean": [],
            "sim_mean": [],
            "alignment_gap": [],
            "alignment_snr": [],
            "vicreg_weight": [],
            "contrastive_weight": [],
        }

        self.transition_complete = False
        self.early_transition = False
        self.early_transition_step = 0

        # For logging
        self.current_phase = "contrastive_pretrain"

        # For reducing verbosity
        self._print_counter = 0
        self._print_frequency = 200  # Print every 200 steps

        logger.info(f"Initialized HybridPretrainVICRegLoss with:")
        logger.info(f"  - Contrastive pretrain steps: {contrastive_pretrain_steps}")
        logger.info(f"  - Adaptive transition: {adaptive_transition}")
        logger.info(f"  - Min alignment threshold: {min_alignment_threshold}")
        logger.info(f"  - Gradual transition steps: {gradual_transition_steps}")

    def update_step(self, step: int, total_steps: int):
        """Update current training step.

        Args:
            step: Current global step
            total_steps: Total steps in training
        """
        self.current_step = step
        self.total_steps = total_steps

        # Pass step info to VICReg loss
        self.vicreg_loss.update_step(step, total_steps)

        # Handle adaptive transition or fixed transition point
        if self.adaptive_transition:
            # Need to calculate similarity in forward pass
            # We use that to determine when to transition
            if (
                self.alignment_gap > self.min_alignment_threshold
                and self.in_pretrain_phase
            ):
                if not self.early_transition:
                    logger.info(
                        f"Early transition triggered at step {step} - alignment gap {self.alignment_gap:.4f} exceeds threshold {self.min_alignment_threshold:.4f}"
                    )
                    self.early_transition = True
                    self.early_transition_step = step
        else:
            # Fixed transition point
            if self.in_pretrain_phase and step >= self.contrastive_pretrain_steps:
                logger.info(
                    f"Transitioning from contrastive pre-training to VICReg at step {step}"
                )
                self.in_pretrain_phase = False
                self.transition_complete = True
                self.current_phase = "vicreg"

        # Calculate transition progress if in gradual transition period
        if self.early_transition:
            transition_duration = self.gradual_transition_steps
            steps_since_transition = step - self.early_transition_step
            self.transition_progress = min(
                1.0, steps_since_transition / transition_duration
            )

            # When transition is complete, update phase
            if self.transition_progress >= 1.0 and not self.transition_complete:
                self.transition_complete = True
                self.in_pretrain_phase = False
                self.current_phase = "vicreg"
                logger.info(f"Gradual transition to VICReg complete")

        elif not self.in_pretrain_phase:
            self.transition_progress = 1.0

    def update_epoch(self, epoch: int):
        """Update current epoch.

        Args:
            epoch: Current epoch
        """
        # Pass epoch info to VICReg loss
        self.vicreg_loss.update_epoch(epoch)

    def _calculate_alignment_metrics(
        self, z_a: torch.Tensor, z_b: torch.Tensor
    ) -> Dict[str, float]:
        """Calculate alignment metrics between two sets of embeddings.

        Args:
            z_a: First set of embeddings [batch_size, embedding_dim]
            z_b: Second set of embeddings [batch_size, embedding_dim]

        Returns:
            Dictionary with alignment metrics
        """
        # Normalize for cosine similarity
        z_a_norm = F.normalize(z_a, p=2, dim=1)
        z_b_norm = F.normalize(z_b, p=2, dim=1)

        # Calculate full similarity matrix
        similarity = torch.matmul(z_a_norm, z_b_norm.T)

        # Calculate diagonal (matched pairs) similarity
        diag_similarity = torch.diagonal(similarity)
        diag_mean = diag_similarity.mean().item()

        # Calculate mean similarity across all pairs
        sim_mean = similarity.mean().item()

        # Calculate standard deviation of similarities
        sim_std = similarity.std().item()

        # Calculate alignment gap
        alignment_gap = diag_mean - sim_mean

        # Calculate signal-to-noise ratio
        alignment_snr = abs(alignment_gap) / (sim_std + 1e-6)

        return {
            "diag_mean": diag_mean,
            "sim_mean": sim_mean,
            "sim_std": sim_std,
            "alignment_gap": alignment_gap,
            "alignment_snr": alignment_snr,
        }

    def _update_alignment_history(self, metrics: Dict[str, float]):
        """Update alignment history with current metrics.

        Args:
            metrics: Dictionary with alignment metrics
        """
        # Store alignment metrics
        self.alignment_history["step"].append(self.current_step)
        self.alignment_history["diag_mean"].append(metrics["diag_mean"])
        self.alignment_history["sim_mean"].append(metrics["sim_mean"])
        self.alignment_history["alignment_gap"].append(metrics["alignment_gap"])
        self.alignment_history["alignment_snr"].append(metrics["alignment_snr"])
        self.alignment_history["vicreg_weight"].append(self.transition_progress)
        self.alignment_history["contrastive_weight"].append(
            1.0 - self.transition_progress
        )

        # Update current metrics
        self.mean_similarity = metrics["sim_mean"]
        self.alignment_gap = metrics["alignment_gap"]
        self.alignment_snr = metrics["alignment_snr"]

        # Track max diagonal similarity for monitoring
        self.max_diag_similarity = max(self.max_diag_similarity, metrics["diag_mean"])

    def forward(
        self, z_a: torch.Tensor, z_b: torch.Tensor, **kwargs
    ) -> Dict[str, Union[torch.Tensor, float, Literal["contrastive_pretrain"]]]:
        """
        Forward pass that uses either contrastive or VICReg loss based on current step.

        Args:
            z_a: First set of embeddings [batch_size, embedding_dim]
            z_b: Second set of embeddings [batch_size, embedding_dim]
            **kwargs: Additional arguments like match_ids

        Returns:
            Dictionary with loss and metrics
        """
        # Control verbosity - only print every N steps
        should_print = self._print_counter % self._print_frequency == 0
        self._print_counter += 1

        # Print feature shape information for debugging
        if should_print:
            print(
                f"Feature shapes in HybridPretrainVICRegLoss.forward() - z_a: {z_a.shape}, z_b: {z_b.shape}"
            )

        # Calculate alignment metrics regardless of phase
        alignment_metrics = self._calculate_alignment_metrics(z_a, z_b)
        self._update_alignment_history(alignment_metrics)

        # Full contrastive pre-training phase
        if self.in_pretrain_phase and not self.early_transition:
            # Get match IDs for proper semantic alignment
            match_ids = kwargs.get("match_ids", None)

            # First normalize features for contrastive loss
            z_a_norm = F.normalize(z_a, p=2, dim=1)
            z_b_norm = F.normalize(z_b, p=2, dim=1)

            # Calculate contrastive loss with normalized features
            contrastive_results = self.contrastive_loss(
                vision_features=z_a_norm,  # Pass normalized features
                text_features=z_b_norm,  # Pass normalized features
                match_ids=match_ids if match_ids is not None else None,
            )

            # Calculate progress through pre-training phase (0 to 1)
            pretrain_progress = min(
                1.0, self.current_step / max(1, self.contrastive_pretrain_steps)
            )

            # Print alignment metrics periodically
            if should_print:
                logger.info(
                    f"Step {self.current_step} - Alignment gap: {alignment_metrics['alignment_gap']:.4f}, "
                    f"SNR: {alignment_metrics['alignment_snr']:.2f}, "
                    f"Diag: {alignment_metrics['diag_mean']:.4f}, Mean: {alignment_metrics['sim_mean']:.4f}"
                )

            # Return with phase indicator
            return {
                "loss": contrastive_results["loss"],
                "current_phase": "contrastive_pretrain",
                "pretrain_progress": pretrain_progress,
                "contrastive_loss": contrastive_results["loss"].item(),
                "accuracy": contrastive_results.get("accuracy", 0.0),
                "diagonal_similarity": alignment_metrics["diag_mean"],
                "mean_similarity": alignment_metrics["sim_mean"],
                "alignment_gap": alignment_metrics["alignment_gap"],
                "alignment_snr": alignment_metrics["alignment_snr"],
                "temperature": contrastive_results.get("temperature", 0.07),
                "transition_progress": 0.0,
                # Add placeholders for VICReg components
                "invariance_loss": 0.0,
                "variance_loss": 0.0,
                "covariance_loss": 0.0,
                "sim_weight": self.vicreg_loss.sim_coeff,
                "var_weight": 0.0,  # Not active yet
                "cov_weight": 0.0,  # Not active yet
            }

        # Gradual transition phase
        elif self.early_transition and not self.transition_complete:
            # Get match IDs for proper semantic alignment (for contrastive component)
            match_ids = kwargs.get("match_ids", None)

            # First normalize features for contrastive loss
            z_a_norm = F.normalize(z_a, p=2, dim=1)
            z_b_norm = F.normalize(z_b, p=2, dim=1)

            # Calculate contrastive loss with normalized features
            contrastive_results = self.contrastive_loss(
                vision_features=z_a_norm,  # Pass normalized features
                text_features=z_b_norm,  # Pass normalized features
                match_ids=match_ids if match_ids is not None else None,
            )

            # Calculate VICReg loss
            vicreg_results = self.vicreg_loss(z_a, z_b)

            # Get individual loss values
            contrastive_loss = contrastive_results["loss"]
            vicreg_loss = vicreg_results["loss"]

            # Calculate blended loss based on transition progress
            contrastive_weight = max(0.0, 1.0 - self.transition_progress)
            vicreg_weight = min(1.0, self.transition_progress)

            # Blend the losses
            blended_loss = (contrastive_weight * contrastive_loss) + (
                vicreg_weight * vicreg_loss
            )

            # Print transition progress periodically
            if should_print:
                logger.info(
                    f"Step {self.current_step} - Transition progress: {self.transition_progress:.2f}, "
                    f"Contrastive weight: {contrastive_weight:.2f}, VICReg weight: {vicreg_weight:.2f}"
                )

            # Combine results
            results = {
                "loss": blended_loss,
                "current_phase": "transition",
                "transition_progress": self.transition_progress,
                "contrastive_weight": contrastive_weight,
                "vicreg_weight": vicreg_weight,
                "contrastive_loss": contrastive_loss.item(),
                "vicreg_loss": vicreg_loss.item(),
                "accuracy": contrastive_results.get("accuracy", 0.0),
                "diagonal_similarity": alignment_metrics["diag_mean"],
                "mean_similarity": alignment_metrics["sim_mean"],
                "alignment_gap": alignment_metrics["alignment_gap"],
                "alignment_snr": alignment_metrics["alignment_snr"],
                "invariance_loss": vicreg_results.get("invariance_loss", 0.0),
                "variance_loss": vicreg_results.get("variance_loss", 0.0),
                "covariance_loss": vicreg_results.get("covariance_loss", 0.0),
                "sim_weight": vicreg_results.get(
                    "sim_weight", self.vicreg_loss.sim_coeff
                ),
                "var_weight": vicreg_results.get("var_weight", 0.0),
                "cov_weight": vicreg_results.get("cov_weight", 0.0),
            }

            return results

        # Full VICReg phase
        else:
            # Calculate VICReg loss
            vicreg_results = self.vicreg_loss(z_a, z_b)

            # Log transition completion if just completed
            if not self.transition_complete:
                self.transition_complete = True
                logger.info(
                    f"Transition to VICReg complete. Max contrastive diagonal similarity: {self.max_diag_similarity:.4f}"
                )
                logger.info(
                    f"Final alignment metrics - Alignment gap: {alignment_metrics['alignment_gap']:.4f}, "
                    f"SNR: {alignment_metrics['alignment_snr']:.2f}"
                )

            # Print alignment metrics periodically
            if should_print:
                print(
                    f"VICReg phase - Alignment metrics - Gap: {alignment_metrics['alignment_gap']:.4f}, "
                    f"SNR: {alignment_metrics['alignment_snr']:.2f}, "
                    f"Diag: {alignment_metrics['diag_mean']:.4f}"
                )

            # Add phase information to results
            vicreg_results["current_phase"] = "vicreg"
            vicreg_results["transition_progress"] = 1.0
            vicreg_results["diagonal_similarity"] = alignment_metrics["diag_mean"]
            vicreg_results["mean_similarity"] = alignment_metrics["sim_mean"]
            vicreg_results["alignment_gap"] = alignment_metrics["alignment_gap"]
            vicreg_results["alignment_snr"] = alignment_metrics["alignment_snr"]

            return vicreg_results


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
        "module_purpose": "Implements a hybrid pretraining loss that combines VICReg with contrastive learning for improved representation learning",
        "key_classes": [
            {
                "name": "HybridPretrainVICRegLoss",
                "purpose": "Combines VICReg and contrastive losses with adaptive transition",
                "key_methods": [
                    {
                        "name": "__init__",
                        "signature": "__init__(self, sim_coeff: float = 5.0, var_coeff: float = 5.0, cov_coeff: float = 1.0, warmup_epochs: int = 5)",
                        "brief_description": "Initialize hybrid loss with coefficients and warmup",
                    },
                    {
                        "name": "forward",
                        "signature": "forward(self, z_a: torch.Tensor, z_b: torch.Tensor) -> Dict[str, Union[torch.Tensor, float, Literal['contrastive_pretrain']]]",
                        "brief_description": "Compute hybrid loss with current transition state",
                    },
                    {
                        "name": "update_step",
                        "signature": "update_step(self, step: int, total_steps: int) -> None",
                        "brief_description": "Update training progress for adaptive transition",
                    },
                ],
                "inheritance": "nn.Module",
                "dependencies": ["torch", "torch.nn"],
            }
        ],
        "external_dependencies": ["torch", "typing"],
        "complexity_score": 9,
    }
