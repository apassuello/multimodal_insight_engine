# src/utils/gradient_handler.py

import torch
import torch.nn as nn
from typing import Any, Dict, List, Optional, Union, Set, Callable
import logging
import matplotlib.pyplot as plt
import numpy as np
import os

logger = logging.getLogger(__name__)


class GradientHandler:
    """
    Handles gradient operations including clipping, monitoring, balancing, and analysis.

    This utility class helps:
    1. Monitor gradients to identify training issues
    2. Clip gradients to improve stability
    3. Balance gradients across modalities to prevent dominance
    4. Visualize gradient flow for diagnosis
    """

    def __init__(
        self,
        model: nn.Module,
        clip_value: Optional[float] = None,
        component_ratios: Optional[Dict[str, float]] = None,
        balance_modalities: bool = False,
        log_frequency: int = 100,
        visualization_dir: Optional[str] = None,
    ):
        """
        Initialize the gradient handler.

        Args:
            model: Model to handle gradients for
            clip_value: Value for gradient clipping (None to disable)
            component_ratios: Target ratios between component gradients (e.g., {"vision": 1.0, "text": 1.0})
            balance_modalities: Whether to balance gradients between modalities
            log_frequency: How often to log gradient statistics
            visualization_dir: Directory to save gradient visualizations (None to disable)
        """
        self.model = model
        self.clip_value = clip_value
        self.component_ratios = component_ratios or {}
        self.balance_modalities = balance_modalities
        self.log_frequency = log_frequency
        self.visualization_dir = visualization_dir

        if visualization_dir:
            os.makedirs(visualization_dir, exist_ok=True)

        # Statistics tracking
        self.step_count = 0
        self.grad_history = {
            "vision_grad_norm": [],
            "text_grad_norm": [],
            "fusion_grad_norm": [],
            "total_grad_norm": [],
            "vision_text_ratio": [],
            "step": [],
        }

        # For warning once per execution
        self._warned_missing_components = set()

    def clip_gradients(self) -> float:
        """
        Clip gradients if clip_value is set.

        Returns:
            Total gradient norm before clipping
        """
        if self.clip_value is None:
            return 0.0

        return nn.utils.clip_grad_norm_(self.model.parameters(), self.clip_value).item()

    def analyze_gradients(self) -> Dict[str, float]:
        """
        Analyze gradient norms across different model components.

        Returns:
            Dictionary with gradient statistics
        """
        self.step_count += 1

        # Initialize statistics
        stats = {
            "total_grad_norm": 0.0,
            "grad_count": 0,
            "has_nan": False,
            "has_inf": False,
            "max_grad_norm": 0.0,
            "min_grad_norm": float("inf"),
            "component_norms": {},
        }

        # Extract and organize parameters by component
        component_params = {}
        component_counts = {}

        # Define key component prefixes to track
        key_components = ["vision_model", "text_model", "fusion", "cross", "projection"]

        # Group parameters by component
        for name, param in self.model.named_parameters():
            if param.grad is None:
                continue

            # Determine parameter component
            component = "other"
            for prefix in key_components:
                if prefix in name:
                    component = prefix
                    break

            # Group by major modality for easier reporting
            if component.startswith("vision"):
                major_component = "vision"
            elif component.startswith("text"):
                major_component = "text"
            elif component in ("fusion", "cross"):
                major_component = "fusion"
            else:
                major_component = "other"

            # Initialize component tracking if needed
            if major_component not in component_params:
                component_params[major_component] = []
                component_counts[major_component] = 0

            # Add parameter and update count
            component_params[major_component].append(param)
            component_counts[major_component] += 1

            # Update global statistics
            param_norm = param.grad.norm().item()
            stats["total_grad_norm"] += param_norm
            stats["grad_count"] += 1
            stats["max_grad_norm"] = max(stats["max_grad_norm"], param_norm)
            if param_norm > 0:
                stats["min_grad_norm"] = min(stats["min_grad_norm"], param_norm)

            # Check for problematic gradients
            if torch.isnan(param.grad).any():
                stats["has_nan"] = True
            if torch.isinf(param.grad).any():
                stats["has_inf"] = True

        # Calculate component gradient norms
        for component, params in component_params.items():
            if params:
                # Consider all parameters in the component together
                grad_vector = []
                for p in params:
                    if p.grad is not None:
                        grad_vector.append(p.grad.view(-1))

                if grad_vector:
                    # Concatenate all gradients and compute norm
                    all_grads = torch.cat(grad_vector)
                    component_norm = all_grads.norm().item()
                    stats["component_norms"][component] = {
                        "norm": component_norm,
                        "count": component_counts[component],
                        "avg_norm": component_norm / component_counts[component],
                    }

        # Compute component ratios if we have both vision and text components
        vision_norm = stats["component_norms"].get("vision", {}).get("norm", 0.0)
        text_norm = stats["component_norms"].get("text", {}).get("norm", 0.0)

        if vision_norm > 0 and text_norm > 0:
            stats["vision_text_ratio"] = vision_norm / text_norm
        else:
            stats["vision_text_ratio"] = 0.0

        # Update history
        self.grad_history["step"].append(self.step_count)
        self.grad_history["vision_grad_norm"].append(vision_norm)
        self.grad_history["text_grad_norm"].append(text_norm)
        self.grad_history["fusion_grad_norm"].append(
            stats["component_norms"].get("fusion", {}).get("norm", 0.0)
        )
        self.grad_history["total_grad_norm"].append(stats["total_grad_norm"])
        self.grad_history["vision_text_ratio"].append(stats["vision_text_ratio"])

        # Log gradient statistics periodically
        if self.step_count % self.log_frequency == 0:
            self._log_gradient_stats(stats)

            # Generate visualization if enabled
            if self.visualization_dir:
                self._visualize_gradients()

        return stats

    def balance_component_gradients(self, optimizer: torch.optim.Optimizer) -> None:
        """
        Balance gradients between components according to target ratios.

        Args:
            optimizer: The optimizer to adjust learning rates for
        """
        if not self.balance_modalities or not self.component_ratios:
            return

        # Analyze current gradients
        stats = self.analyze_gradients()

        # Get component norms
        component_norms = {}
        for component, ratio in self.component_ratios.items():
            if component in stats["component_norms"]:
                component_norms[component] = stats["component_norms"][component]["norm"]
            else:
                if component not in self._warned_missing_components:
                    logger.warning(
                        f"Component '{component}' specified in ratios but not found in model"
                    )
                    self._warned_missing_components.add(component)
                component_norms[component] = 0.0

        # Skip if any component has zero gradient
        if any(norm == 0 for norm in component_norms.values()):
            return

        # Calculate current ratios relative to first component
        first_component = next(iter(self.component_ratios.keys()))
        current_ratios = {}
        for component, norm in component_norms.items():
            if norm > 0 and component_norms[first_component] > 0:
                current_ratios[component] = norm / component_norms[first_component]
            else:
                current_ratios[component] = 1.0

        # Calculate correction factors
        correction_factors = {}
        for component, target_ratio in self.component_ratios.items():
            if component in current_ratios and current_ratios[component] > 0:
                # Target ratio / current ratio gives the correction factor
                correction_factors[component] = target_ratio / current_ratios[component]
            else:
                correction_factors[component] = 1.0

            # Limit correction to avoid extreme changes
            correction_factors[component] = max(
                0.1, min(10.0, correction_factors[component])
            )

        # Apply corrections to learning rates in optimizer
        for i, param_group in enumerate(optimizer.param_groups):
            group_name = param_group.get("name", "")

            # Apply correction if this group matches a component
            for component, factor in correction_factors.items():
                if component in group_name.lower():
                    # Avoid extreme changes by using a dampened correction
                    dampened_factor = 1.0 + 0.1 * (factor - 1.0)  # Dampened adjustment

                    # Apply correction to learning rate
                    old_lr = param_group["lr"]
                    param_group["lr"] = old_lr * dampened_factor

                    # Log significant changes
                    if abs(dampened_factor - 1.0) > 0.05:
                        logger.info(
                            f"Adjusted LR for {group_name}: {old_lr:.6f} -> {param_group['lr']:.6f} "
                            f"(factor: {dampened_factor:.2f})"
                        )
                    break

    def _log_gradient_stats(self, stats: Dict[str, Any]) -> None:
        """
        Log gradient statistics for debugging.

        Args:
            stats: Gradient statistics from analyze_gradients
        """
        # Log problematic gradient conditions first
        if stats["has_nan"]:
            logger.error("NaN values detected in gradients!")
        if stats["has_inf"]:
            logger.error("Infinite values detected in gradients!")

        # Log total gradient statistics
        logger.info(f"Gradient stats (step {self.step_count}):")
        logger.info(
            f"  Total norm: {stats['total_grad_norm']:.4f}, "
            f"Max: {stats['max_grad_norm']:.4f}, "
            f"Min: {stats['min_grad_norm']:.4f}"
        )

        # Log component statistics
        for component, component_stats in stats["component_norms"].items():
            logger.info(
                f"  {component.capitalize()} norm: {component_stats['norm']:.4f} "
                f"(avg: {component_stats['avg_norm']:.4f}, count: {component_stats['count']})"
            )

        # Log vision-text ratio if available
        if stats["vision_text_ratio"] > 0:
            logger.info(
                f"  Vision/Text gradient ratio: {stats['vision_text_ratio']:.2f}"
            )

            # Warn if ratio is extremely unbalanced
            if stats["vision_text_ratio"] > 10.0 or stats["vision_text_ratio"] < 0.1:
                logger.warning(
                    f"Highly unbalanced gradients between vision ({stats['component_norms'].get('vision', {}).get('norm', 0.0):.4f}) "
                    f"and text ({stats['component_norms'].get('text', {}).get('norm', 0.0):.4f})"
                )

    def _visualize_gradients(self) -> None:
        """
        Create visualizations of gradient flow across components.
        """
        if not self.visualization_dir or not self.grad_history["step"]:
            return

        # Create a directory for gradient visualizations if needed
        os.makedirs(self.visualization_dir, exist_ok=True)

        # Create figure with four subplots
        fig, axs = plt.subplots(2, 2, figsize=(14, 10))

        # Plot 1: Component gradient norms over time
        ax = axs[0, 0]
        ax.plot(
            self.grad_history["step"],
            self.grad_history["vision_grad_norm"],
            label="Vision",
        )
        ax.plot(
            self.grad_history["step"], self.grad_history["text_grad_norm"], label="Text"
        )
        ax.plot(
            self.grad_history["step"],
            self.grad_history["fusion_grad_norm"],
            label="Fusion",
        )
        ax.set_title("Component Gradient Norms Over Time")
        ax.set_xlabel("Step")
        ax.set_ylabel("Gradient Norm")
        ax.legend()
        ax.grid(True)

        # Plot 2: Vision/Text gradient ratio over time
        ax = axs[0, 1]
        ax.plot(self.grad_history["step"], self.grad_history["vision_text_ratio"])
        ax.axhline(y=1.0, color="r", linestyle="--", alpha=0.5)  # Ideal balance line
        ax.set_title("Vision/Text Gradient Ratio Over Time")
        ax.set_xlabel("Step")
        ax.set_ylabel("Ratio")
        ax.set_yscale("log")  # Log scale for better visualization
        ax.grid(True)

        # Plot 3: Total gradient norm over time
        ax = axs[1, 0]
        ax.plot(self.grad_history["step"], self.grad_history["total_grad_norm"])
        ax.set_title("Total Gradient Norm Over Time")
        ax.set_xlabel("Step")
        ax.set_ylabel("Gradient Norm")
        ax.grid(True)

        # Plot 4: Recent gradient distribution histogram
        ax = axs[1, 1]
        # Get recent gradient norms
        recent_norms = {}
        for component in ["vision_grad_norm", "text_grad_norm", "fusion_grad_norm"]:
            if len(self.grad_history[component]) > 0:
                # Get most recent batch of values
                n_samples = min(100, len(self.grad_history[component]))
                values = np.array(self.grad_history[component][-n_samples:])
                if np.sum(values) > 0:  # Only include if we have non-zero values
                    recent_norms[component.split("_")[0]] = values

        if recent_norms:
            for component, values in recent_norms.items():
                ax.hist(values, bins=20, alpha=0.5, label=component)
            ax.set_title("Recent Gradient Norm Distribution")
            ax.set_xlabel("Gradient Norm")
            ax.set_ylabel("Frequency")
            ax.legend()
            ax.grid(True)
        else:
            ax.text(
                0.5,
                0.5,
                "Insufficient gradient data",
                horizontalalignment="center",
                verticalalignment="center",
                transform=ax.transAxes,
            )

        # Adjust layout and save
        plt.tight_layout()
        plt.savefig(
            os.path.join(
                self.visualization_dir, f"gradients_step_{self.step_count}.png"
            )
        )
        plt.close(fig)
