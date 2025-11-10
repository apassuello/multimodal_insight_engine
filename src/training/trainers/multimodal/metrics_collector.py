"""MODULE: metrics_collector.py
PURPOSE: Collects, tracks, and visualizes training metrics during model training.

KEY COMPONENTS:
- MetricsCollector: Manages metrics collection and history tracking
- Supports nested metrics (e.g., recalls, precisions with multiple values)
- Provides visualization capabilities via matplotlib

DEPENDENCIES:
- PyTorch (torch)
- Matplotlib for visualization
- NumPy for numerical operations
- Python standard library (os, logging, collections)

SPECIAL NOTES:
- Handles both simple scalar metrics and nested dictionary metrics
- Automatically converts PyTorch tensors to CPU for plotting
- Supports metric grouping and visualization
"""

import os
import logging
from collections import defaultdict
from typing import Dict, List, Optional, Any, Union

import torch
import numpy as np
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)


class MetricsCollector:
    """Collects and tracks training metrics with visualization support."""

    def __init__(self):
        """Initialize the metrics collector."""
        self.history = defaultdict(list)
        self._alignment_history = {
            "step": [],
            "diag_mean": [],
            "sim_mean": [],
            "alignment_gap": [],
            "alignment_snr": [],
        }

    def update(self, metrics: Dict[str, Union[float, Dict[str, float]]], prefix: str = "") -> None:
        """
        Update metrics history.

        Args:
            metrics: Dictionary of metrics (can contain nested dictionaries)
            prefix: Prefix to add to metric names (e.g., 'train', 'val')
        """
        for k, v in metrics.items():
            if isinstance(v, dict):
                # Handle nested metrics (e.g., recalls, precisions)
                for sub_k, sub_v in v.items():
                    key = f"{prefix}_{k}.{sub_k}" if prefix else f"{k}.{sub_k}"
                    self.history[key].append(sub_v)
            else:
                # Handle simple scalar metrics
                key = f"{prefix}_{k}" if prefix else k
                self.history[key].append(v)

    def log_metrics(self, metrics: Dict[str, Union[float, Dict[str, float]]], prefix: str) -> None:
        """
        Log metrics to console and add to history.

        Args:
            metrics: Dictionary of metrics (can contain nested dictionaries)
            prefix: Prefix for metric names (e.g., 'train', 'val')
        """
        # Format metrics for logging
        metrics_parts = []

        for k, v in metrics.items():
            if isinstance(v, dict):
                # Handle nested metrics
                sub_metrics = [f"{k}.{sub_k}={sub_v:.4f}" for sub_k, sub_v in v.items()]
                metrics_parts.extend(sub_metrics)
            else:
                # Handle simple metrics
                metrics_parts.append(f"{k}={v:.4f}")

        metrics_str = ", ".join(metrics_parts)
        print(f"{prefix.capitalize()}: {metrics_str}")

        # Add to history
        self.update(metrics, prefix)

    def get_metric(self, metric_name: str) -> List[float]:
        """
        Get the history of a specific metric.

        Args:
            metric_name: Name of the metric to retrieve

        Returns:
            List of metric values over time
        """
        return self.history.get(metric_name, [])

    def get_latest(self, metric_name: str) -> Optional[float]:
        """
        Get the latest value of a metric.

        Args:
            metric_name: Name of the metric

        Returns:
            Latest metric value, or None if not available
        """
        values = self.history.get(metric_name, [])
        return values[-1] if values else None

    def update_alignment_metrics(
        self,
        step: int,
        diag_mean: float,
        sim_mean: float,
        alignment_gap: float,
        alignment_snr: float,
    ) -> None:
        """
        Update alignment-specific metrics.

        Args:
            step: Current training step
            diag_mean: Mean diagonal similarity
            sim_mean: Mean overall similarity
            alignment_gap: Gap between diagonal and mean similarity
            alignment_snr: Signal-to-noise ratio for alignment
        """
        self._alignment_history["step"].append(step)
        self._alignment_history["diag_mean"].append(diag_mean)
        self._alignment_history["sim_mean"].append(sim_mean)
        self._alignment_history["alignment_gap"].append(alignment_gap)
        self._alignment_history["alignment_snr"].append(alignment_snr)

    def diagnose_training_issues(self) -> str:
        """
        Analyze metrics to identify potential training issues.

        Returns:
            Diagnostic message describing any detected issues
        """
        if not self.history or "train_loss" not in self.history or not self.history["train_loss"]:
            return "No training metrics available for diagnostics."

        issues = []
        losses = self.history["train_loss"]

        # Check for non-decreasing loss
        if len(losses) >= 3:
            recent_losses = losses[-3:]
            if all(recent_losses[i] >= recent_losses[i - 1] for i in range(1, len(recent_losses))):
                issues.append("Loss is not decreasing (plateau detected)")

        # Check for exploding loss
        if losses:
            if losses[-1] > 100 or (len(losses) > 1 and losses[-1] > 2 * losses[0]):
                issues.append("Loss may be exploding")

        # Check for vanishing gradients (if accuracy is tracked)
        if "train_accuracy" in self.history and self.history["train_accuracy"]:
            accuracies = self.history["train_accuracy"]
            if len(accuracies) >= 5:
                recent_acc = accuracies[-5:]
                if all(abs(recent_acc[i] - recent_acc[0]) < 0.01 for i in range(len(recent_acc))):
                    issues.append("Accuracy not improving (possible vanishing gradients)")

        # Check for loss instability
        if len(losses) >= 10:
            recent_losses = losses[-10:]
            loss_std = np.std(recent_losses)
            loss_mean = np.mean(recent_losses)
            if loss_std > 0.5 * loss_mean:
                issues.append("Loss is unstable (high variance)")

        if issues:
            return "Training issues detected:\n" + "\n".join(f"  - {issue}" for issue in issues)
        else:
            return "No obvious training issues detected from metrics."

    def plot_history(self, save_dir: Optional[str] = None) -> None:
        """
        Plot training history metrics.

        Args:
            save_dir: Directory to save plots (if None, plots are not saved)
        """
        if not self.history:
            logger.warning("No metrics to plot")
            return

        # Create directory if not exists
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)

        # Group metrics by type
        metric_groups = {}
        for key in self.history.keys():
            # Skip if there are no values
            if not self.history[key]:
                continue

            # Skip dictionary values
            if self.history[key] and isinstance(self.history[key][0], dict):
                continue

            # Split the key into prefix and metric name
            if "_" in key:
                prefix, metric = key.split("_", 1)
                if metric not in metric_groups:
                    metric_groups[metric] = []
                metric_groups[metric].append((prefix, key))

        # Plot each metric group
        for metric, prefixes in metric_groups.items():
            plt.figure(figsize=(10, 6))

            for prefix, key in prefixes:
                values = self.history[key]
                # Only plot if values are not dictionaries
                if values and not isinstance(values[0], dict):
                    # Convert tensors to CPU numpy arrays
                    cpu_values = []
                    for v in values:
                        if isinstance(v, torch.Tensor):
                            cpu_values.append(v.detach().cpu().item())
                        else:
                            cpu_values.append(v)

                    epochs = range(1, len(cpu_values) + 1)
                    plt.plot(epochs, cpu_values, label=f"{prefix}")

            plt.title(f"{metric} over epochs")
            plt.xlabel("Epoch")
            plt.ylabel(metric)
            plt.legend()
            plt.grid(True)

            if save_dir:
                plt.savefig(os.path.join(save_dir, f"{metric}.png"))
            plt.close()

        # Plot alignment metrics if available
        if self._alignment_history["step"]:
            self._plot_alignment_metrics(save_dir)

    def _plot_alignment_metrics(self, save_dir: Optional[str] = None) -> None:
        """
        Plot alignment-specific metrics.

        Args:
            save_dir: Directory to save plots
        """
        plt.figure(figsize=(12, 8))

        # Plot diagonal similarity vs. mean similarity
        plt.subplot(2, 1, 1)
        steps = self._alignment_history["step"]
        plt.plot(
            steps,
            self._alignment_history["diag_mean"],
            label="Diagonal Similarity",
            color="blue",
        )
        plt.plot(
            steps,
            self._alignment_history["sim_mean"],
            label="Mean Similarity",
            color="red",
            linestyle="--",
        )
        plt.title("Semantic Alignment Progress")
        plt.xlabel("Training Steps")
        plt.ylabel("Cosine Similarity")
        plt.legend()
        plt.grid(True)

        # Plot alignment gap and SNR
        plt.subplot(2, 1, 2)
        plt.plot(
            steps,
            self._alignment_history["alignment_gap"],
            label="Alignment Gap",
            color="green",
        )
        plt.plot(
            steps,
            self._alignment_history["alignment_snr"],
            label="Signal-to-Noise Ratio",
            color="purple",
            linestyle="--",
        )
        plt.title("Alignment Quality Metrics")
        plt.xlabel("Training Steps")
        plt.ylabel("Value")
        plt.legend()
        plt.grid(True)

        if save_dir:
            plt.savefig(os.path.join(save_dir, "alignment_metrics.png"))
        plt.close()

    def get_summary(self) -> Dict[str, Any]:
        """
        Get a summary of all tracked metrics.

        Returns:
            Dictionary containing metric summaries
        """
        summary = {}
        for key, values in self.history.items():
            if values and not isinstance(values[0], dict):
                # Convert tensors if needed
                cpu_values = []
                for v in values:
                    if isinstance(v, torch.Tensor):
                        cpu_values.append(v.detach().cpu().item())
                    else:
                        cpu_values.append(v)

                summary[key] = {
                    "latest": cpu_values[-1] if cpu_values else None,
                    "best": min(cpu_values) if "loss" in key else max(cpu_values),
                    "mean": np.mean(cpu_values),
                    "std": np.std(cpu_values),
                    "count": len(cpu_values),
                }
        return summary

    def clear(self) -> None:
        """Clear all metrics history."""
        self.history.clear()
        for key in self._alignment_history:
            self._alignment_history[key].clear()

    def to_dict(self) -> Dict[str, List]:
        """
        Convert metrics history to a dictionary.

        Returns:
            Dictionary of metric histories
        """
        return dict(self.history)

    def from_dict(self, history_dict: Dict[str, List]) -> None:
        """
        Load metrics history from a dictionary.

        Args:
            history_dict: Dictionary of metric histories
        """
        self.history = defaultdict(list, history_dict)
