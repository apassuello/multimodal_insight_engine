# src/utils/metrics_tracker.py

import torch
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Optional, Any, Tuple, Union
import os
import json
import logging
from collections import defaultdict
import time

logger = logging.getLogger(__name__)

"""
MODULE: metrics_tracker.py
PURPOSE: Tracks and visualizes training metrics across epochs and steps with issue detection
KEY COMPONENTS:
- MetricsTracker: Comprehensive metrics tracking and visualization class
DEPENDENCIES: torch, numpy, matplotlib, typing, os, json, logging, collections, time
SPECIAL NOTES: Includes specialized functionality for tracking multimodal alignment quality
"""


class MetricsTracker:
    """
    Tracks and visualizes training metrics across epochs and steps.

    Features:
    1. Tracks metrics at both step and epoch level
    2. Handles multiple metric groups (train, val, etc.)
    3. Provides early stopping based on validation metrics
    4. Creates visualizations of metric trends
    5. Detects training issues based on metric patterns
    6. Saves metrics to disk for later analysis
    """

    def __init__(
        self,
        log_dir: str,
        early_stopping: bool = False,
        patience: int = 5,
        monitor: str = "val_loss",
        mode: str = "min",
        visualization_frequency: int = 1,
        issue_detection: bool = True,
    ):
        """
        Initialize the metrics tracker.

        Args:
            log_dir: Directory to save logs and visualizations
            early_stopping: Whether to use early stopping
            patience: Number of epochs to wait for improvement before stopping
            monitor: Metric to monitor for early stopping
            mode: 'min' for metrics that should decrease, 'max' for metrics that should increase
            visualization_frequency: How often to create visualizations (in epochs)
            issue_detection: Whether to detect training issues based on metrics
        """
        self.log_dir = log_dir
        self.early_stopping = early_stopping
        self.patience = patience
        self.monitor = monitor
        self.mode = mode
        self.visualization_frequency = visualization_frequency
        self.issue_detection = issue_detection

        # Create log directory
        os.makedirs(log_dir, exist_ok=True)
        os.makedirs(os.path.join(log_dir, "visualizations"), exist_ok=True)

        # Initialize metrics tracking
        self.step_metrics = defaultdict(list)
        self.epoch_metrics = defaultdict(list)
        self.current_epoch = 0
        self.global_step = 0

        # Initialize early stopping
        self.best_score = float("inf") if mode == "min" else float("-inf")
        self.best_epoch = 0
        self.counter = 0
        self.stopped_early = False

        # Track timing information
        self.epoch_start_time = None
        self.training_start_time = time.time()

        # Features for multimodal similarity tracking
        self.alignment_history = {
            "epoch": [],
            "step": [],
            "diag_similarity": [],
            "mean_similarity": [],
            "alignment_gap": [],
            "signal_to_noise": [],
        }

        logger.info(f"Metrics will be saved to {log_dir}")

    def start_epoch(self, epoch: int) -> None:
        """
        Signal the start of a new epoch.

        Args:
            epoch: Current epoch number
        """
        self.current_epoch = epoch
        self.epoch_start_time = time.time()
        logger.info(f"Starting epoch {epoch}")

    def end_epoch(self) -> Dict[str, float]:
        """
        Signal the end of the current epoch and compute summary metrics.

        Returns:
            Dictionary with epoch summary metrics
        """
        # Calculate epoch duration
        epoch_duration = (
            time.time() - self.epoch_start_time if self.epoch_start_time else 0
        )

        # Extract metrics for this epoch
        epoch_summary = {}

        # Log epoch summary
        logger.info(f"Completed epoch {self.current_epoch} in {epoch_duration:.2f}s")

        # Create visualizations if needed
        if self.current_epoch % self.visualization_frequency == 0:
            self.create_visualizations()

        # Check for early stopping
        should_stop = self.check_early_stopping()
        if should_stop:
            logger.info(f"Early stopping triggered after {self.current_epoch} epochs")
            self.stopped_early = True

        # Save metrics to disk
        self.save_metrics()

        # Return epoch summary
        return epoch_summary

    def update_step_metrics(
        self, metrics: Dict[str, Any], group: str = "train"
    ) -> None:
        """
        Update metrics for the current step.

        Args:
            metrics: Dictionary of metrics
            group: Metric group (train, val, etc.)
        """
        # Update global step
        self.global_step += 1

        # Store step info
        self.step_metrics["step"].append(self.global_step)
        self.step_metrics["epoch"].append(self.current_epoch)

        # Store metrics
        for name, value in metrics.items():
            key = f"{group}_{name}"
            # Convert tensors to Python values
            if isinstance(value, torch.Tensor):
                value = value.item()
            self.step_metrics[key].append(value)

    def update_epoch_metrics(
        self, metrics: Dict[str, Any], group: str = "train"
    ) -> None:
        """
        Update metrics for the current epoch.

        Args:
            metrics: Dictionary of metrics
            group: Metric group (train, val, etc.)
        """
        # Store epoch info
        if "epoch" not in self.epoch_metrics:
            self.epoch_metrics["epoch"] = []

        # Only append epoch number once per epoch
        if len(self.epoch_metrics["epoch"]) <= self.current_epoch:
            self.epoch_metrics["epoch"].append(self.current_epoch)

        # Store metrics
        for name, value in metrics.items():
            key = f"{group}_{name}"
            # Convert tensors to Python values
            if isinstance(value, torch.Tensor):
                value = value.item()

            # Initialize list if needed
            if key not in self.epoch_metrics:
                self.epoch_metrics[key] = []

            # Make sure we have the right number of entries
            while len(self.epoch_metrics[key]) < self.current_epoch:
                self.epoch_metrics[key].append(None)

            # Update or append value
            if len(self.epoch_metrics[key]) == self.current_epoch:
                self.epoch_metrics[key].append(value)
            else:
                self.epoch_metrics[key][self.current_epoch] = value

    def check_early_stopping(self) -> bool:
        """
        Check if early stopping criteria are met.

        Returns:
            True if training should stop, False otherwise
        """
        if not self.early_stopping:
            return False

        # Check if we have the monitored metric
        if self.monitor not in self.epoch_metrics:
            return False

        # Get current score
        if len(self.epoch_metrics[self.monitor]) <= self.current_epoch:
            return False

        current_score = self.epoch_metrics[self.monitor][self.current_epoch]

        # Check if it's an improvement
        if (self.mode == "min" and current_score < self.best_score) or (
            self.mode == "max" and current_score > self.best_score
        ):
            # Update best score
            self.best_score = current_score
            self.best_epoch = self.current_epoch
            self.counter = 0
            return False
        else:
            # Increment counter
            self.counter += 1
            logger.info(f"EarlyStopping counter: {self.counter} out of {self.patience}")
            if self.counter >= self.patience:
                return True
            return False

    def save_metrics(self) -> None:
        """
        Save metrics to disk for later analysis.
        """
        # Convert defaultdict to dict
        step_metrics = dict(self.step_metrics)
        epoch_metrics = dict(self.epoch_metrics)

        # Create metrics object
        metrics = {
            "step_metrics": step_metrics,
            "epoch_metrics": epoch_metrics,
            "best_score": self.best_score,
            "best_epoch": self.best_epoch,
            "early_stopped": self.stopped_early,
            "current_epoch": self.current_epoch,
            "global_step": self.global_step,
            "alignment_history": self.alignment_history,
        }

        # Save to file
        metrics_file = os.path.join(self.log_dir, "metrics.json")
        try:
            with open(metrics_file, "w") as f:
                json.dump(metrics, f, indent=2)
            logger.debug(f"Metrics saved to {metrics_file}")
        except Exception as e:
            logger.error(f"Error saving metrics: {e}")

    def create_visualizations(self) -> None:
        """
        Create visualizations of tracked metrics.
        """
        # Skip if we don't have enough data
        if not self.epoch_metrics or len(self.epoch_metrics.get("epoch", [])) == 0:
            return

        vis_dir = os.path.join(self.log_dir, "visualizations")
        os.makedirs(vis_dir, exist_ok=True)

        # Create loss and accuracy plots
        self._create_training_curve_plots(vis_dir)

        # Create alignment plots if we have alignment data
        if len(self.alignment_history["step"]) > 0:
            self._create_alignment_plots(vis_dir)

    def _create_training_curve_plots(self, vis_dir: str) -> None:
        """
        Create loss and accuracy training curves.

        Args:
            vis_dir: Directory to save visualizations
        """
        # Collect metric names by group
        metric_names = set()
        groups = set()

        for key in self.epoch_metrics.keys():
            if key == "epoch":
                continue

            # Split into group and name
            parts = key.split("_", 1)
            if len(parts) == 2:
                group, name = parts
                groups.add(group)
                metric_names.add(name)

        # Create plots for each metric type
        for metric_name in metric_names:
            # Create figure
            plt.figure(figsize=(10, 6))

            # Plot metric for each group
            for group in groups:
                key = f"{group}_{metric_name}"
                if key in self.epoch_metrics and len(self.epoch_metrics[key]) > 0:
                    # Clean up None values for plotting
                    values = []
                    epochs = []
                    for epoch, value in enumerate(self.epoch_metrics[key]):
                        if value is not None:
                            epochs.append(epoch)
                            values.append(value)

                    if epochs and values:
                        plt.plot(epochs, values, marker="o", label=f"{group}")

            # Add chart elements
            plt.title(f"{metric_name.capitalize()} vs. Epoch")
            plt.xlabel("Epoch")
            plt.ylabel(metric_name.capitalize())
            plt.grid(True)
            plt.legend()

            # Save figure
            plot_file = os.path.join(vis_dir, f"{metric_name}_curve.png")
            plt.savefig(plot_file)
            plt.close()
            logger.debug(f"Created {metric_name} plot: {plot_file}")

    def _create_alignment_plots(self, vis_dir: str) -> None:
        """
        Create plots showing multimodal alignment progress.

        Args:
            vis_dir: Directory to save visualizations
        """
        if len(self.alignment_history["step"]) == 0:
            return

        # Create figure
        plt.figure(figsize=(12, 10))

        # Create subplots
        plt.subplot(2, 1, 1)
        plt.plot(
            self.alignment_history["step"],
            self.alignment_history["diag_similarity"],
            label="Diagonal (Matched Pairs)",
        )
        plt.plot(
            self.alignment_history["step"],
            self.alignment_history["mean_similarity"],
            label="Mean Similarity",
        )
        plt.title("Multimodal Similarity Progress")
        plt.xlabel("Training Step")
        plt.ylabel("Cosine Similarity")
        plt.legend()
        plt.grid(True)

        plt.subplot(2, 1, 2)
        plt.plot(
            self.alignment_history["step"],
            self.alignment_history["alignment_gap"],
            label="Alignment Gap",
        )
        plt.plot(
            self.alignment_history["step"],
            self.alignment_history["signal_to_noise"],
            label="Signal-to-Noise Ratio",
        )
        plt.title("Alignment Quality Metrics")
        plt.xlabel("Training Step")
        plt.ylabel("Metric Value")
        plt.legend()
        plt.grid(True)

        # Save figure
        plt.tight_layout()
        plot_file = os.path.join(vis_dir, f"alignment_progress.png")
        plt.savefig(plot_file)
        plt.close()
        logger.debug(f"Created alignment plot: {plot_file}")

    def update_alignment_metrics(
        self,
        diag_similarity: float,
        mean_similarity: float,
        std_similarity: float,
        step: Optional[int] = None,
    ) -> None:
        """
        Update metrics related to multimodal alignment quality.

        Args:
            diag_similarity: Mean similarity for matched (diagonal) pairs
            mean_similarity: Mean similarity across all pairs
            std_similarity: Standard deviation of similarity
            step: Current step (uses global_step if None)
        """
        current_step = step if step is not None else self.global_step

        # Calculate derived metrics
        alignment_gap = diag_similarity - mean_similarity
        signal_to_noise = abs(alignment_gap) / max(std_similarity, 1e-5)

        # Update history
        self.alignment_history["epoch"].append(self.current_epoch)
        self.alignment_history["step"].append(current_step)
        self.alignment_history["diag_similarity"].append(diag_similarity)
        self.alignment_history["mean_similarity"].append(mean_similarity)
        self.alignment_history["alignment_gap"].append(alignment_gap)
        self.alignment_history["signal_to_noise"].append(signal_to_noise)

        # Limited window for memory efficiency
        max_history = 1000  # Store at most this many points
        if len(self.alignment_history["step"]) > max_history:
            for key in self.alignment_history:
                self.alignment_history[key] = self.alignment_history[key][-max_history:]

        # Perform issue detection if enabled
        if self.issue_detection:
            self._detect_alignment_issues(
                diag_similarity, mean_similarity, alignment_gap, signal_to_noise
            )

    def _detect_alignment_issues(
        self,
        diag_similarity: float,
        mean_similarity: float,
        alignment_gap: float,
        signal_to_noise: float,
    ) -> None:
        """
        Detect potential issues in multimodal alignment.

        Args:
            diag_similarity: Similarity for matched pairs
            mean_similarity: Mean similarity across all pairs
            alignment_gap: Difference between diagonal and mean similarity
            signal_to_noise: Ratio of alignment gap to similarity standard deviation
        """
        # Check if diagonal similarity is not much higher than mean
        if alignment_gap < 0.05 and self.current_epoch >= 1:
            logger.warning(
                f"Potential alignment issue: diagonal similarity ({diag_similarity:.4f}) "
                f"is not much higher than mean ({mean_similarity:.4f}), gap={alignment_gap:.4f}"
            )

            # More serious warning for later epochs
            if self.current_epoch >= 3 and alignment_gap < 0.02:
                logger.error(
                    f"CRITICAL ALIGNMENT ISSUE: After {self.current_epoch} epochs, "
                    f"diagonal similarity ({diag_similarity:.4f}) still very close to "
                    f"mean ({mean_similarity:.4f}), gap={alignment_gap:.4f}"
                )
                logger.error("This suggests feature collapse or training issues.")

        # Check for very low signal-to-noise ratio (noisy similarity distribution)
        if signal_to_noise < 1.0 and self.current_epoch >= 1:
            logger.warning(f"Low signal-to-noise ratio: {signal_to_noise:.2f}")

            # More serious for later epochs
            if self.current_epoch >= 3 and signal_to_noise < 0.5:
                logger.error(
                    f"CRITICAL SNR ISSUE: After {self.current_epoch} epochs, "
                    f"signal-to-noise ratio still very low: {signal_to_noise:.2f}"
                )
                logger.error(
                    "This suggests inability to distinguish positive from negative pairs."
                )

    def get_best_metrics(self) -> Dict[str, Any]:
        """
        Get metrics from the best epoch according to the monitored metric.

        Returns:
            Dictionary with best metrics
        """
        best_metrics = {}
        best_metrics["epoch"] = self.best_epoch
        best_metrics["score"] = self.best_score

        # Get all metrics from the best epoch
        for key, values in self.epoch_metrics.items():
            if key != "epoch" and len(values) > self.best_epoch:
                best_metrics[key] = values[self.best_epoch]

        return best_metrics

    def get_latest_metrics(self) -> Dict[str, Any]:
        """
        Get metrics from the most recent epoch.

        Returns:
            Dictionary with latest metrics
        """
        latest_metrics = {}

        # Get all metrics from the latest epoch
        for key, values in self.epoch_metrics.items():
            if len(values) > 0:
                latest_metrics[key] = values[-1]

        return latest_metrics

    def check_for_issues(self) -> List[str]:
        """
        Check for potential training issues based on metric patterns.

        Returns:
            List of issue descriptions
        """
        issues = []

        # Skip if we don't have enough data
        if len(self.epoch_metrics.get("epoch", [])) < 2:
            return issues

        # Check for increasing training loss
        if "train_loss" in self.epoch_metrics:
            train_losses = self.epoch_metrics["train_loss"]
            if len(train_losses) > 3:
                # Check if training loss has been consistently increasing
                recent_losses = [x for x in train_losses[-3:] if x is not None]
                if len(recent_losses) >= 3 and all(
                    recent_losses[i] > recent_losses[i - 1]
                    for i in range(1, len(recent_losses))
                ):
                    issues.append(
                        "Training loss has been consistently increasing for 3+ epochs"
                    )

        # Check for validation loss >> training loss (overfitting)
        if "train_loss" in self.epoch_metrics and "val_loss" in self.epoch_metrics:
            train_losses = self.epoch_metrics["train_loss"]
            val_losses = self.epoch_metrics["val_loss"]

            if len(train_losses) > 0 and len(val_losses) > 0:
                latest_train = train_losses[-1]
                latest_val = val_losses[-1]

                if latest_train is not None and latest_val is not None:
                    ratio = latest_val / latest_train if latest_train > 0 else 1.0

                    if ratio > 2.0:
                        issues.append(
                            f"Validation loss is {ratio:.1f}x higher than training loss, suggesting overfitting"
                        )

        # Check for NaN/inf values
        for key, values in self.epoch_metrics.items():
            if key != "epoch" and len(values) > 0:
                latest = values[-1]
                if latest is not None and (np.isnan(latest) or np.isinf(latest)):
                    issues.append(f"NaN or infinite values detected in {key}")

        # Check for validation metrics not improving
        if (
            self.monitor in self.epoch_metrics
            and self.current_epoch > self.best_epoch + 3
        ):
            epochs_since_best = self.current_epoch - self.best_epoch
            issues.append(
                f"No improvement in {self.monitor} for {epochs_since_best} epochs "
                f"(best: {self.best_score} at epoch {self.best_epoch})"
            )

        # Check for learning rate issues based on training progress
        if "learning_rate" in self.step_metrics and "train_loss" in self.step_metrics:
            recent_steps = min(100, len(self.step_metrics["learning_rate"]))
            if recent_steps > 50:
                recent_lr = self.step_metrics["learning_rate"][-recent_steps:]
                recent_loss = self.step_metrics["train_loss"][-recent_steps:]

                # Calculate correlation between LR and loss
                lr_loss_corr = np.corrcoef(recent_lr, recent_loss)[0, 1]

                if lr_loss_corr > 0.7:
                    issues.append(
                        "Strong positive correlation between learning rate and loss, "
                        "suggesting learning rate might be too high"
                    )

        return issues


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
        "module_purpose": "Tracks and visualizes training metrics across epochs and steps with issue detection",
        "key_classes": [
            {
                "name": "MetricsTracker",
                "purpose": "Tracks and visualizes training metrics with early stopping and issue detection",
                "key_methods": [
                    {
                        "name": "__init__",
                        "signature": "__init__(self, log_dir: str, early_stopping: bool = False, patience: int = 5, monitor: str = 'val_loss', mode: str = 'min', visualization_frequency: int = 1, issue_detection: bool = True)",
                        "brief_description": "Initialize metrics tracker with configuration for monitoring and visualization",
                    },
                    {
                        "name": "update_step_metrics",
                        "signature": "update_step_metrics(self, metrics: Dict[str, Any], group: str = 'train') -> None",
                        "brief_description": "Update metrics for the current step",
                    },
                    {
                        "name": "update_epoch_metrics",
                        "signature": "update_epoch_metrics(self, metrics: Dict[str, Any], group: str = 'train') -> None",
                        "brief_description": "Update metrics for the current epoch",
                    },
                    {
                        "name": "check_early_stopping",
                        "signature": "check_early_stopping(self) -> bool",
                        "brief_description": "Check if early stopping criteria are met",
                    },
                    {
                        "name": "create_visualizations",
                        "signature": "create_visualizations(self) -> None",
                        "brief_description": "Create visualizations of tracked metrics",
                    },
                    {
                        "name": "update_alignment_metrics",
                        "signature": "update_alignment_metrics(self, diag_similarity: float, mean_similarity: float, std_similarity: float, step: Optional[int] = None) -> None",
                        "brief_description": "Update metrics related to multimodal alignment quality",
                    },
                    {
                        "name": "check_for_issues",
                        "signature": "check_for_issues(self) -> List[str]",
                        "brief_description": "Check for potential training issues based on metric patterns",
                    },
                ],
                "inheritance": "object",
                "dependencies": [
                    "torch",
                    "numpy",
                    "matplotlib",
                    "typing",
                    "os",
                    "json",
                    "logging",
                    "collections",
                    "time",
                ],
            }
        ],
        "external_dependencies": ["torch", "numpy", "matplotlib", "json"],
        "complexity_score": 7,  # Complex component with visualization, early stopping, and issue detection
    }
