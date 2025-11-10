"""Tests for MetricsCollector.

Tests metrics collection, history tracking, and visualization.
"""

import os
import tempfile
import shutil

import pytest
import torch
import numpy as np

from src.training.trainers.multimodal import MetricsCollector


@pytest.fixture
def metrics_collector():
    """Create a MetricsCollector instance."""
    return MetricsCollector()


@pytest.fixture
def temp_dir():
    """Create a temporary directory for plots."""
    tmpdir = tempfile.mkdtemp()
    yield tmpdir
    # Cleanup
    if os.path.exists(tmpdir):
        shutil.rmtree(tmpdir)


class TestMetricsCollector:
    """Tests for MetricsCollector class."""

    def test_initialization(self, metrics_collector):
        """Test MetricsCollector initialization."""
        assert len(metrics_collector.history) == 0
        assert len(metrics_collector._alignment_history["step"]) == 0

    def test_update_simple_metrics(self, metrics_collector):
        """Test updating simple scalar metrics."""
        metrics = {"loss": 0.5, "accuracy": 0.85}
        metrics_collector.update(metrics, prefix="train")

        assert "train_loss" in metrics_collector.history
        assert "train_accuracy" in metrics_collector.history
        assert metrics_collector.history["train_loss"] == [0.5]
        assert metrics_collector.history["train_accuracy"] == [0.85]

    def test_update_nested_metrics(self, metrics_collector):
        """Test updating nested dictionary metrics."""
        metrics = {
            "loss": 0.5,
            "recalls": {"top1": 0.8, "top5": 0.95},
        }
        metrics_collector.update(metrics, prefix="val")

        assert "val_loss" in metrics_collector.history
        assert "val_recalls.top1" in metrics_collector.history
        assert "val_recalls.top5" in metrics_collector.history
        assert metrics_collector.history["val_recalls.top1"] == [0.8]
        assert metrics_collector.history["val_recalls.top5"] == [0.95]

    def test_update_without_prefix(self, metrics_collector):
        """Test updating metrics without prefix."""
        metrics = {"loss": 0.3}
        metrics_collector.update(metrics, prefix="")

        assert "loss" in metrics_collector.history
        assert metrics_collector.history["loss"] == [0.3]

    def test_get_metric(self, metrics_collector):
        """Test retrieving metric history."""
        metrics_collector.update({"loss": 0.5}, prefix="train")
        metrics_collector.update({"loss": 0.4}, prefix="train")
        metrics_collector.update({"loss": 0.3}, prefix="train")

        history = metrics_collector.get_metric("train_loss")
        assert history == [0.5, 0.4, 0.3]

    def test_get_metric_nonexistent(self, metrics_collector):
        """Test retrieving non-existent metric."""
        history = metrics_collector.get_metric("nonexistent")
        assert history == []

    def test_get_latest(self, metrics_collector):
        """Test getting latest metric value."""
        metrics_collector.update({"loss": 0.5}, prefix="train")
        metrics_collector.update({"loss": 0.4}, prefix="train")

        latest = metrics_collector.get_latest("train_loss")
        assert latest == 0.4

    def test_get_latest_empty(self, metrics_collector):
        """Test getting latest value for empty metric."""
        latest = metrics_collector.get_latest("nonexistent")
        assert latest is None

    def test_log_metrics(self, metrics_collector, capsys):
        """Test logging metrics to console."""
        metrics = {"loss": 0.5, "accuracy": 0.85}
        metrics_collector.log_metrics(metrics, prefix="train")

        captured = capsys.readouterr()
        assert "Train:" in captured.out
        assert "loss=0.5" in captured.out
        assert "accuracy=0.85" in captured.out

        # Check that metrics were also added to history
        assert metrics_collector.history["train_loss"] == [0.5]

    def test_log_nested_metrics(self, metrics_collector, capsys):
        """Test logging nested metrics."""
        metrics = {
            "loss": 0.5,
            "recalls": {"top1": 0.8, "top5": 0.95},
        }
        metrics_collector.log_metrics(metrics, prefix="val")

        captured = capsys.readouterr()
        assert "Val:" in captured.out
        assert "loss=0.5" in captured.out
        assert "recalls.top1=0.8" in captured.out
        assert "recalls.top5=0.95" in captured.out

    def test_update_alignment_metrics(self, metrics_collector):
        """Test updating alignment metrics."""
        metrics_collector.update_alignment_metrics(
            step=100,
            diag_mean=0.8,
            sim_mean=0.5,
            alignment_gap=0.3,
            alignment_snr=2.5,
        )

        assert metrics_collector._alignment_history["step"] == [100]
        assert metrics_collector._alignment_history["diag_mean"] == [0.8]
        assert metrics_collector._alignment_history["sim_mean"] == [0.5]
        assert metrics_collector._alignment_history["alignment_gap"] == [0.3]
        assert metrics_collector._alignment_history["alignment_snr"] == [2.5]

    def test_diagnose_no_metrics(self, metrics_collector):
        """Test diagnosis with no metrics."""
        result = metrics_collector.diagnose_training_issues()
        assert "No training metrics available" in result

    def test_diagnose_plateau(self, metrics_collector):
        """Test detection of loss plateau."""
        for loss in [0.5, 0.51, 0.52]:
            metrics_collector.update({"loss": loss}, prefix="train")

        result = metrics_collector.diagnose_training_issues()
        assert "plateau" in result.lower()

    def test_diagnose_exploding_loss(self, metrics_collector):
        """Test detection of exploding loss."""
        metrics_collector.update({"loss": 1.0}, prefix="train")
        metrics_collector.update({"loss": 150.0}, prefix="train")

        result = metrics_collector.diagnose_training_issues()
        assert "exploding" in result.lower()

    def test_diagnose_no_issues(self, metrics_collector):
        """Test diagnosis when training is healthy."""
        for loss in [0.5, 0.4, 0.3, 0.2]:
            metrics_collector.update({"loss": loss}, prefix="train")

        result = metrics_collector.diagnose_training_issues()
        assert "No obvious training issues" in result

    def test_get_summary(self, metrics_collector):
        """Test getting metrics summary."""
        for i, loss in enumerate([0.5, 0.4, 0.3, 0.2]):
            metrics_collector.update({"loss": loss, "accuracy": 0.7 + i * 0.05}, prefix="train")

        summary = metrics_collector.get_summary()

        assert "train_loss" in summary
        assert summary["train_loss"]["latest"] == 0.2
        assert summary["train_loss"]["best"] == 0.2  # min for loss
        assert summary["train_loss"]["count"] == 4

        assert "train_accuracy" in summary
        assert summary["train_accuracy"]["best"] == 0.85  # max for accuracy

    def test_clear(self, metrics_collector):
        """Test clearing metrics."""
        metrics_collector.update({"loss": 0.5}, prefix="train")
        metrics_collector.update_alignment_metrics(
            step=100, diag_mean=0.8, sim_mean=0.5, alignment_gap=0.3, alignment_snr=2.5
        )

        metrics_collector.clear()

        assert len(metrics_collector.history) == 0
        assert len(metrics_collector._alignment_history["step"]) == 0

    def test_to_dict(self, metrics_collector):
        """Test converting to dictionary."""
        metrics_collector.update({"loss": 0.5, "accuracy": 0.85}, prefix="train")

        history_dict = metrics_collector.to_dict()

        assert isinstance(history_dict, dict)
        assert "train_loss" in history_dict
        assert "train_accuracy" in history_dict
        assert history_dict["train_loss"] == [0.5]

    def test_from_dict(self, metrics_collector):
        """Test loading from dictionary."""
        history_dict = {
            "train_loss": [0.5, 0.4, 0.3],
            "train_accuracy": [0.7, 0.8, 0.9],
        }

        metrics_collector.from_dict(history_dict)

        assert metrics_collector.history["train_loss"] == [0.5, 0.4, 0.3]
        assert metrics_collector.history["train_accuracy"] == [0.7, 0.8, 0.9]

    def test_plot_history_no_metrics(self, metrics_collector):
        """Test plotting with no metrics."""
        # Should not raise error
        metrics_collector.plot_history()

    def test_plot_history_with_save(self, metrics_collector, temp_dir):
        """Test plotting and saving to directory."""
        for i in range(5):
            metrics_collector.update(
                {"loss": 0.5 - i * 0.1, "accuracy": 0.5 + i * 0.1},
                prefix="train"
            )

        metrics_collector.plot_history(save_dir=temp_dir)

        # Check that plots were created
        assert os.path.exists(os.path.join(temp_dir, "loss.png"))
        assert os.path.exists(os.path.join(temp_dir, "accuracy.png"))

    def test_plot_alignment_metrics(self, metrics_collector, temp_dir):
        """Test plotting alignment metrics."""
        for i in range(10):
            metrics_collector.update_alignment_metrics(
                step=i * 100,
                diag_mean=0.5 + i * 0.05,
                sim_mean=0.3 + i * 0.03,
                alignment_gap=0.2 + i * 0.02,
                alignment_snr=1.0 + i * 0.1,
            )

        metrics_collector.plot_history(save_dir=temp_dir)

        # Check that alignment plot was created
        assert os.path.exists(os.path.join(temp_dir, "alignment_metrics.png"))

    def test_metrics_with_tensors(self, metrics_collector):
        """Test handling PyTorch tensors in metrics."""
        loss_tensor = torch.tensor(0.5)
        accuracy_tensor = torch.tensor(0.85)

        metrics_collector.update(
            {"loss": loss_tensor, "accuracy": accuracy_tensor},
            prefix="train"
        )

        # Should work fine with tensors
        assert len(metrics_collector.history["train_loss"]) == 1

        # Summary should convert to scalars
        summary = metrics_collector.get_summary()
        assert isinstance(summary["train_loss"]["latest"], float)

    def test_multiple_prefixes(self, metrics_collector):
        """Test tracking metrics with different prefixes."""
        metrics_collector.update({"loss": 0.5}, prefix="train")
        metrics_collector.update({"loss": 0.6}, prefix="val")
        metrics_collector.update({"loss": 0.55}, prefix="test")

        assert "train_loss" in metrics_collector.history
        assert "val_loss" in metrics_collector.history
        assert "test_loss" in metrics_collector.history
        assert metrics_collector.history["train_loss"] == [0.5]
        assert metrics_collector.history["val_loss"] == [0.6]
        assert metrics_collector.history["test_loss"] == [0.55]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
