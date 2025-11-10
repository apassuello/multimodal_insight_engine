"""Tests for CheckpointManager.

Tests checkpoint saving, loading, and training state persistence.
"""

import os
import tempfile
import shutil
from collections import defaultdict

import pytest
import torch
import torch.nn as nn
import torch.optim as optim

from src.training.trainers.multimodal import CheckpointManager


class SimpleModel(nn.Module):
    """Simple model for testing."""

    def __init__(self, input_dim=10, hidden_dim=20, output_dim=5):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        return self.fc2(x)


@pytest.fixture
def temp_dir():
    """Create a temporary directory for checkpoints."""
    tmpdir = tempfile.mkdtemp()
    yield tmpdir
    # Cleanup
    if os.path.exists(tmpdir):
        shutil.rmtree(tmpdir)


@pytest.fixture
def model():
    """Create a simple model."""
    return SimpleModel()


@pytest.fixture
def optimizer(model):
    """Create an optimizer."""
    return optim.Adam(model.parameters(), lr=0.001)


@pytest.fixture
def checkpoint_manager(model, optimizer, temp_dir):
    """Create a CheckpointManager instance."""
    return CheckpointManager(
        model=model,
        optimizer=optimizer,
        checkpoint_dir=temp_dir,
        device=torch.device("cpu")
    )


class TestCheckpointManager:
    """Tests for CheckpointManager class."""

    def test_initialization(self, checkpoint_manager, temp_dir):
        """Test CheckpointManager initialization."""
        assert checkpoint_manager.checkpoint_dir == temp_dir
        assert os.path.exists(temp_dir)
        assert checkpoint_manager.current_epoch == 0
        assert checkpoint_manager.global_step == 0
        assert checkpoint_manager.best_val_metric == 0.0
        assert checkpoint_manager.patience_counter == 0

    def test_save_checkpoint(self, checkpoint_manager, temp_dir):
        """Test saving a checkpoint."""
        checkpoint_path = os.path.join(temp_dir, "test_checkpoint.pt")

        # Update state
        checkpoint_manager.update_state(
            current_epoch=5,
            global_step=100,
            best_val_metric=0.85,
            patience_counter=2
        )

        # Save checkpoint
        checkpoint_manager.save_checkpoint(checkpoint_path)

        # Check file exists
        assert os.path.exists(checkpoint_path)

        # Load and verify contents
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        assert "model_state_dict" in checkpoint
        assert "optimizer_state_dict" in checkpoint
        assert checkpoint["current_epoch"] == 5
        assert checkpoint["global_step"] == 100
        assert checkpoint["best_val_metric"] == 0.85
        assert checkpoint["patience_counter"] == 2

    def test_load_checkpoint(self, checkpoint_manager, temp_dir, model, optimizer):
        """Test loading a checkpoint."""
        checkpoint_path = os.path.join(temp_dir, "test_checkpoint.pt")

        # Save initial state
        checkpoint_manager.update_state(
            current_epoch=3,
            global_step=50,
            best_val_metric=0.75,
            patience_counter=1
        )
        checkpoint_manager.save_checkpoint(checkpoint_path)

        # Modify model weights
        with torch.no_grad():
            for param in model.parameters():
                param.fill_(0.5)

        # Create new manager and load checkpoint
        new_manager = CheckpointManager(
            model=model,
            optimizer=optimizer,
            checkpoint_dir=temp_dir,
            device=torch.device("cpu")
        )
        state = new_manager.load_checkpoint(checkpoint_path)

        # Verify state loaded correctly
        assert state["current_epoch"] == 4  # Should be +1 for resuming
        assert state["global_step"] == 50
        assert state["best_val_metric"] == 0.75
        assert state["patience_counter"] == 1
        assert new_manager.current_epoch == 4

    def test_save_and_load_with_history(self, checkpoint_manager, temp_dir):
        """Test saving and loading with training history."""
        checkpoint_path = os.path.join(temp_dir, "test_checkpoint.pt")

        # Create history
        history = defaultdict(list)
        history["train_loss"] = [0.5, 0.4, 0.3]
        history["val_loss"] = [0.6, 0.5, 0.4]

        checkpoint_manager.update_state(
            current_epoch=3,
            history=history
        )

        # Save checkpoint
        checkpoint_manager.save_checkpoint(checkpoint_path)

        # Load in new manager
        new_manager = CheckpointManager(
            model=checkpoint_manager.model,
            optimizer=checkpoint_manager.optimizer,
            checkpoint_dir=temp_dir,
            device=torch.device("cpu")
        )
        state = new_manager.load_checkpoint(checkpoint_path)

        # Verify history
        assert "history" in state
        assert state["history"]["train_loss"] == [0.5, 0.4, 0.3]
        assert state["history"]["val_loss"] == [0.6, 0.5, 0.4]

    def test_get_checkpoint_path(self, checkpoint_manager):
        """Test checkpoint path generation."""
        # Without metric
        path = checkpoint_manager.get_checkpoint_path(epoch=5)
        assert "checkpoint_epoch_5.pt" in path

        # With metric
        path = checkpoint_manager.get_checkpoint_path(epoch=5, metric_value=0.8542)
        assert "checkpoint_epoch_5_metric_0.8542.pt" in path

    def test_save_best_checkpoint(self, checkpoint_manager, temp_dir):
        """Test saving best checkpoint."""
        history = defaultdict(list)
        history["train_loss"] = [0.5, 0.4]

        checkpoint_manager.save_best_checkpoint(
            metric_value=0.92,
            current_epoch=10,
            global_step=200,
            history=history
        )

        best_path = os.path.join(temp_dir, "best_model.pt")
        assert os.path.exists(best_path)

        # Load and verify
        checkpoint = torch.load(best_path, map_location="cpu", weights_only=True)
        assert checkpoint["best_val_metric"] == 0.92
        assert checkpoint["current_epoch"] == 10
        assert checkpoint["global_step"] == 200

    def test_get_latest_checkpoint(self, checkpoint_manager, temp_dir):
        """Test finding latest checkpoint."""
        # No checkpoints
        latest = checkpoint_manager.get_latest_checkpoint()
        assert latest is None

        # Create multiple checkpoints
        for i in range(3):
            path = os.path.join(temp_dir, f"checkpoint_{i}.pt")
            checkpoint_manager.save_checkpoint(path)
            # Small delay to ensure different modification times
            import time
            time.sleep(0.01)

        # Get latest
        latest = checkpoint_manager.get_latest_checkpoint()
        assert latest is not None
        assert "checkpoint_2.pt" in latest

    def test_save_with_scheduler(self, model, optimizer, temp_dir):
        """Test saving checkpoint with learning rate scheduler."""
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10)

        manager = CheckpointManager(
            model=model,
            optimizer=optimizer,
            checkpoint_dir=temp_dir,
            scheduler=scheduler,
            device=torch.device("cpu")
        )

        checkpoint_path = os.path.join(temp_dir, "test_scheduler.pt")
        manager.save_checkpoint(checkpoint_path)

        # Verify scheduler state is saved
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        assert "scheduler_state_dict" in checkpoint

    def test_load_nonexistent_checkpoint(self, checkpoint_manager):
        """Test loading a nonexistent checkpoint."""
        result = checkpoint_manager.load_checkpoint("nonexistent.pt")
        assert result == {}

    def test_update_state(self, checkpoint_manager):
        """Test updating internal state."""
        checkpoint_manager.update_state(
            current_epoch=10,
            global_step=500,
            best_val_metric=0.95,
            patience_counter=3
        )

        assert checkpoint_manager.current_epoch == 10
        assert checkpoint_manager.global_step == 500
        assert checkpoint_manager.best_val_metric == 0.95
        assert checkpoint_manager.patience_counter == 3

    def test_partial_state_update(self, checkpoint_manager):
        """Test partial state updates."""
        checkpoint_manager.update_state(current_epoch=5, global_step=100)
        assert checkpoint_manager.current_epoch == 5
        assert checkpoint_manager.global_step == 100
        assert checkpoint_manager.best_val_metric == 0.0  # Should remain unchanged

        checkpoint_manager.update_state(best_val_metric=0.8)
        assert checkpoint_manager.current_epoch == 5  # Should remain unchanged
        assert checkpoint_manager.best_val_metric == 0.8

    def test_save_checkpoint_with_explicit_parameters(self, checkpoint_manager, temp_dir):
        """Test saving checkpoint with explicit parameters."""
        checkpoint_path = os.path.join(temp_dir, "explicit_params.pt")

        history = {"loss": [0.5, 0.4, 0.3]}

        checkpoint_manager.save_checkpoint(
            path=checkpoint_path,
            current_epoch=7,
            global_step=150,
            best_val_metric=0.88,
            patience_counter=2,
            history=history
        )

        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        assert checkpoint["current_epoch"] == 7
        assert checkpoint["global_step"] == 150
        assert checkpoint["best_val_metric"] == 0.88
        assert checkpoint["patience_counter"] == 2
        assert checkpoint["history"] == history

    def test_model_weights_preserved(self, checkpoint_manager, temp_dir, model):
        """Test that model weights are correctly saved and loaded."""
        checkpoint_path = os.path.join(temp_dir, "weights_test.pt")

        # Get initial weights
        initial_weights = {
            name: param.clone()
            for name, param in model.named_parameters()
        }

        # Save checkpoint
        checkpoint_manager.save_checkpoint(checkpoint_path)

        # Modify weights
        with torch.no_grad():
            for param in model.parameters():
                param.fill_(0.999)

        # Load checkpoint
        checkpoint_manager.load_checkpoint(checkpoint_path)

        # Verify weights match initial values
        for name, param in model.named_parameters():
            assert torch.allclose(param, initial_weights[name])


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
