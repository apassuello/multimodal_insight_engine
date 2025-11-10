"""Tests for TrainingLoop.

Tests training loop execution, gradient handling, and diagnostics.
"""

import tempfile
import shutil

import pytest
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from src.training.trainers.multimodal.training_loop import TrainingLoop


class SimpleModel(nn.Module):
    """Simple model for testing."""

    def __init__(self, input_dim=10, hidden_dim=20, output_dim=5):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        return {"output": self.fc2(x)}


class SimpleLoss(nn.Module):
    """Simple loss for testing."""

    def forward(self, output, target):
        loss = nn.functional.mse_loss(output, target)
        return {"loss": loss, "accuracy": 0.85}


class CurriculumLoss(nn.Module):
    """Loss with curriculum support."""

    def __init__(self):
        super().__init__()
        self.current_epoch = 0
        self.current_step = 0
        self.current_phase = "pretrain"

    def update_epoch(self, epoch):
        self.current_epoch = epoch
        if epoch >= 5:
            self.current_phase = "finetune"

    def update_step(self, step, total_steps):
        self.current_step = step

    def forward(self, output, target):
        loss = nn.functional.mse_loss(output, target)
        return {
            "loss": loss,
            "accuracy": 0.85,
            "current_phase": self.current_phase,
        }


@pytest.fixture
def device():
    """Get test device."""
    return torch.device("cpu")


@pytest.fixture
def model():
    """Create a simple model."""
    return SimpleModel()


@pytest.fixture
def loss_fn():
    """Create a simple loss function."""
    return SimpleLoss()


@pytest.fixture
def optimizer(model):
    """Create an optimizer."""
    return optim.Adam(model.parameters(), lr=0.001)


@pytest.fixture
def dataloader():
    """Create a simple dataloader."""
    # Create dummy data
    x = torch.randn(50, 10)
    y = torch.randn(50, 5)
    dataset = TensorDataset(x, y)
    return DataLoader(dataset, batch_size=10, shuffle=True)


@pytest.fixture
def training_loop(model, loss_fn, optimizer, device):
    """Create a TrainingLoop instance."""
    return TrainingLoop(
        model=model,
        loss_fn=loss_fn,
        optimizer=optimizer,
        device=device,
        mixed_precision=False,
        accumulation_steps=1,
        clip_grad_norm=None,
        log_steps=10,
        enable_diagnostics=True,
        check_feature_collapse=False,
    )


# Helper functions for training
def prepare_model_inputs(batch):
    """Prepare inputs for model."""
    return {"x": batch[0]}


def prepare_loss_inputs(batch, outputs):
    """Prepare inputs for loss function."""
    return {"output": outputs["output"], "target": batch[1]}


def to_device(batch, device=torch.device("cpu")):
    """Move batch to device."""
    return tuple(item.to(device) for item in batch)


class TestTrainingLoop:
    """Tests for TrainingLoop class."""

    def test_initialization(self, training_loop):
        """Test TrainingLoop initialization."""
        assert training_loop.global_step == 0
        assert training_loop.current_epoch == 0
        assert training_loop.scaler is None  # No mixed precision
        assert training_loop.enable_diagnostics is True

    def test_train_epoch(self, training_loop, dataloader):
        """Test training for one epoch."""
        metrics = training_loop.train_epoch(
            dataloader=dataloader,
            epoch=0,
            num_epochs=10,
            prepare_model_inputs_fn=prepare_model_inputs,
            prepare_loss_inputs_fn=prepare_loss_inputs,
            to_device_fn=lambda b: to_device(b, training_loop.device),
        )

        assert "loss" in metrics
        assert isinstance(metrics["loss"], float)
        assert training_loop.global_step > 0

    def test_multiple_epochs(self, training_loop, dataloader):
        """Test training for multiple epochs."""
        initial_loss = None

        for epoch in range(3):
            metrics = training_loop.train_epoch(
                dataloader=dataloader,
                epoch=epoch,
                num_epochs=3,
                prepare_model_inputs_fn=prepare_model_inputs,
                prepare_loss_inputs_fn=prepare_loss_inputs,
                to_device_fn=lambda b: to_device(b, training_loop.device),
            )

            if initial_loss is None:
                initial_loss = metrics["loss"]

        # Loss should change over epochs (not necessarily decrease in this simple test)
        assert metrics["loss"] != initial_loss

    def test_gradient_accumulation(self, model, loss_fn, optimizer, device, dataloader):
        """Test gradient accumulation."""
        loop = TrainingLoop(
            model=model,
            loss_fn=loss_fn,
            optimizer=optimizer,
            device=device,
            accumulation_steps=2,  # Accumulate over 2 steps
        )

        metrics = loop.train_epoch(
            dataloader=dataloader,
            epoch=0,
            num_epochs=1,
            prepare_model_inputs_fn=prepare_model_inputs,
            prepare_loss_inputs_fn=prepare_loss_inputs,
            to_device_fn=lambda b: to_device(b, device),
        )

        assert "loss" in metrics
        assert loop.global_step == len(dataloader)

    def test_gradient_clipping(self, model, loss_fn, optimizer, device, dataloader):
        """Test gradient clipping."""
        loop = TrainingLoop(
            model=model,
            loss_fn=loss_fn,
            optimizer=optimizer,
            device=device,
            clip_grad_norm=1.0,
        )

        metrics = loop.train_epoch(
            dataloader=dataloader,
            epoch=0,
            num_epochs=1,
            prepare_model_inputs_fn=prepare_model_inputs,
            prepare_loss_inputs_fn=prepare_loss_inputs,
            to_device_fn=lambda b: to_device(b, device),
        )

        assert "loss" in metrics

        # Check that gradients were clipped
        total_norm = 0.0
        for p in model.parameters():
            if p.grad is not None:
                total_norm += p.grad.norm().item() ** 2
        total_norm = total_norm ** 0.5

        # After clipping, total norm should be <= clip_grad_norm
        # (may be less if gradients were small to begin with)
        assert total_norm <= 1.1  # Small tolerance for numerical errors

    def test_with_scheduler(self, model, loss_fn, optimizer, device, dataloader):
        """Test training with learning rate scheduler."""
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

        loop = TrainingLoop(
            model=model,
            loss_fn=loss_fn,
            optimizer=optimizer,
            device=device,
            scheduler=scheduler,
        )

        initial_lr = optimizer.param_groups[0]["lr"]

        metrics = loop.train_epoch(
            dataloader=dataloader,
            epoch=0,
            num_epochs=1,
            prepare_model_inputs_fn=prepare_model_inputs,
            prepare_loss_inputs_fn=prepare_loss_inputs,
            to_device_fn=lambda b: to_device(b, device),
        )

        # Learning rate should have changed
        final_lr = optimizer.param_groups[0]["lr"]
        # LR changes after each step, so should be different
        assert final_lr != initial_lr

    def test_curriculum_loss(self, model, optimizer, device, dataloader):
        """Test training with curriculum loss."""
        curriculum_loss = CurriculumLoss()

        loop = TrainingLoop(
            model=model,
            loss_fn=curriculum_loss,
            optimizer=optimizer,
            device=device,
        )

        # Train for multiple epochs to trigger phase change
        for epoch in range(6):
            loop.train_epoch(
                dataloader=dataloader,
                epoch=epoch,
                num_epochs=10,
                prepare_model_inputs_fn=prepare_model_inputs,
                prepare_loss_inputs_fn=prepare_loss_inputs,
                to_device_fn=lambda b: to_device(b, device),
            )

        # Check that curriculum was updated
        assert curriculum_loss.current_epoch == 5
        assert curriculum_loss.current_phase == "finetune"

    def test_set_global_step(self, training_loop):
        """Test setting global step."""
        training_loop.set_global_step(100)
        assert training_loop.global_step == 100

    def test_set_current_epoch(self, training_loop):
        """Test setting current epoch."""
        training_loop.set_current_epoch(5)
        assert training_loop.current_epoch == 5

    def test_disable_diagnostics(self, model, loss_fn, optimizer, device, dataloader):
        """Test training with diagnostics disabled."""
        loop = TrainingLoop(
            model=model,
            loss_fn=loss_fn,
            optimizer=optimizer,
            device=device,
            enable_diagnostics=False,
        )

        metrics = loop.train_epoch(
            dataloader=dataloader,
            epoch=0,
            num_epochs=1,
            prepare_model_inputs_fn=prepare_model_inputs,
            prepare_loss_inputs_fn=prepare_loss_inputs,
            to_device_fn=lambda b: to_device(b, device),
        )

        assert "loss" in metrics

    def test_model_weights_update(self, model, loss_fn, optimizer, device, dataloader):
        """Test that model weights are updated during training."""
        # Get initial weights
        initial_weights = {
            name: param.clone()
            for name, param in model.named_parameters()
        }

        loop = TrainingLoop(
            model=model,
            loss_fn=loss_fn,
            optimizer=optimizer,
            device=device,
        )

        metrics = loop.train_epoch(
            dataloader=dataloader,
            epoch=0,
            num_epochs=1,
            prepare_model_inputs_fn=prepare_model_inputs,
            prepare_loss_inputs_fn=prepare_loss_inputs,
            to_device_fn=lambda b: to_device(b, device),
        )

        # Check that at least some weights changed
        weights_changed = False
        for name, param in model.named_parameters():
            if not torch.allclose(param, initial_weights[name]):
                weights_changed = True
                break

        assert weights_changed, "Model weights should be updated during training"

    def test_loss_anomaly_detection(self, model, optimizer, device, dataloader, capsys):
        """Test detection of loss anomalies."""
        class AnomalousLoss(nn.Module):
            def __init__(self):
                super().__init__()
                self.call_count = 0

            def forward(self, output, target):
                self.call_count += 1
                # Return NaN on second call
                if self.call_count == 2:
                    loss = torch.tensor(float('nan'))
                else:
                    loss = nn.functional.mse_loss(output, target)
                return {"loss": loss, "accuracy": 0.85}

        anomalous_loss = AnomalousLoss()

        loop = TrainingLoop(
            model=model,
            loss_fn=anomalous_loss,
            optimizer=optimizer,
            device=device,
        )

        # This should log an error about NaN loss
        metrics = loop.train_epoch(
            dataloader=dataloader,
            epoch=0,
            num_epochs=1,
            prepare_model_inputs_fn=prepare_model_inputs,
            prepare_loss_inputs_fn=prepare_loss_inputs,
            to_device_fn=lambda b: to_device(b, device),
        )

        # The training should complete despite NaN
        assert "loss" in metrics

    def test_feature_collapse_detection(self, device, dataloader):
        """Test feature collapse detection for multimodal training."""
        class MultimodalModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.vision_encoder = nn.Linear(10, 128)
                self.text_encoder = nn.Linear(10, 128)

            def forward(self, x):
                vision_features = self.vision_encoder(x)
                text_features = self.text_encoder(x)
                return {
                    "vision_features": vision_features,
                    "text_features": text_features,
                }

        class MultimodalLoss(nn.Module):
            def forward(self, vision_features, text_features):
                loss = torch.mean((vision_features - text_features) ** 2)
                return {"loss": loss}

        def prepare_multimodal_inputs(batch):
            return {"x": batch[0]}

        def prepare_multimodal_loss_inputs(batch, outputs):
            return {
                "vision_features": outputs["vision_features"],
                "text_features": outputs["text_features"],
            }

        model = MultimodalModel()
        loss_fn = MultimodalLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)

        loop = TrainingLoop(
            model=model,
            loss_fn=loss_fn,
            optimizer=optimizer,
            device=device,
            check_feature_collapse=True,
        )

        metrics = loop.train_epoch(
            dataloader=dataloader,
            epoch=0,
            num_epochs=1,
            prepare_model_inputs_fn=prepare_multimodal_inputs,
            prepare_loss_inputs_fn=prepare_multimodal_loss_inputs,
            to_device_fn=lambda b: to_device(b, device),
        )

        assert "loss" in metrics

    def test_periodic_evaluation(self, training_loop, dataloader):
        """Test periodic evaluation during training."""
        evaluation_count = [0]  # Use list to allow modification in closure

        def mock_evaluation():
            evaluation_count[0] += 1
            return {"val_loss": 0.5}

        metrics = training_loop.train_epoch(
            dataloader=dataloader,
            epoch=0,
            num_epochs=1,
            prepare_model_inputs_fn=prepare_model_inputs,
            prepare_loss_inputs_fn=prepare_loss_inputs,
            to_device_fn=lambda b: to_device(b, training_loop.device),
            evaluation_fn=mock_evaluation,
            evaluation_steps=2,  # Evaluate every 2 steps
        )

        # Should have called evaluation multiple times
        assert evaluation_count[0] > 0

    def test_nested_metrics(self, model, optimizer, device, dataloader):
        """Test handling of nested metrics."""
        class NestedMetricsLoss(nn.Module):
            def forward(self, output, target):
                loss = nn.functional.mse_loss(output, target)
                return {
                    "loss": loss,
                    "accuracy": 0.85,
                    "recalls": {"top1": 0.8, "top5": 0.95},
                }

        loss_fn = NestedMetricsLoss()

        loop = TrainingLoop(
            model=model,
            loss_fn=loss_fn,
            optimizer=optimizer,
            device=device,
        )

        metrics = loop.train_epoch(
            dataloader=dataloader,
            epoch=0,
            num_epochs=1,
            prepare_model_inputs_fn=prepare_model_inputs,
            prepare_loss_inputs_fn=prepare_loss_inputs,
            to_device_fn=lambda b: to_device(b, device),
        )

        assert "loss" in metrics
        assert "recalls" in metrics
        assert isinstance(metrics["recalls"], dict)
        assert "top1" in metrics["recalls"]
        assert "top5" in metrics["recalls"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
