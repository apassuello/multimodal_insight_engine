"""
Tests for the MultimodalTrainer coordinator.

This module tests the main trainer orchestration, ensuring proper
integration of all 5 extracted modules.
"""

import os
import tempfile
import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from src.training.trainers.multimodal.trainer import (
    MultimodalTrainer,
    ModalityBalancingScheduler,
)


class DictDataset(Dataset):
    """Dataset that returns dictionaries instead of tuples."""

    def __init__(self, num_samples=20):
        self.num_samples = num_samples
        self.images = torch.randn(num_samples, 3, 32, 32)
        self.text = torch.randint(0, 100, (num_samples, 10))

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        return {
            "images": self.images[idx],
            "text_data": self.text[idx],
        }


class SimpleMultimodalModel(nn.Module):
    """Simple model for testing."""

    def __init__(self, dim=64):
        super().__init__()
        self.vision_model = nn.Sequential(
            nn.Linear(3 * 32 * 32, dim),
            nn.ReLU(),
        )
        self.text_model = nn.Sequential(
            nn.Embedding(100, dim),
            nn.Linear(dim, dim),
        )
        self.fusion_module = nn.Linear(dim * 2, dim)
        self.fusion_module.fusion_dim = dim

    def forward(self, images=None, text_data=None, **kwargs):
        outputs = {}
        if images is not None:
            batch_size = images.size(0)
            vision_features = self.vision_model(images.view(batch_size, -1))
            outputs["image_features"] = vision_features
        if text_data is not None:
            # Handle text_data as indices
            if text_data.dim() == 2:
                text_data = text_data[:, 0]  # Take first token
            text_features = self.text_model[0](text_data)
            text_features = self.text_model[1](text_features)
            outputs["text_features"] = text_features
        return outputs


@pytest.fixture
def device():
    """Get the appropriate device for testing."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")


@pytest.fixture
def model():
    """Create a simple multimodal model."""
    return SimpleMultimodalModel(dim=64)


@pytest.fixture
def dataloaders():
    """Create simple dataloaders for testing."""
    # Create dummy data using DictDataset
    train_dataset = DictDataset(num_samples=20)
    val_dataset = DictDataset(num_samples=20)

    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False)

    return train_loader, val_loader


@pytest.fixture
def temp_dir():
    """Create a temporary directory for checkpoints and logs."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir


class TestModalityBalancingScheduler:
    """Tests for ModalityBalancingScheduler."""

    def test_initialization(self, model):
        """Test scheduler initialization."""
        optimizer = torch.optim.Adam(model.parameters())
        scheduler = ModalityBalancingScheduler(optimizer, target_ratio=1.0)

        assert scheduler.optimizer is optimizer
        assert scheduler.target_ratio == 1.0
        assert scheduler.check_interval == 10
        assert scheduler.step_count == 0

    def test_collect_gradient_stats(self, model):
        """Test gradient statistics collection."""
        optimizer = torch.optim.Adam(model.parameters())
        scheduler = ModalityBalancingScheduler(optimizer)

        # Run a forward pass and backward to create gradients
        images = torch.randn(2, 3, 32, 32)
        text = torch.randint(0, 100, (2,))
        outputs = model(images=images, text_data=text)

        loss = outputs["image_features"].sum() + outputs["text_features"].sum()
        loss.backward()

        avg_vision_grad, avg_text_grad = scheduler.collect_gradient_stats(model)

        assert avg_vision_grad >= 0
        assert avg_text_grad >= 0
        assert len(scheduler.vision_grad_history) == 1
        assert len(scheduler.text_grad_history) == 1

    def test_step_interval(self, model):
        """Test that scheduler only adjusts at intervals."""
        optimizer = torch.optim.Adam(model.parameters())
        scheduler = ModalityBalancingScheduler(optimizer, check_interval=5)

        # Create gradients
        images = torch.randn(2, 3, 32, 32)
        text = torch.randint(0, 100, (2,))
        outputs = model(images=images, text_data=text)
        loss = outputs["image_features"].sum() + outputs["text_features"].sum()
        loss.backward()

        # Step 4 times (should not adjust)
        for _ in range(4):
            scheduler.step(model)

        assert scheduler.step_count == 4

        # Step once more (should check on 5th step)
        scheduler.step(model)
        assert scheduler.step_count == 5


class TestMultimodalTrainer:
    """Tests for MultimodalTrainer."""

    def test_initialization(self, model, dataloaders, temp_dir, device):
        """Test trainer initialization."""
        train_loader, val_loader = dataloaders

        trainer = MultimodalTrainer(
            model=model,
            train_dataloader=train_loader,
            val_dataloader=val_loader,
            num_epochs=2,
            checkpoint_dir=os.path.join(temp_dir, "checkpoints"),
            log_dir=os.path.join(temp_dir, "logs"),
            device=device,
        )

        # Verify all modules are initialized
        assert trainer.checkpoint_manager is not None
        assert trainer.metrics_collector is not None
        assert trainer.training_loop is not None
        assert trainer.evaluator is not None
        assert trainer.data_handler is not None

        # Verify configuration
        assert trainer.num_epochs == 2
        assert trainer.device == device
        assert trainer.model is not None
        assert trainer.optimizer is not None
        assert trainer.loss_fn is not None

        # Verify directories created
        assert os.path.exists(os.path.join(temp_dir, "checkpoints"))
        assert os.path.exists(os.path.join(temp_dir, "logs"))

    def test_detect_model_dimension(self, model, dataloaders, temp_dir, device):
        """Test model dimension detection."""
        train_loader, _ = dataloaders

        trainer = MultimodalTrainer(
            model=model,
            train_dataloader=train_loader,
            checkpoint_dir=os.path.join(temp_dir, "checkpoints"),
            log_dir=os.path.join(temp_dir, "logs"),
            device=device,
        )

        dim = trainer._detect_model_dimension()
        assert dim == 64  # Should detect from fusion_module.fusion_dim

    def test_train_single_epoch(self, model, dataloaders, temp_dir, device):
        """Test training for a single epoch."""
        train_loader, val_loader = dataloaders

        trainer = MultimodalTrainer(
            model=model,
            train_dataloader=train_loader,
            val_dataloader=val_loader,
            num_epochs=1,
            checkpoint_dir=os.path.join(temp_dir, "checkpoints"),
            log_dir=os.path.join(temp_dir, "logs"),
            device=device,
            accumulation_steps=2,  # Test gradient accumulation
        )

        history = trainer.train()

        # Verify history contains metrics
        assert "train_loss" in history
        assert len(history["train_loss"]) == 1  # 1 epoch
        assert history["train_loss"][0] > 0  # Loss should be non-zero

        # Verify checkpoint was saved
        checkpoint_path = os.path.join(temp_dir, "checkpoints", "checkpoint_epoch_1.pt")
        assert os.path.exists(checkpoint_path)

    def test_train_multiple_epochs(self, model, dataloaders, temp_dir, device):
        """Test training for multiple epochs."""
        train_loader, val_loader = dataloaders

        trainer = MultimodalTrainer(
            model=model,
            train_dataloader=train_loader,
            val_dataloader=val_loader,
            num_epochs=3,
            checkpoint_dir=os.path.join(temp_dir, "checkpoints"),
            log_dir=os.path.join(temp_dir, "logs"),
            device=device,
        )

        history = trainer.train()

        # Verify history
        assert len(history["train_loss"]) == 3
        # Evaluator returns metrics with 'val_' prefix
        assert any(key.startswith("val_") for key in history.keys())
        # Check for at least one validation metric with 3 epochs
        val_metrics = [v for k, v in history.items() if k.startswith("val_")]
        assert len(val_metrics) > 0
        assert len(val_metrics[0]) == 3

        # Verify checkpoints
        for epoch in range(1, 4):
            checkpoint_path = os.path.join(
                temp_dir, "checkpoints", f"checkpoint_epoch_{epoch}.pt"
            )
            assert os.path.exists(checkpoint_path)

    def test_early_stopping(self, model, dataloaders, temp_dir, device):
        """Test early stopping functionality."""
        train_loader, val_loader = dataloaders

        trainer = MultimodalTrainer(
            model=model,
            train_dataloader=train_loader,
            val_dataloader=val_loader,
            num_epochs=10,  # Set high so early stopping triggers
            early_stopping_patience=2,  # Stop after 2 epochs without improvement
            checkpoint_dir=os.path.join(temp_dir, "checkpoints"),
            log_dir=os.path.join(temp_dir, "logs"),
            device=device,
        )

        # Mock the validation loss to increase (trigger early stopping)
        original_evaluate = trainer.evaluator.evaluate

        def mock_evaluate(*args, **kwargs):
            result = original_evaluate(*args, **kwargs)
            # Make validation loss increase each time
            trainer._mock_loss_counter = getattr(trainer, "_mock_loss_counter", 0) + 1
            result["loss"] = float(trainer._mock_loss_counter)
            return result

        trainer.evaluator.evaluate = mock_evaluate

        history = trainer.train()

        # Should stop early (before 10 epochs)
        # First epoch establishes baseline, next 2 show no improvement, stops on 3rd
        assert len(history["train_loss"]) <= 4  # Should stop early

        # Verify best model was saved
        best_checkpoint = os.path.join(temp_dir, "checkpoints", "best_model.pt")
        assert os.path.exists(best_checkpoint)

    def test_checkpoint_save_and_load(self, model, dataloaders, temp_dir, device):
        """Test saving and loading checkpoints."""
        train_loader, val_loader = dataloaders

        trainer = MultimodalTrainer(
            model=model,
            train_dataloader=train_loader,
            val_dataloader=val_loader,
            num_epochs=2,
            checkpoint_dir=os.path.join(temp_dir, "checkpoints"),
            log_dir=os.path.join(temp_dir, "logs"),
            device=device,
        )

        # Train for 2 epochs
        history = trainer.train()

        # Create a new trainer and load checkpoint
        new_trainer = MultimodalTrainer(
            model=SimpleMultimodalModel(dim=64),
            train_dataloader=train_loader,
            val_dataloader=val_loader,
            num_epochs=2,
            checkpoint_dir=os.path.join(temp_dir, "checkpoints"),
            log_dir=os.path.join(temp_dir, "logs"),
            device=device,
        )

        # Load the checkpoint from epoch 2
        checkpoint_path = os.path.join(temp_dir, "checkpoints", "checkpoint_epoch_2.pt")
        new_trainer.load_checkpoint(checkpoint_path)

        # Verify state was restored
        assert new_trainer.training_loop.current_epoch == 3  # Ready for next epoch
        assert new_trainer.training_loop.global_step > 0

    def test_evaluate_test(self, model, dataloaders, temp_dir, device):
        """Test evaluation on test set."""
        train_loader, _ = dataloaders

        # Create test loader
        test_dataset = DictDataset(num_samples=8)
        test_loader = DataLoader(test_dataset, batch_size=4)

        trainer = MultimodalTrainer(
            model=model,
            train_dataloader=train_loader,
            test_dataloader=test_loader,
            checkpoint_dir=os.path.join(temp_dir, "checkpoints"),
            log_dir=os.path.join(temp_dir, "logs"),
            device=device,
        )

        test_metrics = trainer.evaluate_test()

        # Evaluator returns metrics with 'global_' prefix
        assert "global_avg_recall@1" in test_metrics or "global_accuracy" in test_metrics
        assert len(test_metrics) > 0

    def test_modality_balancing(self, model, dataloaders, temp_dir, device):
        """Test modality gradient balancing."""
        train_loader, val_loader = dataloaders

        trainer = MultimodalTrainer(
            model=model,
            train_dataloader=train_loader,
            val_dataloader=val_loader,
            num_epochs=1,
            balance_modality_gradients=True,  # Enable balancing
            checkpoint_dir=os.path.join(temp_dir, "checkpoints"),
            log_dir=os.path.join(temp_dir, "logs"),
            device=device,
        )

        # Verify grad scheduler was created
        assert trainer.grad_scheduler is not None
        assert isinstance(trainer.grad_scheduler, ModalityBalancingScheduler)

        history = trainer.train()
        assert len(history["train_loss"]) == 1

    def test_mixed_precision(self, model, dataloaders, temp_dir):
        """Test mixed precision training."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available, skipping mixed precision test")

        train_loader, val_loader = dataloaders

        trainer = MultimodalTrainer(
            model=model,
            train_dataloader=train_loader,
            val_dataloader=val_loader,
            num_epochs=1,
            mixed_precision=True,  # Enable mixed precision
            checkpoint_dir=os.path.join(temp_dir, "checkpoints"),
            log_dir=os.path.join(temp_dir, "logs"),
            device=torch.device("cuda"),
        )

        # Verify mixed precision is enabled in training loop
        assert trainer.training_loop.mixed_precision is True
        assert trainer.training_loop.scaler is not None

        history = trainer.train()
        assert len(history["train_loss"]) == 1

    def test_gradient_clipping(self, model, dataloaders, temp_dir, device):
        """Test gradient clipping."""
        train_loader, val_loader = dataloaders

        trainer = MultimodalTrainer(
            model=model,
            train_dataloader=train_loader,
            val_dataloader=val_loader,
            num_epochs=1,
            clip_grad_norm=1.0,  # Enable gradient clipping
            checkpoint_dir=os.path.join(temp_dir, "checkpoints"),
            log_dir=os.path.join(temp_dir, "logs"),
            device=device,
        )

        # Verify gradient clipping is enabled
        assert trainer.training_loop.clip_grad_norm == 1.0

        history = trainer.train()
        assert len(history["train_loss"]) == 1

    def test_custom_optimizer_and_loss(self, model, dataloaders, temp_dir, device):
        """Test initialization with custom optimizer and loss."""
        train_loader, val_loader = dataloaders

        # Create custom loss that matches the expected signature
        class CustomContrastiveLoss(nn.Module):
            def forward(self, vision_features, text_features, **kwargs):
                # Simple loss - just sum of features
                loss = (vision_features - text_features).pow(2).mean()
                return {"loss": loss}

        custom_optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        custom_loss = CustomContrastiveLoss()

        trainer = MultimodalTrainer(
            model=model,
            train_dataloader=train_loader,
            val_dataloader=val_loader,
            optimizer=custom_optimizer,
            loss_fn=custom_loss,
            num_epochs=1,
            checkpoint_dir=os.path.join(temp_dir, "checkpoints"),
            log_dir=os.path.join(temp_dir, "logs"),
            device=device,
        )

        # Verify custom components are used
        assert trainer.optimizer is custom_optimizer
        assert trainer.loss_fn is custom_loss

        history = trainer.train()
        assert len(history["train_loss"]) == 1

    def test_no_validation_dataloader(self, model, dataloaders, temp_dir, device):
        """Test training without validation dataloader."""
        train_loader, _ = dataloaders

        trainer = MultimodalTrainer(
            model=model,
            train_dataloader=train_loader,
            val_dataloader=None,  # No validation
            num_epochs=2,
            checkpoint_dir=os.path.join(temp_dir, "checkpoints"),
            log_dir=os.path.join(temp_dir, "logs"),
            device=device,
        )

        history = trainer.train()

        # Should have training metrics but not validation
        assert "train_loss" in history
        assert "val_loss" not in history
        assert len(history["train_loss"]) == 2

    def test_scheduler_integration(self, model, dataloaders, temp_dir, device):
        """Test integration with learning rate scheduler."""
        train_loader, val_loader = dataloaders

        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.5)

        trainer = MultimodalTrainer(
            model=model,
            train_dataloader=train_loader,
            val_dataloader=val_loader,
            optimizer=optimizer,
            scheduler=scheduler,
            num_epochs=2,
            checkpoint_dir=os.path.join(temp_dir, "checkpoints"),
            log_dir=os.path.join(temp_dir, "logs"),
            device=device,
        )

        # Verify scheduler is stored
        assert trainer.scheduler is scheduler

        history = trainer.train()
        assert len(history["train_loss"]) == 2
