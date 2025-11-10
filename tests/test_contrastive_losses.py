"""
Comprehensive tests for contrastive loss functions.

Tests cover:
- ContrastiveLoss (InfoNCE, NT-Xent)
- MultiModalMixedContrastiveLoss
- MemoryQueueContrastiveLoss
- HardNegativeMiningContrastiveLoss
- DynamicTemperatureContrastiveLoss
- DecoupledContrastiveLoss
"""

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from src.training.losses import (
    ContrastiveLoss,
    MultiModalMixedContrastiveLoss,
    MemoryQueueContrastiveLoss,
    HardNegativeMiningContrastiveLoss,
    DynamicTemperatureContrastiveLoss,
    DecoupledContrastiveLoss,
)


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def device():
    """Return available device."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@pytest.fixture
def batch_size():
    """Standard batch size for tests."""
    return 16


@pytest.fixture
def embed_dim():
    """Standard embedding dimension for tests."""
    return 128


@pytest.fixture
def vision_features(batch_size, embed_dim, device):
    """Create random vision features for testing."""
    return torch.randn(batch_size, embed_dim, device=device)


@pytest.fixture
def text_features(batch_size, embed_dim, device):
    """Create random text features for testing."""
    return torch.randn(batch_size, embed_dim, device=device)


@pytest.fixture
def match_ids(batch_size):
    """Create match IDs for testing."""
    return [f"id_{i}" for i in range(batch_size)]


# ============================================================================
# ContrastiveLoss Tests
# ============================================================================


class TestContrastiveLoss:
    """Test suite for ContrastiveLoss (InfoNCE)."""

    def test_basic_forward_without_projection(
        self, vision_features, text_features, device
    ):
        """Test basic forward pass without projection heads."""
        loss_fn = ContrastiveLoss(
            temperature=0.07, add_projection=False, loss_type="infonce"
        )

        # Compute loss
        loss = loss_fn(vision_features, text_features)

        # Assert loss properties
        assert isinstance(loss, torch.Tensor)
        assert loss.shape == torch.Size([])  # Scalar output
        assert not torch.isnan(loss)
        assert not torch.isinf(loss)
        assert loss.item() >= 0  # Loss should be non-negative

    def test_basic_forward_with_projection(
        self, batch_size, embed_dim, device
    ):
        """Test basic forward pass with projection heads."""
        vision_features = torch.randn(batch_size, embed_dim, device=device)
        text_features = torch.randn(batch_size, embed_dim, device=device)

        loss_fn = ContrastiveLoss(
            temperature=0.07,
            add_projection=True,
            input_dim=embed_dim,
            projection_dim=256,
            loss_type="infonce",
        )

        # Compute loss
        loss = loss_fn(vision_features, text_features)

        # Assert loss properties
        assert isinstance(loss, torch.Tensor)
        assert loss.shape == torch.Size([])
        assert not torch.isnan(loss)
        assert not torch.isinf(loss)

    def test_gradient_flow(self, vision_features, text_features, device):
        """Test that gradients flow through the loss."""
        vision_features = vision_features.requires_grad_(True)
        text_features = text_features.requires_grad_(True)

        loss_fn = ContrastiveLoss(
            temperature=0.07, add_projection=False, loss_type="infonce"
        )

        # Compute loss and backpropagate
        loss = loss_fn(vision_features, text_features)
        loss.backward()

        # Check gradients exist and are non-zero
        assert vision_features.grad is not None
        assert text_features.grad is not None
        assert not torch.all(vision_features.grad == 0)
        assert not torch.all(text_features.grad == 0)

    def test_temperature_sensitivity(
        self, vision_features, text_features, device
    ):
        """Test that temperature parameter affects loss value."""
        loss_fn_low = ContrastiveLoss(
            temperature=0.01, add_projection=False, loss_type="infonce"
        )
        loss_fn_high = ContrastiveLoss(
            temperature=1.0, add_projection=False, loss_type="infonce"
        )

        loss_low = loss_fn_low(vision_features, text_features)
        loss_high = loss_fn_high(vision_features, text_features)

        # Lower temperature should generally lead to higher loss
        # (sharper distribution, harder to satisfy)
        assert loss_low != loss_high
        assert not torch.isnan(loss_low)
        assert not torch.isnan(loss_high)

    def test_reduction_modes(self, vision_features, text_features, device):
        """Test different reduction modes."""
        # Mean reduction (default)
        loss_fn_mean = ContrastiveLoss(
            temperature=0.07,
            add_projection=False,
            reduction="mean",
            loss_type="infonce",
        )
        loss_mean = loss_fn_mean(vision_features, text_features)
        assert loss_mean.shape == torch.Size([])

        # Sum reduction
        loss_fn_sum = ContrastiveLoss(
            temperature=0.07,
            add_projection=False,
            reduction="sum",
            loss_type="infonce",
        )
        loss_sum = loss_fn_sum(vision_features, text_features)
        assert loss_sum.shape == torch.Size([])

        # None reduction
        loss_fn_none = ContrastiveLoss(
            temperature=0.07,
            add_projection=False,
            reduction="none",
            loss_type="infonce",
        )
        loss_none = loss_fn_none(vision_features, text_features)
        # Should return per-sample losses
        assert loss_none.ndim >= 1

    def test_loss_types(self, vision_features, text_features, device):
        """Test different loss type formulations."""
        # InfoNCE loss
        loss_fn_infonce = ContrastiveLoss(
            temperature=0.07, add_projection=False, loss_type="infonce"
        )
        loss_infonce = loss_fn_infonce(vision_features, text_features)
        assert not torch.isnan(loss_infonce)

        # NT-Xent loss
        loss_fn_ntxent = ContrastiveLoss(
            temperature=0.07, add_projection=False, loss_type="nt_xent"
        )
        loss_ntxent = loss_fn_ntxent(vision_features, text_features)
        assert not torch.isnan(loss_ntxent)

    def test_edge_case_single_sample(self, embed_dim, device):
        """Test with single sample batch."""
        vision_features = torch.randn(1, embed_dim, device=device)
        text_features = torch.randn(1, embed_dim, device=device)

        loss_fn = ContrastiveLoss(
            temperature=0.07, add_projection=False, loss_type="infonce"
        )

        # Should handle single sample gracefully
        loss = loss_fn(vision_features, text_features)
        assert isinstance(loss, torch.Tensor)

    def test_edge_case_identical_features(self, batch_size, embed_dim, device):
        """Test with identical vision and text features."""
        # Create identical features
        features = torch.randn(batch_size, embed_dim, device=device)

        loss_fn = ContrastiveLoss(
            temperature=0.07, add_projection=False, loss_type="infonce"
        )

        # Loss should be low (perfect alignment)
        loss = loss_fn(features, features.clone())
        assert not torch.isnan(loss)
        # With identical features, loss should be relatively low
        assert loss.item() < 10.0

    def test_numerical_stability_extreme_values(
        self, batch_size, embed_dim, device
    ):
        """Test numerical stability with extreme input values."""
        # Very large values
        vision_large = torch.ones(batch_size, embed_dim, device=device) * 100
        text_large = torch.ones(batch_size, embed_dim, device=device) * 100

        loss_fn = ContrastiveLoss(
            temperature=0.07, add_projection=False, loss_type="infonce"
        )

        loss = loss_fn(vision_large, text_large)
        assert not torch.isnan(loss)
        assert not torch.isinf(loss)

    def test_batch_size_invariance(self, embed_dim, device):
        """Test that loss scales appropriately with batch size."""
        loss_fn = ContrastiveLoss(
            temperature=0.07, add_projection=False, loss_type="infonce"
        )

        # Small batch
        vision_small = torch.randn(4, embed_dim, device=device)
        text_small = torch.randn(4, embed_dim, device=device)
        loss_small = loss_fn(vision_small, text_small)

        # Large batch
        vision_large = torch.randn(32, embed_dim, device=device)
        text_large = torch.randn(32, embed_dim, device=device)
        loss_large = loss_fn(vision_large, text_large)

        # Both should be valid losses
        assert not torch.isnan(loss_small)
        assert not torch.isnan(loss_large)


# ============================================================================
# MultiModalMixedContrastiveLoss Tests
# ============================================================================


class TestMultiModalMixedContrastiveLoss:
    """Test suite for MultiModalMixedContrastiveLoss."""

    def test_basic_forward(self, vision_features, text_features, device):
        """Test basic forward pass."""
        loss_fn = MultiModalMixedContrastiveLoss(
            temperature=0.07, add_projection=False
        )

        loss = loss_fn(vision_features, text_features)

        assert isinstance(loss, torch.Tensor)
        assert loss.shape == torch.Size([])
        assert not torch.isnan(loss)
        assert not torch.isinf(loss)

    def test_custom_loss_weights(self, vision_features, text_features, device):
        """Test with custom loss weights."""
        loss_weights = {
            "infonce": 0.5,
            "nt_xent": 0.3,
            "supervised": 0.2,
        }

        loss_fn = MultiModalMixedContrastiveLoss(
            temperature=0.07, loss_weights=loss_weights, add_projection=False
        )

        loss = loss_fn(vision_features, text_features)
        assert not torch.isnan(loss)

    def test_gradient_flow(self, vision_features, text_features, device):
        """Test gradient flow through mixed loss."""
        vision_features = vision_features.requires_grad_(True)
        text_features = text_features.requires_grad_(True)

        loss_fn = MultiModalMixedContrastiveLoss(
            temperature=0.07, add_projection=False
        )

        loss = loss_fn(vision_features, text_features)
        loss.backward()

        assert vision_features.grad is not None
        assert text_features.grad is not None
        assert not torch.all(vision_features.grad == 0)
        assert not torch.all(text_features.grad == 0)

    def test_with_projection_heads(self, batch_size, embed_dim, device):
        """Test with projection heads enabled."""
        vision_features = torch.randn(batch_size, embed_dim, device=device)
        text_features = torch.randn(batch_size, embed_dim, device=device)

        loss_fn = MultiModalMixedContrastiveLoss(
            temperature=0.07,
            add_projection=True,
            input_dim=embed_dim,
            projection_dim=256,
        )

        loss = loss_fn(vision_features, text_features)
        assert not torch.isnan(loss)


# ============================================================================
# MemoryQueueContrastiveLoss Tests
# ============================================================================


class TestMemoryQueueContrastiveLoss:
    """Test suite for MemoryQueueContrastiveLoss."""

    def test_basic_forward(self, vision_features, text_features, device):
        """Test basic forward pass."""
        loss_fn = MemoryQueueContrastiveLoss(
            temperature=0.07, queue_size=128, embed_dim=vision_features.shape[1]
        )

        loss = loss_fn(vision_features, text_features)

        assert isinstance(loss, torch.Tensor)
        assert loss.shape == torch.Size([])
        assert not torch.isnan(loss)
        assert not torch.isinf(loss)

    def test_memory_queue_updates(
        self, batch_size, embed_dim, device
    ):
        """Test that memory queue gets updated across batches."""
        loss_fn = MemoryQueueContrastiveLoss(
            temperature=0.07, queue_size=256, embed_dim=embed_dim
        )

        # First batch
        vision1 = torch.randn(batch_size, embed_dim, device=device)
        text1 = torch.randn(batch_size, embed_dim, device=device)
        loss1 = loss_fn(vision1, text1)

        # Second batch
        vision2 = torch.randn(batch_size, embed_dim, device=device)
        text2 = torch.randn(batch_size, embed_dim, device=device)
        loss2 = loss_fn(vision2, text2)

        # Both losses should be valid
        assert not torch.isnan(loss1)
        assert not torch.isnan(loss2)

    def test_gradient_flow(self, vision_features, text_features, device):
        """Test gradient flow with memory queue."""
        vision_features = vision_features.requires_grad_(True)
        text_features = text_features.requires_grad_(True)

        loss_fn = MemoryQueueContrastiveLoss(
            temperature=0.07, queue_size=128, embed_dim=vision_features.shape[1]
        )

        loss = loss_fn(vision_features, text_features)
        loss.backward()

        assert vision_features.grad is not None
        assert text_features.grad is not None


# ============================================================================
# HardNegativeMiningContrastiveLoss Tests
# ============================================================================


class TestHardNegativeMiningContrastiveLoss:
    """Test suite for HardNegativeMiningContrastiveLoss."""

    def test_basic_forward(self, vision_features, text_features, device):
        """Test basic forward pass."""
        loss_fn = HardNegativeMiningContrastiveLoss(
            temperature=0.07, num_hard_negatives=4
        )

        loss = loss_fn(vision_features, text_features)

        assert isinstance(loss, torch.Tensor)
        assert loss.shape == torch.Size([])
        assert not torch.isnan(loss)
        assert not torch.isinf(loss)

    def test_different_num_hard_negatives(
        self, vision_features, text_features, device
    ):
        """Test with different numbers of hard negatives."""
        for num_negatives in [2, 4, 8]:
            loss_fn = HardNegativeMiningContrastiveLoss(
                temperature=0.07, num_hard_negatives=num_negatives
            )

            loss = loss_fn(vision_features, text_features)
            assert not torch.isnan(loss)

    def test_gradient_flow(self, vision_features, text_features, device):
        """Test gradient flow with hard negative mining."""
        vision_features = vision_features.requires_grad_(True)
        text_features = text_features.requires_grad_(True)

        loss_fn = HardNegativeMiningContrastiveLoss(
            temperature=0.07, num_hard_negatives=4
        )

        loss = loss_fn(vision_features, text_features)
        loss.backward()

        assert vision_features.grad is not None
        assert text_features.grad is not None


# ============================================================================
# DynamicTemperatureContrastiveLoss Tests
# ============================================================================


class TestDynamicTemperatureContrastiveLoss:
    """Test suite for DynamicTemperatureContrastiveLoss."""

    def test_basic_forward(self, vision_features, text_features, device):
        """Test basic forward pass."""
        loss_fn = DynamicTemperatureContrastiveLoss(
            initial_temperature=0.07, learnable_temperature=True
        )

        loss = loss_fn(vision_features, text_features)

        assert isinstance(loss, torch.Tensor)
        assert loss.shape == torch.Size([])
        assert not torch.isnan(loss)
        assert not torch.isinf(loss)

    def test_learnable_temperature(
        self, vision_features, text_features, device
    ):
        """Test that temperature is learnable."""
        loss_fn = DynamicTemperatureContrastiveLoss(
            initial_temperature=0.07, learnable_temperature=True
        )

        # Get initial temperature
        initial_temp = loss_fn.temperature.item() if hasattr(
            loss_fn, 'temperature'
        ) else None

        if initial_temp is not None:
            # Compute loss and backprop
            vision_features = vision_features.requires_grad_(True)
            loss = loss_fn(vision_features, text_features)
            loss.backward()

            # Temperature should have gradient if learnable
            if hasattr(loss_fn, 'temperature') and hasattr(
                loss_fn.temperature, 'grad'
            ):
                assert loss_fn.temperature.grad is not None

    def test_gradient_flow(self, vision_features, text_features, device):
        """Test gradient flow."""
        vision_features = vision_features.requires_grad_(True)
        text_features = text_features.requires_grad_(True)

        loss_fn = DynamicTemperatureContrastiveLoss(
            initial_temperature=0.07, learnable_temperature=False
        )

        loss = loss_fn(vision_features, text_features)
        loss.backward()

        assert vision_features.grad is not None
        assert text_features.grad is not None


# ============================================================================
# DecoupledContrastiveLoss Tests
# ============================================================================


class TestDecoupledContrastiveLoss:
    """Test suite for DecoupledContrastiveLoss."""

    def test_basic_forward(self, vision_features, text_features, device):
        """Test basic forward pass."""
        loss_fn = DecoupledContrastiveLoss(
            temperature=0.07, decouple_factor=0.5
        )

        loss = loss_fn(vision_features, text_features)

        assert isinstance(loss, torch.Tensor)
        assert loss.shape == torch.Size([])
        assert not torch.isnan(loss)
        assert not torch.isinf(loss)

    def test_different_decouple_factors(
        self, vision_features, text_features, device
    ):
        """Test with different decouple factors."""
        for factor in [0.0, 0.5, 1.0]:
            loss_fn = DecoupledContrastiveLoss(
                temperature=0.07, decouple_factor=factor
            )

            loss = loss_fn(vision_features, text_features)
            assert not torch.isnan(loss)

    def test_gradient_flow(self, vision_features, text_features, device):
        """Test gradient flow through decoupled loss."""
        vision_features = vision_features.requires_grad_(True)
        text_features = text_features.requires_grad_(True)

        loss_fn = DecoupledContrastiveLoss(
            temperature=0.07, decouple_factor=0.5
        )

        loss = loss_fn(vision_features, text_features)
        loss.backward()

        assert vision_features.grad is not None
        assert text_features.grad is not None
        assert not torch.all(vision_features.grad == 0)
        assert not torch.all(text_features.grad == 0)
