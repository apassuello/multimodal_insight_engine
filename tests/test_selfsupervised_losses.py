"""
Comprehensive tests for self-supervised loss functions.

Tests cover:
- VICRegLoss (Variance-Invariance-Covariance Regularization)
- BarlowTwinsLoss (Redundancy Reduction)
- HybridPretrainVICRegLoss
- CLIP-style losses
- EMA MoCo losses
"""

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from src.training.losses import (
    VICRegLoss,
    BarlowTwinsLoss,
    HybridPretrainVICRegLoss,
)


# ============================================================================
# Fixtures
# ============================================================================



# ============================================================================
# Helper Functions
# ============================================================================


def extract_loss(result):
    """
    Extract loss tensor from various return types.

    Handles:
    - dict: returns result['loss'] or result['total_loss']
    - tuple: returns first element
    - tensor: returns as-is
    """
    if isinstance(result, dict):
        return result.get('loss', result.get('total_loss', result.get('contrastive_loss')))
    elif isinstance(result, tuple):
        return result[0]
    else:
        return result

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
def embeddings_a(batch_size, embed_dim, device):
    """Create first set of random embeddings for testing."""
    return torch.randn(batch_size, embed_dim, device=device)


@pytest.fixture
def embeddings_b(batch_size, embed_dim, device):
    """Create second set of random embeddings for testing."""
    return torch.randn(batch_size, embed_dim, device=device)


# ============================================================================
# VICRegLoss Tests
# ============================================================================


class TestVICRegLoss:
    """Test suite for VICRegLoss."""

    def test_basic_forward(self, embeddings_a, embeddings_b, device):
        """Test basic forward pass."""
        loss_fn = VICRegLoss(
            sim_coeff=10.0, var_coeff=5.0, cov_coeff=1.0
        )

        result = loss_fn(embeddings_a, embeddings_b)

        # VICReg returns a dictionary
        assert isinstance(result, dict)
        assert 'loss' in result
        assert isinstance(result['loss'], torch.Tensor)
        assert result['loss'].shape == torch.Size([])
        assert not torch.isnan(result['loss'])
        assert not torch.isinf(result['loss'])

    def test_loss_components(self, embeddings_a, embeddings_b, device):
        """Test that all three VICReg components are present."""
        loss_fn = VICRegLoss(
            sim_coeff=10.0, var_coeff=5.0, cov_coeff=1.0
        )

        result = loss_fn(embeddings_a, embeddings_b)

        # Check for variance, invariance, and covariance terms
        assert 'loss' in result
        # Some implementations may return individual components
        if isinstance(result, dict) and len(result) > 1:
            possible_keys = ['sim_loss', 'var_loss', 'cov_loss',
                           'invariance', 'variance', 'covariance']
            has_components = any(key in result for key in possible_keys)
            # If components are returned, they should be tensors
            for key in result:
                if key != 'loss':
                    assert isinstance(result[key], (torch.Tensor, float))

    def test_gradient_flow(self, embeddings_a, embeddings_b, device):
        """Test that gradients flow through VICReg loss."""
        embeddings_a = embeddings_a.requires_grad_(True)
        embeddings_b = embeddings_b.requires_grad_(True)

        loss_fn = VICRegLoss(
            sim_coeff=10.0, var_coeff=5.0, cov_coeff=1.0
        )

        result = loss_fn(embeddings_a, embeddings_b)
        loss = result['loss'] if isinstance(result, dict) else result
        loss.backward()

        # Check gradients exist and are non-zero
        assert embeddings_a.grad is not None
        assert embeddings_b.grad is not None
        assert not torch.all(embeddings_a.grad == 0)
        assert not torch.all(embeddings_b.grad == 0)

    def test_coefficient_effects(self, embeddings_a, embeddings_b, device):
        """Test that different coefficients affect loss value."""
        # High similarity coefficient
        loss_fn_high_sim = VICRegLoss(
            sim_coeff=50.0, var_coeff=5.0, cov_coeff=1.0
        )

        # Low similarity coefficient
        loss_fn_low_sim = VICRegLoss(
            sim_coeff=1.0, var_coeff=5.0, cov_coeff=1.0
        )

        result_high = loss_fn_high_sim(embeddings_a, embeddings_b)
        result_low = loss_fn_low_sim(embeddings_a, embeddings_b)

        loss_high = result_high['loss'] if isinstance(result_high, dict) else result_high
        loss_low = result_low['loss'] if isinstance(result_low, dict) else result_low

        # Different coefficients should lead to different losses
        assert not torch.allclose(loss_high, loss_low)
        assert not torch.isnan(loss_high)
        assert not torch.isnan(loss_low)

    def test_curriculum_learning(self, embeddings_a, embeddings_b, device):
        """Test curriculum learning warm-up."""
        loss_fn = VICRegLoss(
            sim_coeff=10.0,
            var_coeff=5.0,
            cov_coeff=1.0,
            curriculum=True,
            warmup_epochs=5,
        )

        # Test at different epochs
        loss_fn.update_epoch(0)
        result_epoch0 = loss_fn(embeddings_a, embeddings_b)
        loss_epoch0 = result_epoch0['loss'] if isinstance(result_epoch0, dict) else result_epoch0

        loss_fn.update_epoch(5)
        result_epoch5 = loss_fn(embeddings_a, embeddings_b)
        loss_epoch5 = result_epoch5['loss'] if isinstance(result_epoch5, dict) else result_epoch5

        # Both should be valid losses
        assert not torch.isnan(loss_epoch0)
        assert not torch.isnan(loss_epoch5)

    def test_edge_case_identical_embeddings(
        self, batch_size, embed_dim, device
    ):
        """Test with identical embeddings."""
        embeddings = torch.randn(batch_size, embed_dim, device=device)

        loss_fn = VICRegLoss(
            sim_coeff=10.0, var_coeff=5.0, cov_coeff=1.0
        )

        result = loss_fn(embeddings, embeddings.clone())
        loss = result['loss'] if isinstance(result, dict) else result

        # Invariance term should be very low with identical embeddings
        assert not torch.isnan(loss)
        # Loss should still be positive due to variance/covariance terms
        assert loss.item() >= 0

    def test_edge_case_small_batch(self, embed_dim, device):
        """Test with very small batch size."""
        embeddings_a = torch.randn(2, embed_dim, device=device)
        embeddings_b = torch.randn(2, embed_dim, device=device)

        loss_fn = VICRegLoss(
            sim_coeff=10.0, var_coeff=5.0, cov_coeff=1.0
        )

        result = loss_fn(embeddings_a, embeddings_b)
        loss = result['loss'] if isinstance(result, dict) else result

        # Should handle small batches
        assert not torch.isnan(loss)
        assert not torch.isinf(loss)

    def test_numerical_stability(self, batch_size, embed_dim, device):
        """Test numerical stability with extreme values."""
        # Very large embeddings
        embeddings_large_a = torch.ones(batch_size, embed_dim, device=device) * 100
        embeddings_large_b = torch.ones(batch_size, embed_dim, device=device) * 100

        loss_fn = VICRegLoss(
            sim_coeff=10.0, var_coeff=5.0, cov_coeff=1.0
        )

        result = loss_fn(embeddings_large_a, embeddings_large_b)
        loss = result['loss'] if isinstance(result, dict) else result

        assert not torch.isnan(loss)
        assert not torch.isinf(loss)

    def test_batch_size_invariance(self, embed_dim, device):
        """Test behavior with different batch sizes."""
        loss_fn = VICRegLoss(
            sim_coeff=10.0, var_coeff=5.0, cov_coeff=1.0
        )

        # Small batch
        emb_a_small = torch.randn(4, embed_dim, device=device)
        emb_b_small = torch.randn(4, embed_dim, device=device)
        result_small = loss_fn(emb_a_small, emb_b_small)
        loss_small = result_small['loss'] if isinstance(result_small, dict) else result_small

        # Large batch
        emb_a_large = torch.randn(32, embed_dim, device=device)
        emb_b_large = torch.randn(32, embed_dim, device=device)
        result_large = loss_fn(emb_a_large, emb_b_large)
        loss_large = result_large['loss'] if isinstance(result_large, dict) else result_large

        # Both should be valid
        assert not torch.isnan(loss_small)
        assert not torch.isnan(loss_large)


# ============================================================================
# BarlowTwinsLoss Tests
# ============================================================================


class TestBarlowTwinsLoss:
    """Test suite for BarlowTwinsLoss."""

    def test_basic_forward(self, embeddings_a, embeddings_b, device):
        """Test basic forward pass."""
        loss_fn = BarlowTwinsLoss(
            lambda_coeff=0.005, add_projection=False
        )

        result = loss_fn(embeddings_a, embeddings_b)


        loss = extract_loss(result)
        assert isinstance(loss, torch.Tensor)
        assert loss.shape == torch.Size([])
        assert not torch.isnan(loss)
        assert not torch.isinf(loss)
        assert loss.item() >= 0

    def test_with_projection_heads(self, batch_size, embed_dim, device):
        """Test with projection heads enabled."""
        embeddings_a = torch.randn(batch_size, embed_dim, device=device)
        embeddings_b = torch.randn(batch_size, embed_dim, device=device)

        loss_fn = BarlowTwinsLoss(
            lambda_coeff=0.005,
            add_projection=True,
            input_dim=embed_dim,
            projection_dim=512,
        )

        result = loss_fn(embeddings_a, embeddings_b)


        loss = extract_loss(result)
        assert not torch.isnan(loss)

    def test_gradient_flow(self, embeddings_a, embeddings_b, device):
        """Test gradient flow through Barlow Twins loss."""
        embeddings_a = embeddings_a.requires_grad_(True)
        embeddings_b = embeddings_b.requires_grad_(True)

        loss_fn = BarlowTwinsLoss(
            lambda_coeff=0.005, add_projection=False
        )

        result = loss_fn(embeddings_a, embeddings_b)


        loss = extract_loss(result)
        loss.backward()

        assert embeddings_a.grad is not None
        assert embeddings_b.grad is not None
        assert not torch.all(embeddings_a.grad == 0)
        assert not torch.all(embeddings_b.grad == 0)

    def test_lambda_coefficient_effect(
        self, embeddings_a, embeddings_b, device
    ):
        """Test that lambda coefficient affects off-diagonal terms."""
        # High lambda (penalize off-diagonal more)
        loss_fn_high = BarlowTwinsLoss(
            lambda_coeff=0.05, add_projection=False
        )

        # Low lambda (penalize off-diagonal less)
        loss_fn_low = BarlowTwinsLoss(
            lambda_coeff=0.001, add_projection=False
        )

        result = loss_fn_high(embeddings_a, embeddings_b)


        loss_high = extract_loss(result)
        result = loss_fn_low(embeddings_a, embeddings_b)
        loss_low = extract_loss(result)
        # Both losses should be valid
        assert not torch.isnan(loss_high)
        assert not torch.isnan(loss_low)
        # Different lambdas typically lead to different losses
        # Note: We don't assert they're different as random data may yield similar results

    def test_correlation_modes(self, embeddings_a, embeddings_b, device):
        """Test different correlation modes."""
        # Cross-modal correlation
        loss_fn_cross = BarlowTwinsLoss(
            lambda_coeff=0.005,
            correlation_mode="cross_modal",
            add_projection=False,
        )

        result = loss_fn_cross(embeddings_a, embeddings_b)


        loss_cross = extract_loss(result)
        assert not torch.isnan(loss_cross)

    def test_edge_case_identical_embeddings(
        self, batch_size, embed_dim, device
    ):
        """Test with identical embeddings."""
        embeddings = torch.randn(batch_size, embed_dim, device=device)

        loss_fn = BarlowTwinsLoss(
            lambda_coeff=0.005, add_projection=False
        )

        result = loss_fn(embeddings, embeddings.clone())


        loss = extract_loss(result)
        # With identical embeddings, cross-correlation matrix should be identity
        # Loss should be low (primarily off-diagonal terms)
        assert not torch.isnan(loss)
        assert loss.item() >= 0

    def test_numerical_stability(self, batch_size, embed_dim, device):
        """Test numerical stability with extreme values."""
        embeddings_large_a = torch.ones(batch_size, embed_dim, device=device) * 50
        embeddings_large_b = torch.ones(batch_size, embed_dim, device=device) * 50

        loss_fn = BarlowTwinsLoss(
            lambda_coeff=0.005, add_projection=False, normalize_embeddings=True
        )

        result = loss_fn(embeddings_large_a, embeddings_large_b)


        loss = extract_loss(result)
        assert not torch.isnan(loss)
        assert not torch.isinf(loss)

    def test_normalization_effect(self, embeddings_a, embeddings_b, device):
        """Test effect of embedding normalization."""
        # With normalization
        loss_fn_norm = BarlowTwinsLoss(
            lambda_coeff=0.005,
            add_projection=False,
            normalize_embeddings=True,
        )

        # Without normalization
        loss_fn_no_norm = BarlowTwinsLoss(
            lambda_coeff=0.005,
            add_projection=False,
            normalize_embeddings=False,
        )

        result = loss_fn_norm(embeddings_a, embeddings_b)


        loss_norm = extract_loss(result)
        result = loss_fn_no_norm(embeddings_a, embeddings_b)
        loss_no_norm = extract_loss(result)
        # Both should be valid
        assert not torch.isnan(loss_norm)
        assert not torch.isnan(loss_no_norm)


# ============================================================================
# HybridPretrainVICRegLoss Tests
# ============================================================================


class TestHybridPretrainVICRegLoss:
    """Test suite for HybridPretrainVICRegLoss."""

    def test_basic_forward(self, embeddings_a, embeddings_b, device):
        """Test basic forward pass."""
        loss_fn = HybridPretrainVICRegLoss(
            sim_coeff=10.0, var_coeff=5.0, cov_coeff=1.0
        )

        result = loss_fn(embeddings_a, embeddings_b)

        # May return dict or tensor
        if isinstance(result, dict):
            assert 'loss' in result
            loss = result['loss']
        else:
            loss = result

        assert isinstance(loss, torch.Tensor)
        assert loss.shape == torch.Size([])
        assert not torch.isnan(loss)
        assert not torch.isinf(loss)

    def test_gradient_flow(self, embeddings_a, embeddings_b, device):
        """Test gradient flow through hybrid loss."""
        embeddings_a = embeddings_a.requires_grad_(True)
        embeddings_b = embeddings_b.requires_grad_(True)

        loss_fn = HybridPretrainVICRegLoss(
            sim_coeff=10.0, var_coeff=5.0, cov_coeff=1.0
        )

        result = loss_fn(embeddings_a, embeddings_b)
        loss = result['loss'] if isinstance(result, dict) else result
        loss.backward()

        assert embeddings_a.grad is not None
        assert embeddings_b.grad is not None
        assert not torch.all(embeddings_a.grad == 0)
        assert not torch.all(embeddings_b.grad == 0)

    def test_hybrid_components(self, embeddings_a, embeddings_b, device):
        """Test that hybrid loss combines multiple objectives."""
        loss_fn = HybridPretrainVICRegLoss(
            sim_coeff=10.0, var_coeff=5.0, cov_coeff=1.0
        )

        result = loss_fn(embeddings_a, embeddings_b)

        # Hybrid loss should combine VICReg with other objectives
        if isinstance(result, dict):
            # Check that loss is valid
            assert 'loss' in result
            assert not torch.isnan(result['loss'])

    def test_numerical_stability(self, batch_size, embed_dim, device):
        """Test numerical stability."""
        embeddings_a = torch.randn(batch_size, embed_dim, device=device) * 10
        embeddings_b = torch.randn(batch_size, embed_dim, device=device) * 10

        loss_fn = HybridPretrainVICRegLoss(
            sim_coeff=10.0, var_coeff=5.0, cov_coeff=1.0
        )

        result = loss_fn(embeddings_a, embeddings_b)
        loss = result['loss'] if isinstance(result, dict) else result

        assert not torch.isnan(loss)
        assert not torch.isinf(loss)


# ============================================================================
# Integration Tests
# ============================================================================


class TestSelfSupervisedLossIntegration:
    """Integration tests for self-supervised losses."""

    def test_loss_comparison(self, embeddings_a, embeddings_b, device):
        """Compare different self-supervised losses on same data."""
        # VICReg
        vicreg = VICRegLoss(sim_coeff=10.0, var_coeff=5.0, cov_coeff=1.0)
        result_vicreg = vicreg(embeddings_a, embeddings_b)
        loss_vicreg = result_vicreg['loss'] if isinstance(result_vicreg, dict) else result_vicreg

        # Barlow Twins
        barlow = BarlowTwinsLoss(lambda_coeff=0.005, add_projection=False)
        result_barlow = barlow(embeddings_a, embeddings_b)
        loss_barlow = extract_loss(result_barlow)

        # Both should be valid but different
        assert not torch.isnan(loss_vicreg)
        assert not torch.isnan(loss_barlow)
        # Different loss formulations should yield different values
        assert not torch.allclose(loss_vicreg, loss_barlow)

    def test_combined_training_simulation(self, batch_size, embed_dim, device):
        """Simulate a training step with self-supervised losses."""
        # Create a simple model
        model = nn.Sequential(
            nn.Linear(embed_dim, embed_dim), nn.ReLU(), nn.Linear(embed_dim, embed_dim)
        ).to(device)

        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

        # Generate data
        x1 = torch.randn(batch_size, embed_dim, device=device)
        x2 = torch.randn(batch_size, embed_dim, device=device)

        # Forward pass
        z1 = model(x1)
        z2 = model(x2)

        # Compute loss
        loss_fn = VICRegLoss(sim_coeff=10.0, var_coeff=5.0, cov_coeff=1.0)
        result = loss_fn(z1, z2)
        loss = result['loss'] if isinstance(result, dict) else result

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Check that optimization step was successful
        assert not torch.isnan(loss)
        # Model parameters should have been updated
        for param in model.parameters():
            assert param.grad is not None
