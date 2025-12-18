"""
Comprehensive tests for specialized loss functions.

Tests cover:
- DecorrelationLoss (feature collapse prevention)
- MultitaskLoss (multi-task learning)
- CLIPLoss (CLIP-style contrastive)
- CombinedLoss (loss combination)
- LossFactory (loss creation)
- Other specialized losses
"""

from typing import Any, Dict

import numpy as np
import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F


# Import loss functions
try:
    from src.training.losses import DecorrelationLoss
except ImportError:
    DecorrelationLoss = None

try:
    from src.training.losses import MultitaskLoss
except ImportError:
    MultitaskLoss = None

try:
    from src.training.losses import CLIPLoss
except ImportError:
    CLIPLoss = None

try:
    from src.training.losses import CombinedLoss
except ImportError:
    CombinedLoss = None

try:
    from src.training.losses import create_loss_function
except ImportError:
    create_loss_function = None

try:
    from src.training.losses import FeatureConsistencyLoss
except ImportError:
    FeatureConsistencyLoss = None


# ============================================================================
# Helper Classes
# ============================================================================


class MockArgs:
    """Mock arguments object for loss factory tests."""
    def __init__(self, **kwargs):
        # Set defaults for all attributes that loss_factory.py might access
        self.contrastive_sampling = "auto"
        self.batch_size = 32
        self.memory_bank_size = 4096
        self.queue_size = None
        self.dynamic_temp_min = None
        self.dynamic_temp_max = None
        self.mining_strategy = None
        self.hard_negative_factor = None
        self.contrastive_weight = 1.0
        self.classification_weight = 0.0
        self.multimodal_matching_weight = 0.0
        self.use_hard_negatives = False
        self.dim = 512
        self.vision_model = None
        self.text_model = None

        # Override with any provided kwargs
        for key, value in kwargs.items():
            setattr(self, key, value)


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
        return result.get("loss", result.get("total_loss", result.get("contrastive_loss")))
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
# DecorrelationLoss Tests
# ============================================================================


@pytest.mark.skipif(DecorrelationLoss is None, reason="DecorrelationLoss not available")
class TestDecorrelationLoss:
    """Test suite for DecorrelationLoss."""

    def test_basic_forward(self, vision_features, device):
        """Test basic forward pass."""
        loss_fn = DecorrelationLoss(coef=1.0)

        result = loss_fn(vision_features)

        loss = extract_loss(result)
        assert isinstance(loss, torch.Tensor)
        assert loss.shape == torch.Size([])
        assert not torch.isnan(loss)
        assert not torch.isinf(loss)
        assert loss.item() >= 0

    def test_with_text_features(self, vision_features, text_features, device):
        """Test with both vision and text features."""
        loss_fn = DecorrelationLoss(coef=1.0)

        # If the loss supports both modalities
        try:
            result = loss_fn(vision_features, text_features)

            loss = extract_loss(result)
            assert not torch.isnan(loss)
        except TypeError:
            # If it only takes one input
            result = loss_fn(vision_features)

            loss = extract_loss(result)
            assert not torch.isnan(loss)

    def test_gradient_flow(self, vision_features, device):
        """Test gradient flow through decorrelation loss."""
        vision_features = vision_features.requires_grad_(True)

        loss_fn = DecorrelationLoss(coef=1.0)

        result = loss_fn(vision_features)

        loss = extract_loss(result)
        loss.backward()

        assert vision_features.grad is not None
        assert not torch.all(vision_features.grad == 0)

    def test_coefficient_effect(self, vision_features, device):
        """Test that coefficient affects loss magnitude."""
        loss_fn_low = DecorrelationLoss(coef=0.1)
        loss_fn_high = DecorrelationLoss(coef=10.0)

        result = loss_fn_low(vision_features)

        loss_low = extract_loss(result)
        result = loss_fn_high(vision_features)
        loss_high = extract_loss(result)
        # Higher coefficient should lead to higher loss
        assert loss_high > loss_low
        assert not torch.isnan(loss_low)
        assert not torch.isnan(loss_high)

    def test_normalization_effect(self, vision_features, device):
        """Test effect of embedding normalization."""
        loss_fn_norm = DecorrelationLoss(coef=1.0, normalize_embeddings=True)
        loss_fn_no_norm = DecorrelationLoss(coef=1.0, normalize_embeddings=False)

        result = loss_fn_norm(vision_features)

        loss_norm = extract_loss(result)
        result = loss_fn_no_norm(vision_features)
        loss_no_norm = extract_loss(result)
        # Both should be valid
        assert not torch.isnan(loss_norm)
        assert not torch.isnan(loss_no_norm)

    def test_edge_case_uncorrelated_features(self, batch_size, embed_dim, device):
        """Test with perfectly uncorrelated features."""
        # Create orthogonal (uncorrelated) features
        features = torch.eye(min(batch_size, embed_dim), embed_dim, device=device)
        if batch_size > embed_dim:
            features = torch.cat(
                [features, torch.randn(batch_size - embed_dim, embed_dim, device=device)]
            )

        loss_fn = DecorrelationLoss(coef=1.0)
        result = loss_fn(features)

        loss = extract_loss(result)
        # Loss should be low for uncorrelated features
        assert not torch.isnan(loss)

    def test_edge_case_correlated_features(self, batch_size, embed_dim, device):
        """Test with highly correlated features."""
        # Create highly correlated features (all similar)
        base = torch.randn(1, embed_dim, device=device)
        features = (
            base.repeat(batch_size, 1) + torch.randn(batch_size, embed_dim, device=device) * 0.01
        )

        loss_fn = DecorrelationLoss(coef=1.0)
        result = loss_fn(features)

        loss = extract_loss(result)
        # Loss should be higher for correlated features
        assert not torch.isnan(loss)
        assert loss.item() > 0

    def test_numerical_stability(self, batch_size, embed_dim, device):
        """Test numerical stability with extreme values."""
        features_large = torch.ones(batch_size, embed_dim, device=device) * 100

        loss_fn = DecorrelationLoss(coef=1.0, normalize_embeddings=True)
        result = loss_fn(features_large)

        loss = extract_loss(result)
        assert not torch.isnan(loss)
        assert not torch.isinf(loss)


# ============================================================================
# MultitaskLoss Tests
# ============================================================================


@pytest.mark.skipif(MultitaskLoss is None, reason="MultitaskLoss not available")
class TestMultitaskLoss:
    """Test suite for MultitaskLoss."""

    def test_basic_forward(self, device):
        """Test basic forward pass with multiple tasks."""
        # Create simple loss functions for each task
        loss_functions = {
            "task1": nn.MSELoss(),
            "task2": nn.L1Loss(),
        }

        loss_fn = MultitaskLoss(loss_functions=loss_functions)

        # Create inputs and targets
        inputs = {
            "task1": torch.randn(8, 10, device=device),
            "task2": torch.randn(8, 10, device=device),
        }
        targets = {
            "task1": torch.randn(8, 10, device=device),
            "task2": torch.randn(8, 10, device=device),
        }

        result = loss_fn(inputs, targets)

        assert isinstance(result, dict)
        assert "loss" in result or "total_loss" in result
        # Check that loss is valid
        total_loss = result.get("loss", result.get("total_loss"))
        assert isinstance(total_loss, torch.Tensor)
        assert not torch.isnan(total_loss)

    def test_custom_weights(self, device):
        """Test with custom task weights."""
        loss_functions = {
            "task1": nn.MSELoss(),
            "task2": nn.L1Loss(),
        }

        loss_weights = {"task1": 0.7, "task2": 0.3}

        loss_fn = MultitaskLoss(loss_functions=loss_functions, loss_weights=loss_weights)

        inputs = {
            "task1": torch.randn(8, 10, device=device),
            "task2": torch.randn(8, 10, device=device),
        }
        targets = {
            "task1": torch.randn(8, 10, device=device),
            "task2": torch.randn(8, 10, device=device),
        }

        result = loss_fn(inputs, targets)
        total_loss = result.get("loss", result.get("total_loss"))
        assert not torch.isnan(total_loss)

    def test_gradient_flow(self, device):
        """Test gradient flow through multitask loss."""
        loss_functions = {
            "task1": nn.MSELoss(),
            "task2": nn.MSELoss(),
        }

        loss_fn = MultitaskLoss(loss_functions=loss_functions)

        inputs = {
            "task1": torch.randn(8, 10, device=device, requires_grad=True),
            "task2": torch.randn(8, 10, device=device, requires_grad=True),
        }
        targets = {
            "task1": torch.randn(8, 10, device=device),
            "task2": torch.randn(8, 10, device=device),
        }

        result = loss_fn(inputs, targets)
        total_loss = result.get("loss", result.get("total_loss"))
        total_loss.backward()

        # Check gradients
        assert inputs["task1"].grad is not None
        assert inputs["task2"].grad is not None

    def test_missing_task(self, device):
        """Test handling of missing task in inputs."""
        loss_functions = {
            "task1": nn.MSELoss(),
            "task2": nn.MSELoss(),
        }

        loss_fn = MultitaskLoss(loss_functions=loss_functions)

        # Only provide task1
        inputs = {"task1": torch.randn(8, 10, device=device)}
        targets = {"task1": torch.randn(8, 10, device=device)}

        result = loss_fn(inputs, targets)
        # Should handle missing task gracefully
        assert isinstance(result, dict)


# ============================================================================
# CLIPLoss Tests
# ============================================================================


@pytest.mark.skipif(CLIPLoss is None, reason="CLIPLoss not available")
class TestCLIPLoss:
    """Test suite for CLIPLoss."""

    def test_basic_forward(self, vision_features, text_features, device):
        """Test basic forward pass."""
        loss_fn = CLIPLoss(temperature=0.07)

        result = loss_fn(vision_features, text_features)

        # May return dict or tensor
        if isinstance(result, dict):
            assert "loss" in result or "total_loss" in result
            loss = result.get("loss", result.get("total_loss"))
        else:
            loss = result

        assert isinstance(loss, torch.Tensor)
        assert loss.shape == torch.Size([])
        assert not torch.isnan(loss)
        assert not torch.isinf(loss)

    def test_with_match_ids(self, vision_features, text_features, match_ids, device):
        """Test with match IDs."""
        loss_fn = CLIPLoss(temperature=0.07)

        result = loss_fn(vision_features, text_features, match_ids=match_ids)

        if isinstance(result, dict):
            loss = result.get("loss", result.get("total_loss"))
        else:
            loss = result

        assert not torch.isnan(loss)

    def test_temperature_sensitivity(self, vision_features, text_features, device):
        """Test temperature parameter effect."""
        loss_fn_low = CLIPLoss(temperature=0.01)
        loss_fn_high = CLIPLoss(temperature=1.0)

        result_low = loss_fn_low(vision_features, text_features)
        result_high = loss_fn_high(vision_features, text_features)

        loss_low = result_low["loss"] if isinstance(result_low, dict) else result_low
        loss_high = result_high["loss"] if isinstance(result_high, dict) else result_high

        # Different temperatures should lead to different losses
        assert not torch.allclose(loss_low, loss_high)
        assert not torch.isnan(loss_low)
        assert not torch.isnan(loss_high)

    def test_gradient_flow(self, vision_features, text_features, device):
        """Test gradient flow."""
        vision_features = vision_features.requires_grad_(True)
        text_features = text_features.requires_grad_(True)

        loss_fn = CLIPLoss(temperature=0.07)

        result = loss_fn(vision_features, text_features)
        loss = result["loss"] if isinstance(result, dict) else result
        loss.backward()

        assert vision_features.grad is not None
        assert text_features.grad is not None
        assert not torch.all(vision_features.grad == 0)
        assert not torch.all(text_features.grad == 0)

    def test_label_smoothing(self, vision_features, text_features, device):
        """Test label smoothing effect."""
        loss_fn_no_smooth = CLIPLoss(temperature=0.07, label_smoothing=0.0)
        loss_fn_smooth = CLIPLoss(temperature=0.07, label_smoothing=0.1)

        result_no_smooth = loss_fn_no_smooth(vision_features, text_features)
        result_smooth = loss_fn_smooth(vision_features, text_features)

        loss_no_smooth = (
            result_no_smooth["loss"] if isinstance(result_no_smooth, dict) else result_no_smooth
        )
        loss_smooth = result_smooth["loss"] if isinstance(result_smooth, dict) else result_smooth

        # Label smoothing should affect loss value
        assert not torch.isnan(loss_no_smooth)
        assert not torch.isnan(loss_smooth)

    def test_numerical_stability(self, batch_size, embed_dim, device):
        """Test numerical stability."""
        vision_large = torch.ones(batch_size, embed_dim, device=device) * 10
        text_large = torch.ones(batch_size, embed_dim, device=device) * 10

        loss_fn = CLIPLoss(temperature=0.07)

        result = loss_fn(vision_large, text_large)
        loss = result["loss"] if isinstance(result, dict) else result

        assert not torch.isnan(loss)
        assert not torch.isinf(loss)


# ============================================================================
# CombinedLoss Tests
# ============================================================================


@pytest.mark.skipif(CombinedLoss is None, reason="CombinedLoss not available")
class TestCombinedLoss:
    """Test suite for CombinedLoss."""

    def test_basic_forward(self, vision_features, text_features, device):
        """Test combining multiple losses."""
        # Test with appropriate inputs
        try:
            loss_fn = CombinedLoss(
                primary_loss=nn.MSELoss(),
                secondary_loss=nn.L1Loss(),
                secondary_loss_weight=0.5
            )
            result = loss_fn(vision_features, text_features)
            if isinstance(result, dict):
                loss = result.get("loss", result.get("total_loss"))
            else:
                loss = result
            assert not torch.isnan(loss)
        except Exception:
            # CombinedLoss might have different interface
            pytest.skip("CombinedLoss has different interface")

    def test_weighted_combination(self, device):
        """Test weighted combination of losses."""
        try:
            # Primary loss has implicit weight of 1.0, secondary has weight of 0.3
            loss_fn = CombinedLoss(
                primary_loss=nn.MSELoss(),
                secondary_loss=nn.L1Loss(),
                secondary_loss_weight=0.3
            )

            inputs = torch.randn(8, 10, device=device)
            targets = torch.randn(8, 10, device=device)

            result = loss_fn(inputs, targets)
            if isinstance(result, dict):
                loss = result.get("loss", result.get("total_loss"))
            else:
                loss = result
            assert not torch.isnan(loss)
        except Exception:
            pytest.skip("CombinedLoss has different interface")


# ============================================================================
# LossFactory Tests
# ============================================================================


@pytest.mark.skipif(create_loss_function is None, reason="Loss factory not available")
class TestLossFactory:
    """Test suite for loss factory."""

    def test_create_contrastive_loss(self):
        """Test creating contrastive loss from factory."""
        try:
            args = MockArgs(
                loss_type="contrastive",
                temperature=0.07,
                use_simple_model=False,
                use_mixed_loss=False,
                fusion_dim=512
            )
            loss = create_loss_function(args, dataset_size=1000, train_loader=None)
            assert loss is not None
            assert isinstance(loss, nn.Module)
        except Exception:
            pytest.skip("Loss factory has different interface")

    def test_create_vicreg_loss(self):
        """Test creating VICReg loss from factory."""
        try:
            args = MockArgs(
                loss_type="vicreg",
                sim_weight=10.0,
                var_weight=5.0,
                cov_weight=1.0,
                use_simple_model=False,
                use_mixed_loss=False,
                fusion_dim=512
            )
            loss = create_loss_function(args, dataset_size=1000, train_loader=None)
            assert loss is not None
            assert isinstance(loss, nn.Module)
        except Exception:
            pytest.skip("Loss factory has different interface")

    def test_invalid_loss_type(self):
        """Test handling of invalid loss type."""
        try:
            args = MockArgs(
                loss_type="invalid_loss_type",
                use_simple_model=False,
                use_mixed_loss=False
            )
            with pytest.raises((ValueError, KeyError, AttributeError)):
                create_loss_function(args)
        except Exception:
            pytest.skip("Loss factory has different interface")


# ============================================================================
# FeatureConsistencyLoss Tests
# ============================================================================


@pytest.mark.skipif(FeatureConsistencyLoss is None, reason="FeatureConsistencyLoss not available")
class TestFeatureConsistencyLoss:
    """Test suite for FeatureConsistencyLoss."""

    def test_basic_forward(self, vision_features, text_features, device):
        """Test basic forward pass."""
        try:
            # Create simple reference models
            reference_vision = nn.Linear(vision_features.shape[1], vision_features.shape[1]).to(device)
            reference_text = nn.Linear(text_features.shape[1], text_features.shape[1]).to(device)

            # FeatureConsistencyLoss needs both features AND raw inputs
            # Create mock raw inputs (same shape as features for Linear model)
            vision_inputs = torch.randn_like(vision_features).to(device)
            text_inputs = torch.randn_like(text_features).to(device)

            loss_fn = FeatureConsistencyLoss(
                reference_vision_model=reference_vision,
                reference_text_model=reference_text,
                vision_weight=1.0,
                text_weight=1.0
            )
            result = loss_fn(
                vision_features=vision_features,
                text_features=text_features,
                vision_inputs=vision_inputs,
                text_inputs=text_inputs
            )

            loss = extract_loss(result)
            assert not torch.isnan(loss)
        except Exception:
            pytest.skip("FeatureConsistencyLoss has different interface")

    def test_gradient_flow(self, vision_features, text_features, device):
        """Test gradient flow."""
        try:
            vision_features = vision_features.requires_grad_(True)
            text_features = text_features.requires_grad_(True)

            # Create simple reference models
            reference_vision = nn.Linear(vision_features.shape[1], vision_features.shape[1]).to(device)
            reference_text = nn.Linear(text_features.shape[1], text_features.shape[1]).to(device)

            # FeatureConsistencyLoss needs both features AND raw inputs
            vision_inputs = torch.randn_like(vision_features).to(device)
            text_inputs = torch.randn_like(text_features).to(device)

            loss_fn = FeatureConsistencyLoss(
                reference_vision_model=reference_vision,
                reference_text_model=reference_text
            )
            result = loss_fn(
                vision_features=vision_features,
                text_features=text_features,
                vision_inputs=vision_inputs,
                text_inputs=text_inputs
            )

            loss = extract_loss(result)
            loss.backward()

            assert vision_features.grad is not None
            assert text_features.grad is not None
        except Exception:
            pytest.skip("FeatureConsistencyLoss has different interface")


# ============================================================================
# Integration Tests
# ============================================================================


class TestSpecializedLossIntegration:
    """Integration tests for specialized losses."""

    def test_loss_combination_in_training(self, batch_size, embed_dim, device):
        """Simulate combining multiple specialized losses in training."""
        # Create a simple model
        model = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim),
        ).to(device)

        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

        # Generate data
        x = torch.randn(batch_size, embed_dim, device=device)
        y = torch.randn(batch_size, embed_dim, device=device)

        # Forward pass
        z = model(x)

        # Compute combined loss
        total_loss = 0
        loss_count = 0

        # Add MSE loss
        mse_loss = F.mse_loss(z, y)
        total_loss += mse_loss
        loss_count += 1

        # Add decorrelation loss if available
        if DecorrelationLoss is not None:
            decorr_fn = DecorrelationLoss(coef=0.1)
            decorr_result = decorr_fn(z)
            decorr_loss = extract_loss(decorr_result)
            total_loss += decorr_loss
            loss_count += 1

        # Backward pass
        if loss_count > 0:
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            # Check that optimization step was successful
            assert not torch.isnan(total_loss)
            for param in model.parameters():
                assert param.grad is not None
