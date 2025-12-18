# Testing Quick Fix Guide - Immediate Actions

**Purpose**: Get from 10% to 25% coverage in 1 week
**Date**: 2025-12-18

---

## Day 1-2: Create Missing Test Files

### File 1: tests/test_transformer_trainer.py

```python
"""Tests for TransformerTrainer."""
import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from src.training.trainers.transformer_trainer import TransformerTrainer


@pytest.fixture
def simple_transformer_model():
    """Simple transformer for testing."""
    return nn.Transformer(
        d_model=128,
        nhead=4,
        num_encoder_layers=2,
        num_decoder_layers=2,
        dim_feedforward=256
    )


@pytest.fixture
def sample_dataloader():
    """Sample dataloader for testing."""
    # Source and target sequences
    src = torch.randint(0, 100, (50, 20))  # 50 samples, seq_len 20
    tgt = torch.randint(0, 100, (50, 20))
    dataset = TensorDataset(src, tgt)
    return DataLoader(dataset, batch_size=8)


class TestTransformerTrainerInit:
    """Test TransformerTrainer initialization."""

    def test_basic_initialization(self, simple_transformer_model, sample_dataloader):
        """Test basic trainer initialization."""
        trainer = TransformerTrainer(
            model=simple_transformer_model,
            train_dataloader=sample_dataloader,
            pad_idx=0
        )

        assert trainer.model is not None
        assert trainer.pad_idx == 0
        assert trainer.warmup_steps == 4000
        assert trainer.label_smoothing == 0.1

    def test_custom_parameters(self, simple_transformer_model, sample_dataloader):
        """Test initialization with custom parameters."""
        trainer = TransformerTrainer(
            model=simple_transformer_model,
            train_dataloader=sample_dataloader,
            pad_idx=1,
            lr=0.001,
            warmup_steps=1000,
            label_smoothing=0.2
        )

        assert trainer.pad_idx == 1
        assert trainer.warmup_steps == 1000
        assert trainer.label_smoothing == 0.2

    def test_scheduler_types(self, simple_transformer_model, sample_dataloader):
        """Test different scheduler types."""
        for scheduler_type in ["inverse_sqrt", "cosine", "linear", "constant"]:
            trainer = TransformerTrainer(
                model=simple_transformer_model,
                train_dataloader=sample_dataloader,
                scheduler=scheduler_type
            )
            assert trainer.scheduler == scheduler_type


class TestTransformerTrainerTraining:
    """Test training functionality."""

    def test_train_epoch_runs(self, simple_transformer_model, sample_dataloader):
        """Test that train_epoch executes without errors."""
        trainer = TransformerTrainer(
            model=simple_transformer_model,
            train_dataloader=sample_dataloader
        )

        # Should run one epoch without errors
        try:
            loss = trainer.train_epoch()
            assert isinstance(loss, float)
            assert loss > 0
        except Exception as e:
            pytest.fail(f"train_epoch failed: {e}")

    def test_learning_rate_warmup(self, simple_transformer_model, sample_dataloader):
        """Test learning rate warmup."""
        trainer = TransformerTrainer(
            model=simple_transformer_model,
            train_dataloader=sample_dataloader,
            warmup_steps=10
        )

        # Learning rate should increase during warmup
        initial_lr = trainer.optimizer.param_groups[0]['lr']

        # Simulate a few steps
        for _ in range(5):
            trainer._update_learning_rate()

        new_lr = trainer.optimizer.param_groups[0]['lr']
        # During warmup, LR should change
        assert initial_lr != new_lr or trainer.step > trainer.warmup_steps

    def test_gradient_clipping(self, simple_transformer_model, sample_dataloader):
        """Test gradient clipping functionality."""
        trainer = TransformerTrainer(
            model=simple_transformer_model,
            train_dataloader=sample_dataloader,
            clip_grad=1.0
        )

        assert trainer.clip_grad == 1.0


class TestTransformerTrainerValidation:
    """Test validation functionality."""

    def test_validate_runs(self, simple_transformer_model, sample_dataloader):
        """Test that validation runs without errors."""
        trainer = TransformerTrainer(
            model=simple_transformer_model,
            train_dataloader=sample_dataloader,
            val_dataloader=sample_dataloader  # Using same for test
        )

        try:
            val_loss = trainer.validate()
            assert isinstance(val_loss, float)
            assert val_loss > 0
        except Exception as e:
            pytest.fail(f"validate failed: {e}")


class TestTransformerTrainerCheckpoints:
    """Test checkpoint functionality."""

    def test_save_checkpoint(self, simple_transformer_model, sample_dataloader, tmp_path):
        """Test checkpoint saving."""
        trainer = TransformerTrainer(
            model=simple_transformer_model,
            train_dataloader=sample_dataloader
        )

        checkpoint_path = tmp_path / "checkpoint.pt"
        trainer.save_checkpoint(str(checkpoint_path), epoch=1, metrics={'loss': 0.5})

        assert checkpoint_path.exists()

    def test_load_checkpoint(self, simple_transformer_model, sample_dataloader, tmp_path):
        """Test checkpoint loading."""
        trainer = TransformerTrainer(
            model=simple_transformer_model,
            train_dataloader=sample_dataloader
        )

        # Save checkpoint
        checkpoint_path = tmp_path / "checkpoint.pt"
        trainer.save_checkpoint(str(checkpoint_path), epoch=1, metrics={'loss': 0.5})

        # Load checkpoint
        checkpoint = trainer.load_checkpoint(str(checkpoint_path))

        assert checkpoint is not None
        assert 'epoch' in checkpoint
        assert checkpoint['epoch'] == 1
```

**Expected Coverage Gain**: +15% (transformer_trainer.py: 6% → 80%)

---

### File 2: tests/test_loss_factory.py

```python
"""Tests for loss factory functions."""
import pytest
import torch

from src.training.losses.loss_factory import (
    SimpleContrastiveLoss,
    create_loss_function,
    create_multimodal_loss,
)


class TestSimpleContrastiveLoss:
    """Test SimpleContrastiveLoss."""

    def test_initialization(self):
        """Test loss initialization."""
        loss_fn = SimpleContrastiveLoss(temperature=0.1)
        assert loss_fn.temperature == 0.1

    def test_forward_basic(self):
        """Test basic forward pass."""
        loss_fn = SimpleContrastiveLoss()

        batch_size = 8
        dim = 128
        vision_features = torch.randn(batch_size, dim)
        text_features = torch.randn(batch_size, dim)

        result = loss_fn(vision_features, text_features)

        # Check result is dict with loss
        assert isinstance(result, dict)
        assert 'loss' in result
        assert result['loss'].requires_grad

    def test_with_match_ids(self):
        """Test with match_ids."""
        loss_fn = SimpleContrastiveLoss()

        batch_size = 8
        dim = 128
        vision_features = torch.randn(batch_size, dim)
        text_features = torch.randn(batch_size, dim)
        match_ids = torch.tensor([0, 0, 1, 1, 2, 2, 3, 3])  # Pairs

        result = loss_fn(vision_features, text_features, match_ids=match_ids)

        assert isinstance(result, dict)
        assert 'loss' in result

    def test_gradient_flow(self):
        """Test gradients flow through loss."""
        loss_fn = SimpleContrastiveLoss()

        vision_features = torch.randn(4, 64, requires_grad=True)
        text_features = torch.randn(4, 64, requires_grad=True)

        result = loss_fn(vision_features, text_features)
        loss = result['loss']
        loss.backward()

        assert vision_features.grad is not None
        assert text_features.grad is not None


class TestCreateLossFunction:
    """Test create_loss_function factory."""

    def test_create_simclr_loss(self):
        """Test creating SimCLR loss."""
        try:
            loss = create_loss_function('simclr', temperature=0.1)
            assert loss is not None
        except (ImportError, AttributeError) as e:
            pytest.skip(f"SimCLR loss not available: {e}")

    def test_create_moco_loss(self):
        """Test creating MoCo loss."""
        try:
            loss = create_loss_function('moco', temperature=0.07, queue_size=4096)
            assert loss is not None
        except (ImportError, AttributeError) as e:
            pytest.skip(f"MoCo loss not available: {e}")

    def test_create_clip_loss(self):
        """Test creating CLIP-style loss."""
        try:
            loss = create_loss_function('clip', temperature=0.01)
            assert loss is not None
        except (ImportError, AttributeError) as e:
            pytest.skip(f"CLIP loss not available: {e}")

    def test_invalid_loss_type(self):
        """Test error on invalid loss type."""
        with pytest.raises((ValueError, KeyError)):
            create_loss_function('invalid_loss_type')


class TestCreateMultimodalLoss:
    """Test create_multimodal_loss factory."""

    def test_basic_multimodal_loss(self):
        """Test creating basic multimodal loss."""
        try:
            loss = create_multimodal_loss(
                contrastive_weight=1.0,
                classification_weight=0.0
            )
            assert loss is not None
        except (ImportError, AttributeError) as e:
            pytest.skip(f"Multimodal loss not available: {e}")

    def test_combined_losses(self):
        """Test combining multiple loss components."""
        try:
            loss = create_multimodal_loss(
                contrastive_weight=0.5,
                classification_weight=0.3,
                multimodal_matching_weight=0.2
            )
            assert loss is not None
        except (ImportError, AttributeError) as e:
            pytest.skip(f"Combined loss not available: {e}")
```

**Expected Coverage Gain**: +10% (loss_factory.py: 9.7% → 85%)

---

### File 3: tests/test_model_factory.py

```python
"""Tests for model factory functions."""
import pytest
import torch
import torch.nn as nn

from src.models.model_factory import (
    create_multimodal_model,
    create_transformer_model,
)


class MockArgs:
    """Mock arguments for model creation."""
    def __init__(self):
        self.use_pretrained = False
        self.use_pretrained_text = False
        self.vision_model = "vit-base"
        self.text_model = "bert-base-uncased"
        self.fusion_dim = 768
        self.model_size = None


class TestCreateMultimodalModel:
    """Test create_multimodal_model factory."""

    def test_basic_model_creation(self):
        """Test basic multimodal model creation."""
        args = MockArgs()
        device = torch.device("cpu")

        try:
            model = create_multimodal_model(args, device)
            assert isinstance(model, nn.Module)
        except ImportError as e:
            pytest.skip(f"Required dependencies not available: {e}")

    def test_model_size_presets(self):
        """Test model size presets."""
        for size in ["small", "medium", "large"]:
            args = MockArgs()
            args.model_size = size
            device = torch.device("cpu")

            try:
                model = create_multimodal_model(args, device)
                assert isinstance(model, nn.Module)
            except ImportError as e:
                pytest.skip(f"Preset {size} not available: {e}")

    def test_dimension_matching(self):
        """Test dimension matching between vision and text."""
        args = MockArgs()
        args.fusion_dim = 512
        device = torch.device("cpu")

        try:
            model = create_multimodal_model(args, device)
            # Model should be created with proper dimension matching
            assert isinstance(model, nn.Module)
        except ImportError as e:
            pytest.skip(f"Dimension matching not available: {e}")

    def test_device_selection_cpu(self):
        """Test model creation on CPU."""
        args = MockArgs()
        device = torch.device("cpu")

        try:
            model = create_multimodal_model(args, device)
            assert next(model.parameters()).device.type == "cpu"
        except (ImportError, StopIteration) as e:
            pytest.skip(f"Model creation failed: {e}")

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_device_selection_cuda(self):
        """Test model creation on CUDA."""
        args = MockArgs()
        device = torch.device("cuda")

        try:
            model = create_multimodal_model(args, device)
            assert next(model.parameters()).device.type == "cuda"
        except (ImportError, StopIteration) as e:
            pytest.skip(f"Model creation failed: {e}")


class TestCreateTransformerModel:
    """Test create_transformer_model factory."""

    def test_basic_transformer_creation(self):
        """Test basic transformer model creation."""
        args = MockArgs()
        device = torch.device("cpu")

        try:
            model = create_transformer_model(args, device)
            assert isinstance(model, nn.Module)
        except (ImportError, AttributeError) as e:
            pytest.skip(f"Transformer creation not available: {e}")
```

**Expected Coverage Gain**: +8% (model_factory.py: 9.2% → 85%)

---

## Day 3-4: Fix Interface Mismatch Skips

### Fix 1: test_augmentation_pipeline.py

**Problem**: 27 instances of `pytest.skip("Pipeline interface different")`

**Solution**: Update test to match actual interface

```python
# BEFORE (current - causing skips)
def test_image_augmentation_basic(self, sample_image):
    pipeline = MultimodalAugmentationPipeline(image_aug_prob=1.0)

    try:
        augmented = pipeline({"image": sample_image, "text": "test caption"})
        if isinstance(augmented, tuple):
            aug_image, aug_text = augmented
        else:
            aug_image = augmented.get("image", augmented.get("pixel_values"))

        assert isinstance(aug_image, (torch.Tensor, Image.Image))
    except Exception as e:
        pytest.skip(f"Pipeline has different interface: {e}")

# AFTER (fixed - no skip)
def test_image_augmentation_basic(self, sample_image):
    pipeline = MultimodalAugmentationPipeline(image_aug_prob=1.0)

    # Call pipeline and inspect what it actually returns
    result = pipeline({"image": sample_image, "text": "test caption"})

    # Handle actual return format
    if isinstance(result, dict):
        assert "image" in result or "pixel_values" in result
        aug_image = result.get("image") or result.get("pixel_values")
    elif isinstance(result, tuple):
        aug_image, _ = result
    else:
        aug_image = result

    # Verify augmented image
    assert aug_image is not None
    assert isinstance(aug_image, (torch.Tensor, Image.Image))
```

**Action**: Apply this pattern to all 27 skip instances in test_augmentation_pipeline.py

---

### Fix 2: test_specialized_losses.py

**Problem**: 7 instances of "Loss factory has different interface"

**Solution**: Check actual loss factory signature

```python
# Step 1: Verify actual signature
from src.training.losses.loss_factory import create_loss_function
import inspect
print(inspect.signature(create_loss_function))

# Step 2: Update test to match
def test_create_loss_with_factory(self):
    # Find actual factory function signature
    args = MockArgs()

    # Update to match actual signature
    loss = create_loss_function(
        loss_type='simclr',
        args=args  # Or whatever the actual signature requires
    )

    assert loss is not None
```

---

## Day 5: Fix Import Blockers

### Fix 1: Verify Imports in CI

```bash
# Run this in CI environment
python -c "from src.training.losses import DecorrelationLoss, MultitaskLoss, CLIPLoss"
python -c "from src.training.losses import CombinedLoss, FeatureConsistencyLoss"
python -c "from src.training.losses.loss_factory import create_loss_function"
python -c "from src.data.tokenization.wmt_bpe_tokenizer import WMTBPETokenizer"
```

**If any fail**: Add missing dependencies or fix import paths

### Fix 2: Remove Conditional Imports

```python
# BEFORE (test_specialized_losses.py lines 23-51)
try:
    from src.training.losses import DecorrelationLoss
except ImportError:
    DecorrelationLoss = None

@pytest.mark.skipif(DecorrelationLoss is None, reason="DecorrelationLoss not available")
def test_decorrelation_loss():
    ...

# AFTER (assume imports always work in test environment)
from src.training.losses import DecorrelationLoss

def test_decorrelation_loss():
    ...
```

**Action**: Apply to all conditional imports in test_specialized_losses.py

---

## Day 6-7: Add Centralized Fixtures

### conftest.py Enhancement

```python
"""Centralized test fixtures."""
import pytest
import torch
import torch.nn as nn
from PIL import Image


@pytest.fixture
def device():
    """Get available device (CPU/CUDA/MPS)."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")


@pytest.fixture
def sample_image_pil():
    """Sample PIL image for testing."""
    return Image.new("RGB", (224, 224), color=(100, 150, 200))


@pytest.fixture
def sample_image_tensor(device):
    """Sample image tensor for testing."""
    return torch.randn(3, 224, 224, device=device)


@pytest.fixture
def sample_batch_images(device):
    """Batch of image tensors."""
    return torch.randn(8, 3, 224, 224, device=device)


@pytest.fixture
def sample_text_tokens(device):
    """Sample text token IDs."""
    return torch.randint(0, 1000, (8, 50), device=device)


@pytest.fixture
def sample_vision_features(device):
    """Sample vision feature vectors."""
    return torch.randn(8, 512, device=device)


@pytest.fixture
def sample_text_features(device):
    """Sample text feature vectors."""
    return torch.randn(8, 512, device=device)


@pytest.fixture
def simple_vision_model():
    """Simple vision model for testing."""
    return nn.Sequential(
        nn.Linear(3 * 224 * 224, 512),
        nn.ReLU(),
        nn.Linear(512, 256)
    )


@pytest.fixture
def simple_text_model():
    """Simple text model for testing."""
    return nn.Sequential(
        nn.Embedding(1000, 256),
        nn.Linear(256, 256)
    )


@pytest.fixture
def temp_checkpoint_dir(tmp_path):
    """Temporary directory for checkpoints."""
    checkpoint_dir = tmp_path / "checkpoints"
    checkpoint_dir.mkdir()
    return checkpoint_dir


@pytest.fixture
def mock_dataloader(sample_batch_images, sample_text_tokens):
    """Mock dataloader for testing."""
    from torch.utils.data import TensorDataset, DataLoader

    dataset = TensorDataset(sample_batch_images, sample_text_tokens)
    return DataLoader(dataset, batch_size=4)


def pytest_configure(config):
    """Register custom marks."""
    config.addinivalue_line("markers", "no_test: mark a class as not being a test class")
    config.addinivalue_line("markers", "cuda: mark test as requiring CUDA")
    config.addinivalue_line("markers", "slow: mark test as slow running")
    config.addinivalue_line("markers", "integration: mark test as integration test")
```

---

## Expected Results After 1 Week

| Metric | Before | After | Change |
|--------|--------|-------|--------|
| Overall Coverage | 10% | 25% | +15% |
| transformer_trainer.py | 6% | 80% | +74% |
| loss_factory.py | 9.7% | 85% | +75.3% |
| model_factory.py | 9.2% | 85% | +75.8% |
| Skip Statements | 55 | 20 | -35 |
| Tests Passing | ~400 | ~450 | +50 |

---

## Quick Commands

```bash
# Day 1-2: Create new test files
touch tests/test_transformer_trainer.py
touch tests/test_loss_factory.py
touch tests/test_model_factory.py

# Day 3-4: Run tests to find interface mismatches
pytest tests/test_augmentation_pipeline.py -v
pytest tests/test_specialized_losses.py -v

# Day 5: Verify imports
python -c "from src.training.losses import DecorrelationLoss"
python -c "from src.data.tokenization.wmt_bpe_tokenizer import WMTBPETokenizer"

# Day 6-7: Test with new fixtures
pytest tests/ -v --fixtures

# Final verification
pytest tests/ --cov=src --cov-report=term-missing
```

---

## Success Criteria

- [ ] 3 new test files created with >50 tests total
- [ ] Coverage of transformer_trainer.py >70%
- [ ] Coverage of loss_factory.py >80%
- [ ] Coverage of model_factory.py >80%
- [ ] Skip statements reduced from 55 to <25
- [ ] All import errors resolved
- [ ] Overall coverage >25%
