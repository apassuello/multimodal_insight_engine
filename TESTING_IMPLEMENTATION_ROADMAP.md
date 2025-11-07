# Testing Implementation Roadmap
## MultiModal Insight Engine - Priority-Based Test Development Plan

**Duration:** 4-6 weeks
**Total Effort:** 175 hours
**Target Coverage:** 65-75%

---

## Phase 1: Critical Component Testing (Weeks 1-3, 80 hours)

### Week 1: Loss Function Testing Infrastructure + Top 5 Losses

**Objective:** Establish loss testing patterns and test the highest-impact loss functions

#### Step 1.1: Create Test Infrastructure (2 hours)

**Create: `tests/unit/training/test_loss_utils.py`**
```python
"""Utilities for loss function testing."""

import torch
import pytest
from typing import Tuple, Dict

@pytest.fixture
def device():
    """Return available device."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

@pytest.fixture
def batch_size():
    return 8

@pytest.fixture
def embedding_dim():
    return 128

@pytest.fixture
def batch_embeddings(batch_size, embedding_dim, device):
    """Create standard batch of embeddings for loss testing."""
    z_a = torch.randn(batch_size, embedding_dim, device=device)
    z_b = torch.randn(batch_size, embedding_dim, device=device)
    return z_a, z_b

def assert_loss_valid(loss_tensor: torch.Tensor, description: str = ""):
    """Assert loss is valid (not NaN, not Inf, is scalar)."""
    assert not torch.isnan(loss_tensor), f"Loss is NaN: {description}"
    assert not torch.isinf(loss_tensor), f"Loss is Inf: {description}"
    assert loss_tensor.numel() == 1, f"Loss should be scalar: {description}"
    assert loss_tensor.dtype in [torch.float32, torch.float64]

def assert_gradients_valid(tensor: torch.Tensor, description: str = ""):
    """Assert tensor has valid gradients."""
    assert tensor.grad is not None, f"No gradients: {description}"
    assert not torch.all(tensor.grad == 0), f"Gradients all zero: {description}"
    assert not torch.isnan(tensor.grad).any(), f"NaN gradients: {description}"
```

#### Step 1.2: Test VICReg Loss (4 hours)

**Create: `tests/unit/training/losses/test_vicreg_loss.py`**
```python
"""Tests for VICReg loss implementation."""

import pytest
import torch
import numpy as np
from src.training.losses.vicreg_loss import VICRegLoss
from tests.unit.training.test_loss_utils import (
    assert_loss_valid, assert_gradients_valid
)

class TestVICRegLossInitialization:
    """Test VICRegLoss initialization."""

    def test_init_default_parameters(self):
        """Test initialization with default parameters."""
        loss = VICRegLoss()
        assert loss.sim_coeff == 10.0
        assert loss.var_coeff == 5.0
        assert loss.cov_coeff == 1.0
        assert loss.epsilon == 1e-3
        assert loss.warmup_epochs == 5

    def test_init_custom_parameters(self):
        """Test initialization with custom parameters."""
        loss = VICRegLoss(
            sim_coeff=20.0,
            var_coeff=10.0,
            cov_coeff=2.0,
            epsilon=1e-4,
            warmup_epochs=3
        )
        assert loss.sim_coeff == 20.0
        assert loss.var_coeff == 10.0
        assert loss.cov_coeff == 2.0
        assert loss.epsilon == 1e-4
        assert loss.warmup_epochs == 3

    def test_init_curriculum_disabled(self):
        """Test initialization with curriculum learning disabled."""
        loss = VICRegLoss(curriculum=False)
        assert loss.curriculum == False

class TestVICRegLossForward:
    """Test VICRegLoss forward pass."""

    def test_forward_output_shape(self, batch_embeddings):
        """Test forward pass returns dict with correct keys."""
        z_a, z_b = batch_embeddings
        loss = VICRegLoss()
        output = loss(z_a, z_b)

        expected_keys = {
            "loss",
            "invariance_loss",
            "variance_loss",
            "covariance_loss",
            "sim_weight",
            "var_weight",
            "cov_weight",
            "warmup_factor"
        }
        assert expected_keys.issubset(output.keys())

    def test_forward_loss_is_scalar(self, batch_embeddings):
        """Test that loss is a scalar tensor."""
        z_a, z_b = batch_embeddings
        loss = VICRegLoss()
        output = loss(z_a, z_b)

        assert_loss_valid(output["loss"], "VICReg main loss")

    def test_forward_all_components_valid(self, batch_embeddings):
        """Test all loss components are valid numbers."""
        z_a, z_b = batch_embeddings
        loss = VICRegLoss()
        output = loss(z_a, z_b)

        assert_loss_valid(
            torch.tensor(output["invariance_loss"]),
            "Invariance component"
        )
        assert_loss_valid(
            torch.tensor(output["variance_loss"]),
            "Variance component"
        )
        assert_loss_valid(
            torch.tensor(output["covariance_loss"]),
            "Covariance component"
        )

class TestVICRegLossComponents:
    """Test individual loss components."""

    def test_invariance_identical_embeddings(self):
        """Invariance loss should be zero for identical embeddings."""
        z = torch.randn(8, 128)
        loss = VICRegLoss(sim_coeff=1.0, var_coeff=0.0, cov_coeff=0.0)
        output = loss(z.clone(), z.clone())

        # Invariance should be near zero
        assert output["invariance_loss"] < 0.01

    def test_variance_constant_embeddings(self):
        """Variance loss should be high for constant embeddings."""
        # All same values -> zero variance -> high variance loss
        z = torch.ones(8, 128)
        loss = VICRegLoss(sim_coeff=0.0, var_coeff=1.0, cov_coeff=0.0)
        output = loss(z, z)

        # Variance loss should penalize this
        assert output["variance_loss"] > 0.5

    def test_variance_unit_variance(self):
        """Variance loss should be low for unit variance embeddings."""
        # Normalize to unit variance
        z = torch.randn(8, 128)
        z = (z - z.mean(dim=0)) / (z.std(dim=0) + 1e-4)

        loss = VICRegLoss(sim_coeff=0.0, var_coeff=1.0, cov_coeff=0.0)
        output = loss(z, z)

        # Should be relatively low
        assert output["variance_loss"] < 0.5

class TestVICRegLossGradientFlow:
    """Test gradient computation."""

    def test_backward_pass(self, batch_embeddings):
        """Test backward pass computes gradients."""
        z_a, z_b = batch_embeddings
        z_a.requires_grad_(True)
        z_b.requires_grad_(True)

        loss = VICRegLoss()
        output = loss(z_a, z_b)
        output["loss"].backward()

        assert_gradients_valid(z_a, "z_a gradients")
        assert_gradients_valid(z_b, "z_b gradients")

    def test_gradient_magnitude_reasonable(self, batch_embeddings):
        """Test gradient magnitudes are reasonable (not exploding)."""
        z_a, z_b = batch_embeddings
        z_a.requires_grad_(True)
        z_b.requires_grad_(True)

        loss = VICRegLoss()
        output = loss(z_a, z_b)
        output["loss"].backward()

        grad_norm_a = torch.norm(z_a.grad)
        grad_norm_b = torch.norm(z_b.grad)

        # Gradients should not explode
        assert grad_norm_a < 100
        assert grad_norm_b < 100
        assert grad_norm_a > 0
        assert grad_norm_b > 0

class TestVICRegLossCurriculum:
    """Test curriculum learning functionality."""

    def test_warmup_factor_progression(self):
        """Test warmup factor increases with epoch."""
        loss = VICRegLoss(curriculum=True, warmup_epochs=5, num_epochs=30)

        factors = []
        for epoch in range(10):
            loss.update_epoch(epoch)
            factors.append(loss.get_warmup_factor())

        # Factors should generally increase
        assert factors[0] < factors[2]
        assert factors[2] < factors[5]

    def test_warmup_factor_bounds(self):
        """Test warmup factor stays in [0.3, 0.5] range."""
        loss = VICRegLoss(curriculum=True, warmup_epochs=5, num_epochs=30)

        for epoch in range(20):
            loss.update_epoch(epoch)
            factor = loss.get_warmup_factor()

            assert 0.0 <= factor <= 1.0

    def test_curriculum_weights_warm_up(self, batch_embeddings):
        """Test that curriculum adjusts weights correctly."""
        z_a, z_b = batch_embeddings

        loss_curriculum = VICRegLoss(curriculum=True, warmup_epochs=5)
        loss_no_curriculum = VICRegLoss(curriculum=False)

        # Early epoch with curriculum
        loss_curriculum.update_epoch(0)
        output_early = loss_curriculum(z_a, z_b)

        # Later epoch with curriculum
        loss_curriculum.update_epoch(8)
        output_late = loss_curriculum(z_a, z_b)

        # var_weight should increase
        assert output_late["var_weight"] >= output_early["var_weight"]

class TestVICRegLossNumericalStability:
    """Test numerical stability with extreme inputs."""

    def test_large_values(self):
        """Test with very large embedding values."""
        z_a = torch.ones(8, 128) * 1e6
        z_b = torch.ones(8, 128) * 1e6
        loss = VICRegLoss()
        output = loss(z_a, z_b)

        assert_loss_valid(output["loss"], "Large values")

    def test_small_values(self):
        """Test with very small embedding values."""
        z_a = torch.ones(8, 128) * 1e-6
        z_b = torch.ones(8, 128) * 1e-6
        loss = VICRegLoss()
        output = loss(z_a, z_b)

        assert_loss_valid(output["loss"], "Small values")

    def test_mixed_magnitude(self):
        """Test with mixed magnitude values."""
        z_a = torch.randn(8, 128) * torch.logspace(-3, 3, 128)
        z_b = torch.randn(8, 128) * torch.logspace(-3, 3, 128)
        loss = VICRegLoss()
        output = loss(z_a, z_b)

        assert_loss_valid(output["loss"], "Mixed magnitude")

    def test_near_zero_batch(self):
        """Test with embeddings very close to zero."""
        z_a = torch.randn(8, 128) * 1e-8
        z_b = torch.randn(8, 128) * 1e-8
        loss = VICRegLoss()
        output = loss(z_a, z_b)

        assert_loss_valid(output["loss"], "Near-zero values")

class TestVICRegLossWithBatchSize:
    """Test with different batch sizes."""

    @pytest.mark.parametrize("batch_size", [1, 2, 4, 8, 16, 32])
    def test_various_batch_sizes(self, batch_size):
        """Test loss works with various batch sizes."""
        z_a = torch.randn(batch_size, 128)
        z_b = torch.randn(batch_size, 128)
        loss = VICRegLoss()
        output = loss(z_a, z_b)

        assert_loss_valid(output["loss"], f"Batch size {batch_size}")

    @pytest.mark.parametrize("embedding_dim", [32, 64, 128, 256, 512])
    def test_various_embedding_dims(self, embedding_dim):
        """Test loss works with various embedding dimensions."""
        z_a = torch.randn(8, embedding_dim)
        z_b = torch.randn(8, embedding_dim)
        loss = VICRegLoss()
        output = loss(z_a, z_b)

        assert_loss_valid(output["loss"], f"Embedding dim {embedding_dim}")
```

**Estimated Time:** 4 hours
**Test Functions:** 20+
**Coverage Impact:** VICReg: 0% → 80%+

#### Step 1.3: Test Contrastive Loss (4 hours)

**Create: `tests/unit/training/losses/test_contrastive_loss.py`**
```python
"""Tests for Contrastive loss implementation."""

import pytest
import torch
from src.training.losses.contrastive_loss import ContrastiveLoss
from tests.unit.training.test_loss_utils import assert_loss_valid

class TestContrastiveLossInitialization:
    """Test ContrastiveLoss initialization."""

    def test_init_default(self):
        """Test default initialization."""
        loss = ContrastiveLoss()
        assert loss.temperature == 0.07
        assert loss.loss_type == "infonce"
        assert loss.reduction == "mean"

    def test_init_with_projections(self):
        """Test initialization with projection heads."""
        loss = ContrastiveLoss(
            add_projection=True,
            projection_dim=256,
            input_dim=512
        )
        assert loss.add_projection == True
        assert loss.projection_dim == 256

class TestContrastiveLossForward:
    """Test forward pass."""

    def test_forward_in_batch_strategy(self):
        """Test forward with in-batch sampling strategy."""
        loss_fn = ContrastiveLoss(sampling_strategy="in-batch")
        z_a = torch.randn(4, 256)
        z_b = torch.randn(4, 256)

        output = loss_fn(z_a, z_b)
        assert_loss_valid(output["loss"], "In-batch strategy")

    def test_forward_different_temperatures(self):
        """Test that temperature affects loss magnitude."""
        z_a = torch.randn(8, 256)
        z_b = torch.randn(8, 256)

        loss_low_temp = ContrastiveLoss(temperature=0.01)(z_a, z_b)
        loss_high_temp = ContrastiveLoss(temperature=1.0)(z_a, z_b)

        # Lower temperature should produce sharper differences
        # But both should be valid
        assert_loss_valid(loss_low_temp["loss"])
        assert_loss_valid(loss_high_temp["loss"])

    @pytest.mark.parametrize("loss_type", ["infonce", "nt_xent", "supervised"])
    def test_different_loss_types(self, loss_type):
        """Test different loss function types."""
        loss_fn = ContrastiveLoss(loss_type=loss_type)
        z_a = torch.randn(8, 256)
        z_b = torch.randn(8, 256)

        try:
            output = loss_fn(z_a, z_b)
            assert_loss_valid(output["loss"], f"Loss type: {loss_type}")
        except NotImplementedError:
            pytest.skip(f"Loss type {loss_type} not implemented")

class TestContrastiveLossGradients:
    """Test gradient flow."""

    def test_gradients_flow(self):
        """Test gradients flow through loss."""
        loss_fn = ContrastiveLoss()
        z_a = torch.randn(4, 256, requires_grad=True)
        z_b = torch.randn(4, 256, requires_grad=True)

        output = loss_fn(z_a, z_b)
        output["loss"].backward()

        assert z_a.grad is not None
        assert z_b.grad is not None
        assert not torch.all(z_a.grad == 0)
```

**Estimated Time:** 4 hours
**Test Functions:** 15+
**Coverage Impact:** ContrastiveLoss: 0% → 75%+

#### Step 1.4: Test Supervised Contrastive Loss (3 hours)

**Create: `tests/unit/training/losses/test_supervised_contrastive_loss.py`**
```python
"""Tests for Supervised Contrastive loss."""

import pytest
import torch
from src.training.losses.supervised_contrastive_loss import SupervisedContrastiveLoss
from tests.unit.training.test_loss_utils import assert_loss_valid

class TestSupervisedContrastiveLoss:
    """Test supervised contrastive loss."""

    def test_same_class_attraction(self):
        """Same class embeddings should be attracted."""
        loss_fn = SupervisedContrastiveLoss(temperature=0.07)

        # Embeddings with labels
        z = torch.randn(8, 128)
        labels = torch.tensor([0, 0, 1, 1, 2, 2, 3, 3])

        output = loss_fn(z, labels)
        assert_loss_valid(output["loss"])

    def test_gradient_flow(self):
        """Test gradient computation."""
        loss_fn = SupervisedContrastiveLoss()
        z = torch.randn(8, 128, requires_grad=True)
        labels = torch.tensor([0, 0, 1, 1, 2, 2, 3, 3])

        output = loss_fn(z, labels)
        output["loss"].backward()

        assert z.grad is not None
        assert not torch.all(z.grad == 0)
```

**Estimated Time:** 3 hours
**Test Functions:** 10+

#### Step 1.5: Test EMA MoCo Loss (3 hours)

**Similar pattern to above**
**Estimated Time:** 3 hours
**Test Functions:** 12+

#### Summary for Week 1:

| Loss Function | Time | Tests | Coverage |
|--------------|------|-------|----------|
| VICReg | 4h | 20+ | 0→80% |
| Contrastive | 4h | 15+ | 0→75% |
| Supervised Contrastive | 3h | 10+ | 0→75% |
| EMA MoCo | 3h | 12+ | 0→70% |
| Infrastructure | 2h | - | - |
| **TOTAL WEEK 1** | **16 hours** | **57+** | **Avg 75%** |

**Outcome:** 5 major loss functions tested, patterns established, reusable test infrastructure

---

### Week 2: Remaining Loss Functions (25 hours)

**Objective:** Complete remaining 13 loss functions with established patterns

**Test Implementations (using patterns from Week 1):**

1. **CLIP Style Loss** (3 hours, 12 tests)
2. **Decorrelation Loss** (3 hours, 10 tests)
3. **Feature Consistency Loss** (3 hours, 10 tests)
4. **Hard Negative Mining Loss** (3 hours, 10 tests)
5. **Dynamic Temperature Loss** (2 hours, 8 tests)
6. **Barlow Twins Loss** (3 hours, 12 tests)
7. **Memory Queue Loss** (3 hours, 12 tests)
8. **Multimodal Mixed Contrastive** (3 hours, 12 tests)
9. **Decoupled Contrastive** (3 hours, 12 tests)
10. **Hybrid Pretrain VICReg** (3 hours, 12 tests)
11. **Combined Loss** (2 hours, 10 tests)
12. **Multitask Loss** (2 hours, 10 tests)
13. **Loss Factory** (3 hours, 15 tests)

**Week 2 Summary:**
- 25 hours
- 145+ test functions
- All 18 untested loss functions achieve 70%+ coverage
- Loss module coverage: 10% → 75%

---

### Week 3: Trainer Testing (30 hours)

#### Step 3.1: Establish Trainer Test Infrastructure (3 hours)

**Create: `tests/unit/training/trainers/test_trainer_utils.py`**
```python
"""Utilities for trainer testing."""

import torch
import torch.nn as nn
import pytest
from typing import Dict, Tuple

@pytest.fixture
def simple_model():
    """Create a simple model for trainer tests."""
    return nn.Sequential(
        nn.Linear(64, 128),
        nn.ReLU(),
        nn.Linear(128, 10)
    )

@pytest.fixture
def simple_dataloader():
    """Create simple dataloader for trainer tests."""
    dataset = torch.utils.data.TensorDataset(
        torch.randn(32, 64),  # inputs
        torch.randint(0, 10, (32,))  # labels
    )
    return torch.utils.data.DataLoader(dataset, batch_size=8)

@pytest.fixture
def trainer_config():
    """Basic trainer configuration."""
    return {
        'epochs': 2,
        'learning_rate': 0.001,
        'batch_size': 8,
        'device': 'cpu',
    }
```

#### Step 3.2: Test Base Trainer (4 hours)

**Create: `tests/unit/training/trainers/test_trainer.py`**
```python
"""Tests for base Trainer class."""

import pytest
import torch
from src.training.trainers.trainer import Trainer

class TestTrainerInitialization:
    """Test trainer initialization."""

    def test_init_with_config(self, simple_model, trainer_config):
        """Test trainer initialization."""
        trainer = Trainer(model=simple_model, config=trainer_config)
        assert trainer.model is simple_model

class TestTrainerTrainStep:
    """Test training step."""

    def test_train_step_updates_weights(self, simple_model, simple_dataloader):
        """Training step should update weights."""
        trainer = Trainer(model=simple_model)

        # Get initial weights
        initial_weights = [p.clone() for p in simple_model.parameters()]

        # Run training step
        batch = next(iter(simple_dataloader))
        trainer.training_step(batch)

        # Check weights changed
        for initial, current in zip(initial_weights, simple_model.parameters()):
            assert not torch.allclose(initial, current)

class TestTrainerValidation:
    """Test validation."""

    def test_validation_step_no_grad(self, simple_model, simple_dataloader):
        """Validation should not compute gradients."""
        trainer = Trainer(model=simple_model)
        simple_model.train()

        batch = next(iter(simple_dataloader))
        with torch.no_grad():
            metrics = trainer.validation_step(batch)

        assert isinstance(metrics, dict)
```

**Estimated Time:** 4 hours

#### Step 3.3: Test MultiModal Trainer (8 hours)

**Create: `tests/unit/training/trainers/test_multimodal_trainer.py`**
```python
"""Tests for MultiModal Trainer."""

import pytest
import torch
from src.training.trainers.multimodal_trainer import MultiModalTrainer

class TestMultiModalTrainerInitialization:
    """Test initialization."""

    def test_init_default_config(self):
        """Initialize with default config."""
        # Mock model and config
        config = {
            'vision_dim': 768,
            'text_dim': 768,
            'hidden_dim': 512,
        }
        trainer = MultiModalTrainer(config=config)
        assert trainer.config['vision_dim'] == 768

    def test_init_custom_config(self):
        """Initialize with custom config."""
        config = {
            'vision_dim': 512,
            'text_dim': 512,
            'hidden_dim': 256,
            'num_heads': 8,
        }
        trainer = MultiModalTrainer(config=config)
        assert trainer.config == config

class TestMultiModalTrainerAlignment:
    """Test vision-language alignment."""

    def test_multimodal_alignment(self):
        """Test that training aligns modalities."""
        # Create trainer with small model
        config = {'hidden_dim': 128}
        trainer = MultiModalTrainer(config=config)

        # Create aligned data (should train well)
        vision_emb = torch.randn(8, 128)
        text_emb = torch.randn(8, 128)

        # Should compute alignment loss
        output = trainer.forward(vision_emb, text_emb)
        assert 'loss' in output
        assert 'alignment_score' in output

class TestMultiModalTrainerConstitutional:
    """Test constitutional constraints."""

    def test_constitutional_constraint(self):
        """Test constitutional constraints are enforced."""
        config = {
            'hidden_dim': 128,
            'use_constitutional': True,
        }
        trainer = MultiModalTrainer(config=config)

        vision_emb = torch.randn(8, 128)
        text_emb = torch.randn(8, 128)

        output = trainer.forward(vision_emb, text_emb)

        # Should have constitutional score
        if 'constitutional_score' in output:
            assert 0 <= output['constitutional_score'] <= 1
```

**Estimated Time:** 8 hours
**Test Functions:** 25+

#### Step 3.4: Test Other Trainers (15 hours)

- Constitutional Trainer (3h)
- Transformer Trainer (4h)
- Vision Transformer Trainer (3h)
- Language Model Trainer (3h)
- Multistage Trainer (2h)

**Week 3 Summary:**
- 30 hours
- 80+ test functions
- All 8 trainers tested
- Trainers coverage: 30% → 70%+
- **Phase 1 Total:** 80 hours, 282+ tests, coverage 45% → 60%

---

## Phase 2: Testing Infrastructure & CI/CD (Weeks 4-5, 30 hours)

### Week 4: Enhanced Test Infrastructure (15 hours)

#### Task 4.1: Enhance conftest.py (3 hours)

**Create comprehensive fixture library:**
```python
# tests/conftest.py

import pytest
import torch
import numpy as np
import tempfile
from pathlib import Path

@pytest.fixture(autouse=True)
def reset_seeds():
    """Reset random seeds for deterministic tests."""
    torch.manual_seed(42)
    np.random.seed(42)

@pytest.fixture
def device():
    """Return appropriate device."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")

@pytest.fixture
def temp_directory():
    """Create temporary directory for test files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)

# Model fixtures
@pytest.fixture
def transformer_model(device):
    """Create small transformer for testing."""
    from src.models import Transformer
    config = TransformerConfig(vocab_size=1000, hidden_dim=64)
    return Transformer(config).to(device)

# Data fixtures
@pytest.fixture
def sample_batch(device):
    """Create sample batch of data."""
    return {
        'input_ids': torch.randint(0, 1000, (4, 32)).to(device),
        'attention_mask': torch.ones(4, 32).to(device),
    }
```

#### Task 4.2: Test Parametrization (4 hours)

**Refactor existing tests to use parametrize:**
```python
# Before
def test_loss_with_mean_reduction():
    loss = Loss(reduction='mean')
    ...

def test_loss_with_sum_reduction():
    loss = Loss(reduction='sum')
    ...

def test_loss_with_none_reduction():
    loss = Loss(reduction='none')
    ...

# After
@pytest.mark.parametrize("reduction", ["mean", "sum", "none"])
def test_loss_reduction(reduction):
    loss = Loss(reduction=reduction)
    ...
```

#### Task 4.3: Test Fixtures Library (5 hours)

**Create fixture modules:**
- `tests/fixtures/models.py` - Model creation
- `tests/fixtures/data.py` - Data and dataloaders
- `tests/fixtures/configs.py` - Configuration builders
- `tests/fixtures/mocks.py` - Mock objects

#### Task 4.4: Pytest Configuration (3 hours)

**Create: `pytest.ini`**
```ini
[pytest]
testpaths = tests
python_files = test_*.py
python_classes = Test*
python_functions = test_*
addopts =
    -v
    --strict-markers
    --tb=short
    -ra
    --cov=src
    --cov-report=html
    --cov-report=xml
    --cov-report=term-missing
markers =
    slow: marks tests as slow (deselect with '-m "not slow"')
    integration: marks tests as integration tests
    unit: marks tests as unit tests
    gpu: marks tests that require GPU
```

**Week 4 Summary:** 15 hours

### Week 5: CI/CD Pipeline & Documentation (15 hours)

#### Task 5.1: GitHub Actions Workflow (8 hours)

**Create: `.github/workflows/test.yml`**
```yaml
name: Tests

on:
  push:
    branches: [ main, develop ]
  pull_request:
    branches: [ main, develop ]

jobs:
  test:
    runs-on: ${{ matrix.os }}
    strategy:
      matrix:
        os: [ubuntu-latest, macos-latest]
        python-version: ['3.9', '3.10', '3.11']

    steps:
    - uses: actions/checkout@v3

    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: ${{ matrix.python-version }}

    - name: Install dependencies
      run: |
        pip install -e .
        pip install pytest pytest-cov

    - name: Run tests with coverage
      run: |
        pytest tests/ \
          --cov=src \
          --cov-report=xml \
          --cov-report=html \
          --cov-fail-under=65

    - name: Upload coverage to Codecov
      uses: codecov/codecov-action@v3
      with:
        file: ./coverage.xml
        fail_ci_if_error: true
```

#### Task 5.2: Test Documentation (5 hours)

**Create: `TESTING_GUIDE.md`**
```markdown
# Testing Guide

## Running Tests

# All tests
pytest

# Specific test file
pytest tests/unit/training/test_losses.py

# Specific test
pytest tests/unit/training/test_losses.py::test_vicreg_loss

# With coverage
pytest --cov=src --cov-report=html

## Writing New Tests

1. Create test file in appropriate directory
2. Use descriptive names (test_<feature>_<scenario>)
3. Use fixtures from conftest.py
4. Assert specific outcomes
5. Test both success and failure cases

## Fixtures

See tests/conftest.py for available fixtures.
```

#### Task 5.3: Coverage Improvements (2 hours)

**Update `.coveragerc`:**
```ini
[run]
source = src
omit =
    tests/*
    */tests/*
    setup.py

[report]
exclude_lines =
    pragma: no cover
    raise NotImplementedError
    if TYPE_CHECKING:
    if __debug__:
    if __name__ == '__main__':
    @abstractmethod
precision = 2

[html]
directory = htmlcov
```

**Phase 2 Summary:** 30 hours
- Better test organization
- Automated CI/CD
- Improved maintainability
- Clear documentation

---

## Phase 3: Quality Enhancements (Weeks 6-7, 25 hours)

### Week 6: Property-Based Testing (12 hours)

**Implement with Hypothesis library:**
```python
# tests/property_based/test_losses_properties.py
from hypothesis import given, strategies as st

@given(
    st.lists(
        st.floats(allow_nan=False, allow_infinity=False, width=32),
        min_size=64,
        max_size=64
    ),
    st.integers(min_value=1, max_value=128)
)
def test_loss_numerical_stability(embeddings, batch_size):
    """Loss should be numerically stable for any valid input."""
    z_a = torch.tensor(embeddings[:batch_size]).float()
    z_b = torch.tensor(embeddings[batch_size:]).float()

    loss = VICRegLoss()
    output = loss(z_a, z_b)

    assert not torch.isnan(output["loss"])
    assert not torch.isinf(output["loss"])
```

### Week 7: Integration & Performance Tests (13 hours)

**Create: `tests/integration/test_end_to_end.py`**
```python
def test_full_training_pipeline():
    """Test complete training pipeline."""
    # Load data
    # Create model
    # Train for 1 epoch
    # Validate
    # Save checkpoint
    # Load checkpoint
    # Verify convergence
```

---

## Phase 4: Maintenance & Documentation (40 hours, ongoing)

### Mutation Testing (10 hours)
- Run mutmut on test suite
- Improve tests to catch more mutations
- Target >80% mutation survival

### Coverage Analysis (10 hours)
- Identify branch coverage gaps
- Add tests for untested branches
- Improve branch coverage from 34.53% to 60%+

### TDD Implementation (10 hours)
- Create TDD guidelines
- Establish pre-commit test coverage checks
- Create example TDD kata

### Documentation (10 hours)
- Complete test documentation
- Add examples
- Create testing best practices guide

---

## Success Metrics & Checkpoints

### Checkpoint 1: End of Week 1 (Critical Functions)
- ✓ 57+ test functions for 5 loss functions
- ✓ 0% → 75%+ coverage for tested losses
- ✓ Test infrastructure established
- **Expected Coverage:** 50%

### Checkpoint 2: End of Week 3 (All Critical Tests)
- ✓ 282+ new test functions
- ✓ All loss functions (18) tested
- ✓ All trainers (8) tested
- ✓ Safety module tests started
- **Expected Coverage:** 60%

### Checkpoint 3: End of Week 5 (Infrastructure)
- ✓ CI/CD pipeline active
- ✓ Enhanced fixtures and utilities
- ✓ Test documentation complete
- ✓ Parallel test execution enabled
- **Expected Coverage:** 62%

### Checkpoint 4: End of Week 7 (Quality)
- ✓ Integration tests passing
- ✓ Performance baselines set
- ✓ Property-based tests working
- ✓ 100+ tests added
- **Expected Coverage:** 65-70%

### Long-Term (Ongoing Maintenance)
- ✓ Coverage maintained >75%
- ✓ New features always have tests
- ✓ TDD practiced consistently
- ✓ Mutation testing >80%

---

## Resource Requirements

### People
- 1 Senior Test Engineer (175 hours)
- OR 2 Mid-Level Engineers (parallel work)
- OR 3 Junior Engineers + 1 Senior (mentoring)

### Tools
- pytest (already installed)
- pytest-cov (already installed)
- hypothesis (property-based testing)
- mutmut (mutation testing)
- Codecov (coverage reporting)

### Time
- 4-6 weeks for full implementation
- 2-3 weeks for critical path only (Phase 1-2)
- Ongoing maintenance: 5-10 hours/week

---

## Risk Mitigation

### Risk: Tests Slow Down Development
**Mitigation:**
- Run only fast unit tests locally
- Full suite runs in CI/CD (parallelized)
- Pytest marks for slow tests

### Risk: Tests Become Brittle
**Mitigation:**
- Use mocks appropriately
- Test behavior, not implementation
- Avoid hard-coded test data

### Risk: Coverage Metrics Misleading
**Mitigation:**
- Focus on meaningful tests
- Use mutation testing to validate
- Code review test quality

### Risk: Insufficient Testing Knowledge
**Mitigation:**
- Use provided templates
- Document patterns
- Provide code review

---

## Conclusion

This roadmap provides a structured approach to achieving 65-75% test coverage in 4-6 weeks. By following the phase-based approach, you can:

1. **Quickly fix critical gaps** (Loss functions, trainers)
2. **Establish sustainable infrastructure** (CI/CD, fixtures)
3. **Improve long-term quality** (Property-based tests, mutation testing)
4. **Maintain progress** (TDD practices, documentation)

The estimated 175 hours of effort will result in:
- 400+ new test functions
- 65%+ code coverage (from 45%)
- Automated testing in CI/CD
- Improved code quality
- Easier maintenance and debugging
- Production-ready testing infrastructure

**Next Step:** Begin Phase 1, Week 1 with loss function test infrastructure and the top 5 loss implementations.
