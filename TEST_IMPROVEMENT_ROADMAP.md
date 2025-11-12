# Test Suite Improvement Roadmap

## Overview

Current state: 739 passing tests, 36% coverage, mixed quality
Target state: 500-600 focused tests, 50%+ coverage, high-quality assertions

---

## Quick Wins (Complete in 1-2 days)

### 1. Fix Critical Collection Error
**Impact**: Unblock ~15 tests immediately

```bash
# Install missing dependency
pip install nltk

# Verify collection now works
python -m pytest tests/test_training_metrics.py --co -q
```

**Expected result**: test_training_metrics.py tests become collectable

---

### 2. Fix HybridPretrainVICRegLoss Test Configuration
**File**: `tests/test_selfsupervised_losses.py`

**Problem**: Tests provide 128-dim embeddings but loss expects 768-dim

**Solution**:
```python
# CURRENT (BROKEN)
def test_basic_forward(self, embeddings_a, embeddings_b, device):
    loss_fn = HybridPretrainVICRegLoss(...)
    result = loss_fn(embeddings_a, embeddings_b)  # embeddings_a shape: [16, 128]
    # ERROR: Expected 768, got 128

# FIXED
def test_basic_forward(self, batch_size, device):
    # Create embeddings with correct dimension
    embed_dim = 128
    embeddings_a = torch.randn(batch_size, embed_dim, device=device)
    embeddings_b = torch.randn(batch_size, embed_dim, device=device)

    loss_fn = HybridPretrainVICRegLoss(
        sim_coeff=10.0,
        var_coeff=5.0,
        cov_coeff=1.0,
        # Set dimensions to match embeddings
        embed_dim=embed_dim,
        projection_dim=256,
    )

    result = loss_fn(embeddings_a, embeddings_b)
    loss = result['loss'] if isinstance(result, dict) else result

    assert not torch.isnan(loss)
    assert not torch.isinf(loss)
```

**Recovery**: +4 tests pass (TestHybridPretrainVICRegLoss)

---

### 3. Register Custom Pytest Markers
**Problem**: Unknown pytest marks cause warnings

```bash
# Add to pytest.ini or pyproject.toml
[tool:pytest]
markers =
    slow: marks tests as slow (deselect with '-m "not slow"')
    gpu: marks tests requiring GPU
    integration: marks integration tests
    smoke: marks smoke/sanity tests
```

**Result**: Clean test output, ability to skip slow tests

---

## Phase 1: Fix Broken Tests (Days 3-5)

### 4. Fix PPO Trainer Tests
**File**: `tests/test_ppo_trainer.py` (20+ failing tests)

**Root cause**: Setup issues, missing mocks

**Fix strategy**:
```python
# Mock environment setup
@pytest.fixture
def mock_env():
    """Create mock environment for PPO tests."""
    env = MagicMock()
    env.observation_space = Box(low=-1, high=1, shape=(4,))
    env.action_space = Discrete(2)
    env.reset.return_value = np.random.randn(4)
    return env

# Use in tests
def test_gae_basic_computation(self, mock_env):
    trainer = PPOTrainer(env=mock_env, ...)
    # Now test works without OS errors
```

**Expected recovery**: +15-20 tests passing

---

### 5. Fix Reward Model Tests
**File**: `tests/test_reward_model.py` (20+ errors)

**Root cause**: Same as PPO - missing mocks/fixtures

**Solution**: Create comprehensive fixtures for model initialization

```python
@pytest.fixture
def reward_model_config():
    """Configuration for reward model tests."""
    return {
        'model_name': 'gpt2-medium',
        'hidden_size': 768,
        'num_labels': 1,
        'dropout': 0.1,
    }

@pytest.fixture
def dummy_dataset():
    """Small dummy dataset for tests."""
    return {
        'input_ids': torch.randint(0, 1000, (10, 20)),
        'attention_mask': torch.ones(10, 20),
        'labels': torch.randn(10, 1),
    }
```

**Expected recovery**: +15-20 tests passing

---

## Phase 2: Strengthen Weak Assertions (Days 6-8)

### 6. Improve Contrastive Loss Tests
**File**: `tests/test_contrastive_losses.py`

**Current pattern** (WEAK):
```python
def test_temperature_sensitivity(self):
    loss_fn_low = ContrastiveLoss(temperature=0.01, ...)
    loss_fn_high = ContrastiveLoss(temperature=1.0, ...)

    loss_low = loss_fn_low(vision_features, text_features)
    loss_high = loss_fn_high(vision_features, text_features)

    assert loss_low != loss_high        # ← Weak: only checks different
    assert not torch.isnan(loss_low)
    assert not torch.isnan(loss_high)
```

**Improved pattern** (STRONG):
```python
def test_temperature_sensitivity(self):
    """Verify temperature affects loss magnitude as expected."""
    loss_fn_low = ContrastiveLoss(temperature=0.01, ...)
    loss_fn_high = ContrastiveLoss(temperature=1.0, ...)

    # Same data, different temperatures
    loss_low = loss_fn_low(vision_features, text_features)
    loss_high = loss_fn_high(vision_features, text_features)

    # Lower temperature = sharper distributions = higher InfoNCE loss
    # (confidence required for positive pairs)
    assert loss_low > loss_high, \
        f"Lower temp should have higher loss: {loss_low} vs {loss_high}"

    # Both should be in valid range
    assert 0 < loss_low.item() < 10
    assert 0 < loss_high.item() < 10

def test_gradient_magnitude_with_temperature(self):
    """Verify temperature affects gradient magnitudes."""
    vision_feat = vision_features.requires_grad_(True)

    # Low temperature
    loss_fn_low = ContrastiveLoss(temperature=0.01, ...)
    loss_low = loss_fn_low(vision_feat, text_features)
    loss_low.backward()
    grads_low = vision_feat.grad.clone()

    # High temperature
    vision_feat.grad.zero_()
    loss_fn_high = ContrastiveLoss(temperature=1.0, ...)
    loss_high = loss_fn_high(vision_feat, text_features)
    loss_high.backward()
    grads_high = vision_feat.grad.clone()

    # Lower temperature should have sharper gradients
    grad_mag_low = torch.norm(grads_low)
    grad_mag_high = torch.norm(grads_high)

    assert grad_mag_low > grad_mag_high / 2, \
        "Lower temp should have stronger gradients"
```

**Effort**: ~4-5 hours (20+ tests to improve)
**Coverage improvement**: +5-8%

---

## Phase 3: Add Missing Integration Tests (Days 9-12)

### 7. Add Loss Convergence Tests

**New test file**: `tests/test_loss_convergence.py`

```python
"""Test that losses actually decrease during training."""

def test_contrastive_loss_convergence():
    """Verify contrastive loss decreases with training."""
    model = nn.Sequential(nn.Linear(10, 128))
    loss_fn = ContrastiveLoss(temperature=0.07)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.1)

    # Get initial loss
    x1 = torch.randn(32, 10)
    x2 = torch.randn(32, 10)
    z1_init = model(x1)
    z2_init = model(x2)
    loss_init = loss_fn(z1_init, z2_init)['loss'].item()

    # Train for a few steps
    losses = []
    for _ in range(50):
        optimizer.zero_grad()
        z1 = model(x1)
        z2 = model(x2)
        loss = loss_fn(z1, z2)['loss']
        loss.backward()
        optimizer.step()
        losses.append(loss.item())

    # Loss should decrease overall (allowing small fluctuations)
    final_loss = losses[-1]
    avg_loss_last_10 = np.mean(losses[-10:])

    assert final_loss < loss_init, \
        f"Loss didn't decrease: {loss_init} → {final_loss}"
    assert avg_loss_last_10 < loss_init * 0.8, \
        "Loss should decrease by at least 20%"

def test_vicreg_loss_convergence():
    """Verify VICReg loss decreases with training."""
    # Similar structure but tests VICRegLoss
    ...
```

**Effort**: ~2-3 hours
**Coverage improvement**: +3-5%
**Value**: Catches real bugs (training doesn't actually improve model)

---

### 8. Add Data Pipeline Integration Tests

**New test file**: `tests/integration/test_data_pipeline.py`

```python
"""Test end-to-end data loading and preprocessing."""

def test_dataset_to_dataloader_pipeline():
    """Verify data flows correctly through pipeline."""
    # Create sample data
    text_data = torch.randn(100, 10)
    image_data = torch.randn(100, 3, 32, 32)

    # Create dataset
    dataset = MultimodalDataset({
        'text': text_data,
        'image': image_data,
    })

    # Create dataloader
    dataloader = create_dataloader(dataset, batch_size=16)

    # Verify batches are correct
    batch = next(iter(dataloader))

    # Batch structure
    assert 'text' in batch and 'image' in batch
    assert batch['text'].shape[0] == 16  # Batch size
    assert batch['image'].shape[0] == 16

    # Data integrity
    assert torch.allclose(batch['text'][0], text_data[0])
    assert torch.allclose(batch['image'][0], image_data[0])

def test_train_val_test_split_integrity():
    """Verify splits don't overlap."""
    data = torch.randn(1000, 10)
    train, val, test = split_data(data)

    train_indices = set(range(len(train)))
    val_indices = set(range(len(train), len(train) + len(val)))
    test_indices = set(range(len(train) + len(val), 1000))

    # No overlap
    assert len(train_indices & val_indices) == 0
    assert len(val_indices & test_indices) == 0
    assert len(train_indices & test_indices) == 0

    # All data used
    assert len(train_indices | val_indices | test_indices) == 1000
```

**Effort**: ~3-4 hours
**Coverage improvement**: +5-7%

---

## Phase 4: Test Infrastructure Improvements (Days 13-15)

### 9. Create Test Configuration Matrix

**New file**: `tests/conftest.py` improvements

```python
"""Global test configuration."""

import pytest
import torch

@pytest.fixture(scope='session')
def seed():
    """Set random seed for reproducibility."""
    torch.manual_seed(42)
    np.random.seed(42)
    return 42

@pytest.fixture(params=['cpu'], ids=['cpu'])
def device(request):
    """Parametrize tests across devices."""
    if request.param == 'cuda' and not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    return torch.device(request.param)

@pytest.fixture
def minimal_batch():
    """Minimal batch for quick tests."""
    return {
        'batch_size': 2,
        'embed_dim': 64,
        'seq_length': 10,
    }

@pytest.fixture
def standard_batch():
    """Standard batch for normal tests."""
    return {
        'batch_size': 16,
        'embed_dim': 128,
        'seq_length': 64,
    }

@pytest.fixture
def large_batch():
    """Large batch for stress tests."""
    return {
        'batch_size': 256,
        'embed_dim': 512,
        'seq_length': 512,
    }
```

---

### 10. Add Test Markers for Organization

```python
# Mark tests by category and priority
@pytest.mark.unit
@pytest.mark.critical
def test_core_functionality():
    """Test critical code path."""
    pass

@pytest.mark.integration
@pytest.mark.slow
def test_end_to_end():
    """Test full training loop."""
    pass

# Run only critical tests: pytest -m "critical"
# Skip slow tests: pytest -m "not slow"
```

---

## Phase 5: Test Quality Standards (Days 16-17)

### Establish Minimum Requirements

**Per-test requirements**:
1. ✅ Clear name explaining what's tested
2. ✅ Docstring with purpose
3. ✅ At least 2 meaningful assertions (not "doesn't crash")
4. ✅ Tests expected behavior, not implementation

**Coverage targets by category**:
- Core utilities (20+ files): 50%+ coverage
- Data loading (10+ files): 60%+ coverage
- Model components (15+ files): 60%+ coverage
- Loss functions (25+ files): 70%+ coverage
- Training loops (10+ files): 40%+ coverage
- Safety/CAI (12+ files): 80%+ coverage

**Assertion requirements**:
```python
# BAD - No real assertions
def test_forward():
    loss = fn(x, y)  # ← Just runs, no verification

# BETTER - Has assertions but weak
def test_forward():
    loss = fn(x, y)
    assert isinstance(loss, torch.Tensor)

# GOOD - Tests actual behavior
def test_forward():
    loss = fn(x, y)
    assert loss > 0, "Loss must be positive"
    assert not torch.isnan(loss), "Loss must be valid"
    assert loss.requires_grad, "Loss must be trainable"

# EXCELLENT - Tests behavior and properties
def test_forward_convergence():
    x1, x2 = get_test_data()
    loss_initial = fn(x1, x2)

    # Train one step
    loss_initial.backward()
    # Verify learning signal
    assert x1.grad is not None
    assert torch.any(x1.grad != 0)

    # Better alignment should give lower loss
    x1_aligned = improve_alignment(x1)
    loss_aligned = fn(x1_aligned, x2)
    assert loss_aligned < loss_initial
```

---

## Implementation Schedule

| Phase | Duration | Tasks | Expected Result |
|-------|----------|-------|-----------------|
| Quick Wins | 1-2 days | Deps, configs, markers | 20-30 tests passing |
| Fix Broken | 3-5 days | PPO, reward model | +20-30 tests passing |
| Strengthen | 5-8 days | Improve assertions | +10-15% coverage |
| Integration | 4-8 days | Add convergence/pipeline | +8-12% coverage |
| Infrastructure | 2-3 days | Config matrix, markers | Better organization |
| Standards | 1-2 days | Document requirements | Quality baseline |

**Total effort**: ~3-4 weeks
**Expected result**: 500-600 high-quality tests, 50%+ coverage

---

## Metrics to Track

### Coverage
```bash
python -m pytest tests/ --cov=src --cov-report=term-missing \
  | grep "TOTAL"

# Current: TOTAL 36%
# Week 1: TOTAL 38% (quick wins)
# Week 2: TOTAL 40-42% (fix broken tests)
# Week 3: TOTAL 45-48% (strengthen + integration)
# Week 4: TOTAL 50%+ (complete)
```

### Test Quality
```python
# Track assertion types
def count_assertions():
    weak = len(re.findall(r'isinstance|\.shape|is not None', test_code))
    strong = len(re.findall(r'backward|\.grad|allclose|decrease', test_code))
    return weak, strong

# Current ratio: ~70% weak, 30% strong
# Target ratio: ~30% weak, 70% strong
```

### Passing vs Meaningful
```bash
# Current: 739 passing, 36% coverage
# Target: 550 passing, 50%+ coverage

# This means removing ~190 trivial tests
# And ensuring remaining tests are meaningful
```

---

## Critical Success Factors

1. **Fix the 51 failing/erroring tests first** - they indicate test setup issues
2. **Don't just add more tests** - remove or fix weak ones
3. **Involve code authors** - they know what should be tested
4. **Test behavior, not implementation** - avoid brittle tests
5. **Focus on high-value tests** - loss convergence > shape checking

---

## Completion Checklist

- [ ] All 843 tests collect successfully
- [ ] Failed tests reduced to <5 (all legitimate)
- [ ] Coverage increased to 45%+
- [ ] Average 3+ assertions per test (meaningful)
- [ ] Integration tests cover critical paths
- [ ] Convergence tests verify training works
- [ ] Test documentation updated
- [ ] CI/CD pipeline configured
