# Priority Test Fixes and Improvements

**Quick reference guide for test improvements**

---

## CRITICAL - Fix First (Day 1)

### 1. Install Missing Dependencies
**Status**: BLOCKING 15+ tests

```bash
# Install missing dependencies
pip install nltk
# Verify
python -m pytest tests/test_training_metrics.py --co -q
```

**Impact**: Unblocks test collection immediately
**Time**: 5 minutes

---

### 2. Fix HybridPretrainVICRegLoss Test Configuration
**File**: `tests/test_selfsupervised_losses.py` (lines ~440-490)

**Problem**: Test provides 128-dim embeddings but loss expects 768-dim

**Current code**:
```python
def test_basic_forward(self, embeddings_a, embeddings_b, device):
    """Test basic forward pass."""
    loss_fn = HybridPretrainVICRegLoss(
        sim_coeff=10.0, var_coeff=5.0, cov_coeff=1.0
    )
    result = loss_fn(embeddings_a, embeddings_b)  # embeddings_a: [16, 128]
    # ❌ RuntimeError: mat1 and mat2 shapes cannot be multiplied (16x128 and 768x512)
```

**Fix**:
```python
def test_basic_forward(self, batch_size, device):
    """Test basic forward pass."""
    # Create embeddings with correct dimension
    embed_dim = 128
    embeddings_a = torch.randn(batch_size, embed_dim, device=device)
    embeddings_b = torch.randn(batch_size, embed_dim, device=device)

    # Configure loss with matching dimensions
    loss_fn = HybridPretrainVICRegLoss(
        sim_coeff=10.0,
        var_coeff=5.0,
        cov_coeff=1.0,
        embed_dim=embed_dim,  # ← Add this
        projection_dim=256,   # ← Add this
    )

    result = loss_fn(embeddings_a, embeddings_b)
    loss = extract_loss(result)

    # Verify loss is valid
    assert not torch.isnan(loss)
    assert not torch.isinf(loss)
    assert loss.item() >= 0

    # Verify gradients flow
    embeddings_a_grad = embeddings_a.requires_grad_(True)
    result = loss_fn(embeddings_a_grad, embeddings_b)
    loss = extract_loss(result)
    loss.backward()
    assert embeddings_a_grad.grad is not None
```

**Tests fixed**:
- TestHybridPretrainVICRegLoss::test_basic_forward
- TestHybridPretrainVICRegLoss::test_gradient_flow
- TestHybridPretrainVICRegLoss::test_hybrid_components
- TestHybridPretrainVICRegLoss::test_numerical_stability

**Time**: 30 minutes

---

## HIGH PRIORITY - Fix This Week (Days 2-3)

### 3. Fix PPO Trainer Test Fixtures
**File**: `tests/test_ppo_trainer.py` (20+ failing tests)

**Problem**: Tests assume certain imports and fixtures but don't properly mock them

**Fix**:
```python
# Add at top of test file
import pytest
from unittest.mock import MagicMock, patch
import numpy as np
from gym import spaces

# Create proper fixtures
@pytest.fixture
def dummy_model():
    """Create a simple model for testing."""
    return torch.nn.Sequential(
        torch.nn.Linear(4, 64),
        torch.nn.ReLU(),
        torch.nn.Linear(64, 2),  # 2 actions
    )

@pytest.fixture
def dummy_reward_model():
    """Create a simple reward model for testing."""
    model = MagicMock()
    model.get_reward.return_value = torch.tensor([0.5])
    return model

@pytest.fixture
def test_config():
    """Configuration for tests."""
    return {
        'learning_rate': 1e-3,
        'gamma': 0.99,
        'gae_lambda': 0.95,
        'batch_size': 16,
        'n_epochs': 3,
        'clip_ratio': 0.2,
    }

# Use in tests
def test_train_step_completes(dummy_model, dummy_reward_model, test_config):
    """Test that training step completes without error."""
    trainer = PPOTrainer(
        policy_model=dummy_model,
        reward_model=dummy_reward_model,
        **test_config
    )

    # Create dummy batch
    observations = torch.randn(16, 4)
    actions = torch.randint(0, 2, (16,))
    rewards = torch.randn(16)
    old_log_probs = torch.randn(16)

    # Should complete without error
    loss = trainer.train_step(
        observations, actions, rewards, old_log_probs
    )

    assert isinstance(loss, torch.Tensor)
    assert loss.item() > 0
```

**Expected recovery**: +10-15 tests passing
**Time**: 2-3 hours

---

### 4. Fix Reward Model Test Setup
**File**: `tests/test_reward_model.py` (20+ errors)

**Problem**: Tests don't properly initialize model dependencies

**Fix**:
```python
@pytest.fixture
def sample_texts():
    """Sample text data for testing."""
    return [
        "This is a good response.",
        "This is a bad response.",
        "This is a neutral response.",
    ] * 4  # 12 samples

@pytest.fixture
def sample_preferences():
    """Preference pairs for testing."""
    return [
        ("Good response", "Bad response", 1),
        ("Good response", "Neutral response", 1),
        ("Neutral response", "Bad response", 0.5),
    ] * 4  # 12 triplets

@pytest.fixture
def dummy_tokenizer():
    """Dummy tokenizer for tests."""
    tokenizer = MagicMock()
    tokenizer.encode.return_value = [1, 2, 3, 4, 5]
    tokenizer.vocab_size = 5000
    tokenizer.pad_token_id = 0
    return tokenizer

@pytest.fixture
def reward_model_small(dummy_tokenizer):
    """Small reward model for testing."""
    model = RewardModel(
        model_name='gpt2',
        tokenizer=dummy_tokenizer,
        hidden_size=256,
        num_labels=1,
    )
    return model.to('cpu')

# Use in tests
def test_forward_pass_shape(reward_model_small, sample_texts):
    """Test forward pass produces correct shape."""
    # Prepare input
    input_ids = torch.randint(0, 1000, (3, 20))
    attention_mask = torch.ones(3, 20)

    # Forward pass
    output = reward_model_small(
        input_ids=input_ids,
        attention_mask=attention_mask,
    )

    # Verify shape
    assert output.shape == torch.Size([3, 1])  # batch_size x num_labels
```

**Expected recovery**: +10-15 tests passing
**Time**: 2-3 hours

---

## MEDIUM PRIORITY - Fix This Week (Days 4-5)

### 5. Strengthen Contrastive Loss Assertions
**File**: `tests/test_contrastive_losses.py` (26 tests with weak assertions)

**Current pattern** (WEAK):
```python
def test_temperature_sensitivity(self):
    loss_low = loss_fn_low(vision_features, text_features)
    loss_high = loss_fn_high(vision_features, text_features)

    assert loss_low != loss_high           # ← Weak
    assert not torch.isnan(loss_low)       # ← Weak
    assert not torch.isnan(loss_high)      # ← Weak
```

**Improved pattern** (STRONG):
```python
def test_temperature_sensitivity(self):
    """Temperature should affect loss magnitude inversely."""
    # Low temperature = sharper softmax = higher loss for random alignment
    loss_low = loss_fn_low(vision_features, text_features)
    loss_high = loss_fn_high(vision_features, text_features)

    # Verify relationship
    assert loss_low.item() > loss_high.item(), \
        f"Low temp loss {loss_low} should be > high temp loss {loss_high}"

    # Verify valid range
    assert 0 < loss_low.item() < 10
    assert 0 < loss_high.item() < 10

def test_alignment_reduces_loss(self):
    """Aligned features should have lower loss."""
    # Random alignment
    loss_random = loss_fn(vision_features, text_features)

    # Perfect alignment
    loss_aligned = loss_fn(vision_features, vision_features)

    # Aligned should have lower loss
    assert loss_aligned < loss_random, \
        f"Aligned loss {loss_aligned} should be < random {loss_random}"

def test_batch_independence(self):
    """Loss shouldn't depend on batch size in relative terms."""
    loss_small = loss_fn(vision_features[:4], text_features[:4])
    loss_large = loss_fn(vision_features, text_features)

    # Both should be in valid range
    assert 0 < loss_small.item() < 10
    assert 0 < loss_large.item() < 10
    # Relative scales should be similar
    ratio = loss_small.item() / max(loss_large.item(), 1e-6)
    assert 0.1 < ratio < 10, "Loss scales shouldn't differ wildly"
```

**Effort**: 4-5 hours (20+ tests to improve)
**Impact**: +5-8% coverage, much stronger assertions

---

### 6. Add Core Convergence Tests
**New file**: `tests/test_loss_convergence.py`

**Purpose**: Verify that losses actually decrease during training (catch real bugs)

**Tests to add**:
```python
def test_contrastive_loss_convergence():
    """Verify contrastive loss decreases with training."""
    model = nn.Sequential(nn.Linear(10, 128), nn.ReLU())
    loss_fn = ContrastiveLoss(temperature=0.07)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.1)

    x1 = torch.randn(32, 10)
    x2 = torch.randn(32, 10)

    # Initial loss
    z1 = model(x1)
    z2 = model(x2)
    loss_init = loss_fn(z1, z2)['loss'].item()

    # Train for 50 steps
    losses = []
    for _ in range(50):
        optimizer.zero_grad()
        z1 = model(x1)
        z2 = model(x2)
        loss = loss_fn(z1, z2)['loss']
        loss.backward()
        optimizer.step()
        losses.append(loss.item())

    # Loss should decrease
    loss_final = losses[-1]
    assert loss_final < loss_init, \
        f"Loss didn't decrease: {loss_init:.4f} → {loss_final:.4f}"

    # Smooth decrease (not just lucky last step)
    avg_last_10 = np.mean(losses[-10:])
    assert avg_last_10 < loss_init * 0.8

def test_vicreg_loss_convergence():
    """Verify VICReg loss decreases with training."""
    # Similar structure
    ...

def test_gradient_flow_all_losses():
    """Verify all loss functions have proper gradient flow."""
    losses_to_test = [
        ContrastiveLoss(),
        VICRegLoss(),
        BarlowTwinsLoss(),
    ]

    for loss_fn in losses_to_test:
        x1 = torch.randn(16, 128, requires_grad=True)
        x2 = torch.randn(16, 128)

        loss = extract_loss(loss_fn(x1, x2))
        loss.backward()

        assert x1.grad is not None
        assert torch.any(x1.grad != 0), f"{loss_fn} has zero gradients!"
```

**Effort**: 2-3 hours
**Impact**: +3-5% coverage, catches training issues

---

## RECOMMENDED ORDER

**Day 1** (30 min - 1 hour):
1. Install nltk
2. Fix HybridPretrainVICRegLoss configuration

**Days 2-3** (3-4 hours):
3. Fix PPO trainer fixtures
4. Fix reward model fixtures

**Days 4-5** (5-6 hours):
5. Strengthen contrastive loss assertions
6. Add convergence tests

**Expected result after Day 5**:
- 31 failed tests → 5-10 failed tests
- 20 errors → 0-5 errors
- 36% coverage → 42-45% coverage
- All critical paths fixed

---

## Verification Commands

```bash
# Check progress
python -m pytest tests/ --ignore=tests/test_training_metrics.py \
  --tb=short -q 2>&1 | tail -20

# Run specific fixed tests
python -m pytest tests/test_selfsupervised_losses.py::TestHybridPretrainVICRegLoss -v
python -m pytest tests/test_ppo_trainer.py::TestComputeGAE::test_gae_basic_computation -v
python -m pytest tests/test_loss_convergence.py -v

# Check coverage of critical modules
python -m pytest tests/ --cov=src/training/losses \
  --cov-report=term-missing -q
```

---

## Files Modified Checklist

Use this to track progress:

- [ ] Install nltk dependency
- [ ] Update test_selfsupervised_losses.py (4 tests)
- [ ] Update test_ppo_trainer.py (fixtures)
- [ ] Update test_reward_model.py (fixtures)
- [ ] Strengthen test_contrastive_losses.py assertions
- [ ] Create test_loss_convergence.py
- [ ] Register pytest markers in pytest.ini
- [ ] Update test documentation

---

## Success Criteria

After implementing these fixes:

- ✅ All critical tests run without collection errors
- ✅ Failed tests reduced from 31 to <10
- ✅ Errors reduced from 20 to <5
- ✅ Coverage increased from 36% to 42%+
- ✅ Assertions in loss tests are meaningful
- ✅ Convergence tests catch real bugs
- ✅ All critical code paths have tests

**Expected time commitment**: 8-10 hours spread over 5 days

---

## Long-term Improvements

After immediate fixes, continue with:

1. **Data pipeline tests** - Verify no data leakage, correct splits
2. **Integration tests** - Full training loops (small scale)
3. **Performance tests** - Benchmark loss computation, memory usage
4. **Mutation testing** - Verify test effectiveness
5. **Fuzzing** - Test robustness to invalid inputs

See `TEST_IMPROVEMENT_ROADMAP.md` for detailed long-term plan.
