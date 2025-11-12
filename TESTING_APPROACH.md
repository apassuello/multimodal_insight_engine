# Testing Approach for ML Loss Functions

## Philosophy: Test the Contract, Not the Implementation

For ML loss functions, we focus on testing **mathematical properties** and **behavioral contracts**, not internal implementation details.

## 7-Layer Testing Strategy

### Layer 1: Shape & Type Validation (Sanity Checks)
```python
def test_basic_forward(self, vision_features, text_features):
    loss = loss_fn(vision_features, text_features)

    # Shape: Loss should be a scalar
    assert isinstance(loss, torch.Tensor)
    assert loss.shape == torch.Size([])  # Scalar

    # Validity: No NaN or Inf
    assert not torch.isnan(loss)
    assert not torch.isinf(loss)

    # Range: Loss should be non-negative
    assert loss.item() >= 0
```

**Purpose:** Catches basic bugs like returning wrong shape or broken computation.

---

### Layer 2: Gradient Flow Verification (Critical for Training)
```python
def test_gradient_flow(self, features_a, features_b):
    features_a = features_a.requires_grad_(True)

    loss = loss_fn(features_a, features_b)
    loss.backward()

    # Gradients must exist
    assert features_a.grad is not None

    # Gradients must be non-zero (learning signal)
    assert not torch.all(features_a.grad == 0)
```

**Purpose:** If gradients don't flow, training silently fails. This prevents "model trains for days with zero learning" scenarios.

---

### Layer 3: Hyperparameter Sensitivity (Behavioral Validation)
```python
def test_coefficient_effects(self):
    # High coefficient
    loss_high = VICRegLoss(sim_coeff=50.0)(embeddings_a, embeddings_b)

    # Low coefficient
    loss_low = VICRegLoss(sim_coeff=1.0)(embeddings_a, embeddings_b)

    # Must be different
    assert not torch.allclose(loss_high, loss_low)
```

**Purpose:** Verifies hyperparameters actually affect the loss (not just ignored).

---

### Layer 4: Edge Case Handling (Robustness)
```python
def test_edge_case_single_sample(self):
    # Batch size = 1 (common failure point)
    features = torch.randn(1, embed_dim)
    loss = loss_fn(features, features)
    assert not torch.isnan(loss)

def test_edge_case_identical_features(self):
    # Perfect alignment case
    features = torch.randn(batch_size, embed_dim)
    loss = loss_fn(features, features.clone())
    # Loss should be low but valid
    assert not torch.isnan(loss)
```

**Purpose:** Edge cases break in production. Single samples happen in eval, identical features test numerical stability.

**Common Edge Cases to Test:**
- Single sample batches (batch_size=1)
- Identical features (perfect alignment)
- Zero/empty inputs
- Very small/large batch sizes
- Extreme values (outliers)

---

### Layer 5: Numerical Stability (Production Safety)
```python
def test_numerical_stability_extreme_values(self):
    # Very large values (can cause overflow)
    features_large = torch.ones(batch_size, embed_dim) * 100
    loss = loss_fn(features_large, features_large)

    assert not torch.isnan(loss)
    assert not torch.isinf(loss)
```

**Purpose:** Real data has outliers. Prevents models from exploding on extreme inputs.

---

### Layer 6: Mathematical Properties (Correctness)

**For VICReg - Test the three components:**
```python
def test_loss_components(self):
    result = vicreg_loss(embeddings_a, embeddings_b)

    # Should return variance, invariance, covariance terms
    assert 'loss' in result
    # Verify all three components contribute
    if 'sim_loss' in result:
        assert isinstance(result['sim_loss'], torch.Tensor)
```

**For Barlow Twins - Test correlation properties:**
```python
def test_edge_case_identical_embeddings(self):
    # With identical embeddings, cross-correlation matrix
    # should be identity → low loss
    loss = barlow_twins(embeddings, embeddings.clone())
    assert loss.item() >= 0  # Valid but should be low
```

**For Contrastive Losses - Test temperature effects:**
```python
def test_temperature_sensitivity(self):
    loss_low_temp = ContrastiveLoss(temperature=0.01)(v, t)
    loss_high_temp = ContrastiveLoss(temperature=1.0)(v, t)

    # Different temperatures should lead to different losses
    assert loss_low_temp != loss_high_temp
```

**Purpose:** Tests the actual mathematical behavior, not just "does it run?"

---

### Layer 7: Integration Testing (Real-World Simulation)
```python
def test_combined_training_simulation(self):
    model = nn.Sequential(nn.Linear(128, 128), nn.ReLU())
    optimizer = torch.optim.Adam(model.parameters())

    # Forward pass
    z1 = model(x1)
    z2 = model(x2)

    # Compute loss
    loss = vicreg_loss(z1, z2)

    # Backward pass
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # Verify training step succeeded
    assert not torch.isnan(loss)
    for param in model.parameters():
        assert param.grad is not None
```

**Purpose:** Tests loss in actual training loop context—catches integration bugs.

---

## Testing Patterns & Best Practices

### Use Fixtures for Reusability
```python
@pytest.fixture
def device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

@pytest.fixture
def batch_size():
    return 16

@pytest.fixture
def vision_features(batch_size, embed_dim, device):
    return torch.randn(batch_size, embed_dim, device=device)
```

### Handle Multiple Return Types
```python
# Some losses return dict, others return tensor
result = loss_fn(features_a, features_b)
loss = result['loss'] if isinstance(result, dict) else result
```

### Use Skip Markers for Optional Imports
```python
@pytest.mark.skipif(DecorrelationLoss is None, reason="DecorrelationLoss not available")
class TestDecorrelationLoss:
    # Tests here
```

### Test Reduction Modes
```python
def test_reduction_modes(self):
    # Mean (scalar)
    loss_mean = loss_fn(reduction='mean')(features_a, features_b)
    assert loss_mean.shape == torch.Size([])

    # None (per-sample)
    loss_none = loss_fn(reduction='none')(features_a, features_b)
    assert loss_none.ndim >= 1
```

---

## Key Principles

1. **✅ Test behavior, not implementation**
   - Don't test internal variable names or private methods
   - Test the public API and expected behavior

2. **✅ Test edge cases, not just happy path**
   - Single samples, empty inputs, extreme values
   - These break in production, catch them in tests

3. **✅ Test properties (gradients, shapes), not exact values**
   - Loss values are stochastic (random data)
   - Test that gradients exist, not their exact values

4. **✅ Make tests fail fast and clearly**
   - Descriptive assertion messages
   - Test one thing per test function

5. **✅ Tests should be deterministic**
   - Use fixed random seeds when needed
   - No flaky tests allowed

6. **✅ Tests should be independent**
   - No shared state between tests
   - Tests can run in any order

---

## What NOT to Test

❌ **Don't test PyTorch internals**
```python
# Bad
assert loss.grad_fn.__class__.__name__ == 'SomeGradFunction'

# Good
assert loss.requires_grad
```

❌ **Don't test exact loss values**
```python
# Bad
assert loss.item() == 0.123456

# Good
assert 0 <= loss.item() < 10  # Reasonable range
```

❌ **Don't test implementation details**
```python
# Bad
assert hasattr(loss_fn, '_internal_buffer')

# Good
loss = loss_fn(features_a, features_b)
assert not torch.isnan(loss)
```

---

## Confidence Level Guidelines

When writing tests without being able to run them:

**HIGH Confidence (95%+):**
- Copied patterns from existing tests
- Read actual implementation code
- Used defensive programming (skip markers, type checks)
- Follows pytest best practices

**MEDIUM Confidence (70-90%):**
- Made assumptions about API signatures
- Didn't verify return types
- No existing test patterns to follow

**LOW Confidence (<70%):**
- Guessing at API signatures
- No documentation available
- Complex internal behavior

---

## Test Coverage Targets

For ML loss functions:

| Component | Target Coverage | Priority |
|-----------|----------------|----------|
| Loss Functions | 70%+ | 🔥 Critical |
| Dataset Loaders | 50%+ | 🛡️ High (Security) |
| Training Utilities | 40%+ | 🔧 Medium |
| Trainers (Smoke Tests) | 25%+ | 🏃 Medium |

**Note:** 100% coverage is not the goal. Focus on:
- Critical paths (loss computation, data loading)
- Security-relevant code (data validation, user inputs)
- Complex logic (gradient operations, distributed training)

---

## Running Tests

```bash
# Run specific test file
python -m pytest tests/test_contrastive_losses.py -v

# Run with coverage
python -m pytest tests/test_contrastive_losses.py \
    --cov=src/training/losses \
    --cov-report=term-missing \
    -v

# Run all tests
./run_tests.sh
```

---

## Debugging Failed Tests

When tests fail:

1. **Read the error message carefully**
   - TypeError: API signature mismatch
   - AssertionError: Expected behavior not met
   - AttributeError: Missing attribute/method

2. **Check the actual vs expected**
   ```python
   # Add debugging
   print(f"Loss shape: {loss.shape}")
   print(f"Loss value: {loss.item()}")
   print(f"Loss type: {type(loss)}")
   ```

3. **Isolate the problem**
   ```bash
   # Run single test
   python -m pytest tests/test_file.py::TestClass::test_function -v
   ```

4. **Common fixes**
   - Adjust API signature (add/remove parameters)
   - Handle different return types (dict vs tensor)
   - Adjust assertions (too strict/too loose)

---

## References

- **Existing test patterns:** `tests/test_losses.py`
- **Loss implementations:** `src/training/losses/`
- **Pytest documentation:** https://docs.pytest.org/
- **PyTorch testing guide:** https://pytorch.org/docs/stable/testing.html
