# Testing Patterns & Best Practices
## MultiModal Insight Engine

---

## 1. Loss Function Testing Pattern

### Basic Structure

```python
"""Test module for custom loss functions."""

import pytest
import torch
from src.training.losses import YourLoss

class TestYourLossInitialization:
    """Test loss initialization."""

    def test_init_default(self):
        """Test with default parameters."""
        loss = YourLoss()
        assert loss.param1 == default_value

    def test_init_custom(self):
        """Test with custom parameters."""
        loss = YourLoss(param1=custom_value)
        assert loss.param1 == custom_value

class TestYourLossForward:
    """Test forward pass."""

    def test_forward_output_shape(self):
        """Test output tensor shape."""
        loss = YourLoss()
        z_a = torch.randn(8, 256)
        z_b = torch.randn(8, 256)

        output = loss(z_a, z_b)

        # For single scalar loss
        assert output.shape == torch.Size([])

        # For dict output (common in this project)
        assert "loss" in output
        assert isinstance(output["loss"], torch.Tensor)

    def test_forward_is_differentiable(self):
        """Test gradients flow through loss."""
        loss = YourLoss()
        z_a = torch.randn(8, 256, requires_grad=True)
        z_b = torch.randn(8, 256, requires_grad=True)

        output = loss(z_a, z_b)

        # Backward for scalar
        if output.dim() == 0:
            output.backward()
        # Backward for dict
        else:
            output["loss"].backward()

        assert z_a.grad is not None
        assert z_b.grad is not None
        assert not torch.all(z_a.grad == 0)

class TestYourLossComponents:
    """Test individual loss components."""

    def test_component_x_behavior(self):
        """Test specific component behavior."""
        # If loss has multiple components, test each
        loss = YourLoss(component_weight=1.0, other_weight=0.0)

        z_a = torch.randn(8, 256)
        z_b = torch.randn(8, 256)
        output = loss(z_a, z_b)

        assert "component_x" in output

class TestYourLossNumericalStability:
    """Test stability with edge cases."""

    @pytest.mark.parametrize("scale", [1e-6, 1e-3, 1e3, 1e6])
    def test_numerical_stability(self, scale):
        """Test with various scales."""
        loss = YourLoss()
        z_a = torch.randn(8, 256) * scale
        z_b = torch.randn(8, 256) * scale

        output = loss(z_a, z_b)

        assert not torch.isnan(output["loss"])
        assert not torch.isinf(output["loss"])

class TestYourLossBatchSizes:
    """Test with different batch sizes."""

    @pytest.mark.parametrize("batch_size", [1, 2, 8, 16, 32])
    def test_batch_size(self, batch_size):
        """Loss should work with any batch size."""
        loss = YourLoss()
        z_a = torch.randn(batch_size, 256)
        z_b = torch.randn(batch_size, 256)

        output = loss(z_a, z_b)
        assert not torch.isnan(output["loss"])
```

### Checklist for Loss Function Tests

- [ ] Initialization with default and custom parameters
- [ ] Output shape validation (scalar vs dict vs other)
- [ ] Gradient flow (backward pass computes gradients)
- [ ] Component-specific tests (if multi-component)
- [ ] Numerical stability (extreme values, scales)
- [ ] Batch size variations (1 to 32+)
- [ ] Embedding dimension variations
- [ ] Zero/constant embeddings
- [ ] Identical embeddings behavior
- [ ] Temperature/scaling effects
- [ ] Reduction modes (if applicable)
- [ ] Weight effects (component weighting)

---

## 2. Trainer Testing Pattern

### Basic Structure

```python
"""Test module for custom trainers."""

import pytest
import torch
from src.training.trainers import YourTrainer

@pytest.fixture
def trainer_config():
    """Minimal valid configuration."""
    return {
        'learning_rate': 0.001,
        'epochs': 2,
        'batch_size': 8,
    }

@pytest.fixture
def mock_model():
    """Simple model for testing."""
    return torch.nn.Sequential(
        torch.nn.Linear(64, 128),
        torch.nn.ReLU(),
        torch.nn.Linear(128, 10)
    )

@pytest.fixture
def mock_dataloader():
    """Simple dataloader for testing."""
    dataset = torch.utils.data.TensorDataset(
        torch.randn(32, 64),
        torch.randint(0, 10, (32,))
    )
    return torch.utils.data.DataLoader(dataset, batch_size=8)

class TestYourTrainerInitialization:
    """Test trainer initialization."""

    def test_init_default(self, mock_model, trainer_config):
        """Test with default configuration."""
        trainer = YourTrainer(model=mock_model)
        assert trainer.model is mock_model

    def test_init_custom(self, mock_model, trainer_config):
        """Test with custom configuration."""
        trainer = YourTrainer(
            model=mock_model,
            **trainer_config
        )
        assert trainer.learning_rate == 0.001

    def test_init_invalid_config(self, mock_model):
        """Test that invalid config raises error."""
        with pytest.raises((ValueError, KeyError)):
            YourTrainer(
                model=mock_model,
                invalid_param=123
            )

class TestYourTrainerTrainingStep:
    """Test training loop."""

    def test_train_step_completes(self, mock_model, mock_dataloader):
        """Training step should complete."""
        trainer = YourTrainer(model=mock_model)
        batch = next(iter(mock_dataloader))

        # Should not raise
        metrics = trainer.training_step(batch)
        assert "loss" in metrics

    def test_train_step_updates_weights(self, mock_model, mock_dataloader):
        """Training should update model weights."""
        trainer = YourTrainer(model=mock_model)
        batch = next(iter(mock_dataloader))

        # Get initial weights
        initial_weights = [
            p.clone() for p in mock_model.parameters()
        ]

        # Train
        trainer.training_step(batch)

        # Check changed
        for init, curr in zip(initial_weights, mock_model.parameters()):
            if init.requires_grad:
                assert not torch.allclose(init, curr)

class TestYourTrainerValidation:
    """Test validation."""

    def test_validation_no_grad(self, mock_model, mock_dataloader):
        """Validation should not compute gradients."""
        trainer = YourTrainer(model=mock_model)
        batch = next(iter(mock_dataloader))

        # Enable grad tracking
        mock_model.train()

        with torch.no_grad():
            metrics = trainer.validation_step(batch)

        assert isinstance(metrics, dict)

class TestYourTrainerCheckpointing:
    """Test save/load."""

    def test_save_checkpoint(self, mock_model, tmp_path):
        """Test saving checkpoint."""
        trainer = YourTrainer(model=mock_model)
        path = tmp_path / "checkpoint.pt"

        trainer.save_checkpoint(str(path))

        assert path.exists()

    def test_load_checkpoint(self, mock_model, tmp_path):
        """Test loading checkpoint."""
        trainer = YourTrainer(model=mock_model)
        path = tmp_path / "checkpoint.pt"

        # Save
        trainer.save_checkpoint(str(path))

        # Load
        trainer2 = YourTrainer(model=mock_model)
        trainer2.load_checkpoint(str(path))

        # Verify state restored
        for p1, p2 in zip(
            trainer.model.parameters(),
            trainer2.model.parameters()
        ):
            assert torch.allclose(p1, p2)
```

### Checklist for Trainer Tests

- [ ] Initialization with default and custom config
- [ ] Training step completes without error
- [ ] Training updates model parameters
- [ ] Validation works without computing gradients
- [ ] Checkpoint save/load cycle
- [ ] Learning rate scheduler integration
- [ ] Early stopping logic
- [ ] Gradient clipping (if used)
- [ ] Multiple epochs training
- [ ] Metrics tracking and logging
- [ ] Device handling (CPU/GPU)
- [ ] Distributed training (if applicable)

---

## 3. Multimodal Component Testing Pattern

### Basic Structure

```python
"""Test multimodal components."""

import pytest
import torch

class TestVisionLanguageAlignment:
    """Test vision-language alignment components."""

    def test_alignment_identical_modalities(self):
        """Identical modality embeddings should align perfectly."""
        # Create same embedding for both modalities
        embedding = torch.randn(8, 256)

        alignment = compute_alignment(
            vision_embedding=embedding,
            text_embedding=embedding
        )

        # Should be perfectly aligned
        assert alignment > 0.95

    def test_alignment_orthogonal_modalities(self):
        """Orthogonal embeddings should have low alignment."""
        # Create orthogonal embeddings
        v_emb = torch.eye(256)[:8]
        t_emb = torch.eye(256)[8:16]

        alignment = compute_alignment(v_emb, t_emb)

        # Should be poorly aligned
        assert alignment < 0.1

    def test_gradient_flow_through_alignment(self):
        """Gradients should flow through alignment."""
        v_emb = torch.randn(8, 256, requires_grad=True)
        t_emb = torch.randn(8, 256, requires_grad=True)

        loss = compute_alignment(v_emb, t_emb)
        loss.backward()

        assert v_emb.grad is not None
        assert t_emb.grad is not None
```

---

## 4. Constitutional AI Testing Pattern

### Basic Structure

```python
"""Test Constitutional AI components."""

import pytest
import torch

class TestPrincipleEvaluation:
    """Test principle evaluation."""

    def test_principle_violation_detection(self):
        """Test detection of principle violations."""
        evaluator = PrincipleEvaluator()

        # Text that violates principle
        bad_text = "harmful_content"

        score = evaluator.evaluate(bad_text)

        # Should have high violation score
        assert score > 0.7

    def test_principle_compliance_detection(self):
        """Test detection of compliant text."""
        evaluator = PrincipleEvaluator()

        # Safe text
        safe_text = "helpful information"

        score = evaluator.evaluate(safe_text)

        # Should have low violation score
        assert score < 0.3
```

---

## 5. Data Pipeline Testing Pattern

### Basic Structure

```python
"""Test data loading and preprocessing."""

import pytest
import torch

class TestDataLoader:
    """Test data loading."""

    def test_dataset_length(self):
        """Test dataset returns correct length."""
        dataset = MyDataset(num_samples=100)
        assert len(dataset) == 100

    def test_dataset_sample_shape(self):
        """Test samples have correct shape."""
        dataset = MyDataset()
        sample = dataset[0]

        assert sample['input'].shape == (seq_len,)
        assert sample['label'].shape == ()

    def test_dataloader_batching(self):
        """Test dataloader creates correct batches."""
        dataset = MyDataset(num_samples=100)
        loader = DataLoader(dataset, batch_size=32)

        batch = next(iter(loader))

        assert batch['input'].shape[0] == 32

    def test_dataloader_multiple_epochs(self):
        """Test dataloader works across multiple epochs."""
        dataset = MyDataset(num_samples=100)
        loader = DataLoader(dataset, batch_size=32)

        for epoch in range(3):
            count = 0
            for batch in loader:
                count += batch['input'].shape[0]

            assert count == 100
```

---

## 6. Fixtures and Helper Functions

### Shared Fixtures

```python
# tests/conftest.py

import pytest
import torch
import numpy as np

# Device fixtures
@pytest.fixture(params=["cpu"])
def device(request):
    """Test on CPU (add cuda if available)."""
    return torch.device(request.param)

# Model fixtures
@pytest.fixture
def simple_transformer(device):
    """Small transformer for quick tests."""
    from src.models import Transformer
    config = TransformerConfig(
        vocab_size=1000,
        hidden_dim=64,
        num_layers=2,
    )
    return Transformer(config).to(device).eval()

# Data fixtures
@pytest.fixture
def dummy_batch(device):
    """Dummy batch for testing."""
    return {
        'input_ids': torch.randint(0, 1000, (4, 32)).to(device),
        'attention_mask': torch.ones(4, 32).to(device),
    }

# Seeding
@pytest.fixture(autouse=True)
def seed_everything():
    """Seed everything for reproducibility."""
    torch.manual_seed(42)
    np.random.seed(42)
```

### Helper Functions

```python
# tests/utils.py

import torch

def assert_tensor_valid(tensor, name=""):
    """Assert tensor is valid (no NaN, no Inf)."""
    assert not torch.isnan(tensor).any(), f"NaN in {name}"
    assert not torch.isinf(tensor).any(), f"Inf in {name}"

def assert_gradients_flow(model, loss_tensor):
    """Assert gradients flow through model."""
    loss_tensor.backward()
    for param in model.parameters():
        if param.requires_grad:
            assert param.grad is not None

def compare_tensors(t1, t2, rtol=1e-5, atol=1e-7):
    """Compare two tensors with tolerance."""
    return torch.allclose(t1, t2, rtol=rtol, atol=atol)
```

---

## 7. Parameterized Testing

### Use parametrize for variants

```python
# Test across batch sizes
@pytest.mark.parametrize("batch_size", [1, 8, 16, 32])
def test_loss_batch_sizes(batch_size):
    z_a = torch.randn(batch_size, 256)
    z_b = torch.randn(batch_size, 256)
    loss = VICRegLoss()
    output = loss(z_a, z_b)
    assert not torch.isnan(output["loss"])

# Test across hyperparameters
@pytest.mark.parametrize("temperature", [0.01, 0.07, 0.1, 1.0])
def test_loss_temperature(temperature):
    loss = ContrastiveLoss(temperature=temperature)
    z_a = torch.randn(8, 256)
    z_b = torch.randn(8, 256)
    output = loss(z_a, z_b)
    assert not torch.isnan(output["loss"])

# Cartesian product
@pytest.mark.parametrize("batch_size", [8, 16])
@pytest.mark.parametrize("embedding_dim", [128, 256])
def test_loss_combinations(batch_size, embedding_dim):
    z_a = torch.randn(batch_size, embedding_dim)
    z_b = torch.randn(batch_size, embedding_dim)
    loss = VICRegLoss()
    output = loss(z_a, z_b)
    assert not torch.isnan(output["loss"])
```

---

## 8. Marking Tests

### Use pytest marks for organization

```python
import pytest

# Mark slow tests
@pytest.mark.slow
def test_full_training_pipeline():
    """This takes 30+ seconds."""
    pass

# Mark GPU tests
@pytest.mark.gpu
def test_cuda_specific_behavior():
    """Only runs on GPU."""
    pass

# Mark integration tests
@pytest.mark.integration
def test_end_to_end_system():
    """Tests multiple components together."""
    pass

# Custom marks
@pytest.mark.multimodal
def test_vision_language_alignment():
    """Tests multimodal functionality."""
    pass
```

**Run subsets:**
```bash
# Only fast tests
pytest -m "not slow"

# Only unit tests
pytest -m "not integration"

# Only specific marker
pytest -m gpu

# Multiple markers (OR logic)
pytest -m "integration or slow"
```

---

## 9. Mocking and Isolation

### When to mock

```python
# ✓ DO MOCK: External dependencies
@patch('requests.get')
def test_api_call(mock_get):
    mock_get.return_value.json.return_value = {'result': 'success'}
    assert call_api() == 'success'

# ✓ DO MOCK: File I/O
@patch('builtins.open', create=True)
def test_file_read(mock_open):
    mock_open.return_value.__enter__.return_value.read.return_value = 'data'
    assert read_file() == 'data'

# ✗ DON'T MOCK: Internal functions (test actual behavior)
# BAD:
def test_loss():
    with patch.object(loss, 'forward', return_value=0):
        # This doesn't test anything meaningful
        pass

# GOOD:
def test_loss():
    output = loss.forward(z_a, z_b)
    assert output["loss"] is not None
```

---

## 10. Common Assertions

### PyTorch-specific assertions

```python
import torch

# Tensor properties
assert tensor.shape == (8, 256)
assert tensor.dtype == torch.float32
assert tensor.device.type == "cpu"

# Tensor values
assert torch.all(tensor > 0)
assert torch.allclose(tensor1, tensor2, atol=1e-5)
assert not torch.isnan(tensor).any()
assert not torch.isinf(tensor).any()

# Gradients
assert tensor.grad is not None
assert not torch.all(tensor.grad == 0)
assert torch.all(tensor.grad < max_grad)

# Model state
assert model.training == True
for param in model.parameters():
    assert param.requires_grad == True
```

---

## 11. Test Documentation Template

```python
def test_example_with_documentation():
    """
    Test that [component] [behavior].

    This test validates that [specific aspect] works correctly by:
    1. [Setup step]
    2. [Action step]
    3. [Assertion step]

    Expected behavior:
    - [Expected outcome 1]
    - [Expected outcome 2]

    Related: Issue #123, PR #456
    """
    # Arrange: Setup test data
    z_a = torch.randn(8, 256)
    z_b = torch.randn(8, 256)

    # Act: Execute the functionality
    loss = VICRegLoss()
    output = loss(z_a, z_b)

    # Assert: Verify results
    assert "loss" in output
    assert not torch.isnan(output["loss"])
```

---

## 12. Tips & Troubleshooting

### Test fails intermittently
- **Cause:** Non-deterministic operations
- **Solution:** Use fixtures to seed random numbers

### Tests are too slow
- **Cause:** Real tensor computations on large batches
- **Solution:** Use smaller batches, smaller models

### Tests fail on GPU but pass on CPU
- **Cause:** Precision differences
- **Solution:** Use appropriate tolerance in comparisons

### Coverage not increasing
- **Cause:** Testing wrong code paths
- **Solution:** Check coverage reports, add tests for uncovered branches

### Fixtures not loading
- **Cause:** conftest.py not in right location
- **Solution:** Ensure conftest.py is in tests/ root

---

## Summary

**Key Principles:**
1. **Test behavior, not implementation**
2. **Use descriptive test names**
3. **Follow the Arrange-Act-Assert pattern**
4. **Keep tests independent and deterministic**
5. **Test edge cases**
6. **Use fixtures for common setup**
7. **Parametrize for multiple scenarios**
8. **Mark tests appropriately**
9. **Document complex test logic**
10. **Keep tests fast and focused**

**Remember:** Better to have 100 meaningful tests than 1,000 trivial ones!
