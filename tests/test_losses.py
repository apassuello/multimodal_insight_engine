import pytest
import torch
import torch.nn.functional as F

from src.training.losses import CrossEntropyLoss, MeanSquaredError


@pytest.fixture
def device():
    """Return available device."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

@pytest.fixture
def batch_size():
    """Return batch size for tests."""
    return 8

@pytest.fixture
def num_classes():
    """Return number of classes for tests."""
    return 5

@pytest.fixture
def logits(batch_size, num_classes, device):
    """Create random logits for testing."""
    return torch.randn(batch_size, num_classes).to(device)

@pytest.fixture
def targets(batch_size, num_classes, device):
    """Create random class indices for testing."""
    return torch.randint(0, num_classes, (batch_size,)).to(device)

@pytest.fixture
def regression_inputs(batch_size, device):
    """Create random regression inputs for testing."""
    return torch.randn(batch_size, 3).to(device)

@pytest.fixture
def regression_targets(batch_size, device):
    """Create random regression targets for testing."""
    return torch.randn(batch_size, 3).to(device)

@pytest.fixture
def sample_weight(batch_size, device):
    """Create random sample weights for testing."""
    return torch.rand(batch_size).to(device)

@pytest.fixture
def cross_entropy_loss():
    """Create cross-entropy loss with default parameters."""
    return CrossEntropyLoss(smoothing=0.1)

@pytest.fixture
def mse_loss():
    """Create mean squared error loss with default parameters."""
    return MeanSquaredError()

def test_cross_entropy_loss_shape(cross_entropy_loss, logits, targets, device):
    """Test that CrossEntropyLoss returns a scalar value."""
    # Make sure logits require gradients
    logits_with_grad = logits.clone().requires_grad_(True)

    loss = cross_entropy_loss(logits_with_grad, targets)

    assert loss.shape == torch.Size([])  # Scalar output
    assert loss.requires_grad  # Should be differentiable

def test_cross_entropy_loss_no_smoothing(logits, targets, device):
    """Test CrossEntropyLoss without label smoothing."""
    # Create loss without smoothing
    loss_fn = CrossEntropyLoss(smoothing=0)

    # Calculate loss
    loss = loss_fn(logits, targets)

    # Compare with PyTorch's built-in cross entropy
    pytorch_loss = F.cross_entropy(logits, targets)

    # Should be very close
    assert torch.allclose(loss, pytorch_loss, atol=1e-5)

def test_cross_entropy_loss_with_smoothing(logits, targets, device):
    """Test CrossEntropyLoss with label smoothing."""
    # Create loss with smoothing
    smoothing = 0.2
    loss_fn = CrossEntropyLoss(smoothing=smoothing)

    # Calculate loss
    loss = loss_fn(logits, targets)

    # When using label smoothing, loss should be different from standard CE loss
    pytorch_loss = F.cross_entropy(logits, targets)

    # Should NOT be equal due to smoothing
    assert not torch.allclose(loss, pytorch_loss, atol=1e-5)

    # Manual implementation of label smoothing for comparison
    n_classes = logits.size(-1)
    one_hot = F.one_hot(targets, n_classes).float()
    smooth_one_hot = one_hot * (1 - smoothing) + (smoothing / n_classes)
    log_probs = F.log_softmax(logits, dim=-1)
    manual_loss = -(smooth_one_hot * log_probs).sum(dim=-1).mean()

    # Should be close to our manual implementation
    assert torch.allclose(loss, manual_loss, atol=1e-5)

def test_cross_entropy_loss_with_weight(logits, targets, sample_weight, device):
    """Test CrossEntropyLoss with sample weights."""
    # Create loss
    loss_fn = CrossEntropyLoss(smoothing=0.1)

    # Calculate loss with weights
    loss = loss_fn(logits, targets, sample_weight)

    # Calculate manual weighted loss for comparison
    n_classes = logits.size(-1)
    one_hot = F.one_hot(targets, n_classes).float()
    smooth_one_hot = one_hot * (1 - 0.1) + (0.1 / n_classes)
    log_probs = F.log_softmax(logits, dim=-1)
    per_sample_loss = -(smooth_one_hot * log_probs).sum(dim=-1)
    weighted_loss = (per_sample_loss * sample_weight).mean()

    # Should be close to our manual implementation
    assert torch.allclose(loss, weighted_loss, atol=1e-5)

def test_cross_entropy_loss_reduction_none(logits, targets, device):
    """Test CrossEntropyLoss with 'none' reduction."""
    # Create loss with 'none' reduction
    loss_fn = CrossEntropyLoss(smoothing=0.1, reduction='none')

    # Calculate loss
    loss = loss_fn(logits, targets)

    # Should have shape [batch_size]
    assert loss.shape == (logits.size(0),)

def test_cross_entropy_loss_reduction_sum(logits, targets, device):
    """Test CrossEntropyLoss with 'sum' reduction."""
    # Create loss with 'sum' reduction
    loss_fn = CrossEntropyLoss(smoothing=0.1, reduction='sum')

    # Calculate loss
    loss = loss_fn(logits, targets)

    # Should be a scalar
    assert loss.shape == torch.Size([])

    # Calculate manual sum for comparison
    loss_none = CrossEntropyLoss(smoothing=0.1, reduction='none')(logits, targets)
    loss_sum_manual = loss_none.sum()

    # Should be close to our manual implementation
    assert torch.allclose(loss, loss_sum_manual, atol=1e-5)

def test_cross_entropy_loss_gradient(logits, targets, device):
    """Test that gradients flow through CrossEntropyLoss."""
    # Create loss
    loss_fn = CrossEntropyLoss(smoothing=0.1)

    # Requires grad
    logits.requires_grad_(True)

    # Calculate loss
    loss = loss_fn(logits, targets)

    # Backpropagate
    loss.backward()

    # Logits should have gradient
    assert logits.grad is not None
    assert not torch.all(logits.grad == 0)

def test_mse_loss_shape(mse_loss, regression_inputs, regression_targets, device):
    """Test that MeanSquaredError returns a scalar value."""
    # Make sure inputs require gradients
    inputs_with_grad = regression_inputs.clone().requires_grad_(True)

    loss = mse_loss(inputs_with_grad, regression_targets)

    assert loss.shape == torch.Size([])  # Scalar output
    assert loss.requires_grad  # Should be differentiable

def test_mse_loss_basic(regression_inputs, regression_targets, device):
    """Test basic MeanSquaredError calculation."""
    # Create loss
    loss_fn = MeanSquaredError()

    # Calculate loss
    loss = loss_fn(regression_inputs, regression_targets)

    # Compare with PyTorch's built-in MSE loss
    pytorch_loss = F.mse_loss(regression_inputs, regression_targets)

    # Should be very close
    assert torch.allclose(loss, pytorch_loss, atol=1e-5)

def test_mse_loss_with_weight(regression_inputs, regression_targets, sample_weight, device):
    """Test MeanSquaredError with sample weights."""
    # Create loss
    loss_fn = MeanSquaredError()

    # Calculate loss with weights
    loss = loss_fn(regression_inputs, regression_targets, sample_weight)

    # Calculate manual weighted loss for comparison
    per_sample_loss = (regression_inputs - regression_targets) ** 2
    per_sample_loss = per_sample_loss.mean(dim=1)  # Average across features
    weighted_loss = (per_sample_loss * sample_weight).mean()

    # Should be close to our manual implementation
    assert torch.allclose(loss, weighted_loss, atol=1e-5)

def test_mse_loss_reduction_none(regression_inputs, regression_targets, device):
    """Test MeanSquaredError with 'none' reduction."""
    # Create loss with 'none' reduction
    loss_fn = MeanSquaredError(reduction='none')

    # Calculate loss
    loss = loss_fn(regression_inputs, regression_targets)

    # Should have shape [batch_size, feature_dim]
    assert loss.shape == regression_inputs.shape

def test_mse_loss_reduction_sum(regression_inputs, regression_targets, device):
    """Test MeanSquaredError with 'sum' reduction."""
    # Create loss with 'sum' reduction
    loss_fn = MeanSquaredError(reduction='sum')

    # Calculate loss
    loss = loss_fn(regression_inputs, regression_targets)

    # Should be a scalar
    assert loss.shape == torch.Size([])

    # Calculate manual sum for comparison
    squared_error = (regression_inputs - regression_targets) ** 2
    loss_sum_manual = squared_error.sum()

    # Should be close to our manual implementation
    assert torch.allclose(loss, loss_sum_manual, atol=1e-5)

def test_mse_loss_gradient(regression_inputs, regression_targets, device):
    """Test that gradients flow through MeanSquaredError."""
    # Create loss
    loss_fn = MeanSquaredError()

    # Requires grad
    regression_inputs.requires_grad_(True)

    # Calculate loss
    loss = loss_fn(regression_inputs, regression_targets)

    # Backpropagate
    loss.backward()

    # Inputs should have gradient
    assert regression_inputs.grad is not None
    assert not torch.all(regression_inputs.grad == 0)

def test_mse_loss_clip_grad(regression_inputs, regression_targets, device):
    """Test MeanSquaredError with gradient clipping."""
    # Create inputs with extreme values
    extreme_inputs = torch.ones_like(regression_inputs) * 1000
    extreme_inputs.requires_grad_(True)

    # Create loss with gradient clipping
    clip_value = 1.0
    loss_fn = MeanSquaredError(clip_grad=clip_value)

    # Calculate loss
    loss = loss_fn(extreme_inputs, regression_targets)

    # Backpropagate
    loss.backward()

    # Check that gradients are clipped
    assert extreme_inputs.grad is not None
    assert torch.all(extreme_inputs.grad <= clip_value + 1e-5)
