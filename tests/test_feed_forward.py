import pytest
import torch
import numpy as np
from src.models.feed_forward import FeedForwardNN, FeedForwardClassifier, MultiLayerPerceptron

@pytest.fixture
def device():
    """Return available device."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

@pytest.fixture
def batch_size():
    """Return batch size for tests."""
    return 16

@pytest.fixture
def input_size():
    """Return input size for tests."""
    return 32

@pytest.fixture
def hidden_sizes():
    """Return hidden sizes for tests."""
    return [64, 32]

@pytest.fixture
def output_size():
    """Return output size for tests."""
    return 10

@pytest.fixture
def ff_network(input_size, hidden_sizes, output_size, device):
    """Create a feed-forward neural network for testing."""
    model = FeedForwardNN(
        input_size=input_size,
        hidden_sizes=hidden_sizes,
        output_size=output_size,
        activation="relu",
        dropout=0.1
    ).to(device)
    return model

@pytest.fixture
def ff_classifier(input_size, hidden_sizes, output_size, device):
    """Create a feed-forward classifier for testing."""
    model = FeedForwardClassifier(
        input_size=input_size,
        hidden_sizes=hidden_sizes,
        num_classes=output_size,
        activation="relu",
        dropout=0.1
    ).to(device)
    return model

@pytest.fixture
def mlp(input_size, hidden_sizes, output_size, device):
    """Create a multi-layer perceptron for testing."""
    model = MultiLayerPerceptron(
        input_dim=input_size,
        hidden_dims=hidden_sizes,
        output_dim=output_size,
        activation="relu",
        dropout=0.1
    ).to(device)
    return model

def test_feed_forward_nn_shape(ff_network, batch_size, input_size, output_size, device):
    """Test that FeedForwardNN preserves the expected output shape."""
    x = torch.randn(batch_size, input_size).to(device)
    output = ff_network(x)
    
    assert output.shape == (batch_size, output_size)

def test_feed_forward_nn_grad_flow(ff_network, batch_size, input_size, device):
    """Test that gradients flow through the FeedForwardNN correctly."""
    x = torch.randn(batch_size, input_size).to(device)
    
    # Track gradients
    ff_network.zero_grad()
    output = ff_network(x)
    
    # Use mean of output as a simple scalar loss
    loss = output.mean()
    loss.backward()
    
    # Check that all parameters have gradients
    for name, param in ff_network.named_parameters():
        if param.requires_grad:
            assert param.grad is not None, f"Parameter {name} has no gradient"
            assert not torch.all(param.grad == 0), f"Parameter {name} has zero gradient"

def test_feed_forward_nn_with_layer_norm(input_size, hidden_sizes, output_size, batch_size, device):
    """Test FeedForwardNN with layer normalization."""
    model = FeedForwardNN(
        input_size=input_size,
        hidden_sizes=hidden_sizes,
        output_size=output_size,
        activation="relu",
        dropout=0.1,
        use_layer_norm=True
    ).to(device)
    
    x = torch.randn(batch_size, input_size).to(device)
    output = model(x)
    
    assert output.shape == (batch_size, output_size)
    
    # Verify layer norm is used by checking if model has LayerNorm modules
    has_layer_norm = False
    for module in model.modules():
        if isinstance(module, torch.nn.LayerNorm):
            has_layer_norm = True
            break
    
    assert has_layer_norm, "Model should contain LayerNorm modules"

def test_feed_forward_nn_with_residual(input_size, hidden_sizes, output_size, batch_size, device):
    """Test FeedForwardNN with residual connections."""
    model = FeedForwardNN(
        input_size=input_size,
        hidden_sizes=[input_size, input_size],  # Same size for residual connections
        output_size=output_size,
        activation="relu",
        dropout=0.1,
        use_residual=True
    ).to(device)
    
    x = torch.randn(batch_size, input_size).to(device)
    output = model(x)
    
    assert output.shape == (batch_size, output_size)

def test_feed_forward_classifier_shape(ff_classifier, batch_size, input_size, output_size, device):
    """Test that FeedForwardClassifier preserves the expected output shape."""
    x = torch.randn(batch_size, input_size).to(device)
    output = ff_classifier(x)
    
    assert output.shape == (batch_size, output_size)

def test_feed_forward_classifier_predict(ff_classifier, batch_size, input_size, device):
    """Test the predict method of FeedForwardClassifier."""
    x = torch.randn(batch_size, input_size).to(device)
    predictions = ff_classifier.predict(x)
    
    assert predictions.shape == (batch_size,)
    assert predictions.max() < ff_classifier.num_classes
    assert predictions.min() >= 0

def test_feed_forward_classifier_predict_proba(ff_classifier, batch_size, input_size, device):
    """Test the predict_proba method of FeedForwardClassifier."""
    x = torch.randn(batch_size, input_size).to(device)
    probabilities = ff_classifier.predict_proba(x)
    
    assert probabilities.shape == (batch_size, ff_classifier.num_classes)
    
    # Check that the probabilities sum to 1 for each sample
    assert torch.allclose(probabilities.sum(dim=1), torch.ones(batch_size).to(device), atol=1e-6)
    
    # Check that all probabilities are between 0 and 1
    assert (probabilities >= 0).all() and (probabilities <= 1).all()

def test_feed_forward_classifier_training_step(ff_classifier, batch_size, input_size, output_size, device):
    """Test the training_step method of FeedForwardClassifier."""
    inputs = torch.randn(batch_size, input_size).to(device)
    targets = torch.randint(0, output_size, (batch_size,)).to(device)
    
    batch = {"inputs": inputs, "targets": targets}
    result = ff_classifier.training_step(batch)
    
    assert "loss" in result
    assert "accuracy" in result
    assert "predictions" in result
    
    assert result["loss"].shape == torch.Size([])  # Scalar loss
    assert result["accuracy"].shape == torch.Size([])  # Scalar accuracy
    assert result["predictions"].shape == (batch_size,)

def test_feed_forward_classifier_validation_step(ff_classifier, batch_size, input_size, output_size, device):
    """Test the validation_step method of FeedForwardClassifier."""
    inputs = torch.randn(batch_size, input_size).to(device)
    targets = torch.randint(0, output_size, (batch_size,)).to(device)
    
    batch = {"inputs": inputs, "targets": targets}
    result = ff_classifier.validation_step(batch)
    
    assert "loss" in result
    assert "accuracy" in result
    assert "predictions" in result
    
    assert result["loss"].shape == torch.Size([])  # Scalar loss
    assert result["accuracy"].shape == torch.Size([])  # Scalar accuracy
    assert result["predictions"].shape == (batch_size,)

def test_feed_forward_classifier_optimizer(ff_classifier):
    """Test the optimizer configuration of the FeedForwardClassifier."""
    optimizer = ff_classifier.configure_optimizers(lr=0.001)
    
    assert isinstance(optimizer, torch.optim.Adam)
    assert optimizer.param_groups[0]["lr"] == 0.001
    
    # Check that all model parameters are in the optimizer
    model_params = set(ff_classifier.parameters())
    optimizer_params = set()
    for group in optimizer.param_groups:
        optimizer_params.update(group["params"])
    
    assert model_params == optimizer_params

def test_mlp_shape(mlp, batch_size, input_size, output_size, device):
    """Test that MultiLayerPerceptron preserves the expected output shape."""
    x = torch.randn(batch_size, input_size).to(device)
    output = mlp(x)
    
    assert output.shape == (batch_size, output_size)

def test_mlp_with_layer_norm(input_size, hidden_sizes, output_size, batch_size, device):
    """Test MultiLayerPerceptron with layer normalization."""
    model = MultiLayerPerceptron(
        input_dim=input_size,
        hidden_dims=hidden_sizes,
        output_dim=output_size,
        activation="relu",
        dropout=0.1,
        use_layer_norm=True
    ).to(device)
    
    x = torch.randn(batch_size, input_size).to(device)
    output = model(x)
    
    assert output.shape == (batch_size, output_size)
    
    # Verify layer norm is used by checking if model has LayerNorm modules
    has_layer_norm = False
    for module in model.modules():
        if isinstance(module, torch.nn.LayerNorm):
            has_layer_norm = True
            break
    
    assert has_layer_norm, "Model should contain LayerNorm modules"

def test_mlp_with_residual(input_size, hidden_sizes, output_size, batch_size, device):
    """Test MultiLayerPerceptron with residual connections."""
    model = MultiLayerPerceptron(
        input_dim=input_size,
        hidden_dims=[input_size, input_size],  # Same size for residual connections
        output_dim=output_size,
        activation="relu",
        dropout=0.1,
        use_residual=True
    ).to(device)
    
    x = torch.randn(batch_size, input_size).to(device)
    output = model(x)
    
    assert output.shape == (batch_size, output_size)

def test_different_activations(input_size, hidden_sizes, output_size, batch_size, device):
    """Test different activation functions in the feed-forward networks."""
    activations = ["relu", "gelu", "tanh", "sigmoid"]
    
    for activation in activations:
        model = FeedForwardNN(
            input_size=input_size,
            hidden_sizes=hidden_sizes,
            output_size=output_size,
            activation=activation,
            dropout=0.1
        ).to(device)
        
        x = torch.randn(batch_size, input_size).to(device)
        output = model(x)
        
        assert output.shape == (batch_size, output_size)

def test_batch_consistency(ff_network, batch_size, input_size, device):
    """Test that outputs are consistent across samples in a batch."""
    # Set model to eval mode to disable dropout
    ff_network.eval()
    
    # Create a batch with identical inputs
    x = torch.randn(1, input_size).to(device)
    x_batch = x.repeat(batch_size, 1)
    
    # Use no_grad to ensure deterministic outputs
    with torch.no_grad():
        output = ff_network(x_batch)
    
    # All outputs should be identical since inputs are identical
    for i in range(1, batch_size):
        assert torch.allclose(output[0], output[i], atol=1e-6)

def test_model_serialization(ff_network, tmp_path, device):
    """Test saving and loading the model."""
    # Set model to eval mode
    ff_network.eval()
    
    # Save the model
    save_path = tmp_path / "feed_forward.pt"
    torch.save(ff_network.state_dict(), save_path)
    
    # Create a new model with the same architecture
    new_model = FeedForwardNN(
        input_size=ff_network.input_size,
        hidden_sizes=ff_network.hidden_sizes,
        output_size=ff_network.output_size,
        activation=ff_network.activation,
        dropout=0.0  # Set dropout to 0 for deterministic behavior
    ).to(device)
    
    # Load the saved state
    new_model.load_state_dict(torch.load(save_path))
    new_model.eval()  # Set new model to eval mode too
    
    # Test that both models produce the same output for the same input
    x = torch.randn(4, ff_network.input_size).to(device)
    with torch.no_grad():  # Use no_grad for deterministic behavior
        output1 = ff_network(x)
        output2 = new_model(x)
    
    assert torch.allclose(output1, output2, atol=1e-6) 