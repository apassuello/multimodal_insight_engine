import os
import torch
import pytest
from src.models.base_model import BaseModel

@pytest.mark.no_test
class TestModel(BaseModel):
    """A test implementation of BaseModel for testing purposes."""
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(10, 5)
    
    def forward(self, x):
        return self.linear(x)

@pytest.fixture
def test_model():
    return TestModel()

@pytest.fixture
def temp_save_path(tmp_path):
    return str(tmp_path / "test_model.pt")

def test_model_initialization(test_model):
    """Test that the model initializes correctly."""
    assert isinstance(test_model, BaseModel)
    assert test_model.model_type == "TestModel"
    assert hasattr(test_model, "linear")

def test_model_forward(test_model):
    """Test the forward pass of the model."""
    x = torch.randn(5, 10)
    output = test_model(x)
    assert output.shape == (5, 5)

def test_model_save_load(test_model, temp_save_path):
    """Test saving and loading model weights."""
    # Save the model
    test_model.save(temp_save_path)
    assert os.path.exists(temp_save_path)
    
    # Create a new model instance
    new_model = TestModel()
    
    # Load the weights
    checkpoint = new_model.load(temp_save_path)
    assert isinstance(checkpoint, dict)
    
    # Test that weights are loaded correctly
    x = torch.randn(5, 10)
    original_output = test_model(x)
    loaded_output = new_model(x)
    assert torch.allclose(original_output, loaded_output)

def test_model_save_with_optimizer(test_model, temp_save_path):
    """Test saving model with optimizer state."""
    optimizer = torch.optim.Adam(test_model.parameters())
    test_model.save(temp_save_path, optimizer=optimizer)
    
    # Load and verify optimizer state
    new_model = TestModel()
    checkpoint = new_model.load(temp_save_path)
    assert 'optimizer_state_dict' in checkpoint

def test_model_save_with_additional_info(test_model, temp_save_path):
    """Test saving model with additional information."""
    additional_info = {'epoch': 5, 'loss': 0.5, 'custom_info': 'test'}
    test_model.save(temp_save_path, additional_info=additional_info)
    
    # Load and verify additional info
    new_model = TestModel()
    checkpoint = new_model.load(temp_save_path)
    assert checkpoint['epoch'] == 5
    assert checkpoint['loss'] == 0.5
    assert checkpoint['custom_info'] == 'test'

def test_model_parameter_counting(test_model):
    """Test parameter counting functionality."""
    num_params = test_model.count_parameters()
    assert isinstance(num_params, int)
    assert num_params > 0
    # Linear layer with 10 input features and 5 output features
    expected_params = 10 * 5 + 5  # weights + bias
    assert num_params == expected_params

def test_model_device(test_model):
    """Test device detection functionality."""
    device = test_model.get_device()
    assert isinstance(device, torch.device)
    
    # Test moving model to different device
    if torch.cuda.is_available():
        test_model.to('cuda')
        assert test_model.get_device().type == 'cuda'
    
    test_model.to('cpu')
    assert test_model.get_device().type == 'cpu'
