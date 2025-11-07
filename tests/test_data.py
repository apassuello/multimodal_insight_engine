import pytest
import torch

from src.data.dataloader import MultimodalDataset, create_dataloader, get_dataloaders
from src.data.preprocessing import DataPreprocessor, create_sequences, split_data


@pytest.fixture
def sample_data():
    """Create sample data for testing."""
    return torch.randn(100, 5)

@pytest.fixture
def sample_multimodal_data():
    """Create sample multimodal data for testing."""
    return {
        'text': torch.randn(100, 10),
        'image': torch.randn(100, 3, 32, 32),
        'audio': torch.randn(100, 20)
    }

def test_data_preprocessor_standard(sample_data):
    """Test standard scaling preprocessing."""
    preprocessor = DataPreprocessor(method='standard')

    # Test fit_transform
    transformed = preprocessor.fit_transform(sample_data)
    assert isinstance(transformed, torch.Tensor)
    assert transformed.shape == sample_data.shape

    # Test inverse transform
    original = preprocessor.inverse_transform(transformed)
    assert torch.allclose(original, sample_data, atol=1e-6)

    # Test transform without fit
    preprocessor = DataPreprocessor(method='standard')
    with pytest.raises(RuntimeError):
        preprocessor.transform(sample_data)

def test_data_preprocessor_minmax(sample_data):
    """Test min-max scaling preprocessing."""
    preprocessor = DataPreprocessor(method='minmax')

    # Test fit_transform
    transformed = preprocessor.fit_transform(sample_data)
    assert isinstance(transformed, torch.Tensor)
    assert transformed.shape == sample_data.shape

    # Test inverse transform
    original = preprocessor.inverse_transform(transformed)
    assert torch.allclose(original, sample_data, atol=1e-6)

def test_create_sequences(sample_data):
    """Test sequence creation from time series data."""
    seq_length = 10
    sequences, targets = create_sequences(sample_data, seq_length)

    assert isinstance(sequences, torch.Tensor)
    assert isinstance(targets, torch.Tensor)
    assert sequences.shape[1] == seq_length
    assert sequences.shape[2] == sample_data.shape[1]
    assert targets.shape[1] == sample_data.shape[1]
    assert len(sequences) == len(sample_data) - seq_length

def test_split_data(sample_data):
    """Test data splitting functionality."""
    train_data, val_data, test_data = split_data(sample_data)

    assert isinstance(train_data, torch.Tensor)
    assert isinstance(val_data, torch.Tensor)
    assert isinstance(test_data, torch.Tensor)

    # Test split ratios
    total_size = len(sample_data)
    assert len(train_data) == int(total_size * 0.8)
    assert len(val_data) == int(total_size * 0.1)
    assert len(test_data) == total_size - len(train_data) - len(val_data)

def test_multimodal_dataset(sample_multimodal_data):
    """Test multimodal dataset functionality."""
    dataset = MultimodalDataset(sample_multimodal_data)

    assert len(dataset) == 100

    # Test item retrieval
    item = dataset[0]
    assert isinstance(item, dict)
    assert all(key in item for key in sample_multimodal_data.keys())
    assert all(isinstance(value, torch.Tensor) for value in item.values())

    # Test invalid data lengths
    invalid_data = sample_multimodal_data.copy()
    invalid_data['text'] = torch.randn(90, 10)  # Different length
    with pytest.raises(ValueError):
        MultimodalDataset(invalid_data)

def test_create_dataloader(sample_multimodal_data):
    """Test dataloader creation."""
    dataset = MultimodalDataset(sample_multimodal_data)
    dataloader = create_dataloader(dataset, batch_size=32)

    assert isinstance(dataloader, torch.utils.data.DataLoader)
    assert dataloader.batch_size == 32

    # Test batch iteration
    batch = next(iter(dataloader))
    assert isinstance(batch, dict)
    assert all(key in batch for key in sample_multimodal_data.keys())
    assert all(isinstance(value, torch.Tensor) for value in batch.values())
    assert all(value.shape[0] == 32 for value in batch.values())

def test_get_dataloaders(sample_multimodal_data):
    """Test creation of train, validation, and test dataloaders."""
    # Create validation and test data
    val_data = {k: v[:20] for k, v in sample_multimodal_data.items()}
    test_data = {k: v[:20] for k, v in sample_multimodal_data.items()}

    train_loader, val_loader, test_loader = get_dataloaders(
        sample_multimodal_data,
        val_data=val_data,
        test_data=test_data,
        batch_size=16  # Smaller batch size that works for all datasets
    )

    assert isinstance(train_loader, torch.utils.data.DataLoader)
    assert isinstance(val_loader, torch.utils.data.DataLoader)
    assert isinstance(test_loader, torch.utils.data.DataLoader)

    # Test batch shapes
    train_batch = next(iter(train_loader))
    val_batch = next(iter(val_loader))
    test_batch = next(iter(test_loader))

    assert all(value.shape[0] == 16 for value in train_batch.values())
    assert all(value.shape[0] == 16 for value in val_batch.values())
    assert all(value.shape[0] == 16 for value in test_batch.values())
