import torch
import pytest
from src.training.trainers.trainer import train_model
from src.models.base_model import BaseModel
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import _LRScheduler

@pytest.fixture
def test_model():
    return TrainingModelForTesting()

@pytest.fixture
def train_dataloader():
    dataset = DatasetForTesting(num_samples=100, input_dim=10)
    return DataLoader(dataset, batch_size=10, shuffle=True)

@pytest.fixture
def val_dataloader():
    dataset = DatasetForTesting(num_samples=20, input_dim=10)
    return DataLoader(dataset, batch_size=10, shuffle=False)

def test_basic_training(test_model, train_dataloader):
    """Test basic training functionality."""
    history = train_model(
        model=test_model,
        train_dataloader=train_dataloader,
        epochs=2,
        learning_rate=0.01
    )
    
    assert isinstance(history, dict)
    assert 'train_loss' in history
    assert 'train_accuracy' in history
    assert len(history['train_loss']) == 2
    assert len(history['train_accuracy']) == 2

def test_training_with_validation(test_model, train_dataloader, val_dataloader):
    """Test training with validation."""
    history = train_model(
        model=test_model,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        epochs=2,
        learning_rate=0.01
    )
    
    assert 'val_loss' in history
    assert 'val_accuracy' in history
    assert len(history['val_loss']) == 2
    assert len(history['val_accuracy']) == 2

def test_training_with_early_stopping(test_model, train_dataloader, val_dataloader):
    """Test training with early stopping."""
    history = train_model(
        model=test_model,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        epochs=10,
        learning_rate=0.01,
        early_stopping_patience=2
    )
    
    # Early stopping might trigger before all epochs
    assert len(history['train_loss']) <= 10
    assert len(history['val_loss']) <= 10

def test_training_with_scheduler(test_model, train_dataloader):
    """Test training with learning rate scheduler."""
    optimizer = torch.optim.Adam(test_model.parameters(), lr=0.01)
    # Using a scheduler that reduces LR by half every epoch
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.5)
    
    # Get initial learning rate
    initial_lr = optimizer.param_groups[0]['lr']
    
    history = train_model(
        model=test_model,
        train_dataloader=train_dataloader,
        epochs=2,
        optimizer=optimizer,
        scheduler=scheduler
    )
    
    assert len(history['train_loss']) == 2
    # Learning rate should be reduced by half twice (once per epoch)
    expected_final_lr = initial_lr * (0.5 ** 2)  # After two epochs
    assert optimizer.param_groups[0]['lr'] == expected_final_lr

def test_training_with_callbacks(test_model, train_dataloader):
    """Test training with callbacks."""
    callback_called = False
    
    def test_callback(model, epoch, history):
        nonlocal callback_called
        callback_called = True
    
    history = train_model(
        model=test_model,
        train_dataloader=train_dataloader,
        epochs=2,
        callbacks=[test_callback]
    )
    
    assert callback_called

def test_training_device_selection(test_model, train_dataloader):
    """Test training on different devices."""
    if torch.cuda.is_available():
        # Test CUDA training
        history = train_model(
            model=test_model,
            train_dataloader=train_dataloader,
            epochs=2,
            device='cuda'
        )
        assert test_model.get_device().type == 'cuda'
    
    # Test CPU training
    history = train_model(
        model=test_model,
        train_dataloader=train_dataloader,
        epochs=2,
        device='cpu'
    )
    assert test_model.get_device().type == 'cpu'

# Simple dataset for testing purposes
class DatasetForTesting(Dataset):
    """A simple dataset for testing."""
    def __init__(self, num_samples=100, input_dim=10):
        self.inputs = torch.randn(num_samples, input_dim)
        self.targets = torch.randint(0, 2, (num_samples,))
    
    def __len__(self):
        return len(self.inputs)
    
    def __getitem__(self, idx):
        return {'inputs': self.inputs[idx], 'targets': self.targets[idx]}

# Model implementation for training tests
class TrainingModelForTesting(BaseModel):
    """A test implementation of BaseModel for training tests."""
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(10, 2)
    
    def forward(self, x):
        return self.linear(x)
    
    def training_step(self, batch):
        outputs = self(batch['inputs'])
        loss = torch.nn.functional.cross_entropy(outputs, batch['targets'])
        predictions = torch.argmax(outputs, dim=1)
        accuracy = (predictions == batch['targets']).float().mean()
        return {'loss': loss, 'accuracy': accuracy}
    
    def validation_step(self, batch):
        return self.training_step(batch)
