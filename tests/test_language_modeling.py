import pytest
import torch
from src.data.language_modeling import (
    LanguageModelingDataset,
    lm_collate_fn,
    create_lm_dataloaders
)
from src.data.tokenization import BPETokenizer

@pytest.fixture
def sample_texts():
    """Sample texts for testing."""
    return [
        "Hello world",
        "This is a test",
        "Machine learning is fun",
        "Another example text"
    ]

@pytest.fixture
def mock_tokenizer():
    """Mock tokenizer for testing."""
    tokenizer = BPETokenizer(num_merges=10)  # Small vocab for testing
    tokenizer.train(
        texts=["Hello world", "This is a test"],
        vocab_size=256,
        min_frequency=1,
        show_progress=False
    )
    return tokenizer

def test_language_modeling_dataset_initialization(sample_texts, mock_tokenizer):
    """Test initialization of LanguageModelingDataset."""
    dataset = LanguageModelingDataset(
        texts=sample_texts,
        tokenizer=mock_tokenizer,
        max_length=10,
        pad_idx=0,
        bos_idx=1,
        eos_idx=2
    )
    
    assert len(dataset) == len(sample_texts)
    assert dataset.max_length == 10
    assert dataset.pad_idx == 0
    assert dataset.bos_idx == 1
    assert dataset.eos_idx == 2

def test_language_modeling_dataset_getitem(sample_texts, mock_tokenizer):
    """Test getting items from the dataset."""
    dataset = LanguageModelingDataset(
        texts=sample_texts,
        tokenizer=mock_tokenizer,
        max_length=10,
        pad_idx=0,
        bos_idx=1,
        eos_idx=2
    )
    
    # Get first item
    item = dataset[0]
    
    # Check structure
    assert isinstance(item, dict)
    assert "input_ids" in item
    assert "labels" in item
    
    # Check tensor types
    assert isinstance(item["input_ids"], torch.Tensor)
    assert isinstance(item["labels"], torch.Tensor)
    
    # Check shapes
    assert item["input_ids"].dim() == 1
    assert item["labels"].dim() == 1
    assert item["input_ids"].size(0) == item["labels"].size(0)
    
    # Check that labels are shifted by one position
    assert torch.all(item["labels"][:-1] == item["input_ids"][1:])

def test_language_modeling_dataset_truncation(sample_texts, mock_tokenizer):
    """Test that sequences are properly truncated."""
    dataset = LanguageModelingDataset(
        texts=sample_texts,
        tokenizer=mock_tokenizer,
        max_length=5,  # Very short max length
        pad_idx=0,
        bos_idx=1,
        eos_idx=2
    )
    
    item = dataset[0]
    assert item["input_ids"].size(0) <= 5
    assert item["labels"].size(0) <= 5
    assert item["input_ids"][-1] == 2  # EOS token should be at the end

def test_lm_collate_fn():
    """Test the collate function for language modeling."""
    # Create a batch of examples
    batch = [
        {
            "input_ids": torch.tensor([1, 2, 3]),
            "labels": torch.tensor([2, 3, 4])
        },
        {
            "input_ids": torch.tensor([1, 2]),
            "labels": torch.tensor([2, 3])
        }
    ]
    
    # Collate the batch
    collated = lm_collate_fn(batch, pad_idx=0)
    
    # Check structure
    assert isinstance(collated, dict)
    assert "input_ids" in collated
    assert "labels" in collated
    assert "attention_mask" in collated
    
    # Check shapes
    assert collated["input_ids"].shape == (2, 3)  # batch_size=2, max_length=3
    assert collated["labels"].shape == (2, 3)
    assert collated["attention_mask"].shape == (2, 3)
    
    # Check padding
    assert collated["input_ids"][1, 2] == 0  # Padding token
    assert collated["labels"][1, 2] == -100  # Ignored in loss
    assert not collated["attention_mask"][1, 2]  # Not attended to

def test_create_lm_dataloaders(sample_texts, mock_tokenizer):
    """Test creation of language modeling dataloaders."""
    train_dataloader, val_dataloader = create_lm_dataloaders(
        texts=sample_texts,
        tokenizer=mock_tokenizer,
        batch_size=2,
        max_length=10,
        val_split=0.25,  # 1 example in validation
        seed=42
    )
    
    # Check dataloader types
    assert isinstance(train_dataloader, torch.utils.data.DataLoader)
    assert isinstance(val_dataloader, torch.utils.data.DataLoader)
    
    # Check batch sizes
    assert train_dataloader.batch_size == 2
    assert val_dataloader.batch_size == 2
    
    # Check dataset sizes
    assert len(train_dataloader.dataset) == 3  # 75% of 4 examples
    assert len(val_dataloader.dataset) == 1    # 25% of 4 examples
    
    # Check that we can iterate over the dataloaders
    train_batch = next(iter(train_dataloader))
    val_batch = next(iter(val_dataloader))
    
    # Check batch structure
    for batch in [train_batch, val_batch]:
        assert isinstance(batch, dict)
        assert "input_ids" in batch
        assert "labels" in batch
        assert "attention_mask" in batch 