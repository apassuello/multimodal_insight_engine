import pytest
import torch
from typing import Dict, List
from src.data.tokenization.utils import (
    TransformerTextDataset,
    create_transformer_dataloaders,
    transformer_collate_fn
)

class MockTokenizer:
    def __init__(self):
        self.special_tokens = {
            "pad_token_idx": 0,
            "bos_token_idx": 1,
            "eos_token_idx": 2,
            "<pad>": 0,
            "<bos>": 1,
            "<eos>": 2,
            "<unk>": 3
        }
        self.vocab = {
            "<pad>": 0,
            "<bos>": 1,
            "<eos>": 2,
            "<unk>": 3,
            "hello": 4,
            "world": 5,
            "test": 6
        }
    
    def encode(self, text: str) -> List[int]:
        return [self.vocab.get(word, self.vocab["<unk>"]) for word in text.split()]

@pytest.fixture
def mock_tokenizer():
    """Create a mock tokenizer for testing."""
    return MockTokenizer()

@pytest.fixture
def sample_texts():
    """Create sample texts for testing."""
    return [
        "hello world",
        "test test",
        "hello test world"
    ]

def test_transformer_dataset_initialization(mock_tokenizer, sample_texts):
    """Test initialization of TransformerTextDataset."""
    dataset = TransformerTextDataset(
        texts=sample_texts,
        tokenizer=mock_tokenizer,
        max_length=10,
        add_bos=True,
        add_eos=True
    )
    
    assert len(dataset) == len(sample_texts)
    assert dataset.max_length == 10
    assert dataset.add_bos is True
    assert dataset.add_eos is True
    assert dataset.pad_idx == mock_tokenizer.special_tokens["pad_token_idx"]
    assert dataset.bos_idx == mock_tokenizer.special_tokens["bos_token_idx"]
    assert dataset.eos_idx == mock_tokenizer.special_tokens["eos_token_idx"]

def test_transformer_dataset_getitem(mock_tokenizer, sample_texts):
    """Test getting items from the dataset."""
    dataset = TransformerTextDataset(
        texts=sample_texts,
        tokenizer=mock_tokenizer,
        max_length=10,
        add_bos=True,
        add_eos=True
    )
    
    item = dataset[0]  # Get first item
    
    assert isinstance(item, dict)
    assert "input_ids" in item
    assert "attention_mask" in item
    assert isinstance(item["input_ids"], torch.Tensor)
    assert isinstance(item["attention_mask"], torch.Tensor)
    
    # Check that BOS and EOS tokens are added
    assert item["input_ids"][0].item() == dataset.bos_idx
    assert item["input_ids"][-1].item() == dataset.eos_idx
    
    # Check attention mask
    assert torch.all(item["attention_mask"] == 1)

def test_transformer_dataset_max_length(mock_tokenizer, sample_texts):
    """Test max length handling in the dataset."""
    max_length = 5
    dataset = TransformerTextDataset(
        texts=sample_texts,
        tokenizer=mock_tokenizer,
        max_length=max_length,
        add_bos=True,
        add_eos=True
    )
    
    item = dataset[2]  # Get the longest text
    assert item["input_ids"].size(0) <= max_length
    assert item["attention_mask"].size(0) <= max_length
    assert item["input_ids"][-1].item() == dataset.eos_idx  # Should end with EOS token

def test_transformer_dataset_no_special_tokens(mock_tokenizer, sample_texts):
    """Test dataset without BOS/EOS tokens."""
    dataset = TransformerTextDataset(
        texts=sample_texts,
        tokenizer=mock_tokenizer,
        add_bos=False,
        add_eos=False
    )
    
    item = dataset[0]
    token_ids = item["input_ids"].tolist()
    assert dataset.bos_idx not in token_ids
    assert dataset.eos_idx not in token_ids

def test_transformer_dataset_return_lists(mock_tokenizer, sample_texts):
    """Test dataset returning lists instead of tensors."""
    dataset = TransformerTextDataset(
        texts=sample_texts,
        tokenizer=mock_tokenizer,
        return_tensors=False
    )
    
    item = dataset[0]
    assert isinstance(item["input_ids"], list)
    assert isinstance(item["attention_mask"], list)

def test_create_transformer_dataloaders(mock_tokenizer, sample_texts):
    """Test creation of transformer dataloaders."""
    train_loader, val_loader = create_transformer_dataloaders(
        train_texts=sample_texts,
        tokenizer=mock_tokenizer,
        val_texts=sample_texts[:1],
        batch_size=2
    )
    
    assert train_loader is not None
    assert val_loader is not None
    assert len(train_loader.dataset) == len(sample_texts)
    assert len(val_loader.dataset) == 1

def test_create_transformer_dataloaders_no_validation(mock_tokenizer, sample_texts):
    """Test dataloader creation without validation data."""
    train_loader, val_loader = create_transformer_dataloaders(
        train_texts=sample_texts,
        tokenizer=mock_tokenizer
    )
    
    assert train_loader is not None
    assert val_loader is None

def test_transformer_collate_fn_tensor_input():
    """Test collate function with tensor inputs."""
    # Create a batch of examples with different lengths
    batch = [
        {
            "input_ids": torch.tensor([1, 4, 5, 2]),
            "attention_mask": torch.tensor([1, 1, 1, 1])
        },
        {
            "input_ids": torch.tensor([1, 6, 2]),
            "attention_mask": torch.tensor([1, 1, 1])
        }
    ]
    
    collated = transformer_collate_fn(batch)
    
    assert isinstance(collated, dict)
    assert "input_ids" in collated
    assert "attention_mask" in collated
    assert collated["input_ids"].shape == (2, 4)  # batch_size=2, max_len=4
    assert collated["attention_mask"].shape == (2, 4)
    
    # Check padding
    assert torch.equal(collated["attention_mask"][1], torch.tensor([1, 1, 1, 0]))

def test_transformer_collate_fn_list_input():
    """Test collate function with list inputs."""
    # Create a batch of examples with different lengths
    batch = [
        {
            "input_ids": [1, 4, 5, 2],
            "attention_mask": [1, 1, 1, 1]
        },
        {
            "input_ids": [1, 6, 2],
            "attention_mask": [1, 1, 1]
        }
    ]
    
    collated = transformer_collate_fn(batch)
    
    assert isinstance(collated, dict)
    assert "input_ids" in collated
    assert "attention_mask" in collated
    assert collated["input_ids"].shape == (2, 4)  # batch_size=2, max_len=4
    assert collated["attention_mask"].shape == (2, 4)
    
    # Check padding
    assert torch.equal(collated["attention_mask"][1], torch.tensor([1, 1, 1, 0]))

def test_transformer_collate_fn_empty_batch():
    """Test collate function with empty batch."""
    collated = transformer_collate_fn([])
    assert isinstance(collated, dict)
    assert len(collated) == 0

def test_transformer_dataset_end_to_end(mock_tokenizer, sample_texts):
    """Test the complete pipeline from dataset to dataloader."""
    # Create dataset
    dataset = TransformerTextDataset(
        texts=sample_texts,
        tokenizer=mock_tokenizer,
        max_length=10
    )
    
    # Create dataloader
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=2,
        collate_fn=transformer_collate_fn
    )
    
    # Get a batch
    batch = next(iter(dataloader))
    
    assert isinstance(batch, dict)
    assert "input_ids" in batch
    assert "attention_mask" in batch
    assert batch["input_ids"].shape[0] == 2  # batch_size
    assert batch["attention_mask"].shape == batch["input_ids"].shape 