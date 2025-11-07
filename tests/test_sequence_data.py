import pytest
import torch

from src.data.sequence_data import (
    TransformerCollator,
    TransformerDataModule,
    TransformerDataset,
    transformer_collate_fn,
)


@pytest.fixture
def device():
    """Get the device to use for testing."""
    if torch.backends.mps.is_available():
        # Get the first MPS device
        return torch.device("mps:0")
    return torch.device("cpu")

@pytest.fixture
def sample_sequences():
    """Sample sequences for testing."""
    return {
        "source": [
            [1, 2, 3],
            [4, 5],
            [6, 7, 8, 9],
            [10, 11, 12, 13],
            [14, 15],
            [16, 17, 18, 19],
            [20, 21],
            [22, 23, 24, 25],
            [26, 27],
            [28, 29, 30, 31]
        ],
        "target": [
            [32, 33, 34],
            [35, 36],
            [37, 38, 39, 40],
            [41, 42, 43, 44],
            [45, 46],
            [47, 48, 49, 50],
            [51, 52],
            [53, 54, 55, 56],
            [57, 58],
            [59, 60, 61, 62]
        ]
    }

def test_transformer_collate_fn(sample_sequences, device):
    """Test the transformer collate function."""
    # Create a batch of examples
    batch = [
        {"src_tokens": src, "tgt_tokens": tgt}
        for src, tgt in zip(sample_sequences["source"], sample_sequences["target"])
    ]

    # Collate the batch
    collated = transformer_collate_fn(batch, pad_idx=0)

    # Check structure
    assert isinstance(collated, dict)
    assert "src" in collated
    assert "tgt" in collated

    # Check shapes
    assert collated["src"].shape == (10, 4)  # batch_size=10, max_src_len=4
    assert collated["tgt"].shape == (10, 4)  # batch_size=10, max_tgt_len=4

    # Check device
    assert collated["src"].device == device
    assert collated["tgt"].device == device

    # Check padding
    assert collated["src"][1, 2] == 0  # Padding token
    assert collated["src"][1, 3] == 0  # Padding token
    assert collated["tgt"][1, 2] == 0  # Padding token
    assert collated["tgt"][1, 3] == 0  # Padding token

    # Check actual data
    assert torch.all(collated["src"][0, :3] == torch.tensor([1, 2, 3], device=device))
    assert torch.all(collated["src"][1, :2] == torch.tensor([4, 5], device=device))
    assert torch.all(collated["src"][2, :4] == torch.tensor([6, 7, 8, 9], device=device))

    assert torch.all(collated["tgt"][0, :3] == torch.tensor([32, 33, 34], device=device))
    assert torch.all(collated["tgt"][1, :2] == torch.tensor([35, 36], device=device))
    assert torch.all(collated["tgt"][2, :4] == torch.tensor([37, 38, 39, 40], device=device))

def test_transformer_dataset_initialization(sample_sequences):
    """Test initialization of TransformerDataset."""
    dataset = TransformerDataset(
        source_sequences=sample_sequences["source"],
        target_sequences=sample_sequences["target"],
        max_src_len=5,
        max_tgt_len=5,
        pad_idx=0,
        bos_idx=1,
        eos_idx=2
    )

    assert len(dataset) == 10
    assert dataset.max_src_len == 5
    assert dataset.max_tgt_len == 5
    assert dataset.pad_idx == 0
    assert dataset.bos_idx == 1
    assert dataset.eos_idx == 2

def test_transformer_dataset_getitem(sample_sequences, device):
    """Test getting items from the dataset."""
    dataset = TransformerDataset(
        source_sequences=sample_sequences["source"],
        target_sequences=sample_sequences["target"],
        max_src_len=5,
        max_tgt_len=5,
        pad_idx=0,
        bos_idx=1,
        eos_idx=2
    )

    # Get first item
    item = dataset[0]

    # Check structure
    assert isinstance(item, dict)
    assert "src_tokens" in item
    assert "tgt_tokens" in item

    # Check tensor types
    assert isinstance(item["src_tokens"], torch.Tensor)
    assert isinstance(item["tgt_tokens"], torch.Tensor)

    # Check shapes
    assert item["src_tokens"].dim() == 1
    assert item["tgt_tokens"].dim() == 1

    # Check device
    assert item["src_tokens"].device == device
    assert item["tgt_tokens"].device == device

    # Check content
    assert torch.all(item["src_tokens"] == torch.tensor([1, 2, 3], device=device))
    assert torch.all(item["tgt_tokens"] == torch.tensor([1, 32, 33, 34, 2], device=device))  # With BOS and EOS

def test_transformer_dataset_truncation(sample_sequences, device):
    """Test that sequences are properly truncated."""
    dataset = TransformerDataset(
        source_sequences=sample_sequences["source"],
        target_sequences=sample_sequences["target"],
        max_src_len=2,
        max_tgt_len=2,
        pad_idx=0,
        bos_idx=1,
        eos_idx=2
    )

    # Get first item
    item = dataset[0]

    # Check truncation
    assert item["src_tokens"].size(0) == 2
    assert item["tgt_tokens"].size(0) == 2

    # Check device
    assert item["src_tokens"].device == device
    assert item["tgt_tokens"].device == device

    # Check content after truncation
    assert torch.all(item["src_tokens"] == torch.tensor([1, 2], device=device))
    assert torch.all(item["tgt_tokens"] == torch.tensor([1, 32], device=device))  # BOS + first token

def test_transformer_collator(sample_sequences, device):
    """Test the TransformerCollator class."""
    collator = TransformerCollator(pad_idx=0)

    # Create a batch of examples
    batch = [
        {"src_tokens": src, "tgt_tokens": tgt}
        for src, tgt in zip(sample_sequences["source"], sample_sequences["target"])
    ]

    # Collate the batch
    collated = collator(batch)

    # Check structure
    assert isinstance(collated, dict)
    assert "src" in collated
    assert "tgt" in collated

    # Check shapes
    assert collated["src"].shape == (10, 4)  # batch_size=10, max_src_len=4
    assert collated["tgt"].shape == (10, 4)  # batch_size=10, max_tgt_len=4

    # Check device
    assert collated["src"].device == device
    assert collated["tgt"].device == device

def test_transformer_data_module_initialization(sample_sequences):
    """Test initialization of TransformerDataModule."""
    data_module = TransformerDataModule(
        source_sequences=sample_sequences["source"],
        target_sequences=sample_sequences["target"],
        batch_size=2,
        max_src_len=5,
        max_tgt_len=5,
        pad_idx=0,
        bos_idx=1,
        eos_idx=2,
        val_split=0.2,
        shuffle=True,
        num_workers=1
    )

    assert data_module.batch_size == 2
    assert data_module.max_src_len == 5
    assert data_module.max_tgt_len == 5
    assert data_module.pad_idx == 0
    assert data_module.bos_idx == 1
    assert data_module.eos_idx == 2
    assert data_module.val_split == 0.2
    assert data_module.shuffle is True
    assert data_module.num_workers == 1

def test_transformer_data_module_dataloaders(sample_sequences, device):
    """Test creation of dataloaders in TransformerDataModule."""
    data_module = TransformerDataModule(
        source_sequences=sample_sequences["source"],
        target_sequences=sample_sequences["target"],
        batch_size=2,
        max_src_len=5,
        max_tgt_len=5,
        pad_idx=0,
        bos_idx=1,
        eos_idx=2,
        val_split=0.2,
        shuffle=True,
        num_workers=1
    )

    # Get dataloaders
    train_dataloader = data_module.get_train_dataloader()
    val_dataloader = data_module.get_val_dataloader()

    # Check dataloader types
    assert isinstance(train_dataloader, torch.utils.data.DataLoader)
    assert isinstance(val_dataloader, torch.utils.data.DataLoader)

    # Check batch sizes
    assert train_dataloader.batch_size == 2
    assert val_dataloader.batch_size == 2

    # Check that we can iterate over the dataloaders
    train_batch = next(iter(train_dataloader))
    val_batch = next(iter(val_dataloader))

    # Check batch structure
    for batch in [train_batch, val_batch]:
        assert isinstance(batch, dict)
        assert "src" in batch
        assert "tgt" in batch
        assert isinstance(batch["src"], torch.Tensor)
        assert isinstance(batch["tgt"], torch.Tensor)
        assert batch["src"].device == device
        assert batch["tgt"].device == device
