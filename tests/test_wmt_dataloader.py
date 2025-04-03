import pytest
import os
import tempfile
from typing import List, Tuple
from src.data.wmt_dataloader import WMTDataLoader

@pytest.fixture
def sample_data() -> Tuple[List[str], List[str]]:
    """Create sample parallel data for testing."""
    source_data = [
        "This is a test sentence.",
        "Another test sentence here.",
        "The third test sentence.",
        "A fourth test sentence.",
        "The fifth and final test sentence."
    ]
    target_data = [
        "C'est une phrase de test.",
        "Une autre phrase de test ici.",
        "La troisième phrase de test.",
        "Une quatrième phrase de test.",
        "La cinquième et dernière phrase de test."
    ]
    return source_data, target_data

@pytest.fixture
def temp_data_dir(sample_data, tmp_path):
    """Create a temporary directory with sample WMT data files."""
    source_data, target_data = sample_data
    
    # Create the data directory
    data_dir = tmp_path / "wmt_test_data"
    data_dir.mkdir()
    
    # Write source and target files
    src_file = data_dir / "news-commentary-v9.en-fr.en"
    tgt_file = data_dir / "news-commentary-v9.en-fr.fr"
    
    with open(src_file, 'w', encoding='utf-8') as f:
        f.write('\n'.join(source_data))
    
    with open(tgt_file, 'w', encoding='utf-8') as f:
        f.write('\n'.join(target_data))
    
    return str(data_dir)

def test_dataloader_initialization(temp_data_dir):
    """Test initialization of WMTDataLoader."""
    dataloader = WMTDataLoader(
        data_dir=temp_data_dir,
        source_lang='en',
        target_lang='fr',
        batch_size=2
    )
    
    assert dataloader.data_dir == temp_data_dir
    assert dataloader.source_lang == 'en'
    assert dataloader.target_lang == 'fr'
    assert dataloader.batch_size == 2
    assert dataloader.max_examples is None
    assert len(dataloader.source_data) == 5
    assert len(dataloader.target_data) == 5

def test_load_data(temp_data_dir, sample_data):
    """Test data loading functionality."""
    source_data, target_data = sample_data
    
    dataloader = WMTDataLoader(
        data_dir=temp_data_dir,
        source_lang='en',
        target_lang='fr',
        shuffle=False
    )
    
    assert dataloader.source_data == source_data
    assert dataloader.target_data == target_data

def test_max_examples(temp_data_dir):
    """Test max_examples parameter."""
    dataloader = WMTDataLoader(
        data_dir=temp_data_dir,
        source_lang='en',
        target_lang='fr',
        max_examples=3
    )
    
    assert len(dataloader.source_data) == 3
    assert len(dataloader.target_data) == 3

def test_batch_iteration(temp_data_dir):
    """Test batch iteration functionality."""
    dataloader = WMTDataLoader(
        data_dir=temp_data_dir,
        source_lang='en',
        target_lang='fr',
        batch_size=2
    )
    
    batches = list(dataloader)
    assert len(batches) == 3  # 5 examples with batch_size=2 should give 3 batches
    
    # Check first batch
    src_batch, tgt_batch = batches[0]
    assert len(src_batch) == 2
    assert len(tgt_batch) == 2
    
    # Check last batch (should be smaller)
    src_batch, tgt_batch = batches[-1]
    assert len(src_batch) == 1
    assert len(tgt_batch) == 1

def test_file_not_found(tmp_path):
    """Test handling of missing data files."""
    with pytest.raises(FileNotFoundError):
        WMTDataLoader(
            data_dir=str(tmp_path),
            source_lang='en',
            target_lang='fr'
        )

def test_different_length_files(tmp_path):
    """Test handling of files with different lengths."""
    # Create data files with different lengths
    data_dir = tmp_path / "wmt_test_data"
    data_dir.mkdir()
    
    src_file = data_dir / "news-commentary-v9.en-fr.en"
    tgt_file = data_dir / "news-commentary-v9.en-fr.fr"
    
    with open(src_file, 'w', encoding='utf-8') as f:
        f.write('Source line 1\nSource line 2\nSource line 3\n')
    
    with open(tgt_file, 'w', encoding='utf-8') as f:
        f.write('Target line 1\nTarget line 2\n')  # One line less
    
    dataloader = WMTDataLoader(
        data_dir=str(data_dir),
        source_lang='en',
        target_lang='fr'
    )
    
    # Should truncate to shorter length
    assert len(dataloader.source_data) == len(dataloader.target_data)
    assert len(dataloader.source_data) == 2

def test_empty_lines_filtering(tmp_path):
    """Test filtering of empty lines."""
    data_dir = tmp_path / "wmt_test_data"
    data_dir.mkdir()
    
    src_file = data_dir / "news-commentary-v9.en-fr.en"
    tgt_file = data_dir / "news-commentary-v9.en-fr.fr"
    
    with open(src_file, 'w', encoding='utf-8') as f:
        f.write('Source line 1\n\nSource line 3\n')
    
    with open(tgt_file, 'w', encoding='utf-8') as f:
        f.write('Target line 1\n\nTarget line 3\n')
    
    dataloader = WMTDataLoader(
        data_dir=str(data_dir),
        source_lang='en',
        target_lang='fr'
    )
    
    # Should filter out empty lines
    assert len(dataloader.source_data) == 2
    assert len(dataloader.target_data) == 2
    assert '' not in dataloader.source_data
    assert '' not in dataloader.target_data

def test_seed_reproducibility(temp_data_dir):
    """Test that setting the same seed produces the same data order."""
    # Create two dataloaders with the same seed
    dataloader1 = WMTDataLoader(
        data_dir=temp_data_dir,
        source_lang='en',
        target_lang='fr',
        seed=42
    )
    
    dataloader2 = WMTDataLoader(
        data_dir=temp_data_dir,
        source_lang='en',
        target_lang='fr',
        seed=42
    )
    
    # Data should be in the same order
    assert dataloader1.source_data == dataloader2.source_data
    assert dataloader1.target_data == dataloader2.target_data 