import os
import pytest
from tempfile import TemporaryDirectory
from src.data.europarl_dataset import EuroparlDataset

def create_test_data(temp_dir: str, src_lang: str = "de", tgt_lang: str = "en"):
    """Helper function to create test data files"""
    # Create test data in pattern 4 format (simple .txt files)
    src_data = [
        "Das ist ein Test.",
        "Wie geht es dir?",
        "Guten Morgen!",
        "Auf Wiedersehen!"
    ]
    tgt_data = [
        "This is a test.",
        "How are you?",
        "Good morning!",
        "Goodbye!"
    ]
    
    src_file = os.path.join(temp_dir, f"{src_lang}.txt")
    tgt_file = os.path.join(temp_dir, f"{tgt_lang}.txt")
    
    with open(src_file, 'w', encoding='utf-8') as f:
        f.write('\n'.join(src_data))
    
    with open(tgt_file, 'w', encoding='utf-8') as f:
        f.write('\n'.join(tgt_data))
    
    return src_data, tgt_data

def test_europarl_dataset_initialization():
    """Test basic initialization of the dataset"""
    dataset = EuroparlDataset(data_dir="data/europarl/")
    assert dataset.data_dir == "data/europarl/"
    assert dataset.src_lang == "de"
    assert dataset.tgt_lang == "en"
    assert dataset.max_examples is None

def test_europarl_dataset_loading():
    """Test data loading functionality"""
    with TemporaryDirectory() as temp_dir:
        src_data, tgt_data = create_test_data(temp_dir)
        
        dataset = EuroparlDataset(
            data_dir=temp_dir,
            src_lang="de",
            tgt_lang="en"
        )
        
        assert len(dataset.src_data) == len(src_data)
        assert len(dataset.tgt_data) == len(tgt_data)
        assert set(dataset.src_data) == set(src_data)
        assert set(dataset.tgt_data) == set(tgt_data)

def test_europarl_dataset_max_examples():
    """Test max_examples parameter"""
    with TemporaryDirectory() as temp_dir:
        create_test_data(temp_dir)
        
        dataset = EuroparlDataset(
            data_dir=temp_dir,
            max_examples=2
        )
        
        assert len(dataset.src_data) == 2
        assert len(dataset.tgt_data) == 2

def test_europarl_dataset_missing_files():
    """Test error handling for missing files"""
    with TemporaryDirectory() as temp_dir:
        with pytest.raises(FileNotFoundError):
            EuroparlDataset(data_dir=temp_dir)

def test_europarl_dataset_empty_lines():
    """Test handling of empty lines"""
    with TemporaryDirectory() as temp_dir:
        # Create data with empty lines
        src_data = ["Line 1", "", "Line 3"]
        tgt_data = ["Line 1", "", "Line 3"]
        
        src_file = os.path.join(temp_dir, "de.txt")
        tgt_file = os.path.join(temp_dir, "en.txt")
        
        with open(src_file, 'w', encoding='utf-8') as f:
            f.write('\n'.join(src_data))
        with open(tgt_file, 'w', encoding='utf-8') as f:
            f.write('\n'.join(tgt_data))
            
        dataset = EuroparlDataset(data_dir=temp_dir)
        
        # Empty lines should be filtered out
        assert len(dataset.src_data) == 2
        assert len(dataset.tgt_data) == 2
        assert "" not in dataset.src_data
        assert "" not in dataset.tgt_data

def test_europarl_dataset_different_lengths():
    """Test handling of source and target files with different lengths"""
    with TemporaryDirectory() as temp_dir:
        # Create data with different lengths
        src_data = ["Line 1", "Line 2", "Line 3"]
        tgt_data = ["Line 1", "Line 2"]
        
        src_file = os.path.join(temp_dir, "de.txt")
        tgt_file = os.path.join(temp_dir, "en.txt")
        
        with open(src_file, 'w', encoding='utf-8') as f:
            f.write('\n'.join(src_data))
        with open(tgt_file, 'w', encoding='utf-8') as f:
            f.write('\n'.join(tgt_data))
            
        dataset = EuroparlDataset(data_dir=temp_dir)
        
        # Should truncate to shorter length
        assert len(dataset.src_data) == 2
        assert len(dataset.tgt_data) == 2 