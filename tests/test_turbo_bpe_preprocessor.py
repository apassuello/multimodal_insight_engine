import pytest
import os
import torch
import pickle
from collections import namedtuple
from src.data.tokenization.turbo_bpe_preprocessor import TurboBPEPreprocessor

# Mock dataset class for testing
Dataset = namedtuple('Dataset', ['src_data', 'tgt_data'])

class MockTokenizer:
    def __init__(self):
        self.special_tokens = {
            "bos_token_idx": 1,
            "eos_token_idx": 2
        }
        self.vocab = {
            "hello": 3,
            "world": 4,
            "<unk>": 0
        }
        self.word_token_cache = {}
    
    def preprocess(self, text):
        return text.lower()
    
    def _tokenize_word(self, word):
        return [word]
    
    def token_to_index(self, token):
        return self.vocab.get(token, self.vocab["<unk>"])

@pytest.fixture
def temp_cache_dir(tmp_path):
    """Create a temporary cache directory."""
    cache_dir = tmp_path / "tokenizer_cache"
    cache_dir.mkdir()
    return str(cache_dir)

@pytest.fixture
def preprocessor(temp_cache_dir):
    """Create a preprocessor instance for testing."""
    return TurboBPEPreprocessor(cache_dir=temp_cache_dir)

@pytest.fixture
def sample_dataset():
    """Create a sample dataset for testing."""
    src_data = ["Hello world", "Test sentence", "Another example"]
    tgt_data = ["Hallo Welt", "Testsatz", "Noch ein Beispiel"]
    return Dataset(src_data=src_data, tgt_data=tgt_data)

@pytest.fixture
def mock_tokenizer():
    """Create a mock tokenizer for testing."""
    return MockTokenizer()

def test_initialization(preprocessor, temp_cache_dir):
    """Test preprocessor initialization."""
    assert preprocessor.cache_dir == temp_cache_dir
    assert os.path.exists(temp_cache_dir)
    assert isinstance(preprocessor.word_cache, dict)
    assert isinstance(preprocessor.dataset_cache, dict)
    assert preprocessor.optimal_batch_size > 0
    assert preprocessor.num_workers > 0
    assert isinstance(preprocessor.device, torch.device)

def test_generate_cache_key(preprocessor, sample_dataset):
    """Test cache key generation."""
    key = preprocessor._generate_cache_key(sample_dataset)
    assert isinstance(key, str)
    assert len(key) > 0

def test_cache_operations(preprocessor, sample_dataset, mock_tokenizer):
    """Test caching operations."""
    # Test saving to cache
    test_data = (["token1"], ["token2"])
    preprocessor.save_preprocessed_data(test_data, sample_dataset)
    
    # Test loading from cache
    cached_data = preprocessor.check_cached_preprocessed_data(sample_dataset)
    assert cached_data == test_data

def test_process_text_batch(preprocessor, mock_tokenizer):
    """Test batch text processing."""
    texts = ["hello world", "test"]
    result = preprocessor._process_text_batch(texts, mock_tokenizer)
    
    assert isinstance(result, list)
    assert len(result) == len(texts)
    assert all(isinstance(ids, list) for ids in result)
    
    # Test caching
    assert "hello world" in preprocessor.word_cache
    assert "test" in preprocessor.word_cache

def test_process_data_chunk(preprocessor, mock_tokenizer):
    """Test data chunk processing."""
    chunk_id = 0
    src_chunk = ["hello world"]
    tgt_chunk = ["hallo welt"]
    special_tokens = {
        'src_bos': 1,
        'src_eos': 2,
        'tgt_bos': 1,
        'tgt_eos': 2
    }
    
    result = preprocessor._process_data_chunk((
        chunk_id,
        src_chunk,
        tgt_chunk,
        mock_tokenizer,
        mock_tokenizer,
        special_tokens
    ))
    
    assert isinstance(result, tuple)
    assert len(result) == 3
    assert result[0] == chunk_id
    assert all(isinstance(seq, list) for seq in result[1])  # src sequences
    assert all(isinstance(seq, list) for seq in result[2])  # tgt sequences

def test_preprocess_with_caching(preprocessor, sample_dataset, mock_tokenizer):
    """Test full preprocessing with caching."""
    # First run - should process and cache
    src_sequences, tgt_sequences = preprocessor.preprocess_with_caching(
        sample_dataset,
        mock_tokenizer,
        mock_tokenizer
    )
    
    assert isinstance(src_sequences, list)
    assert isinstance(tgt_sequences, list)
    assert len(src_sequences) == len(sample_dataset.src_data)
    assert len(tgt_sequences) == len(sample_dataset.tgt_data)
    
    # Second run - should use cache
    cached_src, cached_tgt = preprocessor.preprocess_with_caching(
        sample_dataset,
        mock_tokenizer,
        mock_tokenizer
    )
    
    assert cached_src == src_sequences
    assert cached_tgt == tgt_sequences

def test_optimize_tokenizer(preprocessor, mock_tokenizer):
    """Test tokenizer optimization."""
    optimized = preprocessor.optimize_tokenizer_for_preprocessing(mock_tokenizer)
    
    # Test that optimization added necessary attributes
    assert hasattr(optimized, 'word_token_cache')
    assert hasattr(optimized, '_tokenize_word_original')
    
    # Test optimized tokenization
    word = "test"
    result = optimized._tokenize_word(word)
    assert isinstance(result, list)
    assert word in optimized.word_token_cache

def test_force_regenerate(preprocessor, sample_dataset, mock_tokenizer):
    """Test force regeneration of preprocessed data."""
    # First run - normal processing
    first_run = preprocessor.preprocess_with_caching(
        sample_dataset,
        mock_tokenizer,
        mock_tokenizer
    )
    
    # Second run with force_regenerate=True
    second_run = preprocessor.preprocess_with_caching(
        sample_dataset,
        mock_tokenizer,
        mock_tokenizer,
        force_regenerate=True
    )
    
    # Results should be the same but should have regenerated
    assert second_run == first_run

def test_cache_size_management(preprocessor, mock_tokenizer):
    """Test that cache size is properly managed."""
    # Generate many texts to process
    texts = [f"text{i}" for i in range(1000)]
    
    # Process texts
    preprocessor._process_text_batch(texts, mock_tokenizer)
    
    # Check that cache size is reasonable
    assert len(preprocessor.word_cache) <= 100000  # Max cache size

def test_special_token_handling(preprocessor, sample_dataset, mock_tokenizer):
    """Test handling of special tokens in preprocessing."""
    src_sequences, tgt_sequences = preprocessor.preprocess_with_caching(
        sample_dataset,
        mock_tokenizer,
        mock_tokenizer
    )
    
    # Check that special tokens are added correctly
    for sequence in src_sequences:
        assert sequence[0] == mock_tokenizer.special_tokens["bos_token_idx"]
        assert sequence[-1] == mock_tokenizer.special_tokens["eos_token_idx"]
    
    for sequence in tgt_sequences:
        assert sequence[0] == mock_tokenizer.special_tokens["bos_token_idx"]
        assert sequence[-1] == mock_tokenizer.special_tokens["eos_token_idx"] 