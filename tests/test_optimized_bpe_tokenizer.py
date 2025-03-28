import pytest
import torch
import os
import tempfile
from src.data.tokenization.optimized_bpe_tokenizer import OptimizedBPETokenizer
from src.data.tokenization.vocabulary import Vocabulary

@pytest.fixture
def device():
    """Get the device to use for testing."""
    return torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")

@pytest.fixture
def sample_vocab():
    """Create a sample vocabulary for testing."""
    vocab = Vocabulary()
    for token in ["hello", "world", "test", "token"]:
        vocab.add_token(token)
    return vocab

@pytest.fixture
def sample_merges():
    """Create sample merge operations for testing."""
    return [("h", "e"), ("he", "l"), ("hel", "lo"), ("w", "o"), ("wo", "r"), ("wor", "ld")]

@pytest.fixture
def tokenizer(device, sample_vocab, sample_merges):
    """Create a tokenizer instance for testing."""
    return OptimizedBPETokenizer(
        vocab=sample_vocab,
        merges=sample_merges,
        num_merges=10,
        device=device.type
    )

def test_tokenizer_initialization(tokenizer, device, sample_vocab, sample_merges):
    """Test tokenizer initialization."""
    assert tokenizer.vocab == sample_vocab
    assert tokenizer.merges == sample_merges
    assert tokenizer.num_merges == 10
    assert tokenizer.device == device
    assert len(tokenizer.token_cache) == 0
    assert len(tokenizer.word_token_cache) == 0

def test_preprocess(tokenizer):
    """Test text preprocessing."""
    text = "Hello, World!"
    processed = tokenizer.preprocess(text)
    assert processed == "hello world"  # Should be lowercase and cleaned

def test_tokenize_word_optimized(tokenizer):
    """Test single word tokenization."""
    # Test with a word that can be merged
    tokens = tokenizer._tokenize_word_optimized("hello")
    assert tokens == ["hello"]  # Should merge to a single token
    
    # Test with a word that can't be fully merged
    tokens = tokenizer._tokenize_word_optimized("test")
    assert len(tokens) > 1  # Should be split into subwords
    
    # Test cache functionality
    assert "hello" in tokenizer.word_token_cache
    assert tokenizer.word_token_cache["hello"] == ["hello"]

def test_tokenize(tokenizer):
    """Test full text tokenization."""
    text = "hello world"
    tokens = tokenizer.tokenize(text)
    assert len(tokens) > 0
    assert "hello" in tokens
    assert "world" in tokens
    
    # Test cache functionality
    assert text in tokenizer.token_cache
    assert tokenizer.token_cache[text] == tokens

def test_encode(tokenizer):
    """Test encoding text to token IDs."""
    text = "hello world"
    token_ids = tokenizer.encode(text)
    assert isinstance(token_ids, list)
    assert all(isinstance(id_, int) for id_ in token_ids)
    assert len(token_ids) > 0

def test_batch_encode_optimized(tokenizer):
    """Test optimized batch encoding."""
    texts = ["hello world", "test token", "hello test"]
    
    # Test without batch size
    encoded = tokenizer.batch_encode_optimized(texts)
    assert len(encoded) == len(texts)
    assert all(isinstance(seq, list) for seq in encoded)
    assert all(all(isinstance(id_, int) for id_ in seq) for seq in encoded)
    
    # Test with batch size
    encoded_batched = tokenizer.batch_encode_optimized(texts, batch_size=2)
    assert encoded == encoded_batched  # Results should be the same

def test_process_batch(tokenizer):
    """Test batch processing."""
    texts = ["hello world", "test token"]
    results = tokenizer._process_batch(texts)
    assert len(results) == len(texts)
    assert all(isinstance(seq, list) for seq in results)

def test_save_and_load_pretrained(tokenizer, tmp_path):
    """Test saving and loading the tokenizer."""
    # Save the tokenizer
    save_path = str(tmp_path / "tokenizer")
    tokenizer.save_pretrained(save_path)
    
    # Check that files were created
    assert os.path.exists(save_path)
    assert os.path.exists(os.path.join(save_path, "vocab.json"))
    assert os.path.exists(os.path.join(save_path, "merges.txt"))
    
    # Load the tokenizer
    loaded_tokenizer = OptimizedBPETokenizer.from_pretrained(save_path)
    
    # Test that loaded tokenizer works the same
    text = "hello world"
    original_tokens = tokenizer.tokenize(text)
    loaded_tokens = loaded_tokenizer.tokenize(text)
    assert original_tokens == loaded_tokens

def test_train(tokenizer):
    """Test tokenizer training."""
    texts = [
        "hello world",
        "test token",
        "hello test",
        "world token"
    ]
    
    # Train the tokenizer
    tokenizer.train(texts, vocab_size=100, min_frequency=1)
    
    # Verify that merges were learned
    assert len(tokenizer.merges) > 0
    
    # Test tokenization with trained merges
    tokens = tokenizer.tokenize("hello world")
    assert len(tokens) > 0

def test_special_tokens(tokenizer):
    """Test special tokens property."""
    special_tokens = tokenizer.special_tokens
    assert isinstance(special_tokens, dict)
    assert "<pad>" in special_tokens
    assert "<unk>" in special_tokens
    assert "<bos>" in special_tokens
    assert "<eos>" in special_tokens

def test_vocab_size(tokenizer):
    """Test vocab size property."""
    size = tokenizer.vocab_size
    assert isinstance(size, int)
    assert size > 0

def test_decode(tokenizer):
    """Test decoding token IDs back to text."""
    text = "hello world"
    token_ids = tokenizer.encode(text)
    decoded = tokenizer.decode(token_ids)
    assert isinstance(decoded, str)
    assert len(decoded) > 0

def test_tensor_lookup_creation(tokenizer):
    """Test creation of tensor lookup tables."""
    tokenizer._create_tensor_lookup()
    assert hasattr(tokenizer, "single_char_merge_indices")
    assert hasattr(tokenizer, "single_char_merge_pairs")
    assert isinstance(tokenizer.single_char_merge_indices, torch.Tensor)
    assert isinstance(tokenizer.single_char_merge_pairs, torch.Tensor)

def test_cache_size_limit(tokenizer):
    """Test that cache size limits are respected."""
    # Set a small cache size
    tokenizer.cache_size = 2
    
    # Add more items than the cache size
    texts = ["hello", "world", "test", "token"]
    for text in texts:
        tokenizer._tokenize_word_optimized(text)
    
    # Check that cache size is not exceeded
    assert len(tokenizer.word_token_cache) <= tokenizer.cache_size 