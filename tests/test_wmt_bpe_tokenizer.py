import pytest
import torch
from src.data.tokenization.wmt_bpe_tokenizer import WMTBPETokenizer

@pytest.fixture
def device():
    """Get the device to use for testing."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

@pytest.fixture
def sample_vocab():
    """Create a sample vocabulary for testing."""
    return {
        "hello": 0,
        "world": 1,
        "test": 2,
        "token": 3,
        "<unk>": 4,
        "h": 5,
        "e": 6,
        "l": 7,
        "o": 8,
        "w": 9,
        "r": 10,
        "d": 11,
        "t": 12,
        "s": 13,
        "n": 14
    }

@pytest.fixture
def sample_merges():
    """Create sample merge operations for testing."""
    return [
        # Complete merge sequence for "hello"
        ("h", "e"), ("he", "l"), ("hel", "l"), ("hell", "o"),
        # Complete merge sequence for "world"
        ("w", "o"), ("wo", "r"), ("wor", "l"), ("worl", "d")
    ]

@pytest.fixture
def tokenizer(device, sample_vocab, sample_merges):
    """Create a tokenizer instance for testing."""
    return WMTBPETokenizer(
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
    assert processed == "hello, world!"  # Should be lowercase

def test_tokenize_word_optimized(tokenizer):
    """Test single word tokenization."""
    # Test with a word that can be merged
    tokens = tokenizer._tokenize_word_optimized("hello")
    assert tokens == ["hello"]  # Should merge to a single token
    
    # Test with a word that can't be fully merged
    tokens = tokenizer._tokenize_word_optimized("test")
    assert len(tokens) > 0  # Should be split into subwords
    
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
    print(token_ids)
    assert token_ids[0] == tokenizer.vocab["hello"]
    assert token_ids[1] == tokenizer.vocab["world"]

def test_decode(tokenizer):
    """Test decoding token IDs back to text."""
    original_text = "hello world"
    token_ids = tokenizer.encode(original_text)
    decoded_text = tokenizer.decode(token_ids)
    assert isinstance(decoded_text, str)
    assert decoded_text == original_text

def test_batch_tokenize(tokenizer):
    """Test batch tokenization."""
    texts = ["hello world", "test token"]
    batch_tokens = tokenizer.batch_tokenize(texts)
    assert len(batch_tokens) == len(texts)
    assert all(isinstance(tokens, list) for tokens in batch_tokens)
    assert all(all(isinstance(token, str) for token in tokens) for tokens in batch_tokens)

def test_batch_encode(tokenizer):
    """Test batch encoding."""
    texts = ["hello world", "test token"]
    batch_ids = tokenizer.batch_encode(texts)
    assert len(batch_ids) == len(texts)
    assert all(isinstance(ids, list) for ids in batch_ids)
    assert all(all(isinstance(id_, int) for id_ in ids) for ids in batch_ids)

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

def test_unknown_token_handling(tokenizer):
    """Test handling of unknown tokens."""
    # Test with unknown word
    text = "unknown"
    token_ids = tokenizer.encode(text)
    print(token_ids)
    assert all(id_ == tokenizer.vocab["<unk>"] for id_ in token_ids)
    
    # Test decoding unknown token IDs
    unknown_ids = [999, 1000]  # IDs not in vocab
    decoded = tokenizer.decode(unknown_ids)
    assert all(token == "<unk>" for token in decoded.split()) 