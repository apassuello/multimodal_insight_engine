import os
import sys
import time
import torch
import random
from typing import List, Dict, Tuple, Optional
from tqdm import tqdm
import matplotlib.pyplot as plt
import pytest
# Add parent directory to path to import local modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.data.europarl_dataset import EuroparlDataset
from src.data.tokenization import (
    BPETokenizer,
    WhitespaceTokenizer,
    Vocabulary,
    clean_text,
    create_transformer_dataloaders,
)

@pytest.fixture
def en_tokenizer():
    """Fixture for English BPE tokenizer."""
    tokenizer = BPETokenizer(num_merges=100)  # Small vocab for testing
    tokenizer.train(
        texts=["Hello world", "This is a test", "Machine learning is fun"],
        vocab_size=256,  # Small vocab for testing
        min_frequency=1,
        show_progress=False
    )
    return tokenizer

@pytest.fixture
def de_tokenizer():
    """Fixture for German BPE tokenizer."""
    tokenizer = BPETokenizer(num_merges=100)  # Small vocab for testing
    tokenizer.train(
        texts=["Hallo Welt", "Dies ist ein Test", "Maschinelles Lernen macht Spaß"],
        vocab_size=256,  # Small vocab for testing
        min_frequency=1,
        show_progress=False
    )
    return tokenizer

def set_seed(seed=42):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train_bpe_tokenizers(
    en_texts: List[str],
    de_texts: List[str],
    vocab_size: int = 32000,
    min_frequency: int = 2,
    save_dir: str = "models/tokenizers",
) -> Tuple[BPETokenizer, BPETokenizer]:
    """
    Train BPE tokenizers for German (source) and English (target).
    
    Args:
        en_texts: List of English texts (target language)
        de_texts: List of German texts (source language)
        vocab_size: Target vocabulary size
        min_frequency: Minimum token frequency
        save_dir: Directory to save tokenizers
        
    Returns:
        Tuple of (german_tokenizer, english_tokenizer)
    """
    print(f"Training BPE tokenizers with vocab size {vocab_size}...")
    
    # Create save directories
    os.makedirs(f"{save_dir}/de", exist_ok=True)  # German directory first
    os.makedirs(f"{save_dir}/en", exist_ok=True)  # English directory second
    
    # Train German tokenizer (source language)
    print("Training German tokenizer (source language)...")
    start_time = time.time()
    de_tokenizer = BPETokenizer(num_merges=vocab_size-256)
    de_tokenizer.train(
        texts=de_texts,
        vocab_size=vocab_size,
        min_frequency=min_frequency,
        show_progress=True,
    )
    de_time = time.time() - start_time
    print(f"German tokenizer trained in {de_time:.2f}s")
    print(f"German vocabulary size: {de_tokenizer.vocab_size}")
    
    # Train English tokenizer (target language)
    print("Training English tokenizer (target language)...")
    start_time = time.time()
    en_tokenizer = BPETokenizer(num_merges=vocab_size-256)
    en_tokenizer.train(
        texts=en_texts,
        vocab_size=vocab_size,
        min_frequency=min_frequency,
        show_progress=True,
    )
    en_time = time.time() - start_time
    print(f"English tokenizer trained in {en_time:.2f}s")
    print(f"English vocabulary size: {en_tokenizer.vocab_size}")
    
    # Save tokenizers
    print("Saving tokenizers...")
    de_tokenizer.save_pretrained(f"{save_dir}/de")
    en_tokenizer.save_pretrained(f"{save_dir}/en")
    
    # Return German tokenizer first (source), then English tokenizer (target)
    return de_tokenizer, en_tokenizer


def test_tokenizers(
    en_tokenizer: BPETokenizer,
    de_tokenizer: BPETokenizer, 
    example_sentences: List[str] = None,
) -> None:
    """
    Test the trained tokenizers on example sentences.
    
    Args:
        en_tokenizer: English tokenizer
        de_tokenizer: German tokenizer
        example_sentences: Optional list of example sentences
    """
    if example_sentences is None:
        example_sentences = [
            "Hello, how are you?",
            "I am learning machine translation.",
            "Transformers are powerful models for NLP.",
            "This is an example of English to German translation."
        ]
    
    print("\n=== Testing Tokenizers ===")
    
    for sentence in example_sentences:
        print(f"\nOriginal: {sentence}")
        
        # Tokenize with BPE
        bpe_tokens = en_tokenizer.tokenize(sentence)
        bpe_ids = en_tokenizer.encode(sentence)
        
        print(f"BPE Tokens ({len(bpe_tokens)}): {bpe_tokens}")
        print(f"BPE IDs: {bpe_ids[:10]}..." if len(bpe_ids) > 10 else f"BPE IDs: {bpe_ids}")
        
        # Decode back to text
        decoded = en_tokenizer.decode(bpe_ids)
        print(f"Decoded: {decoded}")
        
        # Compare with whitespace tokenizer for reference
        ws_tokenizer = WhitespaceTokenizer()
        ws_tokens = ws_tokenizer.tokenize(sentence)
        print(f"Whitespace Tokens ({len(ws_tokens)}): {ws_tokens}")


def analyze_tokenization(
    en_tokenizer: BPETokenizer,
    en_texts: List[str],
    sample_size: int = 1000,
) -> None:
    """
    Analyze tokenization statistics compared to whitespace tokenization.
    
    Args:
        en_tokenizer: English tokenizer
        en_texts: List of English texts
        sample_size: Number of texts to sample
    """
    print("\n=== Tokenization Analysis ===")
    
    # Create whitespace tokenizer for comparison
    ws_tokenizer = WhitespaceTokenizer()
    
    # Sample texts
    if sample_size > len(en_texts):
        sample_size = len(en_texts)
    
    sampled_texts = random.sample(en_texts, sample_size)
    
    # Collect statistics
    bpe_token_counts = []
    ws_token_counts = []
    
    for text in tqdm(sampled_texts, desc="Analyzing tokenization"):
        bpe_tokens = en_tokenizer.tokenize(text)
        ws_tokens = ws_tokenizer.tokenize(text)
        
        bpe_token_counts.append(len(bpe_tokens))
        ws_token_counts.append(len(ws_tokens))
    
    # Calculate statistics
    avg_bpe_tokens = sum(bpe_token_counts) / len(bpe_token_counts)
    avg_ws_tokens = sum(ws_token_counts) / len(ws_token_counts)
    
    print(f"Average BPE tokens per text: {avg_bpe_tokens:.2f}")
    print(f"Average whitespace tokens per text: {avg_ws_tokens:.2f}")
    print(f"Ratio (BPE/Whitespace): {avg_bpe_tokens/avg_ws_tokens:.2f}")
    
    # Plot token count distributions
    plt.figure(figsize=(10, 6))
    plt.hist(ws_token_counts, alpha=0.5, label='Whitespace', bins=30)
    plt.hist(bpe_token_counts, alpha=0.5, label='BPE', bins=30)
    plt.xlabel('Number of Tokens')
    plt.ylabel('Frequency')
    plt.title('Token Count Distribution: BPE vs Whitespace')
    plt.legend()
    plt.savefig('token_distribution.png')
    plt.close()
    
    print("Token distribution histogram saved as 'token_distribution.png'")


def demonstrate_oov_handling(en_tokenizer: BPETokenizer) -> None:
    """
    Demonstrate how BPE handles out-of-vocabulary words.
    
    Args:
        en_tokenizer: English tokenizer
    """
    print("\n=== OOV Word Handling ===")
    
    # Create some words that might not be in the training corpus
    oov_words = [
        "untrained",
        "preprocessing",
        "tokenizations",
        "transformer",
        "anthropically",  # Made-up word related to Anthropic
        "multimodality",
    ]
    
    for word in oov_words:
        tokens = en_tokenizer.tokenize(word)
        print(f"Word: {word}")
        print(f"Decomposition: {tokens}")
        print(f"Token count: {len(tokens)}")
        print()


def prepare_for_transformer(
    de_tokenizer: BPETokenizer,  # German tokenizer (source)
    en_tokenizer: BPETokenizer,  # English tokenizer (target)
    de_texts: List[str],         # German texts (source) 
    en_texts: List[str],         # English texts (target)
    max_length: int = 128,
    batch_size: int = 32,
) -> None:
    """
    Demonstrate how to prepare data for transformer training (German to English).
    
    Args:
        de_tokenizer: German tokenizer (source language)
        en_tokenizer: English tokenizer (target language)
        de_texts: List of German texts (source language)
        en_texts: List of English texts (target language)
        max_length: Maximum sequence length
        batch_size: Batch size
    """
    print("\n=== Preparing for Transformer Training (German → English) ===")
    
    # Create training data loaders
    print("Creating source (German) dataloader...")
    de_train_dataloader, _ = create_transformer_dataloaders(
        train_texts=de_texts[:1000],  # Use a subset for demonstration
        tokenizer=de_tokenizer,
        batch_size=batch_size,
        max_length=max_length,
    )
    
    print("Creating target (English) dataloader...")
    en_train_dataloader, _ = create_transformer_dataloaders(
        train_texts=en_texts[:1000],  # Use a subset for demonstration
        tokenizer=en_tokenizer,
        batch_size=batch_size,
        max_length=max_length,
    )
    
    # Show examples of batches
    print("\nExample German batch (source language):")
    de_batch = next(iter(de_train_dataloader))
    print(f"Input IDs shape: {de_batch['input_ids'].shape}")
    print(f"Attention mask shape: {de_batch['attention_mask'].shape}")
    
    # Decode a sample sequence
    sample_seq = de_batch['input_ids'][0].tolist()
    sample_mask = de_batch['attention_mask'][0].tolist()
    
    # Only include tokens where attention mask is 1 (exclude padding)
    active_tokens = [idx for idx, mask in zip(sample_seq, sample_mask) if mask == 1]
    
    print(f"Sample sequence length: {len(active_tokens)}")
    print(f"Decoded German sample: {de_tokenizer.decode(active_tokens)}")
    
    print("\nInstructions for German to English translation training:")
    print("1. Use the German tokenizer to tokenize source texts")
    print("2. Use the English tokenizer to tokenize target texts")
    print("3. Create dataloaders using create_transformer_dataloaders()")
    print("4. Feed the batches to your transformer model")
    print("5. For translation, tokenize with German tokenizer and decode with English tokenizer")


def main():
    # Set seed for reproducibility
    set_seed(42)
    
    print("Loading Europarl dataset...")
    # Load the Europarl dataset instead of IWSLT
    train_dataset = EuroparlDataset(
        data_dir="data/europarl",  # Adjust this path to match your directory structure
        src_lang="de",
        tgt_lang="en",
        max_examples=100000  # Adjust as needed
    )
    
    # Split data into source and target languages
    en_texts = train_dataset.src_data
    de_texts = train_dataset.tgt_data
    
    print(f"Loaded {len(en_texts)} training examples")
    print(f"Example English text: {en_texts[0]}")
    print(f"Example German text: {de_texts[0]}")
    
    # Rest of your code remains the same
    # Train tokenizers
    en_tokenizer, de_tokenizer = train_bpe_tokenizers(
        en_texts=en_texts,
        de_texts=de_texts,
        vocab_size=8000,  # Good starting point for translation
        min_frequency=2,
        save_dir="models/tokenizers",
    )
    
    # Test tokenizers
    test_tokenizers(en_tokenizer, de_tokenizer)
    
    # Analyze tokenization
    analyze_tokenization(
        en_tokenizer=en_tokenizer,
        en_texts=en_texts,
        sample_size=1000,  # Analyze 1000 examples
    )
    
    # Demonstrate OOV handling
    demonstrate_oov_handling(en_tokenizer)
    
    # Prepare for transformer training
    prepare_for_transformer(
        en_tokenizer=en_tokenizer,
        de_tokenizer=de_tokenizer,
        en_texts=en_texts,
        de_texts=de_texts,
        max_length=128,
        batch_size=32,
    )


if __name__ == "__main__":
    main()
