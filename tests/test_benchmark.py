"""
Benchmark preprocessing performance for translation tasks.

This script compares the original preprocessing method with optimized versions
to find the most efficient approach for Apple Silicon hardware.

Usage:
    python benchmark_preprocessing.py

The script will:
1. Load a sample dataset
2. Run each preprocessing method on the same data
3. Measure and report performance metrics
"""

import os
import sys
import time

import matplotlib.pyplot as plt
import torch
from tqdm import tqdm

# Add parent directory to path to import local modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.europarl_dataset import EuroparlDataset

# Import from the project
from src.data.tokenization import BPETokenizer

# Import optimized preprocessing functions
from src.data.tokenization.turbo_bpe_preprocessor import turbo_preprocess_data


# Original preprocessing function - current implementation
def original_preprocess_data_with_bpe(dataset, de_tokenizer, en_tokenizer, batch_size=1000):
    """Original preprocessing implementation (from your current code)."""
    src_sequences = []
    tgt_sequences = []

    # Get special token indices
    src_bos_idx = de_tokenizer.special_tokens["bos_token_idx"]
    src_eos_idx = de_tokenizer.special_tokens["eos_token_idx"]
    tgt_bos_idx = en_tokenizer.special_tokens["bos_token_idx"]
    tgt_eos_idx = en_tokenizer.special_tokens["eos_token_idx"]

    # Process in batches
    for i in tqdm(range(0, len(dataset.src_data), batch_size),
                 total=len(dataset.src_data) // batch_size + 1,
                 desc="Preprocessing batches"):
        # Get batch
        batch_src = dataset.src_data[i:i + batch_size]
        batch_tgt = dataset.tgt_data[i:i + batch_size]

        # Process source and target texts in parallel using batch_encode
        src_token_ids = de_tokenizer.batch_encode(batch_src, batch_size=batch_size)
        tgt_token_ids = en_tokenizer.batch_encode(batch_tgt, batch_size=batch_size)

        # Add special tokens efficiently
        for src_ids, tgt_ids in zip(src_token_ids, tgt_token_ids):
            src_sequences.append([src_bos_idx] + src_ids + [src_eos_idx])
            tgt_sequences.append([tgt_bos_idx] + tgt_ids + [tgt_eos_idx])

    return src_sequences, tgt_sequences

# First optimization: Simple caching (from our earlier solution)
def optimized_preprocess_data_with_bpe(dataset, de_tokenizer, en_tokenizer, batch_size=4000):
    """Optimized preprocessing with word-level caching."""
    # Initialize word token caches if they don't exist
    if not hasattr(de_tokenizer, 'word_token_cache'):
        de_tokenizer.word_token_cache = {}
    if not hasattr(en_tokenizer, 'word_token_cache'):
        en_tokenizer.word_token_cache = {}

    src_sequences = []
    tgt_sequences = []

    # Get special token indices
    src_bos_idx = de_tokenizer.special_tokens["bos_token_idx"]
    src_eos_idx = de_tokenizer.special_tokens["eos_token_idx"]
    tgt_bos_idx = en_tokenizer.special_tokens["bos_token_idx"]
    tgt_eos_idx = en_tokenizer.special_tokens["eos_token_idx"]

    # Process in batches
    for i in tqdm(range(0, len(dataset.src_data), batch_size),
                 total=len(dataset.src_data) // batch_size + 1,
                 desc="Preprocessing batches"):
        # Get batch
        batch_src = dataset.src_data[i:i + batch_size]
        batch_tgt = dataset.tgt_data[i:i + batch_size]

        # Process each source text with caching
        src_batch_tokens = []
        for text in batch_src:
            # Preprocess
            processed = de_tokenizer.preprocess(text)

            # Split into words
            words = processed.split()

            # Collect tokens for all words (using cache when possible)
            text_tokens = []
            for word in words:
                if word in de_tokenizer.word_token_cache:
                    # Use cached tokens
                    tokens = de_tokenizer.word_token_cache[word]
                else:
                    # Tokenize and cache
                    tokens = de_tokenizer._tokenize_word(word)
                    de_tokenizer.word_token_cache[word] = tokens
                text_tokens.extend(tokens)

            # Convert to token IDs
            token_ids = [de_tokenizer.vocab.token_to_index(token) for token in text_tokens]
            src_batch_tokens.append(token_ids)

        # Process each target text with caching
        tgt_batch_tokens = []
        for text in batch_tgt:
            # Preprocess
            processed = en_tokenizer.preprocess(text)

            # Split into words
            words = processed.split()

            # Collect tokens for all words (using cache when possible)
            text_tokens = []
            for word in words:
                if word in en_tokenizer.word_token_cache:
                    # Use cached tokens
                    tokens = en_tokenizer.word_token_cache[word]
                else:
                    # Tokenize and cache
                    tokens = en_tokenizer._tokenize_word(word)
                    en_tokenizer.word_token_cache[word] = tokens
                text_tokens.extend(tokens)

            # Convert to token IDs
            token_ids = [en_tokenizer.vocab.token_to_index(token) for token in text_tokens]
            tgt_batch_tokens.append(token_ids)

        # Add special tokens
        for src_ids, tgt_ids in zip(src_batch_tokens, tgt_batch_tokens):
            src_sequences.append([src_bos_idx] + src_ids + [src_eos_idx])
            tgt_sequences.append([tgt_bos_idx] + tgt_ids + [tgt_eos_idx])

    return src_sequences, tgt_sequences

def run_benchmark():
    """Run the preprocessing benchmarks and visualize results."""
    print("Loading tokenizers...")
    # Load the pre-trained BPE tokenizers
    de_tokenizer = BPETokenizer.from_pretrained("models/tokenizers/german")
    en_tokenizer = BPETokenizer.from_pretrained("models/tokenizers/english")

    print("Loading test dataset...")
    # Load a small dataset for benchmarking
    test_dataset = EuroparlDataset(
        src_lang="de",
        tgt_lang="en",
        max_examples=10000  # 10k examples is enough for benchmarking
    )

    # Run benchmarks
    methods = [
        ("Original", original_preprocess_data_with_bpe, 1000),
        ("Optimized", optimized_preprocess_data_with_bpe, 4000),
        ("Turbo", turbo_preprocess_data, None)
    ]

    results = []

    for name, method, batch_size in methods:
        print(f"\nRunning benchmark for {name} method...")

        # Reset caches to ensure fair comparison
        if hasattr(de_tokenizer, 'word_token_cache'):
            de_tokenizer.word_token_cache = {}
        if hasattr(en_tokenizer, 'word_token_cache'):
            en_tokenizer.word_token_cache = {}

        # Clear GPU cache
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()

        # Run method with timing
        start_time = time.time()

        if batch_size is not None:
            src_sequences, tgt_sequences = method(test_dataset, de_tokenizer, en_tokenizer, batch_size=batch_size)
        else:
            src_sequences, tgt_sequences = method(test_dataset, de_tokenizer, en_tokenizer)

        elapsed = time.time() - start_time

        # Calculate metrics
        examples_per_second = len(test_dataset.src_data) / elapsed

        # Store results
        results.append({
            'method': name,
            'time': elapsed,
            'examples_per_second': examples_per_second,
            'src_sequences': len(src_sequences),
            'tgt_sequences': len(tgt_sequences)
        })

        print(f"{name} method processed {len(test_dataset.src_data)} examples in {elapsed:.2f} seconds")
        print(f"Speed: {examples_per_second:.2f} examples/second")

    # Verify that all methods produced the same output lengths
    first_result = results[0]
    for result in results[1:]:
        if result['src_sequences'] != first_result['src_sequences'] or result['tgt_sequences'] != first_result['tgt_sequences']:
            print(f"WARNING: Method {result['method']} produced different sequence counts compared to {first_result['method']}")

    # Visualize results
    plt.figure(figsize=(12, 6))

    # Processing time comparison (lower is better)
    plt.subplot(1, 2, 1)
    methods = [r['method'] for r in results]
    times = [r['time'] for r in results]

    plt.bar(methods, times)
    plt.title('Processing Time (seconds)')
    plt.ylabel('Seconds')
    plt.xticks(rotation=45)

    # Speed comparison (higher is better)
    plt.subplot(1, 2, 2)
    speeds = [r['examples_per_second'] for r in results]

    plt.bar(methods, speeds)
    plt.title('Processing Speed (examples/second)')
    plt.ylabel('Examples/second')
    plt.xticks(rotation=45)

    plt.tight_layout()
    plt.savefig('preprocessing_benchmark_results.png')
    plt.close()

    print("\nBenchmark results saved to preprocessing_benchmark_results.png")

    # Print summary table
    print("\nBenchmark Summary:")
    print(f"{'Method':<15} {'Time (s)':<15} {'Speed (examples/s)':<20}")
    print("-" * 50)
    for result in results:
        print(f"{result['method']:<15} {result['time']:<15.2f} {result['examples_per_second']:<20.2f}")

if __name__ == "__main__":
    run_benchmark()
