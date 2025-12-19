#!/usr/bin/env python
"""
Script to train a BPE tokenizer on WMT data.
"""

import os
import sys
import argparse
from typing import List, Dict, Optional, Tuple
import random
from tqdm import tqdm
import concurrent.futures
from functools import partial

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.wmt_dataset import WMTDataset
from src.data.tokenization import OptimizedBPETokenizer


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Train a BPE tokenizer on WMT data")
    parser.add_argument(
        "--src_lang", type=str, default="de", help="Source language code"
    )
    parser.add_argument(
        "--tgt_lang", type=str, default="en", help="Target language code"
    )
    parser.add_argument(
        "--train_lang",
        type=str,
        default=None,
        help="Language to train tokenizer on (if None, train a joint tokenizer)",
    )
    parser.add_argument(
        "--vocab_size",
        type=int,
        default=32000,
        help="Vocabulary size for the tokenizer",
    )
    parser.add_argument(
        "--data_dir", type=str, default="data/wmt", help="Directory for WMT data"
    )
    parser.add_argument(
        "--min_frequency", type=int, default=2, help="Minimum token frequency"
    )
    parser.add_argument(
        "--max_examples",
        type=int,
        default=10000000,
        help="Maximum number of examples to use for training",
    )
    parser.add_argument(
        "--year", type=str, default="14", help="WMT dataset year (without 'wmt' prefix)"
    )
    parser.add_argument(
        "--split", type=str, default="train", help="Dataset split to use"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="models/tokenizers",
        help="Output directory for tokenizer",
    )
    parser.add_argument(
        "--preserve_case",
        action="store_true",
        help="Preserve case when training tokenizer",
    )
    parser.add_argument(
        "--preserve_punctuation",
        action="store_true",
        help="Preserve punctuation when training tokenizer",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to use for training (None = auto-detect)",
    )
    parser.add_argument(
        "--subset",
        type=str,
        default=None,
        help="Dataset subset to use, if available (e.g., 'news_commentary_v9')",
    )
    parser.add_argument(
        "--lowercase",
        action="store_true",
        help="Convert text to lowercase before training",
    )
    parser.add_argument(
        "--joint",
        action="store_true",
        help="Train a joint tokenizer for both languages",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=4,
        help="Number of worker processes for parallel processing",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1000,
        help="Batch size for parallel processing",
    )

    args = parser.parse_args()

    # If train_lang is not specified, use src_lang
    if args.train_lang is None:
        args.train_lang = args.src_lang

    return args


def load_wmt_data(
    train_lang: str,
    src_lang: str,
    tgt_lang: str,
    year: str,
    split: str,
    data_dir: str,
    max_examples: Optional[int],
    subset: Optional[str] = None,
) -> List[str]:
    """
    Load WMT dataset.

    Args:
        train_lang: Language to train the tokenizer for
        src_lang: Source language in the parallel corpus
        tgt_lang: Target language in the parallel corpus
        year: Dataset year (without 'wmt' prefix)
        split: Dataset split (train, valid, test)
        data_dir: Directory containing WMT data
        max_examples: Maximum number of examples to load
        subset: Optional dataset subset to use

    Returns:
        List of texts from the dataset for the specified train_lang
    """
    print(f"Loading WMT{year} dataset for language: {train_lang}")
    print(f"Using parallel corpus src={src_lang}, tgt={tgt_lang}")

    if subset:
        print(f"Using subset: {subset}")

    # Load the dataset
    dataset = WMTDataset(
        src_lang=src_lang,
        tgt_lang=tgt_lang,
        year=year,
        split=split,
        max_examples=max_examples,
        data_dir=data_dir,
        subset=subset,
    )

    # Determine which side of the parallel corpus to use
    if train_lang == "joint":
        print(f"Training tokenizer on both source and target languages")
        texts = dataset.src_data + dataset.tgt_data
    elif train_lang == src_lang:
        print(f"Training tokenizer on source language: {src_lang}")
        texts = dataset.src_data
    elif train_lang == tgt_lang:
        print(f"Training tokenizer on target language: {tgt_lang}")
        texts = dataset.tgt_data
    else:
        raise ValueError(
            f"Train language '{train_lang}' must be either source '{src_lang}', target '{tgt_lang}', or 'joint'"
        )

    print(
        f"Loaded {len(texts)} examples from WMT{year} dataset for language: {train_lang}"
    )
    return texts


def train_tokenizer(
    texts: List[str],
    vocab_size: int,
    min_frequency: int,
    preserve_case: bool = False,
    preserve_punctuation: bool = True,
    device: Optional[str] = None,
) -> OptimizedBPETokenizer:
    """
    Train a BPE tokenizer on the provided texts.

    Args:
        texts: List of training texts
        vocab_size: Target vocabulary size
        min_frequency: Minimum frequency for a token
        preserve_case: Whether to preserve case information
        preserve_punctuation: Whether to preserve punctuation
        device: Device to use for training

    Returns:
        Trained tokenizer
    """
    print(f"Training BPE tokenizer with vocab size: {vocab_size}")
    print(
        f"Preserving case: {preserve_case}, Preserving punctuation: {preserve_punctuation}"
    )

    # Create the tokenizer
    tokenizer = OptimizedBPETokenizer(
        vocab=None,
        merges=None,
        num_merges=vocab_size,
        lower_case=not preserve_case,  # Invert preserve_case flag for lower_case parameter
        device=device,
        vectorized=True,
        preserve_punctuation=preserve_punctuation,
        preserve_case=preserve_case,
    )

    # Train the tokenizer
    tokenizer.train(
        texts=texts,
        vocab_size=vocab_size,
        min_frequency=min_frequency,
        show_progress=True,
    )

    print(f"Tokenizer training complete! Vocabulary size: {tokenizer.vocab_size}")
    return tokenizer


def save_tokenizer(
    tokenizer: OptimizedBPETokenizer, output_dir: str, lang: str, year: str
) -> None:
    """
    Save the tokenizer to disk.

    Args:
        tokenizer: Trained tokenizer
        output_dir: Directory to save the tokenizer
        lang: Language code for the tokenizer
        year: WMT year used for training
    """
    # Create output directory - handle 'joint' specially
    if lang == "joint":
        # Joint tokenizer goes in a special combined folder
        save_path = os.path.join(output_dir, f"wmt{year}", "joint")
    else:
        # Language-specific tokenizer goes in its own folder
        save_path = os.path.join(output_dir, f"wmt{year}", lang)

    os.makedirs(save_path, exist_ok=True)

    # Save the tokenizer
    tokenizer.save_pretrained(save_path)

    print(f"Tokenizer saved to: {save_path}")


def test_tokenizer(
    tokenizer: OptimizedBPETokenizer, texts: List[str], num_samples: int = 5
) -> None:
    """
    Test the tokenizer on sample texts.

    Args:
        tokenizer: Trained tokenizer
        texts: List of texts to sample from
        num_samples: Number of samples to test
    """
    print("\nTesting tokenizer on sample texts:")

    # Sample some texts
    import random

    samples = random.sample(texts, min(num_samples, len(texts)))

    for i, text in enumerate(samples):
        # Tokenize and encode
        tokens = tokenizer.tokenize(text)
        ids = tokenizer.encode(text)

        # Decode back
        decoded = tokenizer.decode(ids)

        # Print results
        print(f"\nSample {i+1}:")
        print(f"Original: {text}")
        print(f"Tokens: {tokens}")
        print(f"Token IDs: {ids}")
        print(f"Decoded: {decoded}")
        print(f"Roundtrip success: {text == decoded}")


def process_batch(batch_texts, vocab_size):
    """Process a batch of texts for tokenization training"""
    processed_texts = []
    for text in batch_texts:
        # Basic text cleaning
        text = text.strip()
        if text:  # Skip empty lines
            processed_texts.append(text)
    return processed_texts


def main():
    """Main function to train and save a tokenizer."""
    # Parse arguments
    args = parse_args()

    print(
        f"Training {'joint' if args.joint else 'separate'} tokenizer(s) for WMT{args.year} {args.src_lang}-{args.tgt_lang}"
    )

    # Create output directory
    output_dir = f"models/tokenizers/wmt{args.year}"
    os.makedirs(output_dir, exist_ok=True)

    # Load WMT dataset
    print(f"Loading WMT{args.year} dataset...")
    dataset = WMTDataset(
        src_lang=args.src_lang,
        tgt_lang=args.tgt_lang,
        year=args.year,
        split="train",
        max_examples=args.max_examples,
        data_dir=args.data_dir,
        subset=args.subset,
    )

    print(f"Loaded {len(dataset.src_data)} examples")

    # Process texts in parallel with progress bar
    print("Processing texts...")

    # Create source and target text batches
    batch_size = args.batch_size
    src_batches = [
        dataset.src_data[i : i + batch_size]
        for i in range(0, len(dataset.src_data), batch_size)
    ]
    tgt_batches = [
        dataset.tgt_data[i : i + batch_size]
        for i in range(0, len(dataset.tgt_data), batch_size)
    ]

    # Process source texts in parallel
    src_texts = []
    with tqdm(
        total=len(dataset.src_data), desc=f"Processing {args.src_lang} texts"
    ) as pbar:
        process_func = partial(process_batch, vocab_size=args.vocab_size)
        with concurrent.futures.ProcessPoolExecutor(
            max_workers=args.num_workers
        ) as executor:
            for batch_result in executor.map(process_func, src_batches):
                src_texts.extend(batch_result)
                pbar.update(len(batch_result))

    # Process target texts in parallel
    tgt_texts = []
    with tqdm(
        total=len(dataset.tgt_data), desc=f"Processing {args.tgt_lang} texts"
    ) as pbar:
        process_func = partial(process_batch, vocab_size=args.vocab_size)
        with concurrent.futures.ProcessPoolExecutor(
            max_workers=args.num_workers
        ) as executor:
            for batch_result in executor.map(process_func, tgt_batches):
                tgt_texts.extend(batch_result)
                pbar.update(len(batch_result))

    if args.joint:
        # Train joint tokenizer
        print(f"Training joint tokenizer with vocab size {args.vocab_size}...")
        all_texts = src_texts + tgt_texts

        # Create the tokenizer with the correct parameters
        tokenizer = OptimizedBPETokenizer(
            vocab=None,
            merges=None,
            num_merges=args.vocab_size,
            lower_case=not args.preserve_case,
            device=None,  # Auto-detect
            vectorized=True,
            preserve_punctuation=True,
            preserve_case=args.preserve_case,
        )

        # Train it with progress bar
        with tqdm(desc="Training joint tokenizer") as pbar:
            tokenizer.train(
                texts=all_texts,
                vocab_size=args.vocab_size,
                min_frequency=args.min_frequency,
                show_progress=True,
            )

        # Save tokenizer
        joint_dir = os.path.join(output_dir, "joint")
        os.makedirs(joint_dir, exist_ok=True)
        tokenizer.save_pretrained(joint_dir)
        print(f"Saved joint tokenizer to {joint_dir}")

    else:
        # Train source tokenizer
        print(
            f"Training {args.src_lang} tokenizer with vocab size {args.vocab_size}..."
        )
        src_tokenizer = OptimizedBPETokenizer(
            vocab=None,
            merges=None,
            num_merges=args.vocab_size,
            lower_case=not args.preserve_case,
            device=None,  # Auto-detect
            vectorized=True,
            preserve_punctuation=True,
            preserve_case=args.preserve_case,
        )

        with tqdm(desc=f"Training {args.src_lang} tokenizer") as pbar:
            src_tokenizer.train(
                texts=src_texts,
                vocab_size=args.vocab_size,
                min_frequency=args.min_frequency,
                show_progress=True,
            )

        # Save source tokenizer
        src_dir = os.path.join(output_dir, args.src_lang)
        os.makedirs(src_dir, exist_ok=True)
        src_tokenizer.save_pretrained(src_dir)
        print(f"Saved {args.src_lang} tokenizer to {src_dir}")

        # Train target tokenizer
        print(
            f"Training {args.tgt_lang} tokenizer with vocab size {args.vocab_size}..."
        )
        tgt_tokenizer = OptimizedBPETokenizer(
            vocab=None,
            merges=None,
            num_merges=args.vocab_size,
            lower_case=not args.preserve_case,
            device=None,  # Auto-detect
            vectorized=True,
            preserve_punctuation=True,
            preserve_case=args.preserve_case,
        )

        with tqdm(desc=f"Training {args.tgt_lang} tokenizer") as pbar:
            tgt_tokenizer.train(
                texts=tgt_texts,
                vocab_size=args.vocab_size,
                min_frequency=args.min_frequency,
                show_progress=True,
            )

        # Save target tokenizer
        tgt_dir = os.path.join(output_dir, args.tgt_lang)
        os.makedirs(tgt_dir, exist_ok=True)
        tgt_tokenizer.save_pretrained(tgt_dir)
        print(f"Saved {args.tgt_lang} tokenizer to {tgt_dir}")


if __name__ == "__main__":
    main()
