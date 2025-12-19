#!/usr/bin/env python
"""
Script to train a BPE tokenizer on IWSLT data.
"""

import os
import sys
import argparse
from typing import List, Dict, Optional, Tuple
import random

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.iwslt_dataset import IWSLTDataset
from src.data.tokenization import OptimizedBPETokenizer


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Train a BPE tokenizer on IWSLT data")
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
        "--vocab_size", type=int, default=8000, help="Vocabulary size for the tokenizer"
    )
    parser.add_argument(
        "--data_dir", type=str, default="data/iwslt", help="Directory for IWSLT data"
    )
    parser.add_argument(
        "--min_frequency", type=int, default=2, help="Minimum token frequency"
    )
    parser.add_argument(
        "--max_examples",
        type=int,
        default=5000000,
        help="Maximum number of examples to use for training",
    )
    parser.add_argument("--year", type=str, default="2017", help="IWSLT dataset year")
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

    args = parser.parse_args()

    # If train_lang is not specified, use src_lang
    if args.train_lang is None:
        args.train_lang = args.src_lang

    return args


def load_iwslt_data(
    train_lang: str,
    src_lang: str,
    tgt_lang: str,
    year: str,
    split: str,
    data_dir: str,
    max_examples: Optional[int],
) -> List[str]:
    """
    Load IWSLT dataset.

    Args:
        train_lang: Language to train the tokenizer for
        src_lang: Source language in the parallel corpus
        tgt_lang: Target language in the parallel corpus
        year: Dataset year
        split: Dataset split (train, valid, test)
        data_dir: Directory containing IWSLT data
        max_examples: Maximum number of examples to load

    Returns:
        List of texts from the dataset for the specified train_lang
    """
    print(f"Loading IWSLT {year} dataset for language: {train_lang} from {data_dir}")
    print(f"Using parallel corpus src={src_lang}, tgt={tgt_lang}")

    # Load the dataset
    dataset = IWSLTDataset(
        src_lang=src_lang,
        tgt_lang=tgt_lang,
        year=year,
        split=split,
        max_examples=max_examples,
        data_dir=data_dir,
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

    print(f"Loaded {len(texts)} examples from IWSLT dataset for language: {train_lang}")
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
    tokenizer: OptimizedBPETokenizer, output_dir: str, lang: str
) -> None:
    """
    Save the tokenizer to disk.

    Args:
        tokenizer: Trained tokenizer
        output_dir: Directory to save the tokenizer
        lang: Language code for the tokenizer
    """
    # Create output directory - handle 'joint' specially
    if lang == "joint":
        # Joint tokenizer goes in a special combined folder
        save_path = os.path.join(output_dir, "combined", "joint")
    else:
        # Language-specific tokenizer goes in its own folder
        save_path = os.path.join(output_dir, lang)

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


def main():
    """Main function to train and save a tokenizer."""
    # Parse arguments
    args = parse_args()

    # Load IWSLT data
    texts = load_iwslt_data(
        train_lang=args.train_lang,
        src_lang=args.src_lang,
        tgt_lang=args.tgt_lang,
        year=args.year,
        split=args.split,
        data_dir=args.data_dir,
        max_examples=args.max_examples,
    )

    # Try to use transformers TokenizerFast for faster training
    try:
        from transformers import PreTrainedTokenizerFast, AutoTokenizer

        print("Using transformers TokenizerFast for faster BPE training")

        # Create corpus file first (transformers needs a file)
        corpus_file = os.path.join(args.data_dir, f"{args.train_lang}_corpus.txt")
        with open(corpus_file, "w", encoding="utf-8") as f:
            for text in texts:
                f.write(text + "\n")

        print(f"Created corpus file with {len(texts)} lines")

        # Use command-line tokenizers library through the transformers API
        tokenizer = PreTrainedTokenizerFast(
            tokenizer_file=None,  # Will be set by train_new_from_iterator
            unk_token="<unk>",
            pad_token="<pad>",
            bos_token="<bos>",
            eos_token="<eos>",
            lowercase=not args.preserve_case,
        )

        # Train from the corpus file
        tokenizer = tokenizer.train_new_from_iterator(
            texts, vocab_size=args.vocab_size, min_frequency=args.min_frequency
        )

        # Save the tokenizer
        save_dir = os.path.join(args.output_dir, args.train_lang)
        os.makedirs(save_dir, exist_ok=True)
        tokenizer.save_pretrained(save_dir)

        print(f"TokenizerFast saved to: {save_dir}")

        # Test the tokenizer
        if texts:
            print("\nTesting tokenizer on sample texts:")
            samples = random.sample(texts, min(5, len(texts)))
            for i, text in enumerate(samples):
                tokens = tokenizer.tokenize(text)
                ids = tokenizer.encode(text)
                decoded = tokenizer.decode(ids)

                print(f"\nSample {i+1}:")
                print(f"Original: {text}")
                print(f"Tokens: {tokens}")
                print(f"Token IDs: {ids}")
                print(f"Decoded: {decoded}")

        # Remove the temporary corpus file
        if os.path.exists(corpus_file):
            os.remove(corpus_file)

    except (ImportError, AttributeError, ValueError) as e:
        print(f"Transformers library not available or API mismatch: {e}")
        print("Falling back to OptimizedBPETokenizer")

        # Train tokenizer using our implementation
        tokenizer = train_tokenizer(
            texts=texts,
            vocab_size=args.vocab_size,
            min_frequency=args.min_frequency,
            preserve_case=args.preserve_case,
            preserve_punctuation=args.preserve_punctuation,
            device=args.device,
        )

        # Save tokenizer
        save_tokenizer(
            tokenizer=tokenizer, output_dir=args.output_dir, lang=args.train_lang
        )

        # Test tokenizer
        test_tokenizer(tokenizer, texts)


if __name__ == "__main__":
    main()
