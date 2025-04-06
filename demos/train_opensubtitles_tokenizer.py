#!/usr/bin/env python3
# train_opensubtitles_tokenizer.py
# Script to train an OptimizedBPETokenizer on the OpenSubtitles dataset

import os
import sys
import argparse
from typing import List, Optional
import torch
from tqdm import tqdm

# Add parent directory to path to import local modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import the tokenizer and dataset
from src.data.tokenization import OptimizedBPETokenizer
from src.data.opensubtitles_dataset import OpenSubtitlesDataset


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Train a BPE tokenizer on the OpenSubtitles dataset")
    
    parser.add_argument("--vocab_size", type=int, default=8000, 
                        help="Vocabulary size for the tokenizer")
    parser.add_argument("--src_lang", type=str, default="en", 
                        help="Source language")
    parser.add_argument("--tgt_lang", type=str, default="en", 
                        help="Target language (for parallel datasets)")
    parser.add_argument("--train_lang", type=str, default=None,
                        help="Language to train the tokenizer for (defaults to src_lang)")
    parser.add_argument("--min_frequency", type=int, default=2, 
                        help="Minimum frequency for a token")
    parser.add_argument("--output_dir", type=str, default="models/tokenizers", 
                        help="Directory to save tokenizer")
    parser.add_argument("--data_dir", type=str, default="data/os", 
                        help="Directory containing OpenSubtitles data")
    parser.add_argument("--max_examples", type=int, default=None, 
                        help="Maximum number of examples to use (None = use all)")
    parser.add_argument("--device", type=str, default=None, 
                        help="Device to use for training (None = auto-detect)")
    
    args = parser.parse_args()
    
    # If train_lang is not specified, use src_lang
    if args.train_lang is None:
        args.train_lang = args.src_lang
        
    return args


def load_opensubtitles_data(train_lang: str, src_lang: str, tgt_lang: str, data_dir: str, max_examples: Optional[int]) -> List[str]:
    """
    Load OpenSubtitles dataset.
    
    Args:
        train_lang: Language to train the tokenizer for
        src_lang: Source language in the parallel corpus
        tgt_lang: Target language in the parallel corpus
        data_dir: Directory containing OpenSubtitles data
        max_examples: Maximum number of examples to load
        
    Returns:
        List of texts from the dataset for the specified train_lang
    """
    print(f"Loading OpenSubtitles dataset for language: {train_lang} from {data_dir}")
    print(f"Using parallel corpus src={src_lang}, tgt={tgt_lang}")
    
    # Load the dataset
    dataset = OpenSubtitlesDataset(
        data_dir=data_dir,
        src_lang=src_lang,
        tgt_lang=tgt_lang,
        max_examples=max_examples
    )
    
    # Determine which side of the parallel corpus to use
    if train_lang == src_lang:
        print(f"Training tokenizer on source language: {src_lang}")
        texts = dataset.src_data
    elif train_lang == tgt_lang:
        print(f"Training tokenizer on target language: {tgt_lang}")
        texts = dataset.tgt_data
    else:
        raise ValueError(f"Train language '{train_lang}' must be either source '{src_lang}' or target '{tgt_lang}'")
    
    print(f"Loaded {len(texts)} examples from OpenSubtitles dataset for language: {train_lang}")
    return texts


def train_tokenizer(texts: List[str], vocab_size: int, min_frequency: int, device: Optional[str] = None) -> OptimizedBPETokenizer:
    """
    Train a BPE tokenizer on the provided texts.
    
    Args:
        texts: List of training texts
        vocab_size: Target vocabulary size
        min_frequency: Minimum frequency for a token
        device: Device to use for training
        
    Returns:
        Trained tokenizer
    """
    print(f"Training BPE tokenizer with vocab size: {vocab_size}")
    
    # Create the tokenizer
    tokenizer = OptimizedBPETokenizer(
        vocab=None,
        merges=None,
        num_merges=vocab_size,
        lower_case=True,
        device=device,
        vectorized=True
    )
    
    # Train the tokenizer
    tokenizer.train(
        texts=texts,
        vocab_size=vocab_size,
        min_frequency=min_frequency,
        show_progress=True
    )
    
    print(f"Tokenizer training complete! Vocabulary size: {tokenizer.vocab_size}")
    return tokenizer


def save_tokenizer(tokenizer: OptimizedBPETokenizer, output_dir: str, lang: str) -> None:
    """
    Save the tokenizer to disk.
    
    Args:
        tokenizer: Trained tokenizer
        output_dir: Directory to save the tokenizer
        lang: Language code for the tokenizer
    """
    # Create output directory if it doesn't exist
    os.makedirs(os.path.join(output_dir, lang), exist_ok=True)
    
    # Save the tokenizer
    output_path = os.path.join(output_dir, lang)
    tokenizer.save_pretrained(output_path)
    
    print(f"Tokenizer saved to: {output_path}")


def test_tokenizer(tokenizer: OptimizedBPETokenizer, texts: List[str], num_samples: int = 5) -> None:
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
    
    # Load OpenSubtitles data
    texts = load_opensubtitles_data(
        train_lang=args.train_lang,
        src_lang=args.src_lang,
        tgt_lang=args.tgt_lang,
        data_dir=args.data_dir,
        max_examples=args.max_examples
    )
    
    # Train tokenizer
    tokenizer = train_tokenizer(
        texts=texts,
        vocab_size=args.vocab_size,
        min_frequency=args.min_frequency,
        device=args.device
    )
    
    # Save tokenizer
    save_tokenizer(
        tokenizer=tokenizer,
        output_dir=args.output_dir,
        lang=args.train_lang
    )
    
    # Test tokenizer
    test_tokenizer(tokenizer, texts)


if __name__ == "__main__":
    main() 