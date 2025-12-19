#!/usr/bin/env python3
# inspect_training_data.py
# Script to inspect training data sequences

import os
import sys
import torch
import argparse
import random
from typing import List, Tuple

# Add parent directory to path to import local modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import necessary modules
from src.data.opensubtitles_dataset import OpenSubtitlesDataset
from src.data.tokenization import OptimizedBPETokenizer

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Inspect training data sequences")
    
    parser.add_argument("--src_lang", type=str, default="de", 
                        help="Source language")
    parser.add_argument("--tgt_lang", type=str, default="en", 
                        help="Target language")
    parser.add_argument("--data_dir", type=str, default="data/os", 
                        help="Directory containing OpenSubtitles data")
    parser.add_argument("--tokenizer_path", type=str, default="models/tokenizers", 
                        help="Directory containing tokenizers")
    parser.add_argument("--num_examples", type=int, default=10, 
                        help="Number of examples to show")
    parser.add_argument("--max_examples", type=int, default=5000, 
                        help="Maximum number of examples to load")
    
    return parser.parse_args()

def preprocess_data_with_bpe(dataset, src_tokenizer, tgt_tokenizer):
    """
    Preprocess the dataset for training using BPE tokenizers.
    This replicates the preprocessing in translation_example.py.
    
    Args:
        dataset: Dataset with src_data and tgt_data attributes
        src_tokenizer: Source language tokenizer
        tgt_tokenizer: Target language tokenizer
        
    Returns:
        Tuple of (src_sequences, tgt_sequences)
    """
    src_sequences = []
    tgt_sequences = []
    
    # Get special token indices
    src_bos_idx = src_tokenizer.special_tokens["bos_token_idx"]
    src_eos_idx = src_tokenizer.special_tokens["eos_token_idx"]
    tgt_bos_idx = tgt_tokenizer.special_tokens["bos_token_idx"]
    tgt_eos_idx = tgt_tokenizer.special_tokens["eos_token_idx"]
    
    for src_text, tgt_text in zip(dataset.src_data, dataset.tgt_data):
        # Encode source and target
        src_token_ids = src_tokenizer.encode(src_text)
        tgt_token_ids = tgt_tokenizer.encode(tgt_text)
        
        # Add special tokens
        src_token_ids = [src_bos_idx] + src_token_ids + [src_eos_idx]
        tgt_token_ids = [tgt_bos_idx] + tgt_token_ids + [tgt_eos_idx]
        
        src_sequences.append(src_token_ids)
        tgt_sequences.append(tgt_token_ids)
    
    return src_sequences, tgt_sequences

def main():
    """Main function to inspect training data."""
    # Parse arguments
    args = parse_args()
    
    # Load dataset
    print(f"Loading OpenSubtitles dataset ({args.src_lang}-{args.tgt_lang})...")
    dataset = OpenSubtitlesDataset(
        data_dir=args.data_dir,
        src_lang=args.src_lang,
        tgt_lang=args.tgt_lang,
        max_examples=args.max_examples
    )
    print(f"Loaded {len(dataset.src_data)} parallel sentences")
    
    # Load tokenizers
    print(f"Loading tokenizers from {args.tokenizer_path}...")
    src_tokenizer = OptimizedBPETokenizer.from_pretrained(f"{args.tokenizer_path}/{args.src_lang}")
    tgt_tokenizer = OptimizedBPETokenizer.from_pretrained(f"{args.tokenizer_path}/{args.tgt_lang}")
    print(f"Loaded {args.src_lang} tokenizer with vocab size: {src_tokenizer.vocab_size}")
    print(f"Loaded {args.tgt_lang} tokenizer with vocab size: {tgt_tokenizer.vocab_size}")
    
    # Preprocess data
    print("Preprocessing data...")
    src_sequences, tgt_sequences = preprocess_data_with_bpe(dataset, src_tokenizer, tgt_tokenizer)
    
    # Select random examples
    indices = random.sample(range(len(src_sequences)), min(args.num_examples, len(src_sequences)))
    
    # Display examples
    print("\n=== Training Data Examples ===\n")
    
    for i, idx in enumerate(indices):
        # Get original texts
        src_text = dataset.src_data[idx]
        tgt_text = dataset.tgt_data[idx]
        
        # Get token sequences
        src_tokens = src_sequences[idx]
        tgt_tokens = tgt_sequences[idx]
        
        # Skip BOS/EOS for cleaner output
        src_tokens_content = src_tokens[1:-1]  # Skip BOS and EOS
        tgt_tokens_content = tgt_tokens[1:-1]  # Skip BOS and EOS
        
        # Decode tokens back to text
        src_decoded = src_tokenizer.decode(src_tokens_content)
        tgt_decoded = tgt_tokenizer.decode(tgt_tokens_content)
        
        # Show results
        print(f"Example {i+1}:")
        print(f"Source Original  : {src_text}")
        print(f"Source Tokenized : {src_tokens}")
        print(f"Source Decoded   : {src_decoded}")
        print(f"Source Roundtrip : {'✓' if src_decoded.replace(' ', '') == src_text.lower().replace(' ', '').replace('.', '').replace(',', '').replace('-', '').replace('?', '').replace('!', '') else '✗'}")
        
        print(f"Target Original  : {tgt_text}")
        print(f"Target Tokenized : {tgt_tokens}")
        print(f"Target Decoded   : {tgt_decoded}")
        print(f"Target Roundtrip : {'✓' if tgt_decoded.replace(' ', '') == tgt_text.lower().replace(' ', '').replace('.', '').replace(',', '').replace('-', '').replace('?', '').replace('!', '') else '✗'}")
        
        print("-" * 80)
    
    # Check if we have any examples with unexpected token counts
    print("\n=== Sequence Length Analysis ===\n")
    
    src_lengths = [len(seq) for seq in src_sequences]
    tgt_lengths = [len(seq) for seq in tgt_sequences]
    
    max_src_len = max(src_lengths)
    max_tgt_len = max(tgt_lengths)
    avg_src_len = sum(src_lengths) / len(src_lengths)
    avg_tgt_len = sum(tgt_lengths) / len(tgt_lengths)
    
    print(f"Source Sequences Length - Max: {max_src_len}, Avg: {avg_src_len:.2f}")
    print(f"Target Sequences Length - Max: {max_tgt_len}, Avg: {avg_tgt_len:.2f}")
    
    # Find examples with unusually long token sequences
    long_src_indices = [i for i, length in enumerate(src_lengths) if length > avg_src_len * 3]
    long_tgt_indices = [i for i, length in enumerate(tgt_lengths) if length > avg_tgt_len * 3]
    
    print(f"\nFound {len(long_src_indices)} unusually long source sequences")
    print(f"Found {len(long_tgt_indices)} unusually long target sequences")
    
    # Show some examples of unusually long sequences
    if long_src_indices:
        print("\n=== Examples of Unusually Long Source Sequences ===\n")
        for i, idx in enumerate(long_src_indices[:3]):
            print(f"Example {i+1} (Length: {len(src_sequences[idx])}):")
            print(f"Text: {dataset.src_data[idx]}")
            print("-" * 80)
    
    if long_tgt_indices:
        print("\n=== Examples of Unusually Long Target Sequences ===\n")
        for i, idx in enumerate(long_tgt_indices[:3]):
            print(f"Example {i+1} (Length: {len(tgt_sequences[idx])}):")
            print(f"Text: {dataset.tgt_data[idx]}")
            print("-" * 80)
    
    # Count the number of special tokens in sequences
    bos_src_count = sum(1 for seq in src_sequences if seq[0] == src_tokenizer.special_tokens["bos_token_idx"])
    eos_src_count = sum(1 for seq in src_sequences if seq[-1] == src_tokenizer.special_tokens["eos_token_idx"])
    bos_tgt_count = sum(1 for seq in tgt_sequences if seq[0] == tgt_tokenizer.special_tokens["bos_token_idx"])
    eos_tgt_count = sum(1 for seq in tgt_sequences if seq[-1] == tgt_tokenizer.special_tokens["eos_token_idx"])
    
    print("\n=== Special Token Analysis ===\n")
    print(f"Source sequences with BOS: {bos_src_count}/{len(src_sequences)}")
    print(f"Source sequences with EOS: {eos_src_count}/{len(src_sequences)}")
    print(f"Target sequences with BOS: {bos_tgt_count}/{len(tgt_sequences)}")
    print(f"Target sequences with EOS: {eos_tgt_count}/{len(tgt_sequences)}")
    
    print("\nInspection complete!")


if __name__ == "__main__":
    main() 