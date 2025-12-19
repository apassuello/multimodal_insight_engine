#!/usr/bin/env python
"""
Script to test the IWSLT dataset with the translation example.
"""

import os
import sys
import argparse

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.iwslt_dataset import IWSLTDataset
from src.data.tokenization import OptimizedBPETokenizer


def main():
    """Test the IWSLT dataset."""
    parser = argparse.ArgumentParser(description="Test IWSLT dataset")
    parser.add_argument("--src_lang", type=str, default="de", help="Source language")
    parser.add_argument("--tgt_lang", type=str, default="en", help="Target language")
    parser.add_argument("--year", type=str, default="2017", help="Dataset year")
    parser.add_argument("--split", type=str, default="train", help="Dataset split")
    parser.add_argument(
        "--data_dir", type=str, default="data/iwslt", help="Data directory"
    )
    parser.add_argument(
        "--max_examples", type=int, default=100, help="Maximum examples to load"
    )

    args = parser.parse_args()

    # Load the dataset
    print(f"Loading IWSLT {args.year} dataset ({args.src_lang}-{args.tgt_lang})...")
    dataset = IWSLTDataset(
        src_lang=args.src_lang,
        tgt_lang=args.tgt_lang,
        year=args.year,
        split=args.split,
        max_examples=args.max_examples,
        data_dir=args.data_dir,
    )

    # Print dataset statistics
    print(f"Dataset loaded with {len(dataset.src_data)} examples")

    # Print a few examples
    print("\nSample sentence pairs:")
    for i in range(min(5, len(dataset.src_data))):
        print(f"\nExample {i+1}:")
        print(f"Source ({args.src_lang}): {dataset.src_data[i]}")
        print(f"Target ({args.tgt_lang}): {dataset.tgt_data[i]}")

    # Test with tokenizer if available
    try:
        # Try to load source tokenizer
        src_tokenizer_path = f"models/tokenizers/{args.src_lang}"
        if os.path.exists(src_tokenizer_path):
            print(f"\nLoading source tokenizer from {src_tokenizer_path}...")
            src_tokenizer = OptimizedBPETokenizer.from_pretrained(src_tokenizer_path)

            # Tokenize a sample sentence
            sample_idx = 0
            sample_src = dataset.src_data[sample_idx]
            tokens = src_tokenizer.tokenize(sample_src)
            ids = src_tokenizer.encode(sample_src)
            decoded = src_tokenizer.decode(ids)

            print(f"\nSource sentence: {sample_src}")
            print(f"Tokens: {tokens}")
            print(f"IDs: {ids}")
            print(f"Decoded: {decoded}")
            print(f"Roundtrip success: {sample_src == decoded}")
        else:
            print(f"\nNo source tokenizer found at {src_tokenizer_path}")
            print("Run demos/train_iwslt_tokenizer.sh to train tokenizers")

    except Exception as e:
        print(f"Error loading or using tokenizer: {e}")

    print("\nTo use this dataset in translation_example.py, run:")
    print(
        f"python demos/translation_example.py --dataset iwslt --src_lang {args.src_lang} --tgt_lang {args.tgt_lang}"
    )


if __name__ == "__main__":
    main()
