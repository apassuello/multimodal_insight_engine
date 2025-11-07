"""MODULE: joint_bpe_training.py
PURPOSE: Implements joint BPE (Byte-Pair Encoding) tokenizer training for multilingual text processing, specifically designed for machine translation tasks.

KEY COMPONENTS:
- train_joint_bpe_tokenizer: Main function for training a shared BPE tokenizer on source and target language texts
- main: Example usage and demonstration of the joint BPE tokenizer training

DEPENDENCIES:
- src.data.tokenization (BPETokenizer)

SPECIAL NOTES:
- Creates a shared vocabulary for both source and target languages
- Supports configurable vocabulary size and minimum token frequency
- Saves trained tokenizer for later use in both languages
"""

import os
from typing import List

from src.data.tokenization import BPETokenizer


def train_joint_bpe_tokenizer(
    src_texts: List[str],
    tgt_texts: List[str],
    vocab_size: int = 8000,
    min_frequency: int = 2,
    save_dir: str = "models/tokenizers"
) -> BPETokenizer:
    """
    Train a joint BPE tokenizer for both source and target texts.
    
    This function creates a shared vocabulary by training a BPE tokenizer on
    combined source and target language texts. This approach ensures consistent
    tokenization across both languages and helps capture shared subword patterns.
    
    Args:
        src_texts: List of source language texts
        tgt_texts: List of target language texts
        vocab_size: Target vocabulary size (default: 8000)
        min_frequency: Minimum token frequency for inclusion in vocabulary (default: 2)
        save_dir: Directory to save the trained tokenizer (default: "models/tokenizers")
        
    Returns:
        BPETokenizer: Trained BPE tokenizer with shared vocabulary
        
    Raises:
        ValueError: If either src_texts or tgt_texts is empty
    """
    if not src_texts or not tgt_texts:
        raise ValueError("Source and target texts cannot be empty")

    print(f"Training joint BPE tokenizer with vocab size {vocab_size}...")

    # Combine texts from both languages
    combined_texts = src_texts + tgt_texts

    # Train BPE tokenizer
    tokenizer = BPETokenizer(num_merges=vocab_size-256)
    tokenizer.train(
        texts=combined_texts,
        vocab_size=vocab_size,
        min_frequency=min_frequency,
        show_progress=True,
    )

    # Save tokenizer
    tokenizer.save_pretrained(save_dir)

    return tokenizer


def main():
    """
    Demonstrate the usage of joint BPE tokenizer training.
    
    This function provides an example of how to train and use a joint BPE tokenizer
    for a simple machine translation task with German and English texts.
    """
    # Example usage
    src_texts = ["Hallo, wie geht es dir?", "Ich lerne maschinelle Übersetzung."]
    tgt_texts = ["Hello, how are you?", "I am learning machine translation."]

    # Train joint BPE tokenizer
    joint_tokenizer = train_joint_bpe_tokenizer(src_texts, tgt_texts, vocab_size=8000)

    # Load the tokenizer for both languages
    src_tokenizer = BPETokenizer.from_pretrained("models/tokenizers")
    tgt_tokenizer = BPETokenizer.from_pretrained("models/tokenizers")

    print(f"Source tokenizer vocab size: {src_tokenizer.vocab_size}")
    print(f"Target tokenizer vocab size: {tgt_tokenizer.vocab_size}")


if __name__ == "__main__":
    main()


def extract_file_metadata(file_path=__file__):
    """
    Extract structured metadata about this module.
    
    Args:
        file_path: Path to the source file (defaults to current file)
        
    Returns:
        dict: Structured metadata about the module's purpose and components
    """
    return {
        "filename": os.path.basename(file_path),
        "module_purpose": "Implements joint BPE tokenizer training for multilingual text processing in machine translation tasks",
        "key_classes": [],  # No classes in this module
        "key_functions": [
            {
                "name": "train_joint_bpe_tokenizer",
                "signature": "train_joint_bpe_tokenizer(src_texts: List[str], tgt_texts: List[str], vocab_size: int = 8000, min_frequency: int = 2, save_dir: str = 'models/tokenizers') -> BPETokenizer",
                "brief_description": "Trains a shared BPE tokenizer on combined source and target language texts"
            },
            {
                "name": "main",
                "signature": "main()",
                "brief_description": "Demonstrates the usage of joint BPE tokenizer training with example texts"
            }
        ],
        "external_dependencies": ["src.data.tokenization"],
        "complexity_score": 3,  # Low complexity as it's a straightforward tokenizer training module
    }
