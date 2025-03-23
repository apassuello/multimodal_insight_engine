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
    
    Args:
        src_texts: List of source language texts
        tgt_texts: List of target language texts
        vocab_size: Target vocabulary size
        min_frequency: Minimum token frequency
        save_dir: Directory to save the tokenizer
        
    Returns:
        Trained BPE tokenizer
    """
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