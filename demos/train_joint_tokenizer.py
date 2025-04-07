import argparse
import os
import time
from src.data.tokenization import OptimizedBPETokenizer
from src.data.combined_wmt_translation_dataset import load_dataset_from_file


def train_joint_tokenizer(
    dataset_path="combined_de_en_dataset.jsonl",
    vocab_size=32000,
    output_path="models/tokenizers/combined/joint",
    max_samples=None,
):
    """
    Train a joint BPE tokenizer on both German and English text from the dataset.

    Args:
        dataset_path: Path to the combined dataset JSONL file
        vocab_size: Size of the vocabulary
        output_path: Where to save the tokenizer
        max_samples: Maximum number of samples to use from the dataset
    """
    # Create the output directory if it doesn't exist
    os.makedirs(output_path, exist_ok=True)

    print(f"Loading dataset from {dataset_path}...")
    dataset = load_dataset_from_file(dataset_path, max_samples=max_samples)

    print(
        f"Training joint DE-EN tokenizer with vocabulary size {vocab_size} on {len(dataset)} samples..."
    )

    # Create the OptimizedBPETokenizer with appropriate num_merges
    # The vocab size will include base characters plus merges
    tokenizer = OptimizedBPETokenizer(num_merges=vocab_size - 256)

    # Extract both German and English texts for training
    texts = []
    for source, target in dataset:
        texts.append(source)  # German
        texts.append(target)  # English

    print(f"Total texts for training: {len(texts)}")

    # Train the tokenizer
    start_time = time.time()
    tokenizer.train(
        texts=texts, vocab_size=vocab_size, min_frequency=2, show_progress=True
    )
    training_time = time.time() - start_time

    # Save the tokenizer
    tokenizer.save_pretrained(output_path)

    print(f"Joint DE-EN Tokenizer saved to {output_path}")
    print(f"Training completed in {training_time:.2f}s")
    print(f"Vocabulary size: {tokenizer.vocab_size}")

    # Test the tokenizer on both languages
    test_texts = {
        "en": "This is a test sentence to verify the tokenizer works correctly.",
        "de": "Dies ist ein Testsatz, um zu überprüfen, ob der Tokenizer korrekt funktioniert.",
    }

    print("\n=== Testing Joint Tokenizer ===")
    for lang, text in test_texts.items():
        tokens = tokenizer.tokenize(text)
        ids = tokenizer.encode(text)
        decoded = tokenizer.decode(ids)

        print(f"\n{lang.upper()} Test: {text}")
        print(
            f"Tokens ({len(tokens)}): {tokens[:10]}..."
            if len(tokens) > 10
            else f"Tokens: {tokens}"
        )
        print(f"IDs: {ids[:10]}..." if len(ids) > 10 else f"IDs: {ids}")
        print(f"Decoded: {decoded}")

    return tokenizer


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train a joint BPE tokenizer for German and English"
    )
    parser.add_argument(
        "--dataset_path",
        type=str,
        default="combined_de_en_dataset.jsonl",
        help="Path to the combined dataset JSONL file",
    )
    parser.add_argument(
        "--vocab_size",
        type=int,
        default=32000,
        help="Vocabulary size for the tokenizer",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="models/tokenizers/combined/joint",
        help="Where to save the tokenizer",
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="Maximum number of samples to use from the dataset",
    )

    args = parser.parse_args()

    train_joint_tokenizer(
        dataset_path=args.dataset_path,
        vocab_size=args.vocab_size,
        output_path=args.output_path,
        max_samples=args.max_samples,
    )
