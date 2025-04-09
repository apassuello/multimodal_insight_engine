import argparse
import os
import time
from src.data.tokenization import OptimizedBPETokenizer
from src.data.combined_wmt_translation_dataset import load_dataset_from_file


def load_opensubtitles_dataset(src_file, tgt_file, max_samples=None):
    """
    Load parallel sentences from OpenSubtitles files.

    Args:
        src_file: Path to source language file (e.g., German)
        tgt_file: Path to target language file (e.g., English)
        max_samples: Maximum number of samples to load (None for all)

    Returns:
        List of (source, target) sentence pairs
    """
    print(f"Loading OpenSubtitles dataset from {src_file} and {tgt_file}...")

    dataset = []
    count = 0

    with open(src_file, "r", encoding="utf-8") as src_f, open(
        tgt_file, "r", encoding="utf-8"
    ) as tgt_f:

        for src_line, tgt_line in zip(src_f, tgt_f):
            src_text = src_line.strip()
            tgt_text = tgt_line.strip()

            # Skip empty lines
            if not src_text or not tgt_text:
                continue

            dataset.append((src_text, tgt_text))
            count += 1

            if max_samples and count >= max_samples:
                break

    print(f"Loaded {len(dataset)} parallel sentence pairs")
    return dataset


def train_joint_tokenizer(
    dataset_path="combined_de_en_dataset.jsonl",
    src_file=None,
    tgt_file=None,
    vocab_size=32000,
    output_path="models/tokenizers/combined/joint",
    max_samples=None,
):
    """
    Train a joint BPE tokenizer on both German and English text from the dataset.

    Args:
        dataset_path: Path to the combined dataset JSONL file (if using JSONL format)
        src_file: Path to source language file (for OpenSubtitles format)
        tgt_file: Path to target language file (for OpenSubtitles format)
        vocab_size: Size of the vocabulary
        output_path: Where to save the tokenizer
        max_samples: Maximum number of samples to use from the dataset
    """
    # Create the output directory if it doesn't exist
    os.makedirs(output_path, exist_ok=True)

    # Load dataset based on provided files
    if src_file and tgt_file:
        dataset = load_opensubtitles_dataset(src_file, tgt_file, max_samples)
    else:
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
        "--src_file",
        type=str,
        default=None,
        help="Path to source language file (for OpenSubtitles format)",
    )
    parser.add_argument(
        "--tgt_file",
        type=str,
        default=None,
        help="Path to target language file (for OpenSubtitles format)",
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
        src_file=args.src_file,
        tgt_file=args.tgt_file,
        vocab_size=args.vocab_size,
        output_path=args.output_path,
        max_samples=args.max_samples,
    )
