import argparse
import os
import time
from src.data.tokenization import OptimizedBPETokenizer
from src.data.combined_wmt_translation_dataset import load_dataset_from_file


def train_optimized_bpe_tokenizer(
    language="en",
    dataset_path="combined_de_en_dataset.jsonl",
    vocab_size=16000,
    output_path=None,
    max_samples=None,
):
    """
    Train an OptimizedBPETokenizer on either English or German text from the dataset.

    Args:
        language: 'en' for English or 'de' for German
        dataset_path: Path to the combined dataset JSONL file
        vocab_size: Size of the vocabulary
        output_path: Where to save the tokenizer (defaults to models/tokenizers/combined/{language})
        max_samples: Maximum number of samples to use from the dataset
    """
    if language not in ["en", "de"]:
        raise ValueError("Language must be either 'en' for English or 'de' for German")

    # Create the output directory if it doesn't exist
    output_dir = "models/tokenizers/combined"
    if output_path is None:
        output_path = f"{output_dir}/{language}"

    os.makedirs(output_path, exist_ok=True)

    print(f"Loading dataset from {dataset_path}...")
    dataset = load_dataset_from_file(dataset_path, max_samples=max_samples)

    print(
        f"Training {language} tokenizer with vocabulary size {vocab_size} on {len(dataset)} samples..."
    )

    # Create the OptimizedBPETokenizer with appropriate num_merges
    # The vocab size will include base characters plus merges
    tokenizer = OptimizedBPETokenizer(num_merges=vocab_size - 256)

    # Extract texts for training based on language
    texts = []
    for source, target in dataset:
        if language == "en":
            texts.append(target)  # English is the target
        else:
            texts.append(source)  # German is the source

    # Train the tokenizer
    start_time = time.time()
    tokenizer.train(
        texts=texts,
        vocab_size=vocab_size,
        min_frequency=2,
        show_progress=True,
    )
    training_time = time.time() - start_time

    # Save the tokenizer
    tokenizer.save_pretrained(output_path)

    print(f"{language.upper()} Tokenizer saved to {output_path}")
    print(f"Training completed in {training_time:.2f}s")
    print(f"Vocabulary size: {tokenizer.vocab_size}")

    # Test the tokenizer
    if language == "en":
        test_text = "This is a test sentence to verify the tokenizer works correctly."
    else:
        test_text = "Dies ist ein Testsatz, um zu überprüfen, ob der Tokenizer korrekt funktioniert."

    tokens = tokenizer.tokenize(test_text)
    ids = tokenizer.encode(test_text)
    decoded = tokenizer.decode(ids)

    print(f"\nTest encoding: {test_text}")
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
        description="Train an OptimizedBPE tokenizer for translation"
    )
    parser.add_argument(
        "--language",
        type=str,
        default="en",
        choices=["en", "de"],
        help="Language to train tokenizer for (en or de)",
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
        default=16000,
        help="Vocabulary size for the tokenizer",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default=None,
        help="Where to save the tokenizer (defaults to models/tokenizers/combined/{language})",
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="Maximum number of samples to use from the dataset",
    )

    args = parser.parse_args()

    train_optimized_bpe_tokenizer(
        language=args.language,
        dataset_path=args.dataset_path,
        vocab_size=args.vocab_size,
        output_path=args.output_path,
        max_samples=args.max_samples,
    )
