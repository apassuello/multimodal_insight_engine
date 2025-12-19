import argparse
import os
import time
import re
import json
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


def preprocess_text(
    text, preserve_case=True, preserve_punctuation=True, add_special_tokens=True
):
    """
    Preprocess text by optionally lowercasing and handling punctuation.

    Args:
        text: Input text to preprocess
        preserve_case: Whether to preserve original case
        preserve_punctuation: Whether to preserve punctuation
        add_special_tokens: Whether to add spaces around punctuation for better tokenization

    Returns:
        Preprocessed text
    """
    if not preserve_case:
        text = text.lower()

    if preserve_punctuation:
        if add_special_tokens:
            # Add spaces around punctuation to make them separate tokens
            # This helps the model learn punctuation patterns better
            punct_pattern = r'([.,!?;:()[\]{}"\'-])'
            text = re.sub(punct_pattern, r" \1 ", text)

            # Fix spacing for quotes and parentheses
            text = re.sub(r"\( ", r"(", text)
            text = re.sub(r" \)", r")", text)
            text = re.sub(r'" ', r'"', text)
            text = re.sub(r' "', r'"', text)

            # Handle special German characters
            text = re.sub(r"ß", r"ß", text)  # Preserve eszett

            # Clean up multiple spaces
            text = re.sub(r"\s+", " ", text).strip()
    else:
        # Remove punctuation if not preserving it
        text = re.sub(r"[^\w\s]", "", text)

    return text


def train_joint_tokenizer(
    dataset_path="combined_de_en_dataset.jsonl",
    src_file=None,
    tgt_file=None,
    vocab_size=32000,
    output_path="models/tokenizers/combined/joint",
    max_samples=None,
    preserve_case=True,
    preserve_punctuation=True,
    add_special_tokens=True,
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
        preserve_case: Whether to preserve original case
        preserve_punctuation: Whether to preserve punctuation
        add_special_tokens: Whether to add spaces around punctuation
    """
    # Create the output directory if it doesn't exist
    os.makedirs(output_path, exist_ok=True)

    # Load dataset based on provided files
    if src_file and tgt_file:
        dataset = load_opensubtitles_dataset(src_file, tgt_file, max_samples)
    else:
        print(f"Loading dataset from {dataset_path}...")
        dataset = load_dataset_from_file(dataset_path, max_samples=max_samples)

    tokenizer_type = "case-" + ("sensitive" if preserve_case else "insensitive")
    tokenizer_type += "_punct-" + ("preserved" if preserve_punctuation else "removed")

    print(
        f"Training joint DE-EN {tokenizer_type} tokenizer with vocabulary size {vocab_size} on {len(dataset)} samples..."
    )

    # Create the OptimizedBPETokenizer with appropriate num_merges
    # The vocab size will include base characters plus merges
    lower_case = not preserve_case
    tokenizer = OptimizedBPETokenizer(
        num_merges=vocab_size - 256,
        lower_case=lower_case,
        preserve_punctuation=preserve_punctuation,
        preserve_case=preserve_case,
    )

    # Extract both German and English texts for training, with preprocessing
    texts = []
    for source, target in dataset:
        # Apply preprocessing to preserve important linguistic features
        source = preprocess_text(
            source,
            preserve_case=preserve_case,
            preserve_punctuation=preserve_punctuation,
            add_special_tokens=add_special_tokens,
        )
        target = preprocess_text(
            target,
            preserve_case=preserve_case,
            preserve_punctuation=preserve_punctuation,
            add_special_tokens=add_special_tokens,
        )

        texts.append(source)  # German
        texts.append(target)  # English

    print(f"Total texts for training: {len(texts)}")

    # Train the tokenizer
    start_time = time.time()
    tokenizer.train(
        texts=texts, vocab_size=vocab_size, min_frequency=2, show_progress=True
    )
    training_time = time.time() - start_time

    # Save tokenizer config for reference
    config_info = {
        "vocab_size": vocab_size,
        "preserve_case": preserve_case,
        "preserve_punctuation": preserve_punctuation,
        "add_special_tokens": add_special_tokens,
    }

    # Save config to a separate file
    config_path = os.path.join(output_path, "tokenizer_config.json")
    with open(config_path, "w") as f:
        json.dump(config_info, f, indent=2)

    # Save the tokenizer
    tokenizer.save_pretrained(output_path)

    print(f"Joint DE-EN Tokenizer saved to {output_path}")
    print(f"Training completed in {training_time:.2f}s")
    print(f"Vocabulary size: {tokenizer.vocab_size}")
    print(f"Tokenizer configuration: {tokenizer_type}")

    # Test the tokenizer on both languages with cases that highlight punctuation and case
    test_texts = {
        "en": "This is a test sentence, with punctuation! Does it work correctly?",
        "de": "Dies ist ein Testsatz mit Zeichensetzung! Funktioniert es richtig? Die Frau geht nach Hause.",
        "en_case": "The US President visited Germany in May.",
        "de_case": "Der Präsident besucht Berlin im Mai. Sie sind sehr nett.",
    }

    print("\n=== Testing Joint Tokenizer ===")
    for lang, text in test_texts.items():
        tokens = tokenizer.tokenize(text)
        ids = tokenizer.encode(text)
        decoded = tokenizer.decode(ids)

        print(f"\n{lang.upper()} Test: {text}")
        print(
            f"Tokens ({len(tokens)}): {tokens[:15]}..."
            if len(tokens) > 15
            else f"Tokens: {tokens}"
        )
        print(f"IDs: {ids[:10]}..." if len(ids) > 10 else f"IDs: {ids}")
        print(f"Decoded: {decoded}")

        # Check if case and punctuation are preserved in the decoded output
        case_preserved = (
            any(c.isupper() for c in decoded)
            if any(c.isupper() for c in text)
            else True
        )
        punct_preserved = (
            any(c in decoded for c in ".,!?")
            if any(c in text for c in ".,!?")
            else True
        )

        print(f"Case preserved: {case_preserved}")
        print(f"Punctuation preserved: {punct_preserved}")

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
    parser.add_argument(
        "--preserve_case",
        type=lambda x: x.lower() == "true",
        default=True,
        help="Whether to preserve original case (true/false)",
    )
    parser.add_argument(
        "--preserve_punctuation",
        type=lambda x: x.lower() == "true",
        default=True,
        help="Whether to preserve punctuation (true/false)",
    )
    parser.add_argument(
        "--add_special_tokens",
        type=lambda x: x.lower() == "true",
        default=True,
        help="Whether to add spaces around punctuation to make them separate tokens (true/false)",
    )

    args = parser.parse_args()

    train_joint_tokenizer(
        dataset_path=args.dataset_path,
        src_file=args.src_file,
        tgt_file=args.tgt_file,
        vocab_size=args.vocab_size,
        output_path=args.output_path,
        max_samples=args.max_samples,
        preserve_case=args.preserve_case,
        preserve_punctuation=args.preserve_punctuation,
        add_special_tokens=args.add_special_tokens,
    )
