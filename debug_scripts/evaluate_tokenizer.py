#!/usr/bin/env python3
# evaluate_tokenizer.py
# Script to evaluate BPE tokenizer quality

import os
import sys
import argparse
import random
import re
from typing import List, Dict
from collections import Counter

# Add parent directory to path to import local modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import the tokenizer
from src.data.tokenization import OptimizedBPETokenizer
from src.data.opensubtitles_dataset import OpenSubtitlesDataset


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Evaluate BPE tokenizer quality")
    
    parser.add_argument("--tokenizer_path", type=str, required=True,
                        help="Path to the tokenizer directory")
    parser.add_argument("--test_data", type=str, default="data/os",
                        help="Path to test data directory")
    parser.add_argument("--lang", type=str, required=True,
                        help="Language to evaluate")
    parser.add_argument("--pair_lang", type=str, default=None,
                        help="Paired language (if using parallel corpus)")
    parser.add_argument("--num_examples", type=int, default=1000,
                        help="Number of examples to evaluate")
    
    args = parser.parse_args()
    
    # If pair_lang is not specified, default based on language
    if args.pair_lang is None:
        args.pair_lang = "de" if args.lang == "en" else "en"
        
    return args


def preprocess_text(text: str, tokenizer: OptimizedBPETokenizer) -> str:
    """
    Apply the same preprocessing as the tokenizer.
    
    Args:
        text: Input text
        tokenizer: Tokenizer with preprocessing rules
        
    Returns:
        Preprocessed text
    """
    # Apply lowercase if tokenizer uses it
    if hasattr(tokenizer, 'lower_case') and tokenizer.lower_case:
        text = text.lower()
    
    # Remove punctuation and special characters
    # This matches what the tokenizer does in its preprocess method
    text = re.sub(r'[^\w\s]', '', text)
    
    return text


def load_test_data(data_dir: str, lang: str, pair_lang: str, num_examples: int) -> List[str]:
    """
    Load test data.
    
    Args:
        data_dir: Directory containing test data
        lang: Language to evaluate
        pair_lang: Paired language
        num_examples: Number of examples to load
        
    Returns:
        List of text examples
    """
    print(f"Loading test data for language: {lang} from {data_dir}")
    
    # Load dataset
    src_lang = lang if lang < pair_lang else pair_lang  # Alphabetical order
    tgt_lang = pair_lang if lang < pair_lang else lang
    
    dataset = OpenSubtitlesDataset(
        data_dir=data_dir,
        src_lang=src_lang,
        tgt_lang=tgt_lang,
        max_examples=num_examples * 2  # Load more examples than needed
    )
    
    # Get data for the specified language
    texts = dataset.src_data if lang == src_lang else dataset.tgt_data
    
    # Sample examples if we have more than needed
    if len(texts) > num_examples:
        texts = random.sample(texts, num_examples)
    
    print(f"Loaded {len(texts)} test examples")
    return texts


def evaluate_token_to_word_ratio(tokenizer: OptimizedBPETokenizer, texts: List[str]) -> Dict:
    """
    Calculate the average number of tokens per word and per character.
    
    Args:
        tokenizer: BPE tokenizer
        texts: List of text examples
        
    Returns:
        Dictionary with evaluation metrics
    """
    total_words = 0
    total_chars = 0
    total_tokens = 0
    
    for text in texts:
        # Apply preprocessing
        processed_text = preprocess_text(text, tokenizer)
        
        # Count words, characters, and tokens
        words = processed_text.split()
        chars = len(processed_text.replace(" ", ""))
        tokens = tokenizer.tokenize(processed_text)
        
        total_words += len(words)
        total_chars += chars
        total_tokens += len(tokens)
    
    # Calculate ratios
    tokens_per_word = total_tokens / total_words if total_words > 0 else 0
    tokens_per_char = total_tokens / total_chars if total_chars > 0 else 0
    
    return {
        "total_words": total_words,
        "total_chars": total_chars,
        "total_tokens": total_tokens,
        "tokens_per_word": tokens_per_word,
        "tokens_per_char": tokens_per_char
    }


def evaluate_oov_rate(tokenizer: OptimizedBPETokenizer, texts: List[str]) -> Dict:
    """
    Calculate the out-of-vocabulary (OOV) rate.
    
    Args:
        tokenizer: BPE tokenizer
        texts: List of text examples
        
    Returns:
        Dictionary with OOV metrics
    """
    unknown_token_idx = tokenizer.special_tokens.get("unk_token_idx", 3)  # Default UNK index
    
    total_tokens = 0
    unknown_tokens = 0
    
    for text in texts:
        # Apply preprocessing
        processed_text = preprocess_text(text, tokenizer)
        
        # Encode text
        token_ids = tokenizer.encode(processed_text)
        
        # Count total and unknown tokens
        total_tokens += len(token_ids)
        unknown_tokens += token_ids.count(unknown_token_idx)
    
    # Calculate OOV rate
    oov_rate = unknown_tokens / total_tokens if total_tokens > 0 else 0
    
    return {
        "total_tokens": total_tokens,
        "unknown_tokens": unknown_tokens,
        "oov_rate": oov_rate
    }


def evaluate_roundtrip_accuracy(tokenizer: OptimizedBPETokenizer, texts: List[str]) -> Dict:
    """
    Calculate the percentage of texts that survive roundtrip conversion.
    
    Args:
        tokenizer: BPE tokenizer
        texts: List of text examples
        
    Returns:
        Dictionary with roundtrip metrics
    """
    successful = 0
    partial_matches = 0
    
    for text in texts:
        # Apply the same preprocessing as the tokenizer
        processed_text = preprocess_text(text, tokenizer)
        
        # Perform roundtrip: text -> tokens -> text
        token_ids = tokenizer.encode(processed_text)
        decoded = tokenizer.decode(token_ids)
        
        # Check if roundtrip is successful (compare against processed text)
        if decoded == processed_text:
            successful += 1
        elif decoded.replace(" ", "") == processed_text.replace(" ", ""):
            # If only whitespace differences
            partial_matches += 1
    
    # Calculate success rate
    success_rate = successful / len(texts) if texts else 0
    partial_rate = partial_matches / len(texts) if texts else 0
    
    return {
        "total_texts": len(texts),
        "successful": successful,
        "partial_matches": partial_matches,
        "success_rate": success_rate,
        "partial_rate": partial_rate
    }


def analyze_common_tokens(tokenizer: OptimizedBPETokenizer, texts: List[str], top_n: int = 20) -> Dict:
    """
    Analyze the most common tokens in the texts.
    
    Args:
        tokenizer: BPE tokenizer
        texts: List of text examples
        top_n: Number of top tokens to return
        
    Returns:
        Dictionary with token analysis
    """
    token_counter = Counter()
    
    for text in texts:
        # Apply preprocessing
        processed_text = preprocess_text(text, tokenizer)
        
        # Tokenize text
        tokens = tokenizer.tokenize(processed_text)
        token_counter.update(tokens)
    
    # Get most common tokens
    most_common = token_counter.most_common(top_n)
    
    return {
        "total_unique_tokens": len(token_counter),
        "most_common_tokens": most_common
    }


def print_tokenization_examples(tokenizer: OptimizedBPETokenizer, texts: List[str], num_examples: int = 5):
    """
    Print examples of tokenization.
    
    Args:
        tokenizer: BPE tokenizer
        texts: List of text examples
        num_examples: Number of examples to print
    """
    print("\nTokenization Examples:")
    print("="*80)
    
    # Sample some examples
    sample_texts = random.sample(texts, min(num_examples, len(texts)))
    
    for i, text in enumerate(sample_texts):
        # Store original text for comparison
        original_text = text
        
        # Apply same preprocessing as tokenizer
        processed_text = preprocess_text(text, tokenizer)
        
        # Tokenize and encode
        tokens = tokenizer.tokenize(processed_text)
        ids = tokenizer.encode(processed_text)
        decoded = tokenizer.decode(ids)
        
        # Print results
        print(f"\nExample {i+1}:")
        print(f"Original  : {original_text}")
        print(f"Processed : {processed_text}")  # Show the processed text (lowercased and punctuation removed)
        print(f"Tokens    : {tokens}")
        print(f"Token IDs : {ids}")
        print(f"Decoded   : {decoded}")
        print(f"Roundtrip : {'✓' if processed_text == decoded else '✗'}")
        print("-"*80)


def main():
    """Main function to evaluate tokenizer quality."""
    # Parse arguments
    args = parse_args()
    
    # Load tokenizer
    tokenizer_path = os.path.join(args.tokenizer_path, args.lang)
    print(f"Loading tokenizer from: {tokenizer_path}")
    tokenizer = OptimizedBPETokenizer.from_pretrained(tokenizer_path)
    print(f"Tokenizer loaded with vocabulary size: {tokenizer.vocab_size}")
    
    # Load test data
    texts = load_test_data(
        data_dir=args.test_data,
        lang=args.lang,
        pair_lang=args.pair_lang,
        num_examples=args.num_examples
    )
    
    # Evaluate token-to-word ratio
    ratio_metrics = evaluate_token_to_word_ratio(tokenizer, texts)
    print("\nToken-to-Word Ratio Metrics:")
    print(f"Total words  : {ratio_metrics['total_words']}")
    print(f"Total chars  : {ratio_metrics['total_chars']}")
    print(f"Total tokens : {ratio_metrics['total_tokens']}")
    print(f"Tokens/word  : {ratio_metrics['tokens_per_word']:.3f}")
    print(f"Tokens/char  : {ratio_metrics['tokens_per_char']:.3f}")
    
    # Evaluate OOV rate
    oov_metrics = evaluate_oov_rate(tokenizer, texts)
    print("\nOOV Rate Metrics:")
    print(f"Total tokens    : {oov_metrics['total_tokens']}")
    print(f"Unknown tokens  : {oov_metrics['unknown_tokens']}")
    print(f"OOV rate        : {oov_metrics['oov_rate']:.6f} ({oov_metrics['oov_rate']*100:.4f}%)")
    
    # Evaluate roundtrip accuracy
    roundtrip_metrics = evaluate_roundtrip_accuracy(tokenizer, texts)
    print("\nRoundtrip Accuracy Metrics:")
    print(f"Total texts       : {roundtrip_metrics['total_texts']}")
    print(f"Successful        : {roundtrip_metrics['successful']}")
    print(f"Partial matches   : {roundtrip_metrics['partial_matches']}")
    print(f"Success rate      : {roundtrip_metrics['success_rate']:.4f} ({roundtrip_metrics['success_rate']*100:.2f}%)")
    print(f"Partial rate      : {roundtrip_metrics['partial_rate']:.4f} ({roundtrip_metrics['partial_rate']*100:.2f}%)")
    
    # Analyze common tokens
    token_metrics = analyze_common_tokens(tokenizer, texts)
    print("\nToken Analysis:")
    print(f"Total unique tokens: {token_metrics['total_unique_tokens']}")
    print("\nMost common tokens:")
    for token, count in token_metrics['most_common_tokens']:
        print(f"  {token:<15}: {count}")
    
    # Print tokenization examples
    print_tokenization_examples(tokenizer, texts)
    
    print("\nTokenizer Evaluation Complete!")


if __name__ == "__main__":
    main() 