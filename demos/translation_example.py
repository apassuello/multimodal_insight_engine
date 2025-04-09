# examples/translation_example_fixed.py
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict, Tuple
import os
import sys
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import requests
import random
from tqdm import tqdm
import re
from collections import Counter
import argparse

# Disable interactive mode for matplotlib to prevent opening windows
plt.ioff()

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models.transformer import EncoderDecoderTransformer
from src.data.sequence_data import TransformerDataModule
from src.training.transformer_trainer import TransformerTrainer
from src.training.transformer_utils import create_padding_mask, create_causal_mask
from src.data.tokenization import OptimizedBPETokenizer
import unicodedata
from src.optimization.mixed_precision import MixedPrecisionConverter
from src.data.europarl_dataset import EuroparlDataset
from src.data.opensubtitles_dataset import OpenSubtitlesDataset
from src.data.iwslt_dataset import IWSLTDataset
from debug_scripts.debug_transformer import (
    attach_debugger_to_trainer,
    debug_sample_batch,
)
from src.data.combined_dataset import CombinedDataset


def clean_translation_output(text):
    """
    Minimal cleaning of translation output - only fix capitalization and spaces.

    Args:
        text: Raw translation text

    Returns:
        Cleaned translation text
    """
    # Remove BOS token if present
    text = text.replace("<bos>", "").strip()

    # Fix common special tokens
    text = text.replace("_space", " ")
    text = text.replace("_dash", "-")
    text = text.replace("_comma", ",")
    text = text.replace("_period", ".")
    text = text.replace("_question", "?")

    # Fix double spaces
    text = re.sub(r"\s+", " ", text).strip()

    # Capitalize first letter only
    if text and len(text) > 0:
        text = text[0].upper() + text[1:]

    return text


def generate_translations(
    model,
    tokenizer,  # Single tokenizer for both source and target
    src_sentences,
    max_length=100,
    device=None,
    fallback_token="[UNK]",
    sampling_temp=1.0,  # Temperature for sampling (1.0 = greedy)
    sampling_topk=1,  # Default to greedy decoding (top-1)
):
    """
    Generate translations for a list of source sentences using greedy decoding by default.

    Args:
        model: Trained transformer model
        tokenizer: Joint vocabulary tokenizer
        src_sentences: List of source language sentences
        max_length: Maximum length of generated translations
        device: Device to run inference on
        fallback_token: Token to use when encountering out-of-vocabulary tokens
        sampling_temp: Temperature for sampling (1.0 = greedy)
        sampling_topk: Number of top tokens to consider for sampling (1 = greedy)

    Returns:
        List of translated sentences
    """
    model.eval()
    translations = []
    raw_translations = []

    # Default to CPU if device not provided
    if device is None:
        device = torch.device("cpu")
    elif isinstance(device, str):
        device = torch.device(device)

    # Get vocabulary size
    vocab_size = model.encoder.token_embedding.embedding.weight.shape[0]

    # Check for vocab size mismatch and print warning
    if tokenizer.vocab_size != vocab_size:
        print(
            f"WARNING: Tokenizer vocabulary size ({tokenizer.vocab_size}) doesn't match model ({vocab_size})"
        )
        print("This may result in incorrect translations!")

    # Get token indices for unknown tokens to use as fallbacks
    try:
        unk_token_idx = tokenizer.token_to_id(fallback_token)
        if unk_token_idx is None or unk_token_idx >= vocab_size:
            unk_token_idx = 0  # Use a safe default
    except:
        unk_token_idx = 0

    with torch.no_grad():
        for src_text in src_sentences:
            try:
                # Tokenize source text
                src_ids = tokenizer.encode(src_text)

                # Add special tokens
                src_ids = (
                    [tokenizer.special_tokens["bos_token_idx"]]
                    + src_ids
                    + [tokenizer.special_tokens["eos_token_idx"]]
                )

                # Ensure all tokens are in vocabulary range
                src_ids = [min(token_id, vocab_size - 1) for token_id in src_ids]

                src_tensor = torch.tensor([src_ids], dtype=torch.long).to(device)

                # Create source mask
                src_mask = create_padding_mask(
                    src_tensor, tokenizer.special_tokens["pad_token_idx"]
                )

                # Initialize target with BOS token
                tgt_bos_idx = min(
                    tokenizer.special_tokens["bos_token_idx"], vocab_size - 1
                )
                tgt_ids = [tgt_bos_idx]
                tgt_tensor = torch.tensor([tgt_ids], dtype=torch.long).to(device)

                # Generate translation auto-regressively
                unique_tokens = set()  # Track unique tokens to detect repetition
                repetition_count = 0
                max_repetitions = 5  # Stop after this many repeated tokens
                tokens_generated = 0

                for _ in range(max_length):
                    tokens_generated += 1

                    # Create target mask (causal)
                    tgt_mask = create_causal_mask(tgt_tensor.size(1), device)

                    # Get model prediction
                    logits = model(
                        src_tensor, tgt_tensor, src_mask=src_mask, tgt_mask=tgt_mask
                    )
                    next_token_logits = logits[0, -1]

                    # Apply temperature to logits if not 1.0
                    if sampling_temp != 1.0:
                        next_token_logits = next_token_logits / sampling_temp

                    # Get top-k tokens for potential sampling
                    top_token_probs, top_tokens = torch.topk(
                        next_token_logits, sampling_topk
                    )

                    # Convert to probabilities
                    top_token_probs = torch.softmax(top_token_probs, dim=-1)

                    # Sample from top-k or use greedy decoding
                    if sampling_temp == 1.0 or sampling_topk == 1:
                        # Greedy decoding - just take the highest probability token
                        next_token = top_tokens[0].item()
                    else:
                        # Sample from top-k based on probabilities
                        try:
                            # Convert to a proper probability distribution
                            if (
                                torch.isnan(top_token_probs).any()
                                or not torch.isfinite(top_token_probs).all()
                            ):
                                # Fallback to uniform distribution if probabilities are invalid
                                top_token_probs = torch.ones_like(
                                    top_token_probs
                                ) / len(top_token_probs)

                            # Ensure it sums to 1.0
                            if abs(top_token_probs.sum().item() - 1.0) > 1e-6:
                                top_token_probs = (
                                    top_token_probs / top_token_probs.sum()
                                )

                            # Use explicit integer index for multinomial
                            index_tensor = torch.multinomial(
                                top_token_probs, num_samples=1
                            )
                            next_token_idx = index_tensor[0].item()
                            next_token = top_tokens[int(next_token_idx)].item()
                        except Exception as e:
                            print(
                                f"Sampling error: {e}, falling back to greedy decoding"
                            )
                            next_token = top_tokens[0].item()

                    # Check if we're getting repetitions
                    if (
                        next_token in unique_tokens and tokens_generated > 15
                    ):  # Allow more repetition, only check after 15 tokens
                        repetition_count += 1

                        # Try to avoid repetition by sampling from remaining tokens - only if repetition becomes severe
                        if (
                            repetition_count >= 5 and len(top_tokens) > 1
                        ):  # Increased threshold
                            for alt_idx in range(1, len(top_tokens)):
                                alt_token = top_tokens[alt_idx].item()
                                if (
                                    alt_token not in unique_tokens
                                    or alt_token
                                    == tokenizer.special_tokens["eos_token_idx"]
                                ):
                                    next_token = alt_token
                                    repetition_count = 0
                                    break
                    else:
                        repetition_count = 0

                    unique_tokens.add(next_token)

                    # Stop if too many repetitions or EOS token
                    if repetition_count >= max_repetitions:
                        # Add EOS token if there are too many repetitions
                        next_token = tokenizer.special_tokens["eos_token_idx"]

                    # Add predicted token to target sequence
                    tgt_ids.append(next_token)
                    tgt_tensor = torch.tensor([tgt_ids], dtype=torch.long).to(device)

                    # Stop if EOS token is generated
                    if next_token == tokenizer.special_tokens["eos_token_idx"]:
                        break

                # Process output tokens, removing special tokens
                output_tokens = []
                for i, token_id in enumerate(tgt_ids):
                    # Skip BOS and EOS tokens
                    if (
                        token_id == tokenizer.special_tokens["bos_token_idx"]
                        or token_id == tokenizer.special_tokens["eos_token_idx"]
                    ):
                        continue

                    # Skip unknown tokens
                    if token_id >= vocab_size:
                        continue

                    # Add token to output
                    output_tokens.append(token_id)

                # Decode tokens to text
                if output_tokens:
                    raw_translated_text = tokenizer.decode(output_tokens)
                else:
                    raw_translated_text = "[Empty translation]"

                # Apply post-processing to clean up BPE artifacts
                cleaned_text = clean_translation_output(raw_translated_text)

                # Use fallback if we get mostly repetitive output
                if len(set(cleaned_text.split())) <= 1 and len(cleaned_text) > 5:
                    print(
                        "Warning: Generated repetitive translation. Attempting different decoding..."
                    )
                    # Try again with more randomness in sampling
                    return generate_translations(
                        model,
                        tokenizer,
                        [src_text],
                        max_length,
                        device,
                        fallback_token,
                        sampling_temp=0.7,  # Add some randomness
                        sampling_topk=10,  # Consider more candidates
                    )

                raw_translations.append(raw_translated_text)
                translations.append(cleaned_text)

            except Exception as e:
                print(f"Error during translation: {e}")
                raw_translations.append("[Translation error]")
                translations.append("[Translation error]")

    return translations, raw_translations


def load_pretrained_model(model_path, src_tokenizer, tgt_tokenizer, device):
    """
    Load a pretrained translation model.

    Args:
        model_path: Path to the saved model checkpoint
        src_tokenizer: Source language tokenizer
        tgt_tokenizer: Target language tokenizer
        device: Device to load the model on

    Returns:
        Loaded model
    """
    # Check if model exists
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"No model found at {model_path}")

    # Load checkpoint
    checkpoint = torch.load(model_path, map_location=device)

    # Extract or infer model configuration
    if "model_config" in checkpoint:
        model_config = checkpoint["model_config"]
        print(f"Using saved model configuration: {model_config}")
    else:
        # Try to infer configuration from state dict
        if "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]
            # Check encoder/decoder layers
            encoder_layers = 0
            decoder_layers = 0
            for key in state_dict.keys():
                if "encoder.layers." in key:
                    layer_num = int(key.split("encoder.layers.")[1].split(".")[0])
                    encoder_layers = max(encoder_layers, layer_num + 1)
                if "decoder.layers." in key:
                    layer_num = int(key.split("decoder.layers.")[1].split(".")[0])
                    decoder_layers = max(decoder_layers, layer_num + 1)

            # Get vocabulary sizes from the state dict
            src_vocab_size = state_dict[
                "encoder.token_embedding.embedding.weight"
            ].shape[0]
            tgt_vocab_size = state_dict[
                "decoder.token_embedding.embedding.weight"
            ].shape[0]
            d_model = state_dict["encoder.token_embedding.embedding.weight"].shape[1]

            # Infer other parameters from the first layer to ensure compatibility
            num_heads = None
            for key in state_dict.keys():
                if "self_attn.query_projection.weight" in key:
                    # Infer number of heads from attention head dimension
                    head_dim = state_dict[key].shape[0] // d_model
                    num_heads = d_model // head_dim
                    break

            # Use inferred values, fallback to defaults if needed
            model_config = {
                "src_vocab_size": src_vocab_size,
                "tgt_vocab_size": tgt_vocab_size,
                "d_model": d_model,
                "num_heads": num_heads or 8,
                "num_encoder_layers": encoder_layers,
                "num_decoder_layers": decoder_layers,
                "d_ff": 2048,  # Default, but should be inferred if possible
                "dropout": 0.1,
                "max_seq_length": 100,
                "positional_encoding": "sinusoidal",
                "share_embeddings": False,
            }
            print(f"Inferred model configuration: {model_config}")
        else:
            raise ValueError(
                "Checkpoint does not contain model_config or model_state_dict"
            )

    # Make sure we use the exact vocabulary sizes from the saved model
    if "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
        model_config["src_vocab_size"] = state_dict[
            "encoder.token_embedding.embedding.weight"
        ].shape[0]
        model_config["tgt_vocab_size"] = state_dict[
            "decoder.token_embedding.embedding.weight"
        ].shape[0]

        # Print warning if tokenizer vocab sizes don't match the model
        if src_tokenizer.vocab_size != model_config["src_vocab_size"]:
            print(
                f"WARNING: Source tokenizer vocabulary size ({src_tokenizer.vocab_size}) doesn't match model ({model_config['src_vocab_size']})"
            )

        if tgt_tokenizer.vocab_size != model_config["tgt_vocab_size"]:
            print(
                f"WARNING: Target tokenizer vocabulary size ({tgt_tokenizer.vocab_size}) doesn't match model ({model_config['tgt_vocab_size']})"
            )

    # Create model with the inferred architecture
    model = EncoderDecoderTransformer(
        src_vocab_size=model_config["src_vocab_size"],
        tgt_vocab_size=model_config["tgt_vocab_size"],
        d_model=model_config.get("d_model", 512),
        num_heads=model_config.get("num_heads", 16),
        num_encoder_layers=model_config.get("num_encoder_layers", 6),
        num_decoder_layers=model_config.get("num_decoder_layers", 6),
        d_ff=model_config.get("d_ff", 2048),
        dropout=model_config.get("dropout", 0.2),
        max_seq_length=model_config.get("max_seq_length", 100),
        positional_encoding=model_config.get("positional_encoding", "sinusoidal"),
        share_embeddings=model_config.get("share_embeddings", False),
    )

    # Load model state
    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        # Try to load directly - some checkpoints might store the state dict directly
        model.load_state_dict(checkpoint)

    model.to(device)
    model.eval()

    print(f"Successfully loaded model from {model_path}")
    return model


def preprocess_data(dataset, src_tokenizer, tgt_tokenizer, src_vocab, tgt_vocab):
    """
    Preprocess the dataset for training.

    Args:
        dataset: IWSLT dataset
        src_tokenizer: Source language tokenizer
        tgt_tokenizer: Target language tokenizer
        src_vocab: Source vocabulary
        tgt_vocab: Target vocabulary

    Returns:
        Tokenized source sequences, tokenized target sequences
    """
    src_sequences = []
    tgt_sequences = []

    for src_text, tgt_text in zip(dataset.src_data, dataset.tgt_data):
        # Tokenize
        src_tokens = src_tokenizer(src_text)
        tgt_tokens = tgt_tokenizer(tgt_text)

        # Convert to indices
        src_indices = [src_vocab[token] for token in src_tokens]
        tgt_indices = [tgt_vocab[token] for token in tgt_tokens]

        src_sequences.append(src_indices)
        tgt_sequences.append(tgt_indices)

    return src_sequences, tgt_sequences


def calculate_bleu(hypotheses, references):
    """
    Calculate a simplified BLEU score.

    Args:
        hypotheses: List of generated translations (token lists)
        references: List of reference translations (token lists)

    Returns:
        BLEU score
    """
    # This is a very simplified BLEU implementation
    # In practice, you would use a library like NLTK or sacrebleu

    def count_ngrams(sentence, n):
        """Count n-grams in a sentence."""
        ngrams = {}
        for i in range(len(sentence) - n + 1):
            ngram = tuple(sentence[i : i + n])
            ngrams[ngram] = ngrams.get(ngram, 0) + 1
        return ngrams

    def modified_precision(hypothesis, reference, n):
        """Calculate modified precision for n-grams."""
        hyp_ngrams = count_ngrams(hypothesis, n)
        ref_ngrams = count_ngrams(reference, n)

        if not hyp_ngrams:
            return 0

        # Count matches
        matches = 0
        for ngram, count in hyp_ngrams.items():
            matches += min(count, ref_ngrams.get(ngram, 0))

        return matches / sum(hyp_ngrams.values())

    # Calculate n-gram precisions
    precisions = []
    for n in range(1, 5):
        precision_sum = 0
        for hyp, ref in zip(hypotheses, references):
            precision_sum += modified_precision(hyp, ref, n)

        if len(hypotheses) > 0:
            precisions.append(precision_sum / len(hypotheses))
        else:
            precisions.append(0)

    # Apply weights (equal weights for simplicity)
    weighted_precision = (
        sum(0.25 * p for p in precisions) if all(p > 0 for p in precisions) else 0
    )

    # Calculate brevity penalty
    hyp_length = sum(len(h) for h in hypotheses)
    ref_length = sum(len(r) for r in references)

    if hyp_length < ref_length:
        bp = np.exp(1 - ref_length / hyp_length) if hyp_length > 0 else 0
    else:
        bp = 1

    # Calculate BLEU
    bleu = bp * np.exp(weighted_precision)

    return bleu


def early_stopping(history, patience=10):
    """
    Determine whether to stop training based on validation loss.

    Args:
        history: Dictionary containing training history
        patience: Number of epochs to wait for improvement

    Returns:
        Boolean indicating whether to stop training
    """
    val_losses = history.get("val_loss", [])

    # Not enough history to apply early stopping
    if len(val_losses) < patience:
        return False

    # Get recent validation losses
    recent_losses = val_losses[-patience:]

    # Find the best (lowest) loss in the recent window
    best_loss = min(recent_losses)

    # Stop if no significant improvement for 'patience' epochs
    # Using a 1% improvement threshold to prevent stopping too early
    return recent_losses[-1] > best_loss * 0.99


def preprocess_data_with_bpe(dataset, src_tokenizer, tgt_tokenizer=None):
    """
    Preprocess the dataset for training using BPE tokenizer(s).

    Args:
        dataset: IWSLT dataset or EuroparlDataset
        src_tokenizer: Source language tokenizer
        tgt_tokenizer: Target language tokenizer (optional, if None, uses src_tokenizer as joint tokenizer)

    Returns:
        Lists of tokenized source and target sequences
    """
    src_sequences = []
    tgt_sequences = []

    # If tgt_tokenizer is None, use src_tokenizer as joint tokenizer
    joint_tokenizer = tgt_tokenizer is None
    tgt_tokenizer = tgt_tokenizer or src_tokenizer

    for src_text, tgt_text in zip(dataset.src_data, dataset.tgt_data):
        # Tokenize with appropriate tokenizer(s)
        src_ids = src_tokenizer.encode(src_text)
        tgt_ids = tgt_tokenizer.encode(tgt_text)

        # Add special tokens
        src_ids = (
            [src_tokenizer.special_tokens["bos_token_idx"]]
            + src_ids
            + [src_tokenizer.special_tokens["eos_token_idx"]]
        )
        tgt_ids = (
            [tgt_tokenizer.special_tokens["bos_token_idx"]]
            + tgt_ids
            + [tgt_tokenizer.special_tokens["eos_token_idx"]]
        )

        src_sequences.append(src_ids)
        tgt_sequences.append(tgt_ids)

    return src_sequences, tgt_sequences


def count_parameters(model):
    """Count the number of trainable parameters in a model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def preprocess_input_text(text):
    """
    Standardize input text before tokenization to improve translation quality.

    Args:
        text: Input text to standardize

    Returns:
        Standardized text
    """
    # Normalize whitespace
    text = re.sub(r"\s+", " ", text).strip()

    # Add space around punctuation for better tokenization
    text = re.sub(r"([.,!?;:()])", r" \1 ", text)
    text = re.sub(r"\s+", " ", text).strip()

    # Add space between lowercase and uppercase letters (for German compound words)
    text = re.sub(r"([a-z])([A-Z])", r"\1 \2", text)

    return text


def find_best_model(model_dir, model_prefix):
    """
    Find the best model file to load, with fallbacks.

    Args:
        model_dir: Directory to search for models
        model_prefix: Prefix for the model files

    Returns:
        Path to the best model file to load
    """
    # Try different possible file paths
    candidates = [
        f"{model_dir}/{model_prefix}",  # No extension
        f"{model_dir}/{model_prefix}.pt",  # .pt extension
        f"{model_dir}/{model_prefix}_direct.pt",  # Direct save version
        f"{model_dir}/{model_prefix}_last.pt",  # Last checkpoint
        f"{model_dir}/checkpoints/{model_prefix}/epoch_25.pt",  # Last epoch typically
    ]

    # Find the most recently modified file that exists
    valid_models = []
    for path in candidates:
        if os.path.exists(path):
            valid_models.append((path, os.path.getmtime(path)))

    if not valid_models:
        # Try finding any checkpoint in the checkpoints directory
        checkpoint_dir = f"{model_dir}/checkpoints/{model_prefix}"
        if os.path.exists(checkpoint_dir):
            for filename in os.listdir(checkpoint_dir):
                if filename.startswith("epoch_") and filename.endswith(".pt"):
                    path = os.path.join(checkpoint_dir, filename)
                    valid_models.append((path, os.path.getmtime(path)))

    if valid_models:
        # Sort by modification time (newest first)
        valid_models.sort(key=lambda x: x[1], reverse=True)
        return valid_models[0][0]

    return None


def main(args):
    # Set random seed for reproducibility
    torch.manual_seed(42)
    random.seed(42)
    np.random.seed(42)

    # Set device
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load dataset based on user selection
    print(f"Loading {args.dataset.capitalize()} dataset...")

    if args.dataset == "europarl":
        # Load Europarl dataset
        train_dataset = EuroparlDataset(
            src_lang=args.src_lang,
            tgt_lang=args.tgt_lang,
            max_examples=args.max_train_examples,
        )

        val_dataset = EuroparlDataset(
            src_lang=args.src_lang,
            tgt_lang=args.tgt_lang,
            max_examples=args.max_val_examples,
        )
    elif args.dataset == "opensubtitles":
        # Load OpenSubtitles dataset
        train_dataset = OpenSubtitlesDataset(
            data_dir="data/os",
            src_lang=args.src_lang,
            tgt_lang=args.tgt_lang,
            max_examples=args.max_train_examples,
        )

        val_dataset = OpenSubtitlesDataset(
            data_dir="data/os",
            src_lang=args.src_lang,
            tgt_lang=args.tgt_lang,
            max_examples=args.max_val_examples,
        )
    elif args.dataset == "combined":
        # Load combined dataset
        train_dataset = CombinedDataset(
            src_lang=args.src_lang,
            tgt_lang=args.tgt_lang,
            max_examples=args.max_train_examples,
        )

        val_dataset = CombinedDataset(
            src_lang=args.src_lang,
            tgt_lang=args.tgt_lang,
            max_examples=args.max_val_examples,
        )
    else:
        raise ValueError(f"Unknown dataset: {args.dataset}")

    # Load tokenizer(s)
    print("Loading tokenizer(s)...")
    if args.use_joint_tokenizer:
        # Load joint tokenizer
        src_tokenizer = OptimizedBPETokenizer.from_pretrained(
            args.tokenizer_path or "models/tokenizers/combined/joint"
        )
        tgt_tokenizer = src_tokenizer  # Use same tokenizer for both
        print(f"Loaded joint tokenizer with vocab size: {src_tokenizer.vocab_size}")
    else:
        # Load separate tokenizers
        src_tokenizer = OptimizedBPETokenizer.from_pretrained(
            args.src_tokenizer_path or f"models/tokenizers/{args.src_lang}"
        )
        tgt_tokenizer = OptimizedBPETokenizer.from_pretrained(
            args.tgt_tokenizer_path or f"models/tokenizers/{args.tgt_lang}"
        )
        print(f"Loaded source tokenizer with vocab size: {src_tokenizer.vocab_size}")
        print(f"Loaded target tokenizer with vocab size: {tgt_tokenizer.vocab_size}")

    # Get special token indices for the transformer
    pad_idx = src_tokenizer.special_tokens["pad_token_idx"]
    bos_idx = src_tokenizer.special_tokens["bos_token_idx"]
    eos_idx = src_tokenizer.special_tokens["eos_token_idx"]

    # Preprocess data with appropriate tokenizer(s)
    print("Preprocessing training data...")
    train_src_sequences, train_tgt_sequences = preprocess_data_with_bpe(
        train_dataset, src_tokenizer, tgt_tokenizer
    )

    print("Preprocessing validation data...")
    val_src_sequences, val_tgt_sequences = preprocess_data_with_bpe(
        val_dataset, src_tokenizer, tgt_tokenizer
    )

    # Create data module
    print("Creating data module...")
    data_module = TransformerDataModule(
        source_sequences=train_src_sequences,
        target_sequences=train_tgt_sequences,
        batch_size=args.batch_size,
        max_src_len=100,
        max_tgt_len=100,
        pad_idx=pad_idx,
        bos_idx=bos_idx,
        eos_idx=eos_idx,
        val_split=0.0,
        shuffle=True,
        num_workers=4,
    )

    # Create a separate validation data module
    val_data_module = TransformerDataModule(
        source_sequences=val_src_sequences,
        target_sequences=val_tgt_sequences,
        batch_size=args.batch_size,
        max_src_len=100,
        max_tgt_len=100,
        pad_idx=pad_idx,
        bos_idx=bos_idx,
        eos_idx=eos_idx,
        val_split=0.0,
        shuffle=False,
        num_workers=4,
    )

    # Create transformer model with appropriate vocabulary sizes
    print("Creating transformer model...")
    if args.use_joint_tokenizer:
        # Use same vocabulary size for both encoder and decoder
        vocab_size = src_tokenizer.vocab_size
        model = EncoderDecoderTransformer(
            src_vocab_size=vocab_size,
            tgt_vocab_size=vocab_size,
            d_model=args.d_model,
            num_heads=args.num_heads,
            num_encoder_layers=args.num_encoder_layers,
            num_decoder_layers=args.num_decoder_layers,
            d_ff=args.d_ff,
            dropout=args.dropout,
            max_seq_length=100,
            positional_encoding="sinusoidal",
            share_embeddings=True,  # Enable embedding sharing for joint vocabulary
        )
    else:
        # Use separate vocabulary sizes
        model = EncoderDecoderTransformer(
            src_vocab_size=src_tokenizer.vocab_size,
            tgt_vocab_size=tgt_tokenizer.vocab_size,
            d_model=args.d_model,
            num_heads=args.num_heads,
            num_encoder_layers=args.num_encoder_layers,
            num_decoder_layers=args.num_decoder_layers,
            d_ff=args.d_ff,
            dropout=args.dropout,
            max_seq_length=100,
            positional_encoding="sinusoidal",
            share_embeddings=False,  # Disable embedding sharing for separate vocabularies
        )

    # Print model parameter count
    num_params = count_parameters(model)
    print(f"Model created with {num_params:,} trainable parameters")

    # Create the trainer
    trainer = TransformerTrainer(
        model=model,
        train_dataloader=data_module.get_train_dataloader(),
        val_dataloader=val_data_module.get_val_dataloader(),
        pad_idx=pad_idx,
        lr=args.learning_rate,
        warmup_steps=args.warmup_steps,
        label_smoothing=args.label_smoothing,
        device=device,
        track_perplexity=True,
        use_gradient_scaling=args.use_gradient_scaling,
        early_stopping_patience=args.early_stopping_patience,
    )

    # If loading a pretrained model is requested
    if args.load_model:
        # If the path doesn't exist, try to find a valid model path
        if not os.path.exists(args.load_model):
            dataset_prefix = f"{args.dataset}_{args.src_lang}_{args.tgt_lang}"
            best_model = find_best_model("models", dataset_prefix)
            if best_model:
                args.load_model = best_model
                print(f"Found model: {args.load_model}")
            else:
                print(f"No model found matching {dataset_prefix}")
                if not args.inference_only:
                    print("Will train a new model from scratch")
                    args.load_model = None
                else:
                    raise FileNotFoundError("No model found for inference")

        if args.load_model and os.path.exists(args.load_model):
            print(f"Loading pretrained model from {args.load_model}")
            print(f"Using absolute path: {os.path.abspath(args.load_model)}")
            try:
                # Load the model
                model = load_pretrained_model(
                    args.load_model, src_tokenizer, tgt_tokenizer, device
                )
                print("Loaded pretrained model successfully")
            except Exception as e:
                print(f"Error loading model: {e}")
                # Continue with the new model if loading fails
                pass
        else:
            print("Training new model from scratch")
    else:
        print("Training new model from scratch")

    # Add debugging - this will automatically attach to the trainer
    if args.debug:
        print("Enabling debugging features...")
        # Note: No need to explicitly create a debugger here as it's handled
        # in the main function via attach_debugger_to_trainer

    # Create save directory
    os.makedirs("models", exist_ok=True)

    # Prefix for model file based on dataset and languages
    model_prefix = f"{args.dataset}_{args.src_lang}_{args.tgt_lang}"

    # Define the save path
    save_path = f"models/{model_prefix}_translation"
    print(f"\nModel will be saved to: {os.path.abspath(save_path)}\n")

    # Define a checkpoint directory for saving models during training
    checkpoint_dir = f"models/checkpoints/{model_prefix}"
    os.makedirs(checkpoint_dir, exist_ok=True)

    # Train model if not just doing inference
    if not args.inference_only:
        print("Training model...")

        # Define a simple callback function to save checkpoints at the end of each epoch
        def save_epoch_checkpoint(
            epoch: int, model: nn.Module, trainer: "TransformerTrainer"
        ) -> None:
            epoch_save_path = f"{checkpoint_dir}/epoch_{epoch+1}.pt"
            try:
                checkpoint = {
                    "model_state_dict": model.state_dict(),
                    "epoch": epoch + 1,
                    "model_config": {
                        "src_vocab_size": model.encoder.token_embedding.embedding.weight.shape[
                            0
                        ],
                        "tgt_vocab_size": model.decoder.token_embedding.embedding.weight.shape[
                            0
                        ],
                        "d_model": model.encoder.token_embedding.embedding.weight.shape[
                            1
                        ],
                        "num_encoder_layers": len(model.encoder.layers),
                        "num_decoder_layers": len(model.decoder.layers),
                    },
                }
                torch.save(checkpoint, epoch_save_path)
                print(f"Saved epoch {epoch+1} checkpoint to {epoch_save_path}")
            except Exception as e:
                print(f"Error saving epoch checkpoint: {e}")

        # Set a checkpoint callback in trainer - ignore the linter error about type incompatibility
        # The TransformerTrainer class accepts a callable for this attribute
        # mypy/pylance can't determine this from the code
        # @type-ignore
        trainer.epoch_end_callback = save_epoch_checkpoint

        trainer_history = trainer.train(epochs=args.epochs, save_path=save_path)

        # Check vocabulary sizes after training
        src_vocab_size = model.encoder.token_embedding.embedding.weight.shape[0]
        tgt_vocab_size = model.decoder.token_embedding.embedding.weight.shape[0]
        print(
            f"Model vocabulary sizes after training: source={src_vocab_size}, target={tgt_vocab_size}"
        )

        if (
            src_vocab_size != src_tokenizer.vocab_size
            or tgt_vocab_size != tgt_tokenizer.vocab_size
        ):
            print(
                f"WARNING: Final model vocabulary size doesn't match tokenizer vocabulary size!"
            )
            print(
                f"Source: model={src_vocab_size}, tokenizer={src_tokenizer.vocab_size}"
            )
            print(
                f"Target: model={tgt_vocab_size}, tokenizer={tgt_tokenizer.vocab_size}"
            )

        # Ensure model is saved directly from script (belt and suspenders approach)
        try:
            print(f"\nDirectly saving model from script to: {save_path}")
            # Create a checkpoint dictionary with essential information
            model_checkpoint = {
                "model_state_dict": model.state_dict(),
                "model_config": {
                    "src_vocab_size": src_vocab_size,
                    "tgt_vocab_size": tgt_vocab_size,
                    "d_model": model.encoder.token_embedding.embedding.weight.shape[1],
                    "num_encoder_layers": len(model.encoder.layers),
                    "num_decoder_layers": len(model.decoder.layers),
                    "dropout": getattr(model, "dropout", 0.1),
                },
            }

            # Add a .pt extension for this direct save to distinguish it
            direct_save_path = f"{save_path}_direct.pt"
            torch.save(model_checkpoint, direct_save_path)

            if os.path.exists(direct_save_path):
                file_size = os.path.getsize(direct_save_path) / (
                    1024 * 1024
                )  # Size in MB
                print(f"Direct save successful - File size: {file_size:.2f} MB")
            else:
                print("WARNING: Direct save failed - file was not created")
        except Exception as e:
            print(f"ERROR during direct model save: {e}")

        # Plot training history
        trainer.plot_training_history()
        plt.savefig(f"{model_prefix}_training_history.png")
        plt.close()

        # Plot learning rate schedule
        trainer.plot_learning_rate()
        plt.savefig(f"{model_prefix}_learning_rate_schedule.png")
        plt.close()
    else:
        print("Skipping training, inference only mode")
        trainer_history = {}

    # Generate translation examples
    print("\n===== Translation Examples =====")

    # Sample sentences for translation
    sample_sentences = [
        preprocess_input_text("Ich bin ein Student."),
        preprocess_input_text("Wo ist die Bibliothek?"),
        preprocess_input_text("Künstliche Intelligenz wird unser Leben verändern."),
        preprocess_input_text("Das Buch ist sehr interessant."),
        preprocess_input_text("Ich möchte Deutsch lernen."),
    ]

    # Generate translations with better defaults for cleaner results
    translations, raw_translations = generate_translations(
        model=model,
        tokenizer=src_tokenizer,
        src_sentences=sample_sentences,
        device=device,
        max_length=50,  # Shorter translations are typically better
        sampling_temp=1.0,  # Greedy decoding
        sampling_topk=1,  # Greedy decoding (top-1)
    )

    # Print translations
    print("\nTranslation Results:")
    for i, (src, raw, cleaned) in enumerate(
        zip(sample_sentences, raw_translations, translations)
    ):
        print(f"Source {i+1}: {src}")
        print(f"Raw: {raw}")
        print(f"Cleaned: {cleaned}")
        print()

    return trainer_history


if __name__ == "__main__":
    # Parse arguments
    parser = argparse.ArgumentParser(
        description="Train a transformer model for machine translation with joint vocabulary"
    )

    # Dataset options
    parser.add_argument(
        "--dataset",
        type=str,
        choices=["europarl", "opensubtitles", "combined"],
        default="europarl",
        help="Dataset to use for training",
    )
    parser.add_argument(
        "--max_train_examples",
        type=int,
        default=100000,
        help="Maximum number of training examples to use",
    )
    parser.add_argument(
        "--max_val_examples",
        type=int,
        default=20000,
        help="Maximum number of validation examples to use",
    )
    parser.add_argument(
        "--src_lang", type=str, default="de", help="Source language code"
    )
    parser.add_argument(
        "--tgt_lang", type=str, default="en", help="Target language code"
    )
    parser.add_argument(
        "--tokenizer_path",
        type=str,
        default=None,
        help="Path to joint tokenizer (default: models/tokenizers/combined/joint)",
    )

    # Model options
    parser.add_argument(
        "--d_model", type=int, default=512, help="Dimension of model embeddings"
    )
    parser.add_argument(
        "--num_heads", type=int, default=8, help="Number of attention heads"
    )
    parser.add_argument(
        "--num_encoder_layers", type=int, default=4, help="Number of encoder layers"
    )
    parser.add_argument(
        "--num_decoder_layers", type=int, default=4, help="Number of decoder layers"
    )
    parser.add_argument(
        "--d_ff", type=int, default=2048, help="Dimension of feed-forward network"
    )
    parser.add_argument("--dropout", type=float, default=0.1, help="Dropout rate")

    # Training options
    parser.add_argument(
        "--batch_size", type=int, default=128, help="Batch size for training"
    )
    parser.add_argument(
        "--learning_rate", type=float, default=0.0001, help="Initial learning rate"
    )
    parser.add_argument(
        "--label_smoothing",
        type=float,
        default=0.2,
        help="Label smoothing factor for training",
    )
    parser.add_argument(
        "--warmup_steps",
        type=int,
        default=4000,
        help="Number of warmup steps for learning rate scheduler",
    )
    parser.add_argument(
        "--early_stopping_patience",
        type=int,
        default=None,
        help="Number of epochs to wait before early stopping (None to disable)",
    )
    parser.add_argument(
        "--use_gradient_scaling",
        action="store_true",
        help="Whether to use gradient scaling for mixed precision training",
    )
    parser.add_argument(
        "--use_mixed_precision",
        action="store_true",
        help="Whether to use mixed precision training",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug mode with more verbose output",
    )
    parser.add_argument(
        "--debug_frequency",
        type=int,
        default=100,
        help="How often to print debug information (in training steps)",
    )
    parser.add_argument(
        "--debug_sample_batch",
        action="store_true",
        help="Debug a sample batch before training",
    )
    parser.add_argument(
        "--epochs", type=int, default=3, help="Number of epochs to train for"
    )

    # Model loading and inference options
    parser.add_argument(
        "--load_model",
        type=str,
        default=None,
        help="Path to a pretrained model checkpoint to load",
    )
    parser.add_argument(
        "--inference_only",
        action="store_true",
        help="Run only inference (no training)",
    )

    # New options for separate tokenizers
    parser.add_argument(
        "--src_tokenizer_path",
        type=str,
        default=None,
        help="Path to source language tokenizer",
    )
    parser.add_argument(
        "--tgt_tokenizer_path",
        type=str,
        default=None,
        help="Path to target language tokenizer",
    )
    parser.add_argument(
        "--use_joint_tokenizer",
        type=lambda x: x.lower() == "true",
        default=True,
        help="Use a joint tokenizer instead of separate ones (true/false)",
    )

    args = parser.parse_args()

    # Set random seed for reproducibility
    torch.manual_seed(42)
    random.seed(42)
    np.random.seed(42)

    # Check for mixed precision conflicts
    if args.use_mixed_precision and args.use_gradient_scaling:
        print("Warning: Using both mixed precision and gradient scaling.")
        print("Gradient scaling is typically used alongside mixed precision.")

    # Ensure gradient scaling is enabled with mixed precision
    if args.use_mixed_precision and not args.use_gradient_scaling:
        print("Enabling gradient scaling since mixed precision is enabled.")
        args.use_gradient_scaling = True

    # Set up debugging
    debugger = None
    if args.debug:
        print("Enabling debugging features...")
        # Note: No need to explicitly create a debugger here as it's handled
        # in the main function via attach_debugger_to_trainer

    # Run the training process
    main(args)

    # Note: The visualization code below won't run correctly unless trainer_history is defined
    # in the outer scope. The plotting is now handled within the main function.
    # The create_translation_debugger reference was also removed as it's not defined.
