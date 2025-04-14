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
import concurrent.futures
from functools import partial

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
from src.data.wmt_dataset import WMTDataset  # Add import for WMT dataset
from debug_scripts.debug_transformer import (
    attach_debugger_to_trainer,
    debug_sample_batch,
)
from src.data.combined_dataset import CombinedDataset
from src.data.curriculum_dataset import CurriculumTranslationDataset


def clean_translation_output(text):
    """
    Minimal cleaning of translation output - preserves case and punctuation.

    Args:
        text: Raw translation text

    Returns:
        Cleaned translation text
    """
    # Remove BOS token if present
    text = text.replace("<bos>", "").strip()

    # Fix common special tokens
    text = text.replace("_space_", " ")  # Fix underscore space format from tokenizer
    text = text.replace("_space", " ")  # Also handle alternative format
    text = text.replace("_dash_", "-")
    text = text.replace("_dash", "-")
    text = text.replace("_comma_", ",")
    text = text.replace("_comma", ",")
    text = text.replace("_period_", ".")
    text = text.replace("_period", ".")
    text = text.replace("_question_", "?")
    text = text.replace("_question", "?")
    text = text.replace("_exclamation_", "!")
    text = text.replace("_exclamation", "!")

    # Fix double spaces
    text = re.sub(r"\s+", " ", text).strip()

    # Don't capitalize the first letter automatically
    # Let the model's output preserve the case as-is

    return text


def generate_translations(
    model,
    src_tokenizer,  # Now explicitly naming as source tokenizer
    src_sentences,  # Required parameter
    tgt_tokenizer=None,  # Add target tokenizer as separate parameter
    max_length=50,
    device=None,
    fallback_token="[UNK]",
    sampling_temp=0.2,
    sampling_topk=10,
):
    """
    Generate translations for a list of source sentences using sampling with temperature.

    Args:
        model: Trained transformer model
        src_tokenizer: Source language tokenizer
        src_sentences: List of source language sentences
        tgt_tokenizer: Target language tokenizer (if None, uses src_tokenizer)
        max_length: Maximum length of generated translations
        device: Device to run inference on
        fallback_token: Token to use when encountering out-of-vocabulary tokens
        sampling_temp: Temperature for sampling (lower = more deterministic)
        sampling_topk: Number of top tokens to consider for sampling

    Returns:
        List of translated sentences
    """
    model.eval()
    translations = []
    raw_translations = []

    # Use source tokenizer for target if no target tokenizer provided
    if tgt_tokenizer is None:
        print(
            "WARNING: No target tokenizer provided. Using source tokenizer for decoding."
        )
        tgt_tokenizer = src_tokenizer

    # Default to CPU if device not provided
    if device is None:
        device = torch.device("cpu")
    elif isinstance(device, str):
        device = torch.device(device)

    # Get vocabulary size
    vocab_size = model.encoder.token_embedding.embedding.weight.shape[0]
    tgt_vocab_size = model.decoder.token_embedding.embedding.weight.shape[0]

    # Check for vocab size mismatch and print warning
    if src_tokenizer.vocab_size != vocab_size:
        print(
            f"WARNING: Source tokenizer vocabulary size ({src_tokenizer.vocab_size}) doesn't match model encoder ({vocab_size})"
        )

    if tgt_tokenizer.vocab_size != tgt_vocab_size:
        print(
            f"WARNING: Target tokenizer vocabulary size ({tgt_tokenizer.vocab_size}) doesn't match model decoder ({tgt_vocab_size})"
        )

    # Get token indices for special tokens
    try:
        src_unk_token_idx = src_tokenizer.token_to_id(fallback_token)
        if src_unk_token_idx is None or src_unk_token_idx >= vocab_size:
            src_unk_token_idx = 0  # Use a safe default
    except:
        src_unk_token_idx = 0

    # Get special token indices for cleaner code
    src_bos_idx = src_tokenizer.special_tokens["bos_token_idx"]
    src_eos_idx = src_tokenizer.special_tokens["eos_token_idx"]
    src_pad_idx = src_tokenizer.special_tokens["pad_token_idx"]

    tgt_bos_idx = tgt_tokenizer.special_tokens["bos_token_idx"]
    tgt_eos_idx = tgt_tokenizer.special_tokens["eos_token_idx"]
    tgt_pad_idx = tgt_tokenizer.special_tokens["pad_token_idx"]

    print(
        "Source special token indices:",
        {"bos": src_bos_idx, "eos": src_eos_idx, "pad": src_pad_idx},
    )

    print(
        "Target special token indices:",
        {"bos": tgt_bos_idx, "eos": tgt_eos_idx, "pad": tgt_pad_idx},
    )

    with torch.no_grad():
        for src_idx, src_text in enumerate(src_sentences):
            try:
                print(f"\nTranslating sentence {src_idx+1}: '{src_text}'")

                # Tokenize source text with SOURCE tokenizer
                src_ids = src_tokenizer.encode(src_text)
                print(f"Source tokens: {src_ids}")

                # Add special tokens
                src_ids = [src_bos_idx] + src_ids + [src_eos_idx]
                print(f"Source tokens with special tokens: {src_ids}")

                # Ensure all tokens are in vocabulary range
                src_ids = [min(token_id, vocab_size - 1) for token_id in src_ids]

                # Convert to tensor and move to device
                src_tensor = torch.tensor([src_ids], dtype=torch.long).to(device)

                # Create source mask
                src_mask = create_padding_mask(src_tensor, src_pad_idx)

                # Initialize target with TARGET BOS token
                tgt_bos_idx = min(tgt_bos_idx, tgt_vocab_size - 1)
                tgt_ids = [tgt_bos_idx]
                tgt_tensor = torch.tensor([tgt_ids], dtype=torch.long).to(device)

                # Generate translation auto-regressively
                for _ in range(max_length):
                    # Create target mask (causal)
                    tgt_mask = create_causal_mask(tgt_tensor.size(1), device)

                    # Get model prediction
                    logits = model(
                        src_tensor,
                        tgt_tensor,
                        src_mask=src_mask,
                        tgt_mask=tgt_mask,
                    )
                    next_token_logits = logits[0, -1]

                    # Apply temperature (lower = more deterministic)
                    next_token_logits = next_token_logits / sampling_temp

                    # Get top-k tokens
                    top_k = min(sampling_topk, next_token_logits.size(-1))
                    top_token_logits, top_tokens = torch.topk(next_token_logits, top_k)

                    # Convert to probabilities
                    top_token_probs = torch.softmax(top_token_logits, dim=-1)

                    # Sample or use greedy decoding based on settings
                    if sampling_temp < 0.01 or top_k == 1:  # Effectively greedy
                        next_token = top_tokens[0].item()
                    else:
                        # Sample from top-k based on probabilities
                        try:
                            # Ensure valid probabilities
                            if (
                                torch.isnan(top_token_probs).any()
                                or not torch.isfinite(top_token_probs).all()
                            ):
                                top_token_probs = (
                                    torch.ones_like(top_token_probs) / top_k
                                )

                            # Normalize to sum to 1.0
                            top_token_probs = top_token_probs / top_token_probs.sum()

                            # Sample from distribution
                            index_tensor = torch.multinomial(
                                top_token_probs, num_samples=1
                            )
                            next_token_idx = index_tensor.item()
                            next_token = top_tokens[int(next_token_idx)].item()

                            # Print top tokens for debugging (first sentence only)
                            if src_idx == 0 and len(tgt_ids) < 5:
                                print(
                                    f"Step {len(tgt_ids)}: Top 5 tokens and probabilities:"
                                )
                                for i in range(min(5, len(top_tokens))):
                                    token_id = top_tokens[i].item()
                                    token_str = tgt_tokenizer.decode([token_id])
                                    prob = top_token_probs[i].item()
                                    print(
                                        f"  {i+1}. ID: {token_id}, Token: '{token_str}', Prob: {prob:.4f}"
                                    )
                                print(
                                    f"Selected token: ID: {next_token}, Token: '{tgt_tokenizer.decode([next_token])}'"
                                )

                        except Exception as e:
                            print(
                                f"Sampling error: {e}, falling back to greedy decoding"
                            )
                            next_token = top_tokens[0].item()

                    # Add predicted token to target sequence
                    tgt_ids.append(next_token)
                    tgt_tensor = torch.tensor([tgt_ids], dtype=torch.long).to(device)

                    # Stop if TARGET EOS token
                    if next_token == tgt_eos_idx:
                        break

                # Process output tokens
                output_tokens = []
                for token_id in tgt_ids:
                    # Skip BOS token
                    if token_id == tgt_bos_idx:
                        continue
                    # Stop at EOS token
                    if token_id == tgt_eos_idx:
                        break
                    # Skip tokens outside vocabulary range
                    if token_id >= tgt_vocab_size:
                        continue
                    output_tokens.append(token_id)

                # Debugging info for first sentence
                if src_idx == 0:
                    print(f"Output tokens: {output_tokens}")
                    print(
                        f"Output token strings: {[tgt_tokenizer.decode([t]) for t in output_tokens]}"
                    )

                # Decode tokens to text using TARGET tokenizer
                raw_translated_text = tgt_tokenizer.decode(output_tokens)
                cleaned_text = clean_translation_output(raw_translated_text)

                print(f"Raw translation: '{raw_translated_text}'")
                print(f"Cleaned translation: '{cleaned_text}'")

                raw_translations.append(raw_translated_text)
                translations.append(cleaned_text)

            except Exception as e:
                print(f"Error during translation: {e}")
                import traceback

                traceback.print_exc()
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
    print(f"Loading model from {model_path}...")
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
    load_succeeded = False
    if "model_state_dict" in checkpoint:
        try:
            state_dict = checkpoint["model_state_dict"]

            # Fix output projection key mismatch if needed
            if (
                "output_projection.weight" in state_dict
                and "decoder.output_projection.weight" not in state_dict
            ):
                state_dict["decoder.output_projection.weight"] = state_dict.pop(
                    "output_projection.weight"
                )
                state_dict["decoder.output_projection.bias"] = state_dict.pop(
                    "output_projection.bias"
                )
                print("Fixed output projection key mismatch in state dict")

            # Strict loading - will error if keys don't match
            model.load_state_dict(state_dict, strict=True)
            print("Loaded model state with strict=True")
            load_succeeded = True
        except Exception as e:
            print(f"Error during strict loading: {e}")
            print("Trying non-strict loading...")
            try:
                model.load_state_dict(state_dict, strict=False)
                print("Loaded model state with strict=False")
                load_succeeded = True
            except Exception as e2:
                print(f"Error during non-strict loading: {e2}")
    else:
        # Try to load directly - some checkpoints might store the state dict directly
        try:
            # Fix output projection key mismatch if needed
            if (
                "output_projection.weight" in checkpoint
                and "decoder.output_projection.weight" not in checkpoint
            ):
                checkpoint["decoder.output_projection.weight"] = checkpoint.pop(
                    "output_projection.weight"
                )
                checkpoint["decoder.output_projection.bias"] = checkpoint.pop(
                    "output_projection.bias"
                )
                print("Fixed output projection key mismatch in state dict")

            model.load_state_dict(checkpoint, strict=True)
            print("Loaded model state directly from checkpoint with strict=True")
            load_succeeded = True
        except Exception as e:
            print(f"Error during direct loading: {e}")
            try:
                model.load_state_dict(checkpoint, strict=False)
                print("Loaded model state directly from checkpoint with strict=False")
                load_succeeded = True
            except Exception as e2:
                print(f"Error during direct non-strict loading: {e2}")

    if not load_succeeded:
        print("WARNING: Failed to load model weights properly!")
        print(
            "Keys in state dict:",
            (
                list(state_dict.keys())
                if "state_dict" in locals()
                else "No state dict available"
            ),
        )
        print("Keys in model:", [k for k, _ in model.named_parameters()])
        raise RuntimeError("Failed to load pretrained model properly")

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


# Define process_batch as a top-level function so it can be pickled for multiprocessing
def _process_batch_for_tokenization(batch_data, src_tokenizer_data, tgt_tokenizer_data):
    """Process a batch of text data for tokenization (defined at module level for pickling)"""
    import torch
    from src.data.tokenization import OptimizedBPETokenizer

    # Recreate the tokenizers on CPU to avoid pickling issues with device-specific objects
    src_tokenizer = OptimizedBPETokenizer.from_pretrained(src_tokenizer_data["path"])
    if src_tokenizer_data["path"] != tgt_tokenizer_data["path"]:
        tgt_tokenizer = OptimizedBPETokenizer.from_pretrained(
            tgt_tokenizer_data["path"]
        )
    else:
        tgt_tokenizer = src_tokenizer

    batch_src_sequences = []
    batch_tgt_sequences = []

    # Get special token indices
    src_bos_idx = src_tokenizer.special_tokens["bos_token_idx"]
    src_eos_idx = src_tokenizer.special_tokens["eos_token_idx"]
    tgt_bos_idx = tgt_tokenizer.special_tokens["bos_token_idx"]
    tgt_eos_idx = tgt_tokenizer.special_tokens["eos_token_idx"]

    for src_text, tgt_text in batch_data:
        # Tokenize with appropriate tokenizer(s)
        src_ids = src_tokenizer.encode(src_text)
        tgt_ids = tgt_tokenizer.encode(tgt_text)

        # Add special tokens
        src_ids = [src_bos_idx] + src_ids + [src_eos_idx]
        tgt_ids = [tgt_bos_idx] + tgt_ids + [tgt_eos_idx]

        batch_src_sequences.append(src_ids)
        batch_tgt_sequences.append(tgt_ids)

    return batch_src_sequences, batch_tgt_sequences


def preprocess_data_with_bpe(
    dataset, src_tokenizer, tgt_tokenizer=None, batch_size=64, num_workers=4
):
    """
    Preprocess the dataset for training using BPE tokenizer(s) with optimization.

    Args:
        dataset: IWSLT dataset or EuroparlDataset
        src_tokenizer: Source language tokenizer
        tgt_tokenizer: Target language tokenizer (optional, if None, uses src_tokenizer as joint tokenizer)
        batch_size: Size of batches for processing
        num_workers: Number of parallel workers for multiprocessing

    Returns:
        Lists of tokenized source and target sequences
    """
    src_sequences = []
    tgt_sequences = []

    # If tgt_tokenizer is None, use src_tokenizer as joint tokenizer
    joint_tokenizer = tgt_tokenizer is None
    tgt_tokenizer = tgt_tokenizer or src_tokenizer

    # Create batches
    data_pairs = list(zip(dataset.src_data, dataset.tgt_data))
    total_examples = len(data_pairs)

    # Process in batches with a progress bar
    with tqdm(total=total_examples, desc="Preprocessing data") as pbar:
        # If multiprocessing is enabled and we have at least 2 workers
        if num_workers > 1:
            try:
                # Create batches for processing
                batches = [
                    data_pairs[i : i + batch_size]
                    for i in range(0, len(data_pairs), batch_size)
                ]

                # Create serializable representations of tokenizers (just the paths)
                src_tokenizer_data = {
                    "path": (
                        src_tokenizer.save_dir
                        if hasattr(src_tokenizer, "save_dir")
                        else None
                    )
                }
                tgt_tokenizer_data = {
                    "path": (
                        tgt_tokenizer.save_dir
                        if hasattr(tgt_tokenizer, "save_dir")
                        else None
                    )
                }

                # If save_dir is not available, we need to create temporary directories to save the tokenizers
                import tempfile
                import os
                import shutil
                import multiprocessing as mp
                from concurrent.futures import ProcessPoolExecutor, as_completed

                temp_dirs = []
                if src_tokenizer_data["path"] is None:
                    temp_dir = tempfile.mkdtemp()
                    temp_dirs.append(temp_dir)
                    temp_path = os.path.join(temp_dir, "src_tokenizer")
                    os.makedirs(temp_path, exist_ok=True)
                    src_tokenizer.save_pretrained(temp_path)
                    src_tokenizer_data["path"] = temp_path

                if tgt_tokenizer_data["path"] is None and not joint_tokenizer:
                    temp_dir = tempfile.mkdtemp()
                    temp_dirs.append(temp_dir)
                    temp_path = os.path.join(temp_dir, "tgt_tokenizer")
                    os.makedirs(temp_path, exist_ok=True)
                    tgt_tokenizer.save_pretrained(temp_path)
                    tgt_tokenizer_data["path"] = temp_path

                # Set up process pool with proper resource management
                ctx = mp.get_context("spawn")  # Use spawn context for better stability
                max_workers = min(num_workers, mp.cpu_count() - 1)  # Leave one CPU free

                with ProcessPoolExecutor(
                    max_workers=max_workers, mp_context=ctx
                ) as executor:
                    # Submit all batches
                    future_to_batch = {
                        executor.submit(
                            _process_batch_for_tokenization,
                            batch,
                            src_tokenizer_data,
                            tgt_tokenizer_data,
                        ): i
                        for i, batch in enumerate(batches)
                    }

                    # Process completed batches as they finish
                    for future in as_completed(future_to_batch):
                        try:
                            batch_src, batch_tgt = future.result()
                            src_sequences.extend(batch_src)
                            tgt_sequences.extend(batch_tgt)
                            pbar.update(len(batch_src))
                        except Exception as e:
                            print(
                                f"Error processing batch {future_to_batch[future]}: {e}"
                            )
                            # Continue with remaining batches
                            continue

                # Clean up temporary directories
                for temp_dir in temp_dirs:
                    try:
                        shutil.rmtree(temp_dir)
                    except Exception as e:
                        print(
                            f"Warning: Could not remove temporary directory {temp_dir}: {e}"
                        )

            except Exception as e:
                print(
                    f"Multiprocessing error: {e}. Falling back to sequential processing."
                )
                # Fall back to sequential processing
                num_workers = 1

        # Sequential processing with batches
        if num_workers <= 1:
            for i in range(0, len(data_pairs), batch_size):
                batch = data_pairs[i : i + batch_size]

                # Process directly without creating new tokenizer instances
                batch_src_sequences = []
                batch_tgt_sequences = []

                # Get special token indices
                src_bos_idx = src_tokenizer.special_tokens["bos_token_idx"]
                src_eos_idx = src_tokenizer.special_tokens["eos_token_idx"]
                tgt_bos_idx = tgt_tokenizer.special_tokens["bos_token_idx"]
                tgt_eos_idx = tgt_tokenizer.special_tokens["eos_token_idx"]

                for src_text, tgt_text in batch:
                    # Tokenize with appropriate tokenizer(s)
                    src_ids = src_tokenizer.encode(src_text)
                    tgt_ids = tgt_tokenizer.encode(tgt_text)

                    # Add special tokens
                    src_ids = [src_bos_idx] + src_ids + [src_eos_idx]
                    tgt_ids = [tgt_bos_idx] + tgt_ids + [tgt_eos_idx]

                    batch_src_sequences.append(src_ids)
                    batch_tgt_sequences.append(tgt_ids)

                src_sequences.extend(batch_src_sequences)
                tgt_sequences.extend(batch_tgt_sequences)
                pbar.update(len(batch_src_sequences))

    return src_sequences, tgt_sequences


def count_parameters(model):
    """Count the number of trainable parameters in a model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def preprocess_input_text(text):
    """
    Standardize input text before tokenization to improve translation quality.
    Makes minimal changes to preserve case and punctuation.

    Args:
        text: Input text to standardize

    Returns:
        Standardized text
    """
    # Normalize whitespace
    text = re.sub(r"\s+", " ", text).strip()

    # Don't add spaces around punctuation when using a tokenizer that preserves punctuation
    # The tokenizer will handle punctuation appropriately based on its settings

    # Don't force lowercase, as our tokenizer now preserves case

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

    # Set device for training (not preprocessing)
    training_device = torch.device(
        "mps" if torch.backends.mps.is_available() else "cpu"
    )
    # Always use CPU for preprocessing
    preprocessing_device = torch.device("cpu")

    print(f"Using device for training: {training_device}")
    print(f"Using device for preprocessing: {preprocessing_device}")

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
    elif args.dataset == "iwslt":
        # Load IWSLT dataset
        train_dataset = IWSLTDataset(
            src_lang=args.src_lang,
            tgt_lang=args.tgt_lang,
            year=args.year if hasattr(args, "year") else "2017",
            split="train",
            max_examples=args.max_train_examples,
            data_dir="data/iwslt",
        )

        val_dataset = IWSLTDataset(
            src_lang=args.src_lang,
            tgt_lang=args.tgt_lang,
            year=args.year if hasattr(args, "year") else "2017",
            split="validation",  # Use validation split for evaluation
            max_examples=args.max_val_examples,
            data_dir="data/iwslt",
        )
    elif args.dataset == "wmt":
        # Load WMT dataset
        train_dataset = WMTDataset(
            src_lang=args.src_lang,
            tgt_lang=args.tgt_lang,
            year=args.wmt_year if hasattr(args, "wmt_year") else "14",
            split="train",
            max_examples=args.max_train_examples,
            data_dir="data/wmt",
            subset=args.subset if hasattr(args, "subset") else None,
        )

        val_dataset = WMTDataset(
            src_lang=args.src_lang,
            tgt_lang=args.tgt_lang,
            year=args.wmt_year if hasattr(args, "wmt_year") else "14",
            split="validation",  # Use validation split for evaluation
            max_examples=args.max_val_examples,
            data_dir="data/wmt",
            subset=args.subset if hasattr(args, "subset") else None,
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

    # Load tokenizer(s) - forcing CPU for preprocessing
    print("Loading tokenizer(s)...")
    if args.use_joint_tokenizer:
        # Load joint tokenizer
        if args.dataset == "wmt":
            tokenizer_path = (
                args.tokenizer_path
                or f"models/tokenizers/wmt{args.wmt_year if hasattr(args, 'wmt_year') else '14'}/joint"
            )
        else:
            tokenizer_path = args.tokenizer_path or "models/tokenizers/combined/joint"

        # Ensure we're using CPU for preprocessing
        src_tokenizer = OptimizedBPETokenizer.from_pretrained(tokenizer_path)
        src_tokenizer.device = preprocessing_device  # Force CPU
        # Store path as a custom attribute for multiprocessing
        setattr(src_tokenizer, "save_dir", tokenizer_path)

        tgt_tokenizer = src_tokenizer  # Use same tokenizer for both
        print(f"Loaded joint tokenizer with vocab size: {src_tokenizer.vocab_size}")
    else:
        # Load separate tokenizers
        if args.dataset == "wmt":
            src_tokenizer_path = (
                args.src_tokenizer_path
                or f"models/tokenizers/wmt{args.wmt_year if hasattr(args, 'wmt_year') else '14'}/{args.src_lang}"
            )
            tgt_tokenizer_path = (
                args.tgt_tokenizer_path
                or f"models/tokenizers/wmt{args.wmt_year if hasattr(args, 'wmt_year') else '14'}/{args.tgt_lang}"
            )
        else:
            src_tokenizer_path = (
                args.src_tokenizer_path or f"models/tokenizers/{args.src_lang}"
            )
            tgt_tokenizer_path = (
                args.tgt_tokenizer_path or f"models/tokenizers/{args.tgt_lang}"
            )

        # Ensure we're using CPU for preprocessing
        src_tokenizer = OptimizedBPETokenizer.from_pretrained(src_tokenizer_path)
        src_tokenizer.device = preprocessing_device  # Force CPU
        # Store path as a custom attribute for multiprocessing
        setattr(src_tokenizer, "save_dir", src_tokenizer_path)

        tgt_tokenizer = OptimizedBPETokenizer.from_pretrained(tgt_tokenizer_path)
        tgt_tokenizer.device = preprocessing_device  # Force CPU
        # Store path as a custom attribute for multiprocessing
        setattr(tgt_tokenizer, "save_dir", tgt_tokenizer_path)

        print(f"Loaded source tokenizer with vocab size: {src_tokenizer.vocab_size}")
        print(f"Loaded target tokenizer with vocab size: {tgt_tokenizer.vocab_size}")

    # Get special token indices for the transformer
    pad_idx = src_tokenizer.special_tokens["pad_token_idx"]
    bos_idx = src_tokenizer.special_tokens["bos_token_idx"]
    eos_idx = src_tokenizer.special_tokens["eos_token_idx"]

    # Preprocess data with appropriate tokenizer(s)
    print("Preprocessing training data...")
    train_src_sequences, train_tgt_sequences = preprocess_data_with_bpe(
        train_dataset,
        src_tokenizer,
        tgt_tokenizer,
        batch_size=64,
        num_workers=(
            args.preprocessing_workers if hasattr(args, "preprocessing_workers") else 4
        ),
    )

    print("Preprocessing validation data...")
    val_src_sequences, val_tgt_sequences = preprocess_data_with_bpe(
        val_dataset,
        src_tokenizer,
        tgt_tokenizer,
        batch_size=64,
        num_workers=(
            args.preprocessing_workers if hasattr(args, "preprocessing_workers") else 4
        ),
    )

    # Create data module
    print("\nCreating data module...")
    if args.use_curriculum:
        print(f"Initializing curriculum learning with:")
        print(f"  Strategy: {args.curriculum_strategy}")
        print(f"  Stages: {args.curriculum_stages}")

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
        use_curriculum=args.use_curriculum,
        curriculum_strategy=args.curriculum_strategy,
        curriculum_stages=args.curriculum_stages,
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
        val_split=1.0,  # Use all data for validation
        shuffle=False,
        num_workers=4,
    )

    # Determine model loading path
    model_path = None
    if args.load_model:
        # If the path doesn't exist, try to find a valid model path
        if not os.path.exists(args.load_model):
            dataset_prefix = f"{args.dataset}_{args.src_lang}_{args.tgt_lang}"
            best_model = find_best_model("models", dataset_prefix)
            if best_model:
                model_path = best_model
                print(f"Found model: {model_path}")
            else:
                print(f"No model found matching {dataset_prefix}")
                if not args.inference_only:
                    print("Will train a new model from scratch")
                    model_path = None
                else:
                    raise FileNotFoundError("No model found for inference")
        else:
            model_path = args.load_model
            print(f"Using specified model path: {model_path}")

    # Load model configuration from checkpoint if available
    model_config = None
    if model_path:
        try:
            checkpoint = torch.load(
                model_path, map_location=preprocessing_device
            )  # Load to CPU first
            if "model_config" in checkpoint:
                model_config = checkpoint["model_config"]
                print(f"Loaded model configuration from checkpoint: {model_config}")
            elif "model_state_dict" in checkpoint:
                # Try to infer configuration from state dict
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
                d_model = state_dict["encoder.token_embedding.embedding.weight"].shape[
                    1
                ]

                # Infer number of heads if possible
                num_heads = None
                for key in state_dict.keys():
                    if "self_attn.query_projection.weight" in key:
                        # Infer number of heads from attention head dimension
                        head_dim = state_dict[key].shape[0] // d_model
                        num_heads = d_model // head_dim
                        break

                # Infer d_ff if possible
                d_ff = None
                for key in state_dict.keys():
                    if "feed_forward.linear1.linear.weight" in key:
                        d_ff = state_dict[key].shape[0]
                        break

                model_config = {
                    "src_vocab_size": src_vocab_size,
                    "tgt_vocab_size": tgt_vocab_size,
                    "d_model": d_model,
                    "num_heads": num_heads or args.num_heads,
                    "num_encoder_layers": encoder_layers,
                    "num_decoder_layers": decoder_layers,
                    "d_ff": d_ff or args.d_ff,
                    "dropout": args.dropout,
                    "max_seq_length": 100,
                    "positional_encoding": "sinusoidal",
                    "share_embeddings": args.use_joint_tokenizer,
                }
                print(f"Inferred model configuration from checkpoint: {model_config}")
        except Exception as e:
            print(f"Error loading model configuration: {e}")
            import traceback

            traceback.print_exc()

    # Create a new transformer model with proper configuration
    if model_config:
        # Use configuration from checkpoint
        model = EncoderDecoderTransformer(
            src_vocab_size=model_config["src_vocab_size"],
            tgt_vocab_size=model_config["tgt_vocab_size"],
            d_model=model_config["d_model"],
            num_heads=model_config["num_heads"],
            num_encoder_layers=model_config["num_encoder_layers"],
            num_decoder_layers=model_config["num_decoder_layers"],
            d_ff=model_config["d_ff"],
            dropout=model_config.get("dropout", args.dropout),
            max_seq_length=model_config.get("max_seq_length", 100),
            positional_encoding=model_config.get("positional_encoding", "sinusoidal"),
            share_embeddings=model_config.get(
                "share_embeddings", args.use_joint_tokenizer
            ),
        )
        print("Created model with architecture from checkpoint")
    else:
        # Use command-line arguments for configuration
        print("Creating transformer model from arguments...")
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

    # Print model architecture
    print(f"Model architecture:")
    print(f"  Encoder layers: {len(model.encoder.layers)}")
    print(f"  Decoder layers: {len(model.decoder.layers)}")
    print(
        f"  Hidden size (d_model): {model.encoder.token_embedding.embedding.weight.shape[1]}"
    )
    print(
        f"  Vocabulary sizes: src={model.encoder.token_embedding.embedding.weight.shape[0]}, tgt={model.decoder.token_embedding.embedding.weight.shape[0]}"
    )

    # Move model to the appropriate device for training
    model.to(training_device)

    # Create the trainer
    trainer = TransformerTrainer(
        model=model,
        train_dataloader=data_module.get_train_dataloader(),
        val_dataloader=val_data_module.get_val_dataloader(),
        pad_idx=pad_idx,
        lr=args.learning_rate,
        warmup_steps=args.warmup_steps,
        label_smoothing=args.label_smoothing,
        device=training_device,  # Use MPS for training
        track_perplexity=True,
        use_gradient_scaling=args.use_gradient_scaling,
        early_stopping_patience=args.early_stopping_patience,
        clip_grad=(
            args.clip_grad if args.clip_grad is not None else 0.0
        ),  # Fix for None clip_grad
    )

    # Load the pretrained model if specified
    if model_path:
        print(f"Loading pretrained model from {model_path}")
        print(f"Using absolute path: {os.path.abspath(model_path)}")

        # Try to restore full training state including optimizer and scheduler
        restored = trainer.restore_from_checkpoint(
            model_path,
            strict=True,
            reset_optimizer=args.reset_optimizer,
            reset_scheduler=args.reset_scheduler,
        )

        if restored:
            print("Successfully restored full training state")
        else:
            print(
                "Unable to restore full training state, falling back to model weights only"
            )
            try:
                # Fallback to just loading the model weights
                checkpoint = torch.load(model_path, map_location=training_device)
                if "model_state_dict" in checkpoint:
                    model.load_state_dict(checkpoint["model_state_dict"])
                    print("Loaded model weights only")
                else:
                    print("No model state found in checkpoint")
            except Exception as e:
                print(f"Error loading model: {e}")
                print("Will continue with freshly initialized model")

    # Debug option
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
            """Save checkpoint and update curriculum at end of epoch."""
            epoch_save_path = f"{checkpoint_dir}/epoch_{epoch+1}.pt"
            try:
                # Save full training state through the trainer
                trainer.save_checkpoint(epoch_save_path)
                print(f"Saved epoch {epoch+1} checkpoint to {epoch_save_path}")

                # Update curriculum stage based on completed epoch
                if args.use_curriculum:
                    data_module.update_curriculum_stage(epoch + 1)
                    # Print curriculum statistics
                    stats = data_module.get_curriculum_stats()
                    if stats:
                        print(
                            f"Curriculum stats: Stage {stats['stage']}/{stats['num_stages']-1}, "
                            f"{stats['percent_available']:.1f}% of data available, "
                            f"avg difficulty: {stats['mean_difficulty']:.2f}"
                        )
            except Exception as e:
                print(f"Error saving epoch checkpoint: {e}")

        # Set a checkpoint callback in trainer
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
                "optimizer_state_dict": trainer.optimizer.state_dict(),
                "scheduler_state_dict": trainer.scheduler.state_dict(),
                "current_epoch": trainer.current_epoch,
                "global_step": trainer.global_step,
                "best_val_loss": trainer.best_val_loss,
                "patience_counter": trainer.patience_counter,
                "history": trainer.history,
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

        # Print curriculum progression summary if curriculum learning was used
        if (
            args.use_curriculum
            and hasattr(data_module, "train_dataset")
            and hasattr(
                data_module.train_dataset, "print_curriculum_progression_summary"
            )
            and callable(
                getattr(
                    data_module.train_dataset,
                    "print_curriculum_progression_summary",
                    None,
                )
            )
        ):
            data_module.train_dataset.print_curriculum_progression_summary()
    else:
        print("Skipping training, inference only mode")
        trainer_history = {}

    # Run translation test
    run_translation_test(
        model, src_tokenizer, tgt_tokenizer, training_device, args.greedy_decoding
    )

    return trainer_history


def run_translation_test(
    model, tokenizer, tgt_tokenizer=None, device=None, greedy=False
):
    """
    Run a simple translation test with the model.

    Args:
        model: The transformer model
        tokenizer: Source tokenizer
        tgt_tokenizer: Target tokenizer (if None, uses src_tokenizer)
        device: Device to run on
        greedy: Whether to use greedy decoding
    """
    print("\n===== Translation Examples =====")

    # Use same tokenizer for source and target if no target tokenizer provided
    if tgt_tokenizer is None:
        tgt_tokenizer = tokenizer

    # If device not specified, use the model's device
    if device is None:
        device = next(model.parameters()).device

    # Try a manual token-by-token approach for a test case first
    print("\n=== Testing token-by-token analysis ===")
    model.eval()
    with torch.no_grad():
        # Special token indices
        bos_idx = tokenizer.special_tokens["bos_token_idx"]
        eos_idx = tokenizer.special_tokens["eos_token_idx"]
        pad_idx = tokenizer.special_tokens["pad_token_idx"]

        print(f"Special tokens: BOS={bos_idx}, EOS={eos_idx}, PAD={pad_idx}")

        # Test with a very simple sentence
        test_sentence = "Ich bin ein Student."
        print(f"\nTest sentence: '{test_sentence}'")

        # Tokenize source
        src_ids = tokenizer.encode(test_sentence)
        print(f"Source tokens: {src_ids}")

        # Print token strings
        token_strings = [tokenizer.decode([t]) for t in src_ids]
        print(f"Token strings: {token_strings}")

        # Add special tokens
        src_ids_with_special = [bos_idx] + src_ids + [eos_idx]
        print(f"With special tokens: {src_ids_with_special}")

        # Create tensor and mask
        src_tensor = torch.tensor([src_ids_with_special], dtype=torch.long).to(device)
        src_mask = create_padding_mask(src_tensor, pad_idx)

        # Initialize with BOS token
        tgt_ids = [bos_idx]
        tgt_tensor = torch.tensor([tgt_ids], dtype=torch.long).to(device)

        # Generate 20 tokens step by step, showing all options
        for step in range(20):
            # Create target mask (causal)
            tgt_mask = create_causal_mask(tgt_tensor.size(1), device)

            # Get model prediction
            logits = model(src_tensor, tgt_tensor, src_mask=src_mask, tgt_mask=tgt_mask)
            next_token_logits = logits[0, -1]

            # Get top tokens
            top_k = 5
            top_token_logits, top_tokens = torch.topk(next_token_logits, top_k)

            # Convert to probabilities
            top_token_probs = torch.softmax(top_token_logits, dim=-1)

            # Get the highest probability token
            next_token = top_tokens[0].item()

            # Print top tokens
            print(f"\nStep {step+1} - Top {top_k} tokens:")
            for i in range(top_k):
                token_id = top_tokens[i].item()
                token_text = tgt_tokenizer.decode([token_id])
                prob = top_token_probs[i].item()
                print(
                    f"  {i+1}. ID: {token_id}, Token: '{token_text}', Prob: {prob:.4f}"
                )

            # Add predicted token to target sequence
            tgt_ids.append(next_token)
            tgt_tensor = torch.tensor([tgt_ids], dtype=torch.long).to(device)

            # Stop if EOS token
            if next_token == eos_idx:
                print("Generated EOS token, stopping.")
                break

        # Decode the output
        output_tokens = tgt_ids[1:]  # Remove BOS token
        if output_tokens[-1] == eos_idx:
            output_tokens = output_tokens[:-1]  # Remove EOS if present

        # Print the result
        print("\nRaw generated tokens:", output_tokens)
        decoded_text = tgt_tokenizer.decode(output_tokens)
        print(f"Decoded translation: '{decoded_text}'")
        cleaned = clean_translation_output(decoded_text)
        print(f"Cleaned translation: '{cleaned}'")

    # Sample sentences for translation
    sample_sentences = [
        preprocess_input_text("Ich bin ein Student."),
        preprocess_input_text("Wo ist die Bibliothek?"),
        preprocess_input_text("Künstliche Intelligenz wird unser Leben verändern."),
        preprocess_input_text("Das Buch ist sehr interessant."),
        preprocess_input_text("Ich möchte Deutsch lernen."),
    ]

    # Generate translations with even more conservative settings
    print("\n=== Standard translation generation ===")
    translations, raw_translations = generate_translations(
        model=model,
        src_tokenizer=tokenizer,
        src_sentences=sample_sentences,
        tgt_tokenizer=tgt_tokenizer,
        max_length=30,  # Shorter translations to avoid going off track
        sampling_temp=0.01 if greedy else 0.2,  # Use near-zero temp for greedy decoding
        sampling_topk=1 if greedy else 10,  # Use top-1 for greedy decoding
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


if __name__ == "__main__":
    # Parse arguments
    parser = argparse.ArgumentParser(
        description="Train a transformer model for machine translation with joint vocabulary"
    )

    # Dataset options
    parser.add_argument(
        "--dataset",
        type=str,
        choices=[
            "europarl",
            "opensubtitles",
            "iwslt",
            "wmt",
            "combined",
        ],  # Add wmt to choices
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
        help="Path to joint tokenizer (default: models/tokenizers/combined/joint or wmt{year}/joint)",
    )

    # WMT specific options
    parser.add_argument(
        "--wmt_year",
        type=str,
        default="14",
        help="WMT dataset year without prefix (e.g., '14' for WMT14, only used if dataset=wmt)",
    )
    parser.add_argument(
        "--subset",
        type=str,
        default=None,
        help="WMT dataset subset to use, if available (e.g., 'news_commentary_v9', only used if dataset=wmt)",
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
    parser.add_argument(
        "--reset_optimizer",
        action="store_true",
        help="When loading a model, don't restore optimizer state (useful when changing batch size)",
    )
    parser.add_argument(
        "--reset_scheduler",
        action="store_true",
        help="When loading a model, don't restore scheduler state (useful when changing learning rate)",
    )
    parser.add_argument(
        "--greedy_decoding",
        action="store_true",
        help="Use greedy decoding instead of sampling for translations",
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

    # Add clip_grad to argument parser
    parser.add_argument(
        "--clip_grad",
        type=float,
        default=None,
        help="Gradient clipping value (None to disable)",
    )

    # Curriculum learning options
    parser.add_argument(
        "--use_curriculum",
        action="store_true",
        help="Whether to use curriculum learning",
    )
    parser.add_argument(
        "--curriculum_strategy",
        type=str,
        choices=["length", "vocab", "similarity"],
        default="length",
        help="Strategy for curriculum learning (length, vocab, similarity)",
    )
    parser.add_argument(
        "--curriculum_stages",
        type=int,
        default=5,
        help="Number of curriculum stages",
    )

    # Add IWSLT-specific options
    parser.add_argument(
        "--year",
        type=str,
        default="2017",
        help="IWSLT dataset year (only used if dataset=iwslt)",
    )

    # Add preprocessing optimization options
    parser.add_argument(
        "--preprocessing_workers",
        type=int,
        default=4,
        help="Number of workers for parallel preprocessing",
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
