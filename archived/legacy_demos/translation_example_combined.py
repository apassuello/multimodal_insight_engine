import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict, Tuple, Iterator, Generator
import os
import sys
import argparse
import random
from tqdm import tqdm
import re
import json
from torch.utils.data import Dataset, DataLoader, IterableDataset

# Disable interactive mode for matplotlib to prevent opening windows
plt.ioff()

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models.transformer import EncoderDecoderTransformer
from src.data.sequence_data import TransformerDataModule
from src.training.transformer_trainer import TransformerTrainer
from src.training.transformer_utils import create_padding_mask, create_causal_mask
from src.data.tokenization import OptimizedBPETokenizer
from src.optimization.mixed_precision import MixedPrecisionConverter
from src.data.europarl_dataset import EuroparlDataset
from src.data.opensubtitles_dataset import OpenSubtitlesDataset
from src.data.iwslt_dataset import IWSLTDataset
from src.data.combined_wmt_translation_dataset import (
    load_dataset_from_file as load_combined_dataset,
)


class StreamingTranslationDataset(IterableDataset):
    """A memory-efficient dataset that loads and processes data on-the-fly."""

    def __init__(
        self,
        file_path: str,
        tokenizer: OptimizedBPETokenizer,
        max_samples: int = 0,
        start_idx: int = 0,
        skip_empty: bool = True,
        buffer_size: int = 10000,
    ):
        """
        Args:
            file_path: Path to the JSONL dataset file
            tokenizer: Tokenizer to use for encoding
            max_samples: Maximum number of samples to use (0 for all)
            start_idx: Index to start reading from
            skip_empty: Whether to skip empty examples
            buffer_size: Size of the buffer for shuffling
        """
        self.file_path = file_path
        self.tokenizer = (
            tokenizer  # Keep the tokenizer reference - we'll use single process
        )
        self.max_samples = max_samples
        self.start_idx = start_idx
        self.skip_empty = skip_empty
        self.buffer_size = buffer_size

        # Count total lines in the file - do this once at initialization
        self.total_lines = self._count_lines()

    def _count_lines(self):
        """Count the number of lines in the file."""
        print(f"Counting lines in {self.file_path}...")
        try:
            with open(self.file_path, "r", encoding="utf-8") as f:
                count = sum(1 for _ in f)
            print(f"Total lines: {count}")
            return count
        except Exception as e:
            print(f"Error counting lines: {e}")
            return 1000000  # Default to a large number if counting fails

    def __len__(self):
        """Return an estimate of the dataset size."""
        if self.max_samples > 0:
            return min(self.max_samples, self.total_lines - self.start_idx)
        return max(1, self.total_lines - self.start_idx)  # Ensure at least length 1

    def _stream_examples(self, worker_id: int = 0, num_workers: int = 1):
        """Stream examples from the file, processing only the assigned portion."""
        # Calculate which portion of the file this worker should process
        lines_per_worker = max(1, self.total_lines // max(1, num_workers))
        start_line = self.start_idx + (worker_id * lines_per_worker)
        end_line = (
            start_line + lines_per_worker
            if worker_id < num_workers - 1
            else self.total_lines
        )

        # Limit to max_samples if specified
        if self.max_samples > 0:
            samples_per_worker = max(1, self.max_samples // max(1, num_workers))
            end_line = min(start_line + samples_per_worker, end_line)

        try:
            # Use simple line-by-line reading for reliability
            with open(self.file_path, "r", encoding="utf-8") as f:
                # Skip to the start line
                for _ in range(start_line):
                    next(f, None)

                # Process only up to end_line
                count = 0
                for line_idx, line in enumerate(f):
                    if line_idx + start_line >= end_line:
                        break

                    try:
                        example = json.loads(line)
                        src_text = example.get("source", "")
                        tgt_text = example.get("target", "")

                        if self.skip_empty and (not src_text or not tgt_text):
                            continue

                        # Tokenize in the main process (single-process loading)
                        src_ids = self.tokenizer.encode(src_text)
                        tgt_ids = self.tokenizer.encode(tgt_text)

                        # Add special tokens
                        src_ids = (
                            [self.tokenizer.special_tokens["bos_token_idx"]]
                            + src_ids
                            + [self.tokenizer.special_tokens["eos_token_idx"]]
                        )
                        tgt_ids = (
                            [self.tokenizer.special_tokens["bos_token_idx"]]
                            + tgt_ids
                            + [self.tokenizer.special_tokens["eos_token_idx"]]
                        )

                        yield (src_ids, tgt_ids)
                        count += 1

                    except Exception as e:
                        print(f"Error processing line {line_idx + start_line}: {e}")
                        continue

                print(f"Worker {worker_id}: Processed {count} examples")

        except Exception as e:
            print(f"Worker {worker_id}: Error in _stream_examples: {e}")
            # If there's an error, yield at least one dummy example to prevent empty iterator errors
            src_ids = [
                self.tokenizer.special_tokens["bos_token_idx"],
                self.tokenizer.special_tokens["eos_token_idx"],
            ]
            tgt_ids = [
                self.tokenizer.special_tokens["bos_token_idx"],
                self.tokenizer.special_tokens["eos_token_idx"],
            ]
            yield (src_ids, tgt_ids)

    def __iter__(self):
        """Return an iterator over the dataset."""
        # Always use worker_id=0, num_workers=1 for single-process loading
        generator = self._stream_examples(0, 1)

        # Create a buffer for shuffling
        buffer = []
        item_yielded = False  # Track if we've yielded at least one item

        for example in generator:
            buffer.append(example)
            item_yielded = True

            if len(buffer) >= self.buffer_size:
                # Shuffle the buffer
                random.shuffle(buffer)

                # Yield examples from the buffer
                for i in range(self.buffer_size):
                    yield buffer[i]

                # Keep only remaining examples
                buffer = buffer[self.buffer_size :]

        # Shuffle and yield any remaining examples
        if buffer:
            random.shuffle(buffer)
            for example in buffer:
                yield example

        # If we didn't yield any examples, yield at least one dummy example
        # This prevents empty iterator errors in PyTorch DataLoader
        if not item_yielded:
            print(
                f"Warning: No items yielded from dataset. Creating a dummy example to prevent empty iterator error."
            )
            # Create a dummy example with source and target IDs
            src_ids = [
                self.tokenizer.special_tokens["bos_token_idx"],
                self.tokenizer.special_tokens["eos_token_idx"],
            ]
            tgt_ids = [
                self.tokenizer.special_tokens["bos_token_idx"],
                self.tokenizer.special_tokens["eos_token_idx"],
            ]
            yield (src_ids, tgt_ids)


class MemoryEfficientTransformerDataModule:
    """A memory-efficient data module for transformer training."""

    def __init__(
        self,
        dataset_path: str,
        tokenizer: OptimizedBPETokenizer,
        batch_size: int,
        max_src_len: int,
        max_tgt_len: int,
        pad_idx: int,
        train_samples: int = 0,
        val_samples: int = 0,
        val_start_idx: int = 0,
        num_workers: int = 0,  # Default to 0 for single-process
        buffer_size: int = 10000,
    ):
        """
        Args:
            dataset_path: Path to the JSONL dataset file
            tokenizer: Tokenizer to use for encoding
            batch_size: Batch size
            max_src_len: Maximum source sequence length
            max_tgt_len: Maximum target sequence length
            pad_idx: Padding token index
            train_samples: Maximum number of training samples (0 for all)
            val_samples: Maximum number of validation samples (0 for all)
            val_start_idx: Index to start reading validation data from
            num_workers: Number of workers for data loading (use 0 for single-process)
            buffer_size: Size of the buffer for shuffling
        """
        self.dataset_path = dataset_path
        self.tokenizer = tokenizer
        self.batch_size = batch_size
        self.max_src_len = max_src_len
        self.max_tgt_len = max_tgt_len
        self.pad_idx = pad_idx
        self.train_samples = train_samples
        self.val_samples = val_samples
        self.num_workers = max(0, num_workers)  # Ensure non-negative
        self.buffer_size = buffer_size
        self.val_start_idx = val_start_idx

        # Count total lines just once during initialization
        with open(dataset_path, "r", encoding="utf-8") as f:
            self.total_lines = sum(1 for _ in f)

        # Calculate validation start index if requested validation samples > 0
        if val_samples > 0 and val_start_idx == 0:
            self.val_start_idx = max(0, self.total_lines - val_samples)

        # Store the train and validation datasets
        self.train_dataset = None
        self.val_dataset = None

    def _collate_fn(self, batch):
        """Collate function for the DataLoader."""
        src_sequences = []
        tgt_sequences = []

        for src_seq, tgt_seq in batch:
            # Truncate sequences if they're too long
            src_seq = src_seq[: self.max_src_len]
            tgt_seq = tgt_seq[: self.max_tgt_len]

            src_sequences.append(src_seq)
            tgt_sequences.append(tgt_seq)

        # Pad sequences to the maximum length in the batch
        src_lengths = [len(seq) for seq in src_sequences]
        tgt_lengths = [len(seq) for seq in tgt_sequences]

        max_src_len_batch = min(max(src_lengths), self.max_src_len)
        max_tgt_len_batch = min(max(tgt_lengths), self.max_tgt_len)

        # Pad source sequences
        padded_src = [
            seq + [self.pad_idx] * (max_src_len_batch - len(seq))
            for seq in src_sequences
        ]

        # Pad target sequences
        padded_tgt = [
            seq + [self.pad_idx] * (max_tgt_len_batch - len(seq))
            for seq in tgt_sequences
        ]

        # Convert to tensors
        src_tensor = torch.tensor(padded_src, dtype=torch.long)
        tgt_tensor = torch.tensor(padded_tgt, dtype=torch.long)

        return {"src": src_tensor, "tgt": tgt_tensor}

    def get_train_dataloader(self):
        """Get the training DataLoader."""
        self.train_dataset = StreamingTranslationDataset(
            file_path=self.dataset_path,
            tokenizer=self.tokenizer,
            max_samples=self.train_samples,
            start_idx=0,  # Start from the beginning
            buffer_size=self.buffer_size,
        )

        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            collate_fn=self._collate_fn,
            num_workers=0,  # Always use single-process due to pickling issues
            pin_memory=True,
        )

    def get_val_dataloader(self):
        """Get the validation DataLoader."""
        if self.val_samples <= 0:
            return None

        self.val_dataset = StreamingTranslationDataset(
            file_path=self.dataset_path,
            tokenizer=self.tokenizer,
            max_samples=self.val_samples,
            start_idx=self.val_start_idx,
            buffer_size=self.buffer_size,
        )

        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            collate_fn=self._collate_fn,
            num_workers=0,  # Always use single-process due to pickling issues
            pin_memory=True,
        )

    def get_train_len(self):
        """Get the estimated length of the training dataset."""
        if self.train_dataset is not None:
            return len(self.train_dataset)

        # Estimate based on parameters
        if self.train_samples > 0:
            return min(self.train_samples, self.total_lines)
        return self.total_lines

    def get_val_len(self):
        """Get the estimated length of the validation dataset."""
        if self.val_dataset is not None:
            return len(self.val_dataset)

        # Estimate based on parameters
        if self.val_samples > 0:
            return min(self.val_samples, self.total_lines - self.val_start_idx)
        return 0


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


def load_pretrained_model(model_path, tokenizer, device):
    """
    Load a pretrained translation model.

    Args:
        model_path: Path to the saved model checkpoint
        tokenizer: Joint tokenizer for both languages
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

            # Get vocabulary size from the state dict
            vocab_size = state_dict["encoder.token_embedding.embedding.weight"].shape[0]
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
                "vocab_size": vocab_size,
                "d_model": d_model,
                "num_heads": num_heads or 8,
                "num_encoder_layers": encoder_layers,
                "num_decoder_layers": decoder_layers,
                "d_ff": 2048,  # Default, but should be inferred if possible
                "dropout": 0.1,
                "max_seq_length": 100,
                "positional_encoding": "sinusoidal",
                "share_embeddings": True,  # We're using a shared vocabulary
            }
            print(f"Inferred model configuration: {model_config}")
        else:
            raise ValueError(
                "Checkpoint does not contain model_config or model_state_dict"
            )

    # Make sure we use the exact vocabulary size from the saved model
    if "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
        model_config["vocab_size"] = state_dict[
            "encoder.token_embedding.embedding.weight"
        ].shape[0]

        # Print warning if tokenizer vocab size doesn't match the model
        if tokenizer.vocab_size != model_config["vocab_size"]:
            print(
                f"WARNING: Tokenizer vocabulary size ({tokenizer.vocab_size}) doesn't match model ({model_config['vocab_size']})"
            )

    # Create model with the inferred architecture
    model = EncoderDecoderTransformer(
        src_vocab_size=model_config["vocab_size"],
        tgt_vocab_size=model_config[
            "vocab_size"
        ],  # Same vocab size for both src and tgt
        d_model=model_config.get("d_model", 512),
        num_heads=model_config.get("num_heads", 16),
        num_encoder_layers=model_config.get("num_encoder_layers", 6),
        num_decoder_layers=model_config.get("num_decoder_layers", 6),
        d_ff=model_config.get("d_ff", 2048),
        dropout=model_config.get("dropout", 0.2),
        max_seq_length=model_config.get("max_seq_length", 100),
        positional_encoding=model_config.get("positional_encoding", "sinusoidal"),
        share_embeddings=True,  # Always share embeddings with joint vocabulary
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


def preprocess_data_with_bpe(dataset, tokenizer):
    """
    Preprocess the dataset for training using joint BPE tokenizer.

    Args:
        dataset: Translation dataset (list of (source, target) pairs)
        tokenizer: Joint BPE tokenizer for both languages

    Returns:
        Lists of tokenized source and target sequences
    """
    src_sequences = []
    tgt_sequences = []

    for src_text, tgt_text in dataset:
        # Tokenize with joint BPE vocabulary
        src_ids = tokenizer.encode(src_text)
        tgt_ids = tokenizer.encode(tgt_text)

        # Add special tokens
        src_ids = (
            [tokenizer.special_tokens["bos_token_idx"]]
            + src_ids
            + [tokenizer.special_tokens["eos_token_idx"]]
        )
        tgt_ids = (
            [tokenizer.special_tokens["bos_token_idx"]]
            + tgt_ids
            + [tokenizer.special_tokens["eos_token_idx"]]
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
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    # Set device
    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available() else "cpu"
    )
    print(f"Using device: {device}")

    # Load the joint BPE tokenizer
    print("Loading joint BPE tokenizer...")
    tokenizer = OptimizedBPETokenizer.from_pretrained(
        args.tokenizer_path or "models/tokenizers/combined/joint"
    )
    print(f"Loaded joint tokenizer with vocab size: {tokenizer.vocab_size}")

    # Get special token indices
    pad_idx = tokenizer.special_tokens["pad_token_idx"]
    bos_idx = tokenizer.special_tokens["bos_token_idx"]
    eos_idx = tokenizer.special_tokens["eos_token_idx"]

    # Create memory-efficient data module
    print(f"Creating memory-efficient data module for dataset: {args.dataset}...")

    if args.dataset == "combined":
        dataset_path = args.combined_dataset_path
        print(f"Using combined dataset from: {dataset_path}")
    else:
        # For other datasets, we still use the original loading logic
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
        else:  # opensubtitles
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

        # Convert datasets to standard format and train as before
        # ... existing code for non-combined datasets ...
        return  # Skip the rest of the function

    # Create memory-efficient data module for large datasets
    print(
        f"Using memory-efficient streaming data module with buffer size {args.buffer_size}"
    )
    print(
        f"WARNING: Using single-process data loading due to pickling constraints with the tokenizer"
    )
    effective_batch_size = args.batch_size
    print(f"Using batch size of {effective_batch_size} for training")

    data_module = MemoryEfficientTransformerDataModule(
        dataset_path=dataset_path,
        tokenizer=tokenizer,
        batch_size=effective_batch_size,
        max_src_len=args.max_seq_length,
        max_tgt_len=args.max_seq_length,
        pad_idx=pad_idx,
        train_samples=args.max_train_examples,
        val_samples=args.max_val_examples,
        num_workers=0,  # Force single-process data loading
        buffer_size=args.buffer_size,
    )

    # Create the dataloaders with our modified classes
    train_dataloader = data_module.get_train_dataloader()
    val_dataloader = data_module.get_val_dataloader()

    # If loading a pretrained model is requested
    if args.load_model:
        # If the path doesn't exist, try to find a valid model path
        if not os.path.exists(args.load_model):
            dataset_prefix = f"{args.dataset}_{args.src_lang}_{args.tgt_lang}_joint"
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
                model = load_pretrained_model(args.load_model, tokenizer, device)
                print("Loaded pretrained model successfully")
            except Exception as e:
                print(f"Error loading model: {e}")
                # Continue with the new model if loading fails
                model = None
        else:
            print("Training new model from scratch")
            model = None
    else:
        print("Training new model from scratch")
        model = None

    # Create a new model if one wasn't loaded
    if model is None:
        print("Creating new transformer model with shared embeddings...")
        model = EncoderDecoderTransformer(
            src_vocab_size=tokenizer.vocab_size,
            tgt_vocab_size=tokenizer.vocab_size,  # Same vocabulary size for both
            d_model=args.d_model,
            num_heads=args.num_heads,
            num_encoder_layers=args.num_encoder_layers,
            num_decoder_layers=args.num_decoder_layers,
            d_ff=args.d_ff,
            dropout=args.dropout,
            max_seq_length=args.max_seq_length,
            positional_encoding="sinusoidal",
            share_embeddings=True,  # Enable embedding sharing for joint vocabulary
        )

    # Print model parameter count
    num_params = count_parameters(model)
    print(f"Model has {num_params:,} trainable parameters")

    model.to(device)

    # Create the standard transformer trainer
    print(f"Creating transformer trainer")

    # Use the dataloaders
    # In translation_example_combined.py where you create the trainer:
    trainer = TransformerTrainer(
        model=model,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        pad_idx=pad_idx,
        lr=args.learning_rate,
        warmup_steps=args.warmup_steps,
        label_smoothing=args.label_smoothing,
        device=device,
        track_perplexity=True,
        early_stopping_patience=args.early_stopping_patience,
        clip_grad=args.clip_grad,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        betas=(0.9, 0.98),  # This is important for transformer training
        weight_decay=0.001,  # Reduced from 0.01 to avoid overly aggressive regularization
    )

    # Create save directory
    os.makedirs("models", exist_ok=True)

    # Prefix for model file based on dataset and languages, include "joint" to indicate joint vocabulary
    model_prefix = f"{args.dataset}_{args.src_lang}_{args.tgt_lang}_joint"

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
                        "vocab_size": model.encoder.token_embedding.embedding.weight.shape[
                            0
                        ],
                        "d_model": model.encoder.token_embedding.embedding.weight.shape[
                            1
                        ],
                        "num_encoder_layers": len(model.encoder.layers),
                        "num_decoder_layers": len(model.decoder.layers),
                        "share_embeddings": True,
                    },
                }
                torch.save(checkpoint, epoch_save_path)
                print(f"Saved epoch {epoch+1} checkpoint to {epoch_save_path}")
            except Exception as e:
                print(f"Error saving epoch checkpoint: {e}")

        # Set a checkpoint callback in trainer
        trainer.epoch_end_callback = save_epoch_checkpoint

        trainer_history = trainer.train(epochs=args.epochs, save_path=save_path)

        # Ensure model is saved directly from script
        try:
            print(f"\nDirectly saving model from script to: {save_path}")
            # Create a checkpoint dictionary with essential information
            model_checkpoint = {
                "model_state_dict": model.state_dict(),
                "model_config": {
                    "vocab_size": model.encoder.token_embedding.embedding.weight.shape[
                        0
                    ],
                    "d_model": model.encoder.token_embedding.embedding.weight.shape[1],
                    "num_encoder_layers": len(model.encoder.layers),
                    "num_decoder_layers": len(model.decoder.layers),
                    "dropout": getattr(model, "dropout", 0.1),
                    "share_embeddings": True,
                },
            }

            # Add a .pt extension for this direct save
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

    # Generate translations with the joint vocabulary model
    translations, raw_translations = generate_translations(
        model=model,
        tokenizer=tokenizer,  # Single tokenizer for both languages
        src_sentences=sample_sentences,
        device=device,
        max_length=50,
        sampling_temp=1.0,
        sampling_topk=1,
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
        choices=["combined", "europarl", "opensubtitles"],
        default="combined",
        help="Dataset to use for training",
    )
    parser.add_argument(
        "--combined_dataset_path",
        type=str,
        default="combined_de_en_dataset.jsonl",
        help="Path to the combined dataset JSONL file",
    )
    parser.add_argument(
        "--max_train_examples",
        type=int,
        default=500000,
        help="Maximum number of training examples to use",
    )
    parser.add_argument(
        "--max_val_examples",
        type=int,
        default=5000,
        help="Maximum number of validation examples to use",
    )
    parser.add_argument(
        "--src_lang", type=str, default="de", help="Source language code"
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=4,  # Set a reasonable default value
        help="Number of steps to accumulate gradients before updating weights",
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

    # Memory efficiency options
    parser.add_argument(
        "--buffer_size",
        type=int,
        default=10000,
        help="Size of the buffer for shuffling in streaming dataset",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=4,
        help="Number of worker processes for data loading",
    )
    parser.add_argument(
        "--max_seq_length",
        type=int,
        default=100,
        help="Maximum sequence length for source and target",
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
        "--batch_size", type=int, default=32, help="Batch size for training"
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

    # Add random seed
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed for reproducibility"
    )

    # Add clip_grad parameter
    parser.add_argument(
        "--clip_grad",
        type=float,
        default=1.0,
        help="Gradient clipping value (default: 1.0)",
    )

    args = parser.parse_args()

    # Set random seed for reproducibility
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    # Run the training process
    main(args)
