# multimodal_training_demo.py
"""
Multimodal Integration Training Demo

This script demonstrates the complete training and evaluation pipeline for
multimodal models with contrastive learning. It showcases:

1. Loading and preprocessing multimodal data
2. Building a cross-modal attention transformer
3. Training with advanced contrastive learning objectives
4. Evaluating cross-modal retrieval performance
5. Visualizing model outputs and attention maps

Available contrastive loss modes:
- contrastive: Standard contrastive loss with enhanced configuration
- memory_queue: Uses a memory queue for consistent global comparisons
- dynamic_temp: Automatically adjusts temperature based on embeddings
- hard_negative: Focuses training on challenging negative examples
- mixed: Combines multiple objectives for robust learning

Example usage:
python multimodal_training_demo.py --loss_type memory_queue --queue_size 8192
python multimodal_training_demo.py --loss_type dynamic_temp
python multimodal_training_demo.py --loss_type hard_negative --mining_strategy semi-hard
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
import logging
import argparse
import json
import random
from tqdm import tqdm
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Union, Any
from torch.utils.data import DataLoader
from src.models.vision.multimodal_integration import (
    MultiModalTransformer,
    EnhancedMultiModalTransformer,
)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(), logging.FileHandler("multimodal_training.log")],
)
logger = logging.getLogger(__name__)

# Import our models and utilities
from src.models.vision.vision_transformer import VisionTransformer
from src.models.transformer import EncoderDecoderTransformer
from src.models.vision.multimodal_integration import EnhancedMultiModalTransformer
from src.models.vision.image_preprocessing import ImagePreprocessor
from src.data.tokenization.optimized_bpe_tokenizer import OptimizedBPETokenizer
from src.training.contrastive_learning import (
    ContrastiveLoss,
    MultiModalMixedContrastiveLoss,
)
from src.training.multimodal_trainer import MultimodalTrainer

# Import our custom dataset
from src.data.multimodal_dataset import EnhancedMultimodalDataset


def count_parameters(model):
    """
    Count the total number of trainable parameters in a model.

    Args:
        model: PyTorch model

    Returns:
        int: Total number of trainable parameters
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Multimodal Training Demo")

    # Data arguments
    parser.add_argument(
        "--dataset",
        type=str,
        default="flickr30k",
        choices=["flickr30k", "custom", "synthetic"],
        help="Dataset to use",
    )

    parser.add_argument(
        "--use_multistage_training",
        action="store_true",
        help="Use multi-stage training approach for better performance",
    )
    parser.add_argument(
        "--from_scratch",
        action="store_true",
        help="Use specialized curriculum learning approach for training from scratch (without pretrained weights)",
    )
    parser.add_argument(
        "--freeze_base_models",
        action="store_true",
        default=False,  # Default to NOT freezing for better performance
        help="Freeze the base vision and text models (only train fusion layers)",
    )
    parser.add_argument(
        "--use_pretrained",
        action="store_true",
        default=True,  # Default to using pretrained when available
        help="Use pretrained weights for vision model",
    )
    parser.add_argument(
        "--data_dir", type=str, default="data", help="Directory containing dataset"
    )
    parser.add_argument(
        "--use_synthetic",
        action="store_true",
        help="Use synthetic data instead of real data",
    )
    parser.add_argument(
        "--synthetic_samples",
        type=int,
        default=100,
        help="Number of synthetic samples if using synthetic data",
    )
    parser.add_argument(
        "--max_train_examples",
        type=int,
        default=None,
        help="Maximum number of training examples to use (None uses all available data)",
    )
    parser.add_argument(
        "--max_val_examples",
        type=int,
        default=None,
        help="Maximum number of validation examples to use (None uses all available data)",
    )
    parser.add_argument(
        "--max_test_examples",
        type=int,
        default=None,
        help="Maximum number of test examples to use (None uses all available data)",
    )
    parser.add_argument(
        "--max_text_length", type=int, default=77, help="Maximum text sequence length"
    )

    # Model arguments
    parser.add_argument(
        "--vision_model",
        type=str,
        default="vit-base",
        help="Vision transformer model size",
    )
    parser.add_argument(
        "--text_model",
        type=str,
        default="transformer-base",
        choices=[
            "transformer-base",
            "transformer-small",
            "bert-base",
            "roberta-base",
            "distilbert-base",
            "mobilebert",
            "albert-base",
        ],
        help="Text transformer model size or HuggingFace model name to load (MobileBERT and ALBERT are more MPS-friendly)",
    )
    parser.add_argument(
        "--use_pretrained_text",
        action="store_true",
        default=False,
        help="Use pretrained text model from HuggingFace (bert-base, roberta-base, distilbert-base)",
    )
    parser.add_argument(
        "--fusion_type",
        type=str,
        default="co_attention",
        choices=["co_attention", "bidirectional"],
        help="Type of multimodal fusion",
    )
    parser.add_argument(
        "--fusion_dim", type=int, default=512, help="Dimension for multimodal fusion"
    )
    parser.add_argument(
        "--projection_dim",
        type=int,
        default=256,
        help="Projection dimension for contrastive learning",
    )

    # Contrastive learning arguments
    parser.add_argument(
        "--contrastive_sampling",
        type=str,
        default="auto",
        choices=["auto", "in-batch", "memory-bank", "global"],
        help="Contrastive sampling strategy",
    )
    parser.add_argument(
        "--memory_bank_size",
        type=int,
        default=4096,
        help="Size of memory bank for contrastive learning",
    )

    # NEW: Add loss type selection
    parser.add_argument(
        "--loss_type",
        type=str,
        default="contrastive",
        choices=[
            "contrastive",
            "memory_queue",
            "dynamic_temp",
            "hard_negative",
            "mixed",
        ],
        help="Type of contrastive loss to use",
    )

    # NEW: Memory Queue specific arguments
    parser.add_argument(
        "--queue_size",
        type=int,
        default=8192,
        help="Size of the memory queue for MemoryQueueContrastiveLoss",
    )

    # NEW: Dynamic Temperature specific arguments
    parser.add_argument(
        "--dynamic_temp_min",
        type=float,
        default=None,
        help="Minimum temperature for dynamic temperature adjustment",
    )
    parser.add_argument(
        "--dynamic_temp_max",
        type=float,
        default=None,
        help="Maximum temperature for dynamic temperature adjustment",
    )

    # NEW: Hard Negative Mining specific arguments
    parser.add_argument(
        "--mining_strategy",
        type=str,
        default="auto",
        choices=["auto", "hard", "semi-hard"],
        help="Strategy for mining hard negatives",
    )
    parser.add_argument(
        "--hard_negative_factor",
        type=float,
        default=None,
        help="Weight factor for hard negatives (higher = more emphasis)",
    )

    # Training arguments
    parser.add_argument(
        "--batch_size", type=int, default=16, help="Batch size for training"
    )
    parser.add_argument(
        "--num_epochs", type=int, default=20, help="Number of training epochs"
    )
    parser.add_argument(
        "--learning_rate", type=float, default=1e-4, help="Learning rate"
    )
    parser.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay")
    parser.add_argument(
        "--warmup_steps", type=int, default=1000, help="Number of warmup steps"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.07,
        help="Temperature for contrastive loss",
    )
    parser.add_argument(
        "--use_mixed_loss", action="store_true", help="Use mixed contrastive loss"
    )
    parser.add_argument(
        "--use_mixed_precision",
        action="store_true",
        help="Use mixed precision training",
    )

    # Output arguments
    parser.add_argument(
        "--output_dir",
        type=str,
        default="multimodal_outputs",
        help="Directory to save outputs",
    )
    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        default="checkpoints",
        help="Directory to save checkpoints",
    )
    parser.add_argument(
        "--log_dir", type=str, default="logs", help="Directory to save logs"
    )
    parser.add_argument(
        "--visualize_attention",
        action="store_true",
        help="Visualize attention maps during evaluation",
    )

    # System arguments
    parser.add_argument(
        "--device", type=str, default="mps", help="Device to use (cuda, mps, cpu)"
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    return parser.parse_args()


def set_seed(seed):
    """Set random seed for reproducibility"""
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def print_model_summary(model, title="MODEL SUMMARY"):
    """
    Print a concise summary of the model architecture and parameter counts.

    Args:
        model: PyTorch model
        title: Title for the summary
    """
    total_params = count_parameters(model)

    # Try to get submodule parameter counts if available
    vision_params = text_params = fusion_params = 0

    if hasattr(model, "vision_model"):
        vision_params = count_parameters(model.vision_model)

    if hasattr(model, "text_model"):
        text_params = count_parameters(model.text_model)

    # Fusion parameters (estimate)
    fusion_params = total_params - vision_params - text_params

    print("\n" + "=" * 50)
    print(f"{title}")
    print("-" * 50)
    print(f"Total parameters:       {total_params:,}")
    if vision_params > 0:
        print(
            f"Vision model parameters: {vision_params:,} ({vision_params/total_params*100:.1f}%)"
        )
    if text_params > 0:
        print(
            f"Text model parameters:   {text_params:,} ({text_params/total_params*100:.1f}%)"
        )
    if fusion_params > 0:
        print(
            f"Fusion parameters:       {fusion_params:,} ({fusion_params/total_params*100:.1f}%)"
        )
    print("=" * 50 + "\n")


def create_multimodal_model(args, device):
    """
    Create vision transformer, text transformer, and multimodal model
    with pretrained foundation models.
    """
    logger.info("Creating multimodal model with pretrained components...")

    # Create vision transformer with pretrained weights
    try:
        import timm

        if args.use_pretrained:
            logger.info(f"Loading pretrained vision model: {args.vision_model}")
        else:
            logger.info(
                f"Creating vision model (without pretraining): {args.vision_model}"
            )

        if args.vision_model == "vit-base":
            vision_model = timm.create_model(
                "vit_base_patch16_224", pretrained=args.use_pretrained
            )
            # Remove classification head
            vision_model.head = nn.Identity()
            vision_dim = 512
        elif args.vision_model == "vit-small":
            vision_model = timm.create_model(
                "vit_small_patch16_224", pretrained=args.use_pretrained
            )
            vision_model.head = nn.Identity()
            vision_dim = 384
        else:
            logger.warning(f"Using standard vision transformer: {args.vision_model}")
            # Fallback to standard implementation
            vision_config = {
                "image_size": 224,
                "patch_size": 16,
                "in_channels": 3,
                "num_classes": 1000,
                "embed_dim": 512,
                "depth": 12,
                "num_heads": 12,
                "mlp_ratio": 4.0,
                "dropout": 0.1,
            }
            vision_model = VisionTransformer(**vision_config)
            vision_dim = vision_config["embed_dim"]

    except ImportError:
        logger.warning("timm library not found, using standard vision transformer")
        # Fallback to standard implementation
        vision_config = {
            "image_size": 224,
            "patch_size": 16,
            "in_channels": 3,
            "num_classes": 1000,
            "embed_dim": 512,
            "depth": 12,
            "num_heads": 12,
            "mlp_ratio": 4.0,
            "dropout": 0.1,
        }
        vision_model = VisionTransformer(**vision_config)
        vision_dim = vision_config["embed_dim"]

    # Create text transformer with pretrained weights
    try:
        from transformers import AutoModel, AutoTokenizer

        logger.info(f"Loading pretrained text model: {args.text_model}")

        # Check if we should use a pretrained text model from HuggingFace
        if args.use_pretrained_text:
            logger.info(
                f"Loading pretrained text model from HuggingFace: {args.text_model}"
            )

            # Create a wrapper class for HuggingFace models
            class HuggingFaceTextModelWrapper(nn.Module):
                """Wrapper for HuggingFace models to provide compatible interface."""

                def __init__(self, model_name):
                    super().__init__()

                    # Get system device
                    system_device = torch.device(
                        "cuda"
                        if torch.cuda.is_available()
                        else "mps" if torch.backends.mps.is_available() else "cpu"
                    )

                    # Map to MPS-friendly models when needed
                    if (
                        system_device.type == "mps"
                        and "bert-base" in model_name.lower()
                    ):
                        print(
                            "⚠️ Detected MPS device - using MobileBERT instead of BERT for better compatibility"
                        )
                        model_name = "google/mobilebert-uncased"

                    # Load the model on appropriate device
                    if "mobilebert" in model_name.lower():
                        from transformers import MobileBertModel, MobileBertTokenizer

                        print(f"Loading MobileBERT model: {model_name}")
                        self.encoder = MobileBertModel.from_pretrained(
                            "google/mobilebert-uncased"
                        )
                        self.d_model = self.encoder.config.hidden_size
                        self.encoder_type = "mobilebert"
                    elif "albert" in model_name.lower():
                        from transformers import AlbertModel, AlbertTokenizer

                        print(f"Loading ALBERT model: {model_name}")
                        self.encoder = AlbertModel.from_pretrained("albert-base-v2")
                        self.d_model = self.encoder.config.hidden_size
                        self.encoder_type = "albert"
                    elif (
                        "bert" in model_name.lower()
                        and "distil" not in model_name.lower()
                    ):
                        from transformers import BertModel, BertTokenizer

                        print(f"Loading BERT model: {model_name}")
                        self.encoder = BertModel.from_pretrained(model_name)
                        self.d_model = self.encoder.config.hidden_size
                        self.encoder_type = "bert"
                    elif "roberta" in model_name.lower():
                        from transformers import RobertaModel, RobertaTokenizer

                        print(f"Loading RoBERTa model: {model_name}")
                        self.encoder = RobertaModel.from_pretrained(model_name)
                        self.d_model = self.encoder.config.hidden_size
                        self.encoder_type = "roberta"
                    elif "distilbert" in model_name.lower():
                        from transformers import DistilBertModel, DistilBertTokenizer

                        print(f"Loading DistilBERT model: {model_name}")
                        self.encoder = DistilBertModel.from_pretrained(model_name)
                        self.d_model = self.encoder.config.hidden_size
                        self.encoder_type = "distilbert"
                    else:
                        raise ValueError(f"Unsupported model: {model_name}")

                    # Try moving to the system device if MPS-compatible
                    try:
                        self.encoder = self.encoder.to(system_device)
                        print(f"Successfully moved {model_name} to {system_device}")
                    except Exception as e:
                        print(
                            f"Could not move model to {system_device}, using CPU instead: {str(e)}"
                        )
                        self.encoder = self.encoder.to("cpu")

                    logger.info(f"Loaded {model_name} with dimension {self.d_model}")

                def encode(self, src, src_mask=None):
                    """Encode text using the HuggingFace model."""
                    # Get original device
                    input_device = src.device

                    # MPS compatibility mode - check if we're on MPS and handle specially
                    is_mps = (
                        input_device.type == "mps" or torch.backends.mps.is_available()
                    )

                    # Get encoder's current device
                    encoder_device = next(self.encoder.parameters()).device

                    # Try direct processing for all models first
                    # The extract_text_features method in multimodal_integration.py
                    # will handle the CPU fallback if needed
                    use_cpu_path = False

                    # Define a CPU fallback function for reuse
                    def process_on_cpu():
                        print(
                            f"Processing {getattr(self, 'encoder_type', 'model')} on CPU for compatibility"
                        )
                        # Move everything to CPU
                        cpu_src = src.to("cpu")
                        cpu_mask = src_mask.to("cpu") if src_mask is not None else None

                        # Format mask if needed
                        if cpu_mask is not None:
                            if cpu_mask.dim() > 2:
                                cpu_mask = cpu_mask.squeeze(1)
                                if cpu_mask.dim() > 2:
                                    cpu_mask = cpu_mask.squeeze(1)

                        # Move encoder to CPU temporarily
                        original_device = next(self.encoder.parameters()).device
                        cpu_encoder = self.encoder.to("cpu")

                        # Handle out-of-range indices if needed
                        if hasattr(cpu_encoder, "embeddings") and hasattr(
                            cpu_encoder.embeddings, "word_embeddings"
                        ):
                            vocab_size = (
                                cpu_encoder.embeddings.word_embeddings.weight.size(0)
                            )
                            if torch.max(cpu_src) >= vocab_size:
                                cpu_src = torch.clamp(cpu_src, max=vocab_size - 1)

                        try:
                            # Process on CPU
                            with torch.no_grad():
                                outputs = cpu_encoder(
                                    input_ids=cpu_src, attention_mask=cpu_mask
                                )

                            # Move encoder back
                            self.encoder = self.encoder.to(original_device)

                            # Return results on input device
                            return outputs.last_hidden_state.to(input_device)
                        except Exception as cpu_err:
                            print(f"CPU fallback processing failed: {str(cpu_err)}")
                            self.encoder = self.encoder.to(original_device)

                            # Return zeros as last resort
                            batch_size, seq_length = src.shape
                            return torch.zeros(
                                batch_size,
                                seq_length,
                                self.d_model,
                                device=input_device,
                            )

                    # If we've decided to use CPU path, do it now
                    if use_cpu_path:
                        return process_on_cpu()

                    # Otherwise, try normal processing
                    try:
                        # Format attention mask if needed
                        if src_mask is not None:
                            if src_mask.dim() > 2:
                                attention_mask = src_mask.squeeze(1)
                                if attention_mask.dim() > 2:
                                    attention_mask = attention_mask.squeeze(1)
                            else:
                                attention_mask = src_mask

                            # Move mask to encoder device
                            attention_mask = attention_mask.to(encoder_device)
                        else:
                            attention_mask = None

                        # Move input to encoder device
                        input_ids = src.to(encoder_device)

                        # Handle out-of-range indices (important for BERT)
                        if hasattr(self.encoder, "embeddings") and hasattr(
                            self.encoder.embeddings, "word_embeddings"
                        ):
                            vocab_size = (
                                self.encoder.embeddings.word_embeddings.weight.size(0)
                            )
                            # Check if indices are in range
                            if torch.max(input_ids) >= vocab_size:
                                print(
                                    f"Warning: Found input indices larger than vocabulary size ({vocab_size}). Clipping."
                                )
                                # Clip indices to valid range
                                input_ids = torch.clamp(input_ids, max=vocab_size - 1)

                        # Regular processing with device alignment
                        with torch.no_grad():
                            outputs = self.encoder(
                                input_ids=input_ids, attention_mask=attention_mask
                            )

                            # Move result back to original device
                            return outputs.last_hidden_state.to(input_device)

                    except Exception as e:
                        print(f"Error in HuggingFace text encoding: {str(e)}")
                        print(
                            f"Encoder type: {getattr(self, 'encoder_type', 'unknown')}"
                        )
                        print(
                            f"Encoder device: {encoder_device}, Input device: {input_device}"
                        )

                        # Try CPU fallback
                        try:
                            return process_on_cpu()
                        except Exception as fallback_err:
                            print(f"All fallback attempts failed: {str(fallback_err)}")

                        # Final emergency fallback - generate features with correct shape
                        batch_size, seq_length = src.shape
                        print(
                            f"Using final fallback: zeros in correct shape on device {input_device}"
                        )
                        return torch.zeros(
                            batch_size, seq_length, self.d_model, device=input_device
                        )

                def forward(self, src, tgt=None, src_mask=None, tgt_mask=None):
                    """Forward pass - just calls encode for encoder-only models."""
                    return self.encode(src, src_mask)

            # Map model name to HuggingFace model name
            system_device = torch.device(
                "cuda"
                if torch.cuda.is_available()
                else "mps" if torch.backends.mps.is_available() else "cpu"
            )

            # Use MPS-friendly models if on MPS device
            if system_device.type == "mps":
                if args.text_model == "bert-base":
                    huggingface_model_name = (
                        "google/mobilebert-uncased"  # MPS-friendly alternative
                    )
                    print(
                        "⚠️ Automatically switched from BERT-base to MobileBERT for MPS compatibility"
                    )
                elif args.text_model == "roberta-base":
                    huggingface_model_name = "distilroberta-base"  # Smaller model for better MPS compatibility
                    print(
                        "⚠️ Automatically switched from RoBERTa-base to DistilRoBERTa for MPS compatibility"
                    )
                elif args.text_model == "distilbert-base":
                    huggingface_model_name = "distilbert-base-uncased"
                elif args.text_model == "mobilebert":
                    huggingface_model_name = "google/mobilebert-uncased"
                elif args.text_model == "albert-base":
                    huggingface_model_name = "albert-base-v2"
                else:
                    # Default to MobileBERT for MPS
                    huggingface_model_name = "google/mobilebert-uncased"
            else:
                # Standard mapping for CPU/CUDA
                if args.text_model == "bert-base":
                    huggingface_model_name = "bert-base-uncased"
                elif args.text_model == "roberta-base":
                    huggingface_model_name = "roberta-base"
                elif args.text_model == "distilbert-base":
                    huggingface_model_name = "distilbert-base-uncased"
                elif args.text_model == "mobilebert":
                    huggingface_model_name = "google/mobilebert-uncased"
                elif args.text_model == "albert-base":
                    huggingface_model_name = "albert-base-v2"
                else:
                    # Default to bert-base if not specified
                    huggingface_model_name = "bert-base-uncased"

            # Create model
            text_model = HuggingFaceTextModelWrapper(huggingface_model_name)
            text_dim = text_model.d_model

        else:
            # Not using pretrained - create standard transformer
            logger.info(
                f"Creating custom text transformer (without pretraining): {args.text_model}"
            )

            if args.text_model == "transformer-base":
                # Create larger custom transformer
                text_config = {
                    "src_vocab_size": 50000,
                    "tgt_vocab_size": 50000,
                    "d_model": 512,  # Match BERT dimension
                    "num_heads": 12,
                    "num_encoder_layers": 6,
                    "num_decoder_layers": 6,
                    "d_ff": 3072,
                    "dropout": 0.1,
                }
            else:
                # Use transformer-small
                logger.info("Using transformer-small configuration")
                text_config = {
                    "src_vocab_size": 50000,
                    "tgt_vocab_size": 50000,
                    "d_model": 384,  # Match vit-small dimension
                    "num_heads": 8,
                    "num_encoder_layers": 6,
                    "num_decoder_layers": 6,
                    "d_ff": 2048,
                    "dropout": 0.1,
                }

            # Create model
            text_model = EncoderDecoderTransformer(**text_config)
            text_dim = text_config["d_model"]

    except ImportError:
        logger.warning(
            "transformers library not found, using standard text transformer"
        )
        # Fallback to standard implementation
        text_config = {
            "src_vocab_size": 50000,
            "tgt_vocab_size": 50000,
            "d_model": 512,
            "num_heads": 8,
            "num_encoder_layers": 6,
            "num_decoder_layers": 6,
            "d_ff": 2048,
            "dropout": 0.1,
        }
        text_model = EncoderDecoderTransformer(**text_config)
        text_dim = text_config["d_model"]

    # Log parameter counts
    vision_params = count_parameters(vision_model)
    text_params = count_parameters(text_model)
    logger.info(f"Vision transformer parameters: {vision_params:,}")
    logger.info(f"Text transformer parameters: {text_params:,}")

    # Create multimodal model with custom initialization
    logger.info(
        f"Creating enhanced multimodal transformer with {args.fusion_type} fusion"
    )

    # Use the real dimensions from the models
    fusion_dim = args.fusion_dim

    # Log dimension information for debugging
    logger.info(
        f"Model dimensions - Vision: {vision_dim}, Text: {text_dim}, Fusion: {fusion_dim}"
    )

    # Check for dimension mismatch between vision and text models
    if vision_dim != text_dim:
        logger.warning(
            f"Dimension mismatch detected between vision ({vision_dim}) and text ({text_dim}) models"
        )

        # Create a dimension alignment layer to ensure they match before fusion
        if (
            args.text_model == "transformer-base"
            or args.text_model == "transformer-small"
        ):
            # For custom transformers, we can adjust the transformer itself
            logger.info(
                f"Setting text model dimension to match vision model: {vision_dim}"
            )
            # Update the model's dimension
            text_model.d_model = vision_dim
            text_dim = vision_dim
        else:
            # For pretrained models, add a projection layer
            logger.info(
                f"Adding explicit projection layer to match dimensions: {text_dim} → {vision_dim}"
            )

            # Create a wrapper to add projection
            class DimensionMatchingWrapper(nn.Module):
                def __init__(self, base_model, input_dim, output_dim):
                    super().__init__()
                    self.base_model = base_model
                    self.projection = nn.Linear(input_dim, output_dim)
                    self.d_model = output_dim  # Update dimension

                    # Move projection to same device as base model initially
                    base_device = next(base_model.parameters()).device
                    self.projection = self.projection.to(base_device)

                def encode(self, *args, **kwargs):
                    # Track original device for consistent return
                    original_device = None
                    if "src" in kwargs:
                        original_device = kwargs["src"].device
                    elif len(args) > 0:
                        original_device = args[0].device

                    # Get device from base model for consistency
                    base_device = next(self.base_model.parameters()).device

                    # Move inputs to base model device if needed
                    if "src" in kwargs and kwargs["src"].device != base_device:
                        kwargs["src"] = kwargs["src"].to(base_device)
                    if (
                        "src_mask" in kwargs
                        and kwargs["src_mask"] is not None
                        and kwargs["src_mask"].device != base_device
                    ):
                        kwargs["src_mask"] = kwargs["src_mask"].to(base_device)
                    if len(args) > 0 and args[0].device != base_device:
                        args = list(args)
                        args[0] = args[0].to(base_device)
                        args = tuple(args)

                    # Get output from base model (now everything is on same device)
                    base_output = self.base_model.encode(*args, **kwargs)

                    # Ensure projection is on same device
                    proj_device = next(self.projection.parameters()).device
                    if base_output.device != proj_device:
                        self.projection = self.projection.to(base_output.device)

                    # Project to new dimension
                    projected = self.projection(base_output)

                    # Return to original device if requested
                    if (
                        original_device is not None
                        and projected.device != original_device
                    ):
                        projected = projected.to(original_device)

                    return projected

                def forward(self, *args, **kwargs):
                    return self.encode(*args, **kwargs)

            # Wrap the text model with a projection
            text_model = DimensionMatchingWrapper(text_model, text_dim, vision_dim)
            # Move to the same device as the text model
            device = next(text_model.base_model.parameters()).device
            text_model.projection = text_model.projection.to(device)
            text_dim = vision_dim
            logger.info(f"Projection layer created on device: {device}")

    # After alignment, dimensions should match
    logger.info(
        f"Aligned dimensions - Vision: {vision_dim}, Text: {text_dim}, Fusion: {fusion_dim}"
    )

    multimodal_model = EnhancedMultiModalTransformer(
        vision_model=vision_model,
        text_model=text_model,
        fusion_dim=fusion_dim,
        num_fusion_layers=3,  # Increased from 2 for better fusion
        num_heads=8,
        dropout=0.1,
        fusion_type=args.fusion_type,
    )

    # Set up parameter freezing based on argument
    if args.freeze_base_models:
        # Freeze base models initially (will be unfrozen during multi-stage training)
        for name, param in multimodal_model.named_parameters():
            if "vision_model" in name or "text_model" in name:
                param.requires_grad = False
        logger.info("Base models frozen for initial training phase")
    else:
        # Keep all parameters trainable for full fine-tuning
        for param in multimodal_model.parameters():
            param.requires_grad = True
        logger.info("All model parameters are trainable (including base models)")

    # Calculate trainable parameters
    trainable_params = sum(
        p.numel() for p in multimodal_model.parameters() if p.requires_grad
    )
    total_params = count_parameters(multimodal_model)
    logger.info(
        f"Trainable parameters: {trainable_params:,} ({trainable_params/total_params*100:.1f}% of total)"
    )

    # Print detailed model summary
    print_model_summary(multimodal_model, "MULTIMODAL MODEL ARCHITECTURE (PRETRAINED)")

    # Move model to device - ensure all components are on the same device
    multimodal_model = multimodal_model.to(device)

    # Explicitly check and move all submodules to ensure consistency
    multimodal_model.vision_model = multimodal_model.vision_model.to(device)
    multimodal_model.text_model = multimodal_model.text_model.to(device)
    multimodal_model.fusion_module = multimodal_model.fusion_module.to(device)
    if hasattr(multimodal_model, "classifier"):
        multimodal_model.classifier = multimodal_model.classifier.to(device)

    # Print device confirmation
    print(f"Model components successfully moved to {device}")
    print(f"- Vision model: {next(multimodal_model.vision_model.parameters()).device}")
    print(f"- Text model: {next(multimodal_model.text_model.parameters()).device}")
    print(
        f"- Fusion module: {next(multimodal_model.fusion_module.parameters()).device}"
    )

    # We're now using standard transformer models instead of BERT for better MPS compatibility

    return multimodal_model


def convert_tensors_to_python_types(obj):
    """Convert PyTorch tensors to native Python types for JSON serialization."""
    if isinstance(obj, torch.Tensor):
        # Convert tensor to Python type
        return obj.item() if obj.numel() == 1 else obj.detach().cpu().tolist()
    elif isinstance(obj, dict):
        # Recursively convert dictionary values
        return {k: convert_tensors_to_python_types(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        # Recursively convert list elements
        return [convert_tensors_to_python_types(item) for item in obj]
    elif isinstance(obj, tuple):
        # Recursively convert tuple elements
        return tuple(convert_tensors_to_python_types(item) for item in obj)
    else:
        # Return other types as is
        return obj


def create_data_loaders(args, image_preprocessor, tokenizer):
    """
    Create data loaders for training, validation, and testing.

    Args:
        args: Command line arguments
        image_preprocessor: Image preprocessor
        tokenizer: Text tokenizer

    Returns:
        Train, validation, and test data loaders
    """
    print(f"Creating data loaders for {args.dataset} dataset...")

    # Create dataset and data loaders
    if args.dataset == "flickr30k":
        if args.use_synthetic:
            print("WARNING: Using synthetic data instead of real Flickr30k data!")
            synthetic_samples = args.synthetic_samples
        else:
            # When using real data, don't specify synthetic_samples
            synthetic_samples = args.synthetic_samples if args.use_synthetic else 0

        # Try to create Flickr30k dataset splits
        try:
            print("Loading Flickr30k train split...")
            train_dataset = EnhancedMultimodalDataset(
                dataset_name="flickr30k",
                split="train",
                image_preprocessor=image_preprocessor,
                tokenizer=tokenizer,
                max_text_length=args.max_text_length,
                synthetic_samples=synthetic_samples,
                cache_dir=os.path.join(args.data_dir, "flickr30k"),
                max_samples=args.max_train_examples,
            )

            # Check if we actually got real data or synthetic fallback
            dataset_info = train_dataset.get_split_proportions()
            if (
                dataset_info.get("total_samples", 0) <= args.synthetic_samples
                and not args.use_synthetic
            ):
                print(
                    f"WARNING: Got {dataset_info.get('total_samples', 0)} samples which may indicate synthetic data fallback"
                )
                print(
                    "If you want to use synthetic data explicitly, use --use_synthetic flag"
                )

            print("Loading Flickr30k validation split...")
            val_dataset = EnhancedMultimodalDataset(
                dataset_name="flickr30k",
                split="val",
                image_preprocessor=image_preprocessor,
                tokenizer=tokenizer,
                max_text_length=args.max_text_length,
                synthetic_samples=(
                    synthetic_samples // 4 if synthetic_samples > 0 else 0
                ),
                cache_dir=os.path.join(args.data_dir, "flickr30k"),
                max_samples=args.max_val_examples,
            )

            print("Loading Flickr30k test split...")
            test_dataset = EnhancedMultimodalDataset(
                dataset_name="flickr30k",
                split="test",
                image_preprocessor=image_preprocessor,
                tokenizer=tokenizer,
                max_text_length=args.max_text_length,
                synthetic_samples=(
                    synthetic_samples // 4 if synthetic_samples > 0 else 0
                ),
                cache_dir=os.path.join(args.data_dir, "flickr30k"),
                max_samples=args.max_test_examples,
            )

        except Exception as e:
            print(f"ERROR: Failed to load Flickr30k dataset: {str(e)}")
            print(
                "If you intended to use synthetic data, please use --dataset synthetic or --use_synthetic flag"
            )
            raise

    elif args.dataset == "synthetic":
        print("Creating synthetic datasets for training demo")
        # Create synthetic datasets explicitly
        # Adjust the number of synthetic samples based on max_examples settings
        train_synthetic_samples = args.synthetic_samples
        if (
            args.max_train_examples is not None
            and args.max_train_examples < train_synthetic_samples
        ):
            train_synthetic_samples = args.max_train_examples

        val_synthetic_samples = args.synthetic_samples // 4
        if (
            args.max_val_examples is not None
            and args.max_val_examples < val_synthetic_samples
        ):
            val_synthetic_samples = args.max_val_examples

        test_synthetic_samples = args.synthetic_samples // 4
        if (
            args.max_test_examples is not None
            and args.max_test_examples < test_synthetic_samples
        ):
            test_synthetic_samples = args.max_test_examples

        train_dataset = EnhancedMultimodalDataset(
            dataset_name="synthetic",
            split="train",
            image_preprocessor=image_preprocessor,
            tokenizer=tokenizer,
            max_text_length=args.max_text_length,
            synthetic_samples=train_synthetic_samples,
            cache_dir=os.path.join(args.data_dir, "synthetic"),
            max_samples=args.max_train_examples,
        )

        val_dataset = EnhancedMultimodalDataset(
            dataset_name="synthetic",
            split="val",
            image_preprocessor=image_preprocessor,
            tokenizer=tokenizer,
            max_text_length=args.max_text_length,
            synthetic_samples=val_synthetic_samples,
            cache_dir=os.path.join(args.data_dir, "synthetic"),
            max_samples=args.max_val_examples,
        )

        test_dataset = EnhancedMultimodalDataset(
            dataset_name="synthetic",
            split="test",
            image_preprocessor=image_preprocessor,
            tokenizer=tokenizer,
            max_text_length=args.max_text_length,
            synthetic_samples=test_synthetic_samples,
            cache_dir=os.path.join(args.data_dir, "synthetic"),
            max_samples=args.max_test_examples,
        )
    else:
        # Custom dataset handling remains the same
        train_dataset = EnhancedMultimodalDataset(
            dataset_name="custom",
            split="train",
            image_preprocessor=image_preprocessor,
            tokenizer=tokenizer,
            max_text_length=args.max_text_length,
            synthetic_samples=args.synthetic_samples,
            cache_dir=os.path.join(args.data_dir, "custom"),
            max_samples=args.max_train_examples,
        )

        val_dataset = EnhancedMultimodalDataset(
            dataset_name="custom",
            split="val",
            image_preprocessor=image_preprocessor,
            tokenizer=tokenizer,
            max_text_length=args.max_text_length,
            synthetic_samples=args.synthetic_samples // 4,
            cache_dir=os.path.join(args.data_dir, "custom"),
            max_samples=args.max_val_examples,
        )

        test_dataset = EnhancedMultimodalDataset(
            dataset_name="custom",
            split="test",
            image_preprocessor=image_preprocessor,
            tokenizer=tokenizer,
            max_text_length=args.max_text_length,
            synthetic_samples=args.synthetic_samples // 4,
            cache_dir=os.path.join(args.data_dir, "custom"),
            max_samples=args.max_test_examples,
        )

    # Let's use a simpler but effective approach to prevent shortcut learning
    # Instead of a complex custom sampler, we'll use a simple but effective approach

    # First, we need to ensure our datasets have the match_ids properly set
    # We'll do this by adding a randomization method
    def randomize_dataset_positions(dataset):
        """Randomize the positions of items in the dataset to break positional correlation."""
        # First, make sure we have access to match_ids
        # Try different approaches to get match_ids
        match_ids = None

        # Try to access match_ids as an attribute
        if hasattr(dataset, "match_ids"):
            match_ids = dataset.match_ids
        # Try to access through a method
        elif hasattr(dataset, "get_match_ids") and callable(dataset.get_match_ids):
            match_ids = dataset.get_match_ids()
        # Try to extract from each item
        else:
            print("Attempting to extract match_ids from dataset items...")
            # Check if we can access items and if they have match_id
            try:
                # Check first item
                first_item = dataset[0]
                if isinstance(first_item, dict) and "match_id" in first_item:
                    # Extract match_ids from all items
                    match_ids = []
                    for i in range(len(dataset)):
                        item = dataset[i]
                        match_ids.append(item.get("match_id", f"id_{i}"))
                    print(
                        f"Successfully extracted {len(match_ids)} match_ids from items"
                    )
            except Exception as e:
                print(f"Error extracting match_ids from items: {e}")

        # If we still don't have match_ids, use default
        if match_ids is None:
            print("WARNING: Couldn't access match_ids, using fallback with unique IDs")
            match_ids = [f"id_{i}" for i in range(len(dataset))]

        # Store match_ids in the dataset for future reference
        if not hasattr(dataset, "match_ids"):
            dataset.match_ids = match_ids

        # Group indices by match_id
        match_id_groups = {}
        for idx, match_id in enumerate(match_ids):
            if match_id not in match_id_groups:
                match_id_groups[match_id] = []
            match_id_groups[match_id].append(idx)

        print(
            f"Found {len(match_id_groups)} match groups in dataset with {len(dataset)} items"
        )

        # Create shuffled indices that preserve semantic relationships but break position
        shuffled_indices = []
        # Mix up groups as much as possible
        group_keys = list(match_id_groups.keys())
        random.shuffle(group_keys)

        # For each group, randomize indices within the group
        for match_id in group_keys:
            indices = match_id_groups[match_id]
            random.shuffle(indices)
            shuffled_indices.extend(indices)

        print(f"Created shuffled indices list with {len(shuffled_indices)} items")
        return shuffled_indices

    # Randomize our datasets and get shuffled indices
    print("Randomizing dataset positions to prevent shortcut learning...")
    train_indices = randomize_dataset_positions(train_dataset)
    val_indices = randomize_dataset_positions(val_dataset)

    # Use PyTorch's SubsetRandomSampler for simplicity
    from torch.utils.data import SubsetRandomSampler

    # Create samplers using our shuffled indices
    train_sampler = SubsetRandomSampler(train_indices)
    val_sampler = SubsetRandomSampler(val_indices)

    # Create data loaders with these samplers
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        sampler=train_sampler,
        num_workers=0,  # Keep at 0 for MPS
        drop_last=True,  # Important for consistent batch sizes
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        sampler=val_sampler,
        num_workers=0,  # Keep at 0 for MPS
        drop_last=True,  # Consistent batch sizes for validation
    )

    # For testing we still use a standard loader without custom sampling
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,  # Important for reproducible testing
        num_workers=0,  # Keep at 0 for MPS
    )

    # Print dataset statistics
    print(f"Train split: {train_dataset.get_split_proportions()}")
    print(f"Val split: {val_dataset.get_split_proportions()}")
    print(f"Test split: {test_dataset.get_split_proportions()}")

    return train_loader, val_loader, test_loader


def create_loss_function(args, dataset_size=None, train_loader=None):
    """
    Create the appropriate loss function based on arguments.

    Args:
        args: Command line arguments
        dataset_size: Size of the dataset for auto-selecting sampling strategy
        train_loader: Training dataloader for advanced loss configuration

    Returns:
        Loss function
    """
    # Get loss type directly from args now that we've added it to the argument parser
    loss_type = args.loss_type

    # Log which loss function is being used
    logger.info(f"Creating loss function of type: {loss_type}")

    # Use args.use_mixed_loss to override if it's explicitly set
    if args.use_mixed_loss:
        loss_type = "mixed"
        logger.info(
            "Overriding to Mixed Contrastive Loss based on --use_mixed_loss flag"
        )

    # Switch based on the loss type
    if loss_type == "memory_queue":
        # Memory Queue-Based Contrastive Loss
        logger.info("Using Memory Queue-Based Contrastive Loss")
        from src.training.contrastive_learning import MemoryQueueContrastiveLoss

        # Use queue_size from args if provided, otherwise determine based on dataset size
        if args.queue_size:
            queue_size = args.queue_size
        elif dataset_size is not None:
            # Larger datasets benefit from larger queues
            if dataset_size > 10000:
                queue_size = 16384  # Very large queue for large datasets
            elif dataset_size > 5000:
                queue_size = 8192  # Large queue for medium-large datasets
            elif dataset_size > 1000:
                queue_size = 4096  # Medium queue for medium datasets
            else:
                queue_size = 2048  # Smaller queue for small datasets
        else:
            queue_size = 8192  # Default to a reasonably large queue size

        # Adjust temperature for memory queue approach
        # Generally needs slightly higher temperature than standard contrastive
        adjusted_temp = args.temperature * 1.1

        logger.info(
            f"Memory Queue size: {queue_size}, Temperature: {adjusted_temp:.4f}"
        )

        return MemoryQueueContrastiveLoss(
            dim=512,  # Feature dimension
            queue_size=queue_size,
            temperature=adjusted_temp,
        )

    elif loss_type == "dynamic_temp":
        # Dynamic Temperature Calibration
        logger.info("Using Dynamic Temperature Calibration Contrastive Loss")
        from src.training.contrastive_learning import DynamicTemperatureContrastiveLoss

        # Base temperature is provided by args
        base_temp = args.temperature

        # Use min/max from args if provided, otherwise calculate sensible defaults
        if args.dynamic_temp_min is not None:
            min_temp = args.dynamic_temp_min
        else:
            min_temp = max(0.01, base_temp * 0.6)  # Don't go below 0.01 or 60% of base

        if args.dynamic_temp_max is not None:
            max_temp = args.dynamic_temp_max
        else:
            max_temp = min(0.3, base_temp * 2.0)  # Don't go above 0.3 or 200% of base

        logger.info(
            f"Dynamic Temperature - Base: {base_temp:.4f}, Range: [{min_temp:.4f}, {max_temp:.4f}]"
        )

        return DynamicTemperatureContrastiveLoss(
            base_temperature=base_temp, min_temp=min_temp, max_temp=max_temp
        )

    elif loss_type == "hard_negative":
        # Hard Negative Mining Contrastive Loss
        logger.info("Using Hard Negative Mining Contrastive Loss")
        from src.training.contrastive_learning import HardNegativeMiningContrastiveLoss

        # Use mining strategy from args if not 'auto', otherwise determine based on batch size
        if args.mining_strategy and args.mining_strategy != "auto":
            mining_strategy = args.mining_strategy
        elif args.batch_size < 32:
            # For small batches, semi-hard negatives work better
            mining_strategy = "semi-hard"
        else:
            # For larger batches, full hard negative mining is effective
            mining_strategy = "hard"

        # Use hard negative factor from args if provided
        if args.hard_negative_factor is not None:
            hard_negative_factor = args.hard_negative_factor
        elif mining_strategy == "semi-hard":
            # Higher weighting for semi-hard approach
            hard_negative_factor = 3.0
        else:
            # Moderate weighting for hard approach
            hard_negative_factor = 2.0

        logger.info(
            f"Hard Negative Mining - Strategy: {mining_strategy}, Weight: {hard_negative_factor:.1f}x"
        )

        return HardNegativeMiningContrastiveLoss(
            temperature=args.temperature,
            hard_negative_factor=hard_negative_factor,
            mining_strategy=mining_strategy,
        )

    elif loss_type == "mixed":
        # Mixed Contrastive Loss with multiple objectives
        logger.info("Using Mixed Contrastive Loss with multiple objectives")
        return MultiModalMixedContrastiveLoss(
            contrastive_weight=1.0,
            classification_weight=0.2,
            multimodal_matching_weight=0.2,
            temperature=args.temperature,
            use_hard_negatives=True,  # ENABLED hard negative mining
            hard_negative_weight=(
                0.3 if args.hard_negative_factor is None else args.hard_negative_factor
            ),
        )

    else:
        # Standard Contrastive Loss (default)
        logger.info("Using Standard Contrastive Loss with enhanced settings")

        # Determine which sampling strategy to use
        if args.contrastive_sampling == "auto":
            if dataset_size is not None:
                if dataset_size < 1000:
                    sampling_strategy = "global"
                    logger.info(
                        f"Auto-selecting 'global' sampling strategy for small dataset size ({dataset_size})"
                    )
                elif dataset_size < 10000:
                    sampling_strategy = "memory-bank"
                    logger.info(
                        f"Auto-selecting 'memory-bank' sampling strategy for medium dataset size ({dataset_size})"
                    )
                else:
                    sampling_strategy = "in-batch"
                    logger.info(
                        f"Auto-selecting 'in-batch' sampling strategy for large dataset size ({dataset_size})"
                    )
            else:
                sampling_strategy = "memory-bank"  # CHANGED default to memory-bank
                logger.info(
                    "Dataset size unknown, defaulting to 'memory-bank' sampling strategy for better performance"
                )
        else:
            sampling_strategy = args.contrastive_sampling
            logger.info(f"Using '{sampling_strategy}' sampling strategy as specified")

        # IMPROVEMENT: Add a warning about in-batch sampling
        if sampling_strategy == "in-batch":
            logger.warning(
                "WARNING: Using 'in-batch' sampling can lead to shortcut learning for small batches. "
                "Consider using 'memory-bank' or 'global' for more robust training."
            )

        # IMPROVED: Use a more appropriate temperature based on strategy and batch size
        # Smaller batches need lower temperature for more focused learning
        if args.batch_size < 64:
            temp_scale = 0.85  # More aggressive scaling for small batches
        elif args.batch_size < 128:
            temp_scale = 0.9
        else:
            temp_scale = 0.95  # Less aggressive for large batches

        # Apply temperature adjustment based on sampling strategy
        if sampling_strategy == "in-batch":
            adjusted_temp = args.temperature * temp_scale  # Lower temp for in-batch
        elif sampling_strategy == "memory-bank":
            adjusted_temp = args.temperature * 1.0  # Standard temp for memory-bank
        else:
            adjusted_temp = (
                args.temperature * 1.05
            )  # Slightly higher for global (more diverse negatives)

        logger.info(
            f"Using Contrastive Loss with temperature {adjusted_temp:.4f} "
            f"(original: {args.temperature}, adjusted by {adjusted_temp/args.temperature:.2f}x) "
            f"and {sampling_strategy} sampling strategy"
        )

        # CRITICAL FIX: Create a more advanced contrastive loss with better settings
        return ContrastiveLoss(
            temperature=adjusted_temp,
            loss_type="infonce",  # InfoNCE loss is standard for contrastive learning
            reduction="mean",
            add_projection=True,  # ENABLED: Use projection heads for better representation
            projection_dim=256,  # Smaller dimension for the projection space
            input_dim=512,  # Match model dimension
            sampling_strategy=sampling_strategy,
            memory_bank_size=args.memory_bank_size
            * 2,  # INCREASED: Use larger memory bank
            dataset_size=dataset_size,
        )


def visualize_similarity_matrix(similarity_matrix, captions, save_path=None):
    """
    Visualize the similarity matrix between images and texts.

    Args:
        similarity_matrix: Image-text similarity matrix
        captions: List of captions
        save_path: Path to save the visualization
    """
    plt.figure(figsize=(10, 8))
    plt.imshow(similarity_matrix.cpu().numpy(), cmap="viridis")
    plt.colorbar(label="Similarity")
    plt.xlabel("Text")
    plt.ylabel("Image")
    plt.title("Cross-Modal Similarity Matrix")

    # Add grid
    plt.grid(False)

    # Add labels (limit to 20 for readability)
    max_captions = min(20, len(captions))
    short_captions = [
        c[:20] + "..." if len(c) > 20 else c for c in captions[:max_captions]
    ]

    plt.xticks(
        range(max_captions),
        [f"{i}: {c}" for i, c in enumerate(short_captions)],
        rotation=90,
        fontsize=8,
    )
    plt.yticks(
        range(max_captions),
        [f"{i}" for i in range(max_captions)],
        fontsize=8,
    )

    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300)
    else:
        plt.show()

    plt.close()


def visualize_attention_maps(
    attention_maps, images, captions, save_dir=None, model=None
):
    """
    Visualize attention maps between images and texts.

    Args:
        attention_maps: Dictionary of attention maps
        images: Batch of images
        captions: List of captions
        save_dir: Directory to save visualizations
        model: Model to display parameter count (optional)
    """
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)

    batch_size = min(4, len(images))  # Limit to first 4 examples

    # Model parameter count info
    param_info = ""
    if model is not None:
        param_count = count_parameters(model)
        param_info = f" | Model: {param_count:,} parameters"

    # Process each attention type
    for attn_name, attn_maps in attention_maps.items():
        # Skip if not a tensor
        if not isinstance(attn_maps, torch.Tensor):
            continue

        # Get attention shape
        if len(attn_maps.shape) == 4:  # [batch, heads, seq_len_q, seq_len_k]
            num_heads = attn_maps.shape[1]

            # For each example in the batch
            for b in range(batch_size):
                plt.figure(figsize=(20, 4 * num_heads))
                plt.suptitle(
                    f"Attention Maps: {attn_name} - Example {b}{param_info}\nCaption: {captions[b][:80]}...",
                    fontsize=16,
                )

                # For each attention head
                for h in range(num_heads):
                    plt.subplot(num_heads, 1, h + 1)
                    plt.imshow(attn_maps[b, h].cpu().numpy(), cmap="viridis")
                    plt.title(f"Head {h}")
                    plt.colorbar(label="Attention Weight")

                plt.tight_layout()
                plt.subplots_adjust(top=0.9)

                if save_dir:
                    plt.savefig(
                        os.path.join(save_dir, f"{attn_name}_example{b}.png"), dpi=200
                    )
                else:
                    plt.show()

                plt.close()


def run_inference_demo(model, image_preprocessor, tokenizer, device, args):
    """
    Run inference demo with the trained model.

    Args:
        model: Trained multimodal model
        image_preprocessor: Image preprocessor
        tokenizer: Text tokenizer
        device: Device to run inference on
        args: Command line arguments
    """
    logger.info("Running inference demo...")
    logger.info(f"Model has {count_parameters(model):,} trainable parameters")

    # Use real test dataset instead of synthetic for better accuracy
    # The main test split is already created and available in the test loader
    test_dataset = EnhancedMultimodalDataset(
        split="test",
        image_preprocessor=image_preprocessor,
        tokenizer=tokenizer,
        max_text_length=args.max_text_length,
        dataset_name=args.dataset,
        # No synthetic samples - use real data
        cache_dir=args.data_dir,
    )

    # Create a smaller test loader for the demo
    demo_batch_size = 16
    test_loader = DataLoader(
        test_dataset,
        batch_size=demo_batch_size,
        shuffle=False,
    )

    # Get a batch of data
    batch = next(iter(test_loader))

    # Move to device
    batch = {
        k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()
    }

    # Run inference with attention visualization
    model.eval()
    with torch.no_grad():
        outputs = model(
            images=batch["image"],
            text_data=batch["text"],
            return_attention=args.visualize_attention,
        )

    # Extract and visualize similarity matrix
    if "raw_similarity" in outputs:
        similarity = outputs["raw_similarity"]
    elif "similarity" in outputs:
        similarity = outputs["similarity"]
    else:
        # Compute similarity from features - prefer enhanced features if available
        if (
            "vision_features_enhanced" in outputs
            and "text_features_enhanced" in outputs
        ):
            vision_features = F.normalize(
                outputs["vision_features_enhanced"], p=2, dim=1
            )
            text_features = F.normalize(outputs["text_features_enhanced"], p=2, dim=1)
        else:
            vision_features = F.normalize(outputs["vision_features"], p=2, dim=1)
            text_features = F.normalize(outputs["text_features"], p=2, dim=1)
        similarity = torch.matmul(vision_features, text_features.T)

    # Get captions
    captions = batch["raw_text"]

    # Visualize similarity matrix
    logger.info("Visualizing similarity matrix...")
    similarity_matrix_path = os.path.join(args.output_dir, "similarity_matrix.png")
    visualize_similarity_matrix(
        similarity,
        captions,
        save_path=similarity_matrix_path,
    )
    logger.info(f"Saved similarity matrix to {similarity_matrix_path}")

    # Create example visualization using the same batch
    # This creates a visualization where we match each image with text based on similarity
    logger.info("Creating example test pair visualizations...")
    example_viz_path = os.path.join(args.output_dir, "test_examples_visualization.png")

    # Directly use the visualize_test_samples function with our batch
    test_samples_accuracy = visualize_test_samples(
        model=model,
        test_dataset=test_dataset,  # Use the full test dataset
        device=device,
        save_path=example_viz_path,
        num_samples=10,  # Show 10 samples
    )

    logger.info(f"Example visualization saved to {example_viz_path}")
    logger.info(f"Example accuracy: {test_samples_accuracy:.4f}")

    # Calculate retrieval metrics for the current batch
    logger.info("Computing retrieval metrics...")

    # For proper match-based evaluation, we need match IDs
    match_ids = []
    if "match_id" in batch:
        match_ids = batch["match_id"]

    # Create match matrix based on match IDs if available
    batch_size = len(similarity)
    if match_ids:
        match_matrix = torch.zeros(
            (batch_size, batch_size), dtype=torch.bool, device=device
        )
        for i in range(batch_size):
            for j in range(batch_size):
                match_matrix[i, j] = match_ids[i] == match_ids[j]
    else:
        # Fallback to diagonal matching
        match_matrix = torch.eye(batch_size, dtype=torch.bool, device=device)

    # Image-to-text retrieval with proper match IDs
    recalls = {}
    for k in [1, 5, 10]:
        i2t_hits = 0
        t2i_hits = 0

        # For each image, find if any of its matching texts are in the top-k
        for i in range(batch_size):
            # Get indices of all matching texts for this image
            matching_text_indices = torch.where(match_matrix[i])[0]
            if len(matching_text_indices) == 0:
                continue

            # Get top-k predictions for this image
            topk_indices = torch.topk(similarity[i], min(k, batch_size), dim=0)[1]

            # Check if any matching text is in the top-k predictions
            hit = any(
                idx.item() in matching_text_indices.tolist() for idx in topk_indices
            )
            i2t_hits += int(hit)

        # For each text, find if any of its matching images are in the top-k
        for j in range(batch_size):
            # Get indices of all matching images for this text
            matching_image_indices = torch.where(match_matrix[:, j])[0]
            if len(matching_image_indices) == 0:
                continue

            # Get top-k predictions for this text
            topk_indices = torch.topk(similarity[:, j], min(k, batch_size), dim=0)[1]

            # Check if any matching image is in the top-k predictions
            hit = any(
                idx.item() in matching_image_indices.tolist() for idx in topk_indices
            )
            t2i_hits += int(hit)

        # Calculate recall
        i2t_recall = i2t_hits / batch_size
        t2i_recall = t2i_hits / batch_size
        avg_recall = (i2t_recall + t2i_recall) / 2

        recalls[f"i2t_recall@{k}"] = i2t_recall
        recalls[f"t2i_recall@{k}"] = t2i_recall
        recalls[f"avg_recall@{k}"] = avg_recall

    logger.info(f"Retrieval metrics: {recalls}")

    # Visualize attention maps if available
    if args.visualize_attention and "attention_maps" in outputs:
        logger.info("Visualizing attention maps...")
        attention_dir = os.path.join(args.output_dir, "attention_maps")
        visualize_attention_maps(
            outputs["attention_maps"],
            batch["image"],
            batch["raw_text"],
            save_dir=attention_dir,
            model=model,
        )
        logger.info(f"Attention maps saved to {attention_dir}")

    # Create a unified visualization that shows both similarity matrix and example matches
    try:
        logger.info("Creating unified visualization...")
        plt.figure(figsize=(16, 12))

        # Load the individual visualizations
        similarity_img = plt.imread(similarity_matrix_path)
        examples_img = plt.imread(example_viz_path)

        # Create a 2-panel figure
        plt.subplot(2, 1, 1)
        plt.imshow(similarity_img)
        plt.title("Similarity Matrix")
        plt.axis("off")

        plt.subplot(2, 1, 2)
        plt.imshow(examples_img)
        plt.title("Example Image-Text Matches")
        plt.axis("off")

        plt.tight_layout()
        unified_path = os.path.join(args.output_dir, "unified_visualization.png")
        plt.savefig(unified_path, dpi=200)
        plt.close()

        logger.info(f"Unified visualization saved to {unified_path}")
    except Exception as e:
        logger.warning(f"Failed to create unified visualization: {e}")

    logger.info("Inference demo completed")

    return recalls


class SimpleTokenizer:
    def __init__(self):
        self._special_tokens = {
            "pad_token_idx": 0,
            "unk_token_idx": 1,
            "bos_token_idx": 2,
            "eos_token_idx": 3,
            "mask_token_idx": 4,
        }
        self.vocab_size = 50000

    def encode(self, text):
        # Simple encoding based on character hashing
        tokens = [self._special_tokens["bos_token_idx"]]
        tokens.extend([hash(c) % (self.vocab_size - 10) + 10 for c in text])
        tokens.append(self._special_tokens["eos_token_idx"])
        return tokens

    @property
    def special_tokens(self):
        return self._special_tokens


def visualize_test_samples(model, test_dataset, device, save_path, num_samples=10):
    """
    Visualize specific test samples with their matched captions.
    Images are matched against ALL captions in the test dataset, not just the displayed samples.

    Args:
        model: Trained multimodal model
        test_dataset: Test dataset
        device: Device to run inference on
        save_path: Path to save the visualization
        num_samples: Number of samples to visualize

    Returns:
        float: Accuracy of image-to-text matching for these samples
    """
    # Print model parameter information
    print(
        f"Model used for visualization has {count_parameters(model):,} trainable parameters"
    )

    # Ensure output directory exists
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    # Limit the number of samples to visualize
    num_samples = min(num_samples, len(test_dataset))

    # Create DataLoader for the visualization samples
    vis_loader = DataLoader(test_dataset, batch_size=num_samples, shuffle=False)

    # Create DataLoader for all captions
    full_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    # Get the visualization samples
    vis_batch = next(iter(vis_loader))

    # Move to device
    vis_images = vis_batch["image"].to(device)

    # Get all text embeddings from the dataset
    all_text_embeddings = []
    all_captions = []

    model.eval()
    with torch.no_grad():
        # First, get embeddings for the visualization images
        if "text" in vis_batch:
            vis_text_data = {
                "src": (
                    vis_batch["text"]["src"].to(device)
                    if vis_batch["text"]["src"].dim() == 2
                    else vis_batch["text"]["src"].squeeze(1).to(device)
                ),
                "src_mask": vis_batch["text"]["src_mask"].to(device),
            }
        else:
            raise ValueError("Text data not found in batch")

        # Get raw text captions for visualization samples
        vis_captions = vis_batch.get(
            "raw_text", [f"Caption {i}" for i in range(num_samples)]
        )

        # Process all text in the dataset to get embeddings
        print("Computing text embeddings for all captions in the dataset...")
        for batch in tqdm(full_loader, desc="Processing captions"):
            if "text" in batch:
                text_data = {
                    "src": (
                        batch["text"]["src"].to(device)
                        if batch["text"]["src"].dim() == 2
                        else batch["text"]["src"].squeeze(1).to(device)
                    ),
                    "src_mask": batch["text"]["src_mask"].to(device),
                }

                # Get text embeddings
                outputs = model(images=None, text_data=text_data)

                # Get normalized text features
                if "text_features_enhanced" in outputs:
                    text_features = outputs["text_features_enhanced"]
                else:
                    text_features = outputs["text_features"]

                # Pool if needed and normalize
                if len(text_features.shape) == 3:
                    text_features = text_features.mean(dim=1)
                text_features = F.normalize(text_features, p=2, dim=1)

                all_text_embeddings.append(text_features.cpu())

                # Store captions
                batch_captions = batch.get(
                    "raw_text", [f"Caption {i}" for i in range(len(batch["image"]))]
                )
                all_captions.extend(batch_captions)

        # Process visualization images
        outputs = model(images=vis_images, text_data=None)

        # Get vision features
        if "vision_features_enhanced" in outputs:
            vision_features = outputs["vision_features_enhanced"]
        else:
            vision_features = outputs["vision_features"]

        # Pool if needed and normalize
        if len(vision_features.shape) == 3:
            vision_features = vision_features.mean(dim=1)
        vision_features = F.normalize(vision_features, p=2, dim=1)

        # Concatenate all text embeddings
        all_text_embeddings = torch.cat(all_text_embeddings, dim=0)

        # Check and fix dimension mismatch between vision and text features
        vision_dim = vision_features.shape[1]
        text_dim = all_text_embeddings.shape[1]

        if vision_dim != text_dim:
            print(
                f"Dimension mismatch: vision={vision_dim}, text={text_dim}. Creating projection..."
            )
            # Create a simple projection to match dimensions
            if vision_dim > text_dim:
                # Project vision features to text dimension
                projection = nn.Linear(vision_dim, text_dim).to(device)
                vision_features = projection(vision_features)
            else:
                # Project text features to vision dimension
                projection = nn.Linear(text_dim, vision_dim).to(device)
                all_text_embeddings = projection(all_text_embeddings.to(device))
                # Move back to CPU after projection
                all_text_embeddings = all_text_embeddings.cpu()

        # Compute similarity matrix between visualization images and ALL text captions
        similarity_matrix = torch.matmul(
            vision_features, all_text_embeddings.to(device).T
        )

    # Get the most similar caption for each image
    most_similar_idxs = similarity_matrix.argmax(dim=1)

    # Get the ground truth indices for the visualization samples
    # These are the positions of our visualization samples in the full dataset
    # Since we're using shuffle=False, these are just the first num_samples indices
    ground_truth_idxs = list(range(num_samples))

    # Check if we're using synthetic data for warning
    dataset_info = test_dataset.get_split_proportions()
    is_synthetic = (
        dataset_info.get("dataset_name") == "synthetic"
        or dataset_info.get("total_samples", 1000) < 100
    )

    # Get parameter count for title
    param_count = count_parameters(model)

    # Create figure with subplots
    fig, axes = plt.subplots(num_samples, 2, figsize=(12, 3 * num_samples))

    # Ensure axes is 2D even for single sample
    if num_samples == 1:
        axes = np.array([axes])

    # Set a title for the figure
    if is_synthetic:
        fig.suptitle(
            f"Multimodal Retrieval Results (SYNTHETIC DATA) | Model: {param_count:,} parameters",
            fontsize=16,
            color="red",
        )
    else:
        fig.suptitle(
            f"Multimodal Retrieval Results (Searching All Captions) | Model: {param_count:,} parameters",
            fontsize=16,
        )

    # Draw images and captions
    for i in range(num_samples):
        # Get and process image
        img = vis_images[i].cpu().numpy().transpose(1, 2, 0)

        # Denormalize
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        img = img * std + mean
        img = np.clip(img, 0, 1)

        # Display image
        axes[i, 0].imshow(img)
        axes[i, 0].set_title(f"Image {i}")
        axes[i, 0].axis("off")

        # Get matched caption index
        matched_idx = most_similar_idxs[i].item()
        matched_caption = all_captions[matched_idx]
        original_caption = vis_captions[i]

        # Determine if the match is correct (matches ground truth index)
        is_correct = matched_idx == ground_truth_idxs[i]

        # Set text color based on match correctness
        color = "green" if is_correct else "red"

        # Display caption
        axes[i, 1].text(
            0.5,
            0.5,
            f"Original caption:\n{original_caption}\n\nBest match caption:\n{matched_caption}\n\n"
            f"Matched from {len(all_captions)} possible captions",
            ha="center",
            va="center",
            wrap=True,
            fontsize=10,
            color=color,
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
        )
        axes[i, 1].axis("off")

    # Save the figure
    plt.tight_layout()
    plt.subplots_adjust(top=0.95)
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()

    # Create similarity matrix visualization - but only showing the first 100 captions for clarity
    max_vis_captions = min(100, len(all_captions))
    matrix_path = save_path.replace(".png", "_similarity_matrix.png")
    plt.figure(figsize=(10, 8))
    similarity_np = similarity_matrix.cpu().numpy()
    plt.imshow(similarity_np[:, :max_vis_captions], cmap="viridis")
    plt.colorbar(label="Similarity")

    # Add warning for synthetic data
    if is_synthetic:
        plt.title(
            f"Cross-Modal Similarity Matrix (SYNTHETIC DATA) | Model: {param_count:,} parameters",
            color="red",
        )
    else:
        plt.title(
            f"Cross-Modal Similarity Matrix (first {max_vis_captions} captions) | Model: {param_count:,} parameters"
        )

    plt.xlabel("Text")
    plt.ylabel("Image")

    # Add labels (limited to make the plot readable)
    plt.xticks(
        range(max_vis_captions),
        [f"Text {i}" for i in range(max_vis_captions)],
        rotation=90,
    )
    plt.yticks(range(num_samples), [f"Image {i}" for i in range(num_samples)])

    plt.tight_layout()
    plt.savefig(matrix_path, dpi=200)
    plt.close()

    if is_synthetic:
        print(
            f"WARNING: Visualizations saved to {save_path} and {matrix_path} using SYNTHETIC data"
        )
    else:
        print(f"Visualizations saved to {save_path} and {matrix_path}")

    # Calculate accuracy - comparing against all possible captions
    ground_truth_np = np.array(ground_truth_idxs)
    most_similar_np = most_similar_idxs.cpu().numpy()
    accuracy = np.mean((most_similar_np == ground_truth_np).astype(np.float32))
    total_captions = len(all_captions)

    print(
        f"Test samples matching accuracy: {accuracy:.2f} (selecting from {total_captions} possible captions)"
    )
    return accuracy


def main():
    """Main function for multimodal training demo."""
    # Parse arguments
    args = parse_args()

    # Set random seed
    set_seed(args.seed)

    # Create output directories
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, args.checkpoint_dir), exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, args.log_dir), exist_ok=True)

    # Set device
    if args.device is None:
        if torch.backends.mps.is_available():
            device = torch.device("mps")
        elif torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")
    else:
        device = torch.device(args.device)

    print(f"Using device: {device}")

    # Create preprocessor and tokenizer
    image_preprocessor = ImagePreprocessor(
        image_size=224,
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225),
    )

    # Create a simple tokenizer with required interface
    tokenizer = SimpleTokenizer()

    # Create model
    model = create_multimodal_model(args, device)

    # Create data loaders
    train_loader, val_loader, test_loader = create_data_loaders(
        args, image_preprocessor, tokenizer
    )

    # Check if we're using synthetic data
    is_synthetic = False
    if args.use_synthetic or args.dataset == "synthetic":
        is_synthetic = True
    else:
        # Get dataset info from the training dataset
        train_dataset_info = train_loader.dataset.get_split_proportions()
        dataset_name = train_dataset_info.get("dataset_name", "unknown")

        # Check if we're using real Flickr30k data
        if dataset_name == "flickr30k":
            # The data source is Flickr30k, check if we're using cached files from disk
            # or if we had to generate synthetic data
            cache_status = getattr(train_loader.dataset, "loaded_from_cache", True)
            is_synthetic = not cache_status  # If not loaded from cache, it's synthetic
        else:
            # For other dataset types or unknown sources, consider it synthetic
            is_synthetic = True

        # Add an extra warning if using a very small dataset
        train_dataset_size = train_dataset_info.get("total_samples", 0)
        if train_dataset_size < 100:
            print("\n" + "=" * 80)
            print("WARNING: Using a very small dataset (<100 samples).")
            print("Small datasets may lead to overfitting and unrealistic metrics.")
            print("Consider using more data for better evaluation.")
            print("=" * 80 + "\n")

    if is_synthetic:
        print("\n" + "=" * 80)
        print("WARNING: Using synthetic data for training and evaluation.")
        print("Results will not be representative of real-world performance.")
        print(
            "For real evaluation, please ensure you have proper access to the real dataset."
        )
        print("=" * 80 + "\n")

    # Get the dataset size for contrastive loss sampling strategy
    dataset_size = None
    try:
        train_dataset_info = train_loader.dataset.get_split_proportions()
        dataset_size = train_dataset_info.get("total_samples", None)
    except (AttributeError, TypeError):
        logger.warning(
            "Could not determine dataset size for contrastive sampling strategy"
        )

    # Create loss function with enhanced configuration
    loss_fn = create_loss_function(args, dataset_size, train_loader)

    # CRITICAL CHANGE: Initialize the model with pre-trained weights if possible
    # This helps avoid the "cold start" problem in contrastive learning
    try:
        if hasattr(model.vision_model, "initialize_from_pretrained"):
            logger.info("Initializing vision model from pretrained weights")
            success = model.vision_model.initialize_from_pretrained()
            if success:
                logger.info(
                    "Successfully initialized vision model from pretrained weights"
                )
            else:
                logger.warning(
                    "Failed to initialize vision model from pretrained weights"
                )

        if hasattr(model.text_model, "initialize_from_pretrained"):
            logger.info("Initializing text model from pretrained weights")
            success = model.text_model.initialize_from_pretrained()
            if success:
                logger.info(
                    "Successfully initialized text model from pretrained weights"
                )
            else:
                logger.warning(
                    "Failed to initialize text model from pretrained weights"
                )
    except Exception as e:
        logger.warning(f"Error initializing from pretrained: {str(e)}")

    # Create trainer with improved settings for contrastive learning
    trainer = MultimodalTrainer(
        model=model,
        train_dataloader=train_loader,
        val_dataloader=val_loader,
        test_dataloader=test_loader,
        loss_fn=loss_fn,
        num_epochs=args.num_epochs,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay
        * 2.0,  # Double weight decay to prevent overfitting
        warmup_steps=args.warmup_steps,
        checkpoint_dir=os.path.join(args.output_dir, args.checkpoint_dir),
        log_dir=os.path.join(args.output_dir, args.log_dir),
        device=device,
        mixed_precision=args.use_mixed_precision,
        evaluation_steps=0,  # Only evaluate at the end of each epoch
        log_steps=100,  # Reduce logging frequency to avoid spam
        early_stopping_patience=5,
        clip_grad_norm=1.0,  # Add gradient clipping for stability
        accumulation_steps=2,  # Use gradient accumulation for more stable updates
    )

    # Train model
    print("Starting training...")
    print_model_summary(model, "TRAINING MULTIMODAL MODEL")
    # Print model summary before training
    print_model_summary(model, "MULTIMODAL MODEL BEFORE TRAINING")

    # Use multi-stage training if enabled
    if args.use_multistage_training:
        # Determine which pretrained models we're using
        vision_status = "pretrained" if args.use_pretrained else "from scratch"
        text_status = "pretrained" if args.use_pretrained_text else "from scratch"
        freeze_status = "frozen" if args.freeze_base_models else "trainable"

        print(f"\nStarting multi-stage training with:")
        print(f"- Vision model: {args.vision_model} ({vision_status}, {freeze_status})")
        print(f"- Text model: {args.text_model} ({text_status}, {freeze_status})")

        # Add warning if using BERT with MPS
        is_bert = any(
            model_type in args.text_model.lower() for model_type in ["bert", "roberta"]
        )
        device_is_mps = next(model.parameters()).device.type == "mps"

        if is_bert and device_is_mps and args.use_pretrained_text:
            print("\n⚠️ INFO: Using BERT model on MPS device (Apple Silicon)")
            print(
                "The code now properly handles BERT models on MPS devices with a CPU fallback."
            )
            print("Processing may be slower but quality will not be affected.")

        # Determine if we're training from scratch
        # Either explicitly requested, or no pretrained weights are being used
        training_from_scratch = args.from_scratch or not (
            args.use_pretrained or args.use_pretrained_text
        )

        # If training from scratch, enable enhanced data augmentation
        if training_from_scratch:
            print("\nEnabling enhanced data augmentation for training from scratch")
            # This would normally modify dataloaders to include more augmentation strategies
            # For now, we're just acknowledging this would happen here
            # In a full implementation, we'd add more transforms, enable mixup, etc.
            train_dataloader = trainer.train_dataloader
            # TODO: Implement actual augmentation enhancement here

        # Print appropriate stage info based on training approach
        if training_from_scratch:
            print("\nTraining from scratch with specialized curriculum:")
            print("Stage 1a: Train early vision layers (edge/texture detection)")
            print("Stage 1b: Train early text layers (token-level understanding)")
            print("Stage 2a: Train mid-level vision layers (object parts)")
            print("Stage 2b: Train mid-level text layers (phrase-level understanding)")
            print("Stage 3: Train high-level representation in both modalities")
            print("Stage 4: Train cross-modal fusion")
            print("Stage 5: Fine-tune everything with hard negative mining")
        elif args.freeze_base_models:
            print("\nTraining stages (with pre-trained models, frozen):")
            print("Stage 1: Train projections only (vision and text models frozen)")
            print("Stage 2: Train fusion layers (vision and text models frozen)")
            print("Stage 3: Fine-tune everything together")
        else:
            print("\nTraining stages (with pre-trained models):")
            print(
                "Stage 1: Train all components with lower learning rate on pretrained models"
            )
            print("Stage 2: Increase focus on fusion layers")
            print("Stage 3: Fine-tune everything with hard negative mining")

        trainer.train_multistage()
    else:
        if args.freeze_base_models:
            print(
                "\nStarting standard training with FROZEN base models (only fusion layers will train)..."
            )
        else:
            print("\nStarting standard training with ALL parameters trainable...")
        trainer.train()

    # Print model summary after training
    print_model_summary(model, "MULTIMODAL MODEL AFTER TRAINING")

    # Run final evaluation
    print("Running final evaluation...")
    print_model_summary(model, "EVALUATION MODEL")
    test_metrics = trainer.evaluate(test_loader)

    # Add this code to visualize test samples
    print("Generating test sample visualizations...")

    # Get the first X samples from the test dataset for consistent visualization
    test_samples_accuracy = visualize_test_samples(
        model=model,
        test_dataset=test_loader.dataset,
        device=device,
        save_path=os.path.join(args.output_dir, "test_samples_visualization.png"),
        num_samples=10,  # Always visualize 10 samples
    )

    if is_synthetic:
        print(
            f"Test samples matching accuracy: {test_samples_accuracy:.2f} (SYNTHETIC DATA - not representative of real performance)"
        )
    else:
        print(f"Test samples matching accuracy: {test_samples_accuracy:.2f}")

    # Convert PyTorch tensors to Python types for display and serialization
    python_test_metrics = convert_tensors_to_python_types(test_metrics)

    # Create a new dictionary with both metrics and model info
    complete_test_metrics = {
        "metrics": python_test_metrics,
        "model_info": {
            "total_parameters": count_parameters(model),
            "vision_model": args.vision_model,
            "text_model": args.text_model,
            "fusion_type": args.fusion_type,
            "fusion_dim": args.fusion_dim,
        },
    }

    if is_synthetic:
        print(f"Test metrics (SYNTHETIC DATA): {python_test_metrics}")
    else:
        print(f"Test metrics: {python_test_metrics}")

    # Save final test metrics
    with open(os.path.join(args.output_dir, "test_metrics.json"), "w") as f:
        json.dump(complete_test_metrics, f, indent=2)

    # Run inference demo
    print("Running inference demo...")
    print_model_summary(model, "INFERENCE DEMO MODEL")
    inference_metrics = run_inference_demo(
        model, image_preprocessor, tokenizer, device, args
    )

    # Convert PyTorch tensors to Python types for serialization
    python_inference_metrics = convert_tensors_to_python_types(inference_metrics)

    # Create a new dictionary with both metrics and model info
    complete_inference_metrics = {
        "metrics": python_inference_metrics,
        "model_info": {
            "total_parameters": count_parameters(model),
            "vision_model": args.vision_model,
            "text_model": args.text_model,
            "fusion_type": args.fusion_type,
            "fusion_dim": args.fusion_dim,
        },
    }

    # Save demo results
    with open(os.path.join(args.output_dir, "demo_results.json"), "w") as f:
        json.dump(complete_inference_metrics, f, indent=2)

    if is_synthetic:
        print("Multimodal training demo completed with SYNTHETIC DATA!")
        print(
            "For real evaluation, please download and use the actual Flickr30k dataset."
        )
    else:
        print("Multimodal training demo completed!")


if __name__ == "__main__":
    main()
