# src/utils/argument_configs.py
"""
Argument configuration utilities for multimodal training scripts.

This module provides functions to create and configure argument parsers
for multimodal training scripts with consistent options and defaults.
"""

import argparse
from typing import Dict, Any


def get_multimodal_training_args() -> argparse.ArgumentParser:
    """
    Create argument parser for multimodal training scripts.

    Returns:
        argparse.ArgumentParser: Configured argument parser
    """
    parser = argparse.ArgumentParser(description="Multimodal Training Configuration")

    # Data arguments
    parser.add_argument(
        "--dataset",
        type=str,
        default="flickr30k",
        choices=["flickr30k", "custom", "synthetic"],
        help="Dataset to use",
    )

    parser.add_argument(
        "--use_simple_model",
        action="store_true",
        help="Use simplified model architecture for debugging",
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
    parser.add_argument(
        "--captions_per_image",
        type=int,
        default=1,
        choices=range(1, 6),
        help="Number of captions to use per image (1-5, default: 1)",
    )

    # Semantic batching arguments
    parser.add_argument(
        "--use_semantic_batching",
        action="store_true",
        default=True,
        help="Use semantic grouping in batches for contrastive learning",
    )
    parser.add_argument(
        "--min_samples_per_group",
        type=int,
        default=4,
        help="Minimum number of samples per semantic group in each batch",
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
        default="mobilebert",  # Default to MobileBERT which works well on all devices
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
    # Dimension-matched model size presets
    parser.add_argument(
        "--model_size",
        type=str,
        default=None,
        choices=["small", "medium", "large"],
        help="""Preset for dimension-matched model pairs:
        - small (384): google/vit-base-patch16-384 + microsoft/MiniLM-L12-H384-uncased
        - medium (512): microsoft/beit-large-patch16-512 + flaubert-small-cased
        - large (768): google/vit-base-patch16-224 + bert-base-uncased""",
    )
    parser.add_argument(
        "--use_pretrained_text",
        action="store_true",
        default=True,  # Default to using pretrained text models
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
        "--fusion_dim",
        type=int,
        default=512,
        help="Dimension for multimodal fusion (should match model dimensions)",
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

    # Loss type selection
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

    # Memory Queue specific arguments
    parser.add_argument(
        "--queue_size",
        type=int,
        default=8192,
        help="Size of the memory queue for MemoryQueueContrastiveLoss",
    )

    # Dynamic Temperature specific arguments
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

    # Hard Negative Mining specific arguments
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
        "--learning_rate", type=float, default=1e-5, help="Learning rate"
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

    return parser


def extract_file_metadata(file_path=__file__):
    """
    Extract structured metadata about this module.

    Args:
        file_path: Path to the source file (defaults to current file)

    Returns:
        dict: Structured metadata about the module's purpose and components
    """
    import os

    return {
        "filename": os.path.basename(file_path),
        "module_purpose": "Argument configuration utilities for multimodal training scripts",
        "key_functions": [
            {
                "name": "get_multimodal_training_args",
                "signature": "get_multimodal_training_args() -> argparse.ArgumentParser",
                "brief_description": "Create argument parser for multimodal training scripts",
            }
        ],
        "key_argument_groups": [
            {
                "name": "data_args",
                "description": "Arguments for dataset configuration and processing",
            },
            {
                "name": "semantic_batching_args",
                "description": "Arguments for semantic grouping in batch creation for contrastive learning",
            },
            {
                "name": "model_args",
                "description": "Arguments for model architecture and configuration",
            },
            {
                "name": "contrastive_learning_args",
                "description": "Arguments for contrastive learning setup and optimization",
            },
            {
                "name": "training_args",
                "description": "Arguments for training process configuration",
            },
        ],
        "external_dependencies": ["argparse"],
        "complexity_score": 4,  # Low to medium complexity for argument configuration
    }
