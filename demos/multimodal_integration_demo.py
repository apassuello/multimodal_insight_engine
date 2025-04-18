#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Multimodal Integration Training Demo

This script demonstrates a complete training pipeline for multimodal models with:
1. Dataset loading and preprocessing (Flickr30k or synthetic)
2. Advanced multimodal integration with cross-modal attention
3. Contrastive learning training
4. Evaluation on cross-modal retrieval tasks
5. Visualization of results

Author: Arthur PASSUELLO
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Union
import os
import random
from pathlib import Path
import time
import argparse
import logging
from tqdm import tqdm
import warnings

# Suppress warnings
warnings.filterwarnings("ignore", category=UserWarning)

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Import our modules (adjust paths as needed)
from src.models.vision.vision_transformer import VisionTransformer
from src.models.transformer import EncoderDecoderTransformer
from src.models.vision.multimodal_integration import (
    MultiModalTransformer,
    EnhancedMultiModalTransformer,
)
from src.models.vision.cross_modal_attention import (
    CoAttentionFusion,
    BidirectionalCrossAttention,
)
from src.models.vision.image_preprocessing import ImagePreprocessor
from src.data.tokenization.optimized_bpe_tokenizer import OptimizedBPETokenizer
from src.training.contrastive_learning import (
    ContrastiveLoss,
    MultiModalMixedContrastiveLoss,
)
from src.training.multimodal_trainer import MultimodalTrainer

# Try to import Hugging Face datasets
try:
    from datasets import load_dataset
    from PIL import Image

    DATASETS_AVAILABLE = True
except ImportError:
    logger.warning("Hugging Face datasets not available. Will use synthetic data only.")
    DATASETS_AVAILABLE = False


class FlickrMultiModalDataset(Dataset):
    """Dataset adapter for Flickr30k from Hugging Face."""

    def __init__(
        self,
        split="train",  # Can be 'train', 'val', or 'test'
        image_preprocessor=None,
        tokenizer=None,
        max_text_length=77,
        max_samples=None,
    ):
        """
        Initialize the Flickr30k dataset.

        Args:
            split: Dataset split ('train', 'val', or 'test')
            image_preprocessor: Image preprocessing utility
            tokenizer: Text tokenizer
            max_text_length: Maximum text sequence length
            max_samples: Maximum number of samples to use (for quick testing)
        """
        try:
            if not DATASETS_AVAILABLE:
                raise ImportError("Hugging Face datasets not available")

            # Load the dataset from Hugging Face
            logger.info("Loading Flickr30k dataset...")
            try:
                # Load the full dataset
                dataset_dict = load_dataset("nlphuji/flickr30k")

                # Convert to list format for easier handling
                logger.info(f"Filtering dataset for split: {split}")
                filtered_data = []

                # Get the test split since that's what's available
                if not isinstance(dataset_dict, dict) or "test" not in dataset_dict:
                    raise ValueError("Expected dataset with 'test' split")

                test_dataset = dataset_dict["test"]

                # Iterate through all examples
                for i in range(len(test_dataset)):
                    item = test_dataset[i]
                    if isinstance(item, dict) and item.get("split") == split:
                        # Convert to dictionary and store
                        filtered_data.append(
                            {
                                "images": item["images"],
                                "captions": (
                                    [item["caption"]]
                                    if "caption" in item
                                    else item.get("captions", [])
                                ),
                                "image_id": str(i),
                            }
                        )

                if not filtered_data:
                    raise ValueError(f"No examples found for split '{split}'")

                self.dataset = filtered_data
                logger.info(
                    f"Successfully loaded {len(self.dataset)} examples from Flickr30k {split} split"
                )

            except Exception as e:
                logger.error(f"Error with dataset source: {str(e)}")
                raise  # Re-raise to try alternative sources

        except Exception as e:
            logger.warning(f"Error loading Flickr30k dataset: {str(e)}")
            logger.info("Falling back to synthetic data generation...")
            self._generate_synthetic_data(max_samples or 1000)

        # Limit dataset size if specified
        if max_samples and len(self.dataset) > max_samples:
            logger.info(f"Limiting dataset to {max_samples} samples")
            self.dataset = self.dataset[:max_samples]

        self.image_preprocessor = image_preprocessor
        self.tokenizer = tokenizer
        self.max_text_length = max_text_length

        # Verify tokenizer
        if self.tokenizer is None:
            raise ValueError("Tokenizer is required")
        if not hasattr(self.tokenizer, "encode"):
            raise ValueError("Tokenizer must have 'encode' method")
        if not hasattr(self.tokenizer, "special_tokens"):
            raise ValueError("Tokenizer must have 'special_tokens' attribute")

    def _generate_synthetic_data(self, n_samples=1000):
        """
        Generate synthetic data as a fallback.

        Args:
            n_samples: Number of synthetic samples to generate
        """
        logger.info(f"Generating {n_samples} synthetic samples")
        self.dataset = []

        # Create simple shapes for more diverse synthetic images
        shapes = ["circle", "square", "triangle", "rectangle", "star"]
        colors = ["red", "green", "blue", "yellow", "purple", "orange"]
        objects = ["cat", "dog", "car", "tree", "house", "person", "bird"]
        attributes = ["small", "large", "colorful", "bright", "dark", "shiny"]

        for i in range(n_samples):
            # Create a random RGB image
            img = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)

            # Add some structure to make it more interesting
            x, y = np.mgrid[0:224, 0:224]

            # Randomly choose colors for elements
            primary_color = np.random.randint(0, 3)  # RGB channel index
            secondary_color = (primary_color + 1) % 3

            # Add a random shape
            shape_type = i % 5
            shape_size = np.random.randint(40, 100)
            center_x = np.random.randint(shape_size, 224 - shape_size)
            center_y = np.random.randint(shape_size, 224 - shape_size)

            if shape_type == 0:  # Circle
                circle = (x - center_x) ** 2 + (y - center_y) ** 2 < shape_size**2
                img[circle, primary_color] = np.random.randint(180, 250)
            elif shape_type == 1:  # Square
                square = (abs(x - center_x) < shape_size // 2) & (
                    abs(y - center_y) < shape_size // 2
                )
                img[square, primary_color] = np.random.randint(180, 250)
            elif shape_type == 2:  # Triangle
                tri_y, tri_x = np.mgrid[
                    center_y - shape_size : center_y + shape_size,
                    center_x - shape_size : center_x + shape_size,
                ]
                mask = (0 <= tri_y) & (tri_y < 224) & (0 <= tri_x) & (tri_x < 224)
                tri_y, tri_x = tri_y[mask], tri_x[mask]
                triangle = (tri_x - center_x) < (tri_y - center_y)
                img[tri_y[triangle], tri_x[triangle], primary_color] = (
                    np.random.randint(180, 250)
                )
            elif shape_type == 3:  # Rectangle
                rect = (abs(x - center_x) < shape_size) & (
                    abs(y - center_y) < shape_size // 2
                )
                img[rect, primary_color] = np.random.randint(180, 250)
            else:  # Random noise pattern
                pattern = np.random.rand(224, 224) > 0.8
                img[pattern, primary_color] = np.random.randint(180, 250)

            # Add a secondary element
            secondary_x = (center_x + 112) % 224
            secondary_y = (center_y + 112) % 224
            secondary_size = shape_size // 2

            secondary = (x - secondary_x) ** 2 + (
                y - secondary_y
            ) ** 2 < secondary_size**2
            img[secondary, secondary_color] = np.random.randint(180, 250)

            # Create image from array
            img = Image.fromarray(img)

            # Create descriptive caption
            shape_name = shapes[shape_type]
            color_name = colors[primary_color]
            secondary_color_name = colors[secondary_color]

            caption_templates = [
                f"A {color_name} {shape_name} with a {secondary_color_name} circle",
                f"An image showing a {color_name} {shape_name} and a {secondary_color_name} dot",
                f"A synthetic image with a {color_name} {shape_name} and {secondary_color_name} element",
                f"A {color_name} {shape_name} next to a {secondary_color_name} circle",
                f"A computer-generated image of a {color_name} {shape_name} and {secondary_color_name} circle",
            ]

            caption = random.choice(caption_templates)

            self.dataset.append(
                {
                    "images": img,
                    "captions": [caption],
                    "image_id": str(i),
                }
            )

        logger.info(f"Generated {len(self.dataset)} synthetic examples")

    def __len__(self):
        """Return the number of examples in the dataset."""
        return len(self.dataset)

    def __getitem__(self, idx):
        """
        Get a single example from the dataset.

        Args:
            idx: Index of the example

        Returns:
            Dictionary containing image, text, caption, and index information
        """
        # Get image and caption
        item = self.dataset[idx]
        image = item["images"]  # This is a PIL Image

        # Randomly select one caption if multiple are available
        caption = random.choice(item["captions"])
        # Ensure caption is a string
        if isinstance(caption, (list, tuple)):
            caption = caption[0] if caption else ""
        caption = str(caption)

        # Process image
        if self.image_preprocessor:
            image_tensor = self.image_preprocessor.preprocess(image)
        else:
            raise ValueError("Image preprocessor is required")

        # Process caption
        token_ids = self.tokenizer.encode(caption)

        # Pad or truncate to max_text_length
        if len(token_ids) > self.max_text_length:
            token_ids = token_ids[: self.max_text_length]
        else:
            padding = [self.tokenizer.special_tokens["pad_token_idx"]] * (
                self.max_text_length - len(token_ids)
            )
            token_ids = token_ids + padding

        # Create source tensor
        src = torch.tensor(token_ids, dtype=torch.long)

        # Create source mask (1 for real tokens, 0 for padding)
        src_mask = (
            (src != self.tokenizer.special_tokens["pad_token_idx"])
            .unsqueeze(0)
            .unsqueeze(0)
        )

        return {
            "images": image_tensor,
            "text_data": {
                "src": src.unsqueeze(0),  # Add batch dimension
                "src_mask": src_mask,
            },
            "raw_text": caption,
            "image_id": item["image_id"],
            "idx": torch.tensor(
                idx, dtype=torch.long
            ),  # Add index for retrieval evaluation
        }


def create_dataloaders(
    image_preprocessor,
    tokenizer,
    batch_size=32,
    max_samples=None,
    train_val_test_split=(0.8, 0.1, 0.1),
):
    """
    Create train, validation, and test dataloaders.

    Args:
        image_preprocessor: Image preprocessing utility
        tokenizer: Text tokenizer
        batch_size: Batch size for dataloaders
        max_samples: Maximum number of samples to use
        train_val_test_split: Train/val/test split ratios

    Returns:
        Tuple of (train_dataloader, val_dataloader, test_dataloader)
    """
    # Determine number of workers
    if torch.backends.mps.is_available():
        # For MPS (Apple Silicon), use minimal number of workers
        num_workers = 0
    else:
        num_workers = min(4, os.cpu_count() or 1)

    logger.info(f"Using {num_workers} dataloader workers")

    # Create full dataset
    full_dataset = FlickrMultiModalDataset(
        split="train",  # Use train split for all data
        image_preprocessor=image_preprocessor,
        tokenizer=tokenizer,
        max_samples=max_samples,
    )

    # Calculate split sizes
    total_size = len(full_dataset)
    train_size = int(total_size * train_val_test_split[0])
    val_size = int(total_size * train_val_test_split[1])
    test_size = total_size - train_size - val_size

    # Split dataset
    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
        full_dataset, [train_size, val_size, test_size]
    )

    logger.info(
        f"Dataset split: {train_size} train, {val_size} validation, {test_size} test samples"
    )

    # Create dataloaders
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        drop_last=True,
    )

    val_dataloader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )

    test_dataloader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )

    return train_dataloader, val_dataloader, test_dataloader


def create_models(device, fusion_type="bidirectional"):
    """
    Create Vision Transformer, Text Transformer, and Multimodal Transformer models.

    Args:
        device: Device to place models on
        fusion_type: Type of fusion to use ('bidirectional' or 'co_attention')

    Returns:
        Tuple of (multimodal_model, image_preprocessor, tokenizer)
    """
    # Initialize Vision Transformer
    vit_config = {
        "image_size": 224,
        "patch_size": 16,
        "in_channels": 3,
        "num_classes": 1000,  # ImageNet classes
        "embed_dim": 512,  # Using smaller dim for faster training
        "depth": 6,  # Smaller model for demo
        "num_heads": 8,
        "mlp_ratio": 4.0,
        "dropout": 0.1,
    }

    vision_model = VisionTransformer(**vit_config)
    logger.info(
        f"Created Vision Transformer with {sum(p.numel() for p in vision_model.parameters()):,} parameters"
    )

    # Initialize Text Transformer
    text_config = {
        "src_vocab_size": 50000,  # Approximate vocabulary size
        "tgt_vocab_size": 50000,
        "d_model": 512,
        "num_heads": 8,
        "num_encoder_layers": 6,
        "num_decoder_layers": 0,  # Use encoder-only for embedding
        "d_ff": 2048,
        "dropout": 0.1,
    }

    text_model = EncoderDecoderTransformer(**text_config)
    logger.info(
        f"Created Text Transformer with {sum(p.numel() for p in text_model.parameters()):,} parameters"
    )

    # Initialize Multimodal Transformer
    if fusion_type == "simple":
        multimodal_model = MultiModalTransformer(
            vision_model=vision_model,
            text_model=text_model,
            projection_dim=512,
            dropout=0.1,
        )
        logger.info("Using simple projection-based multimodal integration")
    else:
        multimodal_model = EnhancedMultiModalTransformer(
            vision_model=vision_model,
            text_model=text_model,
            fusion_dim=512,
            num_fusion_layers=2,
            num_heads=8,
            dropout=0.1,
            fusion_type=fusion_type,  # "bidirectional" or "co_attention"
        )
        logger.info(f"Using enhanced multimodal integration with {fusion_type} fusion")

    logger.info(
        f"Created Multimodal Transformer with {sum(p.numel() for p in multimodal_model.parameters()):,} parameters"
    )

    # Move models to device
    multimodal_model = multimodal_model.to(device)

    # Create image preprocessor
    image_preprocessor = ImagePreprocessor(
        image_size=224,
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225),
    )

    # Create text tokenizer
    tokenizer = TokenizerWithSpecials(OptimizedBPETokenizer())

    return multimodal_model, image_preprocessor, tokenizer


class TokenizerWithSpecials:
    """
    Wrapper for tokenizer to ensure it has required special tokens.
    """

    def __init__(self, base_tokenizer):
        self.base_tokenizer = base_tokenizer
        self._special_tokens = {
            "pad_token_idx": 0,
            "unk_token_idx": 1,
            "bos_token_idx": 2,
            "eos_token_idx": 3,
            "mask_token_idx": 4,
        }

    def encode(self, text):
        if hasattr(self.base_tokenizer, "encode"):
            return self.base_tokenizer.encode(text)
        # Fallback encoding method
        return (
            [self._special_tokens["bos_token_idx"]]
            + [hash(token) % 50000 for token in text.split()]
            + [self._special_tokens["eos_token_idx"]]
        )

    @property
    def special_tokens(self):
        return self._special_tokens


def visualize_image_text_matches(
    model, dataset, device, num_examples=5, output_dir="outputs"
):
    """
    Visualize image-text matches.

    Args:
        model: Trained multimodal model
        dataset: Dataset to sample from
        device: Device to use
        num_examples: Number of examples to visualize
        output_dir: Directory to save visualizations
    """
    os.makedirs(output_dir, exist_ok=True)

    # Create dataloader with batch size = num_examples
    dataloader = DataLoader(dataset, batch_size=num_examples, shuffle=True)

    # Get a batch
    batch = next(iter(dataloader))

    # Move data to device
    batch = {
        k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()
    }

    # Run model
    with torch.no_grad():
        outputs = model(images=batch["images"], text_data=batch["text_data"])

    # Get similarity matrix
    if "similarity" in outputs:
        similarity = outputs["similarity"]
    elif "raw_similarity" in outputs:
        similarity = outputs["raw_similarity"]
    else:
        # Calculate similarity from embeddings
        image_emb = outputs["vision_features"]
        text_emb = outputs["text_features"]

        # Normalize
        image_emb = torch.nn.functional.normalize(image_emb, p=2, dim=1)
        text_emb = torch.nn.functional.normalize(text_emb, p=2, dim=1)

        # Compute similarity
        similarity = torch.matmul(image_emb, text_emb.T)

    # Get images and captions
    images = batch["images"].cpu()
    captions = batch["raw_text"]

    # Create figure with subplots
    fig, axes = plt.subplots(num_examples, 2, figsize=(14, 3 * num_examples))

    # Set a title for the figure
    fig.suptitle("Multimodal Integration Results", fontsize=16)

    # Get the most similar caption for each image
    most_similar_idxs = similarity.argmax(dim=1).cpu().numpy()

    # Draw images and captions
    for i in range(num_examples):
        # Get image
        img = images[i].numpy().transpose(1, 2, 0)

        # Un-normalize image
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        img = img * std + mean
        img = np.clip(img, 0, 1)

        # Display image
        axes[i, 0].imshow(img)
        axes[i, 0].set_title(f"Image {i}")
        axes[i, 0].axis("off")

        # Display caption
        gt_caption = captions[i]
        matched_caption = captions[most_similar_idxs[i]]
        is_correct = i == most_similar_idxs[i]

        caption_text = f"Ground truth caption:\n{gt_caption}\n\n"
        caption_text += f"Best match caption:\n{matched_caption}\n\n"
        caption_text += f"Match correct: {'✓' if is_correct else '✗'}"

        axes[i, 1].text(
            0.5,
            0.5,
            caption_text,
            horizontalalignment="center",
            verticalalignment="center",
            wrap=True,
            color="green" if is_correct else "red",
        )
        axes[i, 1].axis("off")

    plt.tight_layout()
    plt.subplots_adjust(top=0.9)
    plt.savefig(os.path.join(output_dir, "image_text_matches.png"))

    # Show similarity matrix as a heatmap
    plt.figure(figsize=(10, 8))
    plt.imshow(similarity.cpu().numpy(), cmap="viridis")
    plt.colorbar(label="Similarity")
    plt.xlabel("Text")
    plt.ylabel("Image")
    plt.title("Cross-Modal Similarity Matrix")

    # Add grid
    plt.grid(False)

    # Add labels
    plt.xticks(
        range(len(captions)),
        [f"Caption {i}" for i in range(len(captions))],
        rotation=90,
    )
    plt.yticks(range(len(captions)), [f"Image {i}" for i in range(len(captions))])

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "similarity_matrix.png"))

    logger.info(f"Visualizations saved to {output_dir}")


def main():
    """Main function to run the multimodal training demo."""
    parser = argparse.ArgumentParser(description="Multimodal Training Demo")
    parser.add_argument(
        "--batch_size", type=int, default=32, help="Batch size for training"
    )
    parser.add_argument(
        "--num_epochs", type=int, default=5, help="Number of epochs to train"
    )
    parser.add_argument(
        "--learning_rate", type=float, default=5e-5, help="Learning rate"
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=1000,
        help="Max number of samples to use (for quick testing)",
    )
    parser.add_argument(
        "--fusion_type",
        type=str,
        default="bidirectional",
        choices=["simple", "bidirectional", "co_attention"],
        help="Fusion type",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="outputs",
        help="Output directory for logs and checkpoints",
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed for reproducibility"
    )
    parser.add_argument(
        "--eval_only",
        action="store_true",
        help="Only run evaluation on a pretrained model",
    )
    parser.add_argument(
        "--checkpoint", type=str, default=None, help="Load model from checkpoint"
    )
    args = parser.parse_args()

    # Set random seed
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    # Set up directories
    checkpoint_dir = os.path.join(args.output_dir, "checkpoints")
    log_dir = os.path.join(args.output_dir, "logs")
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    # Set device
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        logger.info("Using MPS (Apple Silicon) for training")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        logger.info(f"Using CUDA for training ({torch.cuda.get_device_name(0)})")
    else:
        device = torch.device("cpu")
        logger.info("Using CPU for training (this will be slow)")

    # Create models
    model, image_preprocessor, tokenizer = create_models(
        device, fusion_type=args.fusion_type
    )

    # Create dataloaders
    train_dataloader, val_dataloader, test_dataloader = create_dataloaders(
        image_preprocessor=image_preprocessor,
        tokenizer=tokenizer,
        batch_size=args.batch_size,
        max_samples=args.max_samples,
    )

    # Create loss function
    contrastive_loss = ContrastiveLoss(temperature=0.07)

    # Create optimizer
    optimizer = optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=0.01,
        betas=(0.9, 0.999),
    )

    # Create scheduler with warmup
    warmup_steps = len(train_dataloader) // 10  # 10% of steps for warmup
    total_steps = len(train_dataloader) * args.num_epochs

    def lr_lambda(step):
        if step < warmup_steps:
            return float(step) / float(max(1, warmup_steps))
        return max(
            0.0,
            float(total_steps - step) / float(max(1, total_steps - warmup_steps)),
        )

    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # Create trainer
    trainer = MultimodalTrainer(
        model=model,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        test_dataloader=test_dataloader,
        optimizer=optimizer,
        scheduler=scheduler,
        loss_fn=contrastive_loss,
        num_epochs=args.num_epochs,
        checkpoint_dir=checkpoint_dir,
        log_dir=log_dir,
        device=device,
        mixed_precision=device.type == "cuda",  # Use mixed precision only on CUDA
        evaluation_steps=len(train_dataloader) // 2,  # Evaluate twice per epoch
        early_stopping_patience=3,
    )

    # Load checkpoint if specified
    if args.checkpoint:
        trainer.load_checkpoint(args.checkpoint)
        logger.info(f"Loaded checkpoint from {args.checkpoint}")

    # Train or evaluate
    if not args.eval_only:
        logger.info("Starting training...")
        start_time = time.time()
        history = trainer.train()
        train_time = time.time() - start_time
        logger.info(f"Training completed in {train_time:.2f} seconds")

        # Save plots after training
        trainer.plot_history(os.path.join(log_dir, "history"))

    # Final evaluation
    logger.info("Running final evaluation...")
    test_metrics = trainer.evaluate(test_dataloader)
    logger.info("Test metrics:")
    for k, v in test_metrics.items():
        logger.info(f"  {k}: {v:.4f}")

    # Visualize results
    logger.info("Visualizing results...")
    visualize_image_text_matches(
        model=model,
        dataset=test_dataloader.dataset,
        device=device,
        num_examples=min(5, len(test_dataloader.dataset)),
        output_dir=args.output_dir,
    )

    logger.info("Demo completed successfully!")


if __name__ == "__main__":
    main()
