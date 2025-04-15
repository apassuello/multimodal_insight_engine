"""
MultiModal Integration Demo

This script demonstrates the integration of vision and text transformers in a simple
multimodal architecture. It shows how to:
1. Load image and text data
2. Process both modalities with dedicated transformers
3. Project features to a common embedding space
4. Compute cross-modal similarity

The demo uses a small synthetic dataset of image-text pairs to show the functionality.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torchvision import transforms
from PIL import Image
import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Union
import os
import random
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import warnings

# Suppress warnings about dataset loading
warnings.filterwarnings("ignore", category=UserWarning)

# Import our models (adjust imports to match your project structure)
from src.models.vision.vision_transformer import VisionTransformer
from src.models.transformer import EncoderDecoderTransformer
from src.models.vision.multimodal_integration import MultiModalTransformer
from src.models.vision.image_preprocessing import ImagePreprocessor
from src.data.tokenization.optimized_bpe_tokenizer import OptimizedBPETokenizer

# Import Hugging Face datasets
from datasets import load_dataset
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torch


class FlickrMultiModalDataset(Dataset):
    """Dataset adapter for Flickr30k from Hugging Face."""

    def __init__(
        self,
        split="train",  # Can be 'train', 'val', or 'test'
        image_preprocessor=None,
        tokenizer=None,
        max_text_length=77,
    ):
        try:
            # Load the dataset from Hugging Face
            print("Loading Flickr30k dataset...")
            try:
                # Load the full dataset
                dataset_dict = load_dataset("nlphuji/flickr30k")

                # Convert to list format for easier handling
                print(f"Filtering dataset for split: {split}")
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
                                "image": item["image"],
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
                print(
                    f"Successfully loaded {len(self.dataset)} examples from Flickr30k {split} split"
                )

            except Exception as e:
                print(f"Error with primary dataset source: {str(e)}")
                raise  # Re-raise to try alternative sources

        except Exception as e:
            print(f"Error loading Flickr30k dataset: {str(e)}")
            print("Falling back to synthetic data generation...")
            self._generate_synthetic_data()

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

    def _generate_synthetic_data(self):
        """Generate synthetic data as a fallback."""
        # Create a small synthetic dataset with random images and captions
        n_samples = 100
        self.dataset = []

        for i in range(n_samples):
            # Create a random RGB image
            img = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
            img = Image.fromarray(img)

            # Create a simple caption
            caption = f"A synthetic image {i} with random patterns"

            self.dataset.append(
                {
                    "image": img,
                    "captions": [caption],  # Match Flickr format with list of captions
                    "image_id": str(i),
                }
            )

        print(f"Generated {len(self.dataset)} synthetic examples")

    def __len__(self):
        """Return the number of examples in the dataset."""
        return len(self.dataset)

    def __getitem__(self, idx):
        """Get a single example from the dataset."""
        # Get image and caption
        item = self.dataset[idx]
        image = item["image"]  # This is a PIL Image

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
            "image": image_tensor,
            "text": {
                "src": src.unsqueeze(0),  # Add batch dimension
                "src_mask": src_mask,
            },
            "raw_text": caption,
            "image_id": item["image_id"],
        }


class MultiModalDemoDataset(Dataset):
    """
    A simple dataset for multimodal demo that pairs images with text descriptions.

    For demo purposes, we can use a small set of sample images and texts.
    In a real application, you would load this from a proper dataset.
    """

    def __init__(
        self,
        image_dir: str,
        captions_file: str,
        image_preprocessor: ImagePreprocessor,
        tokenizer: OptimizedBPETokenizer,
        max_text_length: int = 77,
    ):
        """
        Initialize the multimodal demo dataset.

        Args:
            image_dir: Directory containing images
            captions_file: File containing captions (one per line)
            image_preprocessor: Image preprocessor to transform images
            tokenizer: Tokenizer for text processing
            max_text_length: Maximum text sequence length
        """
        self.image_dir = Path(image_dir)
        self.captions_file = Path(captions_file)
        self.image_preprocessor = image_preprocessor
        self.tokenizer = tokenizer
        self.max_text_length = max_text_length

        # Load image paths and captions
        self.image_paths = []
        self.captions = []

        # Check if the image directory exists
        if self.image_dir.exists():
            # Find all image files
            self.image_paths = (
                list(self.image_dir.glob("*.jpg"))
                + list(self.image_dir.glob("*.png"))
                + list(self.image_dir.glob("*.jpeg"))
            )

        # Check if the captions file exists
        if self.captions_file.exists():
            with open(self.captions_file, "r", encoding="utf-8") as f:
                self.captions = [line.strip() for line in f if line.strip()]

        # Generate synthetic data if no real data is available
        if not self.image_paths or not self.captions:
            print("No real data found. Generating synthetic data for demo purposes.")
            self._generate_synthetic_data()

        # Ensure same number of images and captions
        min_len = min(len(self.image_paths), len(self.captions))
        self.image_paths = self.image_paths[:min_len]
        self.captions = self.captions[:min_len]

        print(f"Loaded {len(self.image_paths)} image-text pairs")

    def _generate_synthetic_data(self):
        """Generate synthetic data for demo purposes."""
        # For this demo, we'll create random noise images and generic captions
        os.makedirs("demo_data/images", exist_ok=True)

        # Generate 10 synthetic samples
        synthetic_samples = 10

        # Generate random noise images
        for i in range(synthetic_samples):
            # Create a random RGB image (3 channels)
            img = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)

            # Add some structure to make it more interesting
            x, y = np.mgrid[0:224, 0:224]
            circle = (x - 112) ** 2 + (y - 112) ** 2 < 50**2
            img[circle, 0] = 200  # Add a red circle

            # Add a random shape
            shape_type = i % 3
            if shape_type == 0:  # Square
                img[70:140, 70:140, 1] = 200  # Green square
            elif shape_type == 1:  # Rectangle
                img[60:100, 60:160, 2] = 200  # Blue rectangle
            else:  # Triangle
                tri_y, tri_x = np.mgrid[40:140, 40:140]
                triangle = tri_x > tri_y
                img[tri_y[triangle], tri_x[triangle], 1] = 180  # Green triangle

            # Save the image
            img_path = f"demo_data/images/synthetic_{i}.png"
            Image.fromarray(img).save(img_path)
            self.image_paths.append(Path(img_path))

        # Generate generic captions
        shapes = ["circle", "square", "rectangle", "triangle", "shape"]
        colors = ["red", "green", "blue", "colorful", "bright"]
        templates = [
            "An image containing a {color} {shape}",
            "A {color} {shape} on a plain background",
            "A simple {shape} with {color} coloring",
            "Abstract image with a {color} {shape}",
            "Synthetic image showing a {color} {shape}",
        ]

        for i in range(synthetic_samples):
            shape = shapes[i % len(shapes)]
            color = colors[(i + 1) % len(colors)]
            template = templates[i % len(templates)]
            caption = template.format(shape=shape, color=color)
            self.captions.append(caption)

        # Write captions to file
        with open("demo_data/captions.txt", "w", encoding="utf-8") as f:
            for caption in self.captions:
                f.write(caption + "\n")

    def __len__(self):
        """Return the number of samples in the dataset."""
        return len(self.image_paths)

    def __getitem__(self, idx):
        """
        Get a single image-text pair.

        Args:
            idx: Index of the pair

        Returns:
            Dictionary containing:
            - 'image': Preprocessed image tensor
            - 'text': Dictionary with 'src' and 'src_mask' for the tokenized caption
            - 'raw_text': Original caption text
            - 'image_path': Path to the image file
        """
        # Get image path and caption
        image_path = self.image_paths[idx]
        caption = self.captions[idx]

        # Load and preprocess image
        try:
            image = Image.open(image_path).convert("RGB")
            image_tensor = self.image_preprocessor.preprocess(image)
        except Exception as e:
            print(f"Error loading image {image_path}: {e}")
            # Fallback to a blank image
            image_tensor = torch.zeros(3, 224, 224)

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
            "image": image_tensor,
            "text": {
                "src": src.unsqueeze(0),  # Add batch dimension
                "src_mask": src_mask,
            },
            "raw_text": caption,
            "image_path": str(image_path),
        }


def create_multimodal_demo_model(
    device: Union[str, torch.device],
    vit_pretrained: bool = False,
    text_pretrained: bool = False,
):
    """
    Create models for the multimodal demo.

    Args:
        device: Device to place models on (can be string 'cpu'/'cuda'/'mps' or torch.device)
        vit_pretrained: Whether to load pretrained Vision Transformer
        text_pretrained: Whether to load pretrained Text Transformer

    Returns:
        Tuple of (multimodal model, vision preprocessor, text tokenizer)
    """
    # Convert string device to torch.device if needed
    if isinstance(device, str):
        device = torch.device(device)

    # Initialize Vision Transformer
    vit_config = {
        "image_size": 224,
        "patch_size": 16,
        "in_channels": 3,
        "num_classes": 1000,  # ImageNet classes
        "embed_dim": 768,
        "depth": 12,
        "num_heads": 12,
        "mlp_ratio": 4.0,
        "dropout": 0.1,
    }

    vision_model = VisionTransformer(**vit_config)

    # Initialize Text Transformer
    # This is a simplified example - in a real application you would
    # use your actual text transformer architecture
    text_config = {
        "src_vocab_size": 50000,  # Approximate vocabulary size
        "tgt_vocab_size": 50000,
        "d_model": 512,
        "num_heads": 8,
        "num_encoder_layers": 6,
        "num_decoder_layers": 6,
        "d_ff": 2048,
        "dropout": 0.1,
    }

    text_model = EncoderDecoderTransformer(**text_config)

    # Initialize Multimodal Transformer
    multimodal_model = MultiModalTransformer(
        vision_model=vision_model,
        text_model=text_model,
        projection_dim=512,
        dropout=0.1,
    )

    # Move models to device
    multimodal_model = multimodal_model.to(device)

    # Create image preprocessor
    image_preprocessor = ImagePreprocessor(
        image_size=224,
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225),
    )

    # Create text tokenizer (simplified for demo)
    # In a real application, use your actual tokenizer
    tokenizer = OptimizedBPETokenizer()

    return multimodal_model, image_preprocessor, tokenizer


def visualize_multimodal_results(
    images: torch.Tensor,
    captions: List[str],
    similarity_matrix: torch.Tensor,
    num_samples: int = 5,
):
    """
    Visualize the results of multimodal integration.

    Args:
        images: Batch of images
        captions: List of captions
        similarity_matrix: Image-text similarity matrix
        num_samples: Number of samples to display
    """
    # Limit to the specified number of samples
    n = min(num_samples, len(captions), images.shape[0])

    # Create figure with subplots
    fig, axes = plt.subplots(n, 2, figsize=(12, 3 * n))

    # Set a title for the figure
    fig.suptitle("Multimodal Integration Results", fontsize=16)

    # Get the most similar caption for each image
    most_similar_idxs = similarity_matrix.argmax(dim=1)

    # Draw images and captions
    for i in range(n):
        # Get image
        img = images[i].cpu().numpy().transpose(1, 2, 0)

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
        axes[i, 1].text(
            0.5,
            0.5,
            f"Ground truth caption:\n{captions[i]}\n\nBest match caption:\n{captions[most_similar_idxs[i]]}",
            horizontalalignment="center",
            verticalalignment="center",
            wrap=True,
        )
        axes[i, 1].axis("off")

    plt.tight_layout()
    plt.subplots_adjust(top=0.9)
    plt.show()

    # Show similarity matrix as a heatmap
    plt.figure(figsize=(10, 8))
    plt.imshow(similarity_matrix.cpu().numpy(), cmap="viridis")
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
    plt.show()


def run_multimodal_demo_with_flickr():
    """Run the multimodal integration demo with Flickr30k dataset."""
    # First create models on CPU
    multimodal_model, image_preprocessor, base_tokenizer = create_multimodal_demo_model(
        device="cpu"  # Initially create on CPU
    )

    # Always wrap the tokenizer to ensure required methods and attributes
    class TokenizerWithSpecials:
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
            return (
                [self._special_tokens["bos_token_idx"]]
                + [hash(token) % 50000 for token in text.split()]
                + [self._special_tokens["eos_token_idx"]]
            )

        @property
        def special_tokens(self):
            return self._special_tokens

    tokenizer = TokenizerWithSpecials(base_tokenizer)

    # Create dataset
    flickr_dataset = FlickrMultiModalDataset(
        split="train",  # Use training split by default
        image_preprocessor=image_preprocessor,
        tokenizer=tokenizer,  # Now guaranteed to have required methods
        max_text_length=77,
    )

    # Determine device after dataset creation
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        # For MPS, we need to disable multiprocessing
        num_workers = 0
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        num_workers = 2

    print(f"Using device: {device}")

    # Move model to device after dataset creation
    multimodal_model = multimodal_model.to(device)

    # Create dataloader with appropriate num_workers
    batch_size = 16  # Adjust based on your GPU memory
    dataloader = DataLoader(
        flickr_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
    )

    # Get a batch of data
    batch = next(iter(dataloader))

    # Move data to device after loading
    images = batch["image"].to(device)
    texts = {
        "src": batch["text"]["src"].squeeze(1).to(device),
        "src_mask": batch["text"]["src_mask"].to(device),
    }

    # Forward pass through the multimodal model
    with torch.no_grad():
        results = multimodal_model(images, texts)

    # Get similarity matrix
    similarity_matrix = results["similarity"]

    # Visualize results
    captions = batch["raw_text"]
    visualize_multimodal_results(
        images=images.cpu(),  # Move back to CPU for visualization
        captions=captions,
        similarity_matrix=similarity_matrix.cpu(),  # Move back to CPU for visualization
        num_samples=min(5, len(captions)),
    )

    # Calculate matching accuracy
    max_sim_idx = torch.argmax(similarity_matrix, dim=1)
    accuracy = (
        (torch.arange(len(max_sim_idx), device=device) == max_sim_idx)
        .float()
        .mean()
        .item()
    )
    print(f"Cross-modal matching accuracy: {accuracy * 100:.2f}%")

    return {
        "similarity_matrix": similarity_matrix.cpu(),  # Return CPU tensors
        "accuracy": accuracy,
        "captions": captions,
    }


def run_multimodal_demo():
    """Run the multimodal integration demo."""
    # Set device (prefer MPS for Apple Silicon)
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Using device: {device}")

    # Create models
    multimodal_model, image_preprocessor, tokenizer = create_multimodal_demo_model(
        device
    )

    # Create dataset
    demo_dataset = MultiModalDemoDataset(
        image_dir="demo_data/images",
        captions_file="demo_data/captions.txt",
        image_preprocessor=image_preprocessor,
        tokenizer=tokenizer,
        max_text_length=77,
    )

    # Create dataloader
    demo_loader = DataLoader(
        demo_dataset,
        batch_size=5,
        shuffle=True,
    )

    # Get a batch of data
    batch = next(iter(demo_loader))

    # Move data to device
    images = batch["image"].to(device)
    texts = {
        "src": batch["text"]["src"].squeeze(1).to(device),
        "src_mask": batch["text"]["src_mask"].to(device),
    }

    # Forward pass through the multimodal model
    with torch.no_grad():
        results = multimodal_model(images, texts)

    # Get similarity matrix
    similarity_matrix = results["similarity"]

    # Visualize results
    captions = batch["raw_text"]
    visualize_multimodal_results(
        images=images,
        captions=captions,
        similarity_matrix=similarity_matrix,
        num_samples=min(5, len(captions)),
    )

    # Print some statistics
    print(f"Cross-modal similarity matrix shape: {similarity_matrix.shape}")

    # Calculate matching accuracy (diagonal elements should be high)
    diagonal_sim = torch.diag(similarity_matrix)
    print(f"Self-similarity scores: {diagonal_sim}")

    # Get index of maximum similarity for each image
    max_sim_idx = torch.argmax(similarity_matrix, dim=1)

    # Calculate accuracy (how many images matched with their corresponding text)
    accuracy = (
        (torch.arange(len(max_sim_idx), device=device) == max_sim_idx)
        .float()
        .mean()
        .item()
    )
    print(f"Cross-modal matching accuracy: {accuracy * 100:.2f}%")

    return {
        "similarity_matrix": similarity_matrix,
        "accuracy": accuracy,
        "captions": captions,
    }


if __name__ == "__main__":
    print("Running Multimodal Integration Demo")
    results = run_multimodal_demo_with_flickr()
    print("Demo completed!")
