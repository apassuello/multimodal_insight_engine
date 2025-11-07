"""
Visualization utilities for the MultiModal Insight Engine.

This module provides functions for visualizing model performance,
attention patterns, embeddings, and multimodal model outputs for
better interpretability.
"""

import os
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.manifold import TSNE
from tqdm import tqdm


def plot_training_history(history: Dict[str, List[float]],
                         figsize: Tuple[int, int] = (12, 8),
                         save_path: Optional[str] = None) -> None:
    """
    Plot training metrics history.
    
    Args:
        history: Dictionary mapping metric names to lists of values
        figsize: Figure size as (width, height)
        save_path: Optional path to save the figure
    """
    fig, axes = plt.subplots(len(history), 1, figsize=figsize)
    if len(history) == 1:
        axes = [axes]

    for i, (metric, values) in enumerate(history.items()):
        axes[i].plot(values)
        axes[i].set_title(f'{metric} history')
        axes[i].set_xlabel('Epoch')
        axes[i].set_ylabel(metric.capitalize())
        axes[i].grid(True)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)

    plt.show()

def plot_attention_weights(attention_weights: torch.Tensor,
                          tokens: List[str] = None,
                          layer: int = 0,
                          head: int = 0,
                          figsize: Tuple[int, int] = (10, 10),
                          save_path: Optional[str] = None) -> None:
    """
    Visualize attention weights from a transformer model.
    
    Args:
        attention_weights: Tensor of attention weights with shape [layers, heads, seq_len, seq_len]
        tokens: Optional list of token strings for axis labels
        layer: Which layer to visualize
        head: Which attention head to visualize
        figsize: Figure size as (width, height)
        save_path: Optional path to save the figure
    """
    # Extract the specified layer and head
    weights = attention_weights[layer, head].cpu().detach().numpy()

    plt.figure(figsize=figsize)
    ax = sns.heatmap(weights,
                    annot=False,
                    cmap='viridis',
                    xticklabels=tokens if tokens else [],
                    yticklabels=tokens if tokens else [])

    plt.title(f'Attention Weights (Layer {layer}, Head {head})')
    plt.xlabel('Target Tokens')
    plt.ylabel('Source Tokens')

    if tokens:
        plt.xticks(rotation=90)
        plt.yticks(rotation=0)

    if save_path:
        plt.savefig(save_path)

    plt.show()

def plot_embeddings_tsne(embeddings: torch.Tensor,
                        labels: Optional[List[Any]] = None,
                        random_state: int = 42,
                        figsize: Tuple[int, int] = (10, 10),
                        save_path: Optional[str] = None) -> None:
    """
    Visualize embeddings using t-SNE dimensionality reduction.
    
    Args:
        embeddings: Tensor of embeddings to visualize
        labels: Optional list of labels for color-coding
        random_state: Random seed for t-SNE
        figsize: Figure size as (width, height)
        save_path: Optional path to save the figure
    """
    # Convert embeddings to numpy if they're torch tensors
    if isinstance(embeddings, torch.Tensor):
        embeddings = embeddings.cpu().detach().numpy()

    # Apply t-SNE
    tsne = TSNE(n_components=2, random_state=random_state, perplexity=min(30, len(embeddings)-1))
    reduced_embeddings = tsne.fit_transform(embeddings)

    plt.figure(figsize=figsize)

    if labels is not None:
        # If we have labels, use them for coloring
        unique_labels = list(set(labels))
        colors = plt.cm.rainbow(np.linspace(0, 1, len(unique_labels)))

        for i, label in enumerate(unique_labels):
            indices = [j for j, l in enumerate(labels) if l == label]
            plt.scatter(reduced_embeddings[indices, 0],
                       reduced_embeddings[indices, 1],
                       color=colors[i],
                       label=label,
                       alpha=0.7)
        plt.legend()
    else:
        # If no labels, just plot the points
        plt.scatter(reduced_embeddings[:, 0], reduced_embeddings[:, 1], alpha=0.7)

    plt.title('t-SNE Visualization of Embeddings')
    plt.xlabel('Dimension 1')
    plt.ylabel('Dimension 2')
    plt.grid(True)

    if save_path:
        plt.savefig(save_path)

    plt.show()

def count_parameters(model: nn.Module) -> int:
    """
    Count the total number of trainable parameters in a model.
    
    Args:
        model: PyTorch model
        
    Returns:
        int: Total number of trainable parameters
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def visualize_similarity_matrix(
    similarity_matrix: torch.Tensor,
    captions: List[str],
    save_path: Optional[str] = None
) -> None:
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
    attention_maps: Dict[str, torch.Tensor],
    images: torch.Tensor,
    captions: List[str],
    save_dir: Optional[str] = None,
    model: Optional[nn.Module] = None
) -> None:
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


def visualize_test_samples(
    model: nn.Module,
    test_dataset: Any,
    device: torch.device,
    save_path: str,
    num_samples: int = 10
) -> float:
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
    from torch.utils.data import DataLoader
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


def extract_file_metadata(file_path=__file__):
    """
    Extract structured metadata about this module.
    
    Args:
        file_path: Path to the source file (defaults to current file)
        
    Returns:
        dict: Structured metadata about the module's purpose and components
    """
    return {
        "filename": os.path.basename(file_path),
        "module_purpose": "Provides visualization utilities for model performance, attention patterns, embeddings, and multimodal outputs",
        "key_functions": [
            {
                "name": "plot_training_history",
                "signature": "plot_training_history(history: Dict[str, List[float]], figsize: Tuple[int, int] = (12, 8), save_path: Optional[str] = None) -> None",
                "brief_description": "Plot training metrics history over epochs"
            },
            {
                "name": "plot_attention_weights",
                "signature": "plot_attention_weights(attention_weights: torch.Tensor, tokens: List[str] = None, layer: int = 0, head: int = 0, figsize: Tuple[int, int] = (10, 10), save_path: Optional[str] = None) -> None",
                "brief_description": "Visualize attention weights from transformer models"
            },
            {
                "name": "plot_embeddings_tsne",
                "signature": "plot_embeddings_tsne(embeddings: torch.Tensor, labels: Optional[List[Any]] = None, random_state: int = 42, figsize: Tuple[int, int] = (10, 10), save_path: Optional[str] = None) -> None",
                "brief_description": "Visualize embeddings using t-SNE dimensionality reduction"
            },
            {
                "name": "visualize_similarity_matrix",
                "signature": "visualize_similarity_matrix(similarity_matrix: torch.Tensor, captions: List[str], save_path: Optional[str] = None) -> None",
                "brief_description": "Visualize the similarity matrix between images and texts"
            },
            {
                "name": "visualize_attention_maps",
                "signature": "visualize_attention_maps(attention_maps: Dict[str, torch.Tensor], images: torch.Tensor, captions: List[str], save_dir: Optional[str] = None, model: Optional[nn.Module] = None) -> None",
                "brief_description": "Visualize attention maps between images and texts"
            },
            {
                "name": "visualize_test_samples",
                "signature": "visualize_test_samples(model: nn.Module, test_dataset: Any, device: torch.device, save_path: str, num_samples: int = 10) -> float",
                "brief_description": "Visualize test samples with their matched captions"
            },
            {
                "name": "count_parameters",
                "signature": "count_parameters(model: nn.Module) -> int",
                "brief_description": "Count trainable parameters in a model"
            }
        ],
        "external_dependencies": ["matplotlib", "seaborn", "torch", "sklearn", "numpy", "tqdm"],
        "complexity_score": 7  # Increased complexity for multimodal visualization
    }
