# src/evaluation/inference_demo.py
"""
Inference demo utilities for multimodal models.

This module provides functions for running inference with multimodal models,
including visualization of model outputs and performance metrics.
"""

import logging
import os
from typing import Any, Dict

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from ..utils.model_utils import count_parameters
from ..utils.visualization import (
    visualize_attention_maps,
    visualize_similarity_matrix,
    visualize_test_samples,
)

logger = logging.getLogger(__name__)


def run_inference_demo(
    model: nn.Module,
    image_preprocessor: Any,
    tokenizer: Any,
    device: torch.device,
    args: Any
) -> Dict[str, float]:
    """
    Run inference demo with the trained model.

    Args:
        model: Trained multimodal model
        image_preprocessor: Image preprocessor
        tokenizer: Text tokenizer
        device: Device to run inference on
        args: Command line arguments

    Returns:
        Dictionary of retrieval metrics
    """
    logger.info("Running inference demo...")
    logger.info(f"Model has {count_parameters(model):,} trainable parameters")

    # Use real test dataset instead of synthetic for better accuracy
    # The main test split is already created and available in the test loader
    from ..data.multimodal_dataset import EnhancedMultimodalDataset

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
        "module_purpose": "Inference demo utilities for multimodal models",
        "key_functions": [
            {
                "name": "run_inference_demo",
                "signature": "run_inference_demo(model: nn.Module, image_preprocessor: Any, tokenizer: Any, device: torch.device, args: Any) -> Dict[str, float]",
                "brief_description": "Run inference demo with visualizations and metrics"
            }
        ],
        "external_dependencies": ["torch", "matplotlib", "logging"],
        "complexity_score": 7  # Moderately high complexity for inference and visualization
    }
