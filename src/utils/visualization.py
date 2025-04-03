"""
Visualization utilities for the MultiModal Insight Engine.

This module provides functions for visualizing model performance,
attention patterns, and embeddings for better interpretability.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Any, Optional, Tuple, Union
import torch
from sklearn.manifold import TSNE

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
        "module_purpose": "Provides visualization utilities for model performance, attention patterns, and embeddings",
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
            }
        ],
        "external_dependencies": ["matplotlib", "seaborn", "torch", "sklearn", "numpy"],
        "complexity_score": 5  # Moderate complexity for visualization utilities
    }