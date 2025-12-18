# src/data/__init__.py
"""
Data modules for the MultiModal Insight Engine.

This package contains components for loading, preprocessing, and
managing data for various model architectures.
"""

# Import dataset classes
from .combined_dataset import CombinedDataset

# NOTE: Constitutional AI dataset has been extracted to standalone repository
# Location: extracted/constitutional-ai/
# Standalone repo: /Users/apa/ml_projects/constitutional-ai
# For Constitutional AI dataset functionality, use the standalone repository
from .dataloader import create_dataloader
from .language_modeling import LanguageModelingDataset

# Import multimodal data utilities
from .multimodal_data_utils import create_data_loaders, randomize_dataset_positions
from .sequence_data import TransformerCollator, TransformerDataModule, TransformerDataset

# Import tokenization utilities
from .tokenization import BaseTokenizer, BPETokenizer, SimpleTokenizer, Vocabulary


__all__ = [
    "LanguageModelingDataset",
    "TransformerDataset",
    "TransformerCollator",
    "TransformerDataModule",
    "CombinedDataset",
    "create_dataloader",
    "create_data_loaders",
    "randomize_dataset_positions",
    "SimpleTokenizer",
    "BPETokenizer",
    "BaseTokenizer",
    "Vocabulary",
]
