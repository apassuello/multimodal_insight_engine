# src/data/dataset_wrapper.py
"""
Dataset Interface Standardization Wrappers

PURPOSE:
    Provides wrapper classes and utilities to standardize the interface for various datasets,
    ensuring consistent data format and access patterns across different data sources.

KEY COMPONENTS:
    - Dictionary-format dataset wrapper
    - Dictionary-format dataloader creation
    - Standardized dataset loading functions
    - Conversion between tuple and dictionary formats
"""

import torch
from torch.utils.data import Dataset, DataLoader
from typing import Dict, Tuple, List, Any, Callable, Optional, Union
import os


class DictionaryDataset(Dataset):
    """
    Wrapper for datasets that return tuples, converting them to dictionaries.
    This standardizes the interface for all datasets used in the project.
    """

    def __init__(self, dataset: Dataset, keys: List[str] = None):
        """
        Initialize the wrapper.

        Args:
            dataset: The dataset to wrap
            keys: Names to use for the tuple elements. Default is ["image", "label"]
        """
        self.dataset = dataset
        self.keys = keys or ["image", "label"]

    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        return len(self.dataset)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        Get a sample from the dataset as a dictionary.

        Args:
            idx: Index of the sample

        Returns:
            Dictionary with keys from self.keys and values from the dataset
        """
        sample = self.dataset[idx]

        # If already a dictionary with expected keys, return as is
        if isinstance(sample, dict) and all(k in sample for k in self.keys):
            return sample

        # Handle tuple-like returns (convert to dictionary)
        if isinstance(sample, tuple) or isinstance(sample, list):
            # Ensure we have enough keys
            if len(sample) > len(self.keys):
                # More values than keys, create additional keys
                extra_keys = [f"item{i}" for i in range(len(self.keys), len(sample))]
                keys = self.keys + extra_keys
            else:
                keys = self.keys[: len(sample)]

            # Create dictionary from tuple
            return {k: v for k, v in zip(keys, sample)}

        # If single item, assume it's the first key
        return {self.keys[0]: sample}


def create_dictionary_dataloader(
    dataset: Dataset,
    batch_size: int = 32,
    shuffle: bool = True,
    num_workers: int = 4,
    pin_memory: bool = True,
    keys: List[str] = None,
) -> DataLoader:
    """
    Create a DataLoader that returns dictionary-format batches.

    Args:
        dataset: Dataset to wrap
        batch_size: Batch size
        shuffle: Whether to shuffle the data
        num_workers: Number of worker processes
        pin_memory: Whether to pin memory for faster GPU transfer
        keys: Keys to use for the dictionary (default: ["image", "label"])

    Returns:
        DataLoader with dictionary-format batches
    """
    wrapped_dataset = DictionaryDataset(dataset, keys=keys)

    return DataLoader(
        wrapped_dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )


# Enhanced versions of the dataset loader functions


def load_cifar10_dict(
    batch_size: int = 128, num_workers: int = 4, image_size: int = 32
) -> Tuple[DataLoader, DataLoader, List[str]]:
    """
    Load CIFAR-10 dataset with dictionary-format data loaders.

    Args:
        batch_size: Batch size for dataloaders
        num_workers: Number of worker processes for data loading
        image_size: Size to resize images to

    Returns:
        train_loader: Training data loader (dictionary format)
        val_loader: Validation data loader (dictionary format)
        classes: List of class names
    """
    from .dataset_loaders import load_cifar10

    # Get original data loaders
    train_loader_orig, val_loader_orig, classes = load_cifar10(
        batch_size=batch_size, num_workers=num_workers, image_size=image_size
    )

    # Convert to dictionary format
    train_loader = create_dictionary_dataloader(
        train_loader_orig.dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )

    val_loader = create_dictionary_dataloader(
        val_loader_orig.dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_loader, val_loader, classes


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
        "module_purpose": "Provides dataset wrappers to standardize interfaces for different dataset types",
        "key_classes": [
            {
                "name": "DictionaryDataset",
                "purpose": "Wrapper for datasets that return tuples, converting them to dictionaries for a standardized interface",
                "key_methods": [
                    {
                        "name": "__getitem__",
                        "signature": "__getitem__(self, idx: int) -> Dict[str, Any]",
                        "brief_description": "Get a sample from the dataset as a dictionary with standardized keys"
                    }
                ],
                "inheritance": "Dataset",
                "dependencies": ["torch.utils.data.Dataset"]
            }
        ],
        "key_functions": [
            {
                "name": "create_dictionary_dataloader",
                "signature": "create_dictionary_dataloader(dataset: Dataset, batch_size: int = 32, shuffle: bool = True, num_workers: int = 4, pin_memory: bool = True, keys: List[str] = None) -> DataLoader",
                "brief_description": "Create a DataLoader that returns dictionary-format batches"
            },
            {
                "name": "load_cifar10_dict",
                "signature": "load_cifar10_dict(batch_size: int = 128, num_workers: int = 4, image_size: int = 32) -> Tuple[DataLoader, DataLoader, List[str]]",
                "brief_description": "Load CIFAR-10 dataset with dictionary-format data loaders"
            }
        ],
        "external_dependencies": ["torch", "torch.utils.data"],
        "complexity_score": 3  # Low-moderate complexity for dataset wrapping and standardization
    }
