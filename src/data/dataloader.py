import torch
from torch.utils.data import Dataset, DataLoader
from typing import Dict, Any, Optional, Tuple, List, Callable
import numpy as np
import os

class MultimodalDataset(Dataset):
    """A dataset class for handling multimodal data."""
    
    def __init__(self, data_dict: Dict[str, torch.Tensor]):
        """
        Initialize the dataset.
        
        Args:
            data_dict: Dictionary containing different modalities of data
        """
        self.data_dict = data_dict
        self.length = len(next(iter(data_dict.values())))
        
        # Verify all modalities have the same length
        for modality, data in data_dict.items():
            if len(data) != self.length:
                raise ValueError(f"All modalities must have the same length. {modality} has length {len(data)}")
    
    def __len__(self) -> int:
        """Return the length of the dataset."""
        return self.length
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get an item from the dataset.
        
        Args:
            idx: Index of the item to get
            
        Returns:
            Dictionary containing the item from each modality
        """
        return {modality: data[idx] for modality, data in self.data_dict.items()}

def create_dataloader(
    dataset: Dataset,
    batch_size: int = 32,
    shuffle: bool = True,
    num_workers: int = 0,
    pin_memory: bool = True,
    drop_last: bool = False,
    collate_fn: Optional[callable] = None
) -> DataLoader:
    """
    Create a DataLoader from a dataset.
    
    Args:
        dataset: Dataset to create DataLoader from
        batch_size: Batch size for the DataLoader
        shuffle: Whether to shuffle the data
        num_workers: Number of worker processes
        pin_memory: Whether to pin memory for faster data transfer to GPU
        drop_last: Whether to drop the last incomplete batch
        collate_fn: Optional custom collate function for batching
        
    Returns:
        DataLoader instance
    """
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last,
        collate_fn=collate_fn
    )

def collate_fn(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    """
    Custom collate function for batching multimodal data.
    
    Args:
        batch: List of dictionaries containing data from each modality
        
    Returns:
        Dictionary containing batched data from each modality
    """
    result = {}
    for key in batch[0].keys():
        result[key] = torch.stack([item[key] for item in batch])
    return result

def get_dataloaders(
    train_data: Dict[str, torch.Tensor],
    val_data: Optional[Dict[str, torch.Tensor]] = None,
    test_data: Optional[Dict[str, torch.Tensor]] = None,
    batch_size: int = 32,
    num_workers: int = 0
) -> Tuple[DataLoader, Optional[DataLoader], Optional[DataLoader]]:
    """
    Create DataLoaders for train, validation, and test sets.
    
    Args:
        train_data: Training data dictionary
        val_data: Optional validation data dictionary
        test_data: Optional test data dictionary
        batch_size: Batch size for the DataLoaders
        num_workers: Number of worker processes
        
    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    train_dataset = MultimodalDataset(train_data)
    train_loader = create_dataloader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate_fn
    )
    
    val_loader = None
    if val_data is not None:
        val_dataset = MultimodalDataset(val_data)
        val_loader = create_dataloader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            collate_fn=collate_fn
        )
    
    test_loader = None
    if test_data is not None:
        test_dataset = MultimodalDataset(test_data)
        test_loader = create_dataloader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            collate_fn=collate_fn
        )
    
    return train_loader, val_loader, test_loader
    
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
        "module_purpose": "Provides utilities for handling multimodal data with PyTorch DataLoader",
        "key_classes": [
            {
                "name": "MultimodalDataset",
                "purpose": "Dataset class for handling multiple modalities of data with consistent length validation",
                "key_methods": [
                    {
                        "name": "__getitem__",
                        "signature": "__getitem__(self, idx: int) -> Dict[str, torch.Tensor]",
                        "brief_description": "Get an item from each modality at the specified index"
                    },
                    {
                        "name": "__len__",
                        "signature": "__len__(self) -> int",
                        "brief_description": "Return the length of the dataset"
                    }
                ],
                "inheritance": "Dataset",
                "dependencies": ["torch.utils.data.Dataset"]
            }
        ],
        "key_functions": [
            {
                "name": "create_dataloader",
                "signature": "create_dataloader(dataset: Dataset, batch_size: int = 32, shuffle: bool = True, num_workers: int = 0, pin_memory: bool = True, drop_last: bool = False, collate_fn: Optional[callable] = None) -> DataLoader",
                "brief_description": "Create a DataLoader from a dataset with common configuration options"
            },
            {
                "name": "collate_fn",
                "signature": "collate_fn(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]",
                "brief_description": "Custom collate function for batching multimodal data"
            },
            {
                "name": "get_dataloaders",
                "signature": "get_dataloaders(train_data: Dict[str, torch.Tensor], val_data: Optional[Dict[str, torch.Tensor]] = None, test_data: Optional[Dict[str, torch.Tensor]] = None, batch_size: int = 32, num_workers: int = 0) -> Tuple[DataLoader, Optional[DataLoader], Optional[DataLoader]]",
                "brief_description": "Create DataLoaders for train, validation, and test sets"
            }
        ],
        "external_dependencies": ["torch", "numpy"],
        "complexity_score": 3  # Moderate complexity
    }
