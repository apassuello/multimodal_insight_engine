# src/data/tokenization/utils.py
from typing import List, Dict, Tuple, Optional, Union, Any
import torch
import torch.nn.functional as F  # Add this import
from torch.utils.data import Dataset, DataLoader

from .base_tokenizer import BaseTokenizer

class TransformerTextDataset(Dataset):
    """
    A PyTorch dataset for transformer text processing.
    
    This dataset handles tokenization and encoding of text data for
    transformer models.
    """
    
    def __init__(
        self,
        texts: List[str],
        tokenizer: BaseTokenizer,
        max_length: Optional[int] = None,
        add_bos: bool = True,
        add_eos: bool = True,
        return_tensors: bool = True,
    ):
        """
        Initialize the dataset.
        
        Args:
            texts: List of texts
            tokenizer: Tokenizer to use
            max_length: Maximum sequence length (None means no limit)
            add_bos: Whether to add beginning-of-sequence token
            add_eos: Whether to add end-of-sequence token
            return_tensors: Whether to return PyTorch tensors or lists
        """
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.add_bos = add_bos
        self.add_eos = add_eos
        self.return_tensors = return_tensors
        
        # Get special token indices
        special_tokens = tokenizer.special_tokens
        self.pad_idx = special_tokens.get("pad_token_idx", 0)
        self.bos_idx = special_tokens.get("bos_token_idx", 1)
        self.eos_idx = special_tokens.get("eos_token_idx", 2)
    
    def __len__(self) -> int:
        """Get the number of examples in the dataset."""
        return len(self.texts)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        Get an example from the dataset.
        
        Args:
            idx: Index of the example
            
        Returns:
            Dictionary containing 'input_ids' and 'attention_mask'
        """
        text = self.texts[idx]
        
        # Tokenize and encode
        token_ids = self.tokenizer.encode(text)
        
        # Add BOS token
        if self.add_bos:
            token_ids = [self.bos_idx] + token_ids
        
        # Add EOS token
        if self.add_eos:
            token_ids = token_ids + [self.eos_idx]
        
        # Truncate if needed
        if self.max_length is not None and len(token_ids) > self.max_length:
            token_ids = token_ids[:self.max_length]
            if self.add_eos:
                token_ids[-1] = self.eos_idx
        
        # Create attention mask (1 for tokens, 0 for padding)
        attention_mask = [1] * len(token_ids)
        
        # Convert to tensors if requested
        if self.return_tensors:
            token_ids = torch.tensor(token_ids, dtype=torch.long)
            attention_mask = torch.tensor(attention_mask, dtype=torch.long)
        
        return {
            "input_ids": token_ids,
            "attention_mask": attention_mask,
        }


def create_transformer_dataloaders(
    train_texts: List[str],
    tokenizer: BaseTokenizer,
    val_texts: Optional[List[str]] = None,
    batch_size: int = 32,
    max_length: Optional[int] = None,
    shuffle: bool = True,
    num_workers: int = 0,
) -> Tuple[DataLoader, Optional[DataLoader]]:
    """
    Create DataLoaders for transformer training.
    
    Args:
        train_texts: List of training texts
        tokenizer: Tokenizer to use
        val_texts: Optional list of validation texts
        batch_size: Batch size
        max_length: Maximum sequence length
        shuffle: Whether to shuffle the training data
        num_workers: Number of worker processes for data loading
        
    Returns:
        Tuple of (train_dataloader, val_dataloader)
    """
    # Create datasets
    train_dataset = TransformerTextDataset(
        texts=train_texts,
        tokenizer=tokenizer,
        max_length=max_length,
        add_bos=True,
        add_eos=True,
        return_tensors=True,
    )
    
    # Create training dataloader
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=transformer_collate_fn,
    )
    
    # Create validation dataloader if validation texts are provided
    val_dataloader = None
    if val_texts is not None:
        val_dataset = TransformerTextDataset(
            texts=val_texts,
            tokenizer=tokenizer,
            max_length=max_length,
            add_bos=True,
            add_eos=True,
            return_tensors=True,
        )
        
        val_dataloader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            collate_fn=transformer_collate_fn,
        )
    
    return train_dataloader, val_dataloader


def transformer_collate_fn(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    """
    Collate function for transformer batches.
    
    Args:
        batch: List of examples
        
    Returns:
        Batched tensors with padding
    """
    # Get batch size and determine maximum length
    batch_size = len(batch)
    if batch_size == 0:
        return {}
    
    # Determine maximum length in this batch
    max_length = max(example["input_ids"].size(0) for example in batch)
    
    # Get padding index from first example
    pad_idx = 0  # Default
    if hasattr(batch[0]["input_ids"], "new_full"):
        # This is a tensor, so we can create a new one with the same dtype and device
        input_ids = torch.stack([
            F.pad(
                example["input_ids"],
                (0, max_length - example["input_ids"].size(0)),
                value=pad_idx
            )
            for example in batch
        ])
        
        attention_mask = torch.stack([
            F.pad(
                example["attention_mask"],
                (0, max_length - example["attention_mask"].size(0)),
                value=0
            )
            for example in batch
        ])
    else:
        # This is a list, so we'll need to handle it differently
        input_ids = [
            example["input_ids"] + [pad_idx] * (max_length - len(example["input_ids"]))
            for example in batch
        ]
        
        attention_mask = [
            example["attention_mask"] + [0] * (max_length - len(example["attention_mask"]))
            for example in batch
        ]
        
        # Convert to tensors
        input_ids = torch.tensor(input_ids, dtype=torch.long)
        attention_mask = torch.tensor(attention_mask, dtype=torch.long)
    
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
    }