"""MODULE: utils.py
PURPOSE: Provides utility functions and classes for tokenization and text processing tasks focused on transformer models.

KEY COMPONENTS:
- TransformerTextDataset: Dataset class for transformer text processing
- create_transformer_dataloaders: Function to create data loaders for transformer training
- transformer_collate_fn: Collate function for batching and padding text sequences

DEPENDENCIES:
- PyTorch (torch, torch.utils.data)
- Base tokenizers (BaseTokenizer)

SPECIAL NOTES:
- Includes efficient padding and batching mechanisms for variable-length sequences
- Supports both dictionary-based and list-based batch formats
"""

# src/data/tokenization/utils.py
from typing import List, Dict, Tuple, Optional, Union, Any
import torch
import torch.nn.functional as F  # Add this import
from torch.utils.data import Dataset, DataLoader
import os

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
            token_ids = token_ids[: self.max_length]
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


def transformer_collate_fn(
    batch: List[Union[Dict[str, torch.Tensor], List[int]]],
) -> Union[Dict[str, torch.Tensor], torch.Tensor]:
    """
    Collate function for transformer datasets.

    Args:
        batch: List of examples, each either a dict with input_ids and attention_mask
              or a list of token IDs.

    Returns:
        Either a dict with batched input_ids and attention_mask tensors,
        or a single batched tensor of token IDs.
    """
    if not batch:
        return {}

    # Check if batch contains dictionaries or lists
    is_dict_batch = isinstance(batch[0], dict)

    if is_dict_batch:
        # Handle dictionary inputs
        input_ids = [item["input_ids"] for item in batch]
        attention_masks = [item["attention_mask"] for item in batch]

        # Pad sequences
        max_len = max(seq.size(0) for seq in input_ids)
        padded_ids = torch.zeros((len(batch), max_len), dtype=input_ids[0].dtype)
        padded_masks = torch.zeros(
            (len(batch), max_len), dtype=attention_masks[0].dtype
        )

        for i, (ids, mask) in enumerate(zip(input_ids, attention_masks)):
            padded_ids[i, : ids.size(0)] = ids
            padded_masks[i, : mask.size(0)] = mask

        return {"input_ids": padded_ids, "attention_mask": padded_masks}
    else:
        # Handle list inputs
        # Pad sequences
        max_len = max(len(seq) for seq in batch)
        padded = torch.zeros((len(batch), max_len), dtype=torch.long)

        for i, seq in enumerate(batch):
            padded[i, : len(seq)] = torch.tensor(seq)

        return padded


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
        "module_purpose": "Provides utility functions and classes for tokenization and text processing in transformer models",
        "key_classes": [
            {
                "name": "TransformerTextDataset",
                "purpose": "PyTorch dataset for transformer text processing with support for BOS/EOS tokens and attention masks",
                "key_methods": [
                    {
                        "name": "__init__",
                        "signature": "__init__(self, texts: List[str], tokenizer: BaseTokenizer, max_length: Optional[int] = None, add_bos: bool = True, add_eos: bool = True, return_tensors: bool = True)",
                        "brief_description": "Initialize dataset with texts and tokenization parameters",
                    },
                    {
                        "name": "__getitem__",
                        "signature": "__getitem__(self, idx: int) -> Dict[str, Any]",
                        "brief_description": "Get a processed example with input_ids and attention_mask",
                    },
                ],
                "inheritance": "Dataset",
                "dependencies": ["torch.utils.data", ".base_tokenizer"],
            }
        ],
        "key_functions": [
            {
                "name": "create_transformer_dataloaders",
                "signature": "create_transformer_dataloaders(train_texts: List[str], tokenizer: BaseTokenizer, val_texts: Optional[List[str]] = None, batch_size: int = 32, max_length: Optional[int] = None, shuffle: bool = True, num_workers: int = 0) -> Tuple[DataLoader, Optional[DataLoader]]",
                "brief_description": "Create train and validation dataloaders for transformer training with appropriate collation",
            },
            {
                "name": "transformer_collate_fn",
                "signature": "transformer_collate_fn(batch: List[Union[Dict[str, torch.Tensor], List[int]]]) -> Union[Dict[str, torch.Tensor], torch.Tensor]",
                "brief_description": "Collate and pad batches of text sequences for transformer models",
            },
        ],
        "external_dependencies": ["torch", "torch.utils.data"],
        "complexity_score": 5,  # Medium complexity for standard dataset and dataloader functionality
    }
