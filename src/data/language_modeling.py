# src/data/language_modeling.py
import torch
from torch.utils.data import Dataset, DataLoader
from typing import List, Dict, Optional, Union
import random
from tqdm import tqdm

from .tokenization import BPETokenizer

class LanguageModelingDataset(Dataset):
    """
    Dataset for causal language modeling tasks.
    
    This dataset takes tokenized text and creates inputs/targets for
    next-token prediction tasks used in language model training.
    """
    
    def __init__(
        self,
        texts: List[str],
        tokenizer: BPETokenizer,
        max_length: int = 512,
        pad_idx: int = 0,
        bos_idx: int = 1,
        eos_idx: int = 2,
    ):
        """
        Initialize the language modeling dataset.
        
        Args:
            texts: List of text samples
            tokenizer: Tokenizer for encoding texts
            max_length: Maximum sequence length (will truncate longer sequences)
            pad_idx: Padding token index
            bos_idx: Beginning of sequence token index
            eos_idx: End of sequence token index
        """
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.pad_idx = pad_idx
        self.bos_idx = bos_idx
        self.eos_idx = eos_idx
        
        # Process all texts at initialization to avoid repeated processing
        print("Tokenizing dataset...")
        self.examples = []
        
        for text in tqdm(texts):
            # Tokenize text
            token_ids = self.tokenizer.encode(text)
            
            # Add BOS token at the beginning
            token_ids = [self.bos_idx] + token_ids
            
            # Truncate if needed, leaving room for EOS token
            if len(token_ids) >= self.max_length:
                token_ids = token_ids[:self.max_length-1]
            
            # Add EOS token at the end
            token_ids = token_ids + [self.eos_idx]
            
            # Store example
            self.examples.append(token_ids)
    
    def __len__(self) -> int:
        """Return the number of examples in the dataset."""
        return len(self.examples)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get an example from the dataset.
        
        For language modeling, the input is the sequence and the target
        is the same sequence shifted by one position (to predict the next token).
        
        Args:
            idx: Index of the example
            
        Returns:
            Dictionary with input_ids and labels
        """
        token_ids = self.examples[idx]
        
        # Create input sequence (all tokens)
        input_ids = token_ids
        
        # Create target sequence (all tokens except the first, plus padding)
        # This creates the next-token prediction task
        labels = token_ids[1:] + [self.pad_idx]
        
        # Convert to tensors
        input_ids = torch.tensor(input_ids, dtype=torch.long)
        labels = torch.tensor(labels, dtype=torch.long)
        
        return {
            "input_ids": input_ids,
            "labels": labels
        }

def lm_collate_fn(batch: List[Dict[str, torch.Tensor]], pad_idx: int) -> Dict[str, torch.Tensor]:
    """
    Collate function for language modeling batches.
    
    Args:
        batch: List of examples
        pad_idx: Padding token index
        
    Returns:
        Dictionary with batched and padded tensors
    """
    # Get max sequence length in this batch
    max_length = max(example["input_ids"].size(0) for example in batch)
    
    # Initialize padded tensors
    input_ids = torch.full((len(batch), max_length), pad_idx, dtype=torch.long)
    labels = torch.full((len(batch), max_length), -100, dtype=torch.long)  # -100 is ignored in loss
    attention_mask = torch.zeros((len(batch), max_length), dtype=torch.bool)
    
    # Fill tensors with data
    for i, example in enumerate(batch):
        seq_len = example["input_ids"].size(0)
        input_ids[i, :seq_len] = example["input_ids"]
        labels[i, :seq_len] = example["labels"]
        attention_mask[i, :seq_len] = 1  # 1 means attended to, 0 means masked
    
    return {
        "input_ids": input_ids,
        "labels": labels,
        "attention_mask": attention_mask
    }

def create_lm_dataloaders(
    texts: List[str],
    tokenizer: BPETokenizer,
    batch_size: int = 16,
    max_length: int = 512,
    val_split: float = 0.1,
    seed: int = 42,
) -> tuple:
    """
    Create training and validation dataloaders for language modeling.
    
    Args:
        texts: List of text samples
        tokenizer: Tokenizer for encoding texts
        batch_size: Batch size
        max_length: Maximum sequence length
        val_split: Fraction of data to use for validation
        seed: Random seed for reproducibility
        
    Returns:
        Tuple of (train_dataloader, val_dataloader)
    """
    # Set random seed for reproducibility
    random.seed(seed)
    
    # Split texts into train and validation sets
    val_size = int(len(texts) * val_split)
    indices = list(range(len(texts)))
    random.shuffle(indices)
    
    train_indices = indices[val_size:]
    val_indices = indices[:val_size]
    
    train_texts = [texts[i] for i in train_indices]
    val_texts = [texts[i] for i in val_indices]
    
    # Special token indices
    pad_idx = tokenizer.special_tokens["pad_token_idx"]
    bos_idx = tokenizer.special_tokens["bos_token_idx"]
    eos_idx = tokenizer.special_tokens["eos_token_idx"]
    
    # Create datasets
    train_dataset = LanguageModelingDataset(
        texts=train_texts,
        tokenizer=tokenizer,
        max_length=max_length,
        pad_idx=pad_idx,
        bos_idx=bos_idx,
        eos_idx=eos_idx,
    )
    
    val_dataset = LanguageModelingDataset(
        texts=val_texts,
        tokenizer=tokenizer,
        max_length=max_length,
        pad_idx=pad_idx,
        bos_idx=bos_idx,
        eos_idx=eos_idx,
    )
    
    # Create collate function with the padding index
    collate_fn = lambda batch: lm_collate_fn(batch, pad_idx)
    
    # Create dataloaders
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
    )
    
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
    )
    
    return train_dataloader, val_dataloader