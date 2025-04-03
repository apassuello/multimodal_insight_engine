# src/data/sequence_data.py
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Dict, List, Tuple, Optional, Any, Callable
import numpy as np

def transformer_collate_fn(batch: List[Dict[str, List[int]]], pad_idx: int) -> Dict[str, torch.Tensor]:
    """
    Collate function for transformer datasets.
    
    Args:
        batch: List of dictionaries with "src_tokens" and "tgt_tokens"
        pad_idx: Padding token index
        
    Returns:
        Dictionary with batched tensors
    """
    # Get max lengths in this batch
    max_src_len = max(len(item["src_tokens"]) for item in batch)
    max_tgt_len = max(len(item["tgt_tokens"]) for item in batch)
    
    # Create padded tensors
    src_batch = torch.full(
        (len(batch), max_src_len), pad_idx, dtype=torch.long
    )
    tgt_batch = torch.full(
        (len(batch), max_tgt_len), pad_idx, dtype=torch.long
    )
    
    # Fill tensors with actual tokens
    for i, item in enumerate(batch):
        src_tokens = item["src_tokens"]
        tgt_tokens = item["tgt_tokens"]
        
        if isinstance(src_tokens, torch.Tensor):
            src_batch[i, :len(src_tokens)] = src_tokens.clone().detach()
        else:
            src_batch[i, :len(src_tokens)] = torch.tensor(src_tokens, dtype=torch.long)
            
        if isinstance(tgt_tokens, torch.Tensor):
            tgt_batch[i, :len(tgt_tokens)] = tgt_tokens.clone().detach()
        else:
            tgt_batch[i, :len(tgt_tokens)] = torch.tensor(tgt_tokens, dtype=torch.long)
    
    return {
        "src": src_batch.to(torch.device("mps") if torch.backends.mps.is_available() else src_batch.device),
        "tgt": tgt_batch.to(torch.device("mps") if torch.backends.mps.is_available() else tgt_batch.device),
    }

class TransformerDataset(Dataset):
    """
    Dataset for transformer sequence-to-sequence tasks.
    
    This dataset handles tokenized source and target sequences for
    training transformer models.
    """
    
    def __init__(
        self,
        source_sequences: List[List[int]],
        target_sequences: List[List[int]],
        max_src_len: Optional[int] = None,
        max_tgt_len: Optional[int] = None,
        pad_idx: int = 0,
        bos_idx: Optional[int] = None,
        eos_idx: Optional[int] = None,
    ):
        """
        Initialize the transformer dataset.
        
        Args:
            source_sequences: List of tokenized source sequences
            target_sequences: List of tokenized target sequences
            max_src_len: Maximum source sequence length (truncate if longer)
            max_tgt_len: Maximum target sequence length (truncate if longer)
            pad_idx: Padding token index
            bos_idx: Beginning of sequence token index (will be added if provided)
            eos_idx: End of sequence token index (will be added if provided)
        """
        assert len(source_sequences) == len(target_sequences), "Source and target must have same length"
        
        self.source_sequences = source_sequences
        self.target_sequences = target_sequences
        self.max_src_len = max_src_len
        self.max_tgt_len = max_tgt_len
        self.pad_idx = pad_idx
        self.bos_idx = bos_idx
        self.eos_idx = eos_idx
    
    def __len__(self):
        return len(self.source_sequences)
    
    def __getitem__(self, idx):
        # Get raw sequences
        src_seq = self.source_sequences[idx]
        tgt_seq = self.target_sequences[idx]
        
        # Add BOS/EOS tokens if specified
        if self.bos_idx is not None:
            tgt_seq = [self.bos_idx] + tgt_seq
        if self.eos_idx is not None:
            tgt_seq = tgt_seq + [self.eos_idx]
        
        # Truncate if needed
        if self.max_src_len is not None:
            src_seq = src_seq[:self.max_src_len]
        if self.max_tgt_len is not None:
            tgt_seq = tgt_seq[:self.max_tgt_len]
        
        # Convert to tensors properly without triggering warnings
        src_tensor = torch.LongTensor(src_seq)
        tgt_tensor = torch.LongTensor(tgt_seq)
        
        # Move to appropriate device
        device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
        
        return {
            "src_tokens": src_tensor.to(device),
            "tgt_tokens": tgt_tensor.to(device),
        }

class TransformerCollator:
    """Collator class for transformer batches."""
    
    def __init__(self, pad_idx: int):
        """Initialize the collator.
        
        Args:
            pad_idx: Padding token index
        """
        self.pad_idx = pad_idx
    
    def __call__(self, batch: List[Dict[str, List[int]]]) -> Dict[str, torch.Tensor]:
        """Collate a batch of data.
        
        Args:
            batch: List of dictionaries with "src_tokens" and "tgt_tokens"
            
        Returns:
            Dictionary with batched tensors
        """
        return transformer_collate_fn(batch, self.pad_idx)

class TransformerDataModule:
    """
    Data module for transformer sequence-to-sequence tasks.
    
    This class handles loading, preprocessing, and batching data for
    transformer models.
    """
    
    def __init__(
        self,
        source_sequences: List[List[int]],
        target_sequences: List[List[int]],
        batch_size: int = 32,
        max_src_len: Optional[int] = None,
        max_tgt_len: Optional[int] = None,
        pad_idx: int = 0,
        bos_idx: Optional[int] = None,
        eos_idx: Optional[int] = None,
        val_split: float = 0.1,
        shuffle: bool = True,
        num_workers: int = 4,
    ):
        """
        Initialize the transformer data module.
        
        Args:
            source_sequences: List of tokenized source sequences
            target_sequences: List of tokenized target sequences
            batch_size: Batch size
            max_src_len: Maximum source sequence length
            max_tgt_len: Maximum target sequence length
            pad_idx: Padding token index
            bos_idx: Beginning of sequence token index
            eos_idx: End of sequence token index
            val_split: Fraction of data to use for validation
            shuffle: Whether to shuffle training data
            num_workers: Number of workers for data loading
        """
        self.source_sequences = source_sequences
        self.target_sequences = target_sequences
        self.batch_size = batch_size
        self.max_src_len = max_src_len
        self.max_tgt_len = max_tgt_len
        self.pad_idx = pad_idx
        self.bos_idx = bos_idx
        self.eos_idx = eos_idx
        self.val_split = val_split
        self.shuffle = shuffle
        self.num_workers = num_workers
        
        # Set up datasets and dataloaders
        self._setup()
    
    def _setup(self):
        """
        Set up datasets and dataloaders.
        """
        # Create train/validation split
        num_samples = len(self.source_sequences)
        indices = list(range(num_samples))
        
        if self.shuffle:
            np.random.shuffle(indices)
        
        val_size = int(self.val_split * num_samples)
        train_indices = indices[val_size:]
        val_indices = indices[:val_size]
        
        # Create train dataset
        train_src = [self.source_sequences[i] for i in train_indices]
        train_tgt = [self.target_sequences[i] for i in train_indices]
        
        self.train_dataset = TransformerDataset(
            source_sequences=train_src,
            target_sequences=train_tgt,
            max_src_len=self.max_src_len,
            max_tgt_len=self.max_tgt_len,
            pad_idx=self.pad_idx,
            bos_idx=self.bos_idx,
            eos_idx=self.eos_idx,
        )
        
        # Create validation dataset
        if val_size > 0:
            val_src = [self.source_sequences[i] for i in val_indices]
            val_tgt = [self.target_sequences[i] for i in val_indices]
            
            self.val_dataset = TransformerDataset(
                source_sequences=val_src,
                target_sequences=val_tgt,
                max_src_len=self.max_src_len,
                max_tgt_len=self.max_tgt_len,
                pad_idx=self.pad_idx,
                bos_idx=self.bos_idx,
                eos_idx=self.eos_idx,
            )
        else:
            self.val_dataset = None
        
        # Create collator
        collator = TransformerCollator(self.pad_idx)
        
        # Create dataloaders
        self.train_dataloader = DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=self.shuffle,
            num_workers=0,  # Disable multiprocessing for MPS
            collate_fn=collator,
            pin_memory=True,
        )
        
        if self.val_dataset is not None:
            self.val_dataloader = DataLoader(
                self.val_dataset,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=0,  # Disable multiprocessing for MPS
                collate_fn=collator,
                pin_memory=True,
            )
        else:
            self.val_dataloader = None
    
    def get_train_dataloader(self):
        """Get the training dataloader."""
        return self.train_dataloader
    
    def get_val_dataloader(self):
        """Get the validation dataloader."""
        return self.val_dataloader