"""
Fixed semantic batch sampler implementation for multimodal training.

This module implements a semantic batch sampler that ensures each batch contains
multiple examples with the same semantic meaning (same match_id), which is crucial
for effective contrastive learning.
"""

import torch
import random
import logging
from torch.utils.data import Sampler, Dataset
from collections import defaultdict
from typing import List, Dict, Optional, Iterator, Any

logger = logging.getLogger(__name__)


class FixedSemanticBatchSampler(Sampler):
    """
    A batch sampler that ensures each batch contains multiple examples with the same match_id.
    This is critical for contrastive learning to have positive pairs within each batch.
    
    Improved with:
    - Robust match_id extraction from different dataset formats
    - Better handling of minimum samples per group requirement
    - More balanced batch composition
    - Detailed logging for debugging
    """
    
    def __init__(
        self,
        dataset: Dataset,
        batch_size: int,
        min_samples_per_group: int = 5,
        max_samples_per_group: Optional[int] = None,
        shuffle: bool = True,
        drop_last: bool = True,
        verbose: bool = False
    ):
        """
        Initialize the semantic batch sampler.
        
        Args:
            dataset: Dataset to sample from
            batch_size: Size of each batch
            min_samples_per_group: Minimum samples per semantic group in a batch
            max_samples_per_group: Maximum samples per semantic group in a batch
            shuffle: Whether to shuffle the dataset
            drop_last: Whether to drop the last incomplete batch
            verbose: Whether to print detailed debug information
        """
        self.dataset = dataset
        self.batch_size = batch_size
        self.min_samples_per_group = min_samples_per_group
        self.max_samples_per_group = max_samples_per_group or batch_size // 2
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.verbose = verbose
        
        # Extract match_ids from the dataset
        self.match_ids = self._extract_match_ids()
        
        # Group indices by match_id
        self.grouped_indices = self._group_indices_by_match_id()
        
        # Filter groups to ensure they have enough samples
        self.valid_groups = self._filter_valid_groups()
        
        if self.verbose:
            logger.info(f"Found {len(self.grouped_indices)} total semantic groups")
            logger.info(f"Found {len(self.valid_groups)} valid groups with at least {self.min_samples_per_group} samples")
            
            # Show distribution of group sizes
            group_sizes = [len(indices) for indices in self.valid_groups.values()]
            if group_sizes:
                logger.info(f"Group sizes - Min: {min(group_sizes)}, Max: {max(group_sizes)}, Avg: {sum(group_sizes)/len(group_sizes):.1f}")
        
        # Build batches
        self.batches = self._build_batches()
        
        if self.verbose:
            logger.info(f"Created {len(self.batches)} batches with semantic grouping")
    
    def _extract_match_ids(self) -> List[str]:
        """Extract match_ids from the dataset."""
        match_ids = []
        
        # Try different extraction methods
        
        # Method 1: Check if the dataset has match_ids attribute
        if hasattr(self.dataset, "match_ids"):
            if self.verbose:
                logger.info("Extracting match_ids from dataset.match_ids attribute")
            return self.dataset.match_ids
        
        # Method 2: Check if the dataset has get_match_ids method
        if hasattr(self.dataset, "get_match_ids") and callable(self.dataset.get_match_ids):
            if self.verbose:
                logger.info("Extracting match_ids using dataset.get_match_ids() method")
            return self.dataset.get_match_ids()
        
        # Method 3: Try to extract from dataset items
        try:
            # Check first item to see if it has match_id
            first_item = self.dataset[0]
            if isinstance(first_item, dict) and "match_id" in first_item:
                if self.verbose:
                    logger.info("Extracting match_ids from each dataset item")
                
                for i in range(len(self.dataset)):
                    try:
                        item = self.dataset[i]
                        match_id = item.get("match_id", f"id_{i}")
                        # Ensure match_id is a string for consistent processing
                        if not isinstance(match_id, str):
                            match_id = str(match_id)
                        match_ids.append(match_id)
                    except Exception as e:
                        logger.warning(f"Error accessing item {i}: {e}")
                        match_ids.append(f"id_{i}")  # Fallback
                
                return match_ids
        except Exception as e:
            logger.warning(f"Error extracting match_ids from items: {e}")
        
        # Fallback: use indices as match_ids
        if self.verbose:
            logger.warning("Using fallback: indices as match_ids")
        return [f"id_{i}" for i in range(len(self.dataset))]
    
    def _group_indices_by_match_id(self) -> Dict[str, List[int]]:
        """Group dataset indices by their match_id."""
        grouped_indices = defaultdict(list)
        
        for idx, match_id in enumerate(self.match_ids):
            grouped_indices[match_id].append(idx)
        
        return dict(grouped_indices)
    
    def _filter_valid_groups(self) -> Dict[str, List[int]]:
        """Filter groups to ensure they have at least min_samples_per_group samples."""
        valid_groups = {}
        
        for match_id, indices in self.grouped_indices.items():
            if len(indices) >= self.min_samples_per_group:
                valid_groups[match_id] = indices.copy()
        
        return valid_groups
    
    def _build_batches(self) -> List[List[int]]:
        """Build batches by grouping examples with the same match_id."""
        batches = []
        
        # Get list of group ids
        group_ids = list(self.valid_groups.keys())
        
        # Shuffle group ids if requested
        if self.shuffle:
            random.shuffle(group_ids)
        
        # Keep track of available groups
        available_groups = {group_id: self.valid_groups[group_id].copy() for group_id in group_ids}
        
        # Shuffle indices within each group
        if self.shuffle:
            for indices in available_groups.values():
                random.shuffle(indices)
        
        # Create batches
        while available_groups:
            # Create a new batch
            batch = []
            
            # Get groups that still have enough samples
            valid_group_ids = [
                gid for gid, indices in available_groups.items() 
                if len(indices) >= self.min_samples_per_group
            ]
            
            # If no groups have enough samples, we're done
            if not valid_group_ids:
                break
            
            # Shuffle the valid groups for better mixing
            if self.shuffle:
                random.shuffle(valid_group_ids)
            
            # Keep track of which groups we've used in this batch
            used_groups = []
            
            # Fill the batch with samples from different groups
            while batch_size_left := (self.batch_size - len(batch)):
                # If no valid groups left, break
                if not valid_group_ids:
                    break
                
                # Select a group
                group_id = valid_group_ids.pop(0)
                used_groups.append(group_id)
                
                # Calculate how many samples to take from this group
                # Take minimum of:
                # 1. What's needed to fill the batch
                # 2. How many samples are available in the group
                # 3. The maximum samples per group parameter
                samples_to_take = min(
                    batch_size_left,
                    len(available_groups[group_id]),
                    self.max_samples_per_group
                )
                
                # Make sure we're not taking too few samples
                if samples_to_take < self.min_samples_per_group:
                    # If we can't take enough samples, skip this group
                    continue
                
                # Take samples from the group
                group_samples = available_groups[group_id][:samples_to_take]
                available_groups[group_id] = available_groups[group_id][samples_to_take:]
                batch.extend(group_samples)
                
                # If the group doesn't have enough samples left, remove it
                if len(available_groups[group_id]) < self.min_samples_per_group:
                    available_groups.pop(group_id)
            
            # If we couldn't fill the batch, and we're dropping last, skip it
            if len(batch) < self.batch_size and self.drop_last:
                continue
            
            # If using drop_last=False, we need to pad the batch
            if len(batch) < self.batch_size and not self.drop_last:
                # Pad with random samples from used groups
                padding_needed = self.batch_size - len(batch)
                all_used_samples = []
                for gid in used_groups:
                    if gid in self.valid_groups:
                        all_used_samples.extend(self.valid_groups[gid])
                
                if all_used_samples:
                    padding = random.choices(all_used_samples, k=padding_needed)
                    batch.extend(padding)
            
            # Add batch to list of batches
            batches.append(batch)
            
            # Remove empty groups
            available_groups = {
                gid: indices for gid, indices in available_groups.items() 
                if indices
            }
        
        return batches
    
    def __iter__(self) -> Iterator[List[int]]:
        """Iterate through the batches."""
        # Shuffle batches for each epoch if requested
        if self.shuffle:
            random.shuffle(self.batches)
        
        for batch in self.batches:
            yield batch
    
    def __len__(self) -> int:
        """Return the number of batches."""
        return len(self.batches)


def create_semantic_dataloader(
    dataset: Dataset,
    batch_size: int,
    min_samples_per_group: int = 5,
    shuffle: bool = True,
    num_workers: int = 0,
    verbose: bool = False
) -> torch.utils.data.DataLoader:
    """
    Create a DataLoader with semantic batch sampling.
    
    Args:
        dataset: Dataset to load
        batch_size: Batch size
        min_samples_per_group: Minimum samples per semantic group
        shuffle: Whether to shuffle the dataset
        num_workers: Number of workers for data loading
        verbose: Whether to print detailed information
        
    Returns:
        DataLoader with semantic batch sampling
    """
    # Create semantic batch sampler
    batch_sampler = FixedSemanticBatchSampler(
        dataset=dataset,
        batch_size=batch_size,
        min_samples_per_group=min_samples_per_group,
        shuffle=shuffle,
        drop_last=True,
        verbose=verbose
    )
    
    # Create DataLoader with the batch sampler
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_sampler=batch_sampler,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return dataloader