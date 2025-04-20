# src/data/multimodal_data_utils.py
"""
Utilities for multimodal data loading and processing.

This module provides functions for loading, processing, and preparing
multimodal data for training and evaluation.
"""

import os
import torch
import random
import logging
import numpy as np
from typing import Dict, List, Tuple, Any, Optional, Iterator
from torch.utils.data import DataLoader, SubsetRandomSampler, Sampler, BatchSampler
from collections import defaultdict

from .multimodal_dataset import EnhancedMultimodalDataset

logger = logging.getLogger(__name__)


class SemanticGroupBatchSampler(BatchSampler):
    """
    Custom batch sampler that ensures each batch contains multiple examples from the same semantic groups.
    
    This is crucial for contrastive learning, as it ensures each batch has positive pairs for comparison.
    Without semantic grouping, the contrastive learning process becomes ineffective as most batches won't
    contain related samples to learn meaningful similarities.
    
    Args:
        dataset: Dataset containing samples with match_ids
        batch_size: Size of each batch
        drop_last: Whether to drop the last incomplete batch
        min_samples_per_group: Minimum number of samples required from each group per batch
        groups_per_batch: Number of different semantic groups to include in each batch
    """
    
    def __init__(
        self,
        dataset: Any,
        batch_size: int,
        drop_last: bool = True,
        min_samples_per_group: int = 2,
        groups_per_batch: Optional[int] = None,
    ):
        self.dataset = dataset
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.min_samples_per_group = min_samples_per_group
        
        # Extract match_ids from dataset
        self.match_ids = self._get_match_ids()
        
        # Group samples by match_id
        self.groups = self._group_samples_by_match_id()
        
        # Filter groups to ensure they have at least min_samples_per_group
        self.valid_groups = {
            group_id: indices
            for group_id, indices in self.groups.items()
            if len(indices) >= self.min_samples_per_group
        }
        
        # Calculate how many groups to include per batch
        if groups_per_batch is None:
            # Default: aim to have batch_size / (2 * min_samples_per_group) different groups
            # This ensures we have enough samples per group while maintaining diversity
            self.groups_per_batch = max(1, batch_size // (2 * min_samples_per_group))
        else:
            self.groups_per_batch = groups_per_batch
            
        # Ensure we have enough valid groups
        if len(self.valid_groups) < 1:
            raise ValueError(
                f"No semantic groups have at least {self.min_samples_per_group} samples. "
                f"Consider reducing min_samples_per_group or providing more data."
            )
            
        logger.info(
            f"SemanticGroupBatchSampler initialized with {len(self.valid_groups)} valid semantic groups "
            f"(out of {len(self.groups)} total groups)"
        )
        logger.info(
            f"Each batch will contain ~{self.groups_per_batch} groups with at least "
            f"{self.min_samples_per_group} samples each"
        )
        
        # Calculate total length
        self._calculate_length()
    
    def _get_match_ids(self) -> List[str]:
        """Extract match_ids from the dataset."""
        # Try different approaches to get match_ids
        if hasattr(self.dataset, "match_ids"):
            return self.dataset.match_ids
        elif hasattr(self.dataset, "get_match_ids") and callable(self.dataset.get_match_ids):
            return self.dataset.get_match_ids()
        else:
            # Try to extract from each item
            try:
                # Check first item to see if it contains match_id
                first_item = self.dataset[0]
                if isinstance(first_item, dict) and "match_id" in first_item:
                    match_ids = []
                    for i in range(len(self.dataset)):
                        item = self.dataset[i]
                        match_ids.append(item.get("match_id", f"id_{i}"))
                    return match_ids
            except Exception as e:
                logger.warning(f"Error extracting match_ids from items: {e}")
            
            # Fallback: use indices as unique match_ids
            logger.warning("Could not find match_ids in dataset, using indices as fallback")
            return [f"id_{i}" for i in range(len(self.dataset))]
    
    def _group_samples_by_match_id(self) -> Dict[str, List[int]]:
        """Group sample indices by their match_id."""
        groups = defaultdict(list)
        for idx, match_id in enumerate(self.match_ids):
            groups[match_id].append(idx)
        return dict(groups)
    
    def _calculate_length(self):
        """Calculate the number of batches that will be generated."""
        # Calculate how many complete batches we can make
        if self.drop_last:
            # We need to estimate how many batches we can create with our constraints
            total_valid_samples = sum(len(indices) for indices in self.valid_groups.values())
            # Approximate number of batches we can create
            self.length = total_valid_samples // self.batch_size
        else:
            # With drop_last=False, we need to account for the last incomplete batch
            total_valid_samples = sum(len(indices) for indices in self.valid_groups.values())
            self.length = (total_valid_samples + self.batch_size - 1) // self.batch_size
    
    def __iter__(self) -> Iterator[List[int]]:
        """Iterate through batches that maintain semantic grouping."""
        # Shuffle groups for each epoch
        group_ids = list(self.valid_groups.keys())
        random.shuffle(group_ids)
        
        # Create a copy of groups data to modify during iteration
        groups_data = {
            group_id: indices.copy() 
            for group_id, indices in self.valid_groups.items()
        }
        
        # Shuffle samples within each group
        for indices in groups_data.values():
            random.shuffle(indices)
        
        # Keep track of which groups still have enough samples
        active_groups = set(group_ids)
        
        # Generate batches
        while active_groups and sum(len(groups_data[g]) for g in active_groups) >= self.batch_size:
            batch = []
            
            # Select groups for this batch (up to groups_per_batch)
            batch_groups = random.sample(
                list(active_groups),
                min(self.groups_per_batch, len(active_groups))
            )
            
            # Calculate samples per group for this batch
            # Start by ensuring min_samples_per_group for each selected group
            remaining_space = self.batch_size - self.min_samples_per_group * len(batch_groups)
            
            # Distribute remaining space proportionally based on group sizes
            if remaining_space > 0:
                group_sizes = {g: len(groups_data[g]) for g in batch_groups}
                total_size = sum(group_sizes.values())
                
                # Calculate extra samples for each group, ensuring each gets at least min_samples_per_group
                extra_samples = {}
                for g in batch_groups:
                    # Proportional allocation of remaining samples
                    share = int(remaining_space * group_sizes[g] / total_size) if total_size > 0 else 0
                    # Ensure we don't take more samples than available
                    extra_samples[g] = min(share, group_sizes[g] - self.min_samples_per_group)
                
                # Distribute any remaining samples
                leftover = remaining_space - sum(extra_samples.values())
                sorted_groups = sorted(
                    batch_groups, 
                    key=lambda g: (extra_samples[g], -len(groups_data[g]))
                )
                for g in sorted_groups:
                    if leftover <= 0:
                        break
                    available = min(leftover, group_sizes[g] - self.min_samples_per_group - extra_samples[g])
                    extra_samples[g] += available
                    leftover -= available
                
                # Final samples per group
                samples_per_group = {
                    g: self.min_samples_per_group + extra_samples[g] 
                    for g in batch_groups
                }
            else:
                # If batch size is too small, just distribute evenly
                samples_per_group = {
                    g: min(self.min_samples_per_group, len(groups_data[g]))
                    for g in batch_groups
                }
            
            # Take samples from each group
            for group_id, num_samples in samples_per_group.items():
                # Take samples from the group
                group_samples = groups_data[group_id][:num_samples]
                groups_data[group_id] = groups_data[group_id][num_samples:]
                batch.extend(group_samples)
                
                # If group has too few samples left, remove it from active groups
                if len(groups_data[group_id]) < self.min_samples_per_group:
                    active_groups.remove(group_id)
            
            # Handle case where we couldn't fill the batch
            if len(batch) < self.batch_size:
                # If we're dropping last, only yield if we have enough samples
                if not self.drop_last or len(batch) >= self.batch_size:
                    yield batch
            else:
                # Yield the batch
                yield batch
    
    def __len__(self) -> int:
        """Return the number of batches."""
        return self.length


def randomize_dataset_positions(dataset: Any) -> List[int]:
    """
    Randomize the positions of items in the dataset to break positional correlation.
    
    This function is critical for preventing shortcut learning in contrastive training
    by ensuring that matching items don't occupy the same positions across batches.
    
    Args:
        dataset: Dataset containing items with match_ids
        
    Returns:
        List of indices in randomized order
    """
    # First, make sure we have access to match_ids
    # Try different approaches to get match_ids
    match_ids = None

    # Try to access match_ids as an attribute
    if hasattr(dataset, "match_ids"):
        match_ids = dataset.match_ids
    # Try to access through a method
    elif hasattr(dataset, "get_match_ids") and callable(dataset.get_match_ids):
        match_ids = dataset.get_match_ids()
    # Try to extract from each item
    else:
        print("Attempting to extract match_ids from dataset items...")
        # Check if we can access items and if they have match_id
        try:
            # Check first item
            first_item = dataset[0]
            if isinstance(first_item, dict) and "match_id" in first_item:
                # Extract match_ids from all items
                match_ids = []
                for i in range(len(dataset)):
                    item = dataset[i]
                    match_ids.append(item.get("match_id", f"id_{i}"))
                print(
                    f"Successfully extracted {len(match_ids)} match_ids from items"
                )
        except Exception as e:
            print(f"Error extracting match_ids from items: {e}")

    # If we still don't have match_ids, use default
    if match_ids is None:
        print("WARNING: Couldn't access match_ids, using fallback with unique IDs")
        match_ids = [f"id_{i}" for i in range(len(dataset))]

    # Store match_ids in the dataset for future reference
    if not hasattr(dataset, "match_ids"):
        dataset.match_ids = match_ids

    # Group indices by match_id
    match_id_groups = {}
    for idx, match_id in enumerate(match_ids):
        if match_id not in match_id_groups:
            match_id_groups[match_id] = []
        match_id_groups[match_id].append(idx)

    print(
        f"Found {len(match_id_groups)} match groups in dataset with {len(dataset)} items"
    )

    # Create shuffled indices that preserve semantic relationships but break position
    shuffled_indices = []
    # Mix up groups as much as possible
    group_keys = list(match_id_groups.keys())
    random.shuffle(group_keys)

    # For each group, randomize indices within the group
    for match_id in group_keys:
        indices = match_id_groups[match_id]
        random.shuffle(indices)
        shuffled_indices.extend(indices)

    print(f"Created shuffled indices list with {len(shuffled_indices)} items")
    return shuffled_indices


def create_data_loaders(
    args: Any,
    image_preprocessor: Any,
    tokenizer: Any
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create data loaders for training, validation, and testing.

    Args:
        args: Command line arguments with options including:
            - dataset: Dataset name ('flickr30k', 'custom', 'synthetic')
            - use_synthetic: Whether to use synthetic data
            - captions_per_image: Number of captions to use per image (1-5)
        image_preprocessor: Image preprocessor
        tokenizer: Text tokenizer

    Returns:
        Train, validation, and test data loaders
    """
    print(f"Creating data loaders for {args.dataset} dataset...")

    # Create dataset and data loaders
    if args.dataset == "flickr30k":
        if args.use_synthetic:
            print("WARNING: Using synthetic data instead of real Flickr30k data!")
            synthetic_samples = args.synthetic_samples
        else:
            # When using real data, don't specify synthetic_samples
            synthetic_samples = args.synthetic_samples if args.use_synthetic else 0

        # Try to create Flickr30k dataset splits
        try:
            print("Loading Flickr30k train split...")
            # Get captions_per_image from args if available, otherwise default to 1
            captions_per_image = getattr(args, 'captions_per_image', 1)
            
            train_dataset = EnhancedMultimodalDataset(
                dataset_name="flickr30k",
                split="train",
                image_preprocessor=image_preprocessor,
                tokenizer=tokenizer,
                max_text_length=args.max_text_length,
                synthetic_samples=synthetic_samples,
                cache_dir=os.path.join(args.data_dir, "flickr30k"),
                max_samples=args.max_train_examples,
                captions_per_image=captions_per_image,
            )

            # Check if we actually got real data or synthetic fallback
            dataset_info = train_dataset.get_split_proportions()
            if (
                dataset_info.get("total_samples", 0) <= args.synthetic_samples
                and not args.use_synthetic
            ):
                print(
                    f"WARNING: Got {dataset_info.get('total_samples', 0)} samples which may indicate synthetic data fallback"
                )
                print(
                    "If you want to use synthetic data explicitly, use --use_synthetic flag"
                )

            print("Loading Flickr30k validation split...")
            val_dataset = EnhancedMultimodalDataset(
                dataset_name="flickr30k",
                split="val",
                image_preprocessor=image_preprocessor,
                tokenizer=tokenizer,
                max_text_length=args.max_text_length,
                synthetic_samples=(
                    synthetic_samples // 4 if synthetic_samples > 0 else 0
                ),
                cache_dir=os.path.join(args.data_dir, "flickr30k"),
                max_samples=args.max_val_examples,
                captions_per_image=captions_per_image,
            )

            print("Loading Flickr30k test split...")
            test_dataset = EnhancedMultimodalDataset(
                dataset_name="flickr30k",
                split="test",
                image_preprocessor=image_preprocessor,
                tokenizer=tokenizer,
                max_text_length=args.max_text_length,
                synthetic_samples=(
                    synthetic_samples // 4 if synthetic_samples > 0 else 0
                ),
                cache_dir=os.path.join(args.data_dir, "flickr30k"),
                max_samples=args.max_test_examples,
                captions_per_image=captions_per_image,
            )

        except Exception as e:
            print(f"ERROR: Failed to load Flickr30k dataset: {str(e)}")
            print(
                "If you intended to use synthetic data, please use --dataset synthetic or --use_synthetic flag"
            )
            raise

    elif args.dataset == "synthetic":
        print("Creating synthetic datasets for training demo")
        # Create synthetic datasets explicitly
        # Adjust the number of synthetic samples based on max_examples settings
        train_synthetic_samples = args.synthetic_samples
        if (
            args.max_train_examples is not None
            and args.max_train_examples < train_synthetic_samples
        ):
            train_synthetic_samples = args.max_train_examples

        val_synthetic_samples = args.synthetic_samples // 4
        if (
            args.max_val_examples is not None
            and args.max_val_examples < val_synthetic_samples
        ):
            val_synthetic_samples = args.max_val_examples

        test_synthetic_samples = args.synthetic_samples // 4
        if (
            args.max_test_examples is not None
            and args.max_test_examples < test_synthetic_samples
        ):
            test_synthetic_samples = args.max_test_examples

        train_dataset = EnhancedMultimodalDataset(
            dataset_name="synthetic",
            split="train",
            image_preprocessor=image_preprocessor,
            tokenizer=tokenizer,
            max_text_length=args.max_text_length,
            synthetic_samples=train_synthetic_samples,
            cache_dir=os.path.join(args.data_dir, "synthetic"),
            max_samples=args.max_train_examples,
        )

        val_dataset = EnhancedMultimodalDataset(
            dataset_name="synthetic",
            split="val",
            image_preprocessor=image_preprocessor,
            tokenizer=tokenizer,
            max_text_length=args.max_text_length,
            synthetic_samples=val_synthetic_samples,
            cache_dir=os.path.join(args.data_dir, "synthetic"),
            max_samples=args.max_val_examples,
        )

        test_dataset = EnhancedMultimodalDataset(
            dataset_name="synthetic",
            split="test",
            image_preprocessor=image_preprocessor,
            tokenizer=tokenizer,
            max_text_length=args.max_text_length,
            synthetic_samples=test_synthetic_samples,
            cache_dir=os.path.join(args.data_dir, "synthetic"),
            max_samples=args.max_test_examples,
        )
    else:
        # Custom dataset handling
        train_dataset = EnhancedMultimodalDataset(
            dataset_name="custom",
            split="train",
            image_preprocessor=image_preprocessor,
            tokenizer=tokenizer,
            max_text_length=args.max_text_length,
            synthetic_samples=args.synthetic_samples,
            cache_dir=os.path.join(args.data_dir, "custom"),
            max_samples=args.max_train_examples,
        )

        val_dataset = EnhancedMultimodalDataset(
            dataset_name="custom",
            split="val",
            image_preprocessor=image_preprocessor,
            tokenizer=tokenizer,
            max_text_length=args.max_text_length,
            synthetic_samples=args.synthetic_samples // 4,
            cache_dir=os.path.join(args.data_dir, "custom"),
            max_samples=args.max_val_examples,
        )

        test_dataset = EnhancedMultimodalDataset(
            dataset_name="custom",
            split="test",
            image_preprocessor=image_preprocessor,
            tokenizer=tokenizer,
            max_text_length=args.max_text_length,
            synthetic_samples=args.synthetic_samples // 4,
            cache_dir=os.path.join(args.data_dir, "custom"),
            max_samples=args.max_test_examples,
        )

    # Check if we need to use semantic grouping for batches
    # Get the setting from args or default to True for contrastive learning
    use_semantic_batching = getattr(args, 'use_semantic_batching', True)
    min_samples_per_group = getattr(args, 'min_samples_per_group', 2)
    
    if use_semantic_batching:
        print(f"Using SemanticGroupBatchSampler with min_samples_per_group={min_samples_per_group}")
        try:
            # Create batch samplers that maintain semantic relationships
            train_batch_sampler = SemanticGroupBatchSampler(
                dataset=train_dataset,
                batch_size=args.batch_size,
                drop_last=True,
                min_samples_per_group=min_samples_per_group
            )
            
            val_batch_sampler = SemanticGroupBatchSampler(
                dataset=val_dataset,
                batch_size=args.batch_size,
                drop_last=True,
                min_samples_per_group=min_samples_per_group
            )
            
            # Create data loaders with batch samplers
            train_loader = DataLoader(
                train_dataset,
                batch_sampler=train_batch_sampler,
                num_workers=0,  # Keep at 0 for MPS
            )
            
            val_loader = DataLoader(
                val_dataset,
                batch_sampler=val_batch_sampler,
                num_workers=0,  # Keep at 0 for MPS
            )
            
            print(f"SemanticGroupBatchSampler created {len(train_batch_sampler)} training batches")
            print(f"SemanticGroupBatchSampler created {len(val_batch_sampler)} validation batches")
            
        except Exception as e:
            logger.warning(f"Failed to create semantic batch samplers: {e}")
            logger.warning("Falling back to standard sampling")
            use_semantic_batching = False
    
    if not use_semantic_batching:
        # Fallback to traditional randomization
        print("Using traditional random sampling (without semantic grouping)")
        # Randomize our datasets to prevent shortcut learning
        print("Randomizing dataset positions to prevent shortcut learning...")
        train_indices = randomize_dataset_positions(train_dataset)
        val_indices = randomize_dataset_positions(val_dataset)

        # Use PyTorch's SubsetRandomSampler for simplicity
        train_sampler = SubsetRandomSampler(train_indices)
        val_sampler = SubsetRandomSampler(val_indices)

        # Create data loaders with these samplers
        train_loader = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            sampler=train_sampler,
            num_workers=0,  # Keep at 0 for MPS
            drop_last=True,  # Important for consistent batch sizes
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            sampler=val_sampler,
            num_workers=0,  # Keep at 0 for MPS
            drop_last=True,  # Consistent batch sizes for validation
        )

    # For testing we still use a standard loader without custom sampling
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,  # Important for reproducible testing
        num_workers=0,  # Keep at 0 for MPS
    )

    # Print dataset statistics
    print(f"Train split: {train_dataset.get_split_proportions()}")
    print(f"Val split: {val_dataset.get_split_proportions()}")
    print(f"Test split: {test_dataset.get_split_proportions()}")

    return train_loader, val_loader, test_loader


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
        "module_purpose": "Utilities for multimodal data loading and processing",
        "key_components": [
            {
                "name": "SemanticGroupBatchSampler",
                "signature": "class SemanticGroupBatchSampler(BatchSampler)",
                "brief_description": "Custom batch sampler that ensures semantic groups are maintained in each batch for contrastive learning"
            }
        ],
        "key_functions": [
            {
                "name": "randomize_dataset_positions",
                "signature": "randomize_dataset_positions(dataset: Any) -> List[int]",
                "brief_description": "Randomize dataset positions to prevent shortcut learning"
            },
            {
                "name": "create_data_loaders",
                "signature": "create_data_loaders(args: Any, image_preprocessor: Any, tokenizer: Any) -> Tuple[DataLoader, DataLoader, DataLoader]",
                "brief_description": "Create data loaders for training, validation, and testing with semantic grouping capabilities"
            }
        ],
        "external_dependencies": ["torch", "random", "logging", "numpy"],
        "complexity_score": 7  # Moderate to high complexity for semantic batch handling
    }