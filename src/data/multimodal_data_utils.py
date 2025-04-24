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
from dataclasses import dataclass

from .multimodal_dataset import EnhancedMultimodalDataset

logger = logging.getLogger(__name__)


# Key implementation details only, not full code
class SemanticBatchSampler(Sampler):
    # Existing implementation should be enhanced with:

    def __init__(self, match_ids, batch_size, min_samples_per_group=4):
        """
        Create semantically meaningful batches for contrastive learning.

        Args:
            match_ids: List of match IDs that define semantic groupings
            batch_size: Total batch size
            min_samples_per_group: Minimum samples per semantic group
        """
        self.match_ids = match_ids
        self.batch_size = batch_size
        self.min_samples_per_group = min_samples_per_group

        # Create semantic groups mapping
        self.semantic_groups = self._create_semantic_groups()

    def _create_semantic_groups(self):
        # Group samples by match_id
        semantic_groups = defaultdict(list)
        for idx, match_id in enumerate(self.match_ids):
            semantic_groups[match_id].append(idx)

        # Filter groups that are too small
        return {
            k: v
            for k, v in semantic_groups.items()
            if len(v) >= self.min_samples_per_group
        }

    def __iter__(self):
        # Select semantic groups until batch is filled
        # Ensure we select groups with multiple samples
        # Balance between variety and number of positives
        pass

    def __len__(self):
        # Calculate the number of batches that will be generated
        return len(self.semantic_groups)


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
        max_samples_per_group: Optional[int] = None,
        cap_strategy: str = "random",
        groups_per_batch: Optional[int] = None,
    ):
        """
        Initialize the semantic group batch sampler.
        
        Args:
            dataset: Dataset containing samples with match_ids
            batch_size: Size of each batch
            drop_last: Whether to drop the last incomplete batch
            min_samples_per_group: Minimum number of samples required from each group
            max_samples_per_group: Maximum number of samples to use from each group 
                                  (if None, no cap is applied)
            cap_strategy: Strategy for capping group size ('random' or 'split')
            groups_per_batch: Number of different semantic groups to include in each batch
        """
        self.dataset = dataset
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.min_samples_per_group = min_samples_per_group
        self.max_samples_per_group = max_samples_per_group
        self.cap_strategy = cap_strategy

        # Extract match_ids from dataset
        self.match_ids = self._get_match_ids()

        # Group samples by match_id
        self.groups = self._group_samples_by_match_id()

        # Apply group size capping if specified
        if self.max_samples_per_group is not None:
            if self.max_samples_per_group < self.min_samples_per_group:
                logger.warning(
                    f"max_samples_per_group ({self.max_samples_per_group}) is less than "
                    f"min_samples_per_group ({self.min_samples_per_group}). "
                    f"Setting max_samples_per_group to {self.min_samples_per_group}."
                )
                self.max_samples_per_group = self.min_samples_per_group
                
            # Track original group statistics before capping
            original_group_sizes = [len(indices) for indices in self.groups.values()]
            max_original = max(original_group_sizes) if original_group_sizes else 0
            avg_original = sum(original_group_sizes) / len(original_group_sizes) if original_group_sizes else 0
            
            # Apply capping strategy
            if self.cap_strategy == "random":
                self.groups = self._apply_random_capping()
            elif self.cap_strategy == "split":
                self.groups = self._apply_split_capping()
            else:
                logger.warning(f"Unknown cap_strategy: {self.cap_strategy}. Using 'random' instead.")
                self.groups = self._apply_random_capping()
                
            # Log capping statistics
            capped_group_sizes = [len(indices) for indices in self.groups.values()]
            max_capped = max(capped_group_sizes) if capped_group_sizes else 0
            avg_capped = sum(capped_group_sizes) / len(capped_group_sizes) if capped_group_sizes else 0
            
            logger.info(
                f"Applied {self.cap_strategy} capping: "
                f"Original max={max_original}, avg={avg_original:.2f} → "
                f"Capped max={max_capped}, avg={avg_capped:.2f}"
            )
            
            # Validate that capping worked - check there are no groups exceeding max_samples_per_group
            oversized_groups = sum(1 for size in capped_group_sizes if size > self.max_samples_per_group)
            if oversized_groups > 0:
                logger.warning(f"WARNING: {oversized_groups} groups still exceed max_samples_per_group limit!")
            else:
                logger.info(f"All groups successfully capped to max_samples_per_group={self.max_samples_per_group}")

        # Filter groups to ensure they have at least min_samples_per_group
        self.valid_groups = {
            group_id: indices
            for group_id, indices in self.groups.items()
            if len(indices) >= self.min_samples_per_group
        }

        # Calculate how many groups to include per batch
        if groups_per_batch is None:
            # Ensure we have at least 2 different groups per batch for better contrast
            self.groups_per_batch = max(2, batch_size // (2 * min_samples_per_group))
        else:
            self.groups_per_batch = groups_per_batch

        # Ensure we have enough valid groups
        if len(self.valid_groups) < 1:
            raise ValueError(
                f"No semantic groups have at least {min_samples_per_group} samples. "
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
        elif hasattr(self.dataset, "get_match_ids") and callable(
            self.dataset.get_match_ids
        ):
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
            logger.warning(
                "Could not find match_ids in dataset, using indices as fallback"
            )
            return [f"id_{i}" for i in range(len(self.dataset))]

    def _group_samples_by_match_id(self) -> Dict[str, List[int]]:
        """Group sample indices by their match_id."""
        groups = defaultdict(list)
        for idx, match_id in enumerate(self.match_ids):
            groups[match_id].append(idx)
        return dict(groups)
        
    def _apply_random_capping(self) -> Dict[str, List[int]]:
        """
        Cap group sizes by randomly sampling from each group.
        
        This approach:
        1. Keeps groups intact (doesn't create new groups)
        2. Randomly selects subset of samples from large groups
        3. Different samples may be selected each epoch (with shuffling)
        
        Returns:
            Dictionary of capped groups
        """
        capped_groups = {}
        
        for group_id, indices in self.groups.items():
            if len(indices) <= self.max_samples_per_group:
                # Group already within size limit
                capped_groups[group_id] = indices
            else:
                # Randomly sample from the group to cap its size
                # Use numpy for stable random sampling
                random_indices = np.random.choice(
                    len(indices), self.max_samples_per_group, replace=False
                )
                capped_groups[group_id] = [indices[i] for i in random_indices]
                
        return capped_groups
        
    def _apply_split_capping(self) -> Dict[str, List[int]]:
        """
        Cap group sizes by splitting large groups into multiple smaller groups.
        
        This approach:
        1. Preserves all samples (no data discarded)
        2. Creates new semantic groups for large groups
        3. Keeps each group under the max_samples_per_group limit
        
        Returns:
            Dictionary of capped and split groups
        """
        split_groups = {}
        
        for group_id, indices in self.groups.items():
            if len(indices) <= self.max_samples_per_group:
                # Group already within size limit
                split_groups[group_id] = indices
            else:
                # Split the group into multiple smaller groups
                # Shuffle indices first to ensure random distribution
                shuffled_indices = indices.copy()
                random.shuffle(shuffled_indices)
                
                # Calculate how many subgroups to create
                num_subgroups = (len(indices) + self.max_samples_per_group - 1) // self.max_samples_per_group
                
                # Create subgroups
                for i in range(num_subgroups):
                    start_idx = i * self.max_samples_per_group
                    end_idx = min((i + 1) * self.max_samples_per_group, len(shuffled_indices))
                    subgroup_indices = shuffled_indices[start_idx:end_idx]
                    
                    # Skip if this subgroup is too small
                    if len(subgroup_indices) < self.min_samples_per_group:
                        continue
                        
                    # Create a new subgroup ID
                    subgroup_id = f"{group_id}_split_{i}"
                    split_groups[subgroup_id] = subgroup_indices
                    
        return split_groups

    def _calculate_length(self):
        """Calculate the number of batches that will be generated."""
        # Calculate how many complete batches we can make
        if self.drop_last:
            # We need to estimate how many batches we can create with our constraints
            total_valid_samples = sum(
                len(indices) for indices in self.valid_groups.values()
            )
            # Approximate number of batches we can create
            self.length = total_valid_samples // self.batch_size
        else:
            # With drop_last=False, we need to account for the last incomplete batch
            total_valid_samples = sum(
                len(indices) for indices in self.valid_groups.values()
            )
            self.length = (total_valid_samples + self.batch_size - 1) // self.batch_size

    def __iter__(self) -> Iterator[List[int]]:
        """Iterate through batches that maintain semantic grouping."""
        # Shuffle groups for each epoch
        group_ids = list(self.valid_groups.keys())
        random.shuffle(group_ids)

        # Create a copy of groups data to modify during iteration
        groups_data = {
            group_id: indices.copy() for group_id, indices in self.valid_groups.items()
        }

        # Shuffle samples within each group
        for indices in groups_data.values():
            random.shuffle(indices)

        # Keep track of which groups still have enough samples
        active_groups = set(group_ids)

        # Generate batches
        while (
            active_groups
            and sum(len(groups_data[g]) for g in active_groups) >= self.batch_size
        ):
            batch = []

            # Select groups for this batch (up to groups_per_batch)
            # Ensure we select at least 2 different groups for better contrast
            num_groups_to_select = min(self.groups_per_batch, len(active_groups))
            if num_groups_to_select < 2 and len(active_groups) >= 2:
                num_groups_to_select = 2

            batch_groups = random.sample(list(active_groups), num_groups_to_select)

            # Calculate samples per group for this batch
            # Start by ensuring min_samples_per_group for each selected group
            remaining_space = self.batch_size - self.min_samples_per_group * len(
                batch_groups
            )

            # Distribute remaining space proportionally based on group sizes
            if remaining_space > 0:
                group_sizes = {g: len(groups_data[g]) for g in batch_groups}
                total_size = sum(group_sizes.values())

                # Calculate extra samples for each group
                extra_samples = {}
                for g in batch_groups:
                    # Proportional allocation of remaining samples
                    share = (
                        int(remaining_space * group_sizes[g] / total_size)
                        if total_size > 0
                        else 0
                    )
                    # Ensure we don't take more samples than available
                    extra_samples[g] = min(
                        share, group_sizes[g] - self.min_samples_per_group
                    )

                # Distribute any remaining samples
                leftover = remaining_space - sum(extra_samples.values())
                sorted_groups = sorted(
                    batch_groups, key=lambda g: (extra_samples[g], -len(groups_data[g]))
                )
                for g in sorted_groups:
                    if leftover <= 0:
                        break
                    available = min(
                        leftover,
                        group_sizes[g] - self.min_samples_per_group - extra_samples[g],
                    )
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
                print(f"Successfully extracted {len(match_ids)} match_ids from items")
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
    args: Any, image_preprocessor: Any, tokenizer: Any
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
            captions_per_image = getattr(args, "captions_per_image", 1)

            # Pass through the semantic grouping parameters
            min_samples_per_group = getattr(args, "min_samples_per_group", 2)
            max_samples_per_group = getattr(args, "max_samples_per_group", None)
            cap_strategy = getattr(args, "cap_strategy", "random")
            
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
                min_samples_per_group=min_samples_per_group,
                max_samples_per_group=max_samples_per_group,
                cap_strategy=cap_strategy,
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
                min_samples_per_group=min_samples_per_group,
                max_samples_per_group=max_samples_per_group,
                cap_strategy=cap_strategy,
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
    use_semantic_batching = getattr(args, "use_semantic_batching", True)
    min_samples_per_group = getattr(args, "min_samples_per_group", 2)
    max_samples_per_group = getattr(args, "max_samples_per_group", None)
    cap_strategy = getattr(args, "cap_strategy", "random") 

    if use_semantic_batching:
        # Log the configuration settings
        if max_samples_per_group is not None:
            print(
                f"Using SemanticGroupBatchSampler with min_samples_per_group={min_samples_per_group}, "
                f"max_samples_per_group={max_samples_per_group}, cap_strategy='{cap_strategy}'"
            )
        else:
            print(
                f"Using SemanticGroupBatchSampler with min_samples_per_group={min_samples_per_group}"
            )
            
        try:
            # Create batch samplers that maintain semantic relationships
            train_batch_sampler = SemanticGroupBatchSampler(
                dataset=train_dataset,
                batch_size=args.batch_size,
                drop_last=True,
                min_samples_per_group=min_samples_per_group,
                max_samples_per_group=max_samples_per_group,
                cap_strategy=cap_strategy,
            )

            val_batch_sampler = SemanticGroupBatchSampler(
                dataset=val_dataset,
                batch_size=args.batch_size,
                drop_last=True,
                min_samples_per_group=min_samples_per_group,
                max_samples_per_group=max_samples_per_group,
                cap_strategy=cap_strategy,
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

            print(
                f"SemanticGroupBatchSampler created {len(train_batch_sampler)} training batches"
            )
            print(
                f"SemanticGroupBatchSampler created {len(val_batch_sampler)} validation batches"
            )

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
                "brief_description": "Custom batch sampler that ensures semantic groups are maintained in each batch for contrastive learning",
            }
        ],
        "key_functions": [
            {
                "name": "randomize_dataset_positions",
                "signature": "randomize_dataset_positions(dataset: Any) -> List[int]",
                "brief_description": "Randomize dataset positions to prevent shortcut learning",
            },
            {
                "name": "create_data_loaders",
                "signature": "create_data_loaders(args: Any, image_preprocessor: Any, tokenizer: Any) -> Tuple[DataLoader, DataLoader, DataLoader]",
                "brief_description": "Create data loaders for training, validation, and testing with semantic grouping capabilities",
            },
        ],
        "external_dependencies": ["torch", "random", "logging", "numpy"],
        "complexity_score": 7,  # Moderate to high complexity for semantic batch handling
    }


@dataclass
class FeatureStats:
    mean: torch.Tensor
    std: torch.Tensor
    count: int


class MultimodalDataset:
    def __init__(
        self,
        vision_data: Dict[str, torch.Tensor],
        text_data: Dict[str, torch.Tensor],
        match_ids: List[str],
        feature_dim: int = 512,
        diversity_weight: float = 0.1,
        min_group_size: int = 2,
        max_group_size: int = 10,
    ):
        """
        Initialize the enhanced multimodal dataset with feature diversity and semantic grouping.

        Args:
            vision_data: Dictionary mapping IDs to vision feature tensors
            text_data: Dictionary mapping IDs to text feature tensors
            match_ids: List of IDs indicating matching pairs
            feature_dim: Dimension of the feature vectors
            diversity_weight: Weight for diversity loss in feature computation
            min_group_size: Minimum size for semantic groups
            max_group_size: Maximum size for semantic groups
        """
        self.vision_data = vision_data
        self.text_data = text_data
        self.match_ids = match_ids
        self.feature_dim = feature_dim
        self.diversity_weight = diversity_weight
        self.min_group_size = min_group_size
        self.max_group_size = max_group_size

        # Initialize feature statistics
        self.vision_stats = FeatureStats(
            mean=torch.zeros(feature_dim), std=torch.ones(feature_dim), count=0
        )
        self.text_stats = FeatureStats(
            mean=torch.zeros(feature_dim), std=torch.ones(feature_dim), count=0
        )

        # Initialize semantic groups
        self.semantic_groups = defaultdict(list)

        # Update statistics and create initial groups
        self._update_feature_statistics()
        self._create_semantic_groups()

    def _update_feature_statistics(self) -> None:
        """Update running statistics for vision and text features."""
        # Update vision statistics
        vision_features = torch.stack(list(self.vision_data.values()))
        self.vision_stats.mean = torch.mean(vision_features, dim=0)
        self.vision_stats.std = torch.std(vision_features, dim=0)
        self.vision_stats.count = len(vision_features)

        # Update text statistics
        text_features = torch.stack(list(self.text_data.values()))
        self.text_stats.mean = torch.mean(text_features, dim=0)
        self.text_stats.std = torch.std(text_features, dim=0)
        self.text_stats.count = len(text_features)

    def _compute_feature_diversity(self, features: torch.Tensor) -> torch.Tensor:
        """
        Compute diversity score for a set of features.

        Args:
            features: Tensor of shape [num_features, feature_dim]

        Returns:
            Diversity score tensor
        """
        # Normalize features
        normalized_features = (features - self.vision_stats.mean) / (
            self.vision_stats.std + 1e-6
        )

        # Compute pairwise cosine similarities
        similarities = torch.mm(normalized_features, normalized_features.t())

        # Compute diversity score (1 - average similarity)
        diversity = 1 - torch.mean(similarities)

        return diversity

    def _create_semantic_groups(self) -> None:
        """Create semantic groups based on feature similarity."""
        # Clear existing groups
        self.semantic_groups.clear()

        # Get all IDs
        all_ids = list(self.vision_data.keys())

        # Compute similarity matrix
        vision_features = torch.stack([self.vision_data[id] for id in all_ids])
        similarities = torch.mm(vision_features, vision_features.t())

        # Create groups using hierarchical clustering
        for i, id1 in enumerate(all_ids):
            if id1 in self.semantic_groups:
                continue

            # Find similar items
            group = [id1]
            for j, id2 in enumerate(all_ids):
                if i != j and similarities[i, j] > 0.7:  # Similarity threshold
                    group.append(id2)

            # Filter group size
            if self.min_group_size <= len(group) <= self.max_group_size:
                group_id = f"group_{len(self.semantic_groups)}"
                self.semantic_groups[group_id] = group

    def get_group_features(self, group_id: str) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get vision and text features for a specific group.

        Args:
            group_id: ID of the semantic group

        Returns:
            Tuple of (vision_features, text_features) for the group
        """
        if group_id not in self.semantic_groups:
            raise ValueError(f"Group {group_id} not found")

        group_ids = self.semantic_groups[group_id]
        vision_features = torch.stack([self.vision_data[id] for id in group_ids])
        text_features = torch.stack([self.text_data[id] for id in group_ids])

        return vision_features, text_features

    def get_diverse_features(
        self, num_features: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get diverse features by selecting from different semantic groups.

        Args:
            num_features: Number of features to return

        Returns:
            Tuple of (vision_features, text_features)
        """
        selected_features = []
        selected_texts = []

        # Select features from different groups
        for group_id in list(self.semantic_groups.keys())[:num_features]:
            vision_feats, text_feats = self.get_group_features(group_id)
            selected_features.append(
                vision_feats[0]
            )  # Take first feature from each group
            selected_texts.append(text_feats[0])

        return torch.stack(selected_features), torch.stack(selected_texts)

    def update_features(
        self,
        new_vision_data: Dict[str, torch.Tensor],
        new_text_data: Dict[str, torch.Tensor],
    ) -> None:
        """
        Update the dataset with new features.

        Args:
            new_vision_data: New vision features to add
            new_text_data: New text features to add
        """
        self.vision_data.update(new_vision_data)
        self.text_data.update(new_text_data)

        # Update statistics and groups
        self._update_feature_statistics()
        self._create_semantic_groups()
