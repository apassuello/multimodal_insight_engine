#!/usr/bin/env python
"""
Diagnostic Script: Check Match IDs in Multimodal Dataset

This script helps diagnose whether your dataloader is correctly providing
semantic groups via match_ids for contrastive learning.

CRITICAL ISSUE TO CHECK:
- If all match_ids are unique in each batch, contrastive learning fails
- We need multiple samples with the same match_id for proper positive pairs
"""

import sys
import os
from pathlib import Path
from collections import Counter
import torch
from torch.utils.data import DataLoader

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def diagnose_dataloader(dataloader, num_batches=10):
    """
    Analyze match_ids in dataloader batches.

    Args:
        dataloader: Your multimodal dataloader
        num_batches: Number of batches to analyze
    """
    print("=" * 80)
    print("MATCH ID DIAGNOSTIC REPORT")
    print("=" * 80)

    all_batch_stats = []

    for batch_idx, batch in enumerate(dataloader):
        if batch_idx >= num_batches:
            break

        # Extract match_ids from batch
        if isinstance(batch, dict):
            match_ids = batch.get('match_ids', batch.get('match_id', None))
        elif isinstance(batch, (tuple, list)) and len(batch) > 3:
            match_ids = batch[3] if len(batch) > 3 else None
        else:
            match_ids = None

        if match_ids is None:
            print(f"\n❌ Batch {batch_idx}: NO MATCH_IDS FOUND!")
            print("   This means contrastive loss will use diagonal matching")
            print("   → FEATURE COLLAPSE EXPECTED!")
            continue

        # Convert to list if tensor
        if isinstance(match_ids, torch.Tensor):
            match_ids = match_ids.tolist()

        # Convert all to strings for comparison
        match_ids = [str(mid) for mid in match_ids]

        # Analyze distribution
        batch_size = len(match_ids)
        unique_ids = len(set(match_ids))
        id_counts = Counter(match_ids)

        # Calculate statistics
        max_group_size = max(id_counts.values())
        num_duplicates = sum(1 for count in id_counts.values() if count > 1)
        duplicate_ratio = num_duplicates / unique_ids if unique_ids > 0 else 0

        # Determine health status
        if unique_ids == batch_size:
            status = "🔴 CRITICAL"
            issue = "All match_ids unique - NO SEMANTIC GROUPING!"
        elif unique_ids == 1:
            status = "🔴 CRITICAL"
            issue = "All match_ids identical - EVERYTHING MATCHES!"
        elif duplicate_ratio < 0.3:
            status = "🟡 WARNING"
            issue = f"Only {duplicate_ratio*100:.1f}% of IDs have duplicates"
        else:
            status = "✅ HEALTHY"
            issue = f"{duplicate_ratio*100:.1f}% of IDs have duplicates"

        print(f"\nBatch {batch_idx}: {status}")
        print(f"  Batch size: {batch_size}")
        print(f"  Unique IDs: {unique_ids}")
        print(f"  Max group size: {max_group_size}")
        print(f"  Issue: {issue}")

        # Show example groups
        if num_duplicates > 0:
            print(f"  Example groups:")
            shown = 0
            for mid, count in id_counts.most_common():
                if count > 1 and shown < 3:
                    indices = [i for i, x in enumerate(match_ids) if x == mid]
                    print(f"    - ID '{mid}': {count} samples at indices {indices}")
                    shown += 1

        all_batch_stats.append({
            'batch_size': batch_size,
            'unique_ids': unique_ids,
            'duplicate_ratio': duplicate_ratio,
            'max_group_size': max_group_size,
            'status': status
        })

    # Summary statistics
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)

    if not all_batch_stats:
        print("❌ NO BATCHES ANALYZED - CHECK YOUR DATALOADER!")
        return False

    critical_batches = sum(1 for s in all_batch_stats if 'CRITICAL' in s['status'])
    warning_batches = sum(1 for s in all_batch_stats if 'WARNING' in s['status'])
    healthy_batches = sum(1 for s in all_batch_stats if 'HEALTHY' in s['status'])

    avg_duplicate_ratio = sum(s['duplicate_ratio'] for s in all_batch_stats) / len(all_batch_stats)
    avg_unique_ratio = sum(s['unique_ids'] / s['batch_size'] for s in all_batch_stats) / len(all_batch_stats)

    print(f"Batches analyzed: {len(all_batch_stats)}")
    print(f"  🔴 Critical: {critical_batches}")
    print(f"  🟡 Warning: {warning_batches}")
    print(f"  ✅ Healthy: {healthy_batches}")
    print(f"\nAverage duplicate ratio: {avg_duplicate_ratio*100:.1f}%")
    print(f"Average unique ID ratio: {avg_unique_ratio*100:.1f}%")

    # Final verdict
    print("\n" + "=" * 80)
    print("VERDICT")
    print("=" * 80)

    if critical_batches > len(all_batch_stats) * 0.5:
        print("🔴 CRITICAL ISSUE DETECTED!")
        print("Your dataloader is NOT providing proper semantic groups.")
        print("\nLikely causes:")
        print("1. Dataset not implementing match_id properly")
        print("2. Dataloader not using a semantic sampler")
        print("3. Batch construction is random without grouping")
        print("\nThis WILL cause feature collapse!")
        print("\nAction required:")
        print("- Check src/data/multimodal_dataset.py")
        print("- Verify batch sampler groups semantically similar items")
        print("- Ensure each batch has 2-5 samples per semantic group")
        return False
    elif warning_batches > healthy_batches:
        print("🟡 SUBOPTIMAL CONFIGURATION")
        print("Your semantic grouping is weak.")
        print("This may contribute to slower learning or partial collapse.")
        print("\nRecommendation:")
        print("- Increase number of samples per semantic group in batches")
        print("- Use a custom batch sampler that ensures diversity")
        return False
    else:
        print("✅ MATCH IDS LOOK HEALTHY")
        print("Semantic grouping appears to be working correctly.")
        print("Feature collapse is likely caused by other factors.")
        print("\nNext steps:")
        print("1. Check temperature parameter (should be 0.2-0.3 initially)")
        print("2. Verify batch size is large enough (>= 64)")
        print("3. Check gradient flow through projection layers")
        return True


def create_dummy_dataloader():
    """Create a dummy dataloader for testing this script."""

    class DummyDataset(torch.utils.data.Dataset):
        def __init__(self, num_samples=100):
            self.num_samples = num_samples
            # Simulate Flickr30k where each image has 5 captions
            # Create 20 images, each with 5 captions
            self.match_ids = []
            for img_id in range(20):
                for caption_id in range(5):
                    self.match_ids.append(f"image_{img_id}")

        def __len__(self):
            return self.num_samples

        def __getitem__(self, idx):
            return {
                'image': torch.randn(3, 224, 224),
                'text': torch.randint(0, 1000, (77,)),
                'match_ids': self.match_ids[idx]
            }

    dataset = DummyDataset()
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    return dataloader


if __name__ == "__main__":
    print("Multimodal Match ID Diagnostic Tool")
    print("=" * 80)

    # Option 1: Test with your actual dataloader
    # Uncomment and modify this section to use your real dataloader
    """
    from src.data.multimodal_dataset import MultimodalDataset
    from torch.utils.data import DataLoader

    dataset = MultimodalDataset(
        data_root="/path/to/flickr30k",
        split="train",
        # ... your other parameters
    )

    dataloader = DataLoader(
        dataset,
        batch_size=64,
        shuffle=True,
        num_workers=4
    )

    diagnose_dataloader(dataloader, num_batches=20)
    """

    # Option 2: Test with dummy data to verify the script works
    print("\nRunning with DUMMY DATA for demonstration...")
    print("Replace this with your actual dataloader to diagnose real issues.\n")

    dummy_loader = create_dummy_dataloader()
    result = diagnose_dataloader(dummy_loader, num_batches=5)

    print("\n" + "=" * 80)
    print("To use with your real dataloader:")
    print("1. Uncomment the 'Option 1' section in this script")
    print("2. Update paths and parameters for your dataset")
    print("3. Run: python debug_scripts/diagnose_match_ids.py")
    print("=" * 80)
