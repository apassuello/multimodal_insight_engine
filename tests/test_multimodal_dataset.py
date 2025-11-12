"""
Comprehensive tests for multimodal dataset loading and preprocessing.

Tests cover:
- MultimodalDataset (security-critical: 1,732 lines, currently 5%)
- Dataset initialization and configuration
- Metadata loading and validation
- Path handling and security (path traversal prevention)
- Image loading and error handling
- Text processing and tokenization
- Hard negative mining
"""

import pytest
import torch
from torch.utils.data import DataLoader
import os
import json
import tempfile
from pathlib import Path
from PIL import Image
import numpy as np

from src.data.multimodal_dataset import MultimodalDataset


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def temp_dataset_dir():
    """Create a temporary directory for test dataset."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def fake_image_dataset(temp_dataset_dir):
    """Create a fake multimodal dataset with images and metadata."""
    # Create images directory
    image_dir = temp_dataset_dir / "images"
    image_dir.mkdir()

    # Create some fake images
    images_data = []
    for i in range(10):
        # Create a simple colored image
        img = Image.new('RGB', (224, 224), color=(i*25, i*25, i*25))
        img_path = image_dir / f"img_{i:03d}.jpg"
        img.save(img_path)

        images_data.append({
            "image_path": f"img_{i:03d}.jpg",
            "caption": f"This is caption number {i}",
            "split": "train" if i < 7 else "val",
            "label": f"class_{i % 3}"  # 3 classes
        })

    # Create metadata file
    metadata = {
        "train": [d for d in images_data if d["split"] == "train"],
        "val": [d for d in images_data if d["split"] == "val"]
    }

    metadata_path = temp_dataset_dir / "metadata.json"
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f)

    return temp_dataset_dir, metadata


@pytest.fixture
def fake_flat_metadata_dataset(temp_dataset_dir):
    """Create dataset with flat (list) metadata structure."""
    image_dir = temp_dataset_dir / "images"
    image_dir.mkdir()

    # Create images
    images_data = []
    for i in range(5):
        img = Image.new('RGB', (224, 224), color=(i*50, i*50, i*50))
        img_path = image_dir / f"img_{i}.jpg"
        img.save(img_path)

        images_data.append({
            "image_path": f"img_{i}.jpg",
            "caption": f"Caption {i}",
            "split": "train"
        })

    # Flat list metadata
    metadata_path = temp_dataset_dir / "metadata.json"
    with open(metadata_path, 'w') as f:
        json.dump(images_data, f)

    return temp_dataset_dir, images_data


# ============================================================================
# MultimodalDataset Initialization Tests
# ============================================================================


class TestMultimodalDatasetInitialization:
    """Test dataset initialization and configuration."""

    def test_basic_initialization(self, fake_image_dataset):
        """Test basic dataset initialization."""
        data_root, metadata = fake_image_dataset

        dataset = MultimodalDataset(
            data_root=str(data_root),
            split="train"
        )

        assert len(dataset) == 7  # 7 train samples
        assert dataset.split == "train"
        assert dataset.data_root == str(data_root)

    def test_initialization_with_val_split(self, fake_image_dataset):
        """Test initialization with validation split."""
        data_root, metadata = fake_image_dataset

        dataset = MultimodalDataset(
            data_root=str(data_root),
            split="val"
        )

        assert len(dataset) == 3  # 3 val samples

    def test_initialization_with_limit_samples(self, fake_image_dataset):
        """Test sample limiting."""
        data_root, metadata = fake_image_dataset

        dataset = MultimodalDataset(
            data_root=str(data_root),
            split="train",
            limit_samples=3
        )

        assert len(dataset) == 3

    def test_initialization_with_custom_keys(self, temp_dataset_dir):
        """Test initialization with custom metadata keys."""
        image_dir = temp_dataset_dir / "images"
        image_dir.mkdir()

        # Create image
        img = Image.new('RGB', (224, 224))
        img.save(image_dir / "test.jpg")

        # Custom keys
        metadata = {
            "train": [{
                "img": "test.jpg",
                "text": "Test caption"
            }]
        }

        with open(temp_dataset_dir / "metadata.json", 'w') as f:
            json.dump(metadata, f)

        dataset = MultimodalDataset(
            data_root=str(temp_dataset_dir),
            split="train",
            image_key="img",
            caption_key="text"
        )

        assert len(dataset) == 1

    def test_missing_metadata_file(self, temp_dataset_dir):
        """Test error handling for missing metadata file."""
        with pytest.raises(FileNotFoundError):
            MultimodalDataset(
                data_root=str(temp_dataset_dir),
                split="train",
                metadata_file="nonexistent.json"
            )

    def test_invalid_split(self, fake_image_dataset):
        """Test error handling for invalid split."""
        data_root, metadata = fake_image_dataset

        with pytest.raises(ValueError):
            MultimodalDataset(
                data_root=str(data_root),
                split="invalid_split"
            )


# ============================================================================
# Metadata Loading Tests
# ============================================================================


class TestMetadataLoading:
    """Test metadata loading and validation."""

    def test_load_dict_metadata(self, fake_image_dataset):
        """Test loading dictionary-structured metadata."""
        data_root, metadata = fake_image_dataset

        dataset = MultimodalDataset(
            data_root=str(data_root),
            split="train"
        )

        # Should load train split
        assert len(dataset) == 7

    def test_load_flat_list_metadata(self, fake_flat_metadata_dataset):
        """Test loading flat list metadata."""
        data_root, metadata = fake_flat_metadata_dataset

        dataset = MultimodalDataset(
            data_root=str(data_root),
            split="train"
        )

        assert len(dataset) == 5

    def test_metadata_validation_missing_images(self, temp_dataset_dir):
        """Test that samples with missing images are filtered out."""
        image_dir = temp_dataset_dir / "images"
        image_dir.mkdir()

        # Create one image but metadata for two
        img = Image.new('RGB', (224, 224))
        img.save(image_dir / "existing.jpg")

        metadata = {
            "train": [
                {"image_path": "existing.jpg", "caption": "Exists"},
                {"image_path": "missing.jpg", "caption": "Missing"}
            ]
        }

        with open(temp_dataset_dir / "metadata.json", 'w') as f:
            json.dump(metadata, f)

        dataset = MultimodalDataset(
            data_root=str(temp_dataset_dir),
            split="train"
        )

        # Should only have 1 sample (missing image filtered out)
        assert len(dataset) == 1

    def test_metadata_validation_missing_caption(self, temp_dataset_dir):
        """Test that samples without captions are filtered out."""
        image_dir = temp_dataset_dir / "images"
        image_dir.mkdir()

        img = Image.new('RGB', (224, 224))
        img.save(image_dir / "test.jpg")

        metadata = {
            "train": [
                {"image_path": "test.jpg", "caption": "Has caption"},
                {"image_path": "test.jpg"}  # No caption
            ]
        }

        with open(temp_dataset_dir / "metadata.json", 'w') as f:
            json.dump(metadata, f)

        dataset = MultimodalDataset(
            data_root=str(temp_dataset_dir),
            split="train"
        )

        assert len(dataset) == 1


# ============================================================================
# Path Handling & Security Tests
# ============================================================================


class TestPathHandling:
    """Test path handling and security."""

    def test_relative_path_handling(self, temp_dataset_dir):
        """Test that relative paths are converted to absolute."""
        image_dir = temp_dataset_dir / "images"
        image_dir.mkdir()

        img = Image.new('RGB', (224, 224))
        img.save(image_dir / "test.jpg")

        metadata = {
            "train": [{
                "image_path": "test.jpg",  # Relative path
                "caption": "Test"
            }]
        }

        with open(temp_dataset_dir / "metadata.json", 'w') as f:
            json.dump(metadata, f)

        dataset = MultimodalDataset(
            data_root=str(temp_dataset_dir),
            split="train"
        )

        # Path should be converted to absolute
        sample_path = dataset.samples[0]["image_path"]
        assert os.path.isabs(sample_path)
        assert os.path.exists(sample_path)

    def test_absolute_path_handling(self, temp_dataset_dir):
        """Test that absolute paths are preserved."""
        image_dir = temp_dataset_dir / "images"
        image_dir.mkdir()

        img = Image.new('RGB', (224, 224))
        img_abs_path = image_dir / "test.jpg"
        img.save(img_abs_path)

        metadata = {
            "train": [{
                "image_path": str(img_abs_path),  # Absolute path
                "caption": "Test"
            }]
        }

        with open(temp_dataset_dir / "metadata.json", 'w') as f:
            json.dump(metadata, f)

        dataset = MultimodalDataset(
            data_root=str(temp_dataset_dir),
            split="train"
        )

        assert len(dataset) == 1

    def test_path_traversal_prevention(self, temp_dataset_dir):
        """Test that path traversal attempts are handled safely."""
        image_dir = temp_dataset_dir / "images"
        image_dir.mkdir()

        # Try path traversal
        metadata = {
            "train": [{
                "image_path": "../../../etc/passwd",  # Path traversal attempt
                "caption": "Test"
            }]
        }

        with open(temp_dataset_dir / "metadata.json", 'w') as f:
            json.dump(metadata, f)

        dataset = MultimodalDataset(
            data_root=str(temp_dataset_dir),
            split="train"
        )

        # Should have 0 samples (file doesn't exist or is filtered)
        assert len(dataset) == 0


# ============================================================================
# Data Loading Tests
# ============================================================================


class TestDataLoading:
    """Test actual data loading via __getitem__."""

    def test_getitem_basic(self, fake_image_dataset):
        """Test basic __getitem__ functionality."""
        data_root, metadata = fake_image_dataset

        dataset = MultimodalDataset(
            data_root=str(data_root),
            split="train"
        )

        # Get first sample
        sample = dataset[0]

        # Check return type
        assert isinstance(sample, dict)
        assert 'image' in sample or 'pixel_values' in sample
        assert 'text' in sample or 'caption' in sample

    def test_getitem_returns_tensors(self, fake_image_dataset):
        """Test that __getitem__ returns tensors."""
        data_root, metadata = fake_image_dataset

        dataset = MultimodalDataset(
            data_root=str(data_root),
            split="train"
        )

        sample = dataset[0]

        # Image should be tensor or PIL Image
        image_key = 'image' if 'image' in sample else 'pixel_values'
        assert isinstance(sample[image_key], (torch.Tensor, Image.Image))

    def test_dataloader_integration(self, fake_image_dataset):
        """Test dataset works with DataLoader."""
        data_root, metadata = fake_image_dataset

        dataset = MultimodalDataset(
            data_root=str(data_root),
            split="train"
        )

        # Create dataloader
        dataloader = DataLoader(dataset, batch_size=2, shuffle=False)

        # Get one batch
        batch = next(iter(dataloader))

        assert isinstance(batch, dict)
        # Batch size should be 2
        for key in batch:
            if isinstance(batch[key], torch.Tensor):
                assert batch[key].shape[0] == 2


# ============================================================================
# Hard Negative Mining Tests
# ============================================================================


class TestHardNegativeMining:
    """Test hard negative mining functionality."""

    def test_class_indexing(self, fake_image_dataset):
        """Test that class indices are built correctly."""
        data_root, metadata = fake_image_dataset

        dataset = MultimodalDataset(
            data_root=str(data_root),
            split="train",
            label_key="label"
        )

        # Should have class indices
        assert len(dataset.class_to_indices) > 0

        # Each class should have at least one sample
        for class_label, indices in dataset.class_to_indices.items():
            assert len(indices) > 0

    def test_hard_negative_same_class(self, fake_image_dataset):
        """Test getting hard negative from same class."""
        data_root, metadata = fake_image_dataset

        dataset = MultimodalDataset(
            data_root=str(data_root),
            split="train",
            label_key="label"
        )

        # Get hard negative for first sample
        neg_idx = dataset.get_hard_negative(0, neg_type="same_class")

        # Should be valid index
        assert 0 <= neg_idx < len(dataset)

        # Should have same label (if possible)
        if len(dataset.class_to_indices[dataset.samples[0]["label"]]) > 1:
            assert dataset.samples[0]["label"] == dataset.samples[neg_idx]["label"]

    def test_hard_negative_different_class(self, fake_image_dataset):
        """Test getting hard negative from different class."""
        data_root, metadata = fake_image_dataset

        dataset = MultimodalDataset(
            data_root=str(data_root),
            split="train",
            label_key="label"
        )

        neg_idx = dataset.get_hard_negative(0, neg_type="different_class")

        # Should be valid index
        assert 0 <= neg_idx < len(dataset)

        # Should have different label (if multiple classes exist)
        if len(dataset.class_to_indices) > 1:
            assert dataset.samples[0]["label"] != dataset.samples[neg_idx]["label"]

    def test_hard_negative_no_labels(self, fake_flat_metadata_dataset):
        """Test hard negative mining without labels."""
        data_root, metadata = fake_flat_metadata_dataset

        dataset = MultimodalDataset(
            data_root=str(data_root),
            split="train",
            label_key=None  # No labels
        )

        # Should still return valid index
        neg_idx = dataset.get_hard_negative(0, neg_type="same_class")
        assert 0 <= neg_idx < len(dataset)


# ============================================================================
# Edge Cases & Error Handling
# ============================================================================


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_empty_dataset(self, temp_dataset_dir):
        """Test dataset with no valid samples."""
        image_dir = temp_dataset_dir / "images"
        image_dir.mkdir()

        # Metadata with no matching images
        metadata = {
            "train": [{
                "image_path": "nonexistent.jpg",
                "caption": "Test"
            }]
        }

        with open(temp_dataset_dir / "metadata.json", 'w') as f:
            json.dump(metadata, f)

        dataset = MultimodalDataset(
            data_root=str(temp_dataset_dir),
            split="train"
        )

        assert len(dataset) == 0

    def test_single_sample_dataset(self, temp_dataset_dir):
        """Test dataset with single sample."""
        image_dir = temp_dataset_dir / "images"
        image_dir.mkdir()

        img = Image.new('RGB', (224, 224))
        img.save(image_dir / "test.jpg")

        metadata = {
            "train": [{
                "image_path": "test.jpg",
                "caption": "Single sample"
            }]
        }

        with open(temp_dataset_dir / "metadata.json", 'w') as f:
            json.dump(metadata, f)

        dataset = MultimodalDataset(
            data_root=str(temp_dataset_dir),
            split="train"
        )

        assert len(dataset) == 1
        sample = dataset[0]
        assert isinstance(sample, dict)

    def test_large_caption(self, temp_dataset_dir):
        """Test handling of very long captions."""
        image_dir = temp_dataset_dir / "images"
        image_dir.mkdir()

        img = Image.new('RGB', (224, 224))
        img.save(image_dir / "test.jpg")

        # Very long caption
        long_caption = "word " * 1000

        metadata = {
            "train": [{
                "image_path": "test.jpg",
                "caption": long_caption
            }]
        }

        with open(temp_dataset_dir / "metadata.json", 'w') as f:
            json.dump(metadata, f)

        dataset = MultimodalDataset(
            data_root=str(temp_dataset_dir),
            split="train",
            max_text_length=77
        )

        # Should handle long caption
        assert len(dataset) == 1

    def test_unicode_caption(self, temp_dataset_dir):
        """Test handling of Unicode characters in captions."""
        image_dir = temp_dataset_dir / "images"
        image_dir.mkdir()

        img = Image.new('RGB', (224, 224))
        img.save(image_dir / "test.jpg")

        metadata = {
            "train": [{
                "image_path": "test.jpg",
                "caption": "Hello 世界 🌍 Привет"
            }]
        }

        with open(temp_dataset_dir / "metadata.json", 'w') as f:
            json.dump(metadata, f, ensure_ascii=False)

        dataset = MultimodalDataset(
            data_root=str(temp_dataset_dir),
            split="train"
        )

        assert len(dataset) == 1
