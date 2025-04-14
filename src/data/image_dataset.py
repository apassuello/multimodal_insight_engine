# src/data/image_dataset.py
"""
Image Dataset for Vision Transformer Models

PURPOSE:
    Provides a flexible dataset implementation for loading and preprocessing image data
    for vision transformer models. Supports different directory structures and class mappings.

KEY COMPONENTS:
    - Support for common image formats (jpg, png, jpeg)
    - Automatic class discovery from directory structure
    - Optional custom class mapping support
    - Integration with image preprocessing pipeline
"""

import torch
from torch.utils.data import Dataset
import os
from PIL import Image
from typing import Optional, Callable, List, Tuple, Dict, Any
import json
from pathlib import Path

from ..models.vision.image_preprocessing import ImagePreprocessor


class ImageDataset(Dataset):
    """
    Dataset for image classification with Vision Transformer.
    """

    def __init__(
        self,
        root_dir: str,
        transform: Optional[Callable] = None,
        image_size: int = 224,
        split: str = "train",
        class_mapping_file: Optional[str] = None,
    ):
        """
        Initialize the image dataset.

        Args:
            root_dir: Root directory containing images
            transform: Optional transform to apply (if None, a default ImagePreprocessor will be used)
            image_size: Image size for preprocessing
            split: Dataset split ('train', 'val', or 'test')
            class_mapping_file: Optional path to class mapping file (JSON)
        """
        self.root_dir = Path(root_dir)
        self.split = split
        self.image_size = image_size

        # Image processor (either custom or default)
        self.transform = transform or ImagePreprocessor(image_size=image_size)

        # Load class mapping if provided
        self.class_mapping = None
        if class_mapping_file and os.path.exists(class_mapping_file):
            with open(class_mapping_file, "r") as f:
                self.class_mapping = json.load(f)

        # Find all image files
        self.image_paths: List[Path] = []
        self.labels: List[int] = []

        # Check if split directory exists
        split_dir = self.root_dir / split
        if split_dir.exists():
            # Assume directory structure is root/split/class/image.jpg
            for class_dir in split_dir.iterdir():
                if class_dir.is_dir():
                    class_name = class_dir.name
                    class_idx = self._get_class_idx(class_name)

                    for img_path in class_dir.glob("*.jpg"):
                        self.image_paths.append(img_path)
                        self.labels.append(class_idx)

                    for img_path in class_dir.glob("*.png"):
                        self.image_paths.append(img_path)
                        self.labels.append(class_idx)

                    for img_path in class_dir.glob("*.jpeg"):
                        self.image_paths.append(img_path)
                        self.labels.append(class_idx)
        else:
            # Assume flat directory with label information elsewhere
            for img_path in self.root_dir.glob(f"**/*.jpg"):
                self.image_paths.append(img_path)
                # Placeholder label, should be overridden
                self.labels.append(0)

            for img_path in self.root_dir.glob(f"**/*.png"):
                self.image_paths.append(img_path)
                self.labels.append(0)

            for img_path in self.root_dir.glob(f"**/*.jpeg"):
                self.image_paths.append(img_path)
                self.labels.append(0)

    def _get_class_idx(self, class_name: str) -> int:
        """
        Get class index from class name.

        Args:
            class_name: Name of the class

        Returns:
            Class index
        """
        if self.class_mapping is not None:
            # Use provided mapping
            return self.class_mapping.get(class_name, -1)
        else:
            # Create mapping on the fly
            if not hasattr(self, "_class_to_idx"):
                self._class_to_idx = {}

            if class_name not in self._class_to_idx:
                self._class_to_idx[class_name] = len(self._class_to_idx)

            return self._class_to_idx[class_name]

    def __len__(self) -> int:
        """
        Get the number of samples in the dataset.

        Returns:
            Number of samples
        """
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a sample from the dataset.

        Args:
            idx: Index of the sample

        Returns:
            Dictionary containing:
            - 'image': Image tensor of shape [C, H, W]
            - 'label': Class label
        """
        img_path = self.image_paths[idx]
        label = self.labels[idx]

        # Load and preprocess image
        image = Image.open(img_path).convert("RGB")

        # Apply transform
        if isinstance(self.transform, ImagePreprocessor):
            image_tensor = self.transform.preprocess(image)
        else:
            image_tensor = self.transform(image)

        return {
            "image": image_tensor,
            "label": torch.tensor(label, dtype=torch.long),
            "path": str(img_path),
        }


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
        "module_purpose": "Provides dataset functionality for loading and preprocessing image data for vision transformer models",
        "key_classes": [
            {
                "name": "ImageDataset",
                "purpose": "Dataset for loading and preprocessing images for vision transformer models",
                "key_methods": [
                    {
                        "name": "_get_class_idx",
                        "signature": "_get_class_idx(self, class_name: str) -> int",
                        "brief_description": "Get class index from class name using mapping or dynamic creation"
                    },
                    {
                        "name": "__getitem__",
                        "signature": "__getitem__(self, idx: int) -> Dict[str, torch.Tensor]",
                        "brief_description": "Load, preprocess and return an image with its label and path"
                    }
                ],
                "inheritance": "Dataset",
                "dependencies": ["torch.utils.data.Dataset", "PIL.Image", "ImagePreprocessor"]
            }
        ],
        "key_functions": [],
        "external_dependencies": ["torch", "PIL", "pathlib", "json"],
        "complexity_score": 4  # Moderate complexity for image loading/preprocessing
    }
