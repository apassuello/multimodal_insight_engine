# src/data/image_dataset.py
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
