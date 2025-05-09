"""MODULE: multimodal_dataset.py
PURPOSE: Implements dataset classes for handling multimodal data pairs (image-text) with support for various data formats and preprocessing.

KEY COMPONENTS:
- MultimodalDataset: Base dataset class for handling image-text pairs
- Support for multiple data formats (raw files, preprocessed features)
- Configurable data loading and preprocessing pipelines
- Memory-efficient data handling with lazy loading
- Semantic grouping capabilities for contrastive learning

DEPENDENCIES:
- torch
- PIL
- numpy
- torchvision
- multimodal_data_utils
"""

import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import PIL.Image as Image
import os
import json
import time
import random
from typing import Dict, List, Tuple, Optional, Callable, Union, Any
import numpy as np
from collections import defaultdict

from ..models.vision.image_preprocessing import ImagePreprocessor


class MultimodalDataset(Dataset):
    """
    Dataset for multimodal learning with images and text.

    This dataset handles image-text pairs for contrastive learning,
    with options for in-batch negatives and hard negative mining.
    """

    def __init__(
        self,
        data_root: str,
        image_processor: Optional[Union[ImagePreprocessor, transforms.Compose]] = None,
        text_tokenizer=None,  # Type depends on your tokenizer implementation
        max_text_length: int = 77,
        split: str = "train",
        transform_image: Optional[Callable] = None,
        transform_text: Optional[Callable] = None,
        metadata_file: str = "metadata.json",
        image_key: str = "image_path",
        caption_key: str = "caption",
        label_key: Optional[str] = "label",
        image_dir: str = "images",
        limit_samples: Optional[int] = None,
        return_metadata: bool = False,
    ):
        """
        Initialize multimodal dataset.

        Args:
            data_root: Root directory of the dataset
            image_processor: Processor for images (ImagePreprocessor or torchvision transforms)
            text_tokenizer: Tokenizer for text processing
            max_text_length: Maximum text length after tokenization
            split: Data split ("train", "val", or "test")
            transform_image: Additional transforms for images
            transform_text: Additional transforms for text
            metadata_file: JSON file with dataset metadata
            image_key: Key for image path in metadata
            caption_key: Key for caption in metadata
            label_key: Key for class labels in metadata (optional)
            image_dir: Directory containing images (relative to data_root)
            limit_samples: Maximum number of samples to use (for debugging)
            return_metadata: Whether to return full metadata for each sample
        """
        super().__init__()
        self.data_root = data_root
        self.image_dir = os.path.join(data_root, image_dir)
        self.split = split
        self.image_key = image_key
        self.caption_key = caption_key
        self.label_key = label_key
        self.return_metadata = return_metadata

        # Set up image processor
        if image_processor is None:
            # Default image preprocessing
            self.image_processor = ImagePreprocessor(image_size=224)
        else:
            self.image_processor = image_processor

        # Set up text tokenizer
        self.text_tokenizer = text_tokenizer
        self.max_text_length = max_text_length

        # Additional transforms
        self.transform_image = transform_image
        self.transform_text = transform_text

        # Load metadata
        metadata_path = os.path.join(data_root, metadata_file)
        self.samples = self._load_metadata(metadata_path, split)

        # Limit number of samples if specified
        if limit_samples is not None and limit_samples > 0:
            self.samples = self.samples[:limit_samples]

        # Build class-based indices for hard negative mining
        self.class_to_indices = self._build_class_indices()

        print(f"Loaded {len(self.samples)} samples for {split} split")

    def _load_metadata(self, metadata_path: str, split: str) -> List[Dict]:
        """
        Load dataset metadata.

        Args:
            metadata_path: Path to metadata JSON file
            split: Data split to use

        Returns:
            List of metadata dictionaries for samples in the split
        """
        if not os.path.exists(metadata_path):
            raise FileNotFoundError(f"Metadata file not found: {metadata_path}")

        with open(metadata_path, "r") as f:
            full_metadata = json.load(f)

        # Check if metadata has split information
        if isinstance(full_metadata, dict) and "train" in full_metadata:
            # Metadata is organized by split
            if split not in full_metadata:
                raise ValueError(f"Split '{split}' not found in metadata")
            samples = full_metadata[split]
        elif isinstance(full_metadata, list):
            # Metadata is a list of samples - filter by split if available
            if any("split" in sample for sample in full_metadata):
                samples = [
                    sample for sample in full_metadata if sample.get("split") == split
                ]
            else:
                # No split information - use all samples
                samples = full_metadata
        else:
            raise ValueError("Unsupported metadata format")

        # Validate samples
        valid_samples = []
        for sample in samples:
            image_path = sample.get(self.image_key)
            caption = sample.get(self.caption_key)

            if image_path and caption:
                # Ensure image path is absolute
                if not os.path.isabs(image_path):
                    image_path = os.path.join(self.image_dir, image_path)
                    sample[self.image_key] = image_path

                # Check if image exists
                if os.path.exists(image_path):
                    valid_samples.append(sample)

        return valid_samples

    def _build_class_indices(self) -> Dict[str, List[int]]:
        """
        Build mapping from class labels to sample indices.

        This is used for mining hard negatives during training.

        Returns:
            Dictionary mapping class labels to lists of sample indices
        """
        if not self.label_key:
            return {}

        class_indices = defaultdict(list)

        for i, sample in enumerate(self.samples):
            if self.label_key in sample:
                label = sample[self.label_key]
                class_indices[label].append(i)

        return dict(class_indices)

    def get_hard_negative(self, idx: int, neg_type: str = "same_class") -> int:
        """
        Get hard negative sample index.

        Args:
            idx: Index of the anchor sample
            neg_type: Type of hard negative ("same_class" or "different_class")

        Returns:
            Index of hard negative sample
        """
        if not self.label_key or not self.class_to_indices:
            # No class information - use random sample
            return random.randint(0, len(self.samples) - 1)

        anchor_sample = self.samples[idx]
        anchor_label = anchor_sample.get(self.label_key)

        if anchor_label is None:
            # No label for anchor - use random sample
            return random.randint(0, len(self.samples) - 1)

        if neg_type == "same_class":
            # Get sample with same class (hard negative - similar content)
            class_indices = self.class_to_indices.get(anchor_label, [])

            if len(class_indices) <= 1:
                # No other samples with same class - use random sample
                return random.randint(0, len(self.samples) - 1)

            # Get random sample with same class (excluding anchor)
            valid_indices = [i for i in class_indices if i != idx]
            return random.choice(valid_indices)

        elif neg_type == "different_class":
            # Get sample with different class
            other_labels = [
                label for label in self.class_to_indices.keys() if label != anchor_label
            ]

            if not other_labels:
                # No other classes - use random sample
                return random.randint(0, len(self.samples) - 1)

            # Get random sample from random different class
            neg_label = random.choice(other_labels)
            return random.choice(self.class_to_indices[neg_label])

        else:
            # Invalid neg_type - use random sample
            return random.randint(0, len(self.samples) - 1)

    def __len__(self) -> int:
        """Get the number of samples in the dataset."""
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a sample from the dataset.

        Args:
            idx: Sample index

        Returns:
            Dictionary with processed image and text tensors
        """
        sample = self.samples[idx]

        # Get image path and caption
        image_path = sample[self.image_key]
        caption = sample[self.caption_key]

        # Load and process image
        try:
            image = Image.open(image_path).convert("RGB")

            # Apply additional image transforms if specified
            if self.transform_image:
                image = self.transform_image(image)

            # Process with image processor
            if isinstance(self.image_processor, ImagePreprocessor):
                image_tensor = self.image_processor.preprocess(image)
            else:
                # Use as transform composition
                image_tensor = self.image_processor(image)

        except Exception as e:
            print(f"Error loading image {image_path}: {e}")
            # Return a black image as fallback
            image_tensor = torch.zeros(3, 224, 224)

        # Process text
        try:
            # Apply additional text transforms if specified
            if self.transform_text:
                caption = self.transform_text(caption)

            # Tokenize text if tokenizer is available
            if self.text_tokenizer:
                if self.max_text_length > 0:
                    # Tokenize with length limit
                    token_ids = self.text_tokenizer.encode(caption)

                    # Pad or truncate to max_text_length
                    if len(token_ids) > self.max_text_length:
                        token_ids = token_ids[: self.max_text_length]
                    else:
                        padding = [
                            self.text_tokenizer.special_tokens["pad_token_idx"]
                        ] * (self.max_text_length - len(token_ids))
                        token_ids = token_ids + padding

                    # Create text tensor
                    text_tensor = torch.tensor(token_ids, dtype=torch.long)

                    # Create text mask (1 for real tokens, 0 for padding)
                    text_mask = (
                        text_tensor
                        != self.text_tokenizer.special_tokens["pad_token_idx"]
                    ).float()
                else:
                    # Tokenize without length limit
                    token_ids = self.text_tokenizer.encode(caption)
                    text_tensor = torch.tensor(token_ids, dtype=torch.long)
                    text_mask = torch.ones_like(text_tensor, dtype=torch.float)
            else:
                # Return raw text if no tokenizer
                text_tensor = caption
                text_mask = torch.ones(1, dtype=torch.float)  # Dummy mask
        except Exception as e:
            print(f"Error processing text '{caption}': {e}")
            # Return empty tensor as fallback
            if self.text_tokenizer and self.max_text_length > 0:
                text_tensor = torch.full(
                    (self.max_text_length,),
                    self.text_tokenizer.special_tokens["pad_token_idx"],
                    dtype=torch.long,
                )
                text_mask = torch.zeros(self.max_text_length, dtype=torch.float)
            else:
                text_tensor = ""
                text_mask = torch.zeros(1, dtype=torch.float)

        # Build result dictionary
        result = {
            "image": image_tensor,
            "text": text_tensor,
            "text_mask": text_mask,
        }

        # Add metadata if requested
        if self.return_metadata:
            result["metadata"] = sample

        # Add label if available
        if self.label_key and self.label_key in sample:
            result["label"] = sample[self.label_key]

        return result


class Flickr30kDataset(MultimodalDataset):
    """
    Dataset adapter for Flickr30k from Hugging Face.

    This extends the MultimodalDataset class to work with Hugging Face's
    Flickr30k dataset, with fallback to synthetic data generation.
    """

    def __init__(
        self,
        data_root: str = "",
        image_processor: Optional[Union[ImagePreprocessor, transforms.Compose]] = None,
        text_tokenizer=None,
        max_text_length: int = 77,
        split: str = "train",
        transform_image: Optional[Callable] = None,
        transform_text: Optional[Callable] = None,
        limit_samples: Optional[int] = None,
        return_metadata: bool = False,
    ):
        """
        Initialize Flickr30k dataset.

        Args:
            data_root: Root directory for cached data (optional)
            image_processor: Processor for images
            text_tokenizer: Tokenizer for text processing
            max_text_length: Maximum text length after tokenization
            split: Data split ("train", "val", or "test")
            transform_image: Additional transforms for images
            transform_text: Additional transforms for text
            limit_samples: Maximum number of samples to use
            return_metadata: Whether to return full metadata for each sample
        """
        # Set the default data root to data/flickr30k if not provided
        if not data_root:
            data_root = os.path.join(
                os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
                "data",
                "flickr30k",
            )

        # Create directory if it doesn't exist
        os.makedirs(data_root, exist_ok=True)

        # Initialize base class with data_root
        super().__init__(
            data_root=data_root,
            image_processor=image_processor,
            text_tokenizer=text_tokenizer,
            max_text_length=max_text_length,
            split=split,
            transform_image=transform_image,
            transform_text=transform_text,
            limit_samples=limit_samples,
            return_metadata=return_metadata,
        )

        # Set up tokenizer verification
        if self.text_tokenizer is None:
            raise ValueError("Tokenizer is required for Flickr30kDataset")
        if not hasattr(self.text_tokenizer, "encode"):
            raise ValueError("Tokenizer must have 'encode' method")
        if not hasattr(self.text_tokenizer, "special_tokens"):
            raise ValueError("Tokenizer must have 'special_tokens' attribute")

        # Define cache paths
        self.cache_dir = os.path.join(data_root, f"cache_{split}")
        self.cache_metadata = os.path.join(self.cache_dir, "metadata.json")
        self.cache_samples = os.path.join(self.cache_dir, "samples.pkl")

        # Try to load from cache first
        if self._load_from_cache():
            print(
                f"Successfully loaded {len(self.samples)} examples from cache for {split} split"
            )
        else:
            # Load the dataset from HuggingFace if not in cache
            try:
                # Load from Hugging Face datasets
                from datasets import load_dataset
                import tqdm

                print(f"Loading Flickr30k dataset for split: {split}...")

                try:
                    # Load the full dataset
                    dataset_dict = load_dataset(
                        "nlphuji/flickr30k",
                        cache_dir=os.path.join(
                            os.path.dirname(data_root), "nlphuji___flickr30k"
                        ),
                    )

                    # Convert to list format for easier handling
                    self.samples = []

                    # Get the test split since that's what's available
                    if not isinstance(dataset_dict, dict) or "test" not in dataset_dict:
                        raise ValueError("Expected dataset with 'test' split")

                    test_dataset = dataset_dict["test"]

                    # Create a progress bar for processing
                    progress_bar = tqdm.tqdm(
                        total=len(test_dataset), desc=f"Processing Flickr30k {split}"
                    )

                    # Iterate through all examples
                    for i in range(len(test_dataset)):
                        item = test_dataset[i]
                        if isinstance(item, dict) and item.get("split") == split:
                            # Convert to dictionary and store
                            sample = {
                                "image_path": item["image"],
                                "caption": (
                                    item["caption"]
                                    if "caption" in item
                                    else item.get("captions", [""])[0]
                                ),
                                "image_id": str(i),
                            }
                            self.samples.append(sample)
                        progress_bar.update(1)

                    progress_bar.close()

                    if not self.samples:
                        raise ValueError(f"No examples found for split '{split}'")

                    print(
                        f"Successfully loaded {len(self.samples)} examples from Flickr30k {split} split"
                    )

                    # Save to cache for next time
                    self._save_to_cache()

                except Exception as e:
                    print(f"Error with primary dataset source: {str(e)}")
                    raise  # Re-raise to try alternative sources

            except Exception as e:
                print(f"Error loading Flickr30k dataset: {str(e)}")
                print("Falling back to synthetic data generation...")
                self._generate_synthetic_data()

        # Limit samples if specified
        if limit_samples is not None and limit_samples > 0:
            self.samples = self.samples[:limit_samples]

        # Build class indices (empty for Flickr30k as we don't have class labels)
        self.class_to_indices = {}

    def _load_from_cache(self) -> bool:
        """
        Try to load the dataset from cache.

        Returns:
            bool: True if successfully loaded from cache, False otherwise
        """
        import pickle

        if os.path.exists(self.cache_metadata) and os.path.exists(self.cache_samples):
            try:
                # Load samples from pickle file
                with open(self.cache_samples, "rb") as f:
                    self.samples = pickle.load(f)

                # Ensure cache is valid
                if not self.samples:
                    return False

                return True
            except Exception as e:
                print(f"Error loading from cache: {str(e)}")
                return False
        return False

    def _save_to_cache(self) -> None:
        """Save the dataset to cache for faster loading next time."""
        import pickle
        import json

        # Create cache directory if it doesn't exist
        os.makedirs(self.cache_dir, exist_ok=True)

        try:
            # Save samples to pickle file
            with open(self.cache_samples, "wb") as f:
                pickle.dump(self.samples, f)

            # Save metadata
            metadata = {
                "num_samples": len(self.samples),
                "split": self.split,
                "date_created": time.strftime("%Y-%m-%d %H:%M:%S"),
            }

            with open(self.cache_metadata, "w") as f:
                json.dump(metadata, f, indent=2)

            print(f"Saved {len(self.samples)} samples to cache at {self.cache_dir}")
        except Exception as e:
            print(f"Error saving to cache: {str(e)}")

    def _generate_synthetic_data(self):
        """Generate synthetic data as a fallback."""
        logger.info(f"Generating {self.synthetic_samples} synthetic samples")

        # Mark as not loaded from cache (it's synthetic)
        self.loaded_from_cache = False

        # Define cache paths for synthetic data
        data_root = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
            "data",
            "flickr30k",
        )
        os.makedirs(data_root, exist_ok=True)

        synthetic_dir = os.path.join(data_root, f"synthetic_{self.split}")
        os.makedirs(synthetic_dir, exist_ok=True)

        cache_dir = os.path.join(data_root, f"cache_{self.split}")
        cache_metadata = os.path.join(cache_dir, "metadata.json")
        cache_samples = os.path.join(cache_dir, "samples.pkl")

        import tqdm

        # Create a progress bar for generating synthetic data
        progress_bar = tqdm.tqdm(
            total=self.synthetic_samples,
            desc=f"Generating synthetic data for {self.split}",
        )

        self.dataset = []
        for i in range(self.synthetic_samples):
            # Create a random RGB image with some structure
            img = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)

            # Add some structure to make it more interesting
            x, y = np.mgrid[0:224, 0:224]

            # Add a shape based on index
            shape_type = i % 4
            if shape_type == 0:  # Circle
                circle = (x - 112) ** 2 + (y - 112) ** 2 < 50**2
                img[circle, 0] = 200  # Red circle
                shape_name = "circle"
                color_name = "red"
            elif shape_type == 1:  # Square
                img[70:140, 70:140, 1] = 200  # Green square
                shape_name = "square"
                color_name = "green"
            elif shape_type == 2:  # Rectangle
                img[60:100, 60:160, 2] = 200  # Blue rectangle
                shape_name = "rectangle"
                color_name = "blue"
            else:  # Triangle
                tri_y, tri_x = np.mgrid[40:140, 40:140]
                triangle = tri_x > tri_y
                img[tri_y[triangle], tri_x[triangle], 1] = 180  # Green triangle
                shape_name = "triangle"
                color_name = "green"

            # Convert to PIL Image
            img = Image.fromarray(img)

            # Create a simple caption
            caption = f"A {color_name} {shape_name} on a plain background"

            self.dataset.append(
                {
                    "image": img,
                    "captions": [caption],
                    "image_id": str(i),
                    "idx": i,  # Keep track of index for evaluation
                }
            )

            progress_bar.update(1)

        progress_bar.close()
        logger.info(f"Generated {len(self.dataset)} synthetic examples")

        # Save synthetic data to cache
        try:
            import pickle
            import json

            # Create cache directory if it doesn't exist
            os.makedirs(cache_dir, exist_ok=True)

            # Save samples to pickle file
            with open(cache_samples, "wb") as f:
                pickle.dump(self.dataset, f)

            # Save metadata
            metadata = {
                "num_samples": len(self.dataset),
                "split": self.split,
                "date_created": time.strftime("%Y-%m-%d %H:%M:%S"),
                "type": "synthetic",
            }

            with open(cache_metadata, "w") as f:
                json.dump(metadata, f, indent=2)

            logger.info(f"Saved synthetic data to cache at {cache_dir}")
        except Exception as e:
            logger.warning(f"Error saving synthetic data to cache: {str(e)}")


import logging
from datasets import load_dataset

logger = logging.getLogger(__name__)


class EnhancedMultimodalDataset(Dataset):
    """
    Enhanced multimodal dataset for image-text pairs with proper index tracking.
    Works with Flickr30k and falls back to synthetic data generation when needed.
    """

    def __init__(
        self,
        split: str = "train",  # Can be 'train', 'val', or 'test'
        image_preprocessor=None,
        tokenizer=None,
        max_text_length: int = 77,
        dataset_name: str = "flickr30k",
        synthetic_samples: int = 100,
        cache_dir: Optional[str] = None,
        max_samples: Optional[int] = None,
        captions_per_image: int = 1,  # New parameter: Number of captions to use per image (1-5)
        min_samples_per_group: int = 2,  # Pass through semantic grouping parameters
        max_samples_per_group: Optional[int] = None,
        cap_strategy: str = "random",
    ):
        """
        Initialize the enhanced multimodal dataset.

        Args:
            split: Dataset split ('train', 'val', or 'test')
            image_preprocessor: Image preprocessor for transforming images
            tokenizer: Text tokenizer for encoding captions
            max_text_length: Maximum text sequence length after tokenization
            dataset_name: Name of the dataset to load ('flickr30k' or 'custom')
            synthetic_samples: Number of synthetic samples to generate if needed
            cache_dir: Directory to cache the dataset
            max_samples: Maximum number of samples to use (None uses all available data)
            captions_per_image: Number of captions to use per image (1-5, default: 1)
                                Use 1 for efficiency, 5 for maximum data utilization
        """
        self.split = split
        self.image_preprocessor = image_preprocessor
        self.tokenizer = tokenizer
        self.max_text_length = max_text_length
        self.dataset_name = dataset_name
        self.synthetic_samples = synthetic_samples
        self.cache_dir = cache_dir
        self.max_samples = max_samples
        self.loaded_from_cache = False  # Flag to track if data was loaded from cache

        # Store semantic grouping parameters
        self.min_samples_per_group = min_samples_per_group
        self.max_samples_per_group = max_samples_per_group
        self.cap_strategy = cap_strategy

        # Log semantic grouping parameters
        if self.max_samples_per_group is not None:
            logger.info(
                f"Using group size limits: min={self.min_samples_per_group}, max={self.max_samples_per_group}, strategy={self.cap_strategy}"
            )
        else:
            logger.info(
                f"Using min_samples_per_group={self.min_samples_per_group} without upper limit"
            )

        # Validate and store captions_per_image parameter
        if captions_per_image < 1 or captions_per_image > 5:
            logger.warning(
                f"Invalid captions_per_image value: {captions_per_image}. Using 1 as default."
            )
            self.captions_per_image = 1
        else:
            self.captions_per_image = captions_per_image

        logger.info(f"Using {self.captions_per_image} captions per image")

        # Verify tokenizer
        if self.tokenizer is None:
            raise ValueError("Tokenizer is required")
        if not hasattr(self.tokenizer, "encode"):
            raise ValueError("Tokenizer must have 'encode' method")
        if not hasattr(self.tokenizer, "special_tokens"):
            raise ValueError("Tokenizer must have 'special_tokens' attribute")

        # Load dataset
        try:
            if dataset_name == "flickr30k":
                self._load_flickr30k()
            else:
                raise ValueError(f"Unsupported dataset: {dataset_name}")
        except Exception as e:
            logger.warning(f"Error loading {dataset_name}: {str(e)}")
            logger.info("Falling back to synthetic data generation...")
            self._generate_synthetic_data()

        # Limit samples if requested
        if self.max_samples is not None and len(self.dataset) > self.max_samples:
            # Keep a random subset to ensure diversity
            indices = list(range(len(self.dataset)))
            random.shuffle(indices)
            selected_indices = indices[: self.max_samples]
            self.dataset = [self.dataset[i] for i in selected_indices]
            logger.info(
                f"Limited dataset to {len(self.dataset)} samples (max_samples={self.max_samples})"
            )

        # Properly implement match IDs to create a semantic connection for contrastive learning
        # We'll make sure we have groups of semantically related items rather than using index-based matching
        self.match_ids = []
        match_id_counter = 0
        image_id_to_match_id = {}  # Map image_ids to semantic match_ids

        # CRITICAL FIX: Create semantically meaningful match groups
        # Proper semantic grouping is essential for contrastive learning

        # CRITICAL FIX FOR FLICKR30K - ARTIFICIAL SEMANTIC GROUPS
        # We've discovered that Flickr30k (as cached) doesn't have multiple captions per image
        # We need to artificially create semantic groups to enable contrastive learning

        # First, print diagnostics about the dataset structure
        has_image_id = len([item for item in self.dataset if "image_id" in item])
        logger.info(
            f"Dataset has {len(self.dataset)} items, {has_image_id} with image_id"
        )

        if len(self.dataset) > 0:
            sample_item = self.dataset[0]
            logger.info(f"Sample item keys: {sample_item.keys()}")

            # Check if we have captions embedded in the items
            if "captions" in sample_item:
                captions = sample_item["captions"]
                if isinstance(captions, list) and len(captions) > 0:
                    logger.info(f"Found captions list with {len(captions)} items")

        # SYNTHETIC STRATEGY: For Flickr30k, we'll create artificial semantic groups
        # by clustering similar items together

        # Option 1: Group by visual similarity by using nearby indices as proxies
        # This assumes consecutive images in the dataset might have some relationship
        # For training, this provides variation while maintaining some coherence
        group_size = 5  # Each group will have this many items
        num_groups = (
            len(self.dataset) + group_size - 1
        ) // group_size  # Ceiling division

        logger.info(
            f"Creating {num_groups} artificial semantic groups with ~{group_size} items each"
        )

        try:
            import numpy as np
            from sklearn.cluster import KMeans
            import torch
            from tqdm import tqdm

            logger.info(
                "Creating embedding-based semantic groups using pretrained features..."
            )

            # Import a pretrained model for feature extraction
            try:
                from torchvision.models import resnet18, ResNet18_Weights

                pretrained_model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
                pretrained_model.fc = torch.nn.Identity()  # Remove classification layer
                device = torch.device(
                    "cuda"
                    if torch.cuda.is_available()
                    else "mps" if torch.backends.mps.is_available() else "cpu"
                )
                pretrained_model = pretrained_model.to(device)
                pretrained_model.eval()

                # Create preprocessing transform
                from torchvision import transforms

                preprocess = transforms.Compose(
                    [
                        transforms.Resize(256),
                        transforms.CenterCrop(224),
                        transforms.ToTensor(),
                        transforms.Normalize(
                            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                        ),
                    ]
                )

                # Extract features from each image
                features = []
                logger.info("Extracting image features for semantic grouping...")
                with torch.no_grad():
                    for item in tqdm(self.dataset):
                        try:
                            # Handle different image formats
                            if isinstance(item["image"], str):
                                # It's a path - load the image
                                from PIL import Image

                                img = Image.open(item["image"]).convert("RGB")
                                img_tensor = preprocess(img).unsqueeze(0).to(device)
                            elif isinstance(item["image"], torch.Tensor):
                                # It's already a tensor - ensure proper shape and normalization
                                img_tensor = item["image"].unsqueeze(0).to(device)
                            else:
                                # It's likely a PIL image
                                img_tensor = (
                                    preprocess(item["image"]).unsqueeze(0).to(device)
                                )

                            # Extract features
                            feature = (
                                pretrained_model(img_tensor).squeeze().cpu().numpy()
                            )
                            features.append(feature)
                        except Exception as e:
                            # On error, add zero features
                            logger.warning(f"Error extracting features: {e}")
                            features.append(np.zeros(512))  # ResNet18 feature size

                # Convert to numpy array
                features = np.array(features)

                # Use max_samples_per_group if set in command line args
                # This ensures coordination between initial group creation and batch sampling
                if (
                    hasattr(self, "max_samples_per_group")
                    and self.max_samples_per_group is not None
                ):
                    target_group_size = self.max_samples_per_group
                    logger.info(
                        f"Using max_samples_per_group={target_group_size} for clustering"
                    )
                else:
                    target_group_size = 5  # Default target group size

                # Calculate optimal number of clusters to achieve target group size
                n_clusters = max(
                    10, len(self.dataset) // target_group_size
                )  # At least 10 clusters
                n_clusters = min(
                    n_clusters, len(self.dataset) // 2
                )  # But no more than half the dataset size

                # Apply K-means clustering
                logger.info(f"Clustering into {n_clusters} semantic groups...")
                kmeans = KMeans(n_clusters=n_clusters, random_state=42)
                clusters = kmeans.fit_predict(features)

                # Group indices by cluster_id
                cluster_groups = defaultdict(list)
                for idx, cluster_id in enumerate(clusters):
                    cluster_groups[cluster_id].append(idx)

                # Apply capping to clusters if needed
                if (
                    hasattr(self, "max_samples_per_group")
                    and self.max_samples_per_group is not None
                ):
                    for cluster_id, indices in list(cluster_groups.items()):
                        if len(indices) > self.max_samples_per_group:
                            logger.info(
                                f"Capping cluster {cluster_id} from {len(indices)} to {self.max_samples_per_group} samples"
                            )
                            # Randomly sample indices
                            np.random.shuffle(indices)
                            cluster_groups[cluster_id] = indices[
                                : self.max_samples_per_group
                            ]

                # Assign match_ids based on clusters
                # Make sure all items in dataset are updated
                for idx in range(len(self.dataset)):
                    # Default match_id for any item not in a cluster
                    if isinstance(self.dataset[idx], dict):
                        self.dataset[idx]["match_id"] = f"unassigned_{idx}"

                # Now assign proper cluster-based match_ids
                for cluster_id, indices in cluster_groups.items():
                    for idx in indices:
                        if isinstance(self.dataset[idx], dict):
                            self.dataset[idx][
                                "match_id"
                            ] = f"semantic_group_{cluster_id}"

                logger.info(
                    f"Successfully created {n_clusters} embedding-based semantic groups"
                )

            except ImportError as e:
                logger.warning(
                    f"Couldn't import required libraries for embedding-based grouping: {e}"
                )
                logger.warning("Falling back to sequential grouping")

                # Original sequential grouping as fallback
                for group_idx in range(num_groups):
                    match_id = f"sequential_group_{group_idx}"
                    start_idx = group_idx * group_size
                    end_idx = min(start_idx + group_size, len(self.dataset))
                    for idx in range(start_idx, end_idx):
                        self.dataset[idx]["match_id"] = match_id

        except Exception as e:
            logger.error(f"Error in embedding-based grouping: {e}")
            logger.warning("Falling back to sequential grouping")

            # Original sequential grouping as fallback
            for group_idx in range(num_groups):
                match_id = f"sequential_group_{group_idx}"
                start_idx = group_idx * group_size
                end_idx = min(start_idx + group_size, len(self.dataset))
                for idx in range(start_idx, end_idx):
                    self.dataset[idx]["match_id"] = match_id

        # Store all match_ids for convenience, with error handling
        self.match_ids = []
        for i, item in enumerate(self.dataset):
            try:
                if isinstance(item, dict) and "match_id" in item:
                    self.match_ids.append(item["match_id"])
                else:
                    # For items without match_id, create a unique fallback ID
                    fallback_id = f"fallback_id_{i}"
                    if isinstance(item, dict):
                        item["match_id"] = fallback_id
                    self.match_ids.append(fallback_id)
                    logger.warning(
                        f"Item {i} missing match_id, using fallback: {fallback_id}"
                    )
            except Exception as e:
                # Create a unique fallback ID
                fallback_id = f"error_id_{i}"
                if isinstance(item, dict):
                    item["match_id"] = fallback_id
                self.match_ids.append(fallback_id)
                logger.warning(
                    f"Error with item {i}: {e}, using fallback: {fallback_id}"
                )

        # Diagnostics: check how many unique match_ids we created
        unique_match_ids = len(set(self.match_ids))
        items_with_match_id = sum(1 for item in self.dataset if "match_id" in item)
        logger.info(
            f"Created {unique_match_ids} artificial semantic groups for {items_with_match_id} items"
        )

        # Calculate average group size
        id_counts = defaultdict(int)
        for match_id in self.match_ids:
            id_counts[match_id] += 1

        # Apply min_samples_per_group and max_samples_per_group filtering to log accurate stats
        if hasattr(self, "min_samples_per_group") and self.min_samples_per_group > 1:
            # Remove groups that are too small
            id_counts = {
                k: v for k, v in id_counts.items() if v >= self.min_samples_per_group
            }
            logger.info(
                f"After applying min_samples_per_group={self.min_samples_per_group}: {len(id_counts)} valid groups remain"
            )

        group_sizes = list(id_counts.values())
        avg_group_size = sum(group_sizes) / len(group_sizes) if group_sizes else 0
        logger.info(
            f"Group statistics: min={min(group_sizes) if group_sizes else 0}, "
            f"max={max(group_sizes) if group_sizes else 0}, avg={avg_group_size:.2f}"
        )

        # Print statistics about semantic groups
        unique_match_ids = len(set(self.match_ids))
        avg_group_size = len(self.dataset) / max(1, unique_match_ids)
        logger.info(
            f"Created {unique_match_ids} semantic match groups with average size {avg_group_size:.2f}"
        )

        # Thoroughly break position correlation by completely shuffling (for all splits)
        # This is critical to prevent the model from learning shortcuts
        combined = list(zip(self.dataset, self.match_ids))
        random.shuffle(combined)
        self.dataset, self.match_ids = zip(*combined)

        # Unpack the shuffled data
        self.dataset = list(self.dataset)  # Convert back to list
        self.match_ids = list(self.match_ids)  # Convert back to list

        # Double-check shuffling by comparing first 10 indices and match_ids
        logger.info(f"First 10 match_ids after shuffling: {self.match_ids[:10]}")
        # This should show varying match_ids, not a simple pattern

        logger.info(f"Loaded {len(self.dataset)} image-text pairs for {split} split")

    def _load_flickr30k(self):
        """Load Flickr30k dataset from Hugging Face."""
        logger.info(f"Loading Flickr30k dataset for {self.split} split...")

        # Define cache paths for flickr30k
        data_root = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
            "data",
            "flickr30k",
        )
        os.makedirs(data_root, exist_ok=True)

        cache_dir = os.path.join(data_root, f"cache_{self.split}")
        cache_metadata = os.path.join(cache_dir, "metadata.json")
        cache_samples = os.path.join(cache_dir, "samples.pkl")

        # Try to load from cache first
        if os.path.exists(cache_metadata) and os.path.exists(cache_samples):
            try:
                import pickle

                # Load samples from pickle file
                with open(cache_samples, "rb") as f:
                    loaded_dataset = pickle.load(f)

                # Ensure cache is valid
                if loaded_dataset:
                    self.dataset = loaded_dataset
                    self.loaded_from_cache = True  # Set flag to indicate cache was used
                    logger.info(
                        f"Successfully loaded {len(self.dataset)} examples from cache for {self.split} split"
                    )
                    return
            except Exception as e:
                logger.warning(f"Error loading from cache: {str(e)}")
                # Continue with loading from HuggingFace

        try:
            # Load the dataset from Hugging Face
            import tqdm

            dataset_dict = load_dataset(
                "nlphuji/flickr30k",
                cache_dir=self.cache_dir
                or os.path.join(os.path.dirname(data_root), "nlphuji___flickr30k"),
            )

            # Flickr30k on HuggingFace has specific structure - adapt as needed
            if not isinstance(dataset_dict, dict) or "test" not in dataset_dict:
                raise ValueError("Expected dataset with 'test' split")

            # The data is stored in the 'test' split with a 'split' field specifying
            # the actual train/val/test designation
            test_dataset = dataset_dict["test"]

            # Create a progress bar for processing
            progress_bar = tqdm.tqdm(
                total=len(test_dataset),
                desc=f"Downloading and processing Flickr30k {self.split}",
            )

            # Filter by split
            self.dataset = []
            for i in range(len(test_dataset)):
                item = test_dataset[i]
                if isinstance(item, dict) and item.get("split") == self.split:
                    self.dataset.append(
                        {
                            "image": item["image"],
                            "captions": (
                                [item["caption"]]
                                if "caption" in item
                                else item.get("captions", [])
                            ),
                            "image_id": str(i),
                            "idx": i,  # Keep track of original index
                        }
                    )
                progress_bar.update(1)

            progress_bar.close()

            if not self.dataset:
                raise ValueError(f"No examples found for split '{self.split}'")

            logger.info(
                f"Successfully loaded {len(self.dataset)} examples from Flickr30k {self.split} split"
            )

            # Data was loaded from Hugging Face directly, so it's real data
            self.loaded_from_cache = True

            # Save to cache for next time
            try:
                import pickle
                import json

                # Create cache directory if it doesn't exist
                os.makedirs(cache_dir, exist_ok=True)

                # Save samples to pickle file
                with open(cache_samples, "wb") as f:
                    pickle.dump(self.dataset, f)

                # Save metadata
                metadata = {
                    "num_samples": len(self.dataset),
                    "split": self.split,
                    "date_created": time.strftime("%Y-%m-%d %H:%M:%S"),
                }

                with open(cache_metadata, "w") as f:
                    json.dump(metadata, f, indent=2)

                logger.info(
                    f"Saved {len(self.dataset)} samples to cache at {cache_dir}"
                )
            except Exception as e:
                logger.warning(f"Error saving to cache: {str(e)}")

        except Exception as e:
            logger.error(f"Error loading Flickr30k: {str(e)}")
            raise

    def _generate_synthetic_data(self):
        """Generate synthetic data as a fallback."""
        logger.info(f"Generating {self.synthetic_samples} synthetic samples")

        # Mark as not loaded from cache (it's synthetic)
        self.loaded_from_cache = False

        # Define cache paths for synthetic data
        data_root = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
            "data",
            "flickr30k",
        )
        os.makedirs(data_root, exist_ok=True)

        synthetic_dir = os.path.join(data_root, f"synthetic_{self.split}")
        os.makedirs(synthetic_dir, exist_ok=True)

        cache_dir = os.path.join(data_root, f"cache_{self.split}")
        cache_metadata = os.path.join(cache_dir, "metadata.json")
        cache_samples = os.path.join(cache_dir, "samples.pkl")

        import tqdm

        # Create a progress bar for generating synthetic data
        progress_bar = tqdm.tqdm(
            total=self.synthetic_samples,
            desc=f"Generating synthetic data for {self.split}",
        )

        self.dataset = []
        for i in range(self.synthetic_samples):
            # Create a random RGB image with some structure
            img = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)

            # Add some structure to make it more interesting
            x, y = np.mgrid[0:224, 0:224]

            # Add a shape based on index
            shape_type = i % 4
            if shape_type == 0:  # Circle
                circle = (x - 112) ** 2 + (y - 112) ** 2 < 50**2
                img[circle, 0] = 200  # Red circle
                shape_name = "circle"
                color_name = "red"
            elif shape_type == 1:  # Square
                img[70:140, 70:140, 1] = 200  # Green square
                shape_name = "square"
                color_name = "green"
            elif shape_type == 2:  # Rectangle
                img[60:100, 60:160, 2] = 200  # Blue rectangle
                shape_name = "rectangle"
                color_name = "blue"
            else:  # Triangle
                tri_y, tri_x = np.mgrid[40:140, 40:140]
                triangle = tri_x > tri_y
                img[tri_y[triangle], tri_x[triangle], 1] = 180  # Green triangle
                shape_name = "triangle"
                color_name = "green"

            # Convert to PIL Image
            img = Image.fromarray(img)

            # Create a simple caption
            caption = f"A {color_name} {shape_name} on a plain background"

            self.dataset.append(
                {
                    "image": img,
                    "captions": [caption],
                    "image_id": str(i),
                    "idx": i,  # Keep track of index for evaluation
                }
            )

            progress_bar.update(1)

        progress_bar.close()
        logger.info(f"Generated {len(self.dataset)} synthetic examples")

        # Save synthetic data to cache
        try:
            import pickle
            import json

            # Create cache directory if it doesn't exist
            os.makedirs(cache_dir, exist_ok=True)

            # Save samples to pickle file
            with open(cache_samples, "wb") as f:
                pickle.dump(self.dataset, f)

            # Save metadata
            metadata = {
                "num_samples": len(self.dataset),
                "split": self.split,
                "date_created": time.strftime("%Y-%m-%d %H:%M:%S"),
                "type": "synthetic",
            }

            with open(cache_metadata, "w") as f:
                json.dump(metadata, f, indent=2)

            logger.info(f"Saved synthetic data to cache at {cache_dir}")
        except Exception as e:
            logger.warning(f"Error saving synthetic data to cache: {str(e)}")

    def __len__(self):
        """
        Return the number of examples in the dataset.

        When captions_per_image > 1, this multiplies the base dataset length
        to create multiple entries per image.
        """
        if not hasattr(self, "dataset") or self.dataset is None:
            return 0

        base_length = len(self.dataset)

        # If using only 1 caption per image, return the original length
        if self.captions_per_image <= 1:
            return base_length

        # For multi-caption mode, we need to estimate the effective length
        # Check first few items to see how many captions they have on average
        caption_counts = []
        sample_size = min(100, base_length)  # Check up to 100 samples

        for i in range(sample_size):
            item = self.dataset[i]
            if "captions" in item and isinstance(item["captions"], list):
                caption_counts.append(
                    min(len(item["captions"]), self.captions_per_image)
                )
            else:
                caption_counts.append(1)

        # Calculate average number of captions per item
        if caption_counts:
            avg_captions = sum(caption_counts) / len(caption_counts)
            # Use the minimum of requested captions and average available
            effective_captions = min(self.captions_per_image, avg_captions)
        else:
            effective_captions = 1

        # Return the expanded dataset length
        return int(base_length * effective_captions)

    def __getitem__(self, idx):
        """
        Get a single example from the dataset with proper format for MultimodalTrainer.

        Args:
            idx: Index of the example to retrieve

        Returns:
            Dictionary containing:
            - 'image': Preprocessed image tensor
            - 'text': Dictionary with 'src' and 'src_mask'
            - 'raw_text': Original caption text
            - 'match_id': ID that determines which items should match
        """
        # Get image and caption
        item = self.dataset[idx]

        # Handle different key formats in dataset
        if "image" in item:
            image = item["image"]  # This might be a PIL Image or path
        elif "image_path" in item:
            # Load image from path
            try:
                image = Image.open(item["image_path"]).convert("RGB")
            except Exception as e:
                logger.warning(f"Error loading image from path: {str(e)}")
                image = Image.new("RGB", (224, 224), (0, 0, 0))
        else:
            logger.warning(f"No image data found in item {idx}")
            image = Image.new("RGB", (224, 224), (0, 0, 0))

        # Caption selection based on captions_per_image parameter
        if (
            "captions" in item
            and isinstance(item["captions"], list)
            and item["captions"]
        ):
            # Get image_id for deterministic selection
            image_id = item.get("image_id", str(idx))

            if self.captions_per_image == 1:
                # Use a deterministic selection based on image_id hash for single caption
                caption_idx = abs(hash(image_id)) % len(item["captions"])
                caption = item["captions"][caption_idx]
            else:
                # For multiple captions, select a subset based on the index modulo
                # This enables us to create multiple entries per image in the dataset expansion
                available_captions = item["captions"]
                num_captions = min(self.captions_per_image, len(available_captions))

                # Use modulo of idx to determine which caption to use for this specific entry
                # This distributes different captions across different __getitem__ calls
                effective_idx = idx % num_captions

                # For deterministic selection, hash the image_id with the effective_idx
                caption_idx = abs(hash(f"{image_id}_{effective_idx}")) % len(
                    available_captions
                )
                caption = available_captions[caption_idx]

                # Store the original caption index for tracking
                item["caption_idx"] = caption_idx
        elif "caption" in item:
            caption = item["caption"]
        else:
            caption = "Empty caption"

        # Ensure caption is a string
        if isinstance(caption, (list, tuple)):
            caption = caption[0] if caption else ""
        caption = str(caption)

        # Process image
        try:
            if self.image_preprocessor:
                image_tensor = self.image_preprocessor.preprocess(image)
            else:
                raise ValueError("Image preprocessor is required")
        except Exception as e:
            logger.warning(f"Error processing image: {str(e)}")
            image_tensor = torch.zeros(3, 224, 224)

        # Process caption
        try:
            if self.tokenizer is None:
                raise ValueError("Tokenizer is not available")

            token_ids = self.tokenizer.encode(caption)

            # Pad or truncate to max_text_length
            if len(token_ids) > self.max_text_length:
                token_ids = token_ids[: self.max_text_length]
            else:
                padding = [self.tokenizer.special_tokens["pad_token_idx"]] * (
                    self.max_text_length - len(token_ids)
                )
                token_ids = token_ids + padding

            # Create source tensor
            src = torch.tensor(token_ids, dtype=torch.long)

            # Create source mask
            mask = src != self.tokenizer.special_tokens["pad_token_idx"]
            src_mask = mask.unsqueeze(0).unsqueeze(0)
        except Exception as e:
            logger.warning(f"Error processing text: {str(e)}")
            pad_token_idx = 0
            if self.tokenizer and hasattr(self.tokenizer, "special_tokens"):
                pad_token_idx = self.tokenizer.special_tokens.get("pad_token_idx", 0)

            src = torch.zeros(self.max_text_length, dtype=torch.long)
            src_mask = torch.zeros(1, 1, self.max_text_length, dtype=torch.bool)

        # Ensure match_id is properly included in the return dictionary
        # This is critical for content-based contrastive learning
        match_id = item.get("match_id")

        # As a fallback, use the stored match_ids if available
        if (
            match_id is None
            and hasattr(self, "match_ids")
            and idx < len(self.match_ids)
        ):
            match_id = self.match_ids[idx]

        # Last resort - create a unique ID for this item (should never happen)
        if match_id is None:
            match_id = f"fallback_{idx}"

        # Replace the warning with a more specific check
        # We don't want to warn about group_123 when idx=23, which is a false positive
        # Only warn if the exact idx is used as a match_id
        if match_id == f"fallback_{idx}" and self.split == "train":
            logger.warning(
                f"Using fallback match_id ({match_id}) for item {idx} - this may cause shortcut learning!"
            )

        # Return properly formatted batch item
        return {
            "image": image_tensor,
            "text": {
                "src": src,
                "src_mask": src_mask,
            },
            "raw_text": caption,
            "match_id": match_id,  # Important for content-based matching
            "idx": idx,  # Keep original index for reference
        }

    def get_split_proportions(self):
        """Return dataset split information for logging."""
        return {
            "total_samples": len(self.dataset),
            "split": self.split,
            "dataset_name": self.dataset_name,
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
        "module_purpose": "Implements dataset classes for handling multimodal data pairs with support for various data formats and preprocessing",
        "key_classes": [
            {
                "name": "MultimodalDataset",
                "purpose": "Base dataset class for handling image-text pairs with configurable preprocessing",
                "key_methods": [
                    {
                        "name": "__init__",
                        "signature": "__init__(self, data_root: str, image_processor: Optional[Union[ImagePreprocessor, transforms.Compose]] = None, text_tokenizer=None, max_text_length: int = 77, split: str = 'train', transform_image: Optional[Callable] = None, transform_text: Optional[Callable] = None, metadata_file: str = 'metadata.json', image_key: str = 'image_path', caption_key: str = 'caption', label_key: Optional[str] = 'label', image_dir: str = 'images', limit_samples: Optional[int] = None, return_metadata: bool = False)",
                        "brief_description": "Initialize dataset with data path and preprocessing options",
                    },
                    {
                        "name": "__getitem__",
                        "signature": "__getitem__(self, idx: int) -> Dict[str, Any]",
                        "brief_description": "Get a processed image-text pair with optional transformations",
                    },
                    {
                        "name": "_load_metadata",
                        "signature": "_load_metadata(self, metadata_path: str, split: str) -> List[Dict]",
                        "brief_description": "Load and validate dataset metadata from file",
                    },
                    {
                        "name": "get_hard_negative",
                        "signature": "get_hard_negative(self, idx: int, neg_type: str = 'same_class') -> int",
                        "brief_description": "Get index of hard negative sample for contrastive learning",
                    },
                ],
                "inheritance": "Dataset",
                "dependencies": ["torch.utils.data", "PIL", "torchvision"],
            },
            {
                "name": "Flickr30kDataset",
                "purpose": "Specialized dataset for Flickr30k with 5 captions per image support",
                "key_methods": [
                    {
                        "name": "__init__",
                        "signature": "__init__(self, data_root: str = '', image_processor: Optional[Union[ImagePreprocessor, transforms.Compose]] = None, text_tokenizer=None, max_text_length: int = 77, split: str = 'train', transform_image: Optional[Callable] = None, transform_text: Optional[Callable] = None, limit_samples: Optional[int] = None, return_metadata: bool = False)",
                        "brief_description": "Initialize Flickr30k dataset with specialized configuration",
                    },
                    {
                        "name": "_load_from_cache",
                        "signature": "_load_from_cache(self) -> bool",
                        "brief_description": "Load preprocessed dataset from cache if available",
                    },
                    {
                        "name": "_generate_synthetic_data",
                        "signature": "_generate_synthetic_data(self)",
                        "brief_description": "Create synthetic data for testing when real data unavailable",
                    },
                ],
                "inheritance": "MultimodalDataset",
                "dependencies": ["torch.utils.data", "PIL", "torchvision"],
            },
            {
                "name": "EnhancedMultimodalDataset",
                "purpose": "Advanced dataset with semantic grouping, caching, and multiple caption support",
                "key_methods": [
                    {
                        "name": "__init__",
                        "signature": "__init__(self, split: str = 'train', image_preprocessor=None, tokenizer=None, max_text_length: int = 77, dataset_name: str = 'flickr30k', synthetic_samples: int = 100, cache_dir: Optional[str] = None, max_samples: Optional[int] = None, captions_per_image: int = 1, min_samples_per_group: int = 2, max_samples_per_group: Optional[int] = None, cap_strategy: str = 'random')",
                        "brief_description": "Initialize enhanced dataset with advanced configuration options",
                    },
                    {
                        "name": "__getitem__",
                        "signature": "__getitem__(self, idx)",
                        "brief_description": "Get sample with proper format for MultimodalTrainer including match_id",
                    },
                    {
                        "name": "_load_flickr30k",
                        "signature": "_load_flickr30k(self)",
                        "brief_description": "Load and process Flickr30k dataset with semantic grouping",
                    },
                ],
                "inheritance": "Dataset",
                "dependencies": ["torch.utils.data", "PIL", "Image"],
            },
        ],
        "external_dependencies": ["torch", "PIL", "numpy", "torchvision"],
        "complexity_score": 8,
    }
