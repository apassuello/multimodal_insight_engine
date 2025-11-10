import os
import tensorflow as tf
import torch
import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Union
from tqdm import tqdm

class WikipediaDataset:
    """
    Dataset class for the Wikipedia Web2M dataset.
    
    This class handles loading and preprocessing data from the WikiWeb2M dataset
    stored in TFRecord format.
    """
    
    def __init__(
        self,
        data_dir: str = "data/wiki",
        split: str = "train",
        max_examples: Optional[int] = None,
        cache_processed_data: bool = True,
        cache_dir: Optional[str] = None,
        image_size: int = 224,
        random_seed: int = 42
    ):
        """
        Initialize the Wikipedia dataset.
        
        Args:
            data_dir: Directory containing the Wikipedia data
            split: Data split to use ('train', 'val', or 'test')
            max_examples: Maximum number of examples to use (None = use all)
            cache_processed_data: Whether to cache processed data to disk
            cache_dir: Directory to store cached data (defaults to data_dir/cache)
            image_size: Size to resize images to
            random_seed: Random seed for reproducibility
        """
        self.data_dir = data_dir
        self.split = split
        self.max_examples = max_examples
        self.cache_processed_data = cache_processed_data
        self.cache_dir = cache_dir or os.path.join(data_dir, "cache")
        self.image_size = image_size
        
        # Ensure cache directory exists if caching is enabled
        if self.cache_processed_data and not os.path.exists(self.cache_dir):
            os.makedirs(self.cache_dir)
        
        # Set random seed for reproducibility
        np.random.seed(random_seed)
        tf.random.set_seed(random_seed)
        
        # Load the data
        self.data = self.load_data()
        
    def _get_file_paths(self) -> List[str]:
        """
        Get file paths for the specified split.
        
        Returns:
            List of file paths for the specified split
        """
        if self.split == "train":
            # Handle sharded training files
            pattern = os.path.join(self.data_dir, f"wikiweb2m-{self.split}.tfrecord.gz-*")
            return sorted(tf.io.gfile.glob(pattern))
        else:
            # Handle single validation and test files
            filepath = os.path.join(self.data_dir, f"wikiweb2m-{self.split}.tfrecord.gz")
            if tf.io.gfile.exists(filepath):
                return [filepath]
            else:
                raise FileNotFoundError(f"Could not find file for {self.split} split: {filepath}")
    
    def _parse_example(self, example_proto):
        """
        Parse a single example from the TFRecord.
        
        Args:
            example_proto: Serialized example from TFRecord
            
        Returns:
            Parsed feature dictionary
        """
        feature_description = {
            'image/encoded': tf.io.FixedLenFeature([], tf.string),
            'image/format': tf.io.FixedLenFeature([], tf.string),
            'image/height': tf.io.FixedLenFeature([], tf.int64),
            'image/width': tf.io.FixedLenFeature([], tf.int64),
            'text/encoded': tf.io.FixedLenFeature([], tf.string),
            'text/format': tf.io.FixedLenFeature([], tf.string),
            'webpage/url': tf.io.FixedLenFeature([], tf.string, default_value=''),
            'webpage/title': tf.io.FixedLenFeature([], tf.string, default_value=''),
        }
        
        return tf.io.parse_single_example(example_proto, feature_description)
    
    def _process_image(self, image_data):
        """
        Process image data from raw bytes.
        
        Args:
            image_data: Raw image bytes
            
        Returns:
            Processed image tensor
        """
        image = tf.image.decode_image(image_data, channels=3)
        image = tf.image.resize(image, [self.image_size, self.image_size])
        image = tf.cast(image, tf.float32) / 255.0  # Normalize to [0, 1]
        return image
    
    def _process_text(self, text_data):
        """
        Process text data from raw bytes.
        
        Args:
            text_data: Raw text bytes
            
        Returns:
            Processed text string
        """
        return text_data.decode('utf-8')
    
    def _get_cache_path(self):
        """Get path for cached processed data."""
        max_str = f"_max{self.max_examples}" if self.max_examples else ""
        return os.path.join(self.cache_dir, f"wikiweb2m_{self.split}{max_str}.pt")
    
    def load_data(self) -> Dict[str, List[Any]]:
        """
        Load and preprocess the data.
        
        Returns:
            Dictionary containing processed data
        """
        cache_path = self._get_cache_path() if self.cache_processed_data else None
        
        # Try to load from cache first
        if cache_path and os.path.exists(cache_path):
            print(f"Loading cached processed data from {cache_path}")
            return torch.load(cache_path, weights_only=True)
        
        # Get file paths for the specified split
        file_paths = self._get_file_paths()
        if not file_paths:
            raise FileNotFoundError(f"No files found for split '{self.split}' in {self.data_dir}")
            
        # Initialize data containers
        data = {
            'images': [],
            'texts': [],
            'urls': [],
            'titles': []
        }
        
        # Process each file
        total_examples = 0
        for file_path in file_paths:
            print(f"Processing {file_path}")
            dataset = tf.data.TFRecordDataset([file_path], compression_type="GZIP")
            
            for example_proto in tqdm(dataset):
                if self.max_examples is not None and total_examples >= self.max_examples:
                    break
                    
                parsed = self._parse_example(example_proto)
                
                # Process image
                image_data = parsed['image/encoded'].numpy()
                image = self._process_image(image_data).numpy()
                data['images'].append(image)
                
                # Process text
                text_data = parsed['text/encoded'].numpy()
                text = self._process_text(text_data)
                data['texts'].append(text)
                
                # Get metadata
                data['urls'].append(parsed['webpage/url'].numpy().decode('utf-8'))
                data['titles'].append(parsed['webpage/title'].numpy().decode('utf-8'))
                
                total_examples += 1
            
            if self.max_examples is not None and total_examples >= self.max_examples:
                break
        
        print(f"Loaded {total_examples} examples from {self.split} split")
        
        # Convert lists to tensors
        data['images'] = torch.tensor(np.array(data['images']), dtype=torch.float32)
        
        # Cache processed data if enabled
        if cache_path:
            print(f"Saving processed data to cache: {cache_path}")
            torch.save(data, cache_path)
        
        return data
    
    def to_pytorch_dataset(self):
        """
        Convert to a PyTorch dataset compatible with the MultimodalDataset class.
        
        Returns:
            Dictionary of tensors ready for MultimodalDataset
        """
        from src.data.dataloader import MultimodalDataset
        
        # Create tensor dictionary
        tensor_dict = {
            'image': self.data['images'],
            'text': self.data['texts'],  # This will be handled by the collate function
            'metadata': {
                'url': self.data['urls'],
                'title': self.data['titles']
            }
        }
        
        return MultimodalDataset(tensor_dict)

def create_wiki_dataloaders(
    data_dir: str = "data/wiki",
    batch_size: int = 32,
    max_examples: Optional[Dict[str, int]] = None,
    num_workers: int = 0,
    image_size: int = 224,
    random_seed: int = 42
) -> Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
    """
    Create DataLoaders for train, validation, and test sets.
    
    Args:
        data_dir: Directory containing the Wikipedia data
        batch_size: Batch size for the DataLoaders
        max_examples: Optional dictionary specifying max examples per split
        num_workers: Number of worker processes
        image_size: Size to resize images to
        random_seed: Random seed for reproducibility
        
    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    from src.data.dataloader import create_dataloader, collate_fn
    
    # Default to no limits if not specified
    if max_examples is None:
        max_examples = {}
    
    # Create datasets for each split
    train_dataset = WikipediaDataset(
        data_dir=data_dir,
        split="train",
        max_examples=max_examples.get("train"),
        image_size=image_size,
        random_seed=random_seed
    ).to_pytorch_dataset()
    
    val_dataset = WikipediaDataset(
        data_dir=data_dir,
        split="val",
        max_examples=max_examples.get("val"),
        image_size=image_size,
        random_seed=random_seed
    ).to_pytorch_dataset()
    
    test_dataset = WikipediaDataset(
        data_dir=data_dir,
        split="test",
        max_examples=max_examples.get("test"),
        image_size=image_size,
        random_seed=random_seed
    ).to_pytorch_dataset()
    
    # Create dataloaders
    train_loader = create_dataloader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate_fn
    )
    
    val_loader = create_dataloader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn
    )
    
    test_loader = create_dataloader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn
    )
    
    return train_loader, val_loader, test_loader

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
        "module_purpose": "Provides a dataset class for loading and preprocessing Wikipedia Web2M data from TFRecord format",
        "key_classes": [
            {
                "name": "WikipediaDataset",
                "purpose": "Handles loading and preprocessing multimodal (image-text) data from WikiWeb2M TFRecords",
                "key_methods": [
                    {
                        "name": "__init__",
                        "signature": "__init__(self, data_dir: str = 'data/wiki', split: str = 'train', max_examples: Optional[int] = None, cache_processed_data: bool = True, cache_dir: Optional[str] = None, image_size: int = 224, random_seed: int = 42)",
                        "brief_description": "Initialize the dataset with data split and processing options"
                    },
                    {
                        "name": "load_data",
                        "signature": "load_data(self) -> Dict[str, List[Any]]",
                        "brief_description": "Load and preprocess data from TFRecord files with caching capability"
                    },
                    {
                        "name": "to_pytorch_dataset",
                        "signature": "to_pytorch_dataset(self)",
                        "brief_description": "Convert to a PyTorch dataset compatible with MultimodalDataset"
                    }
                ],
                "inheritance": "object",
                "dependencies": ["tensorflow", "torch", "numpy", "tqdm"]
            }
        ],
        "key_functions": [
            {
                "name": "create_wiki_dataloaders",
                "signature": "create_wiki_dataloaders(data_dir: str = 'data/wiki', batch_size: int = 32, max_examples: Optional[Dict[str, int]] = None, num_workers: int = 0, image_size: int = 224, random_seed: int = 42) -> Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader, torch.utils.data.DataLoader]",
                "brief_description": "Create DataLoaders for train, validation, and test sets"
            }
        ],
        "external_dependencies": ["tensorflow", "torch", "numpy", "tqdm"],
        "complexity_score": 5  # High complexity for handling TFRecord data and image processing
    } 