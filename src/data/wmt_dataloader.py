import os
import random
from typing import List, Optional, Tuple


class WMTDataLoader:
    def __init__(self, data_dir: str, source_lang: str, target_lang: str, batch_size: int = 32, max_examples: Optional[int] = None, seed: int = 42, shuffle: bool = True):
        self.data_dir = data_dir
        self.source_lang = source_lang
        self.target_lang = target_lang
        self.batch_size = batch_size
        self.max_examples = max_examples
        self.shuffle = shuffle
        random.seed(seed)

        # Load the data
        self.source_data, self.target_data = self.load_data()

    def load_data(self) -> Tuple[List[str], List[str]]:
        # Define possible file patterns
        possible_patterns = [
            (f"{self.data_dir}/news-commentary-v9.{self.source_lang}-{self.target_lang}.{self.source_lang}",
             f"{self.data_dir}/news-commentary-v9.{self.source_lang}-{self.target_lang}.{self.target_lang}")
        ]

        # Find the correct file pattern
        src_file, tgt_file = None, None
        for src_pattern, tgt_pattern in possible_patterns:
            if os.path.exists(src_pattern) and os.path.exists(tgt_pattern):
                src_file, tgt_file = src_pattern, tgt_pattern
                print(f"Found WMT files using pattern: {src_pattern.split('/')[-1]}")
                break

        if src_file is None or tgt_file is None:
            raise FileNotFoundError(f"Could not find WMT data files for {self.source_lang}-{self.target_lang} in directory {self.data_dir}.")

        # Read data files
        with open(src_file, 'r', encoding='utf-8') as f:
            src_data = [line.strip() for line in f if line.strip()]

        with open(tgt_file, 'r', encoding='utf-8') as f:
            tgt_data = [line.strip() for line in f if line.strip()]

        # Ensure same length
        if len(src_data) != len(tgt_data):
            print(f"Warning: Source and target files have different lengths. Source: {len(src_data)}, Target: {len(tgt_data)}")
            min_len = min(len(src_data), len(tgt_data))
            src_data = src_data[:min_len]
            tgt_data = tgt_data[:min_len]

        # Filter out empty lines and lines that are too long or short
        filtered_pairs = [(src, tgt) for src, tgt in zip(src_data, tgt_data) if src and tgt and len(src.split()) <= 100 and len(tgt.split()) <= 100]

        # Shuffle and limit
        if self.shuffle:
            random.shuffle(filtered_pairs)
        if self.max_examples is not None and self.max_examples < len(filtered_pairs):
            filtered_pairs = filtered_pairs[:self.max_examples]

        # Unzip the pairs
        src_data, tgt_data = zip(*filtered_pairs) if filtered_pairs else ([], [])

        print(f"Loaded {len(src_data)} parallel sentences")

        return list(src_data), list(tgt_data)

    def __iter__(self) -> Tuple[List[str], List[str]]:
        for i in range(0, len(self.source_data), self.batch_size):
            source_batch = self.source_data[i:i + self.batch_size]
            target_batch = self.target_data[i:i + self.batch_size]
            yield source_batch, target_batch

# Example usage:
# dataloader = WMTDataLoader(data_dir='data/wmt/training', source_lang='en', target_lang='fr', max_examples=1000)
# for source_batch, target_batch in dataloader:
#     # Process batches

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
        "module_purpose": "Provides a data loader for WMT (Workshop on Machine Translation) parallel corpus",
        "key_classes": [
            {
                "name": "WMTDataLoader",
                "purpose": "Loads and preprocesses WMT parallel corpus data with batching capability",
                "key_methods": [
                    {
                        "name": "__init__",
                        "signature": "__init__(self, data_dir: str, source_lang: str, target_lang: str, batch_size: int = 32, max_examples: Optional[int] = None, seed: int = 42, shuffle: bool = True)",
                        "brief_description": "Initialize the WMT data loader with configurable batch size and filtering"
                    },
                    {
                        "name": "load_data",
                        "signature": "load_data(self) -> Tuple[List[str], List[str]]",
                        "brief_description": "Load and preprocess parallel data"
                    },
                    {
                        "name": "__iter__",
                        "signature": "__iter__(self) -> Tuple[List[str], List[str]]",
                        "brief_description": "Yield batches of source and target sentences"
                    }
                ],
                "inheritance": "object",
                "dependencies": ["os", "random"]
            }
        ],
        "external_dependencies": ["os", "random"],
        "complexity_score": 3  # Moderate complexity
    }
