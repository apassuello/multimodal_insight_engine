# Create this file as src/data/europarl_dataset.py

import os
import random
from typing import List, Optional, Tuple


class EuroparlDataset:
    """
    Dataset class for the Europarl parallel corpus.
    
    This class handles loading and preprocessing parallel text data from the
    Europarl corpus for machine translation tasks.
    """

    def __init__(
        self,
        data_dir: str = "data/europarl",
        src_lang: str = "de",
        tgt_lang: str = "en",
        max_examples: Optional[int] = None,
        random_seed: int = 42
    ):
        """
        Initialize the Europarl dataset.
        
        Args:
            data_dir: Directory containing the Europarl data
            src_lang: Source language code
            tgt_lang: Target language code
            max_examples: Maximum number of examples to use (None = use all)
            random_seed: Random seed for reproducibility
        """
        self.data_dir = data_dir
        self.src_lang = src_lang
        self.tgt_lang = tgt_lang
        self.max_examples = max_examples

        # Set random seed for reproducibility
        random.seed(random_seed)

        # Load the data
        self.src_data, self.tgt_data = self.load_data()

    def load_data(self) -> Tuple[List[str], List[str]]:
        """
        Load and preprocess the parallel data.
        
        This function tries multiple possible file structures for Europarl data.
        
        Returns:
            Tuple containing lists of source and target sentences
        """
        # Try multiple possible file structures
        possible_patterns = [
            # Pattern 1: Direct language files in the main directory
            (f"{self.data_dir}/europarl-v7.{self.src_lang}-{self.tgt_lang}.{self.src_lang}",
             f"{self.data_dir}/europarl-v7.{self.src_lang}-{self.tgt_lang}.{self.tgt_lang}"),

            # Pattern 2: Language pair subdirectory
            (f"{self.data_dir}/{self.src_lang}-{self.tgt_lang}/europarl.{self.src_lang}",
             f"{self.data_dir}/{self.src_lang}-{self.tgt_lang}/europarl.{self.tgt_lang}"),

            # Pattern 3: Language files with different naming
            (f"{self.data_dir}/europarl.{self.src_lang}",
             f"{self.data_dir}/europarl.{self.tgt_lang}"),

            # Pattern 4: Simple text files named by language
            (f"{self.data_dir}/{self.src_lang}.txt",
             f"{self.data_dir}/{self.tgt_lang}.txt"),
        ]

        # Try each pattern until we find files that exist
        src_file, tgt_file = None, None
        for src_pattern, tgt_pattern in possible_patterns:
            if os.path.exists(src_pattern) and os.path.exists(tgt_pattern):
                src_file, tgt_file = src_pattern, tgt_pattern
                print(f"Found Europarl files using pattern: {src_pattern.split('/')[-1]}")
                break

        # If no pattern matched, raise an error
        if src_file is None or tgt_file is None:
            raise FileNotFoundError(
                f"Could not find Europarl data files for {self.src_lang}-{self.tgt_lang} "
                f"in directory {self.data_dir}. Please check the file structure."
            )

        # Read data files
        print(f"Loading source data from: {src_file}")
        with open(src_file, 'r', encoding='utf-8') as f:
            src_data = [line.strip() for line in f if line.strip()]

        print(f"Loading target data from: {tgt_file}")
        with open(tgt_file, 'r', encoding='utf-8') as f:
            tgt_data = [line.strip() for line in f if line.strip()]

        # Ensure same length
        if len(src_data) != len(tgt_data):
            print(f"Warning: Source and target files have different lengths. "
                  f"Source: {len(src_data)}, Target: {len(tgt_data)}")
            min_len = min(len(src_data), len(tgt_data))
            src_data = src_data[:min_len]
            tgt_data = tgt_data[:min_len]

        # Filter out empty lines and lines that are too long or short
        filtered_pairs = []
        for src, tgt in zip(src_data, tgt_data):
            # Skip if either is empty
            if not src or not tgt:
                continue

            # Skip if either is too long (optional, adjust as needed)
            if len(src.split()) > 100 or len(tgt.split()) > 100:
                continue

            filtered_pairs.append((src, tgt))

        # Shuffle and limit
        random.shuffle(filtered_pairs)
        if self.max_examples is not None and self.max_examples < len(filtered_pairs):
            filtered_pairs = filtered_pairs[:self.max_examples]

        # Unzip the pairs
        src_data, tgt_data = zip(*filtered_pairs) if filtered_pairs else ([], [])

        print(f"Loaded {len(src_data)} parallel sentences")

        return list(src_data), list(tgt_data)

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
        "module_purpose": "Provides a dataset class for loading and preprocessing Europarl parallel corpus data",
        "key_classes": [
            {
                "name": "EuroparlDataset",
                "purpose": "Handles loading and preprocessing parallel text data from the Europarl corpus",
                "key_methods": [
                    {
                        "name": "__init__",
                        "signature": "__init__(self, data_dir: str = 'data/europarl', src_lang: str = 'de', tgt_lang: str = 'en', max_examples: Optional[int] = None, random_seed: int = 42)",
                        "brief_description": "Initialize the dataset with language pair and optional filtering"
                    },
                    {
                        "name": "load_data",
                        "signature": "load_data(self) -> Tuple[List[str], List[str]]",
                        "brief_description": "Load and preprocess parallel data with multiple file pattern detection"
                    }
                ],
                "inheritance": "object",
                "dependencies": ["os", "random"]
            }
        ],
        "external_dependencies": ["os", "random"],
        "complexity_score": 4  # Moderate complexity for handling multiple file formats
    }
