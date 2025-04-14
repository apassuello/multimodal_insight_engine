"""MODULE: wmt_dataset.py
PURPOSE: Provides the WMTDataset class for loading and processing WMT translation datasets
with automatic download capabilities from Hugging Face.

KEY COMPONENTS:
- WMTDataset: Dataset loader for WMT translation data with robust fallback mechanisms

DEPENDENCIES:
- os, sys
- requests
- random
- tqdm
- datasets (Hugging Face)
"""

import os
import sys
import random
from tqdm import tqdm
from typing import List, Tuple, Optional, Dict, Any
import json


class WMTDataset:
    """Dataset class for the WMT translation dataset from Hugging Face datasets."""

    # Available years from Hugging Face
    AVAILABLE_YEARS = ["14", "16", "19"]

    # Default language pairs available for different WMT years
    DEFAULT_LANGUAGE_PAIRS = {
        "14": ["cs-en", "de-en", "fr-en", "hi-en", "ru-en"],
        "16": ["cs-en", "de-en", "fi-en", "ro-en", "ru-en", "tr-en"],
        "19": ["cs-en", "de-en", "fi-en", "gu-en", "kk-en", "lt-en", "ru-en", "zh-en"],
    }

    def __init__(
        self,
        src_lang="de",
        tgt_lang="en",
        year="14",
        split="train",
        max_examples=None,
        data_dir="data/wmt",
        random_seed=42,
        subset=None,
    ):
        """
        Initialize the WMT dataset.

        Args:
            src_lang: Source language code
            tgt_lang: Target language code
            year: Dataset year (without "wmt" prefix, e.g. "14" for WMT14)
            split: Data split (train, validation, test)
            max_examples: Maximum number of examples to use (None = use all)
            data_dir: Directory to store/cache the WMT data
            random_seed: Random seed for reproducibility
            subset: Optional dataset subset name (e.g., "news_commentary_v9")
        """
        self.src_lang = src_lang
        self.tgt_lang = tgt_lang
        self.year = str(year)  # Ensure year is a string
        self.split = self._map_split(split)  # Map split names to HF format
        self.max_examples = max_examples
        self.data_dir = data_dir
        self.subset = subset

        # Set random seed for reproducibility
        random.seed(random_seed)

        # Paths
        os.makedirs(self.data_dir, exist_ok=True)

        # Load data
        self.src_data, self.tgt_data = self.load_data()

        # Print dataset info
        print(f"Loaded {len(self.src_data)} {split} examples from WMT{self.year}")

    def _map_split(self, split: str) -> str:
        """
        Map common split names to those used in Hugging Face datasets.

        Args:
            split: Input split name

        Returns:
            Mapped split name for Hugging Face datasets
        """
        split_mapping = {
            "train": "train",
            "valid": "validation",
            "validation": "validation",
            "dev": "validation",
            "test": "test",
        }

        return split_mapping.get(split.lower(), split)

    def _get_config_name(self) -> str:
        """
        Get the configuration name for the dataset based on language pair.

        Returns:
            Configuration name for Hugging Face datasets
        """
        # For WMT datasets, config is typically language pair in format "lang1-lang2"
        lang_pair = f"{self.src_lang}-{self.tgt_lang}"

        # Check if the language pair exists for this year
        if lang_pair not in self.DEFAULT_LANGUAGE_PAIRS.get(self.year, []):
            # Try the reverse pair
            lang_pair_rev = f"{self.tgt_lang}-{self.src_lang}"
            if lang_pair_rev in self.DEFAULT_LANGUAGE_PAIRS.get(self.year, []):
                return lang_pair_rev

        return lang_pair

    def load_data(self) -> Tuple[List[str], List[str]]:
        """
        Load and preprocess WMT data from Hugging Face datasets.

        Returns:
            Tuple containing lists of source and target sentences
        """
        try:
            from datasets import load_dataset, get_dataset_config_names

            # Determine dataset name and config
            dataset_name = f"wmt{self.year}"
            lang_pair = self._get_config_name()

            print(f"Loading {dataset_name} dataset with language pair {lang_pair}")

            # Get available configs for this dataset if we need to validate
            try:
                available_configs = get_dataset_config_names(dataset_name)
                if lang_pair not in available_configs:
                    available_pairs = [c for c in available_configs if "-" in c]
                    raise ValueError(
                        f"Language pair {lang_pair} not available for {dataset_name}. "
                        f"Available language pairs: {', '.join(available_pairs)}"
                    )
            except Exception as e:
                print(f"Warning: Could not verify available configs: {e}")

            # Load the dataset
            dataset = load_dataset(
                dataset_name,
                lang_pair,
                split=self.split,
                cache_dir=os.path.join(self.data_dir, "cache"),
                trust_remote_code=True,
            )

            # Subset filtering note - added for awareness
            if self.subset:
                print(
                    f"Note: Subset filtering for '{self.subset}' needs to be done manually if needed"
                )

            # Extract source and target texts
            src_data = []
            tgt_data = []

            # Check if we need to swap languages based on language pair ordering
            swap_languages = False
            if f"{self.tgt_lang}-{self.src_lang}" == lang_pair:
                swap_languages = True
                print(f"Note: Source and target are swapped in the dataset")

            # Convert to list for easier processing
            print("Processing dataset examples...")
            examples = list(dataset)

            if not examples:
                raise ValueError(f"No examples found in the dataset")

            # Check structure of examples to determine how to extract translations
            sample = examples[0]

            # Manual subset filtering if needed
            if self.subset:
                filtered_examples = []
                for example in examples:
                    if "subset" in example and example["subset"] == self.subset:
                        filtered_examples.append(example)

                if filtered_examples:
                    examples = filtered_examples
                    print(
                        f"Filtered to {len(examples)} examples in subset: {self.subset}"
                    )
                else:
                    print(
                        f"Warning: No examples found in subset '{self.subset}', using all data"
                    )

            # Determine how to extract the translations based on dataset structure
            if "translation" in sample:
                # WMT14 and above format with "translation" field containing language pairs
                for example in tqdm(examples, desc="Extracting translations"):
                    translation = example["translation"]
                    if swap_languages:
                        src_data.append(translation[self.tgt_lang])
                        tgt_data.append(translation[self.src_lang])
                    else:
                        src_data.append(translation[self.src_lang])
                        tgt_data.append(translation[self.tgt_lang])
            else:
                # Handle other formats or print available keys
                print(f"Dataset structure: {sample.keys()}")
                raise ValueError(
                    f"Unsupported dataset structure. Expected 'translation' field, "
                    f"but found: {list(sample.keys())}"
                )

            # Clean up - remove empty lines
            cleaned_data = [
                (s, t) for s, t in zip(src_data, tgt_data) if s.strip() and t.strip()
            ]

            if cleaned_data:
                src_data, tgt_data = zip(*cleaned_data)
                src_data, tgt_data = list(src_data), list(tgt_data)
            else:
                src_data, tgt_data = [], []

            # Limit dataset size if specified
            if self.max_examples is not None and len(src_data) > self.max_examples:
                # Shuffle with a fixed random seed for reproducibility
                combined = list(zip(src_data, tgt_data))
                random.shuffle(combined)
                combined = combined[: self.max_examples]
                src_data, tgt_data = zip(*combined)
                src_data, tgt_data = list(src_data), list(tgt_data)

            # Cache the dataset to files for faster loading next time
            self._save_to_cache(src_data, tgt_data)

            return src_data, tgt_data

        except ImportError:
            print("Error: The 'datasets' library is required to use WMTDataset")
            print("Please install it using: pip install datasets")
            raise

        except Exception as e:
            print(f"Error loading WMT dataset: {e}")

            # Try to load from cache if available
            cached_data = self._load_from_cache()
            if cached_data:
                print(f"Loaded data from cache instead")
                return cached_data

            raise RuntimeError(
                f"Failed to load WMT{self.year} data for {self.src_lang}-{self.tgt_lang}. "
                f"Error: {str(e)}"
            )

    def _get_cache_path(self) -> Tuple[str, str]:
        """Get paths to cache files for source and target data."""
        cache_dir = os.path.join(self.data_dir, "processed")
        os.makedirs(cache_dir, exist_ok=True)

        subset_str = f".{self.subset}" if self.subset else ""
        src_file = os.path.join(
            cache_dir,
            f"wmt{self.year}.{self.src_lang}-{self.tgt_lang}.{self.split}{subset_str}.{self.src_lang}",
        )
        tgt_file = os.path.join(
            cache_dir,
            f"wmt{self.year}.{self.src_lang}-{self.tgt_lang}.{self.split}{subset_str}.{self.tgt_lang}",
        )

        return src_file, tgt_file

    def _save_to_cache(self, src_data: List[str], tgt_data: List[str]) -> None:
        """Save dataset to cache files."""
        src_file, tgt_file = self._get_cache_path()

        try:
            with open(src_file, "w", encoding="utf-8") as f:
                f.write("\n".join(src_data))

            with open(tgt_file, "w", encoding="utf-8") as f:
                f.write("\n".join(tgt_data))

            print(f"Saved {len(src_data)} examples to cache files")
        except Exception as e:
            print(f"Warning: Failed to save cache files: {e}")

    def _load_from_cache(self) -> Optional[Tuple[List[str], List[str]]]:
        """Try to load dataset from cache files."""
        src_file, tgt_file = self._get_cache_path()

        if os.path.exists(src_file) and os.path.exists(tgt_file):
            try:
                with open(src_file, "r", encoding="utf-8") as f:
                    src_data = f.read().strip().split("\n")

                with open(tgt_file, "r", encoding="utf-8") as f:
                    tgt_data = f.read().strip().split("\n")

                if len(src_data) > 0 and len(tgt_data) > 0:
                    # Apply max_examples limit if needed
                    if self.max_examples and len(src_data) > self.max_examples:
                        combined = list(zip(src_data, tgt_data))
                        random.shuffle(combined)
                        combined = combined[: self.max_examples]
                        src_data, tgt_data = zip(*combined)
                        src_data, tgt_data = list(src_data), list(tgt_data)

                    return src_data, tgt_data
            except Exception as e:
                print(f"Error loading from cache: {e}")

        return None


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
        "module_purpose": "Provides a dataset class for loading and preprocessing WMT dataset for machine translation",
        "key_classes": [
            {
                "name": "WMTDataset",
                "purpose": "Handles loading and preprocessing parallel text data from the WMT dataset",
                "key_methods": [
                    {
                        "name": "__init__",
                        "signature": "__init__(self, src_lang='de', tgt_lang='en', year='14', split='train', max_examples=None, data_dir='data/wmt', random_seed=42, subset=None)",
                        "brief_description": "Initialize the dataset with source/target languages and processing options",
                    },
                    {
                        "name": "load_data",
                        "signature": "load_data(self) -> Tuple[List[str], List[str]]",
                        "brief_description": "Load and preprocess parallel corpora from WMT dataset",
                    },
                ],
                "inheritance": "object",
                "dependencies": ["os", "random", "tqdm"],
            }
        ],
        "external_dependencies": ["datasets"],
        "complexity_score": 4,
    }
