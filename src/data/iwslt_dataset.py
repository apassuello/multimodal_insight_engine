"""MODULE: iwslt_dataset.py
PURPOSE: Provides the IWSLTDataset class for loading and processing IWSLT translation datasets
with automatic download capabilities and fallback to synthetic data generation.

KEY COMPONENTS:
- IWSLTDataset: Dataset loader for IWSLT translation data with robust fallback mechanisms

DEPENDENCIES:
- os, sys
- requests
- random
- tqdm
"""

import os
import sys
import requests
import random
from tqdm import tqdm
import tarfile
import io
from typing import List, Tuple, Optional


class IWSLTDataset:
    """Dataset class for the IWSLT translation dataset with enhanced fallback to synthetic data."""

    # Available years to try loading data from
    AVAILABLE_YEARS = ["2017", "2016", "2015", "2014", "2013", "2012"]

    def __init__(
        self,
        src_lang="de",
        tgt_lang="en",
        year="2017",
        split="train",
        max_examples=None,
        data_dir="data/iwslt",
        random_seed=42,
        combine_years=True,
    ):
        """
        Initialize the IWSLT dataset.

        Args:
            src_lang: Source language code
            tgt_lang: Target language code
            year: Dataset year to start with
            split: Data split (train, valid, test)
            max_examples: Maximum number of examples to use (None = use all)
            data_dir: Directory containing the IWSLT data
            random_seed: Random seed for reproducibility
            combine_years: Whether to combine data from multiple years to reach max_examples
        """
        self.src_lang = src_lang
        self.tgt_lang = tgt_lang
        self.year = year
        self.split = split
        self.max_examples = max_examples
        self.data_dir = data_dir
        self.combine_years = combine_years

        # Set random seed for reproducibility
        random.seed(random_seed)

        # Paths
        os.makedirs(self.data_dir, exist_ok=True)

        # Load data
        self.src_data, self.tgt_data = self.load_data()

        # Print dataset info
        print(f"Loaded {len(self.src_data)} {split} examples")

    def download_data(self, year=None):
        """
        Download IWSLT dataset for a specific year if not already present.

        Args:
            year: Specific year to download (defaults to self.year)

        Returns:
            Tuple of source and target file paths, or None if download failed
        """
        # Use the instance year if not specified
        year = year or self.year

        # Define data paths for this year
        src_file = f"{self.data_dir}/iwslt{year}.{self.src_lang}-{self.tgt_lang}.{self.split}.{self.src_lang}"
        tgt_file = f"{self.data_dir}/iwslt{year}.{self.src_lang}-{self.tgt_lang}.{self.split}.{self.tgt_lang}"

        # Check if data exists
        files_exist = os.path.exists(src_file) and os.path.exists(tgt_file)
        if files_exist:
            # Verify the files are not empty and contain adequate data
            try:
                with open(src_file, "r", encoding="utf-8") as f:
                    src_content = f.read().strip()
                with open(tgt_file, "r", encoding="utf-8") as f:
                    tgt_content = f.read().strip()

                # Check if we have data
                src_lines = src_content.count("\n") + 1
                tgt_lines = tgt_content.count("\n") + 1

                if src_lines > 0 and tgt_lines > 0:
                    print(
                        f"IWSLT {year} {self.src_lang}-{self.tgt_lang} {self.split} data already exists with {src_lines} examples"
                    )
                    return src_file, tgt_file
                else:
                    print(
                        f"IWSLT files for year {year} exist but are empty. Recreating..."
                    )
            except Exception as e:
                print(
                    f"Error reading existing files for year {year}: {e}. Recreating..."
                )

        # Attempt to download from official sources
        try:
            success = self._download_from_huggingface(year)
            if success:
                return src_file, tgt_file
        except Exception as e:
            print(f"Error downloading from HuggingFace for year {year}: {e}")
            try:
                success = self._download_from_official_source(year)
                if success:
                    return src_file, tgt_file
            except Exception as e2:
                print(f"Error downloading from official source for year {year}: {e2}")
                # No synthetic data fallback - return None to indicate failure
                print(
                    f"Failed to download IWSLT data for year {year}. Please fix the download issue manually."
                )
                return None

        # Return paths if files exist, otherwise None
        if os.path.exists(src_file) and os.path.exists(tgt_file):
            return src_file, tgt_file
        else:
            return None

    def _download_from_huggingface(self, year=None):
        """Download dataset from HuggingFace datasets."""
        requested_year = year or self.year
        actual_year = requested_year  # Track which year's dataset was actually loaded
        print(
            f"Downloading IWSLT {requested_year} {self.src_lang}-{self.tgt_lang} {self.split} data from HuggingFace..."
        )

        try:
            from datasets import load_dataset, get_dataset_config_names

            # First try the TED talks dataset for years 2014-2016
            if requested_year in ["2014", "2015", "2016"]:
                try:
                    print(
                        f"Attempting to load from IWSLT/ted_talks_iwslt dataset for year {requested_year}..."
                    )
                    dataset = load_dataset(
                        "IWSLT/ted_talks_iwslt",
                        language_pair=(self.src_lang, self.tgt_lang),
                        year=requested_year,
                        split=self.split,
                    )

                    # Extract source and target texts
                    src_texts = []
                    tgt_texts = []

                    # Convert to list for easier processing
                    examples = list(dataset)

                    for example in examples:
                        if isinstance(example, dict) and "translation" in example:
                            translation = example["translation"]
                            if isinstance(translation, dict):
                                if (
                                    self.src_lang in translation
                                    and self.tgt_lang in translation
                                ):
                                    src_texts.append(translation[self.src_lang])
                                    tgt_texts.append(translation[self.tgt_lang])

                    if src_texts and tgt_texts:
                        src_file = f"{self.data_dir}/iwslt{requested_year}.{self.src_lang}-{self.tgt_lang}.{self.split}.{self.src_lang}"
                        tgt_file = f"{self.data_dir}/iwslt{requested_year}.{self.src_lang}-{self.tgt_lang}.{self.split}.{self.tgt_lang}"

                        with open(src_file, "w", encoding="utf-8") as f:
                            f.write("\n".join(src_texts))

                        with open(tgt_file, "w", encoding="utf-8") as f:
                            f.write("\n".join(tgt_texts))

                        print(
                            f"Downloaded {len(src_texts)} examples from TED talks dataset for year {requested_year}"
                        )
                        return True
                    else:
                        print(
                            f"No valid translation pairs found in the TED talks dataset for year {requested_year}"
                        )
                except Exception as e:
                    print(
                        f"Error downloading from TED talks dataset for year {requested_year}: {e}"
                    )
                    # Continue to try the old approach

            # Continue with the original approach for 2017 or if TED talks dataset failed
            # Load the IWSLT dataset - correct the configuration format
            dataset_config = f"iwslt{requested_year}-{self.src_lang}-{self.tgt_lang}"

            # Try with the standard format first
            try:
                dataset = load_dataset(
                    f"iwslt{requested_year}", dataset_config, split=self.split
                )
            except (ValueError, FileNotFoundError, ImportError) as e:
                # Try with the fixed name format "iwslt2017"
                try:
                    dataset = load_dataset(
                        "iwslt2017", dataset_config, split=self.split
                    )
                    actual_year = "2017"  # Actually using 2017 dataset
                except (ValueError, FileNotFoundError) as e:
                    # If that fails, try the reverse configuration
                    dataset_config = (
                        f"iwslt{requested_year}-{self.tgt_lang}-{self.src_lang}"
                    )
                    try:
                        dataset = load_dataset(
                            f"iwslt{requested_year}", dataset_config, split=self.split
                        )
                        # If this works, we need to swap src and tgt in our extraction
                        swap_languages = True
                    except (ValueError, FileNotFoundError) as e:
                        try:
                            dataset = load_dataset(
                                "iwslt2017", dataset_config, split=self.split
                            )
                            swap_languages = True
                            actual_year = "2017"  # Actually using 2017 dataset
                        except (ValueError, FileNotFoundError) as e:
                            # Try all available configurations and pick the one that matches our languages
                            try:
                                available_configs = get_dataset_config_names(
                                    f"iwslt{requested_year}"
                                )
                            except (ValueError, FileNotFoundError):
                                try:
                                    available_configs = get_dataset_config_names(
                                        "iwslt2017"
                                    )
                                except (ValueError, FileNotFoundError, OSError, ConnectionError, RuntimeError) as e:
                                    available_configs = []

                            matching_configs = [
                                c
                                for c in available_configs
                                if (
                                    f"{self.src_lang}-{self.tgt_lang}" in c
                                    or f"{self.tgt_lang}-{self.src_lang}" in c
                                )
                            ]

                            if matching_configs:
                                dataset_config = matching_configs[0]
                                swap_languages = (
                                    f"{self.tgt_lang}-{self.src_lang}" in dataset_config
                                )
                                try:
                                    dataset = load_dataset(
                                        f"iwslt{requested_year}",
                                        dataset_config,
                                        split=self.split,
                                    )
                                except (ValueError, FileNotFoundError, OSError, ConnectionError, RuntimeError) as e:
                                    dataset = load_dataset(
                                        "iwslt2017", dataset_config, split=self.split
                                    )
                                    actual_year = "2017"  # Actually using 2017 dataset
                            else:
                                # Try one more fallback using iwslt dataset
                                try:
                                    available_configs = get_dataset_config_names(
                                        "iwslt"
                                    )
                                    matching_configs = [
                                        c
                                        for c in available_configs
                                        if (
                                            f"{self.src_lang}" in c
                                            and f"{self.tgt_lang}" in c
                                        )
                                    ]

                                    if matching_configs:
                                        dataset_config = matching_configs[0]
                                        swap_languages = (
                                            f"{self.tgt_lang}" in dataset_config
                                            and f"{self.tgt_lang}"
                                            in dataset_config.split("-")[0]
                                        )
                                        dataset = load_dataset(
                                            "iwslt", dataset_config, split=self.split
                                        )
                                        actual_year = (
                                            "iwslt"  # Using generic IWSLT dataset
                                        )
                                    else:
                                        return False
                                except (ValueError, FileNotFoundError, OSError, ConnectionError, RuntimeError) as e:
                                    return False
            else:
                swap_languages = False

            # Extract source and target texts more safely
            src_texts = []
            tgt_texts = []

            # Convert to list of dictionaries for safer access
            examples = list(dataset)

            for example in examples:
                if isinstance(example, dict) and "translation" in example:
                    translation = example["translation"]
                    if isinstance(translation, dict):
                        if (
                            self.src_lang in translation
                            and self.tgt_lang in translation
                        ):
                            if swap_languages:
                                # Swap the source and target
                                src_texts.append(translation[self.tgt_lang])
                                tgt_texts.append(translation[self.src_lang])
                            else:
                                src_texts.append(translation[self.src_lang])
                                tgt_texts.append(translation[self.tgt_lang])

            # Save to files - use the actual year in the filename
            if src_texts and tgt_texts:
                src_file = f"{self.data_dir}/iwslt{actual_year}.{self.src_lang}-{self.tgt_lang}.{self.split}.{self.src_lang}"
                tgt_file = f"{self.data_dir}/iwslt{actual_year}.{self.src_lang}-{self.tgt_lang}.{self.split}.{self.tgt_lang}"

                with open(src_file, "w", encoding="utf-8") as f:
                    f.write("\n".join(src_texts))

                with open(tgt_file, "w", encoding="utf-8") as f:
                    f.write("\n".join(tgt_texts))

                # If this isn't the originally requested year data, print a warning
                if actual_year != requested_year:
                    print(
                        f"Warning: IWSLT {requested_year} data not found. Using IWSLT {actual_year} data instead."
                    )

                print(
                    f"Downloaded {len(src_texts)} examples from HuggingFace for year {actual_year}"
                )

                # If we loaded a different year's data than requested, return None
                # This forces the code to continue looking for other years rather than
                # reusing the same data with different filenames
                if actual_year != requested_year:
                    return None

                return True
            else:
                print(
                    f"No valid translation pairs found in the dataset for year {requested_year}"
                )
                return False

        except Exception as e:
            print(f"Error downloading from HuggingFace for year {requested_year}: {e}")
            return False

    def _download_from_official_source(self, year=None):
        """Attempt to download from the official IWSLT website."""
        year = year or self.year
        print(
            f"Downloading IWSLT {year} {self.src_lang}-{self.tgt_lang} {self.split} data from official source..."
        )

        import requests
        import tarfile
        import io

        # This is a simplified example - the actual URL structure would need to be adjusted
        # based on the specific IWSLT release
        base_url = (
            f"https://wit3.fbk.eu/archive/{year}/texts/{self.src_lang}/{self.tgt_lang}"
        )
        tarball_url = f"{base_url}/{self.src_lang}-{self.tgt_lang}.tgz"

        try:
            # Download tarball
            response = requests.get(tarball_url)
            response.raise_for_status()

            # Extract from tarball
            with tarfile.open(fileobj=io.BytesIO(response.content)) as tar:
                # Extract relevant files (this would need to be adjusted based on the archive structure)
                src_file_in_tar = f"{self.split}.{self.src_lang}"
                tgt_file_in_tar = f"{self.split}.{self.tgt_lang}"

                # Check if files exist in the tarball
                src_file_info = (
                    tar.getmember(src_file_in_tar)
                    if src_file_in_tar in [m.name for m in tar.getmembers()]
                    else None
                )
                tgt_file_info = (
                    tar.getmember(tgt_file_in_tar)
                    if tgt_file_in_tar in [m.name for m in tar.getmembers()]
                    else None
                )

                # Extract and save files if they exist
                if src_file_info and tgt_file_info:
                    # Extract and save files
                    src_file_obj = tar.extractfile(src_file_info)
                    tgt_file_obj = tar.extractfile(tgt_file_info)

                    if src_file_obj and tgt_file_obj:
                        src_file = f"{self.data_dir}/iwslt{year}.{self.src_lang}-{self.tgt_lang}.{self.split}.{self.src_lang}"
                        tgt_file = f"{self.data_dir}/iwslt{year}.{self.src_lang}-{self.tgt_lang}.{self.split}.{self.tgt_lang}"

                        with open(src_file, "wb") as f:
                            f.write(src_file_obj.read())

                        with open(tgt_file, "wb") as f:
                            f.write(tgt_file_obj.read())

                        return True
                    else:
                        raise FileNotFoundError(
                            f"Could not extract {src_file_in_tar} or {tgt_file_in_tar} from the tarball"
                        )
                else:
                    raise FileNotFoundError(
                        f"Could not find {src_file_in_tar} or {tgt_file_in_tar} in the tarball"
                    )

            print(f"Downloaded from official source for year {year}")
            return True

        except Exception as e:
            print(f"Official source download failed for year {year}: {e}")
            return False

    def load_data(self) -> Tuple[List[str], List[str]]:
        """
        Load and preprocess parallel data, combining from multiple years if needed.

        Returns:
            Tuple containing lists of source and target sentences

        Raises:
            RuntimeError: When no valid data could be found from any year
        """
        all_src_data = []
        all_tgt_data = []
        years_attempted = []
        years_loaded = set()  # Track which years were actually loaded

        # First try loading from the specified year
        years_attempted.append(self.year)
        files = self.download_data()

        if files:
            src_file, tgt_file = files
            # Extract actual year from filename
            actual_year = src_file.split("/")[-1].split(".")[0].replace("iwslt", "")
            years_loaded.add(actual_year)

            # Read the files
            with open(src_file, "r", encoding="utf-8") as f:
                src_data = f.read().strip().split("\n")

            with open(tgt_file, "r", encoding="utf-8") as f:
                tgt_data = f.read().strip().split("\n")

            # Skip empty lines
            src_data = [line for line in src_data if line.strip()]
            tgt_data = [line for line in tgt_data if line.strip()]

            # Ensure same length
            min_len = min(len(src_data), len(tgt_data))
            if min_len < max(len(src_data), len(tgt_data)):
                print(
                    f"Warning: Source and target data have different lengths for year {self.year}. Truncating to {min_len} examples."
                )
                src_data = src_data[:min_len]
                tgt_data = tgt_data[:min_len]

            # Add to our collection
            all_src_data.extend(src_data)
            all_tgt_data.extend(tgt_data)

        # If we need more examples and combine_years is enabled, try other years
        if (
            self.combine_years
            and self.max_examples
            and len(all_src_data) < self.max_examples
        ):
            remaining = self.max_examples - len(all_src_data)
            print(
                f"Loaded {len(all_src_data)} examples from year {self.year}, need {remaining} more."
            )

            # Try other years in order
            for year in self.AVAILABLE_YEARS:
                # Skip the primary year we already loaded
                if year == self.year:
                    continue

                years_attempted.append(year)
                print(f"Attempting to load additional data from year {year}...")
                files = self.download_data(year)

                if not files:
                    print(f"No data available for year {year}, skipping.")
                    continue

                src_file, tgt_file = files
                # Extract actual year from filename
                actual_year = src_file.split("/")[-1].split(".")[0].replace("iwslt", "")

                # Skip if we've already loaded this year's data (avoid duplicates)
                if actual_year in years_loaded:
                    print(
                        f"Already loaded data from year {actual_year}, skipping to avoid duplicates."
                    )
                    continue

                years_loaded.add(actual_year)

                # Read the files
                with open(src_file, "r", encoding="utf-8") as f:
                    src_data = f.read().strip().split("\n")

                with open(tgt_file, "r", encoding="utf-8") as f:
                    tgt_data = f.read().strip().split("\n")

                # Skip empty lines
                src_data = [line for line in src_data if line.strip()]
                tgt_data = [line for line in tgt_data if line.strip()]

                # Ensure same length
                min_len = min(len(src_data), len(tgt_data))
                if min_len < max(len(src_data), len(tgt_data)):
                    print(
                        f"Warning: Source and target data have different lengths for year {actual_year}. Truncating to {min_len} examples."
                    )
                    src_data = src_data[:min_len]
                    tgt_data = tgt_data[:min_len]

                # Add to our collection
                examples_to_add = min(remaining, len(src_data))
                all_src_data.extend(src_data[:examples_to_add])
                all_tgt_data.extend(tgt_data[:examples_to_add])

                print(f"Added {examples_to_add} examples from year {actual_year}")

                # Check if we have enough
                remaining = self.max_examples - len(all_src_data)
                if remaining <= 0:
                    break

        # Check if we have any data
        if not all_src_data or not all_tgt_data:
            years_str = ", ".join(years_attempted)
            raise RuntimeError(
                f"No valid IWSLT data could be loaded for years: {years_str}. "
                f"Please download the data manually or check your network connection."
            )

        if years_loaded:
            print(
                f"Successfully loaded data from years: {', '.join(sorted(years_loaded))}"
            )
            if len(years_loaded) < len(years_attempted):
                missing_years = set([str(y) for y in years_attempted]) - years_loaded
                print(
                    f"Warning: Could not load data for years: {', '.join(sorted(missing_years))}"
                )
                print(
                    "Note: Only IWSLT 2017 may be available through HuggingFace datasets."
                )

        # Limit dataset size if specified
        if self.max_examples is not None and len(all_src_data) > self.max_examples:
            # Shuffle with a fixed random seed for reproducibility
            combined = list(zip(all_src_data, all_tgt_data))
            random.shuffle(combined)
            all_src_data, all_tgt_data = zip(*combined[: self.max_examples])
            all_src_data, all_tgt_data = list(all_src_data), list(all_tgt_data)

        return all_src_data, all_tgt_data


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
        "module_purpose": "Provides a dataset class for loading and preprocessing IWSLT dataset for machine translation",
        "key_classes": [
            {
                "name": "IWSLTDataset",
                "purpose": "Handles loading and preprocessing parallel text data from the IWSLT dataset",
                "key_methods": [
                    {
                        "name": "__init__",
                        "signature": "__init__(self, src_lang='de', tgt_lang='en', year='2017', split='train', max_examples=None, data_dir='data/iwslt', random_seed=42, combine_years=True)",
                        "brief_description": "Initialize the dataset with source/target languages and processing options",
                    },
                    {
                        "name": "download_data",
                        "signature": "download_data(self, year=None)",
                        "brief_description": "Download and prepare the IWSLT dataset for a specific year with fallback to synthetic data generation",
                    },
                    {
                        "name": "load_data",
                        "signature": "load_data(self) -> Tuple[List[str], List[str]]",
                        "brief_description": "Load and preprocess parallel corpora, combining data from multiple years if needed",
                    },
                ],
                "inheritance": "object",
                "dependencies": ["os", "random", "requests", "tqdm", "tarfile", "io"],
            }
        ],
        "external_dependencies": ["datasets"],
        "complexity_score": 5,  # Higher complexity due to multiple download methods and synthetic data generation
    }
