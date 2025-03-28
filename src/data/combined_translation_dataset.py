from typing import List, Optional, Dict
from .europarl_dataset import EuroparlDataset
from .opensubtitles_dataset import OpenSubtitlesDataset

class CombinedTranslationDataset:
    """Combines multiple translation datasets."""
    
    def __init__(
        self,
        src_lang: str = "de",
        tgt_lang: str = "en",
        datasets: Dict[str, int] = None,  # Dict of dataset_name: max_examples
        seed: int = 42
    ):
        """
        Initialize the combined dataset.
        
        Args:
            src_lang: Source language code
            tgt_lang: Target language code
            datasets: Dictionary mapping dataset names to max examples
                     e.g., {"europarl": 100000, "opensubtitles": 100000}
            seed: Random seed for reproducibility
        """
        self.src_lang = src_lang
        self.tgt_lang = tgt_lang
        
        # Initialize empty lists
        self.src_data = []
        self.tgt_data = []
        
        # If no datasets specified, use default configuration
        if datasets is None:
            datasets = {
                "europarl": 3,  # Load 3 examples from Europarl
                "opensubtitles": 2  # Load 2 examples from OpenSubtitles
            }
            
        # Load all specified datasets
        for dataset_name, max_examples in datasets.items():
            if dataset_name == "europarl":
                dataset = EuroparlDataset(
                    src_lang=src_lang,
                    tgt_lang=tgt_lang,
                    max_examples=max_examples,
                )
            elif dataset_name == "opensubtitles":
                dataset = OpenSubtitlesDataset(
                    src_lang=src_lang,
                    tgt_lang=tgt_lang,
                    max_examples=max_examples,
                )
            else:
                raise ValueError(f"Unknown dataset: {dataset_name}")
            
            # Only add data if max_examples > 0
            if max_examples > 0:
                self.src_data.extend(dataset.src_data[:max_examples])
                self.tgt_data.extend(dataset.tgt_data[:max_examples])
        
        print(f"Combined dataset contains {len(self.src_data)} parallel sentences") 