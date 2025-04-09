import os
from typing import List, Tuple, Optional
from .europarl_dataset import EuroparlDataset
from .opensubtitles_dataset import OpenSubtitlesDataset


class CombinedDataset:
    """
    A dataset class that combines examples from both Europarl and OpenSubtitles datasets.
    """

    def __init__(
        self,
        src_lang: str = "de",
        tgt_lang: str = "en",
        max_examples: Optional[int] = None,
    ):
        """
        Initialize the combined dataset.

        Args:
            src_lang: Source language code (default: "de")
            tgt_lang: Target language code (default: "en")
            max_examples: Maximum number of examples to use (default: None)
        """
        # Initialize both datasets
        self.europarl = EuroparlDataset(
            src_lang=src_lang,
            tgt_lang=tgt_lang,
            max_examples=max_examples // 2 if max_examples else None,
        )

        self.opensubtitles = OpenSubtitlesDataset(
            data_dir="data/os",
            src_lang=src_lang,
            tgt_lang=tgt_lang,
            max_examples=max_examples // 2 if max_examples else None,
        )

        # Combine source and target data
        self.src_data = self.europarl.src_data + self.opensubtitles.src_data
        self.tgt_data = self.europarl.tgt_data + self.opensubtitles.tgt_data

        # Shuffle the combined data
        import random

        combined = list(zip(self.src_data, self.tgt_data))
        random.shuffle(combined)
        self.src_data, self.tgt_data = zip(*combined)

        # Convert back to lists
        self.src_data = list(self.src_data)
        self.tgt_data = list(self.tgt_data)

        # Limit to max_examples if specified
        if max_examples and len(self.src_data) > max_examples:
            self.src_data = self.src_data[:max_examples]
            self.tgt_data = self.tgt_data[:max_examples]

        print(f"Loaded {len(self.src_data)} examples from combined dataset")
        print(f"  - Europarl: {len(self.europarl.src_data)} examples")
        print(f"  - OpenSubtitles: {len(self.opensubtitles.src_data)} examples")

    def __len__(self) -> int:
        """Return the number of examples in the dataset."""
        return len(self.src_data)

    def __getitem__(self, idx: int) -> Tuple[str, str]:
        """Get a source-target pair by index."""
        return self.src_data[idx], self.tgt_data[idx]
