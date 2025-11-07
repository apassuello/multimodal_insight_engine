# src/data/curriculum_dataset.py
"""
Curriculum Learning Implementation for Translation Datasets

PURPOSE:
    Implements curriculum learning strategies for machine translation, allowing models to learn
    from simpler examples before moving to more complex ones. This progressive learning approach
    can lead to better and faster convergence.

KEY COMPONENTS:
    - Difficulty calculation based on different strategies (length, vocabulary, similarity)
    - Stage-based progression through the curriculum
    - Tools for monitoring curriculum progression and statistics
"""

import os
from collections import Counter
from typing import Any, Dict, List

import numpy as np
import torch
from torch.utils.data import Dataset


class CurriculumTranslationDataset(Dataset):
    """
    Dataset for curriculum learning in translation tasks.

    This dataset implements various curriculum learning strategies for machine translation:
    - "length": Sort by sentence length (shorter sentences first)
    - "vocab": Sort by vocabulary complexity (common words first)
    - "similarity": Sort by source-target length ratio (similar lengths first)

    The curriculum progresses in stages, exposing more complex examples as training advances.
    """

    def __init__(
        self,
        source_sequences: List[List[int]],
        target_sequences: List[List[int]],
        curriculum_strategy: str = "length",
        num_stages: int = 5,
        pad_idx: int = 0,
        max_src_len: int = 100,
        max_tgt_len: int = 100,
    ):
        """
        Initialize the curriculum dataset.

        Args:
            source_sequences: List of source token sequences (already encoded)
            target_sequences: List of target token sequences (already encoded)
            curriculum_strategy: Strategy for curriculum (length, vocab, similarity)
            num_stages: Number of curriculum stages
            pad_idx: Padding token index
            max_src_len: Maximum source sequence length
            max_tgt_len: Maximum target sequence length
        """
        print("\n=== Initializing Curriculum Learning Dataset ===")
        print(f"Total examples: {len(source_sequences)}")
        print(f"Strategy: {curriculum_strategy}")
        print(f"Number of stages: {num_stages}")

        self.source_sequences = source_sequences
        self.target_sequences = target_sequences
        self.curriculum_strategy = curriculum_strategy
        self.num_stages = num_stages
        self.pad_idx = pad_idx
        self.max_src_len = max_src_len
        self.max_tgt_len = max_tgt_len

        # Start at the first curriculum stage
        self.current_stage = 0

        print("\nCalculating difficulty scores...")
        # Calculate difficulty scores for each example
        self.difficulties = self._calculate_difficulties()

        # Sort examples by difficulty
        print("Sorting examples by difficulty...")
        self.sorted_indices = np.argsort(self.difficulties).tolist()

        print("\nCurriculum initialization complete!")
        print(f"Initialized curriculum learning with strategy '{curriculum_strategy}'")
        print(f"Curriculum will progress through {num_stages} stages")

        # Calculate statistics for reporting
        difficulties = np.array(self.difficulties)
        print(f"\nDifficulty statistics:")
        print(f"  Min: {difficulties.min():.2f}")
        print(f"  Max: {difficulties.max():.2f}")
        print(f"  Mean: {difficulties.mean():.2f}")
        print(f"  Median: {np.median(difficulties):.2f}")

        # Print initial curriculum samples
        print("\nInitial curriculum setup:")
        self.print_curriculum_samples(num_samples=5)
        print("=== Curriculum Learning Dataset Initialization Complete ===\n")

    def _calculate_difficulties(self) -> List[float]:
        """Calculate difficulty scores for all examples based on selected strategy."""
        difficulties = []

        if self.curriculum_strategy == "length":
            # Length-based curriculum: shorter sentences are easier
            for src, tgt in zip(self.source_sequences, self.target_sequences):
                # Use maximum length as difficulty
                difficulty = max(len(src), len(tgt))
                difficulties.append(difficulty)

        elif self.curriculum_strategy == "vocab":
            # Vocabulary-based curriculum: common words are easier
            # First, calculate token frequencies across the dataset
            counter = Counter()
            for seq in self.source_sequences + self.target_sequences:
                counter.update(seq)

            # Calculate inverse frequency (rarity) for each token
            total_tokens = sum(counter.values())
            token_rarity = {
                token: total_tokens / count for token, count in counter.items()
            }

            # Calculate average token rarity for each example
            for src, tgt in zip(self.source_sequences, self.target_sequences):
                # Average rarity across both source and target
                src_rarity = sum(token_rarity.get(t, 1.0) for t in src) / max(
                    1, len(src)
                )
                tgt_rarity = sum(token_rarity.get(t, 1.0) for t in tgt) / max(
                    1, len(tgt)
                )
                difficulty = (src_rarity + tgt_rarity) / 2
                difficulties.append(difficulty)

        elif self.curriculum_strategy == "similarity":
            # Similarity-based curriculum: sentences with similar lengths are easier
            for src, tgt in zip(self.source_sequences, self.target_sequences):
                # Ratio of longer to shorter (always ≥ 1.0)
                longer = max(len(src), len(tgt))
                shorter = min(len(src), len(tgt))
                # Avoid division by zero
                if shorter == 0:
                    difficulty = 10.0  # Arbitrary high value
                else:
                    difficulty = longer / shorter
                difficulties.append(difficulty)

        else:
            # Default to length-based if strategy not recognized
            print(
                f"Warning: Curriculum strategy '{self.curriculum_strategy}' not recognized."
            )
            print("Defaulting to length-based curriculum.")
            return self._calculate_difficulties_by_length()

        return difficulties

    def _calculate_difficulties_by_length(self) -> List[float]:
        """Calculate difficulty scores based on sequence length."""
        difficulties = []
        for src, tgt in zip(self.source_sequences, self.target_sequences):
            difficulty = max(len(src), len(tgt))
            difficulties.append(difficulty)
        return difficulties

    def update_stage(self, new_stage: int) -> None:
        """
        Update the curriculum stage.

        Args:
            new_stage: New curriculum stage (0 to num_stages-1)
        """
        if new_stage < 0 or new_stage >= self.num_stages:
            print(
                f"Warning: Invalid curriculum stage {new_stage}. Must be between 0 and {self.num_stages-1}."
            )
            return

        prev_stage = self.current_stage
        self.current_stage = new_stage

        # Calculate percent of data available at current stage
        percent = 100 * self.get_stage_percent()

        if prev_stage != new_stage:
            print(
                f"Curriculum advanced to stage {new_stage}/{self.num_stages-1} ({percent:.1f}% of data)"
            )
            # Print sample examples from the current stage
            self.print_curriculum_samples()

    def print_curriculum_samples(self, num_samples: int = 3) -> None:
        """
        Print sample examples from the current curriculum stage.

        Args:
            num_samples: Number of samples to print
        """
        available_length = self.__len__()
        total_length = len(self.source_sequences)
        available_indices = self.sorted_indices[:available_length]

        # Get difficulty scores for available examples
        available_difficulties = [self.difficulties[i] for i in available_indices]

        # Print statistics about current stage
        print(f"\nCurriculum Stage {self.current_stage} Samples:")
        print(f"Strategy: {self.curriculum_strategy}")
        print(
            f"Available examples: {available_length}/{total_length} ({100 * available_length/total_length:.1f}%)"
        )
        print(
            f"Difficulty range: {min(available_difficulties):.2f} - {max(available_difficulties):.2f}"
        )
        print(
            f"Average difficulty: {sum(available_difficulties)/len(available_difficulties):.2f}"
        )

        # Print sample examples
        print("\nSample examples from current stage:")
        for i in range(min(num_samples, available_length)):
            idx = available_indices[i]
            src = self.source_sequences[idx]
            tgt = self.target_sequences[idx]
            difficulty = self.difficulties[idx]

            print(f"\nExample {i+1} (Difficulty: {difficulty:.2f}):")
            print(f"Source length: {len(src)}, Target length: {len(tgt)}")
            print(f"Source tokens: {src}")
            print(f"Target tokens: {tgt}")

    def get_stage_percent(self) -> float:
        """
        Get the percentage of data available at the current stage.

        Returns:
            Float between 0 and 1 representing portion of data available
        """
        # Progressively include more data as stages advance
        # Stage 0: 20% of data, Stage 1: 40% of data, etc.
        return (self.current_stage + 1) / self.num_stages

    def __len__(self) -> int:
        """Get the length of the dataset at the current curriculum stage."""
        total_length = len(self.source_sequences)
        available_percent = self.get_stage_percent()
        # Ensure at least 100 examples are available even at the first stage
        return max(100, int(total_length * available_percent))

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get an item from the dataset based on curriculum stage.

        Args:
            idx: Index into the curriculum-filtered dataset

        Returns:
            Dictionary with source and target tensors
        """
        # Map the input index to the sorted curriculum index
        available_length = self.__len__()
        if idx >= available_length:
            idx = idx % available_length

        # Get the actual index based on difficulty
        actual_idx = self.sorted_indices[idx]

        # Get source and target sequences
        src = self.source_sequences[actual_idx]
        tgt = self.target_sequences[actual_idx]

        # Truncate sequences if needed
        src = src[: self.max_src_len]
        tgt = tgt[: self.max_tgt_len]

        return {
            "src_tokens": torch.tensor(src, dtype=torch.long),
            "tgt_tokens": torch.tensor(tgt, dtype=torch.long),
        }

    def get_curriculum_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the current curriculum stage.

        Returns:
            Dictionary of curriculum statistics
        """
        available_length = self.__len__()
        total_length = len(self.source_sequences)
        available_indices = self.sorted_indices[:available_length]

        available_difficulties = [self.difficulties[i] for i in available_indices]
        if not available_difficulties:
            return {"error": "No examples available at current stage"}

        return {
            "stage": self.current_stage,
            "num_stages": self.num_stages,
            "examples_available": available_length,
            "total_examples": total_length,
            "percent_available": 100 * available_length / total_length,
            "min_difficulty": min(available_difficulties),
            "max_difficulty": max(available_difficulties),
            "mean_difficulty": sum(available_difficulties)
            / len(available_difficulties),
        }

    def print_curriculum_progression_summary(self) -> None:
        """
        Print a summary of the curriculum progression.
        """
        print("\nCurriculum Learning Progression Summary:")
        print(f"Strategy: {self.curriculum_strategy}")
        print(f"Total stages: {self.num_stages}")
        print(f"Final stage reached: {self.current_stage}/{self.num_stages-1}")

        # Calculate statistics for each stage
        total_examples = len(self.source_sequences)
        for stage in range(self.num_stages + 1):
            # Calculate how many examples would be available at this stage
            available_percent = (stage + 1) / self.num_stages
            available_count = max(100, int(total_examples * available_percent))
            available_indices = self.sorted_indices[:available_count]

            # Get difficulty scores for this stage
            stage_difficulties = [self.difficulties[i] for i in available_indices]

            print(f"\nStage {stage}/{self.num_stages-1}:")
            print(
                f"  Examples available: {available_count}/{total_examples} ({100 * available_count/total_examples:.1f}%)"
            )
            print(
                f"  Difficulty range: {min(stage_difficulties):.2f} - {max(stage_difficulties):.2f}"
            )
            print(
                f"  Average difficulty: {sum(stage_difficulties)/len(stage_difficulties):.2f}"
            )


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
        "module_purpose": "Implements curriculum learning for translation datasets, gradually increasing difficulty during training",
        "key_classes": [
            {
                "name": "CurriculumTranslationDataset",
                "purpose": "Dataset that implements curriculum learning strategies for translation tasks",
                "key_methods": [
                    {
                        "name": "_calculate_difficulties",
                        "signature": "_calculate_difficulties(self) -> List[float]",
                        "brief_description": "Calculate difficulty scores for all examples based on selected strategy"
                    },
                    {
                        "name": "update_stage",
                        "signature": "update_stage(self, new_stage: int) -> None",
                        "brief_description": "Update the curriculum stage to expose more complex examples"
                    },
                    {
                        "name": "__getitem__",
                        "signature": "__getitem__(self, idx: int) -> Dict[str, torch.Tensor]",
                        "brief_description": "Get an item from the dataset based on curriculum stage"
                    },
                    {
                        "name": "get_curriculum_stats",
                        "signature": "get_curriculum_stats(self) -> Dict[str, Any]",
                        "brief_description": "Get statistics about the current curriculum stage"
                    }
                ],
                "inheritance": "Dataset",
                "dependencies": ["torch.utils.data.Dataset", "numpy", "collections.Counter"]
            }
        ],
        "key_functions": [],
        "external_dependencies": ["torch", "numpy", "collections"],
        "complexity_score": 7  # Relatively complex due to multiple curriculum strategies and dynamic filtering
    }
