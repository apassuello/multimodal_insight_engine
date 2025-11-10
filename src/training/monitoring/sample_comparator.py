"""MODULE: sample_comparator.py
PURPOSE: Track response quality changes during training
KEY COMPONENTS:
- SampleComparison: Dataclass for comparison results
- SampleComparator: Callback that tracks before/after sample changes
DEPENDENCIES: difflib, dataclasses, typing, events, callbacks
SPECIAL NOTES: Detects catastrophic forgetting via similarity tracking
"""

from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional
import difflib
from .events import TrainingEvent, TrainingPhase


@dataclass(frozen=True, slots=True)
class SampleComparison:
    """
    Result of comparing original and updated responses.

    Design: Frozen dataclass for immutability and memory efficiency.

    Attributes:
        iteration: Training iteration when comparison was made
        prompt: The prompt that generated the responses
        original: Original response from baseline
        updated: Updated response from current model
        similarity_ratio: Similarity ratio (0.0-1.0) from difflib
        char_changes: Number of character-level changes
        line_changes: Number of line-level changes
    """
    iteration: int
    prompt: str
    original: str
    updated: str
    similarity_ratio: float
    char_changes: int
    line_changes: int

    def is_degraded(self, threshold: float = 0.7) -> bool:
        """
        Check if response quality degraded significantly.

        Args:
            threshold: Similarity threshold below which response is considered degraded

        Returns:
            True if similarity_ratio < threshold
        """
        return self.similarity_ratio < threshold


class SampleComparator:
    """
    Tracks response quality changes during training.

    Features:
    - Collects baseline responses from early iterations
    - Periodically re-generates responses for same prompts
    - Computes text diffs to detect quality changes
    - Detects catastrophic forgetting

    Design: Implements TrainingCallback protocol for event-driven updates.

    Attributes:
        sample_size: Number of baseline samples to collect
        comparison_frequency: How often to re-generate and compare (iterations)
        degradation_threshold: Similarity threshold for degradation detection
        baseline_samples: Stored baseline prompt-response pairs
        comparison_history: List of all comparisons made
        baseline_collected: Whether baseline collection is complete
    """
    __slots__ = (
        'sample_size',
        'comparison_frequency',
        'degradation_threshold',
        'baseline_samples',
        'comparison_history',
        'baseline_collected',
    )

    def __init__(
        self,
        sample_size: int = 10,
        comparison_frequency: int = 100,
        degradation_threshold: float = 0.7,
    ):
        """
        Initialize sample comparator.

        Args:
            sample_size: Number of baseline samples to collect (default: 10)
            comparison_frequency: Iterations between comparisons (default: 100)
            degradation_threshold: Similarity threshold for degradation (default: 0.7)
        """
        self.sample_size = sample_size
        self.comparison_frequency = comparison_frequency
        self.degradation_threshold = degradation_threshold
        self.baseline_samples: List[Dict[str, str]] = []
        self.comparison_history: List[SampleComparison] = []
        self.baseline_collected = False

    def on_event(self, event: TrainingEvent) -> None:
        """
        Handle training events to collect baselines and trigger comparisons.

        Collects baseline samples from early RESPONSE_GENERATED events.
        Triggers comparisons at ITERATION_END events based on frequency.

        Args:
            event: Training event with metrics and metadata
        """
        # Collect baseline samples from early iterations
        if (
            event.phase == TrainingPhase.RESPONSE_GENERATED
            and not self.baseline_collected
        ):
            self._collect_baseline(event)

        # Trigger comparison at specified frequency
        if (
            event.phase == TrainingPhase.ITERATION_END
            and self.baseline_collected
            and self.comparison_frequency > 0
            and event.iteration > 0
            and event.iteration % self.comparison_frequency == 0
        ):
            # Note: Actual re-generation requires model access
            # This is handled by integration with the trainer
            # Here we just mark that comparison should happen
            pass

    def _collect_baseline(self, event: TrainingEvent) -> None:
        """
        Collect baseline samples from RESPONSE_GENERATED events.

        Args:
            event: Training event with prompts and responses in metadata
        """
        if len(self.baseline_samples) >= self.sample_size:
            self.baseline_collected = True
            return

        # Extract prompts and responses from metadata
        prompts = event.metadata.get('prompts', [])
        responses = event.metadata.get('responses', [])

        if not prompts or not responses:
            return

        # Store first prompt-response pair from this batch
        if prompts and responses:
            self.baseline_samples.append({
                'prompt': prompts[0],
                'response': responses[0],
                'iteration': event.iteration,
            })

        # Mark as collected if we reached sample_size
        if len(self.baseline_samples) >= self.sample_size:
            self.baseline_collected = True

    def compute_diff(self, original: str, updated: str) -> Dict[str, Any]:
        """
        Compute text differences using difflib.

        Uses SequenceMatcher for similarity ratio and ndiff for change counting.

        Args:
            original: Original text from baseline
            updated: Updated text from current model

        Returns:
            Dictionary with:
            - similarity_ratio: Similarity score (0.0-1.0)
            - char_changes: Number of character-level changes
            - line_changes: Number of line-level changes
        """
        # Compute similarity ratio
        similarity = difflib.SequenceMatcher(None, original, updated).ratio()

        # Count character-level changes
        char_changes = self._count_char_changes(original, updated)

        # Count line-level changes
        line_changes = self._count_line_changes(original, updated)

        return {
            'similarity_ratio': similarity,
            'char_changes': char_changes,
            'line_changes': line_changes,
        }

    def _count_char_changes(self, original: str, updated: str) -> int:
        """
        Count character-level changes between two strings.

        Args:
            original: Original string
            updated: Updated string

        Returns:
            Number of character changes (insertions + deletions + replacements)
        """
        # Use SequenceMatcher to get edit operations
        matcher = difflib.SequenceMatcher(None, original, updated)
        changes = 0

        for tag, i1, i2, j1, j2 in matcher.get_opcodes():
            if tag != 'equal':
                # Count the maximum of characters affected in either string
                changes += max(i2 - i1, j2 - j1)

        return changes

    def _count_line_changes(self, original: str, updated: str) -> int:
        """
        Count line-level changes between two texts.

        Args:
            original: Original text
            updated: Updated text

        Returns:
            Number of lines that differ
        """
        original_lines = original.splitlines()
        updated_lines = updated.splitlines()

        # Use ndiff to find line changes
        diff = list(difflib.ndiff(original_lines, updated_lines))

        # Count lines with changes (starting with + or -)
        changes = sum(1 for line in diff if line.startswith('+ ') or line.startswith('- '))

        return changes

    def add_comparison(
        self,
        iteration: int,
        prompt: str,
        original: str,
        updated: str,
    ) -> SampleComparison:
        """
        Create and store a comparison between original and updated responses.

        This method is called by the trainer when re-generating responses
        for comparison.

        Args:
            iteration: Current training iteration
            prompt: The prompt used
            original: Original baseline response
            updated: Updated response from current model

        Returns:
            SampleComparison object with diff results
        """
        diff_result = self.compute_diff(original, updated)

        comparison = SampleComparison(
            iteration=iteration,
            prompt=prompt,
            original=original,
            updated=updated,
            similarity_ratio=diff_result['similarity_ratio'],
            char_changes=diff_result['char_changes'],
            line_changes=diff_result['line_changes'],
        )

        self.comparison_history.append(comparison)
        return comparison

    def detect_catastrophic_forgetting(self) -> bool:
        """
        Detect catastrophic forgetting by analyzing comparison history.

        Catastrophic forgetting is detected when a majority of recent
        comparisons show significant quality degradation.

        Returns:
            True if catastrophic forgetting is detected
        """
        if not self.comparison_history:
            return False

        # Check recent comparisons (last 10 or all if fewer)
        recent_comparisons = self.comparison_history[-10:]

        # Count how many are degraded
        degraded_count = sum(
            1 for comp in recent_comparisons
            if comp.is_degraded(self.degradation_threshold)
        )

        # Trigger if more than 50% are degraded
        degradation_ratio = degraded_count / len(recent_comparisons)
        return degradation_ratio > 0.5

    def get_comparison_summary(self) -> Dict[str, Any]:
        """
        Generate summary statistics from comparison history.

        Returns:
            Dictionary with:
            - num_comparisons: Total comparisons made
            - avg_similarity: Average similarity ratio
            - min_similarity: Minimum similarity seen
            - max_similarity: Maximum similarity seen
            - degraded_count: Number of degraded comparisons
            - catastrophic_forgetting: Whether forgetting detected
        """
        if not self.comparison_history:
            return {
                'num_comparisons': 0,
                'avg_similarity': 0.0,
                'min_similarity': 0.0,
                'max_similarity': 0.0,
                'degraded_count': 0,
                'catastrophic_forgetting': False,
            }

        similarities = [comp.similarity_ratio for comp in self.comparison_history]
        degraded_count = sum(
            1 for comp in self.comparison_history
            if comp.is_degraded(self.degradation_threshold)
        )

        return {
            'num_comparisons': len(self.comparison_history),
            'avg_similarity': sum(similarities) / len(similarities),
            'min_similarity': min(similarities),
            'max_similarity': max(similarities),
            'degraded_count': degraded_count,
            'catastrophic_forgetting': self.detect_catastrophic_forgetting(),
        }
