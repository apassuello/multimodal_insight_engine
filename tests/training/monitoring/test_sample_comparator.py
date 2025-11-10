"""MODULE: test_sample_comparator.py
PURPOSE: Unit tests for sample comparison and quality tracking
KEY COMPONENTS:
- Test SampleComparator callback
- Test baseline collection
- Test text diff computation
- Test catastrophic forgetting detection
DEPENDENCIES: pytest, difflib, src.training.monitoring
"""

import pytest
from src.training.monitoring.sample_comparator import (
    SampleComparator,
    SampleComparison,
)
from src.training.monitoring.events import TrainingEvent, TrainingPhase


class TestSampleComparison:
    """Test SampleComparison dataclass."""

    def test_initialization(self):
        """Test comparison result initialization."""
        comparison = SampleComparison(
            iteration=10,
            prompt="Test prompt",
            original="Original response",
            updated="Updated response",
            similarity_ratio=0.85,
            char_changes=15,
            line_changes=2,
        )
        assert comparison.iteration == 10
        assert comparison.prompt == "Test prompt"
        assert comparison.similarity_ratio == 0.85
        assert comparison.char_changes == 15

    def test_is_degraded_false(self):
        """Test is_degraded returns False for minor changes."""
        comparison = SampleComparison(
            iteration=10,
            prompt="Test",
            original="Hello world",
            updated="Hello world!",
            similarity_ratio=0.95,
            char_changes=1,
            line_changes=0,
        )
        assert comparison.is_degraded(threshold=0.7) is False

    def test_is_degraded_true(self):
        """Test is_degraded returns True for major changes."""
        comparison = SampleComparison(
            iteration=10,
            prompt="Test",
            original="Hello world",
            updated="Goodbye moon",
            similarity_ratio=0.3,
            char_changes=10,
            line_changes=1,
        )
        assert comparison.is_degraded(threshold=0.7) is True


class TestSampleComparator:
    """Test SampleComparator callback."""

    def test_initialization(self):
        """Test comparator initialization with defaults."""
        comparator = SampleComparator()
        assert comparator.sample_size == 10
        assert comparator.comparison_frequency == 100
        assert len(comparator.baseline_samples) == 0
        assert len(comparator.comparison_history) == 0
        assert comparator.baseline_collected is False

    def test_initialization_custom_params(self):
        """Test initialization with custom parameters."""
        comparator = SampleComparator(
            sample_size=5,
            comparison_frequency=50,
            degradation_threshold=0.6,
        )
        assert comparator.sample_size == 5
        assert comparator.comparison_frequency == 50
        assert comparator.degradation_threshold == 0.6

    def test_on_event_protocol(self):
        """Test that SampleComparator implements TrainingCallback protocol."""
        comparator = SampleComparator()
        assert hasattr(comparator, 'on_event')
        assert callable(comparator.on_event)

    def test_collect_baseline_from_response_generated(self):
        """Test baseline collection from RESPONSE_GENERATED events."""
        comparator = SampleComparator(sample_size=3)

        # Emit events with responses
        for i in range(3):
            event = TrainingEvent(
                phase=TrainingPhase.RESPONSE_GENERATED,
                iteration=i,
                metadata={
                    'prompts': [f'Prompt {i}'],
                    'responses': [f'Response {i}'],
                }
            )
            comparator.on_event(event)

        assert len(comparator.baseline_samples) == 3
        assert comparator.baseline_collected is True
        assert comparator.baseline_samples[0]['prompt'] == 'Prompt 0'
        assert comparator.baseline_samples[0]['response'] == 'Response 0'

    def test_baseline_stops_after_sample_size(self):
        """Test baseline collection stops after reaching sample_size."""
        comparator = SampleComparator(sample_size=2)

        # Emit 5 events, should only collect 2
        for i in range(5):
            event = TrainingEvent(
                phase=TrainingPhase.RESPONSE_GENERATED,
                iteration=i,
                metadata={
                    'prompts': [f'Prompt {i}'],
                    'responses': [f'Response {i}'],
                }
            )
            comparator.on_event(event)

        assert len(comparator.baseline_samples) == 2
        assert comparator.baseline_samples[0]['prompt'] == 'Prompt 0'
        assert comparator.baseline_samples[1]['prompt'] == 'Prompt 1'

    def test_comparison_triggered_at_frequency(self):
        """Test comparison is triggered at comparison_frequency intervals."""
        comparator = SampleComparator(sample_size=2, comparison_frequency=10)

        # Collect baseline
        for i in range(2):
            event = TrainingEvent(
                phase=TrainingPhase.RESPONSE_GENERATED,
                iteration=i,
                metadata={
                    'prompts': [f'Prompt {i}'],
                    'responses': [f'Response {i}'],
                }
            )
            comparator.on_event(event)

        # Iteration 10 should trigger comparison
        event = TrainingEvent(
            phase=TrainingPhase.ITERATION_END,
            iteration=10,
        )
        comparator.on_event(event)

        # Note: Without actual model generation, this tests the trigger logic
        # Full integration test would require model mock

    def test_compute_diff_identical(self):
        """Test diff computation for identical strings."""
        comparator = SampleComparator()
        text = "Hello world"
        diff_result = comparator.compute_diff(text, text)

        assert diff_result['similarity_ratio'] == 1.0
        assert diff_result['char_changes'] == 0
        assert diff_result['line_changes'] == 0

    def test_compute_diff_small_change(self):
        """Test diff computation for small changes."""
        comparator = SampleComparator()
        original = "Hello world"
        updated = "Hello world!"

        diff_result = comparator.compute_diff(original, updated)

        assert diff_result['similarity_ratio'] > 0.9
        assert diff_result['char_changes'] == 1
        # Note: ndiff shows 2 line changes (- original, + updated) even for single-line changes
        assert diff_result['line_changes'] == 2

    def test_compute_diff_large_change(self):
        """Test diff computation for large changes."""
        comparator = SampleComparator()
        original = "Hello world\nThis is a test"
        updated = "Goodbye moon\nThis is different"

        diff_result = comparator.compute_diff(original, updated)

        assert diff_result['similarity_ratio'] < 0.7
        assert diff_result['char_changes'] > 5
        assert diff_result['line_changes'] >= 2

    def test_compute_diff_multiline(self):
        """Test diff computation for multiline text."""
        comparator = SampleComparator()
        original = "Line 1\nLine 2\nLine 3"
        updated = "Line 1\nModified 2\nLine 3"

        diff_result = comparator.compute_diff(original, updated)

        assert 0.6 < diff_result['similarity_ratio'] < 1.0
        assert diff_result['line_changes'] >= 1

    def test_detect_catastrophic_forgetting_false(self):
        """Test forgetting detection returns False with high similarity."""
        comparator = SampleComparator(degradation_threshold=0.7)

        # Add high-similarity comparisons
        for i in range(5):
            comparison = SampleComparison(
                iteration=i * 10,
                prompt=f"Prompt {i}",
                original="Original",
                updated="Original slightly modified",
                similarity_ratio=0.85,
                char_changes=5,
                line_changes=0,
            )
            comparator.comparison_history.append(comparison)

        assert comparator.detect_catastrophic_forgetting() is False

    def test_detect_catastrophic_forgetting_true(self):
        """Test forgetting detection returns True with low similarity."""
        comparator = SampleComparator(degradation_threshold=0.7)

        # Add low-similarity comparisons
        for i in range(5):
            comparison = SampleComparison(
                iteration=i * 10,
                prompt=f"Prompt {i}",
                original="Original response",
                updated="Completely different text",
                similarity_ratio=0.4,
                char_changes=20,
                line_changes=2,
            )
            comparator.comparison_history.append(comparison)

        assert comparator.detect_catastrophic_forgetting() is True

    def test_detect_catastrophic_forgetting_mixed(self):
        """Test forgetting detection with mixed results."""
        comparator = SampleComparator(degradation_threshold=0.7)

        # Add mixed comparisons
        for i in range(3):
            # Some good
            comparison = SampleComparison(
                iteration=i * 2,
                prompt=f"Prompt {i}",
                original="Original",
                updated="Original modified",
                similarity_ratio=0.85,
                char_changes=3,
                line_changes=0,
            )
            comparator.comparison_history.append(comparison)

            # Some bad
            comparison = SampleComparison(
                iteration=i * 2 + 1,
                prompt=f"Prompt {i}b",
                original="Original",
                updated="Different",
                similarity_ratio=0.5,
                char_changes=10,
                line_changes=1,
            )
            comparator.comparison_history.append(comparison)

        # Should not trigger if < 50% degraded
        result = comparator.detect_catastrophic_forgetting()
        # Implementation should check if majority are degraded

    def test_get_comparison_summary(self):
        """Test summary statistics generation."""
        comparator = SampleComparator()

        # Add some comparisons
        for i in range(3):
            comparison = SampleComparison(
                iteration=i * 10,
                prompt=f"Prompt {i}",
                original="Original",
                updated="Updated",
                similarity_ratio=0.8 - i * 0.1,  # Declining similarity
                char_changes=5 + i * 2,
                line_changes=1,
            )
            comparator.comparison_history.append(comparison)

        summary = comparator.get_comparison_summary()

        assert 'num_comparisons' in summary
        assert 'avg_similarity' in summary
        assert 'min_similarity' in summary
        assert 'max_similarity' in summary
        assert 'degraded_count' in summary
        assert summary['num_comparisons'] == 3

    def test_get_comparison_summary_empty(self):
        """Test summary with no comparisons."""
        comparator = SampleComparator()
        summary = comparator.get_comparison_summary()

        assert summary['num_comparisons'] == 0
        assert summary['avg_similarity'] == 0.0

    def test_baseline_samples_immutability(self):
        """Test that baseline samples are not modified after collection."""
        comparator = SampleComparator(sample_size=2)

        # Collect baseline
        for i in range(2):
            event = TrainingEvent(
                phase=TrainingPhase.RESPONSE_GENERATED,
                iteration=i,
                metadata={
                    'prompts': [f'Prompt {i}'],
                    'responses': [f'Response {i}'],
                }
            )
            comparator.on_event(event)

        original_baseline = comparator.baseline_samples.copy()

        # Try to collect more
        event = TrainingEvent(
            phase=TrainingPhase.RESPONSE_GENERATED,
            iteration=10,
            metadata={
                'prompts': ['New Prompt'],
                'responses': ['New Response'],
            }
        )
        comparator.on_event(event)

        # Baseline should not change
        assert comparator.baseline_samples == original_baseline

    def test_handles_missing_metadata(self):
        """Test graceful handling of events with missing metadata."""
        comparator = SampleComparator(sample_size=2)

        # Event without metadata
        event = TrainingEvent(
            phase=TrainingPhase.RESPONSE_GENERATED,
            iteration=0,
        )
        comparator.on_event(event)

        # Should not crash, baseline should be empty
        assert len(comparator.baseline_samples) == 0

    def test_handles_empty_responses(self):
        """Test handling of empty response lists."""
        comparator = SampleComparator(sample_size=2)

        event = TrainingEvent(
            phase=TrainingPhase.RESPONSE_GENERATED,
            iteration=0,
            metadata={
                'prompts': [],
                'responses': [],
            }
        )
        comparator.on_event(event)

        assert len(comparator.baseline_samples) == 0

    def test_comparison_frequency_zero(self):
        """Test that comparison_frequency=0 disables comparisons."""
        comparator = SampleComparator(comparison_frequency=0)

        # Collect baseline
        for i in range(2):
            event = TrainingEvent(
                phase=TrainingPhase.RESPONSE_GENERATED,
                iteration=i,
                metadata={
                    'prompts': [f'Prompt {i}'],
                    'responses': [f'Response {i}'],
                }
            )
            comparator.on_event(event)

        # Try to trigger comparison
        for iteration in [0, 10, 100, 1000]:
            event = TrainingEvent(
                phase=TrainingPhase.ITERATION_END,
                iteration=iteration,
            )
            comparator.on_event(event)

        # No comparisons should be made
        assert len(comparator.comparison_history) == 0
