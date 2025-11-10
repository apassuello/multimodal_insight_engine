"""MODULE: test_events.py
PURPOSE: Unit tests for training event system
KEY COMPONENTS:
- Test TrainingPhase enum
- Test TrainingEvent dataclass
DEPENDENCIES: pytest, src.training.monitoring.events
"""

import pytest
import time
from src.training.monitoring.events import TrainingEvent, TrainingPhase


class TestTrainingPhase:
    """Test TrainingPhase enum."""

    def test_all_phases_exist(self):
        """Test that all expected phases are defined."""
        expected_phases = [
            'INITIALIZATION',
            'TRAINING_START',
            'EPOCH_START',
            'ITERATION_START',
            'RESPONSE_GENERATED',
            'REWARD_COMPUTED',
            'POLICY_UPDATE',
            'VALUE_UPDATE',
            'ITERATION_END',
            'EPOCH_END',
            'TRAINING_END',
        ]
        for phase in expected_phases:
            assert hasattr(TrainingPhase, phase)

    def test_phases_distinct(self):
        """Test that all phases are distinct."""
        phases = list(TrainingPhase)
        assert len(phases) == len(set(phases))

    def test_phase_order(self):
        """Test that phases have consistent ordering."""
        # TrainingPhase uses auto() so values should be sequential
        phases = list(TrainingPhase)
        values = [p.value for p in phases]
        assert values == sorted(values)


class TestTrainingEvent:
    """Test TrainingEvent dataclass."""

    def test_basic_creation(self):
        """Test creating a basic event."""
        event = TrainingEvent(
            phase=TrainingPhase.ITERATION_START,
            iteration=5,
            epoch=1
        )
        assert event.phase == TrainingPhase.ITERATION_START
        assert event.iteration == 5
        assert event.epoch == 1
        assert event.metrics == {}
        assert event.metadata == {}

    def test_creation_with_metrics(self):
        """Test creating event with metrics."""
        metrics = {'loss': 2.5, 'reward': 0.7}
        event = TrainingEvent(
            phase=TrainingPhase.ITERATION_END,
            iteration=10,
            metrics=metrics
        )
        assert event.metrics == metrics
        assert event.get_metric('loss') == 2.5
        assert event.get_metric('reward') == 0.7

    def test_creation_with_metadata(self):
        """Test creating event with metadata."""
        metadata = {'response': 'test response', 'prompt': 'test prompt'}
        event = TrainingEvent(
            phase=TrainingPhase.RESPONSE_GENERATED,
            iteration=3,
            metadata=metadata
        )
        assert event.metadata == metadata
        assert event.get_metadata('response') == 'test response'
        assert event.get_metadata('prompt') == 'test prompt'

    def test_default_epoch(self):
        """Test default epoch value is -1."""
        event = TrainingEvent(
            phase=TrainingPhase.TRAINING_START,
            iteration=0
        )
        assert event.epoch == -1

    def test_timestamp_auto_generated(self):
        """Test that timestamp is automatically generated."""
        before = time.time()
        event = TrainingEvent(
            phase=TrainingPhase.ITERATION_START,
            iteration=0
        )
        after = time.time()
        assert before <= event.timestamp <= after

    def test_get_metric_with_default(self):
        """Test getting metric with default value."""
        event = TrainingEvent(
            phase=TrainingPhase.ITERATION_END,
            iteration=1
        )
        assert event.get_metric('loss', 0.0) == 0.0
        assert event.get_metric('nonexistent', 99.0) == 99.0

    def test_has_metric(self):
        """Test checking if metric exists."""
        event = TrainingEvent(
            phase=TrainingPhase.ITERATION_END,
            iteration=1,
            metrics={'loss': 1.5}
        )
        assert event.has_metric('loss')
        assert not event.has_metric('reward')

    def test_get_metadata_with_default(self):
        """Test getting metadata with default value."""
        event = TrainingEvent(
            phase=TrainingPhase.RESPONSE_GENERATED,
            iteration=1
        )
        assert event.get_metadata('key', 'default') == 'default'
        assert event.get_metadata('nonexistent') is None

    def test_immutability(self):
        """Test that events are immutable (frozen)."""
        event = TrainingEvent(
            phase=TrainingPhase.ITERATION_START,
            iteration=0
        )
        with pytest.raises(AttributeError):
            event.iteration = 10

    def test_slots_memory_efficiency(self):
        """Test that event uses slots for memory efficiency."""
        event = TrainingEvent(
            phase=TrainingPhase.ITERATION_START,
            iteration=0
        )
        # Frozen dataclasses with slots don't have __dict__
        assert not hasattr(event, '__dict__')

    def test_multiple_metrics(self):
        """Test event with multiple metrics."""
        metrics = {
            'loss': 2.3,
            'reward': 0.65,
            'kl_div': 0.01,
            'value_loss': 0.5
        }
        event = TrainingEvent(
            phase=TrainingPhase.ITERATION_END,
            iteration=42,
            metrics=metrics
        )
        for key, value in metrics.items():
            assert event.get_metric(key) == value

    def test_empty_metrics_and_metadata(self):
        """Test that empty metrics and metadata don't share references."""
        event1 = TrainingEvent(
            phase=TrainingPhase.ITERATION_START,
            iteration=0
        )
        event2 = TrainingEvent(
            phase=TrainingPhase.ITERATION_START,
            iteration=1
        )
        # Each event should have its own empty dicts
        assert event1.metrics is not event2.metrics
        assert event1.metadata is not event2.metadata
