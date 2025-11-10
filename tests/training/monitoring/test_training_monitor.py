"""MODULE: test_training_monitor.py
PURPOSE: Unit tests for TrainingMonitor orchestrator
KEY COMPONENTS:
- Test TrainingMonitor lifecycle
- Test callback integration
- Test metrics aggregation
- Test early stopping
DEPENDENCIES: pytest, src.training.monitoring
"""

import pytest
from pathlib import Path
import tempfile
import shutil
from src.training.monitoring import (
    TrainingMonitor,
    TrainingCallback,
    TrainingEvent,
    TrainingPhase,
    VerbosityLevel,
    MetricsStore
)


class MockCallback:
    """Mock callback for testing."""

    def __init__(self):
        self.events = []
        self.phases_seen = set()

    def on_event(self, event: TrainingEvent) -> None:
        """Record events."""
        self.events.append(event)
        self.phases_seen.add(event.phase)


class TestTrainingMonitor:
    """Test TrainingMonitor orchestrator."""

    def setup_method(self):
        """Create temporary directory for each test."""
        self.temp_dir = tempfile.mkdtemp()

    def teardown_method(self):
        """Clean up temporary directory."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_initialization_default(self):
        """Test initialization with defaults."""
        monitor = TrainingMonitor()
        assert monitor.verbosity == VerbosityLevel.SIMPLE
        assert monitor.num_callbacks == 0
        assert not monitor.is_training_active

    def test_initialization_custom(self):
        """Test initialization with custom values."""
        monitor = TrainingMonitor(
            verbosity=VerbosityLevel.VERBOSE,
            output_dir=self.temp_dir
        )
        assert monitor.verbosity == VerbosityLevel.VERBOSE
        assert monitor.output_dir == Path(self.temp_dir)

    def test_output_dir_creation(self):
        """Test that output directory is created."""
        output_dir = Path(self.temp_dir) / "nested" / "monitoring"
        monitor = TrainingMonitor(output_dir=str(output_dir))
        assert output_dir.exists()
        assert output_dir.is_dir()

    def test_register_callback(self):
        """Test registering a callback."""
        monitor = TrainingMonitor()
        callback = MockCallback()
        assert monitor.num_callbacks == 0

        monitor.register_callback(callback)
        assert monitor.num_callbacks == 1

    def test_register_multiple_callbacks(self):
        """Test registering multiple callbacks."""
        monitor = TrainingMonitor()
        callback1 = MockCallback()
        callback2 = MockCallback()

        monitor.register_callback(callback1)
        monitor.register_callback(callback2)
        assert monitor.num_callbacks == 2

    def test_unregister_callback(self):
        """Test unregistering a callback."""
        monitor = TrainingMonitor()
        callback = MockCallback()
        monitor.register_callback(callback)
        assert monitor.num_callbacks == 1

        monitor.unregister_callback(callback)
        assert monitor.num_callbacks == 0

    def test_monitor_context_lifecycle(self):
        """Test monitor context manager lifecycle."""
        monitor = TrainingMonitor()
        callback = MockCallback()
        monitor.register_callback(callback)

        assert not monitor.is_training_active

        with monitor.monitor_context():
            assert monitor.is_training_active

        assert not monitor.is_training_active

        # Check that START and END events were emitted
        assert len(callback.events) == 2
        assert callback.events[0].phase == TrainingPhase.TRAINING_START
        assert callback.events[1].phase == TrainingPhase.TRAINING_END

    def test_monitor_context_exception_handling(self):
        """Test that monitor context handles exceptions properly."""
        monitor = TrainingMonitor()
        callback = MockCallback()
        monitor.register_callback(callback)

        with pytest.raises(RuntimeError):
            with monitor.monitor_context():
                assert monitor.is_training_active
                raise RuntimeError("Test error")

        # Should still emit END event and deactivate
        assert not monitor.is_training_active
        assert len(callback.events) == 2
        assert callback.events[1].phase == TrainingPhase.TRAINING_END

    def test_on_event_dispatching(self):
        """Test that on_event dispatches to callbacks."""
        monitor = TrainingMonitor()
        callback = MockCallback()
        monitor.register_callback(callback)

        with monitor.monitor_context():
            event = TrainingEvent(
                phase=TrainingPhase.ITERATION_END,
                iteration=5,
                metrics={'loss': 2.5}
            )
            monitor.on_event(event)

        # Should have START, our event, and END
        assert len(callback.events) == 3
        assert callback.events[1] == event

    def test_on_event_inactive_training(self):
        """Test that on_event does nothing when training inactive."""
        monitor = TrainingMonitor()
        callback = MockCallback()
        monitor.register_callback(callback)

        event = TrainingEvent(
            phase=TrainingPhase.ITERATION_END,
            iteration=5
        )
        monitor.on_event(event)

        # Should not dispatch event when not in context
        assert len(callback.events) == 0

    def test_metrics_recording(self):
        """Test that metrics are recorded to store."""
        monitor = TrainingMonitor()

        with monitor.monitor_context():
            event1 = TrainingEvent(
                phase=TrainingPhase.ITERATION_END,
                iteration=1,
                metrics={'loss': 2.5, 'reward': 0.6}
            )
            event2 = TrainingEvent(
                phase=TrainingPhase.ITERATION_END,
                iteration=2,
                metrics={'loss': 2.3, 'reward': 0.65}
            )
            monitor.on_event(event1)
            monitor.on_event(event2)

        store = monitor.get_metrics_store()
        assert store.has_metric('loss')
        assert store.has_metric('reward')

        loss_buffer = store.get_buffer('loss')
        assert len(loss_buffer) == 2

        reward_buffer = store.get_buffer('reward')
        assert len(reward_buffer) == 2

    def test_early_stopping_initial_state(self):
        """Test initial early stopping state."""
        monitor = TrainingMonitor()
        should_stop, reason = monitor.should_stop_early()
        assert not should_stop
        assert reason is None

    def test_request_early_stop(self):
        """Test requesting early stop."""
        monitor = TrainingMonitor()
        monitor.request_early_stop("Quality degradation detected")

        should_stop, reason = monitor.should_stop_early()
        assert should_stop
        assert reason == "Quality degradation detected"

    def test_early_stopping_reset_on_context(self):
        """Test that early stopping is reset in new context."""
        monitor = TrainingMonitor()
        callback = MockCallback()
        monitor.register_callback(callback)

        # First training run with early stop
        with monitor.monitor_context():
            monitor.request_early_stop("Test reason")
            should_stop, reason = monitor.should_stop_early()
            assert should_stop

        # Second training run should reset
        with monitor.monitor_context():
            should_stop, reason = monitor.should_stop_early()
            assert not should_stop
            assert reason is None

    def test_early_stop_metadata_in_end_event(self):
        """Test that early stop is recorded in END event metadata."""
        monitor = TrainingMonitor()
        callback = MockCallback()
        monitor.register_callback(callback)

        with monitor.monitor_context():
            monitor.request_early_stop("Test early stop")

        end_event = callback.events[-1]
        assert end_event.phase == TrainingPhase.TRAINING_END
        assert end_event.metadata['stopped_early'] is True

    def test_no_early_stop_metadata(self):
        """Test END event metadata when no early stop."""
        monitor = TrainingMonitor()
        callback = MockCallback()
        monitor.register_callback(callback)

        with monitor.monitor_context():
            pass  # Normal completion

        end_event = callback.events[-1]
        assert end_event.phase == TrainingPhase.TRAINING_END
        assert end_event.metadata['stopped_early'] is False

    def test_get_metrics_store(self):
        """Test getting metrics store."""
        monitor = TrainingMonitor()
        store = monitor.get_metrics_store()
        assert isinstance(store, MetricsStore)

    def test_multiple_training_sessions(self):
        """Test running multiple training sessions."""
        monitor = TrainingMonitor()
        callback = MockCallback()
        monitor.register_callback(callback)

        # First session
        with monitor.monitor_context():
            event = TrainingEvent(
                phase=TrainingPhase.ITERATION_END,
                iteration=1
            )
            monitor.on_event(event)

        session1_events = len(callback.events)

        # Second session
        with monitor.monitor_context():
            event = TrainingEvent(
                phase=TrainingPhase.ITERATION_END,
                iteration=1
            )
            monitor.on_event(event)

        # Should have events from both sessions
        assert len(callback.events) > session1_events

    def test_complex_workflow(self):
        """Test complex training workflow."""
        monitor = TrainingMonitor(verbosity=VerbosityLevel.VERBOSE)
        callback = MockCallback()
        monitor.register_callback(callback)

        with monitor.monitor_context():
            # Simulate training loop
            for iteration in range(3):
                # Iteration start
                monitor.on_event(TrainingEvent(
                    phase=TrainingPhase.ITERATION_START,
                    iteration=iteration
                ))

                # Response generated
                monitor.on_event(TrainingEvent(
                    phase=TrainingPhase.RESPONSE_GENERATED,
                    iteration=iteration,
                    metadata={'response': f'response_{iteration}'}
                ))

                # Reward computed
                monitor.on_event(TrainingEvent(
                    phase=TrainingPhase.REWARD_COMPUTED,
                    iteration=iteration,
                    metrics={'reward': 0.5 + iteration * 0.1}
                ))

                # Policy update
                monitor.on_event(TrainingEvent(
                    phase=TrainingPhase.POLICY_UPDATE,
                    iteration=iteration,
                    metrics={'policy_loss': 2.0 - iteration * 0.1}
                ))

                # Iteration end
                monitor.on_event(TrainingEvent(
                    phase=TrainingPhase.ITERATION_END,
                    iteration=iteration,
                    metrics={'total_loss': 1.5 - iteration * 0.05}
                ))

        # Verify all phases were seen
        expected_phases = {
            TrainingPhase.TRAINING_START,
            TrainingPhase.ITERATION_START,
            TrainingPhase.RESPONSE_GENERATED,
            TrainingPhase.REWARD_COMPUTED,
            TrainingPhase.POLICY_UPDATE,
            TrainingPhase.ITERATION_END,
            TrainingPhase.TRAINING_END,
        }
        assert expected_phases.issubset(callback.phases_seen)

        # Verify metrics were recorded
        store = monitor.get_metrics_store()
        assert store.has_metric('reward')
        assert store.has_metric('policy_loss')
        assert store.has_metric('total_loss')

        # Verify metric counts
        assert len(store.get_buffer('reward')) == 3
        assert len(store.get_buffer('policy_loss')) == 3
        assert len(store.get_buffer('total_loss')) == 3

    def test_callback_receives_all_event_data(self):
        """Test that callbacks receive complete event data."""
        monitor = TrainingMonitor()
        callback = MockCallback()
        monitor.register_callback(callback)

        with monitor.monitor_context():
            event = TrainingEvent(
                phase=TrainingPhase.ITERATION_END,
                iteration=42,
                epoch=3,
                metrics={'loss': 1.23, 'reward': 0.78},
                metadata={'sample': 'test data'}
            )
            monitor.on_event(event)

        received_event = callback.events[1]  # Skip START event
        assert received_event.iteration == 42
        assert received_event.epoch == 3
        assert received_event.get_metric('loss') == 1.23
        assert received_event.get_metric('reward') == 0.78
        assert received_event.get_metadata('sample') == 'test data'

    def test_verbosity_levels(self):
        """Test different verbosity levels."""
        simple_monitor = TrainingMonitor(verbosity=VerbosityLevel.SIMPLE)
        assert simple_monitor.verbosity == VerbosityLevel.SIMPLE

        verbose_monitor = TrainingMonitor(verbosity=VerbosityLevel.VERBOSE)
        assert verbose_monitor.verbosity == VerbosityLevel.VERBOSE

    def test_is_training_active_property(self):
        """Test is_training_active property."""
        monitor = TrainingMonitor()
        assert not monitor.is_training_active

        with monitor.monitor_context():
            assert monitor.is_training_active

        assert not monitor.is_training_active

    def test_num_callbacks_property(self):
        """Test num_callbacks property."""
        monitor = TrainingMonitor()
        assert monitor.num_callbacks == 0

        callback1 = MockCallback()
        callback2 = MockCallback()
        monitor.register_callback(callback1)
        assert monitor.num_callbacks == 1

        monitor.register_callback(callback2)
        assert monitor.num_callbacks == 2

        monitor.unregister_callback(callback1)
        assert monitor.num_callbacks == 1

    def test_context_manager_yield(self):
        """Test that context manager yields self."""
        monitor = TrainingMonitor()

        with monitor.monitor_context() as ctx:
            assert ctx is monitor
