"""MODULE: test_callbacks.py
PURPOSE: Unit tests for callback system
KEY COMPONENTS:
- Test TrainingCallback Protocol
- Test CallbackManager
DEPENDENCIES: pytest, src.training.monitoring.callbacks, src.training.monitoring.events
"""

import pytest
from src.training.monitoring.callbacks import CallbackManager, TrainingCallback
from src.training.monitoring.events import TrainingEvent, TrainingPhase


class MockCallback:
    """Mock callback for testing."""

    def __init__(self):
        self.events_received = []
        self.call_count = 0

    def on_event(self, event: TrainingEvent) -> None:
        """Record received events."""
        self.events_received.append(event)
        self.call_count += 1


class ErrorCallback:
    """Callback that raises errors for testing error handling."""

    def __init__(self, error_type=RuntimeError):
        self.error_type = error_type

    def on_event(self, event: TrainingEvent) -> None:
        """Always raises an error."""
        raise self.error_type("Intentional test error")


class TestTrainingCallback:
    """Test TrainingCallback Protocol."""

    def test_protocol_implementation(self):
        """Test that MockCallback implements the protocol."""
        callback = MockCallback()
        assert hasattr(callback, 'on_event')
        assert callable(callback.on_event)

    def test_callback_receives_event(self):
        """Test that callback can receive and process events."""
        callback = MockCallback()
        event = TrainingEvent(
            phase=TrainingPhase.ITERATION_START,
            iteration=5
        )
        callback.on_event(event)
        assert len(callback.events_received) == 1
        assert callback.events_received[0] == event


class TestCallbackManager:
    """Test CallbackManager."""

    def test_initialization(self):
        """Test manager initialization."""
        manager = CallbackManager()
        assert manager.num_callbacks == 0
        assert manager.is_enabled

    def test_register_callback(self):
        """Test registering a callback."""
        manager = CallbackManager()
        callback = MockCallback()
        manager.register(callback)
        assert manager.num_callbacks == 1

    def test_register_multiple_callbacks(self):
        """Test registering multiple callbacks."""
        manager = CallbackManager()
        callback1 = MockCallback()
        callback2 = MockCallback()
        manager.register(callback1)
        manager.register(callback2)
        assert manager.num_callbacks == 2

    def test_register_invalid_callback(self):
        """Test registering object without on_event method."""
        manager = CallbackManager()
        invalid_callback = object()
        with pytest.raises(TypeError, match="must implement on_event"):
            manager.register(invalid_callback)

    def test_unregister_callback(self):
        """Test unregistering a callback."""
        manager = CallbackManager()
        callback = MockCallback()
        manager.register(callback)
        assert manager.num_callbacks == 1
        manager.unregister(callback)
        assert manager.num_callbacks == 0

    def test_unregister_nonexistent_callback(self):
        """Test unregistering callback that was never registered."""
        manager = CallbackManager()
        callback = MockCallback()
        # Should not raise error
        manager.unregister(callback)
        assert manager.num_callbacks == 0

    def test_emit_to_single_callback(self):
        """Test emitting event to single callback."""
        manager = CallbackManager()
        callback = MockCallback()
        manager.register(callback)

        event = TrainingEvent(
            phase=TrainingPhase.ITERATION_END,
            iteration=10,
            metrics={'loss': 2.5}
        )
        manager.emit(event)

        assert callback.call_count == 1
        assert len(callback.events_received) == 1
        assert callback.events_received[0] == event

    def test_emit_to_multiple_callbacks(self):
        """Test emitting event to multiple callbacks."""
        manager = CallbackManager()
        callback1 = MockCallback()
        callback2 = MockCallback()
        manager.register(callback1)
        manager.register(callback2)

        event = TrainingEvent(
            phase=TrainingPhase.EPOCH_START,
            iteration=0,
            epoch=2
        )
        manager.emit(event)

        assert callback1.call_count == 1
        assert callback2.call_count == 1
        assert callback1.events_received[0] == event
        assert callback2.events_received[0] == event

    def test_emit_when_disabled(self):
        """Test that emit does nothing when manager is disabled."""
        manager = CallbackManager()
        callback = MockCallback()
        manager.register(callback)
        manager.disable()

        event = TrainingEvent(
            phase=TrainingPhase.ITERATION_START,
            iteration=0
        )
        manager.emit(event)

        assert callback.call_count == 0

    def test_enable_disable(self):
        """Test enable/disable functionality."""
        manager = CallbackManager()
        assert manager.is_enabled

        manager.disable()
        assert not manager.is_enabled

        manager.enable()
        assert manager.is_enabled

    def test_error_handling(self, capsys):
        """Test that callback errors don't stop other callbacks."""
        manager = CallbackManager()
        callback1 = ErrorCallback()
        callback2 = MockCallback()
        manager.register(callback1)
        manager.register(callback2)

        event = TrainingEvent(
            phase=TrainingPhase.ITERATION_START,
            iteration=0
        )
        manager.emit(event)

        # callback2 should still receive event despite callback1 error
        assert callback2.call_count == 1

        # Check that warning was printed
        captured = capsys.readouterr()
        assert "Warning" in captured.out
        assert "ErrorCallback" in captured.out

    def test_clear_callbacks(self):
        """Test clearing all callbacks."""
        manager = CallbackManager()
        callback1 = MockCallback()
        callback2 = MockCallback()
        manager.register(callback1)
        manager.register(callback2)
        assert manager.num_callbacks == 2

        manager.clear()
        assert manager.num_callbacks == 0

    def test_multiple_events(self):
        """Test handling multiple events in sequence."""
        manager = CallbackManager()
        callback = MockCallback()
        manager.register(callback)

        events = [
            TrainingEvent(phase=TrainingPhase.TRAINING_START, iteration=0),
            TrainingEvent(phase=TrainingPhase.ITERATION_START, iteration=1),
            TrainingEvent(phase=TrainingPhase.ITERATION_END, iteration=1),
            TrainingEvent(phase=TrainingPhase.TRAINING_END, iteration=-1),
        ]

        for event in events:
            manager.emit(event)

        assert callback.call_count == len(events)
        assert len(callback.events_received) == len(events)
        for i, event in enumerate(events):
            assert callback.events_received[i] == event

    def test_callback_registration_order_preserved(self):
        """Test that callbacks are called in registration order."""
        manager = CallbackManager()

        class OrderTracker:
            def __init__(self):
                self.order = []

        tracker = OrderTracker()

        class OrderedCallback:
            def __init__(self, name, tracker):
                self.name = name
                self.tracker = tracker

            def on_event(self, event):
                self.tracker.order.append(self.name)

        callback1 = OrderedCallback("first", tracker)
        callback2 = OrderedCallback("second", tracker)
        callback3 = OrderedCallback("third", tracker)

        manager.register(callback1)
        manager.register(callback2)
        manager.register(callback3)

        event = TrainingEvent(
            phase=TrainingPhase.ITERATION_START,
            iteration=0
        )
        manager.emit(event)

        assert tracker.order == ["first", "second", "third"]

    def test_slots_memory_efficiency(self):
        """Test that CallbackManager uses slots."""
        manager = CallbackManager()
        assert hasattr(manager, '__slots__')
