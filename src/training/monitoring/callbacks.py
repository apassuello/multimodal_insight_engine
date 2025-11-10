"""MODULE: callbacks.py
PURPOSE: Callback system for training monitoring
KEY COMPONENTS:
- TrainingCallback: Protocol for callback implementations
- CallbackManager: Efficient callback dispatcher
DEPENDENCIES: typing, events
SPECIAL NOTES: Uses Protocol for zero-coupling, type-safe callbacks
"""

from typing import Protocol, List
from .events import TrainingEvent


class TrainingCallback(Protocol):
    """
    Protocol for training callbacks.

    Callbacks receive training events and can observe/react to training progress.
    Using Protocol allows zero-coupling - callbacks don't need to inherit from
    a base class.

    Example:
        class MyCallback:
            def on_event(self, event: TrainingEvent) -> None:
                if event.phase == TrainingPhase.ITERATION_END:
                    print(f"Iteration {event.iteration}: Loss = {event.get_metric('loss')}")

        manager = CallbackManager()
        manager.register(MyCallback())
    """

    def on_event(self, event: TrainingEvent) -> None:
        """
        Handle a training event.

        Args:
            event: Immutable training event with metrics and metadata
        """
        ...


class CallbackManager:
    """
    Efficient callback dispatcher with minimal overhead.

    Manages callback registration and event dispatching with O(n) complexity
    where n is the number of registered callbacks (typically < 10).

    Design:
    - Uses __slots__ for memory efficiency
    - Maintains callback list for fast iteration
    - Provides enable/disable for performance-critical sections
    """
    __slots__ = ('_callbacks', '_enabled')

    def __init__(self) -> None:
        self._callbacks: List[TrainingCallback] = []
        self._enabled: bool = True

    def register(self, callback: TrainingCallback) -> None:
        """
        Register a callback to receive events.

        Args:
            callback: Object implementing TrainingCallback protocol
        """
        if not hasattr(callback, 'on_event'):
            raise TypeError(
                f"Callback must implement on_event method. "
                f"Got: {type(callback).__name__}"
            )
        self._callbacks.append(callback)

    def unregister(self, callback: TrainingCallback) -> None:
        """
        Remove a callback from receiving events.

        Args:
            callback: Previously registered callback
        """
        if callback in self._callbacks:
            self._callbacks.remove(callback)

    def emit(self, event: TrainingEvent) -> None:
        """
        Dispatch event to all registered callbacks.

        Args:
            event: Training event to dispatch
        """
        if not self._enabled:
            return

        for callback in self._callbacks:
            try:
                callback.on_event(event)
            except Exception as e:
                # Don't let callback errors stop training
                print(f"Warning: Callback {type(callback).__name__} raised error: {e}")

    def enable(self) -> None:
        """Enable event dispatching."""
        self._enabled = True

    def disable(self) -> None:
        """Disable event dispatching (for performance-critical sections)."""
        self._enabled = False

    def clear(self) -> None:
        """Remove all registered callbacks."""
        self._callbacks.clear()

    @property
    def num_callbacks(self) -> int:
        """Get number of registered callbacks."""
        return len(self._callbacks)

    @property
    def is_enabled(self) -> bool:
        """Check if callback dispatching is enabled."""
        return self._enabled
