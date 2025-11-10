"""MODULE: training_monitor.py
PURPOSE: Main orchestrator for training monitoring system
KEY COMPONENTS:
- TrainingMonitor: Central coordinator managing all callbacks
DEPENDENCIES: callbacks, events, metrics, verbosity
SPECIAL NOTES: Entry point for integrated monitoring
"""

from contextlib import contextmanager
from pathlib import Path
from typing import Optional

from .callbacks import CallbackManager, TrainingCallback
from .events import TrainingEvent, TrainingPhase
from .metrics import MetricsStore
from .verbosity import VerbosityLevel


class TrainingMonitor:
    """
    Central orchestrator for Constitutional AI training monitoring.

    Manages the training monitoring lifecycle:
    - Callback registration and event dispatching
    - Metrics aggregation across components
    - Early stopping decisions based on quality alerts
    - Context management for clean setup/teardown

    Example:
        monitor = TrainingMonitor(VerbosityLevel.VERBOSE, "./output")

        # Register callbacks
        monitor.register_callback(TerminalDisplay(monitor.verbosity))
        monitor.register_callback(QualityAnalyzer())

        # Monitor training
        with monitor.monitor_context():
            trainer.train(...)

        # Generate reports
        monitor.save_reports()
    """

    def __init__(
        self,
        verbosity: VerbosityLevel = VerbosityLevel.SIMPLE,
        output_dir: str = "output/training_monitor"
    ):
        """
        Initialize training monitor.

        Args:
            verbosity: Display detail level
            output_dir: Directory for saving reports and plots
        """
        self.verbosity = verbosity
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Core components
        self._callback_manager = CallbackManager()
        self._metrics_store = MetricsStore(capacity=10_000)

        # State tracking
        self._training_active = False
        self._should_stop = False
        self._stop_reason: Optional[str] = None

    def register_callback(self, callback: TrainingCallback) -> None:
        """
        Register a callback to receive training events.

        Args:
            callback: Callback implementing TrainingCallback protocol
        """
        self._callback_manager.register(callback)

    def unregister_callback(self, callback: TrainingCallback) -> None:
        """
        Remove a callback from receiving events.

        Args:
            callback: Previously registered callback
        """
        self._callback_manager.unregister(callback)

    @contextmanager
    def monitor_context(self):
        """
        Context manager for monitoring lifecycle.

        Handles setup and teardown:
        - Emits TRAINING_START event on enter
        - Emits TRAINING_END event on exit
        - Ensures clean state even if training fails

        Example:
            with monitor.monitor_context():
                trainer.train(...)
        """
        try:
            # Setup
            self._training_active = True
            self._should_stop = False
            self._stop_reason = None

            # Notify callbacks
            start_event = TrainingEvent(
                phase=TrainingPhase.TRAINING_START,
                iteration=0,
                epoch=0
            )
            self._callback_manager.emit(start_event)

            yield self

        finally:
            # Teardown
            self._training_active = False

            end_event = TrainingEvent(
                phase=TrainingPhase.TRAINING_END,
                iteration=-1,
                epoch=-1,
                metadata={'stopped_early': self._should_stop}
            )
            self._callback_manager.emit(end_event)

    def on_event(self, event: TrainingEvent) -> None:
        """
        Process a training event.

        Records metrics and dispatches to callbacks.

        Args:
            event: Training event to process
        """
        if not self._training_active:
            return

        # Record metrics
        for name, value in event.metrics.items():
            self._metrics_store.record(name, value)

        # Dispatch to callbacks
        self._callback_manager.emit(event)

    def should_stop_early(self) -> tuple[bool, Optional[str]]:
        """
        Check if training should stop early.

        Returns:
            Tuple of (should_stop, reason)
        """
        return self._should_stop, self._stop_reason

    def request_early_stop(self, reason: str) -> None:
        """
        Request early stopping of training.

        Args:
            reason: Explanation for why training should stop
        """
        self._should_stop = True
        self._stop_reason = reason

    def get_metrics_store(self) -> MetricsStore:
        """Get the metrics store for analysis."""
        return self._metrics_store

    @property
    def is_training_active(self) -> bool:
        """Check if training is currently active."""
        return self._training_active

    @property
    def num_callbacks(self) -> int:
        """Get number of registered callbacks."""
        return self._callback_manager.num_callbacks
