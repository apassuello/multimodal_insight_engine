"""MODULE: terminal_display.py
PURPOSE: Rich terminal display for training monitoring
KEY COMPONENTS:
- TerminalDisplay: Live terminal UI with SIMPLE and VERBOSE modes
DEPENDENCIES: rich, time, typing, events, verbosity, metrics
SPECIAL NOTES: Rate-limited updates (max 2 Hz), colored trend indicators
"""

import time
from typing import Optional, Dict
from rich.console import Console
from rich.table import Table
from rich.live import Live
from rich.panel import Panel
from rich.text import Text
from rich.layout import Layout

from .events import TrainingEvent, TrainingPhase
from .verbosity import VerbosityLevel
from .metrics import MetricsStore


class TerminalDisplay:
    """
    Rich terminal display for training monitoring.

    Features:
    - SIMPLE mode: One-line status updates
    - VERBOSE mode: Detailed tables with metrics and trends
    - Rate-limited updates (max 2 Hz to avoid flickering)
    - Color-coded trend indicators
    - Memory-efficient with bounded history

    Example:
        display = TerminalDisplay(verbosity=VerbosityLevel.VERBOSE)
        monitor.register_callback(display)
    """

    def __init__(
        self,
        verbosity: VerbosityLevel = VerbosityLevel.SIMPLE,
        metrics_store: Optional[MetricsStore] = None,
        update_rate_hz: float = 2.0
    ):
        """
        Initialize terminal display.

        Args:
            verbosity: Display detail level (SIMPLE or VERBOSE)
            metrics_store: Shared metrics store for trend analysis
            update_rate_hz: Maximum display update frequency
        """
        self.verbosity = verbosity
        self.metrics_store = metrics_store
        self.min_update_interval = 1.0 / update_rate_hz

        # Display state
        self.console = Console()
        self.last_update_time = 0.0
        self.current_iteration = 0
        self.current_epoch = 0
        self.latest_metrics: Dict[str, float] = {}

        # Live display (only for verbose mode)
        self._live: Optional[Live] = None
        self._training_active = False

    def on_event(self, event: TrainingEvent) -> None:
        """
        Handle training events and update display.

        Args:
            event: Training event to process
        """
        # Update state
        self.current_iteration = event.iteration
        self.current_epoch = event.epoch
        self.latest_metrics.update(event.metrics)

        # Handle lifecycle events
        if event.phase == TrainingPhase.TRAINING_START:
            self._on_training_start()
        elif event.phase == TrainingPhase.TRAINING_END:
            self._on_training_end(event)
        elif event.phase == TrainingPhase.ITERATION_END:
            self._on_iteration_end(event)

    def _on_training_start(self) -> None:
        """Handle training start."""
        self._training_active = True
        self.console.print("\n[bold cyan]Training Started[/bold cyan]")
        self.console.print(f"Display mode: [yellow]{self.verbosity.name}[/yellow]\n")

        # Start live display for verbose mode
        if self.verbosity == VerbosityLevel.VERBOSE:
            self._live = Live(
                self._create_verbose_display(),
                console=self.console,
                refresh_per_second=2
            )
            self._live.start()

    def _on_training_end(self, event: TrainingEvent) -> None:
        """Handle training end."""
        self._training_active = False

        # Stop live display
        if self._live is not None:
            self._live.stop()
            self._live = None

        # Print summary
        stopped_early = event.metadata.get('stopped_early', False)
        if stopped_early:
            self.console.print("\n[yellow]Training stopped early[/yellow]")
        else:
            self.console.print("\n[bold green]Training completed[/bold green]")

        # Print final metrics
        if self.latest_metrics:
            self.console.print("\n[bold]Final Metrics:[/bold]")
            for name, value in sorted(self.latest_metrics.items()):
                self.console.print(f"  {name}: {value:.4f}")

    def _on_iteration_end(self, event: TrainingEvent) -> None:
        """Handle iteration end."""
        current_time = time.time()

        # Rate limiting
        if current_time - self.last_update_time < self.min_update_interval:
            return

        self.last_update_time = current_time

        # Update display based on mode
        if self.verbosity == VerbosityLevel.SIMPLE:
            self._update_simple_display(event)
        elif self.verbosity == VerbosityLevel.VERBOSE and self._live is not None:
            self._live.update(self._create_verbose_display())

    def _update_simple_display(self, event: TrainingEvent) -> None:
        """Update display in SIMPLE mode."""
        # Build one-line status
        parts = [f"Iter {self.current_iteration}"]

        if self.current_epoch >= 0:
            parts.append(f"Epoch {self.current_epoch}")

        # Add key metrics
        if 'loss' in event.metrics:
            parts.append(f"Loss: {event.metrics['loss']:.4f}")
        if 'reward' in event.metrics:
            parts.append(f"Reward: {event.metrics['reward']:.4f}")

        status = " | ".join(parts)
        self.console.print(f"[dim]{status}[/dim]")

    def _create_verbose_display(self) -> Layout:
        """Create verbose display layout."""
        layout = Layout()
        layout.split_column(
            Layout(self._create_status_panel(), size=3),
            Layout(self._create_metrics_table())
        )
        return layout

    def _create_status_panel(self) -> Panel:
        """Create status panel for verbose mode."""
        status_text = Text()
        status_text.append(f"Iteration: {self.current_iteration}  ", style="bold cyan")
        if self.current_epoch >= 0:
            status_text.append(f"Epoch: {self.current_epoch}", style="bold yellow")

        return Panel(
            status_text,
            title="Training Status",
            border_style="cyan"
        )

    def _create_metrics_table(self) -> Table:
        """Create metrics table for verbose mode."""
        table = Table(
            title="Metrics",
            show_header=True,
            header_style="bold magenta",
            border_style="blue"
        )

        table.add_column("Metric", style="cyan", no_wrap=True)
        table.add_column("Current", justify="right", style="white")
        table.add_column("Mean", justify="right", style="dim")
        table.add_column("Std", justify="right", style="dim")
        table.add_column("Trend", justify="center", style="yellow")

        # Add rows for each metric
        if self.latest_metrics:
            for name in sorted(self.latest_metrics.keys()):
                self._add_metric_row(table, name)
        else:
            table.add_row("No metrics yet", "-", "-", "-", "-")

        return table

    def _add_metric_row(self, table: Table, metric_name: str) -> None:
        """
        Add a metric row to the table.

        Args:
            table: Rich table to add row to
            metric_name: Name of the metric
        """
        current_value = self.latest_metrics.get(metric_name, 0.0)

        # Get statistics and trend if store is available
        if self.metrics_store is not None and self.metrics_store.has_metric(metric_name):
            stats = self.metrics_store.get_statistics(metric_name)
            trend_value = self.metrics_store.compute_trend(metric_name, window=10)

            mean = stats['mean']
            std = stats['std']
            trend_indicator = self._format_trend(trend_value)
        else:
            mean = current_value
            std = 0.0
            trend_indicator = "-"

        table.add_row(
            metric_name,
            f"{current_value:.4f}",
            f"{mean:.4f}",
            f"{std:.4f}",
            trend_indicator
        )

    def _format_trend(self, trend: float) -> str:
        """
        Format trend value with colored indicators.

        Args:
            trend: Trend slope value

        Returns:
            Formatted trend string with color
        """
        if abs(trend) < 1e-4:
            return "[dim]→[/dim]"  # Stable
        elif trend > 0:
            # Increasing - could be good (reward) or bad (loss)
            return f"[green]↑[/green] +{trend:.2e}"
        else:
            # Decreasing
            return f"[red]↓[/red] {trend:.2e}"

    def set_metrics_store(self, store: MetricsStore) -> None:
        """
        Set metrics store for trend analysis.

        Args:
            store: Metrics store to use
        """
        self.metrics_store = store

    @property
    def is_active(self) -> bool:
        """Check if display is currently active."""
        return self._training_active
