"""MODULE: ppo_metrics_tracker.py
PURPOSE: Track PPO-specific training metrics (verbose mode only)
KEY COMPONENTS:
- PPOSnapshot: Dataclass for PPO metrics snapshot
- PPOMetricsTracker: Callback that tracks PPO training mechanics
DEPENDENCIES: dataclasses, typing, statistics, events, callbacks, verbosity
SPECIAL NOTES: Only active in VERBOSE mode, tracks detailed PPO mechanics
"""

from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional
import statistics
from .events import TrainingEvent, TrainingPhase
from .verbosity import VerbosityLevel


@dataclass(frozen=True, slots=True)
class PPOSnapshot:
    """
    Snapshot of PPO metrics at a specific training step.

    Design: Frozen dataclass for immutability and memory efficiency.
    All metric fields are optional since different phases provide different metrics.

    Attributes:
        iteration: Training iteration number
        phase: Training phase (POLICY_UPDATE or VALUE_UPDATE)
        policy_loss: Policy network loss (from policy updates)
        value_loss: Value network loss (from value updates)
        clip_fraction: Fraction of policy updates that were clipped
        gradient_norm: L2 norm of gradients
        learning_rate: Current learning rate
        advantage_mean: Mean of computed advantages
        advantage_std: Standard deviation of advantages
    """
    iteration: int
    phase: TrainingPhase
    policy_loss: Optional[float] = None
    value_loss: Optional[float] = None
    clip_fraction: Optional[float] = None
    gradient_norm: Optional[float] = None
    learning_rate: Optional[float] = None
    advantage_mean: Optional[float] = None
    advantage_std: Optional[float] = None


class PPOMetricsTracker:
    """
    Tracks PPO-specific training metrics for verbose mode.

    Features:
    - Tracks policy loss, value loss, clip fraction
    - Monitors gradient norms and learning rate schedules
    - Aggregates advantage statistics
    - Computes trends and summaries
    - Only active in VERBOSE mode (minimal overhead in SIMPLE mode)

    Design: Implements TrainingCallback protocol for event-driven tracking.

    Attributes:
        verbosity: Verbosity level controlling whether tracking is enabled
        ppo_history: List of all PPO snapshots
        is_enabled: Whether tracker is active (True for VERBOSE mode only)
    """
    __slots__ = ('verbosity', 'ppo_history', 'is_enabled')

    def __init__(self, verbosity: VerbosityLevel):
        """
        Initialize PPO metrics tracker.

        Args:
            verbosity: Verbosity level (only tracks in VERBOSE mode)
        """
        self.verbosity = verbosity
        self.ppo_history: List[PPOSnapshot] = []
        self.is_enabled = (verbosity == VerbosityLevel.VERBOSE)

    def on_event(self, event: TrainingEvent) -> None:
        """
        Handle training events to track PPO-specific metrics.

        Only processes POLICY_UPDATE and VALUE_UPDATE events in VERBOSE mode.

        Args:
            event: Training event with PPO metrics
        """
        if not self.is_enabled:
            return

        # Only track PPO-specific phases
        if event.phase == TrainingPhase.POLICY_UPDATE:
            self._track_policy_update(event)
        elif event.phase == TrainingPhase.VALUE_UPDATE:
            self._track_value_update(event)

    def _track_policy_update(self, event: TrainingEvent) -> None:
        """
        Track policy update metrics.

        Captures: policy_loss, clip_fraction, gradient_norm, learning_rate,
        advantage statistics.

        Args:
            event: Training event from policy update phase
        """
        snapshot = PPOSnapshot(
            iteration=event.iteration,
            phase=TrainingPhase.POLICY_UPDATE,
            policy_loss=event.get_metric('policy_loss'),
            clip_fraction=event.get_metric('clip_fraction') if 'clip_fraction' in event.metrics else None,
            gradient_norm=event.get_metric('gradient_norm') if 'gradient_norm' in event.metrics else None,
            learning_rate=event.get_metric('learning_rate') if 'learning_rate' in event.metrics else None,
            advantage_mean=event.get_metric('advantage_mean') if 'advantage_mean' in event.metrics else None,
            advantage_std=event.get_metric('advantage_std') if 'advantage_std' in event.metrics else None,
        )
        self.ppo_history.append(snapshot)

    def _track_value_update(self, event: TrainingEvent) -> None:
        """
        Track value update metrics.

        Captures: value_loss, gradient_norm, learning_rate.

        Args:
            event: Training event from value update phase
        """
        snapshot = PPOSnapshot(
            iteration=event.iteration,
            phase=TrainingPhase.VALUE_UPDATE,
            value_loss=event.get_metric('value_loss'),
            gradient_norm=event.get_metric('gradient_norm') if 'gradient_norm' in event.metrics else None,
            learning_rate=event.get_metric('learning_rate') if 'learning_rate' in event.metrics else None,
        )
        self.ppo_history.append(snapshot)

    def get_ppo_summary(self) -> Dict[str, Any]:
        """
        Generate summary of PPO training dynamics.

        Returns:
            Dictionary with:
            - num_policy_updates: Count of policy updates
            - num_value_updates: Count of value updates
            - avg_policy_loss: Average policy loss
            - avg_value_loss: Average value loss
            - avg_clip_fraction: Average clip fraction
            - avg_gradient_norm: Average gradient norm
        """
        if not self.ppo_history:
            return {
                'num_policy_updates': 0,
                'num_value_updates': 0,
                'avg_policy_loss': 0.0,
                'avg_value_loss': 0.0,
                'avg_clip_fraction': 0.0,
                'avg_gradient_norm': 0.0,
            }

        # Separate policy and value updates
        policy_snapshots = [s for s in self.ppo_history if s.phase == TrainingPhase.POLICY_UPDATE]
        value_snapshots = [s for s in self.ppo_history if s.phase == TrainingPhase.VALUE_UPDATE]

        # Compute averages
        avg_policy_loss = self._safe_mean([s.policy_loss for s in policy_snapshots if s.policy_loss is not None])
        avg_value_loss = self._safe_mean([s.value_loss for s in value_snapshots if s.value_loss is not None])
        avg_clip_fraction = self._safe_mean([s.clip_fraction for s in policy_snapshots if s.clip_fraction is not None])

        # Gradient norms from both policy and value updates
        all_gradient_norms = [s.gradient_norm for s in self.ppo_history if s.gradient_norm is not None]
        avg_gradient_norm = self._safe_mean(all_gradient_norms)

        return {
            'num_policy_updates': len(policy_snapshots),
            'num_value_updates': len(value_snapshots),
            'avg_policy_loss': avg_policy_loss,
            'avg_value_loss': avg_value_loss,
            'avg_clip_fraction': avg_clip_fraction,
            'avg_gradient_norm': avg_gradient_norm,
        }

    def get_policy_loss_trend(self) -> Optional[float]:
        """
        Compute trend in policy loss (positive = increasing, negative = decreasing).

        Uses simple linear regression slope over policy losses.

        Returns:
            Trend value, or None if insufficient data
        """
        policy_losses = [
            s.policy_loss for s in self.ppo_history
            if s.phase == TrainingPhase.POLICY_UPDATE and s.policy_loss is not None
        ]

        if len(policy_losses) < 2:
            return None

        # Simple linear trend (slope)
        n = len(policy_losses)
        x_mean = (n - 1) / 2
        y_mean = sum(policy_losses) / n

        numerator = sum((i - x_mean) * (y - y_mean) for i, y in enumerate(policy_losses))
        denominator = sum((i - x_mean) ** 2 for i in range(n))

        if denominator == 0:
            return 0.0

        return numerator / denominator

    def get_value_loss_trend(self) -> Optional[float]:
        """
        Compute trend in value loss (positive = increasing, negative = decreasing).

        Returns:
            Trend value, or None if insufficient data
        """
        value_losses = [
            s.value_loss for s in self.ppo_history
            if s.phase == TrainingPhase.VALUE_UPDATE and s.value_loss is not None
        ]

        if len(value_losses) < 2:
            return None

        # Simple linear trend (slope)
        n = len(value_losses)
        x_mean = (n - 1) / 2
        y_mean = sum(value_losses) / n

        numerator = sum((i - x_mean) * (y - y_mean) for i, y in enumerate(value_losses))
        denominator = sum((i - x_mean) ** 2 for i in range(n))

        if denominator == 0:
            return 0.0

        return numerator / denominator

    def get_clip_fraction_stats(self) -> Dict[str, float]:
        """
        Get statistics on clip fraction values.

        Returns:
            Dictionary with mean, std, min, max of clip fractions
        """
        clip_fractions = [
            s.clip_fraction for s in self.ppo_history
            if s.clip_fraction is not None
        ]

        if not clip_fractions:
            return {'mean': 0.0, 'std': 0.0, 'min': 0.0, 'max': 0.0}

        return {
            'mean': statistics.mean(clip_fractions),
            'std': statistics.stdev(clip_fractions) if len(clip_fractions) > 1 else 0.0,
            'min': min(clip_fractions),
            'max': max(clip_fractions),
        }

    def get_advantage_stats(self) -> Dict[str, Any]:
        """
        Get aggregated advantage statistics.

        Returns:
            Dictionary with:
            - mean_of_means: Average of advantage means
            - mean_of_stds: Average of advantage standard deviations
            - history: List of (mean, std) tuples over time
        """
        advantage_data = [
            (s.advantage_mean, s.advantage_std) for s in self.ppo_history
            if s.advantage_mean is not None and s.advantage_std is not None
        ]

        if not advantage_data:
            return {
                'mean_of_means': 0.0,
                'mean_of_stds': 0.0,
                'history': [],
            }

        means = [mean for mean, _ in advantage_data]
        stds = [std for _, std in advantage_data]

        return {
            'mean_of_means': statistics.mean(means),
            'mean_of_stds': statistics.mean(stds),
            'history': advantage_data,
        }

    def get_recent_snapshots(self, n: int = 10) -> List[PPOSnapshot]:
        """
        Get the N most recent PPO snapshots.

        Args:
            n: Number of recent snapshots to return (default: 10)

        Returns:
            List of recent snapshots, most recent first
        """
        if not self.ppo_history:
            return []

        # Return last n snapshots in reverse order (most recent first)
        return list(reversed(self.ppo_history[-n:]))

    def _safe_mean(self, values: List[Optional[float]]) -> float:
        """
        Compute mean safely, handling empty lists and None values.

        Args:
            values: List of values (may contain None)

        Returns:
            Mean of non-None values, or 0.0 if no valid values
        """
        valid_values = [v for v in values if v is not None]
        if not valid_values:
            return 0.0
        return statistics.mean(valid_values)
