"""MODULE: test_ppo_metrics_tracker.py
PURPOSE: Unit tests for PPO-specific metrics tracking
KEY COMPONENTS:
- Test PPOMetricsTracker callback
- Test policy update tracking
- Test value update tracking
- Test PPO summary generation
- Test verbosity control
DEPENDENCIES: pytest, src.training.monitoring
"""

import pytest
from src.training.monitoring.ppo_metrics_tracker import (
    PPOMetricsTracker,
    PPOSnapshot,
)
from src.training.monitoring import (
    TrainingEvent,
    TrainingPhase,
    VerbosityLevel,
)


class TestPPOSnapshot:
    """Test PPOSnapshot dataclass."""

    def test_initialization_policy(self):
        """Test snapshot initialization for policy update."""
        snapshot = PPOSnapshot(
            iteration=10,
            phase=TrainingPhase.POLICY_UPDATE,
            policy_loss=0.5,
            clip_fraction=0.3,
            gradient_norm=2.5,
        )
        assert snapshot.iteration == 10
        assert snapshot.phase == TrainingPhase.POLICY_UPDATE
        assert snapshot.policy_loss == 0.5
        assert snapshot.clip_fraction == 0.3
        assert snapshot.gradient_norm == 2.5

    def test_initialization_value(self):
        """Test snapshot initialization for value update."""
        snapshot = PPOSnapshot(
            iteration=10,
            phase=TrainingPhase.VALUE_UPDATE,
            value_loss=0.3,
            gradient_norm=1.5,
        )
        assert snapshot.iteration == 10
        assert snapshot.phase == TrainingPhase.VALUE_UPDATE
        assert snapshot.value_loss == 0.3
        assert snapshot.gradient_norm == 1.5


class TestPPOMetricsTracker:
    """Test PPOMetricsTracker callback."""

    def test_initialization_verbose(self):
        """Test tracker initialization in verbose mode."""
        tracker = PPOMetricsTracker(verbosity=VerbosityLevel.VERBOSE)
        assert tracker.verbosity == VerbosityLevel.VERBOSE
        assert len(tracker.ppo_history) == 0
        assert tracker.is_enabled is True

    def test_initialization_simple(self):
        """Test tracker initialization in simple mode."""
        tracker = PPOMetricsTracker(verbosity=VerbosityLevel.SIMPLE)
        assert tracker.verbosity == VerbosityLevel.SIMPLE
        assert tracker.is_enabled is False

    def test_on_event_protocol(self):
        """Test that PPOMetricsTracker implements TrainingCallback protocol."""
        tracker = PPOMetricsTracker(verbosity=VerbosityLevel.VERBOSE)
        assert hasattr(tracker, 'on_event')
        assert callable(tracker.on_event)

    def test_disabled_in_simple_mode(self):
        """Test tracker is disabled in SIMPLE mode."""
        tracker = PPOMetricsTracker(verbosity=VerbosityLevel.SIMPLE)

        # Send policy update event
        event = TrainingEvent(
            phase=TrainingPhase.POLICY_UPDATE,
            iteration=10,
            metrics={'policy_loss': 0.5}
        )
        tracker.on_event(event)

        # Should not record anything
        assert len(tracker.ppo_history) == 0

    def test_tracks_policy_update(self):
        """Test tracking policy update events."""
        tracker = PPOMetricsTracker(verbosity=VerbosityLevel.VERBOSE)

        event = TrainingEvent(
            phase=TrainingPhase.POLICY_UPDATE,
            iteration=10,
            metrics={
                'policy_loss': 0.5,
                'clip_fraction': 0.3,
                'gradient_norm': 2.5,
                'learning_rate': 3e-4,
            }
        )
        tracker.on_event(event)

        assert len(tracker.ppo_history) == 1
        snapshot = tracker.ppo_history[0]
        assert snapshot.iteration == 10
        assert snapshot.phase == TrainingPhase.POLICY_UPDATE
        assert snapshot.policy_loss == 0.5
        assert snapshot.clip_fraction == 0.3
        assert snapshot.gradient_norm == 2.5
        assert snapshot.learning_rate == 3e-4

    def test_tracks_value_update(self):
        """Test tracking value update events."""
        tracker = PPOMetricsTracker(verbosity=VerbosityLevel.VERBOSE)

        event = TrainingEvent(
            phase=TrainingPhase.VALUE_UPDATE,
            iteration=10,
            metrics={
                'value_loss': 0.3,
                'gradient_norm': 1.5,
                'learning_rate': 3e-4,
            }
        )
        tracker.on_event(event)

        assert len(tracker.ppo_history) == 1
        snapshot = tracker.ppo_history[0]
        assert snapshot.iteration == 10
        assert snapshot.phase == TrainingPhase.VALUE_UPDATE
        assert snapshot.value_loss == 0.3
        assert snapshot.gradient_norm == 1.5

    def test_tracks_advantage_statistics(self):
        """Test tracking advantage statistics."""
        tracker = PPOMetricsTracker(verbosity=VerbosityLevel.VERBOSE)

        event = TrainingEvent(
            phase=TrainingPhase.POLICY_UPDATE,
            iteration=10,
            metrics={
                'policy_loss': 0.5,
                'advantage_mean': 0.1,
                'advantage_std': 0.5,
            }
        )
        tracker.on_event(event)

        snapshot = tracker.ppo_history[0]
        assert snapshot.advantage_mean == 0.1
        assert snapshot.advantage_std == 0.5

    def test_ignores_other_phases(self):
        """Test tracker ignores events from other phases."""
        tracker = PPOMetricsTracker(verbosity=VerbosityLevel.VERBOSE)

        # Send events from non-PPO phases
        phases = [
            TrainingPhase.ITERATION_START,
            TrainingPhase.ITERATION_END,
            TrainingPhase.RESPONSE_GENERATED,
            TrainingPhase.REWARD_COMPUTED,
        ]

        for phase in phases:
            event = TrainingEvent(
                phase=phase,
                iteration=10,
                metrics={'some_metric': 1.0}
            )
            tracker.on_event(event)

        # Should not record anything
        assert len(tracker.ppo_history) == 0

    def test_multiple_updates_tracked(self):
        """Test multiple updates are tracked in history."""
        tracker = PPOMetricsTracker(verbosity=VerbosityLevel.VERBOSE)

        # Send 10 policy updates
        for i in range(10):
            event = TrainingEvent(
                phase=TrainingPhase.POLICY_UPDATE,
                iteration=i,
                metrics={'policy_loss': float(i) * 0.1}
            )
            tracker.on_event(event)

        assert len(tracker.ppo_history) == 10
        assert tracker.ppo_history[0].policy_loss == 0.0
        assert tracker.ppo_history[9].policy_loss == 0.9

    def test_get_ppo_summary_empty(self):
        """Test PPO summary with no history."""
        tracker = PPOMetricsTracker(verbosity=VerbosityLevel.VERBOSE)
        summary = tracker.get_ppo_summary()

        assert summary['num_policy_updates'] == 0
        assert summary['num_value_updates'] == 0
        assert summary['avg_clip_fraction'] == 0.0
        assert summary['avg_policy_loss'] == 0.0
        assert summary['avg_value_loss'] == 0.0

    def test_get_ppo_summary_with_data(self):
        """Test PPO summary with tracking data."""
        tracker = PPOMetricsTracker(verbosity=VerbosityLevel.VERBOSE)

        # Add policy updates
        for i in range(5):
            event = TrainingEvent(
                phase=TrainingPhase.POLICY_UPDATE,
                iteration=i,
                metrics={
                    'policy_loss': 0.5,
                    'clip_fraction': 0.3,
                }
            )
            tracker.on_event(event)

        # Add value updates
        for i in range(3):
            event = TrainingEvent(
                phase=TrainingPhase.VALUE_UPDATE,
                iteration=i,
                metrics={'value_loss': 0.2}
            )
            tracker.on_event(event)

        summary = tracker.get_ppo_summary()

        assert summary['num_policy_updates'] == 5
        assert summary['num_value_updates'] == 3
        assert summary['avg_clip_fraction'] == 0.3
        assert summary['avg_policy_loss'] == 0.5
        assert summary['avg_value_loss'] == 0.2

    def test_handles_missing_metrics(self):
        """Test graceful handling of missing metrics."""
        tracker = PPOMetricsTracker(verbosity=VerbosityLevel.VERBOSE)

        # Send event without optional metrics
        event = TrainingEvent(
            phase=TrainingPhase.POLICY_UPDATE,
            iteration=10,
            metrics={'policy_loss': 0.5}  # Missing clip_fraction, gradient_norm
        )
        tracker.on_event(event)

        snapshot = tracker.ppo_history[0]
        assert snapshot.policy_loss == 0.5
        assert snapshot.clip_fraction is None
        assert snapshot.gradient_norm is None

    def test_get_policy_loss_trend(self):
        """Test computing policy loss trend."""
        tracker = PPOMetricsTracker(verbosity=VerbosityLevel.VERBOSE)

        # Add increasing policy losses
        for i in range(10):
            event = TrainingEvent(
                phase=TrainingPhase.POLICY_UPDATE,
                iteration=i,
                metrics={'policy_loss': float(i) * 0.1}
            )
            tracker.on_event(event)

        trend = tracker.get_policy_loss_trend()
        assert trend is not None
        assert trend > 0  # Increasing trend

    def test_get_value_loss_trend(self):
        """Test computing value loss trend."""
        tracker = PPOMetricsTracker(verbosity=VerbosityLevel.VERBOSE)

        # Add decreasing value losses
        for i in range(10):
            event = TrainingEvent(
                phase=TrainingPhase.VALUE_UPDATE,
                iteration=i,
                metrics={'value_loss': 1.0 - float(i) * 0.05}
            )
            tracker.on_event(event)

        trend = tracker.get_value_loss_trend()
        assert trend is not None
        assert trend < 0  # Decreasing trend

    def test_get_clip_fraction_stats(self):
        """Test clip fraction statistics."""
        tracker = PPOMetricsTracker(verbosity=VerbosityLevel.VERBOSE)

        # Add various clip fractions
        clip_fractions = [0.1, 0.2, 0.3, 0.4, 0.5]
        for i, cf in enumerate(clip_fractions):
            event = TrainingEvent(
                phase=TrainingPhase.POLICY_UPDATE,
                iteration=i,
                metrics={
                    'policy_loss': 0.5,
                    'clip_fraction': cf,
                }
            )
            tracker.on_event(event)

        stats = tracker.get_clip_fraction_stats()
        assert stats['mean'] == 0.3  # Mean of [0.1, 0.2, 0.3, 0.4, 0.5]
        assert stats['min'] == 0.1
        assert stats['max'] == 0.5
        assert 'std' in stats

    def test_get_advantage_stats(self):
        """Test advantage statistics aggregation."""
        tracker = PPOMetricsTracker(verbosity=VerbosityLevel.VERBOSE)

        # Add advantage statistics
        for i in range(5):
            event = TrainingEvent(
                phase=TrainingPhase.POLICY_UPDATE,
                iteration=i,
                metrics={
                    'policy_loss': 0.5,
                    'advantage_mean': 0.1 + i * 0.05,
                    'advantage_std': 0.5,
                }
            )
            tracker.on_event(event)

        stats = tracker.get_advantage_stats()
        assert 'mean_of_means' in stats
        assert 'mean_of_stds' in stats
        assert len(stats['history']) == 5

    def test_recent_snapshots(self):
        """Test getting recent snapshots."""
        tracker = PPOMetricsTracker(verbosity=VerbosityLevel.VERBOSE)

        # Add 20 snapshots
        for i in range(20):
            event = TrainingEvent(
                phase=TrainingPhase.POLICY_UPDATE,
                iteration=i,
                metrics={'policy_loss': float(i)}
            )
            tracker.on_event(event)

        recent = tracker.get_recent_snapshots(n=5)
        assert len(recent) == 5
        # Should be most recent
        assert recent[0].iteration == 19
        assert recent[-1].iteration == 15

    def test_recent_snapshots_less_than_n(self):
        """Test get_recent_snapshots with fewer snapshots than requested."""
        tracker = PPOMetricsTracker(verbosity=VerbosityLevel.VERBOSE)

        # Add 3 snapshots
        for i in range(3):
            event = TrainingEvent(
                phase=TrainingPhase.POLICY_UPDATE,
                iteration=i,
                metrics={'policy_loss': float(i)}
            )
            tracker.on_event(event)

        recent = tracker.get_recent_snapshots(n=10)
        assert len(recent) == 3

    def test_mixed_policy_and_value_updates(self):
        """Test tracking both policy and value updates."""
        tracker = PPOMetricsTracker(verbosity=VerbosityLevel.VERBOSE)

        # Alternate policy and value updates
        for i in range(5):
            # Policy update
            event = TrainingEvent(
                phase=TrainingPhase.POLICY_UPDATE,
                iteration=i * 2,
                metrics={'policy_loss': 0.5}
            )
            tracker.on_event(event)

            # Value update
            event = TrainingEvent(
                phase=TrainingPhase.VALUE_UPDATE,
                iteration=i * 2 + 1,
                metrics={'value_loss': 0.3}
            )
            tracker.on_event(event)

        assert len(tracker.ppo_history) == 10

        # Check separation in summary
        summary = tracker.get_ppo_summary()
        assert summary['num_policy_updates'] == 5
        assert summary['num_value_updates'] == 5

    def test_gradient_norm_tracking(self):
        """Test gradient norm is tracked."""
        tracker = PPOMetricsTracker(verbosity=VerbosityLevel.VERBOSE)

        event = TrainingEvent(
            phase=TrainingPhase.POLICY_UPDATE,
            iteration=10,
            metrics={
                'policy_loss': 0.5,
                'gradient_norm': 2.5,
            }
        )
        tracker.on_event(event)

        snapshot = tracker.ppo_history[0]
        assert snapshot.gradient_norm == 2.5

    def test_learning_rate_tracking(self):
        """Test learning rate is tracked."""
        tracker = PPOMetricsTracker(verbosity=VerbosityLevel.VERBOSE)

        event = TrainingEvent(
            phase=TrainingPhase.POLICY_UPDATE,
            iteration=10,
            metrics={
                'policy_loss': 0.5,
                'learning_rate': 3e-4,
            }
        )
        tracker.on_event(event)

        snapshot = tracker.ppo_history[0]
        assert snapshot.learning_rate == 3e-4
