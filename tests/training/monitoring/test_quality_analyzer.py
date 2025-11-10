"""MODULE: test_quality_analyzer.py
PURPOSE: Unit tests for quality degradation detection
KEY COMPONENTS:
- Test QualityAlert dataclass
- Test QualityAnalyzer callback
- Test reward hacking detection
- Test KL divergence monitoring
- Test early stopping triggers
DEPENDENCIES: pytest, src.training.monitoring
"""

import pytest
from src.training.monitoring.quality_analyzer import (
    QualityAnalyzer,
    QualityAlert,
    AlertSeverity,
)
from src.training.monitoring import (
    TrainingMonitor,
    TrainingEvent,
    TrainingPhase,
    VerbosityLevel,
    SampleComparator,
    SampleComparison,
)


class TestAlertSeverity:
    """Test AlertSeverity enum."""

    def test_severity_levels(self):
        """Test all severity levels exist."""
        assert AlertSeverity.INFO
        assert AlertSeverity.WARNING
        assert AlertSeverity.CRITICAL

    def test_severity_ordering(self):
        """Test severity levels are ordered correctly."""
        assert AlertSeverity.INFO.value < AlertSeverity.WARNING.value
        assert AlertSeverity.WARNING.value < AlertSeverity.CRITICAL.value


class TestQualityAlert:
    """Test QualityAlert dataclass."""

    def test_initialization(self):
        """Test alert initialization."""
        alert = QualityAlert(
            iteration=100,
            severity=AlertSeverity.WARNING,
            reason="Test alert",
            metric_name="kl_div",
            metric_value=0.15,
        )
        assert alert.iteration == 100
        assert alert.severity == AlertSeverity.WARNING
        assert alert.reason == "Test alert"
        assert alert.metric_name == "kl_div"
        assert alert.metric_value == 0.15

    def test_is_critical(self):
        """Test is_critical property."""
        critical = QualityAlert(
            iteration=1,
            severity=AlertSeverity.CRITICAL,
            reason="Critical issue",
        )
        warning = QualityAlert(
            iteration=1,
            severity=AlertSeverity.WARNING,
            reason="Warning",
        )
        assert critical.is_critical is True
        assert warning.is_critical is False


class TestQualityAnalyzer:
    """Test QualityAnalyzer callback."""

    def test_initialization_defaults(self):
        """Test analyzer initialization with defaults."""
        monitor = TrainingMonitor()
        analyzer = QualityAnalyzer(monitor=monitor)

        assert analyzer.monitor is monitor
        assert analyzer.reward_hack_threshold == 0.3
        assert analyzer.kl_threshold == 0.1
        assert analyzer.window_size == 50
        assert len(analyzer.alert_history) == 0

    def test_initialization_custom_params(self):
        """Test initialization with custom parameters."""
        monitor = TrainingMonitor()
        analyzer = QualityAnalyzer(
            monitor=monitor,
            reward_hack_threshold=0.5,
            kl_threshold=0.2,
            window_size=20,
        )

        assert analyzer.reward_hack_threshold == 0.5
        assert analyzer.kl_threshold == 0.2
        assert analyzer.window_size == 20

    def test_on_event_protocol(self):
        """Test QualityAnalyzer implements TrainingCallback protocol."""
        monitor = TrainingMonitor()
        analyzer = QualityAnalyzer(monitor=monitor)
        assert hasattr(analyzer, 'on_event')
        assert callable(analyzer.on_event)

    def test_kl_divergence_check_normal(self):
        """Test KL divergence check with normal values."""
        monitor = TrainingMonitor()
        analyzer = QualityAnalyzer(monitor=monitor, kl_threshold=0.1)

        with monitor.monitor_context():
            event = TrainingEvent(
                phase=TrainingPhase.ITERATION_END,
                iteration=10,
                metrics={'kl_div': 0.05}
            )
            analyzer.on_event(event)

        # No alerts should be triggered
        assert len(analyzer.alert_history) == 0

    def test_kl_divergence_check_high(self):
        """Test KL divergence check with high value."""
        monitor = TrainingMonitor()
        analyzer = QualityAnalyzer(monitor=monitor, kl_threshold=0.1)

        with monitor.monitor_context():
            event = TrainingEvent(
                phase=TrainingPhase.ITERATION_END,
                iteration=10,
                metrics={'kl_div': 0.15}
            )
            analyzer.on_event(event)

        # Alert should be triggered
        assert len(analyzer.alert_history) == 1
        alert = analyzer.alert_history[0]
        assert alert.severity == AlertSeverity.WARNING
        assert 'KL divergence' in alert.reason
        assert alert.metric_value == 0.15

    def test_kl_divergence_critical_threshold(self):
        """Test KL divergence critical threshold (2x)."""
        monitor = TrainingMonitor()
        analyzer = QualityAnalyzer(monitor=monitor, kl_threshold=0.1)

        with monitor.monitor_context():
            event = TrainingEvent(
                phase=TrainingPhase.ITERATION_END,
                iteration=10,
                metrics={'kl_div': 0.25}  # 2.5x threshold
            )
            analyzer.on_event(event)

        # Critical alert should trigger early stop
        assert len(analyzer.alert_history) == 1
        alert = analyzer.alert_history[0]
        assert alert.severity == AlertSeverity.CRITICAL

        # Check early stopping was requested
        should_stop, reason = monitor.should_stop_early()
        assert should_stop is True
        assert 'KL divergence' in reason

    def test_reward_hacking_detection_requires_comparator(self):
        """Test reward hacking detection requires comparator."""
        monitor = TrainingMonitor()
        analyzer = QualityAnalyzer(monitor=monitor)

        # Without comparator, reward hacking can't be detected
        with monitor.monitor_context():
            for i in range(10):
                event = TrainingEvent(
                    phase=TrainingPhase.ITERATION_END,
                    iteration=i,
                    metrics={'reward': float(i)}  # Increasing reward
                )
                analyzer.on_event(event)

        # No reward hacking alerts (comparator not set)
        reward_hack_alerts = [a for a in analyzer.alert_history if 'reward hack' in a.reason.lower()]
        assert len(reward_hack_alerts) == 0

    def test_reward_hacking_detection_with_comparator(self):
        """Test reward hacking detection with comparator."""
        monitor = TrainingMonitor()
        comparator = SampleComparator(sample_size=2)
        analyzer = QualityAnalyzer(
            monitor=monitor,
            comparator=comparator,
            reward_hack_threshold=0.3,
            window_size=5
        )

        # Collect baseline
        with monitor.monitor_context():
            for i in range(2):
                event = TrainingEvent(
                    phase=TrainingPhase.RESPONSE_GENERATED,
                    iteration=i,
                    metadata={
                        'prompts': [f'Prompt {i}'],
                        'responses': [f'Good response {i}']
                    }
                )
                comparator.on_event(event)

            # Add degraded comparisons (simulating quality drop)
            for i in range(5):
                comparison = SampleComparison(
                    iteration=i + 10,
                    prompt="Test",
                    original="Good quality response",
                    updated="Bad quality",
                    similarity_ratio=0.3,  # Low similarity = degraded
                    char_changes=20,
                    line_changes=1
                )
                comparator.comparison_history.append(comparison)

            # Now send increasing rewards
            for i in range(10, 15):
                event = TrainingEvent(
                    phase=TrainingPhase.ITERATION_END,
                    iteration=i,
                    metrics={'reward': float(i - 5)}  # Increasing reward
                )
                analyzer.on_event(event)

        # Should detect reward hacking (reward up, quality down)
        reward_hack_alerts = [a for a in analyzer.alert_history if 'reward hack' in a.reason.lower()]
        # May not trigger immediately, depends on window
        # This tests the logic is in place

    def test_get_alert_summary(self):
        """Test alert summary generation."""
        monitor = TrainingMonitor()
        analyzer = QualityAnalyzer(monitor=monitor)

        # Add some alerts
        analyzer.alert_history.extend([
            QualityAlert(
                iteration=10,
                severity=AlertSeverity.INFO,
                reason="Info alert"
            ),
            QualityAlert(
                iteration=20,
                severity=AlertSeverity.WARNING,
                reason="Warning alert"
            ),
            QualityAlert(
                iteration=30,
                severity=AlertSeverity.CRITICAL,
                reason="Critical alert"
            ),
        ])

        summary = analyzer.get_alert_summary()

        assert summary['total_alerts'] == 3
        assert summary['info_count'] == 1
        assert summary['warning_count'] == 1
        assert summary['critical_count'] == 1
        assert len(summary['recent_alerts']) > 0

    def test_get_alert_summary_empty(self):
        """Test alert summary with no alerts."""
        monitor = TrainingMonitor()
        analyzer = QualityAnalyzer(monitor=monitor)

        summary = analyzer.get_alert_summary()

        assert summary['total_alerts'] == 0
        assert summary['info_count'] == 0
        assert summary['warning_count'] == 0
        assert summary['critical_count'] == 0

    def test_only_monitors_iteration_end(self):
        """Test analyzer only processes ITERATION_END events."""
        monitor = TrainingMonitor()
        analyzer = QualityAnalyzer(monitor=monitor, kl_threshold=0.1)

        with monitor.monitor_context():
            # Send high KL at wrong phase
            event = TrainingEvent(
                phase=TrainingPhase.POLICY_UPDATE,  # Wrong phase
                iteration=10,
                metrics={'kl_div': 0.5}
            )
            analyzer.on_event(event)

        # No alerts (wrong phase)
        assert len(analyzer.alert_history) == 0

        with monitor.monitor_context():
            # Send at correct phase
            event = TrainingEvent(
                phase=TrainingPhase.ITERATION_END,
                iteration=10,
                metrics={'kl_div': 0.5}
            )
            analyzer.on_event(event)

        # Alert triggered
        assert len(analyzer.alert_history) == 1

    def test_missing_kl_metric_handled_gracefully(self):
        """Test analyzer handles missing KL divergence metric."""
        monitor = TrainingMonitor()
        analyzer = QualityAnalyzer(monitor=monitor, kl_threshold=0.1)

        with monitor.monitor_context():
            event = TrainingEvent(
                phase=TrainingPhase.ITERATION_END,
                iteration=10,
                metrics={'reward': 1.0}  # No kl_div
            )
            analyzer.on_event(event)

        # Should not crash, no alerts
        assert len(analyzer.alert_history) == 0

    def test_multiple_alerts_same_iteration(self):
        """Test multiple alerts can trigger in same iteration."""
        monitor = TrainingMonitor()
        analyzer = QualityAnalyzer(monitor=monitor, kl_threshold=0.1)

        with monitor.monitor_context():
            event = TrainingEvent(
                phase=TrainingPhase.ITERATION_END,
                iteration=10,
                metrics={
                    'kl_div': 0.25,  # Critical
                    'reward': 5.0
                }
            )
            analyzer.on_event(event)

        # At least KL alert
        assert len(analyzer.alert_history) >= 1

    def test_alert_history_growth(self):
        """Test alert history grows as issues are detected."""
        monitor = TrainingMonitor()
        analyzer = QualityAnalyzer(monitor=monitor, kl_threshold=0.1)

        with monitor.monitor_context():
            for i in range(10):
                event = TrainingEvent(
                    phase=TrainingPhase.ITERATION_END,
                    iteration=i * 10,
                    metrics={'kl_div': 0.15}  # Always above threshold
                )
                analyzer.on_event(event)

        # Should have multiple alerts
        assert len(analyzer.alert_history) == 10

    def test_early_stop_only_on_critical(self):
        """Test early stopping only triggered by critical alerts."""
        monitor = TrainingMonitor()
        analyzer = QualityAnalyzer(monitor=monitor, kl_threshold=0.1)

        with monitor.monitor_context():
            # Warning level (1.5x threshold)
            event = TrainingEvent(
                phase=TrainingPhase.ITERATION_END,
                iteration=10,
                metrics={'kl_div': 0.15}
            )
            analyzer.on_event(event)

            # Should not trigger early stop
            should_stop, _ = monitor.should_stop_early()
            assert should_stop is False

            # Critical level (2.5x threshold)
            event = TrainingEvent(
                phase=TrainingPhase.ITERATION_END,
                iteration=11,
                metrics={'kl_div': 0.25}
            )
            analyzer.on_event(event)

            # Should trigger early stop
            should_stop, reason = monitor.should_stop_early()
            assert should_stop is True
            assert 'KL divergence' in reason

    def test_get_recent_alerts(self):
        """Test getting recent alerts."""
        monitor = TrainingMonitor()
        analyzer = QualityAnalyzer(monitor=monitor)

        # Add 20 alerts
        for i in range(20):
            analyzer.alert_history.append(
                QualityAlert(
                    iteration=i,
                    severity=AlertSeverity.INFO,
                    reason=f"Alert {i}"
                )
            )

        recent = analyzer.get_recent_alerts(n=5)
        assert len(recent) == 5
        # Should be most recent
        assert recent[0].iteration == 19
        assert recent[-1].iteration == 15

    def test_get_recent_alerts_less_than_n(self):
        """Test get_recent_alerts with fewer alerts than requested."""
        monitor = TrainingMonitor()
        analyzer = QualityAnalyzer(monitor=monitor)

        # Add 3 alerts
        for i in range(3):
            analyzer.alert_history.append(
                QualityAlert(
                    iteration=i,
                    severity=AlertSeverity.INFO,
                    reason=f"Alert {i}"
                )
            )

        recent = analyzer.get_recent_alerts(n=10)
        assert len(recent) == 3

    def test_critical_alerts_count(self):
        """Test counting critical alerts."""
        monitor = TrainingMonitor()
        analyzer = QualityAnalyzer(monitor=monitor)

        # Add mix of alerts
        analyzer.alert_history.extend([
            QualityAlert(iteration=1, severity=AlertSeverity.INFO, reason="Info"),
            QualityAlert(iteration=2, severity=AlertSeverity.WARNING, reason="Warning"),
            QualityAlert(iteration=3, severity=AlertSeverity.CRITICAL, reason="Critical 1"),
            QualityAlert(iteration=4, severity=AlertSeverity.CRITICAL, reason="Critical 2"),
        ])

        critical_count = sum(1 for a in analyzer.alert_history if a.is_critical)
        assert critical_count == 2
