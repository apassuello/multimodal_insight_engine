"""MODULE: quality_analyzer.py
PURPOSE: Detect quality degradation and trigger early stopping
KEY COMPONENTS:
- AlertSeverity: Enum for alert severity levels
- QualityAlert: Dataclass for alert records
- QualityAnalyzer: Callback that monitors training quality
DEPENDENCIES: dataclasses, enum, typing, events, callbacks, metrics, sample_comparator
SPECIAL NOTES: Detects reward hacking, KL divergence issues, triggers early stopping
"""

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Dict, List, Any, Optional
from .events import TrainingEvent, TrainingPhase


class AlertSeverity(Enum):
    """
    Severity levels for quality alerts.

    Levels:
    - INFO: Informational, no action needed
    - WARNING: Potential issue, monitor closely
    - CRITICAL: Serious issue, triggers early stopping
    """
    INFO = auto()
    WARNING = auto()
    CRITICAL = auto()


@dataclass(frozen=True, slots=True)
class QualityAlert:
    """
    Record of a quality issue detected during training.

    Design: Frozen dataclass for immutability and memory efficiency.

    Attributes:
        iteration: Training iteration when alert triggered
        severity: Alert severity level
        reason: Human-readable description of the issue
        metric_name: Name of metric that triggered alert (optional)
        metric_value: Value of metric that triggered alert (optional)
    """
    iteration: int
    severity: AlertSeverity
    reason: str
    metric_name: Optional[str] = None
    metric_value: Optional[float] = None

    @property
    def is_critical(self) -> bool:
        """Check if this is a critical alert."""
        return self.severity == AlertSeverity.CRITICAL


class QualityAnalyzer:
    """
    Monitors training quality and detects degradation.

    Features:
    - Detects reward hacking (reward increases, quality decreases)
    - Monitors KL divergence for policy drift
    - Tracks alert history
    - Triggers early stopping on critical issues

    Design: Implements TrainingCallback protocol for event-driven monitoring.

    Attributes:
        monitor: TrainingMonitor instance for early stopping requests
        comparator: Optional SampleComparator for quality tracking
        reward_hack_threshold: Similarity threshold for reward hacking detection
        kl_threshold: KL divergence threshold for warnings
        window_size: Number of recent iterations to analyze for trends
        alert_history: List of all alerts triggered
    """
    __slots__ = (
        'monitor',
        'comparator',
        'reward_hack_threshold',
        'kl_threshold',
        'window_size',
        'alert_history',
    )

    def __init__(
        self,
        monitor: 'TrainingMonitor',
        comparator: Optional['SampleComparator'] = None,
        reward_hack_threshold: float = 0.3,
        kl_threshold: float = 0.1,
        window_size: int = 50,
    ):
        """
        Initialize quality analyzer.

        Args:
            monitor: TrainingMonitor for accessing metrics and early stopping
            comparator: Optional SampleComparator for quality vs reward analysis
            reward_hack_threshold: Degradation threshold for reward hacking (default: 0.3)
            kl_threshold: KL divergence threshold for warnings (default: 0.1)
            window_size: Window size for trend analysis (default: 50)
        """
        self.monitor = monitor
        self.comparator = comparator
        self.reward_hack_threshold = reward_hack_threshold
        self.kl_threshold = kl_threshold
        self.window_size = window_size
        self.alert_history: List[QualityAlert] = []

    def on_event(self, event: TrainingEvent) -> None:
        """
        Handle training events to detect quality issues.

        Monitors ITERATION_END events for quality degradation signals.

        Args:
            event: Training event with metrics and metadata
        """
        # Only monitor at iteration end
        if event.phase != TrainingPhase.ITERATION_END:
            return

        # Check KL divergence
        self._check_kl_divergence(event)

        # Check reward hacking (if comparator available)
        if self.comparator is not None:
            self._check_reward_hacking(event)

    def _check_kl_divergence(self, event: TrainingEvent) -> None:
        """
        Check if KL divergence exceeds thresholds.

        Thresholds:
        - WARNING: kl_div > threshold
        - CRITICAL: kl_div > 2 * threshold (triggers early stop)

        Args:
            event: Training event with KL divergence metric
        """
        kl_div = event.get_metric('kl_div', 0.0)

        if kl_div == 0.0:
            # No KL divergence metric available
            return

        if kl_div > 2 * self.kl_threshold:
            # Critical: KL divergence very high
            alert = QualityAlert(
                iteration=event.iteration,
                severity=AlertSeverity.CRITICAL,
                reason=f"KL divergence critically high: {kl_div:.4f} (threshold: {self.kl_threshold:.4f})",
                metric_name='kl_div',
                metric_value=kl_div,
            )
            self.alert_history.append(alert)

            # Request early stopping
            self.monitor.request_early_stop(alert.reason)

        elif kl_div > self.kl_threshold:
            # Warning: KL divergence above threshold
            alert = QualityAlert(
                iteration=event.iteration,
                severity=AlertSeverity.WARNING,
                reason=f"KL divergence high: {kl_div:.4f} (threshold: {self.kl_threshold:.4f})",
                metric_name='kl_div',
                metric_value=kl_div,
            )
            self.alert_history.append(alert)

    def _check_reward_hacking(self, event: TrainingEvent) -> None:
        """
        Detect reward hacking by comparing reward and quality trends.

        Reward hacking occurs when:
        - Reward trend is increasing
        - Sample quality (from comparator) is decreasing

        Args:
            event: Training event with reward metric
        """
        if self.comparator is None or len(self.comparator.comparison_history) < 3:
            # Need comparator and some comparison history
            return

        # Get recent reward trend from metrics store
        metrics_store = self.monitor.get_metrics_store()
        reward_trend = metrics_store.compute_trend('reward', window=self.window_size)

        if reward_trend is None or reward_trend <= 0:
            # No increasing reward trend
            return

        # Check if quality is degrading
        recent_comparisons = self.comparator.comparison_history[-self.window_size:]
        degraded_count = sum(
            1 for comp in recent_comparisons
            if comp.is_degraded(self.reward_hack_threshold)
        )

        degradation_ratio = degraded_count / len(recent_comparisons)

        if degradation_ratio > 0.5:
            # More than 50% of samples degraded while reward increasing
            alert = QualityAlert(
                iteration=event.iteration,
                severity=AlertSeverity.CRITICAL,
                reason=(
                    f"Potential reward hacking detected: "
                    f"reward trend +{reward_trend:.3f}, "
                    f"quality degradation {degradation_ratio:.1%}"
                ),
                metric_name='reward',
                metric_value=event.get_metric('reward', 0.0),
            )
            self.alert_history.append(alert)

            # Request early stopping for reward hacking
            self.monitor.request_early_stop(alert.reason)

    def get_alert_summary(self) -> Dict[str, Any]:
        """
        Generate summary statistics for alerts.

        Returns:
            Dictionary with:
            - total_alerts: Total number of alerts
            - info_count: Number of INFO alerts
            - warning_count: Number of WARNING alerts
            - critical_count: Number of CRITICAL alerts
            - recent_alerts: List of recent alert summaries
        """
        if not self.alert_history:
            return {
                'total_alerts': 0,
                'info_count': 0,
                'warning_count': 0,
                'critical_count': 0,
                'recent_alerts': [],
            }

        # Count by severity
        info_count = sum(1 for a in self.alert_history if a.severity == AlertSeverity.INFO)
        warning_count = sum(1 for a in self.alert_history if a.severity == AlertSeverity.WARNING)
        critical_count = sum(1 for a in self.alert_history if a.severity == AlertSeverity.CRITICAL)

        # Get recent alerts (last 10)
        recent = self.get_recent_alerts(n=10)
        recent_summaries = [
            {
                'iteration': alert.iteration,
                'severity': alert.severity.name,
                'reason': alert.reason,
            }
            for alert in recent
        ]

        return {
            'total_alerts': len(self.alert_history),
            'info_count': info_count,
            'warning_count': warning_count,
            'critical_count': critical_count,
            'recent_alerts': recent_summaries,
        }

    def get_recent_alerts(self, n: int = 10) -> List[QualityAlert]:
        """
        Get the N most recent alerts.

        Args:
            n: Number of recent alerts to return (default: 10)

        Returns:
            List of recent alerts, most recent first
        """
        if not self.alert_history:
            return []

        # Return last n alerts in reverse order (most recent first)
        return list(reversed(self.alert_history[-n:]))
