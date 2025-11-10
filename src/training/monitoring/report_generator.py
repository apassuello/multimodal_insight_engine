"""MODULE: report_generator.py
PURPOSE: Generate comprehensive training reports in markdown and JSON formats
KEY COMPONENTS:
- ReportGenerator: Creates markdown and JSON reports from training data
DEPENDENCIES: json, pathlib, typing, datetime, training monitor components
SPECIAL NOTES: Generates human-readable markdown and machine-readable JSON
"""

import json
from pathlib import Path
from typing import Dict, Any, Optional, List
from datetime import datetime

from .training_monitor import TrainingMonitor
from .sample_comparator import SampleComparator
from .quality_analyzer import QualityAnalyzer, AlertSeverity


class ReportGenerator:
    """
    Generates comprehensive training reports.

    Features:
    - Markdown reports with metrics, plots, alerts
    - JSON reports with structured data
    - Summary statistics
    - Quality alerts
    - Sample comparisons
    - PPO mechanics (when available)

    Design: Aggregates data from monitor components and formats into reports.

    Attributes:
        output_dir: Directory to save reports
    """
    __slots__ = ('output_dir',)

    def __init__(self, output_dir: Path):
        """
        Initialize report generator.

        Args:
            output_dir: Directory to save reports
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def generate_markdown_report(
        self,
        monitor: TrainingMonitor,
        comparator: SampleComparator,
        analyzer: QualityAnalyzer,
        ppo_tracker: Optional[Any] = None
    ) -> None:
        """
        Generate markdown report.

        Creates training_report.md with:
        - Summary statistics
        - Metrics table
        - Quality alerts
        - Sample changes
        - PPO mechanics (if tracker provided)

        Args:
            monitor: TrainingMonitor instance
            comparator: SampleComparator instance
            analyzer: QualityAnalyzer instance
            ppo_tracker: Optional PPOMetricsTracker instance
        """
        lines = []

        # Header
        lines.append("# Training Report")
        lines.append("")
        lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append("")

        # Summary
        lines.append("## Summary")
        lines.append("")

        metrics_store = monitor.get_metrics_store()
        metric_names = metrics_store.get_metric_names()

        # Count iterations (use reward metric if available, otherwise any metric)
        total_iterations = 0
        if metric_names:
            first_metric = list(metric_names)[0]
            buffer = metrics_store.get_buffer(first_metric)
            total_iterations = len(buffer._buffer)

        lines.append(f"- **Total Iterations**: {total_iterations}")

        # Early stopping
        should_stop, stop_reason = monitor.should_stop_early()
        early_stopped = "Yes" if should_stop else "No"
        lines.append(f"- **Early Stopped**: {early_stopped}")
        if should_stop:
            lines.append(f"  - Reason: {stop_reason}")

        lines.append("")

        # Metrics
        lines.append("## Metrics")
        lines.append("")

        if metric_names:
            lines.append("| Metric | Mean | Std | Min | Max | Trend |")
            lines.append("|--------|------|-----|-----|-----|-------|")

            for metric_name in sorted(metric_names):
                stats = metrics_store.get_statistics(metric_name)
                trend = metrics_store.compute_trend(metric_name)

                trend_str = "↑" if trend and trend > 0.001 else "↓" if trend and trend < -0.001 else "→"

                lines.append(
                    f"| {metric_name} | {stats['mean']:.4f} | {stats['std']:.4f} | "
                    f"{stats['min']:.4f} | {stats['max']:.4f} | {trend_str} |"
                )

            lines.append("")
        else:
            lines.append("*No metrics recorded*")
            lines.append("")

        # Quality Alerts
        lines.append("## Quality Alerts")
        lines.append("")

        if analyzer.alert_history:
            alert_summary = analyzer.get_alert_summary()
            lines.append(f"- **Total Alerts**: {alert_summary['total_alerts']}")
            lines.append(f"  - INFO: {alert_summary['info_count']}")
            lines.append(f"  - WARNING: {alert_summary['warning_count']}")
            lines.append(f"  - CRITICAL: {alert_summary['critical_count']}")
            lines.append("")

            # Recent alerts
            recent_alerts = analyzer.get_recent_alerts(n=10)
            if recent_alerts:
                lines.append("### Recent Alerts")
                lines.append("")
                for alert in recent_alerts:
                    severity_icon = "🔴" if alert.is_critical else "🟡" if alert.severity == AlertSeverity.WARNING else "🔵"
                    lines.append(f"- {severity_icon} **Iteration {alert.iteration}**: {alert.reason}")

                lines.append("")
        else:
            lines.append("*No alerts triggered*")
            lines.append("")

        # Sample Comparisons
        lines.append("## Sample Comparisons")
        lines.append("")

        if comparator.comparison_history:
            summary = comparator.get_comparison_summary()
            lines.append(f"- **Total Comparisons**: {summary['num_comparisons']}")
            lines.append(f"- **Average Similarity**: {summary['avg_similarity']:.4f}")
            lines.append(f"- **Degraded Samples**: {summary['degraded_count']}")
            lines.append("")

            catastrophic_forgetting = comparator.detect_catastrophic_forgetting()
            if catastrophic_forgetting:
                lines.append("⚠️ **Catastrophic forgetting detected!**")
                lines.append("")
        elif comparator.baseline_collected:
            lines.append(f"*Baseline collected ({len(comparator.baseline_samples)} samples), no comparisons yet*")
            lines.append("")
        else:
            lines.append("*No baseline samples collected*")
            lines.append("")

        # PPO Mechanics
        if ppo_tracker is not None:
            lines.append("## PPO Mechanics")
            lines.append("")

            ppo_summary = ppo_tracker.get_ppo_summary()
            lines.append(f"- **Policy Updates**: {ppo_summary['num_policy_updates']}")
            lines.append(f"- **Value Updates**: {ppo_summary['num_value_updates']}")
            lines.append(f"- **Avg Policy Loss**: {ppo_summary['avg_policy_loss']:.4f}")
            lines.append(f"- **Avg Value Loss**: {ppo_summary['avg_value_loss']:.4f}")
            lines.append(f"- **Avg Clip Fraction**: {ppo_summary['avg_clip_fraction']:.4f}")
            lines.append(f"- **Avg Gradient Norm**: {ppo_summary['avg_gradient_norm']:.4f}")
            lines.append("")

            # Trends
            policy_trend = ppo_tracker.get_policy_loss_trend()
            value_trend = ppo_tracker.get_value_loss_trend()

            if policy_trend is not None:
                trend_str = "improving" if policy_trend < 0 else "degrading"
                lines.append(f"- **Policy Loss Trend**: {trend_str} (slope: {policy_trend:.6f})")

            if value_trend is not None:
                trend_str = "improving" if value_trend < 0 else "degrading"
                lines.append(f"- **Value Loss Trend**: {trend_str} (slope: {value_trend:.6f})")

            lines.append("")

        # Write report
        report_path = self.output_dir / "training_report.md"
        report_path.write_text("\n".join(lines))

    def generate_json_report(
        self,
        monitor: TrainingMonitor,
        comparator: SampleComparator,
        analyzer: QualityAnalyzer,
        ppo_tracker: Optional[Any] = None
    ) -> None:
        """
        Generate JSON report with structured data.

        Creates training_report.json with:
        {
          "summary": {...},
          "metrics": {...},
          "alerts": [...],
          "sample_comparisons": {...},
          "ppo_summary": {...}  // if tracker provided
        }

        Args:
            monitor: TrainingMonitor instance
            comparator: SampleComparator instance
            analyzer: QualityAnalyzer instance
            ppo_tracker: Optional PPOMetricsTracker instance
        """
        data: Dict[str, Any] = {}

        # Summary
        metrics_store = monitor.get_metrics_store()
        metric_names = metrics_store.get_metric_names()

        total_iterations = 0
        if metric_names:
            first_metric = list(metric_names)[0]
            buffer = metrics_store.get_buffer(first_metric)
            total_iterations = len(buffer._buffer)

        should_stop, stop_reason = monitor.should_stop_early()

        data["summary"] = {
            "total_iterations": total_iterations,
            "early_stopped": should_stop,
            "stop_reason": stop_reason if should_stop else None,
            "generated_at": datetime.now().isoformat()
        }

        # Metrics
        data["metrics"] = {}
        for metric_name in metric_names:
            stats = metrics_store.get_statistics(metric_name)
            trend = metrics_store.compute_trend(metric_name)

            data["metrics"][metric_name] = {
                "mean": float(stats["mean"]),
                "std": float(stats["std"]),
                "min": float(stats["min"]),
                "max": float(stats["max"]),
                "count": int(stats["count"]),
                "trend": float(trend) if trend is not None else None
            }

        # Alerts
        data["alerts"] = []
        for alert in analyzer.alert_history:
            data["alerts"].append({
                "iteration": alert.iteration,
                "severity": alert.severity.name,
                "reason": alert.reason,
                "metric_name": alert.metric_name,
                "metric_value": alert.metric_value
            })

        # Sample comparisons
        comparison_summary = comparator.get_comparison_summary()
        data["sample_comparisons"] = {
            "num_comparisons": comparison_summary["num_comparisons"],
            "avg_similarity": comparison_summary["avg_similarity"],
            "min_similarity": comparison_summary["min_similarity"],
            "max_similarity": comparison_summary["max_similarity"],
            "degraded_count": comparison_summary["degraded_count"],
            "catastrophic_forgetting": comparator.detect_catastrophic_forgetting()
        }

        # PPO summary
        if ppo_tracker is not None:
            ppo_summary = ppo_tracker.get_ppo_summary()
            policy_trend = ppo_tracker.get_policy_loss_trend()
            value_trend = ppo_tracker.get_value_loss_trend()

            data["ppo_summary"] = {
                "num_policy_updates": ppo_summary["num_policy_updates"],
                "num_value_updates": ppo_summary["num_value_updates"],
                "avg_policy_loss": ppo_summary["avg_policy_loss"],
                "avg_value_loss": ppo_summary["avg_value_loss"],
                "avg_clip_fraction": ppo_summary["avg_clip_fraction"],
                "avg_gradient_norm": ppo_summary["avg_gradient_norm"],
                "policy_loss_trend": policy_trend,
                "value_loss_trend": value_trend
            }

        # Write report
        report_path = self.output_dir / "training_report.json"
        report_path.write_text(json.dumps(data, indent=2))

    def generate_all_reports(
        self,
        monitor: TrainingMonitor,
        comparator: SampleComparator,
        analyzer: QualityAnalyzer,
        ppo_tracker: Optional[Any] = None
    ) -> None:
        """
        Generate both markdown and JSON reports.

        Args:
            monitor: TrainingMonitor instance
            comparator: SampleComparator instance
            analyzer: QualityAnalyzer instance
            ppo_tracker: Optional PPOMetricsTracker instance
        """
        self.generate_markdown_report(monitor, comparator, analyzer, ppo_tracker)
        self.generate_json_report(monitor, comparator, analyzer, ppo_tracker)
