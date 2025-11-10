"""MODULE: test_report_generator.py
PURPOSE: Unit tests for report generation (markdown and JSON)
KEY COMPONENTS:
- Test ReportGenerator initialization
- Test markdown report generation
- Test JSON report generation
- Test report content validation
DEPENDENCIES: pytest, json, pathlib, src.training.monitoring
"""

import pytest
import json
from pathlib import Path
from src.training.monitoring.report_generator import ReportGenerator
from src.training.monitoring import (
    TrainingMonitor,
    TrainingEvent,
    TrainingPhase,
    SampleComparator,
    QualityAnalyzer,
    PPOMetricsTracker,
    VerbosityLevel,
    QualityAlert,
    AlertSeverity,
)


class TestReportGenerator:
    """Test ReportGenerator class."""

    def test_initialization(self, tmp_path):
        """Test report generator initialization."""
        report_gen = ReportGenerator(output_dir=tmp_path)
        assert report_gen.output_dir == tmp_path

    def test_initialization_creates_output_dir(self, tmp_path):
        """Test output directory is created if it doesn't exist."""
        output_dir = tmp_path / "reports"
        assert not output_dir.exists()

        report_gen = ReportGenerator(output_dir=output_dir)
        assert output_dir.exists()
        assert output_dir.is_dir()

    def test_generate_markdown_report(self, tmp_path):
        """Test markdown report generation."""
        monitor = TrainingMonitor(output_dir=tmp_path)
        comparator = SampleComparator()
        analyzer = QualityAnalyzer(monitor=monitor)

        # Add some metrics
        with monitor.monitor_context():
            for i in range(10):
                event = TrainingEvent(
                    phase=TrainingPhase.ITERATION_END,
                    iteration=i,
                    metrics={
                        'reward': float(i * 0.1),
                        'policy_loss': float(i * 0.05),
                        'kl_div': float(i * 0.001)
                    }
                )
                monitor.on_event(event)

        report_gen = ReportGenerator(output_dir=tmp_path)
        report_gen.generate_markdown_report(monitor, comparator, analyzer)

        # Check file was created
        report_path = tmp_path / "training_report.md"
        assert report_path.exists()

        # Check content
        content = report_path.read_text()
        assert "# Training Report" in content
        assert "## Summary" in content
        assert "## Metrics" in content

    def test_generate_json_report(self, tmp_path):
        """Test JSON report generation."""
        monitor = TrainingMonitor(output_dir=tmp_path)
        comparator = SampleComparator()
        analyzer = QualityAnalyzer(monitor=monitor)

        # Add metrics
        with monitor.monitor_context():
            for i in range(10):
                event = TrainingEvent(
                    phase=TrainingPhase.ITERATION_END,
                    iteration=i,
                    metrics={'reward': float(i * 0.1)}
                )
                monitor.on_event(event)

        report_gen = ReportGenerator(output_dir=tmp_path)
        report_gen.generate_json_report(monitor, comparator, analyzer)

        # Check file was created
        report_path = tmp_path / "training_report.json"
        assert report_path.exists()

        # Check valid JSON
        with open(report_path) as f:
            data = json.load(f)

        assert "summary" in data
        assert "metrics" in data
        assert "alerts" in data

    def test_markdown_report_includes_ppo_metrics(self, tmp_path):
        """Test markdown report includes PPO metrics when tracker provided."""
        monitor = TrainingMonitor(output_dir=tmp_path)
        comparator = SampleComparator()
        analyzer = QualityAnalyzer(monitor=monitor)
        ppo_tracker = PPOMetricsTracker(verbosity=VerbosityLevel.VERBOSE)

        # Add PPO events
        with monitor.monitor_context():
            for i in range(5):
                event = TrainingEvent(
                    phase=TrainingPhase.POLICY_UPDATE,
                    iteration=i,
                    metrics={
                        'policy_loss': float(i * 0.1),
                        'clip_fraction': 0.2
                    }
                )
                ppo_tracker.on_event(event)

        report_gen = ReportGenerator(output_dir=tmp_path)
        report_gen.generate_markdown_report(
            monitor, comparator, analyzer, ppo_tracker=ppo_tracker
        )

        report_path = tmp_path / "training_report.md"
        content = report_path.read_text()

        assert "## PPO Mechanics" in content
        assert "Policy Loss" in content
        assert "Clip Fraction" in content

    def test_markdown_report_early_stopping(self, tmp_path):
        """Test markdown report shows early stopping information."""
        monitor = TrainingMonitor(output_dir=tmp_path)
        comparator = SampleComparator()
        analyzer = QualityAnalyzer(monitor=monitor)

        with monitor.monitor_context():
            # Trigger early stop
            monitor.request_early_stop("Test early stop")

        report_gen = ReportGenerator(output_dir=tmp_path)
        report_gen.generate_markdown_report(monitor, comparator, analyzer)

        report_path = tmp_path / "training_report.md"
        content = report_path.read_text()

        assert "Early Stopped: Yes" in content or "early stop" in content.lower()

    def test_markdown_report_includes_alerts(self, tmp_path):
        """Test markdown report includes quality alerts."""
        monitor = TrainingMonitor(output_dir=tmp_path)
        comparator = SampleComparator()
        analyzer = QualityAnalyzer(monitor=monitor, kl_threshold=0.1)

        # Add events that trigger alerts
        with monitor.monitor_context():
            event = TrainingEvent(
                phase=TrainingPhase.ITERATION_END,
                iteration=10,
                metrics={'kl_div': 0.15}  # Above threshold
            )
            analyzer.on_event(event)

        report_gen = ReportGenerator(output_dir=tmp_path)
        report_gen.generate_markdown_report(monitor, comparator, analyzer)

        report_path = tmp_path / "training_report.md"
        content = report_path.read_text()

        assert "## Quality Alerts" in content
        assert "KL divergence" in content or "WARNING" in content

    def test_json_report_structure(self, tmp_path):
        """Test JSON report has correct structure."""
        monitor = TrainingMonitor(output_dir=tmp_path)
        comparator = SampleComparator()
        analyzer = QualityAnalyzer(monitor=monitor)

        with monitor.monitor_context():
            for i in range(5):
                monitor.on_event(TrainingEvent(
                    phase=TrainingPhase.ITERATION_END,
                    iteration=i,
                    metrics={'reward': float(i)}
                ))

        report_gen = ReportGenerator(output_dir=tmp_path)
        report_gen.generate_json_report(monitor, comparator, analyzer)

        with open(tmp_path / "training_report.json") as f:
            data = json.load(f)

        # Check structure
        assert "summary" in data
        assert "total_iterations" in data["summary"]
        assert "metrics" in data
        assert isinstance(data["metrics"], dict)
        assert "alerts" in data
        assert isinstance(data["alerts"], list)

    def test_json_report_with_ppo_tracker(self, tmp_path):
        """Test JSON report includes PPO data when provided."""
        monitor = TrainingMonitor(output_dir=tmp_path)
        comparator = SampleComparator()
        analyzer = QualityAnalyzer(monitor=monitor)
        ppo_tracker = PPOMetricsTracker(verbosity=VerbosityLevel.VERBOSE)

        # Add PPO events
        with monitor.monitor_context():
            for i in range(3):
                ppo_tracker.on_event(TrainingEvent(
                    phase=TrainingPhase.POLICY_UPDATE,
                    iteration=i,
                    metrics={'policy_loss': float(i * 0.1)}
                ))

        report_gen = ReportGenerator(output_dir=tmp_path)
        report_gen.generate_json_report(
            monitor, comparator, analyzer, ppo_tracker=ppo_tracker
        )

        with open(tmp_path / "training_report.json") as f:
            data = json.load(f)

        assert "ppo_summary" in data
        assert "num_policy_updates" in data["ppo_summary"]

    def test_markdown_report_metrics_table(self, tmp_path):
        """Test markdown report includes metrics table."""
        monitor = TrainingMonitor(output_dir=tmp_path)
        comparator = SampleComparator()
        analyzer = QualityAnalyzer(monitor=monitor)

        with monitor.monitor_context():
            for i in range(10):
                monitor.on_event(TrainingEvent(
                    phase=TrainingPhase.ITERATION_END,
                    iteration=i,
                    metrics={
                        'reward': float(i * 0.1),
                        'policy_loss': float(i * 0.05)
                    }
                ))

        report_gen = ReportGenerator(output_dir=tmp_path)
        report_gen.generate_markdown_report(monitor, comparator, analyzer)

        content = (tmp_path / "training_report.md").read_text()

        # Should have table format
        assert "|" in content  # Markdown table delimiter

    def test_generate_both_reports(self, tmp_path):
        """Test generating both markdown and JSON reports."""
        monitor = TrainingMonitor(output_dir=tmp_path)
        comparator = SampleComparator()
        analyzer = QualityAnalyzer(monitor=monitor)

        with monitor.monitor_context():
            for i in range(5):
                monitor.on_event(TrainingEvent(
                    phase=TrainingPhase.ITERATION_END,
                    iteration=i,
                    metrics={'reward': float(i)}
                ))

        report_gen = ReportGenerator(output_dir=tmp_path)
        report_gen.generate_all_reports(monitor, comparator, analyzer)

        # Both files should exist
        assert (tmp_path / "training_report.md").exists()
        assert (tmp_path / "training_report.json").exists()

    def test_report_with_sample_comparisons(self, tmp_path):
        """Test report includes sample comparison data."""
        monitor = TrainingMonitor(output_dir=tmp_path)
        comparator = SampleComparator(sample_size=2)
        analyzer = QualityAnalyzer(monitor=monitor)

        # Collect baseline
        for i in range(2):
            comparator.on_event(TrainingEvent(
                phase=TrainingPhase.RESPONSE_GENERATED,
                iteration=i,
                metadata={
                    'prompts': [f'Prompt {i}'],
                    'responses': [f'Response {i}']
                }
            ))

        report_gen = ReportGenerator(output_dir=tmp_path)
        report_gen.generate_markdown_report(monitor, comparator, analyzer)

        content = (tmp_path / "training_report.md").read_text()
        # Should mention sample comparisons
        assert "Sample" in content or "Comparison" in content or "baseline" in content.lower()

    def test_empty_report(self, tmp_path):
        """Test report generation with no training data."""
        monitor = TrainingMonitor(output_dir=tmp_path)
        comparator = SampleComparator()
        analyzer = QualityAnalyzer(monitor=monitor)

        report_gen = ReportGenerator(output_dir=tmp_path)
        report_gen.generate_markdown_report(monitor, comparator, analyzer)

        # Should still create report
        assert (tmp_path / "training_report.md").exists()

        content = (tmp_path / "training_report.md").read_text()
        assert "# Training Report" in content

    def test_json_serializable(self, tmp_path):
        """Test JSON report is properly serializable."""
        monitor = TrainingMonitor(output_dir=tmp_path)
        comparator = SampleComparator()
        analyzer = QualityAnalyzer(monitor=monitor)

        # Add complex data
        with monitor.monitor_context():
            for i in range(5):
                monitor.on_event(TrainingEvent(
                    phase=TrainingPhase.ITERATION_END,
                    iteration=i,
                    metrics={
                        'reward': float(i * 0.1),
                        'loss': float(i * 0.05)
                    }
                ))

        report_gen = ReportGenerator(output_dir=tmp_path)
        report_gen.generate_json_report(monitor, comparator, analyzer)

        # Should be valid JSON (no errors)
        with open(tmp_path / "training_report.json") as f:
            data = json.load(f)

        # Should be re-serializable
        json.dumps(data)

    def test_report_indentation(self, tmp_path):
        """Test JSON report is properly indented."""
        monitor = TrainingMonitor(output_dir=tmp_path)
        comparator = SampleComparator()
        analyzer = QualityAnalyzer(monitor=monitor)

        report_gen = ReportGenerator(output_dir=tmp_path)
        report_gen.generate_json_report(monitor, comparator, analyzer)

        content = (tmp_path / "training_report.json").read_text()

        # Should have indentation (pretty printed)
        assert "  " in content or "\t" in content


class TestReportGeneratorIntegration:
    """Integration tests for ReportGenerator."""

    def test_full_training_report(self, tmp_path):
        """Test complete training report with all components."""
        monitor = TrainingMonitor(output_dir=tmp_path)
        comparator = SampleComparator(sample_size=2)
        analyzer = QualityAnalyzer(monitor=monitor, kl_threshold=0.1)
        ppo_tracker = PPOMetricsTracker(verbosity=VerbosityLevel.VERBOSE)

        with monitor.monitor_context():
            # Collect baseline
            for i in range(2):
                event = TrainingEvent(
                    phase=TrainingPhase.RESPONSE_GENERATED,
                    iteration=i,
                    metadata={
                        'prompts': [f'Prompt {i}'],
                        'responses': [f'Response {i}']
                    }
                )
                comparator.on_event(event)

            # Training iterations
            for i in range(20):
                # Iteration start
                monitor.on_event(TrainingEvent(
                    phase=TrainingPhase.ITERATION_START,
                    iteration=i
                ))

                # Policy update
                policy_event = TrainingEvent(
                    phase=TrainingPhase.POLICY_UPDATE,
                    iteration=i,
                    metrics={
                        'policy_loss': float(i * 0.1),
                        'clip_fraction': 0.2
                    }
                )
                ppo_tracker.on_event(policy_event)
                monitor.on_event(policy_event)

                # Value update
                value_event = TrainingEvent(
                    phase=TrainingPhase.VALUE_UPDATE,
                    iteration=i,
                    metrics={'value_loss': float(i * 0.05)}
                )
                ppo_tracker.on_event(value_event)
                monitor.on_event(value_event)

                # Iteration end
                end_event = TrainingEvent(
                    phase=TrainingPhase.ITERATION_END,
                    iteration=i,
                    metrics={
                        'reward': float(i * 0.1),
                        'kl_div': float(i * 0.01)
                    }
                )
                analyzer.on_event(end_event)
                monitor.on_event(end_event)

        # Generate reports
        report_gen = ReportGenerator(output_dir=tmp_path)
        report_gen.generate_all_reports(
            monitor, comparator, analyzer, ppo_tracker=ppo_tracker
        )

        # Verify both reports exist
        assert (tmp_path / "training_report.md").exists()
        assert (tmp_path / "training_report.json").exists()

        # Verify markdown content
        md_content = (tmp_path / "training_report.md").read_text()
        assert "# Training Report" in md_content
        assert "PPO Mechanics" in md_content

        # Verify JSON structure
        with open(tmp_path / "training_report.json") as f:
            json_data = json.load(f)

        assert json_data["summary"]["total_iterations"] == 20
        assert "ppo_summary" in json_data
        assert len(json_data["alerts"]) > 0  # Should have KL alerts
