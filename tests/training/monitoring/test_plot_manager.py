"""MODULE: test_plot_manager.py
PURPOSE: Unit tests for visualization and plotting
KEY COMPONENTS:
- Test PlotManager initialization
- Test plot generation methods
- Test file output
- Test matplotlib integration
DEPENDENCIES: pytest, unittest.mock, pathlib, src.training.monitoring
"""

import pytest
from pathlib import Path
from unittest.mock import Mock, MagicMock, patch, call
from src.training.monitoring.plot_manager import PlotManager
from src.training.monitoring import (
    MetricsStore,
    SampleComparator,
    QualityAnalyzer,
    TrainingMonitor,
)


class TestPlotManager:
    """Test PlotManager class."""

    def test_initialization(self, tmp_path):
        """Test plot manager initialization."""
        plot_manager = PlotManager(output_dir=tmp_path)
        assert plot_manager.output_dir == tmp_path
        assert isinstance(plot_manager.figures, dict)
        assert len(plot_manager.figures) == 0

    def test_initialization_creates_output_dir(self, tmp_path):
        """Test output directory is created if it doesn't exist."""
        output_dir = tmp_path / "plots"
        assert not output_dir.exists()

        plot_manager = PlotManager(output_dir=output_dir)
        assert output_dir.exists()
        assert output_dir.is_dir()

    @patch('src.training.monitoring.plot_manager.plt')
    def test_create_loss_plot(self, mock_plt, tmp_path):
        """Test loss plot creation."""
        metrics_store = MetricsStore()
        # Add some sample data
        for i in range(10):
            metrics_store.record('policy_loss', float(i))
            metrics_store.record('value_loss', float(i * 0.5))

        plot_manager = PlotManager(output_dir=tmp_path)

        # Mock figure and axes
        mock_fig = MagicMock()
        mock_axes = [MagicMock(), MagicMock(), MagicMock()]
        mock_plt.subplots.return_value = (mock_fig, mock_axes)

        plot_manager.create_loss_plot(metrics_store)

        # Verify plt.subplots was called
        mock_plt.subplots.assert_called_once()
        # Verify figure was saved
        mock_fig.savefig.assert_called_once()
        # Verify save path
        save_path = mock_fig.savefig.call_args[0][0]
        assert str(save_path).endswith('loss_curves.png')

    @patch('src.training.monitoring.plot_manager.plt')
    def test_create_loss_plot_empty_data(self, mock_plt, tmp_path):
        """Test loss plot with no data."""
        metrics_store = MetricsStore()
        plot_manager = PlotManager(output_dir=tmp_path)

        # Mock figure
        mock_fig = MagicMock()
        mock_axes = [MagicMock(), MagicMock(), MagicMock()]
        mock_plt.subplots.return_value = (mock_fig, mock_axes)

        # Should not crash with empty data
        plot_manager.create_loss_plot(metrics_store)

        # Should still save figure (even if empty)
        mock_fig.savefig.assert_called_once()

    @patch('src.training.monitoring.plot_manager.plt')
    def test_create_reward_plot(self, mock_plt, tmp_path):
        """Test reward plot creation."""
        metrics_store = MetricsStore()
        for i in range(10):
            metrics_store.record('reward', float(i * 0.1))

        plot_manager = PlotManager(output_dir=tmp_path)

        mock_fig = MagicMock()
        mock_ax = MagicMock()
        mock_plt.subplots.return_value = (mock_fig, mock_ax)

        plot_manager.create_reward_plot(metrics_store)

        mock_plt.subplots.assert_called_once()
        mock_fig.savefig.assert_called_once()
        save_path = mock_fig.savefig.call_args[0][0]
        assert str(save_path).endswith('rewards.png')

    @patch('src.training.monitoring.plot_manager.plt')
    def test_create_kl_plot(self, mock_plt, tmp_path):
        """Test KL divergence plot creation."""
        metrics_store = MetricsStore()
        for i in range(10):
            metrics_store.record('kl_div', float(i * 0.01))

        plot_manager = PlotManager(output_dir=tmp_path)

        mock_fig = MagicMock()
        mock_ax = MagicMock()
        mock_plt.subplots.return_value = (mock_fig, mock_ax)

        plot_manager.create_kl_plot(metrics_store, kl_threshold=0.1)

        mock_plt.subplots.assert_called_once()
        mock_fig.savefig.assert_called_once()
        save_path = mock_fig.savefig.call_args[0][0]
        assert str(save_path).endswith('kl_divergence.png')

    @patch('src.training.monitoring.plot_manager.plt')
    def test_create_quality_plot(self, mock_plt, tmp_path):
        """Test quality metrics plot creation."""
        monitor = TrainingMonitor()
        comparator = SampleComparator()
        analyzer = QualityAnalyzer(monitor=monitor)

        plot_manager = PlotManager(output_dir=tmp_path)

        mock_fig = MagicMock()
        mock_axes = [MagicMock(), MagicMock()]
        mock_plt.subplots.return_value = (mock_fig, mock_axes)

        plot_manager.create_quality_plot(comparator, analyzer)

        mock_plt.subplots.assert_called_once()
        mock_fig.savefig.assert_called_once()
        save_path = mock_fig.savefig.call_args[0][0]
        assert str(save_path).endswith('quality_metrics.png')

    @patch('src.training.monitoring.plot_manager.plt')
    def test_create_ppo_plot(self, mock_plt, tmp_path):
        """Test PPO mechanics plot creation."""
        from src.training.monitoring import PPOMetricsTracker, VerbosityLevel

        ppo_tracker = PPOMetricsTracker(verbosity=VerbosityLevel.VERBOSE)

        plot_manager = PlotManager(output_dir=tmp_path)

        mock_fig = MagicMock()
        # Create a custom mock that supports both axes[0][0] and axes[0, 0] indexing
        mock_ax = MagicMock()

        class MockAxes:
            def __getitem__(self, key):
                if isinstance(key, tuple):
                    # Handle axes[0, 0] style
                    return mock_ax
                else:
                    # Handle axes[0] style - return self for chaining
                    return self

        mock_axes = MockAxes()
        mock_plt.subplots.return_value = (mock_fig, mock_axes)

        plot_manager.create_ppo_plot(ppo_tracker)

        mock_plt.subplots.assert_called_once()
        mock_fig.savefig.assert_called_once()
        save_path = mock_fig.savefig.call_args[0][0]
        assert str(save_path).endswith('ppo_mechanics.png')

    @patch('src.training.monitoring.plot_manager.plt')
    def test_create_all_plots(self, mock_plt, tmp_path):
        """Test creating all plots at once."""
        metrics_store = MetricsStore()
        for i in range(10):
            metrics_store.record('policy_loss', float(i))
            metrics_store.record('reward', float(i * 0.1))
            metrics_store.record('kl_div', float(i * 0.01))

        monitor = TrainingMonitor()
        comparator = SampleComparator()
        analyzer = QualityAnalyzer(monitor=monitor)

        plot_manager = PlotManager(output_dir=tmp_path)

        # Mock all figures
        mock_fig = MagicMock()
        mock_ax = MagicMock()
        mock_axes = [MagicMock(), MagicMock(), MagicMock()]
        mock_plt.subplots.return_value = (mock_fig, mock_ax)

        # Override for loss plot
        def subplots_side_effect(*args, **kwargs):
            nrows = kwargs.get('nrows', 1)
            ncols = kwargs.get('ncols', 1)
            if nrows == 1 and ncols == 3:
                return (mock_fig, mock_axes)
            elif nrows == 1 and ncols == 2:
                return (mock_fig, [MagicMock(), MagicMock()])
            else:
                return (mock_fig, mock_ax)

        mock_plt.subplots.side_effect = subplots_side_effect

        plot_manager.create_all_plots(
            metrics_store=metrics_store,
            comparator=comparator,
            analyzer=analyzer
        )

        # Should call subplots multiple times (once per plot)
        assert mock_plt.subplots.call_count >= 4

    def test_figures_stored(self, tmp_path):
        """Test that figures are stored in figures dict."""
        plot_manager = PlotManager(output_dir=tmp_path)

        with patch('src.training.monitoring.plot_manager.plt') as mock_plt:
            mock_fig = MagicMock()
            mock_ax = MagicMock()
            mock_plt.subplots.return_value = (mock_fig, mock_ax)

            metrics_store = MetricsStore()
            plot_manager.create_reward_plot(metrics_store)

            # Figure should be stored
            assert 'rewards' in plot_manager.figures
            assert plot_manager.figures['rewards'] is mock_fig

    @patch('src.training.monitoring.plot_manager.plt')
    def test_close_all_figures(self, mock_plt, tmp_path):
        """Test closing all matplotlib figures."""
        plot_manager = PlotManager(output_dir=tmp_path)

        # Create some figures
        mock_fig = MagicMock()
        mock_ax = MagicMock()
        mock_plt.subplots.return_value = (mock_fig, mock_ax)

        metrics_store = MetricsStore()
        plot_manager.create_reward_plot(metrics_store)

        # Close all
        plot_manager.close_all()

        # Should call plt.close('all')
        mock_plt.close.assert_called_with('all')
        # Should clear figures dict
        assert len(plot_manager.figures) == 0

    @patch('src.training.monitoring.plot_manager.plt')
    def test_plot_with_style(self, mock_plt, tmp_path):
        """Test that plots use proper matplotlib style."""
        plot_manager = PlotManager(output_dir=tmp_path, style='seaborn-v0_8-darkgrid')

        metrics_store = MetricsStore()
        for i in range(10):
            metrics_store.record('reward', float(i))

        mock_fig = MagicMock()
        mock_ax = MagicMock()
        mock_plt.subplots.return_value = (mock_fig, mock_ax)

        plot_manager.create_reward_plot(metrics_store)

        # Should apply style
        mock_plt.style.use.assert_called_with('seaborn-v0_8-darkgrid')

    @patch('src.training.monitoring.plot_manager.plt')
    def test_plot_dpi_setting(self, mock_plt, tmp_path):
        """Test DPI setting for saved plots."""
        plot_manager = PlotManager(output_dir=tmp_path, dpi=300)

        metrics_store = MetricsStore()
        mock_fig = MagicMock()
        mock_ax = MagicMock()
        mock_plt.subplots.return_value = (mock_fig, mock_ax)

        plot_manager.create_reward_plot(metrics_store)

        # Check that savefig was called with correct DPI
        mock_fig.savefig.assert_called_once()
        call_kwargs = mock_fig.savefig.call_args[1]
        assert call_kwargs.get('dpi') == 300

    def test_invalid_output_dir_raises(self):
        """Test that invalid output directory raises error."""
        # This should work - it creates the directory
        plot_manager = PlotManager(output_dir=Path('/tmp/test_plots'))
        assert plot_manager.output_dir == Path('/tmp/test_plots')


class TestPlotManagerIntegration:
    """Integration tests for PlotManager with real data."""

    @patch('src.training.monitoring.plot_manager.plt')
    def test_full_workflow(self, mock_plt, tmp_path):
        """Test complete plotting workflow."""
        # Create monitor with real data
        monitor = TrainingMonitor()
        comparator = SampleComparator()
        analyzer = QualityAnalyzer(monitor=monitor)

        # Add metrics
        metrics_store = monitor.get_metrics_store()
        for i in range(20):
            metrics_store.record('policy_loss', float(i * 0.1))
            metrics_store.record('value_loss', float(i * 0.05))
            metrics_store.record('reward', float(i * 0.2))
            metrics_store.record('kl_div', float(i * 0.001))

        # Create plots
        plot_manager = PlotManager(output_dir=tmp_path)

        # Mock matplotlib
        mock_fig = MagicMock()
        mock_ax = MagicMock()
        mock_axes = [MagicMock(), MagicMock(), MagicMock()]

        def subplots_side_effect(*args, **kwargs):
            ncols = kwargs.get('ncols', 1)
            if ncols == 3:
                return (mock_fig, mock_axes)
            else:
                return (mock_fig, mock_ax)

        mock_plt.subplots.side_effect = subplots_side_effect

        plot_manager.create_all_plots(
            metrics_store=metrics_store,
            comparator=comparator,
            analyzer=analyzer
        )

        # Should have created multiple plots
        assert mock_plt.subplots.call_count >= 4
