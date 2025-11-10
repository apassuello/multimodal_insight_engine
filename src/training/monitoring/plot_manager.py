"""MODULE: plot_manager.py
PURPOSE: Generate matplotlib visualizations for training metrics
KEY COMPONENTS:
- PlotManager: Creates training visualization plots
DEPENDENCIES: matplotlib, pathlib, typing, numpy, metrics, quality components
SPECIAL NOTES: Uses matplotlib with style control and configurable DPI
"""

from pathlib import Path
from typing import Dict, Any, Optional
import numpy as np

# Try to import matplotlib, but allow for mocking during tests
try:
    import matplotlib
    matplotlib.use('Agg')  # Use non-interactive backend
    import matplotlib.pyplot as plt
except ImportError:
    plt = None  # Will be set by tests or remain None

from .metrics import MetricsStore
from .sample_comparator import SampleComparator
from .quality_analyzer import QualityAnalyzer


class PlotManager:
    """
    Creates training visualization plots for monitoring system.

    Features:
    - Loss curves (policy, value, total)
    - Reward progression with trend lines
    - KL divergence monitoring with threshold lines
    - Quality metrics (sample similarity, alerts)
    - PPO mechanics (clip fraction, advantages)
    - Configurable DPI and style

    Design: Creates matplotlib figures and saves them to output directory.

    Attributes:
        output_dir: Directory to save plots
        figures: Dictionary storing matplotlib figure objects
        style: Matplotlib style to use
        dpi: DPI for saved figures
    """
    __slots__ = ('output_dir', 'figures', 'style', 'dpi')

    def __init__(
        self,
        output_dir: Path,
        style: str = 'seaborn-v0_8-whitegrid',
        dpi: int = 150
    ):
        """
        Initialize plot manager.

        Args:
            output_dir: Directory to save plots
            style: Matplotlib style (default: seaborn-v0_8-whitegrid)
            dpi: DPI for saved figures (default: 150)
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.figures: Dict[str, Any] = {}
        self.style = style
        self.dpi = dpi

        # Warn if matplotlib not available (but tests can mock it)
        if plt is None:
            import warnings
            warnings.warn("Matplotlib not available. Plotting disabled.")

    def create_loss_plot(self, metrics_store: MetricsStore) -> None:
        """
        Create loss curves plot.

        Plots:
        - Policy loss
        - Value loss
        - Total loss (if available)

        Saves to: {output_dir}/loss_curves.png

        Args:
            metrics_store: MetricsStore with loss metrics
        """
        if plt is None:
            return

        plt.style.use(self.style)

        fig, axes = plt.subplots(1, 3, figsize=(15, 4))

        # Policy loss
        if metrics_store.has_metric('policy_loss'):
            policy_loss = metrics_store.get_buffer('policy_loss').to_array()
            axes[0].plot(policy_loss, label='Policy Loss', color='#2E86AB')
            axes[0].set_title('Policy Loss')
            axes[0].set_xlabel('Iteration')
            axes[0].set_ylabel('Loss')
            axes[0].grid(True, alpha=0.3)
        else:
            axes[0].text(0.5, 0.5, 'No policy loss data', ha='center', va='center')
            axes[0].set_title('Policy Loss')

        # Value loss
        if metrics_store.has_metric('value_loss'):
            value_loss = metrics_store.get_buffer('value_loss').to_array()
            axes[1].plot(value_loss, label='Value Loss', color='#A23B72')
            axes[1].set_title('Value Loss')
            axes[1].set_xlabel('Iteration')
            axes[1].set_ylabel('Loss')
            axes[1].grid(True, alpha=0.3)
        else:
            axes[1].text(0.5, 0.5, 'No value loss data', ha='center', va='center')
            axes[1].set_title('Value Loss')

        # Total loss
        if metrics_store.has_metric('total_loss'):
            total_loss = metrics_store.get_buffer('total_loss').to_array()
            axes[2].plot(total_loss, label='Total Loss', color='#F18F01')
            axes[2].set_title('Total Loss')
            axes[2].set_xlabel('Iteration')
            axes[2].set_ylabel('Loss')
            axes[2].grid(True, alpha=0.3)
        else:
            axes[2].text(0.5, 0.5, 'No total loss data', ha='center', va='center')
            axes[2].set_title('Total Loss')

        plt.tight_layout()
        save_path = self.output_dir / 'loss_curves.png'
        fig.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
        self.figures['loss_curves'] = fig

    def create_reward_plot(self, metrics_store: MetricsStore) -> None:
        """
        Plot reward progression with trend line.

        Saves to: {output_dir}/rewards.png

        Args:
            metrics_store: MetricsStore with reward metrics
        """
        if plt is None:
            return

        plt.style.use(self.style)

        fig, ax = plt.subplots(1, 1, figsize=(10, 5))

        if metrics_store.has_metric('reward'):
            rewards = metrics_store.get_buffer('reward').to_array()
            iterations = np.arange(len(rewards))

            # Plot rewards
            ax.plot(iterations, rewards, label='Reward', color='#06A77D', alpha=0.7)

            # Add trend line if enough data
            if len(rewards) >= 2:
                z = np.polyfit(iterations, rewards, 1)
                p = np.poly1d(z)
                ax.plot(iterations, p(iterations), '--',
                       label=f'Trend (slope={z[0]:.4f})',
                       color='#D62246', linewidth=2)

            ax.set_title('Reward Progression')
            ax.set_xlabel('Iteration')
            ax.set_ylabel('Reward')
            ax.legend()
            ax.grid(True, alpha=0.3)
        else:
            ax.text(0.5, 0.5, 'No reward data', ha='center', va='center',
                   transform=ax.transAxes)
            ax.set_title('Reward Progression')

        plt.tight_layout()
        save_path = self.output_dir / 'rewards.png'
        fig.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
        self.figures['rewards'] = fig

    def create_kl_plot(
        self,
        metrics_store: MetricsStore,
        kl_threshold: float = 0.1
    ) -> None:
        """
        Plot KL divergence with warning threshold.

        Saves to: {output_dir}/kl_divergence.png

        Args:
            metrics_store: MetricsStore with kl_div metrics
            kl_threshold: Warning threshold to display
        """
        if plt is None:
            return

        plt.style.use(self.style)

        fig, ax = plt.subplots(1, 1, figsize=(10, 5))

        if metrics_store.has_metric('kl_div'):
            kl_div = metrics_store.get_buffer('kl_div').to_array()
            iterations = np.arange(len(kl_div))

            ax.plot(iterations, kl_div, label='KL Divergence', color='#118AB2')

            # Add threshold lines
            ax.axhline(y=kl_threshold, color='#FFB703', linestyle='--',
                      label=f'Warning Threshold ({kl_threshold})')
            ax.axhline(y=kl_threshold * 2, color='#FB8500', linestyle='--',
                      label=f'Critical Threshold ({kl_threshold * 2})')

            ax.set_title('KL Divergence Monitoring')
            ax.set_xlabel('Iteration')
            ax.set_ylabel('KL Divergence')
            ax.legend()
            ax.grid(True, alpha=0.3)
        else:
            ax.text(0.5, 0.5, 'No KL divergence data', ha='center', va='center',
                   transform=ax.transAxes)
            ax.set_title('KL Divergence Monitoring')

        plt.tight_layout()
        save_path = self.output_dir / 'kl_divergence.png'
        fig.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
        self.figures['kl_divergence'] = fig

    def create_quality_plot(
        self,
        comparator: SampleComparator,
        analyzer: QualityAnalyzer
    ) -> None:
        """
        Plot quality metrics and alerts.

        Plots:
        - Sample similarity over time
        - Alert timeline

        Saves to: {output_dir}/quality_metrics.png

        Args:
            comparator: SampleComparator with comparison history
            analyzer: QualityAnalyzer with alert history
        """
        if plt is None:
            return

        plt.style.use(self.style)

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # Sample similarity
        if comparator.comparison_history:
            iterations = [c.iteration for c in comparator.comparison_history]
            similarities = [c.similarity_ratio for c in comparator.comparison_history]

            axes[0].plot(iterations, similarities, 'o-', label='Similarity',
                        color='#06A77D', alpha=0.7)
            axes[0].axhline(y=comparator.degradation_threshold, color='#FFB703',
                           linestyle='--', label=f'Threshold ({comparator.degradation_threshold})')
            axes[0].set_title('Sample Similarity Over Time')
            axes[0].set_xlabel('Iteration')
            axes[0].set_ylabel('Similarity Ratio')
            axes[0].legend()
            axes[0].grid(True, alpha=0.3)
        else:
            axes[0].text(0.5, 0.5, 'No comparison data', ha='center', va='center',
                        transform=axes[0].transAxes)
            axes[0].set_title('Sample Similarity Over Time')

        # Alert timeline
        if analyzer.alert_history:
            from .quality_analyzer import AlertSeverity

            iterations = [a.iteration for a in analyzer.alert_history]
            severities = [a.severity.value for a in analyzer.alert_history]

            # Color by severity
            colors = []
            for a in analyzer.alert_history:
                if a.severity == AlertSeverity.CRITICAL:
                    colors.append('#D62246')
                elif a.severity == AlertSeverity.WARNING:
                    colors.append('#FFB703')
                else:
                    colors.append('#118AB2')

            axes[1].scatter(iterations, severities, c=colors, s=100, alpha=0.7)
            axes[1].set_title('Quality Alerts Timeline')
            axes[1].set_xlabel('Iteration')
            axes[1].set_ylabel('Severity')
            axes[1].set_yticks([1, 2, 3])
            axes[1].set_yticklabels(['INFO', 'WARNING', 'CRITICAL'])
            axes[1].grid(True, alpha=0.3)
        else:
            axes[1].text(0.5, 0.5, 'No alerts', ha='center', va='center',
                        transform=axes[1].transAxes)
            axes[1].set_title('Quality Alerts Timeline')

        plt.tight_layout()
        save_path = self.output_dir / 'quality_metrics.png'
        fig.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
        self.figures['quality_metrics'] = fig

    def create_ppo_plot(self, ppo_tracker) -> None:
        """
        Plot PPO-specific mechanics.

        Plots:
        - Clip fraction over time
        - Advantage statistics
        - Gradient norms
        - Learning rate schedule

        Saves to: {output_dir}/ppo_mechanics.png

        Args:
            ppo_tracker: PPOMetricsTracker instance
        """
        if plt is None:
            return

        plt.style.use(self.style)

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        # Clip fraction
        clip_fractions = [s.clip_fraction for s in ppo_tracker.ppo_history
                         if s.clip_fraction is not None]
        if clip_fractions:
            iterations = list(range(len(clip_fractions)))
            axes[0, 0].plot(iterations, clip_fractions, color='#2E86AB')
            axes[0, 0].set_title('Clip Fraction')
            axes[0, 0].set_xlabel('Update')
            axes[0, 0].set_ylabel('Clip Fraction')
            axes[0, 0].grid(True, alpha=0.3)
        else:
            axes[0, 0].text(0.5, 0.5, 'No clip fraction data',
                           ha='center', va='center', transform=axes[0, 0].transAxes)
            axes[0, 0].set_title('Clip Fraction')

        # Advantage mean
        advantage_means = [s.advantage_mean for s in ppo_tracker.ppo_history
                          if s.advantage_mean is not None]
        if advantage_means:
            iterations = list(range(len(advantage_means)))
            axes[0, 1].plot(iterations, advantage_means, color='#A23B72')
            axes[0, 1].axhline(y=0, color='black', linestyle='--', alpha=0.3)
            axes[0, 1].set_title('Advantage Mean')
            axes[0, 1].set_xlabel('Update')
            axes[0, 1].set_ylabel('Advantage')
            axes[0, 1].grid(True, alpha=0.3)
        else:
            axes[0, 1].text(0.5, 0.5, 'No advantage data',
                           ha='center', va='center', transform=axes[0, 1].transAxes)
            axes[0, 1].set_title('Advantage Mean')

        # Gradient norms
        gradient_norms = [s.gradient_norm for s in ppo_tracker.ppo_history
                         if s.gradient_norm is not None]
        if gradient_norms:
            iterations = list(range(len(gradient_norms)))
            axes[1, 0].plot(iterations, gradient_norms, color='#F18F01')
            axes[1, 0].set_title('Gradient Norm')
            axes[1, 0].set_xlabel('Update')
            axes[1, 0].set_ylabel('L2 Norm')
            axes[1, 0].grid(True, alpha=0.3)
        else:
            axes[1, 0].text(0.5, 0.5, 'No gradient norm data',
                           ha='center', va='center', transform=axes[1, 0].transAxes)
            axes[1, 0].set_title('Gradient Norm')

        # Learning rate
        learning_rates = [s.learning_rate for s in ppo_tracker.ppo_history
                         if s.learning_rate is not None]
        if learning_rates:
            iterations = list(range(len(learning_rates)))
            axes[1, 1].plot(iterations, learning_rates, color='#06A77D')
            axes[1, 1].set_title('Learning Rate Schedule')
            axes[1, 1].set_xlabel('Update')
            axes[1, 1].set_ylabel('Learning Rate')
            axes[1, 1].grid(True, alpha=0.3)
        else:
            axes[1, 1].text(0.5, 0.5, 'No learning rate data',
                           ha='center', va='center', transform=axes[1, 1].transAxes)
            axes[1, 1].set_title('Learning Rate Schedule')

        plt.tight_layout()
        save_path = self.output_dir / 'ppo_mechanics.png'
        fig.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
        self.figures['ppo_mechanics'] = fig

    def create_all_plots(
        self,
        metrics_store: MetricsStore,
        comparator: SampleComparator,
        analyzer: QualityAnalyzer,
        ppo_tracker: Optional[Any] = None,
        kl_threshold: float = 0.1
    ) -> None:
        """
        Generate all plots at once.

        Args:
            metrics_store: MetricsStore with training metrics
            comparator: SampleComparator with comparison history
            analyzer: QualityAnalyzer with alert history
            ppo_tracker: Optional PPOMetricsTracker for PPO-specific plots
            kl_threshold: KL divergence warning threshold
        """
        self.create_loss_plot(metrics_store)
        self.create_reward_plot(metrics_store)
        self.create_kl_plot(metrics_store, kl_threshold=kl_threshold)
        self.create_quality_plot(comparator, analyzer)

        if ppo_tracker is not None:
            self.create_ppo_plot(ppo_tracker)

    def close_all(self) -> None:
        """
        Close all matplotlib figures and clear figure cache.

        Useful to free memory after generating plots.
        """
        if plt is not None:
            plt.close('all')
        self.figures.clear()
