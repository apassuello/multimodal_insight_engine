"""
PPO Training with Comprehensive Monitoring Demo

This demo demonstrates how to use the training monitoring system with PPO:
1. Set up monitoring with callbacks (SampleComparator, QualityAnalyzer, PPOMetricsTracker)
2. Train a PPO model with real-time tracking
3. Generate plots and reports
4. Demonstrate early stopping on quality degradation

Usage:
    # Quick demo (10 iterations)
    python demos/ppo_monitoring_demo.py --quick

    # Full monitoring demo (50 iterations)
    python demos/ppo_monitoring_demo.py --num_steps 50

    # With verbose PPO mechanics tracking
    python demos/ppo_monitoring_demo.py --verbose --num_steps 30
"""

import argparse
import os
import sys
from pathlib import Path

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Check if monitoring system is available
try:
    from src.training.monitoring import (
        TrainingMonitor,
        SampleComparator,
        QualityAnalyzer,
        PPOMetricsTracker,
        PlotManager,
        ReportGenerator,
        VerbosityLevel,
        TerminalDisplay
    )
    MONITORING_AVAILABLE = True
except ImportError as e:
    print(f"⚠️  Monitoring system not available: {e}")
    MONITORING_AVAILABLE = False
    sys.exit(1)


def create_mock_ppo_trainer():
    """
    Create a mock PPO trainer for demonstration.

    In real usage, you would use:
    from src.safety.constitutional.ppo_trainer import PPOTrainer
    """
    from typing import List, Dict, Any
    import numpy as np

    class MockPPOTrainer:
        """Mock PPO trainer that simulates training dynamics."""

        def __init__(self, monitor=None):
            self.monitor = monitor

        def _emit_event(self, event):
            """Emit event to monitor."""
            if self.monitor is not None:
                self.monitor.on_event(event)

        def train(
            self,
            prompts: List[str],
            num_steps: int = 20,
            batch_size: int = 4
        ) -> Dict[str, Any]:
            """Simulate PPO training loop."""
            from src.training.monitoring import TrainingEvent, TrainingPhase

            print(f"Starting mock PPO training for {num_steps} steps")

            # Emit TRAINING_START
            self._emit_event(TrainingEvent(
                phase=TrainingPhase.TRAINING_START,
                iteration=0,
                metadata={'num_steps': num_steps, 'batch_size': batch_size}
            ))

            for step in range(num_steps):
                # Emit ITERATION_START
                self._emit_event(TrainingEvent(
                    phase=TrainingPhase.ITERATION_START,
                    iteration=step
                ))

                # Sample batch
                batch_prompts = prompts[:min(batch_size, len(prompts))]

                # Generate mock responses
                responses = [
                    f"Response to '{p[:30]}...' (iteration {step})"
                    for p in batch_prompts
                ]

                # Emit RESPONSE_GENERATED
                self._emit_event(TrainingEvent(
                    phase=TrainingPhase.RESPONSE_GENERATED,
                    iteration=step,
                    metadata={'prompts': batch_prompts, 'responses': responses}
                ))

                # Simulate reward (gradually increasing with some noise)
                base_reward = 0.5 + (step / num_steps) * 0.3
                reward = base_reward + np.random.normal(0, 0.05)

                # Emit REWARD_COMPUTED
                self._emit_event(TrainingEvent(
                    phase=TrainingPhase.REWARD_COMPUTED,
                    iteration=step,
                    metrics={'reward': float(reward)}
                ))

                # Simulate 4 epochs of PPO optimization
                for epoch in range(4):
                    # Policy update
                    policy_loss = 0.5 - (step / num_steps) * 0.2 + np.random.normal(0, 0.05)
                    kl_div = 0.01 + (step / num_steps) * 0.02 + np.random.normal(0, 0.005)
                    clip_fraction = 0.15 + np.random.normal(0, 0.03)

                    self._emit_event(TrainingEvent(
                        phase=TrainingPhase.POLICY_UPDATE,
                        iteration=step,
                        metrics={
                            'policy_loss': float(max(0.1, policy_loss)),
                            'kl_div': float(max(0.0, kl_div)),
                            'clip_fraction': float(np.clip(clip_fraction, 0, 0.5)),
                            'gradient_norm': float(0.5 + np.random.normal(0, 0.1)),
                            'learning_rate': 1e-5,
                            'advantage_mean': float(np.random.normal(0, 0.5)),
                            'advantage_std': float(0.5 + np.random.normal(0, 0.1))
                        }
                    ))

                    # Value update
                    value_loss = 0.3 - (step / num_steps) * 0.1 + np.random.normal(0, 0.03)

                    self._emit_event(TrainingEvent(
                        phase=TrainingPhase.VALUE_UPDATE,
                        iteration=step,
                        metrics={
                            'value_loss': float(max(0.05, value_loss)),
                            'gradient_norm': float(0.3 + np.random.normal(0, 0.05))
                        }
                    ))

                # Emit ITERATION_END
                self._emit_event(TrainingEvent(
                    phase=TrainingPhase.ITERATION_END,
                    iteration=step,
                    metrics={
                        'policy_loss': float(max(0.1, policy_loss)),
                        'value_loss': float(max(0.05, value_loss)),
                        'kl_div': float(max(0.0, kl_div)),
                        'reward': float(reward)
                    }
                ))

                # Check for early stopping
                if self.monitor is not None:
                    should_stop, stop_reason = self.monitor.should_stop_early()
                    if should_stop:
                        print(f"\n⚠️  Early stopping triggered at step {step + 1}: {stop_reason}")
                        break

            # Emit TRAINING_END
            self._emit_event(TrainingEvent(
                phase=TrainingPhase.TRAINING_END,
                iteration=step,
                metadata={'completed_steps': step + 1}
            ))

            return {'completed_steps': step + 1}

    return MockPPOTrainer


def run_monitored_training_demo(args):
    """Run PPO training with comprehensive monitoring."""

    print("=" * 70)
    print("PPO TRAINING WITH COMPREHENSIVE MONITORING DEMO")
    print("=" * 70)

    # Setup output directory
    output_dir = Path("./training_outputs")
    output_dir.mkdir(exist_ok=True)

    # Determine verbosity
    verbosity = VerbosityLevel.VERBOSE if args.verbose else VerbosityLevel.SIMPLE

    print(f"\n📊 Setting up monitoring system (verbosity: {verbosity.name})...")

    # Create monitor
    monitor = TrainingMonitor(
        output_dir=output_dir,
        verbosity=verbosity
    )

    # Create callbacks
    comparator = SampleComparator(
        sample_size=5,
        comparison_frequency=5,
        degradation_threshold=0.7
    )

    analyzer = QualityAnalyzer(
        monitor=monitor,
        comparator=comparator,
        kl_threshold=0.05,
        reward_hack_threshold=0.3
    )

    ppo_tracker = None
    if verbosity == VerbosityLevel.VERBOSE:
        ppo_tracker = PPOMetricsTracker(verbosity=verbosity)

    # Create terminal display (optional, for rich console output)
    display = TerminalDisplay(verbosity=verbosity)

    # Register callbacks
    monitor.register_callback(comparator)
    monitor.register_callback(analyzer)
    monitor.register_callback(display)

    if ppo_tracker:
        monitor.register_callback(ppo_tracker)

    print(f"   ✓ Registered {monitor.num_callbacks} callbacks")
    print(f"   ✓ Output directory: {output_dir}")

    # Create training prompts
    prompts = [
        "Explain the concept of fairness in AI systems.",
        "What are the ethical considerations in AI development?",
        "How can we ensure AI systems respect privacy?",
        "Describe the importance of transparency in machine learning.",
        "What role does human oversight play in AI?",
        "How should AI handle sensitive information?",
        "What are best practices for AI safety?",
        "Explain the concept of AI alignment.",
    ]

    # Create PPO trainer with monitoring
    print(f"\n🚀 Creating PPO trainer...")
    PPOTrainerClass = create_mock_ppo_trainer()
    trainer = PPOTrainerClass(monitor=monitor)

    # Run training with monitoring context
    print(f"\n🔄 Starting training with {args.num_steps} steps...\n")

    with monitor.monitor_context():
        results = trainer.train(
            prompts=prompts,
            num_steps=args.num_steps,
            batch_size=args.batch_size
        )

    # Generate reports and plots
    print(f"\n📈 Generating visualizations and reports...")

    # Create plots
    plot_manager = PlotManager(output_dir=output_dir, dpi=150)

    try:
        plot_manager.create_all_plots(
            metrics_store=monitor.get_metrics_store(),
            comparator=comparator,
            analyzer=analyzer,
            ppo_tracker=ppo_tracker,
            kl_threshold=0.05
        )
        print(f"   ✓ Plots saved to {output_dir}/")
    except Exception as e:
        print(f"   ⚠️  Could not generate plots (matplotlib not available): {e}")

    # Generate reports
    report_gen = ReportGenerator(output_dir=output_dir)
    report_gen.generate_all_reports(
        monitor=monitor,
        comparator=comparator,
        analyzer=analyzer,
        ppo_tracker=ppo_tracker
    )
    print(f"   ✓ Reports saved to {output_dir}/")

    # Print summary
    print(f"\n" + "=" * 70)
    print("TRAINING SUMMARY")
    print("=" * 70)

    metrics_store = monitor.get_metrics_store()
    metric_names = metrics_store.get_metric_names()

    if metric_names:
        print("\nFinal Metrics:")
        for metric_name in sorted(metric_names):
            stats = metrics_store.get_statistics(metric_name)
            trend = metrics_store.compute_trend(metric_name)
            trend_str = "↑" if trend and trend > 0.001 else "↓" if trend and trend < -0.001 else "→"
            print(f"  {metric_name:20s}: {stats['mean']:.4f} ± {stats['std']:.4f}  {trend_str}")

    # Alert summary
    alert_summary = analyzer.get_alert_summary()
    print(f"\nQuality Alerts:")
    print(f"  INFO:     {alert_summary['info_count']}")
    print(f"  WARNING:  {alert_summary['warning_count']}")
    print(f"  CRITICAL: {alert_summary['critical_count']}")

    # Sample comparison summary
    comp_summary = comparator.get_comparison_summary()
    if comp_summary['num_comparisons'] > 0:
        print(f"\nSample Comparisons:")
        print(f"  Total: {comp_summary['num_comparisons']}")
        print(f"  Avg Similarity: {comp_summary['avg_similarity']:.4f}")
        print(f"  Degraded: {comp_summary['degraded_count']}")

        if comparator.detect_catastrophic_forgetting():
            print(f"  ⚠️  CATASTROPHIC FORGETTING DETECTED!")

    # PPO summary
    if ppo_tracker:
        ppo_summary = ppo_tracker.get_ppo_summary()
        print(f"\nPPO Mechanics (Verbose Mode):")
        print(f"  Policy Updates: {ppo_summary['num_policy_updates']}")
        print(f"  Value Updates:  {ppo_summary['num_value_updates']}")
        print(f"  Avg Clip Fraction: {ppo_summary['avg_clip_fraction']:.4f}")

    print(f"\n" + "=" * 70)
    print(f"✅ Demo complete! Check {output_dir}/ for detailed reports and plots.")
    print("=" * 70)


def main():
    parser = argparse.ArgumentParser(description="PPO Monitoring Demo")
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Quick demo with 10 iterations"
    )
    parser.add_argument(
        "--num_steps",
        type=int,
        default=20,
        help="Number of training steps (default: 20)"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=4,
        help="Batch size (default: 4)"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose mode with PPO mechanics tracking"
    )

    args = parser.parse_args()

    if args.quick:
        args.num_steps = 10
        args.verbose = False

    if not MONITORING_AVAILABLE:
        print("❌ Monitoring system not available. Cannot run demo.")
        return 1

    try:
        run_monitored_training_demo(args)
        return 0
    except KeyboardInterrupt:
        print("\n\n⚠️  Demo interrupted by user.")
        return 1
    except Exception as e:
        print(f"\n\n❌ Error during demo: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
