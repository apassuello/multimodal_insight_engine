# Training Monitoring System - User Guide

## 📚 Table of Contents

1. [Overview](#overview)
2. [Installation](#installation)
3. [Quick Start](#quick-start)
4. [Core Concepts](#core-concepts)
5. [Basic Usage](#basic-usage)
6. [Advanced Features](#advanced-features)
7. [Integration Guide](#integration-guide)
8. [Configuration Options](#configuration-options)
9. [Output & Reports](#output--reports)
10. [Troubleshooting](#troubleshooting)
11. [API Reference](#api-reference)

---

## Overview

The Training Monitoring System provides comprehensive tracking and analysis for Constitutional AI training with PPO (Proximal Policy Optimization). It helps you:

- **Monitor training progress** in real-time with rich terminal output
- **Detect quality degradation** automatically (catastrophic forgetting, reward hacking)
- **Track PPO mechanics** (clip fraction, advantages, gradients)
- **Generate visualizations** (loss curves, reward trends, quality metrics)
- **Create detailed reports** (markdown and JSON formats)
- **Enable early stopping** when quality issues are detected

### Key Features

✅ **Zero breaking changes** - completely optional
✅ **Minimal dependencies** - works without torch/matplotlib
✅ **Event-driven architecture** - clean protocol-based design
✅ **Memory efficient** - ring buffers with bounded capacity
✅ **Production ready** - 202 comprehensive tests
✅ **Flexible verbosity** - SIMPLE or VERBOSE modes

---

## Installation

### Minimal Installation (Testing/Demo)

```bash
# Core dependencies only
pip install numpy pytest

# Optional: Rich terminal output
pip install rich

# Optional: Visualization support
pip install matplotlib
```

### Full Installation (Production Use)

```bash
# All dependencies for real training
pip install numpy pytest rich matplotlib
pip install torch transformers  # For actual PPO training
```

**Note**: The monitoring system works **without** torch for testing and demos!

---

## Quick Start

### 1. Run the Demo

```bash
# Quick demo (10 iterations, ~10 seconds)
python demos/ppo_monitoring_demo.py --quick

# Verbose mode with PPO mechanics
python demos/ppo_monitoring_demo.py --verbose --num_steps 30

# Full demo
python demos/ppo_monitoring_demo.py --num_steps 100
```

### 2. Check the Output

```bash
ls training_outputs/

# You'll see:
# - training_report.md       (Human-readable summary)
# - training_report.json     (Machine-readable data)
# - loss_curves.png          (if matplotlib installed)
# - rewards.png
# - kl_divergence.png
# - quality_metrics.png
# - ppo_mechanics.png        (verbose mode only)
```

### 3. View the Report

```bash
cat training_outputs/training_report.md
```

---

## Core Concepts

### Architecture

The monitoring system uses an **event-driven architecture**:

```
PPO Trainer
    ↓ (emits events)
TrainingMonitor
    ↓ (dispatches to)
Callbacks (SampleComparator, QualityAnalyzer, PPOMetricsTracker, etc.)
    ↓ (process events)
Reports & Plots
```

### Training Events

Events are emitted at key training phases:

- `TRAINING_START` / `TRAINING_END` - Training lifecycle
- `ITERATION_START` / `ITERATION_END` - Per-iteration tracking
- `RESPONSE_GENERATED` - Generated responses (for quality tracking)
- `REWARD_COMPUTED` - Reward signals
- `POLICY_UPDATE` - Policy optimization step
- `VALUE_UPDATE` - Value function update

### Callbacks

Callbacks implement the `TrainingCallback` protocol and process events:

- **SampleComparator** - Tracks response quality over time
- **QualityAnalyzer** - Detects degradation (KL div, reward hacking)
- **PPOMetricsTracker** - Tracks PPO-specific metrics (verbose mode)
- **TerminalDisplay** - Rich console output
- **Custom callbacks** - Implement your own!

### Verbosity Levels

- **SIMPLE** - Essential metrics only, minimal overhead
- **VERBOSE** - All PPO mechanics, detailed tracking

---

## Basic Usage

### Minimal Example

```python
from src.training.monitoring import TrainingMonitor
from src.safety.constitutional.ppo_trainer import PPOTrainer

# Create monitor
monitor = TrainingMonitor(output_dir="./outputs")

# Create trainer with monitoring
trainer = PPOTrainer(
    policy_model=policy_model,
    value_model=value_model,
    reward_model=reward_model,
    tokenizer=tokenizer,
    device=device,
    monitor=monitor  # <-- Add monitor here
)

# Train with monitoring
with monitor.monitor_context():
    results = trainer.train(prompts, num_steps=100)
```

### With Quality Tracking

```python
from src.training.monitoring import (
    TrainingMonitor,
    SampleComparator,
    QualityAnalyzer,
    VerbosityLevel
)

# Setup monitoring
monitor = TrainingMonitor(
    output_dir="./outputs",
    verbosity=VerbosityLevel.SIMPLE
)

# Add quality tracking
comparator = SampleComparator(
    sample_size=10,           # Track 10 baseline samples
    comparison_frequency=5,    # Compare every 5 iterations
    degradation_threshold=0.7  # Trigger if similarity < 0.7
)

analyzer = QualityAnalyzer(
    monitor=monitor,
    comparator=comparator,
    kl_threshold=0.1,          # Warn if KL > 0.1
    reward_hack_threshold=0.3  # Detect reward hacking
)

# Register callbacks
monitor.register_callback(comparator)
monitor.register_callback(analyzer)

# Train
trainer = PPOTrainer(..., monitor=monitor)
with monitor.monitor_context():
    results = trainer.train(prompts, num_steps=100)
```

### Generate Reports

```python
from src.training.monitoring import PlotManager, ReportGenerator

# After training, generate visualizations
plot_manager = PlotManager(output_dir="./outputs")
plot_manager.create_all_plots(
    metrics_store=monitor.get_metrics_store(),
    comparator=comparator,
    analyzer=analyzer
)

# Generate reports
report_gen = ReportGenerator(output_dir="./outputs")
report_gen.generate_all_reports(
    monitor=monitor,
    comparator=comparator,
    analyzer=analyzer
)

print("✅ Reports saved to ./outputs/")
```

---

## Advanced Features

### 1. PPO Mechanics Tracking (Verbose Mode)

Track detailed PPO training dynamics:

```python
from src.training.monitoring import PPOMetricsTracker, VerbosityLevel

monitor = TrainingMonitor(
    output_dir="./outputs",
    verbosity=VerbosityLevel.VERBOSE  # Enable verbose mode
)

# Track PPO mechanics
ppo_tracker = PPOMetricsTracker(verbosity=VerbosityLevel.VERBOSE)
monitor.register_callback(ppo_tracker)

# After training, get PPO summary
ppo_summary = ppo_tracker.get_ppo_summary()
print(f"Avg Clip Fraction: {ppo_summary['avg_clip_fraction']:.4f}")
print(f"Policy Updates: {ppo_summary['num_policy_updates']}")

# Get loss trends
policy_trend = ppo_tracker.get_policy_loss_trend()
if policy_trend < 0:
    print("✅ Policy loss improving!")
```

### 2. Early Stopping

Automatically stop training when quality degrades:

```python
analyzer = QualityAnalyzer(
    monitor=monitor,
    kl_threshold=0.1  # Will trigger early stop if KL > 0.2 (2x threshold)
)

monitor.register_callback(analyzer)

# Training will stop automatically if critical alerts are triggered
with monitor.monitor_context():
    results = trainer.train(prompts, num_steps=100)
    # May stop early if quality degrades!

# Check if early stopping occurred
should_stop, reason = monitor.should_stop_early()
if should_stop:
    print(f"⚠️  Training stopped early: {reason}")
```

### 3. Custom Callbacks

Implement your own monitoring logic:

```python
from src.training.monitoring import TrainingCallback, TrainingEvent, TrainingPhase

class MyCustomCallback:
    """Custom callback example."""

    def on_event(self, event: TrainingEvent) -> None:
        """Process training events."""
        if event.phase == TrainingPhase.ITERATION_END:
            reward = event.get_metric('reward', 0.0)
            if reward > 0.8:
                print(f"🎉 High reward at iteration {event.iteration}: {reward:.4f}")

# Register your callback
monitor.register_callback(MyCustomCallback())
```

### 4. Catastrophic Forgetting Detection

Automatically detect if the model forgets training examples:

```python
comparator = SampleComparator(
    sample_size=20,            # Track more samples
    comparison_frequency=3,     # Compare more frequently
    degradation_threshold=0.75  # Stricter threshold
)

monitor.register_callback(comparator)

# After training, check for catastrophic forgetting
if comparator.detect_catastrophic_forgetting():
    print("⚠️  CATASTROPHIC FORGETTING DETECTED!")

    # Get detailed comparison summary
    summary = comparator.get_comparison_summary()
    print(f"Degraded samples: {summary['degraded_count']}/{summary['num_comparisons']}")
    print(f"Avg similarity: {summary['avg_similarity']:.4f}")
```

### 5. Selective Plot Generation

Generate only specific plots:

```python
plot_manager = PlotManager(output_dir="./outputs")

# Generate individual plots
plot_manager.create_loss_plot(monitor.get_metrics_store())
plot_manager.create_reward_plot(monitor.get_metrics_store())
plot_manager.create_kl_plot(monitor.get_metrics_store(), kl_threshold=0.1)

# Or generate all at once
plot_manager.create_all_plots(
    metrics_store=monitor.get_metrics_store(),
    comparator=comparator,
    analyzer=analyzer,
    ppo_tracker=ppo_tracker,  # Optional
    kl_threshold=0.1
)

# Clean up matplotlib figures
plot_manager.close_all()
```

### 6. Metrics Access

Access raw metrics for custom analysis:

```python
# Get metrics store
metrics_store = monitor.get_metrics_store()

# Get statistics for a metric
reward_stats = metrics_store.get_statistics('reward')
print(f"Mean: {reward_stats['mean']:.4f}")
print(f"Std: {reward_stats['std']:.4f}")
print(f"Min: {reward_stats['min']:.4f}, Max: {reward_stats['max']:.4f}")

# Compute trend
reward_trend = metrics_store.compute_trend('reward')
if reward_trend > 0:
    print("📈 Rewards are improving!")

# Get recent values
recent_rewards = metrics_store.get_recent('reward', n=10)
print(f"Last 10 rewards: {recent_rewards}")

# Get all metric names
metric_names = metrics_store.get_metric_names()
print(f"Tracked metrics: {metric_names}")
```

---

## Integration Guide

### Integrating with Existing PPO Trainer

The monitoring system is already integrated into `src/safety/constitutional/ppo_trainer.py`. Just add the `monitor` parameter:

```python
from src.safety.constitutional.ppo_trainer import PPOTrainer

# Old code (still works!)
trainer = PPOTrainer(
    policy_model=policy_model,
    value_model=value_model,
    reward_model=reward_model,
    tokenizer=tokenizer,
    device=device
)

# New code (with monitoring)
from src.training.monitoring import TrainingMonitor

monitor = TrainingMonitor(output_dir="./outputs")
trainer = PPOTrainer(
    policy_model=policy_model,
    value_model=value_model,
    reward_model=reward_model,
    tokenizer=tokenizer,
    device=device,
    monitor=monitor  # <-- Just add this!
)
```

**No breaking changes!** Old code works exactly as before.

### Integrating with Custom Trainers

If you have your own training loop:

```python
from src.training.monitoring import TrainingMonitor, TrainingEvent, TrainingPhase

class MyCustomTrainer:
    def __init__(self, monitor=None):
        self.monitor = monitor

    def _emit_event(self, event):
        """Helper to emit events."""
        if self.monitor is not None:
            self.monitor.on_event(event)

    def train(self, prompts, num_steps):
        """Training loop with monitoring."""

        # Emit TRAINING_START
        self._emit_event(TrainingEvent(
            phase=TrainingPhase.TRAINING_START,
            iteration=0,
            metadata={'num_steps': num_steps}
        ))

        for step in range(num_steps):
            # Emit ITERATION_START
            self._emit_event(TrainingEvent(
                phase=TrainingPhase.ITERATION_START,
                iteration=step
            ))

            # Your training logic here
            # ...

            # Emit ITERATION_END with metrics
            self._emit_event(TrainingEvent(
                phase=TrainingPhase.ITERATION_END,
                iteration=step,
                metrics={
                    'loss': loss_value,
                    'reward': reward_value
                }
            ))

            # Check for early stopping
            if self.monitor is not None:
                should_stop, reason = self.monitor.should_stop_early()
                if should_stop:
                    print(f"Early stopping: {reason}")
                    break

        # Emit TRAINING_END
        self._emit_event(TrainingEvent(
            phase=TrainingPhase.TRAINING_END,
            iteration=step
        ))
```

---

## Configuration Options

### TrainingMonitor Options

```python
monitor = TrainingMonitor(
    output_dir=Path("./outputs"),      # Output directory for reports
    verbosity=VerbosityLevel.SIMPLE,   # SIMPLE or VERBOSE
    metrics_capacity=10000             # Max metrics per buffer (default: 10000)
)
```

### SampleComparator Options

```python
comparator = SampleComparator(
    sample_size=10,                    # Number of baseline samples to track
    comparison_frequency=5,             # Compare every N iterations
    degradation_threshold=0.7          # Similarity threshold (0-1)
)
```

### QualityAnalyzer Options

```python
analyzer = QualityAnalyzer(
    monitor=monitor,                   # Required: TrainingMonitor instance
    comparator=comparator,             # Optional: for reward hacking detection
    kl_threshold=0.1,                  # KL divergence warning threshold
    reward_hack_threshold=0.3,         # Reward hacking detection threshold
    window_size=50                     # Window for trend analysis
)
```

### PPOMetricsTracker Options

```python
ppo_tracker = PPOMetricsTracker(
    verbosity=VerbosityLevel.VERBOSE   # Only active in VERBOSE mode
)
```

### PlotManager Options

```python
plot_manager = PlotManager(
    output_dir=Path("./outputs"),
    style='seaborn-v0_8-whitegrid',    # Matplotlib style
    dpi=150                            # Plot resolution
)
```

---

## Output & Reports

### Directory Structure

After running training with monitoring:

```
outputs/
├── training_report.md        # Human-readable summary
├── training_report.json      # Machine-readable data
├── loss_curves.png          # Policy, value, total losses
├── rewards.png              # Reward progression with trend
├── kl_divergence.png        # KL div with thresholds
├── quality_metrics.png      # Sample similarity & alerts
└── ppo_mechanics.png        # PPO-specific plots (verbose mode)
```

### Markdown Report Contents

The markdown report includes:

- **Summary**: Total iterations, early stopping status
- **Metrics Table**: Final statistics with trends
- **Quality Alerts**: Breakdown by severity (INFO/WARNING/CRITICAL)
- **Sample Comparisons**: Catastrophic forgetting detection
- **PPO Mechanics**: Verbose mode statistics

Example:

```markdown
# Training Report

## Summary
- **Total Iterations**: 100
- **Early Stopped**: No

## Metrics
| Metric       | Mean   | Std    | Min    | Max    | Trend |
|--------------|--------|--------|--------|--------|-------|
| policy_loss  | 0.3425 | 0.0742 | 0.2341 | 0.4821 | ↓     |
| reward       | 0.7677 | 0.0762 | 0.5095 | 0.8934 | ↑     |
| kl_div       | 0.0329 | 0.0064 | 0.0199 | 0.0445 | →     |

## Quality Alerts
- INFO: 0
- WARNING: 2
- CRITICAL: 0

...
```

### JSON Report Structure

```json
{
  "summary": {
    "total_iterations": 100,
    "early_stopped": false,
    "stop_reason": null,
    "generated_at": "2025-01-10T13:06:42.123456"
  },
  "metrics": {
    "reward": {
      "mean": 0.7677,
      "std": 0.0762,
      "min": 0.5095,
      "max": 0.8934,
      "count": 100,
      "trend": 0.0032
    }
  },
  "alerts": [
    {
      "iteration": 45,
      "severity": "WARNING",
      "reason": "KL divergence above threshold: 0.1234",
      "metric_name": "kl_div",
      "metric_value": 0.1234
    }
  ],
  "sample_comparisons": {
    "num_comparisons": 20,
    "avg_similarity": 0.8542,
    "degraded_count": 2,
    "catastrophic_forgetting": false
  }
}
```

---

## Troubleshooting

### Common Issues

#### 1. Import Errors

**Problem**: `ModuleNotFoundError: No module named 'src.training.monitoring'`

**Solution**:
```bash
# Make sure you're in the project root
cd /path/to/multimodal_insight_engine

# Check Python path
python -c "import sys; print(sys.path)"
```

#### 2. Matplotlib Not Available

**Problem**: `UserWarning: Matplotlib not available. Plotting disabled.`

**Solution**:
```bash
# Install matplotlib (optional)
pip install matplotlib

# Or just skip it - reports will still be generated!
```

#### 3. Rich Terminal Not Working

**Problem**: Terminal output looks plain or has formatting issues

**Solution**:
```bash
# Install rich for better terminal output
pip install rich

# Or run with SIMPLE verbosity
monitor = TrainingMonitor(verbosity=VerbosityLevel.SIMPLE)
```

#### 4. Training Stops Early Unexpectedly

**Problem**: Training stops with "Early stopping triggered"

**This is a feature!** The system detected quality degradation. To investigate:

```python
# Check alert history
alerts = analyzer.get_alert_summary()
print(f"Critical alerts: {alerts['critical_count']}")

recent_alerts = analyzer.get_recent_alerts(n=5)
for alert in recent_alerts:
    print(f"Iteration {alert.iteration}: {alert.reason}")

# Adjust thresholds if needed
analyzer = QualityAnalyzer(
    monitor=monitor,
    kl_threshold=0.2,  # More lenient threshold
)
```

#### 5. Memory Usage Growing

**Problem**: High memory usage during long training runs

**Solution**: The system uses ring buffers with bounded capacity (10K by default). To reduce:

```python
monitor = TrainingMonitor(
    output_dir="./outputs",
    metrics_capacity=5000  # Reduce buffer size
)
```

#### 6. No Events Being Recorded

**Problem**: Reports show no data

**Solution**: Make sure you're using the monitor context:

```python
# ❌ Wrong - no context
results = trainer.train(prompts, num_steps=100)

# ✅ Correct - with context
with monitor.monitor_context():
    results = trainer.train(prompts, num_steps=100)
```

### Debug Mode

Enable verbose logging:

```python
import logging

logging.basicConfig(level=logging.DEBUG)

# Now run your training - you'll see detailed logs
```

---

## API Reference

### TrainingMonitor

```python
class TrainingMonitor:
    """Main monitoring orchestrator."""

    def __init__(
        self,
        output_dir: Path,
        verbosity: VerbosityLevel = VerbosityLevel.SIMPLE,
        metrics_capacity: int = 10000
    )

    def register_callback(self, callback: TrainingCallback) -> None
    def unregister_callback(self, callback: TrainingCallback) -> None
    def monitor_context(self) -> ContextManager
    def on_event(self, event: TrainingEvent) -> None
    def request_early_stop(self, reason: str) -> None
    def should_stop_early(self) -> Tuple[bool, Optional[str]]
    def get_metrics_store(self) -> MetricsStore
```

### TrainingEvent

```python
@dataclass(frozen=True, slots=True)
class TrainingEvent:
    phase: TrainingPhase
    iteration: int
    epoch: int = 0
    timestamp: float = field(default_factory=time.time)
    metrics: Dict[str, float] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def get_metric(self, name: str, default: float = 0.0) -> float
    def has_metric(self, name: str) -> bool
    def get_metadata(self, key: str, default: Any = None) -> Any
```

### SampleComparator

```python
class SampleComparator:
    """Tracks response quality changes."""

    def __init__(
        self,
        sample_size: int = 10,
        comparison_frequency: int = 5,
        degradation_threshold: float = 0.7
    )

    def compute_diff(self, original: str, updated: str) -> Dict[str, Any]
    def detect_catastrophic_forgetting(self) -> bool
    def get_comparison_summary(self) -> Dict[str, Any]
```

### QualityAnalyzer

```python
class QualityAnalyzer:
    """Detects quality degradation."""

    def __init__(
        self,
        monitor: TrainingMonitor,
        comparator: Optional[SampleComparator] = None,
        kl_threshold: float = 0.1,
        reward_hack_threshold: float = 0.3,
        window_size: int = 50
    )

    def get_alert_summary(self) -> Dict[str, Any]
    def get_recent_alerts(self, n: int = 10) -> List[QualityAlert]
```

### PPOMetricsTracker

```python
class PPOMetricsTracker:
    """Tracks PPO-specific metrics (verbose mode only)."""

    def __init__(self, verbosity: VerbosityLevel)

    def get_ppo_summary(self) -> Dict[str, Any]
    def get_policy_loss_trend(self) -> Optional[float]
    def get_value_loss_trend(self) -> Optional[float]
    def get_clip_fraction_stats(self) -> Dict[str, float]
    def get_advantage_stats(self) -> Dict[str, float]
```

### PlotManager

```python
class PlotManager:
    """Generates matplotlib visualizations."""

    def __init__(
        self,
        output_dir: Path,
        style: str = 'seaborn-v0_8-whitegrid',
        dpi: int = 150
    )

    def create_loss_plot(self, metrics_store: MetricsStore) -> None
    def create_reward_plot(self, metrics_store: MetricsStore) -> None
    def create_kl_plot(self, metrics_store: MetricsStore, kl_threshold: float) -> None
    def create_quality_plot(self, comparator: SampleComparator, analyzer: QualityAnalyzer) -> None
    def create_ppo_plot(self, ppo_tracker: PPOMetricsTracker) -> None
    def create_all_plots(...) -> None
    def close_all(self) -> None
```

### ReportGenerator

```python
class ReportGenerator:
    """Generates training reports."""

    def __init__(self, output_dir: Path)

    def generate_markdown_report(
        self,
        monitor: TrainingMonitor,
        comparator: SampleComparator,
        analyzer: QualityAnalyzer,
        ppo_tracker: Optional[PPOMetricsTracker] = None
    ) -> None

    def generate_json_report(...) -> None
    def generate_all_reports(...) -> None
```

---

## Examples

### Example 1: Basic Monitoring

```python
from src.training.monitoring import TrainingMonitor
from src.safety.constitutional.ppo_trainer import PPOTrainer

monitor = TrainingMonitor(output_dir="./outputs")

trainer = PPOTrainer(..., monitor=monitor)

with monitor.monitor_context():
    trainer.train(prompts, num_steps=50)
```

### Example 2: Quality Tracking with Early Stopping

```python
from src.training.monitoring import (
    TrainingMonitor, SampleComparator, QualityAnalyzer
)

monitor = TrainingMonitor(output_dir="./outputs")

comparator = SampleComparator(sample_size=15, comparison_frequency=3)
analyzer = QualityAnalyzer(monitor=monitor, comparator=comparator, kl_threshold=0.08)

monitor.register_callback(comparator)
monitor.register_callback(analyzer)

trainer = PPOTrainer(..., monitor=monitor)

with monitor.monitor_context():
    trainer.train(prompts, num_steps=100)
    # Will stop early if quality degrades!

should_stop, reason = monitor.should_stop_early()
if should_stop:
    print(f"Stopped: {reason}")
```

### Example 3: Verbose PPO Tracking with Full Reports

```python
from src.training.monitoring import (
    TrainingMonitor, SampleComparator, QualityAnalyzer,
    PPOMetricsTracker, PlotManager, ReportGenerator,
    VerbosityLevel
)

# Setup with verbose mode
monitor = TrainingMonitor(
    output_dir="./outputs",
    verbosity=VerbosityLevel.VERBOSE
)

comparator = SampleComparator()
analyzer = QualityAnalyzer(monitor=monitor, comparator=comparator)
ppo_tracker = PPOMetricsTracker(verbosity=VerbosityLevel.VERBOSE)

monitor.register_callback(comparator)
monitor.register_callback(analyzer)
monitor.register_callback(ppo_tracker)

# Train
trainer = PPOTrainer(..., monitor=monitor)
with monitor.monitor_context():
    trainer.train(prompts, num_steps=100)

# Generate comprehensive reports
plot_mgr = PlotManager(output_dir="./outputs", dpi=300)
plot_mgr.create_all_plots(
    metrics_store=monitor.get_metrics_store(),
    comparator=comparator,
    analyzer=analyzer,
    ppo_tracker=ppo_tracker
)

report_gen = ReportGenerator(output_dir="./outputs")
report_gen.generate_all_reports(monitor, comparator, analyzer, ppo_tracker)

print("✅ Complete reports saved to ./outputs/")
```

### Example 4: Custom Callback

```python
from src.training.monitoring import TrainingCallback, TrainingEvent, TrainingPhase

class RewardLogger(TrainingCallback):
    """Log high rewards to a file."""

    def __init__(self, threshold: float = 0.8):
        self.threshold = threshold
        self.high_rewards = []

    def on_event(self, event: TrainingEvent) -> None:
        if event.phase == TrainingPhase.REWARD_COMPUTED:
            reward = event.get_metric('reward', 0.0)
            if reward > self.threshold:
                self.high_rewards.append((event.iteration, reward))
                print(f"🎉 High reward at iter {event.iteration}: {reward:.4f}")

# Use it
monitor = TrainingMonitor(output_dir="./outputs")
monitor.register_callback(RewardLogger(threshold=0.85))

trainer = PPOTrainer(..., monitor=monitor)
with monitor.monitor_context():
    trainer.train(prompts, num_steps=100)
```

---

## Additional Resources

- **Implementation Plan**: `MONITORING_IMPLEMENTATION_PLAN.md`
- **Demo Script**: `demos/ppo_monitoring_demo.py`
- **Tests**: `tests/training/monitoring/`
- **Source Code**: `src/training/monitoring/`

---

## Support

For issues, questions, or contributions:

1. Check the [Troubleshooting](#troubleshooting) section
2. Review the [API Reference](#api-reference)
3. Run the demo to understand the workflow
4. Examine the test suite for usage examples

---

**Happy Monitoring! 🚀**
