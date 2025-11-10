# Training Monitoring System - Implementation Plan

## Overview
Comprehensive training monitoring system for Constitutional AI with event-driven architecture, adaptive verbosity, and quality degradation detection.

## Current Status

### ✅ COMPLETED: Phase 1 - Core Infrastructure (Days 1-2)

**Implemented Components:**
1. **verbosity.py** - VerbosityLevel enum (SIMPLE/VERBOSE)
2. **events.py** - TrainingEvent (frozen dataclass) and TrainingPhase enum (11 lifecycle phases)
3. **callbacks.py** - Protocol-based TrainingCallback and CallbackManager
4. **metrics.py** - MetricsBuffer (ring buffer, 10K capacity) and MetricsStore
5. **training_monitor.py** - Main orchestrator with context manager
6. **__init__.py** - Public API exports
7. **terminal_display.py** - Rich terminal display (SIMPLE and VERBOSE modes)

**Test Coverage:**
- 104 tests, all passing
- Files: `test_verbosity.py`, `test_events.py`, `test_callbacks.py`, `test_metrics.py`, `test_training_monitor.py`

**Key Design Patterns:**
- Protocol-based callbacks for zero coupling
- Frozen dataclasses with __slots__ for memory efficiency
- Ring buffer (deque) for bounded metrics storage
- Lazy NumPy conversion with caching
- Rate-limited display updates (max 2 Hz)

---

## 🚧 TODO: Phase 2 - Display & Comparison (Days 3-4)

### Files to Implement:

#### 1. `sample_comparator.py`
**Purpose:** Track before/after sample changes to detect quality degradation

**Key Components:**
```python
class SampleComparator(TrainingCallback):
    """
    Tracks response quality changes during training.

    Features:
    - Store initial responses (first N iterations)
    - Re-generate same prompts periodically
    - Compute diff statistics (character changes, semantic drift)
    - Detect catastrophic forgetting
    """

    def __init__(
        self,
        sample_size: int = 10,
        comparison_frequency: int = 100
    ):
        self.baseline_samples = []
        self.comparison_history = []

    def on_event(self, event: TrainingEvent) -> None:
        """Collect baseline and track changes."""
        pass

    def compute_diff(self, original: str, updated: str) -> Dict[str, Any]:
        """
        Use difflib to compute:
        - Character-level changes
        - Line-level changes
        - Similarity ratio
        """
        pass

    def detect_catastrophic_forgetting(self) -> bool:
        """Check if responses degraded significantly."""
        pass
```

**Dependencies:**
- `difflib` (standard library) for text comparison
- Store prompts and responses from early iterations
- Trigger re-generation at intervals

**Integration:**
```python
comparator = SampleComparator(sample_size=10, comparison_frequency=50)
monitor.register_callback(comparator)
```

---

#### 2. Update `terminal_display.py`
**TODO:** Add test coverage for terminal_display.py

**Test File:** `test_terminal_display.py`

**Test Coverage Needed:**
- Simple mode output formatting
- Verbose mode table creation
- Rate limiting (max 2 Hz updates)
- Lifecycle events (start/end)
- Metrics store integration
- Trend indicator formatting

---

## 🚧 TODO: Phase 3 - Quality Analysis (Days 5-6)

### Files to Implement:

#### 3. `quality_analyzer.py`
**Purpose:** Detect reward hacking, coherence loss, and trigger early stopping

**Key Components:**
```python
class QualityAnalyzer(TrainingCallback):
    """
    Monitors training quality and detects degradation.

    Checks:
    1. Reward hacking: Reward increases but sample quality decreases
    2. Coherence loss: Responses become less coherent over time
    3. KL divergence explosion: Policy drift from reference
    4. Value-reward mismatch: Value head disagrees with rewards

    Triggers early stopping when critical issues detected.
    """

    def __init__(
        self,
        monitor: TrainingMonitor,
        reward_hack_threshold: float = 0.3,
        kl_threshold: float = 0.1,
        window_size: int = 50
    ):
        self.monitor = monitor
        self.thresholds = {...}
        self.alert_history = []

    def on_event(self, event: TrainingEvent) -> None:
        """Analyze each iteration for quality issues."""
        if event.phase == TrainingPhase.ITERATION_END:
            self._check_reward_hacking(event)
            self._check_kl_divergence(event)
            self._check_coherence(event)

    def _check_reward_hacking(self, event: TrainingEvent) -> None:
        """
        Detect reward hacking:
        - Reward trend increasing
        - Sample quality (from comparator) decreasing
        - Trigger alert if both conditions met
        """
        reward_trend = self.monitor.get_metrics_store().compute_trend('reward')
        # Compare with sample quality trend from comparator
        pass

    def _check_kl_divergence(self, event: TrainingEvent) -> None:
        """Check if KL divergence exceeds threshold."""
        kl_div = event.get_metric('kl_div', 0.0)
        if kl_div > self.thresholds['kl']:
            self._trigger_alert("KL divergence too high", event)

    def _trigger_alert(self, reason: str, event: TrainingEvent) -> None:
        """Log alert and potentially request early stop."""
        self.alert_history.append({
            'iteration': event.iteration,
            'reason': reason,
            'severity': 'critical'
        })

        # Request early stop for critical issues
        if severity == 'critical':
            self.monitor.request_early_stop(reason)
```

**Integration:**
```python
analyzer = QualityAnalyzer(
    monitor=monitor,
    reward_hack_threshold=0.3,
    kl_threshold=0.1
)
monitor.register_callback(analyzer)
```

---

## 🚧 TODO: Phase 4 - PPO Mechanics Tracker (Days 7-8)

### Files to Implement:

#### 4. `ppo_metrics_tracker.py`
**Purpose:** Track detailed PPO training mechanics (verbose mode only)

**Key Components:**
```python
class PPOMetricsTracker(TrainingCallback):
    """
    Tracks PPO-specific metrics for verbose mode.

    Metrics:
    - Policy loss components (clipped vs unclipped)
    - Value loss trends
    - Advantage statistics (mean, std, distribution)
    - Gradient norms
    - Learning rate schedule
    - Clip fraction (how often clipping occurs)
    """

    def __init__(self, verbosity: VerbosityLevel):
        self.verbosity = verbosity
        self.ppo_history = []

    def on_event(self, event: TrainingEvent) -> None:
        """Track PPO-specific metrics."""
        if self.verbosity != VerbosityLevel.VERBOSE:
            return

        if event.phase == TrainingPhase.POLICY_UPDATE:
            self._track_policy_update(event)
        elif event.phase == TrainingPhase.VALUE_UPDATE:
            self._track_value_update(event)

    def _track_policy_update(self, event: TrainingEvent) -> None:
        """
        Track policy update details:
        - policy_loss
        - clip_fraction (from metadata)
        - gradient_norm
        """
        pass

    def _track_value_update(self, event: TrainingEvent) -> None:
        """Track value head training."""
        pass

    def get_ppo_summary(self) -> Dict[str, Any]:
        """Generate summary of PPO training dynamics."""
        return {
            'avg_clip_fraction': ...,
            'policy_improvement': ...,
            'value_accuracy': ...
        }
```

**Integration:**
```python
ppo_tracker = PPOMetricsTracker(verbosity=VerbosityLevel.VERBOSE)
monitor.register_callback(ppo_tracker)
```

---

## 🚧 TODO: Phase 5 - Visualization (Days 9-10)

### Files to Implement:

#### 5. `plot_manager.py`
**Purpose:** Generate matplotlib plots for training metrics

**Key Components:**
```python
class PlotManager:
    """
    Creates training visualization plots.

    Plots:
    1. Loss curves (policy, value, total)
    2. Reward progression with trend lines
    3. KL divergence over time
    4. Quality metrics (coherence, sample similarity)
    5. PPO mechanics (clip fraction, advantages)
    """

    def __init__(self, output_dir: Path):
        self.output_dir = output_dir
        self.figures = {}

    def create_loss_plot(self, metrics_store: MetricsStore) -> None:
        """
        Create loss curves plot:
        - Policy loss
        - Value loss
        - Total loss
        - Save to {output_dir}/loss_curves.png
        """
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        # ... plotting code
        fig.savefig(self.output_dir / "loss_curves.png", dpi=150)

    def create_reward_plot(self, metrics_store: MetricsStore) -> None:
        """Plot reward progression with trend line."""
        pass

    def create_kl_plot(self, metrics_store: MetricsStore) -> None:
        """Plot KL divergence with warning threshold."""
        pass

    def create_quality_plot(
        self,
        comparator: SampleComparator,
        analyzer: QualityAnalyzer
    ) -> None:
        """Plot quality metrics and alerts."""
        pass

    def create_all_plots(
        self,
        metrics_store: MetricsStore,
        comparator: SampleComparator,
        analyzer: QualityAnalyzer
    ) -> None:
        """Generate all plots at once."""
        self.create_loss_plot(metrics_store)
        self.create_reward_plot(metrics_store)
        self.create_kl_plot(metrics_store)
        self.create_quality_plot(comparator, analyzer)
```

**Dependencies:**
- `matplotlib` for plotting
- Access to MetricsStore, SampleComparator, QualityAnalyzer

**Usage:**
```python
plot_manager = PlotManager(output_dir=monitor.output_dir)
plot_manager.create_all_plots(
    metrics_store=monitor.get_metrics_store(),
    comparator=comparator,
    analyzer=analyzer
)
```

---

#### 6. `report_generator.py`
**Purpose:** Generate markdown and JSON training reports

**Key Components:**
```python
class ReportGenerator:
    """
    Generates comprehensive training reports.

    Outputs:
    1. Markdown report with metrics, plots, alerts
    2. JSON report with structured data
    """

    def __init__(self, output_dir: Path):
        self.output_dir = output_dir

    def generate_markdown_report(
        self,
        monitor: TrainingMonitor,
        comparator: SampleComparator,
        analyzer: QualityAnalyzer,
        ppo_tracker: Optional[PPOMetricsTracker] = None
    ) -> None:
        """
        Generate markdown report:

        # Training Report

        ## Summary
        - Total iterations: X
        - Training time: Y
        - Final metrics: {...}
        - Early stop: Yes/No

        ## Metrics
        [Metrics table with final values]

        ## Plots
        ![Loss Curves](loss_curves.png)
        ![Rewards](rewards.png)

        ## Quality Alerts
        [List of alerts from analyzer]

        ## Sample Changes
        [Diff examples from comparator]

        ## PPO Mechanics (if verbose)
        [PPO-specific metrics]
        """
        markdown = self._build_markdown(...)
        (self.output_dir / "training_report.md").write_text(markdown)

    def generate_json_report(
        self,
        monitor: TrainingMonitor,
        comparator: SampleComparator,
        analyzer: QualityAnalyzer
    ) -> None:
        """
        Generate JSON report with structured data:
        {
          "metrics": {...},
          "alerts": [...],
          "sample_changes": [...],
          "configuration": {...}
        }
        """
        import json
        report = self._build_json_data(...)
        (self.output_dir / "training_report.json").write_text(
            json.dumps(report, indent=2)
        )
```

**Usage:**
```python
report_gen = ReportGenerator(output_dir=monitor.output_dir)
report_gen.generate_markdown_report(monitor, comparator, analyzer)
report_gen.generate_json_report(monitor, comparator, analyzer)
```

---

## 🚧 TODO: Phase 6 - Integration (Day 11)

### Files to Modify:

#### 7. `src/safety/constitutional/ppo_trainer.py`
**Purpose:** Integrate monitoring into RLAIFTrainer

**Changes Needed:**

```python
from src.training.monitoring import (
    TrainingMonitor,
    TrainingEvent,
    TrainingPhase,
    VerbosityLevel
)

class RLAIFTrainer:
    def __init__(
        self,
        ...,
        monitor: Optional[TrainingMonitor] = None
    ):
        self.monitor = monitor
        # ... existing code

    def train(
        self,
        prompts: List[str],
        num_iterations: int = 100,
        batch_size: int = 4
    ) -> Dict[str, Any]:
        """Train with monitoring."""

        # Use monitor context if available
        if self.monitor is not None:
            context = self.monitor.monitor_context()
        else:
            from contextlib import nullcontext
            context = nullcontext()

        with context:
            for iteration in range(num_iterations):
                # Emit ITERATION_START
                self._emit_event(TrainingEvent(
                    phase=TrainingPhase.ITERATION_START,
                    iteration=iteration
                ))

                # Generate responses
                responses = self._generate_responses(prompts, batch_size)
                self._emit_event(TrainingEvent(
                    phase=TrainingPhase.RESPONSE_GENERATED,
                    iteration=iteration,
                    metadata={'responses': responses, 'prompts': prompts}
                ))

                # Compute rewards
                rewards = self.compute_rewards(prompts, responses)
                self._emit_event(TrainingEvent(
                    phase=TrainingPhase.REWARD_COMPUTED,
                    iteration=iteration,
                    metrics={'reward': float(rewards.mean())}
                ))

                # PPO update
                policy_loss = self._update_policy(...)
                self._emit_event(TrainingEvent(
                    phase=TrainingPhase.POLICY_UPDATE,
                    iteration=iteration,
                    metrics={'policy_loss': float(policy_loss)}
                ))

                value_loss = self._update_value(...)
                self._emit_event(TrainingEvent(
                    phase=TrainingPhase.VALUE_UPDATE,
                    iteration=iteration,
                    metrics={'value_loss': float(value_loss)}
                ))

                # Iteration end with all metrics
                self._emit_event(TrainingEvent(
                    phase=TrainingPhase.ITERATION_END,
                    iteration=iteration,
                    metrics={
                        'loss': float(policy_loss + value_loss),
                        'reward': float(rewards.mean()),
                        'kl_div': float(kl_divergence),
                    }
                ))

                # Check for early stopping
                if self.monitor is not None:
                    should_stop, reason = self.monitor.should_stop_early()
                    if should_stop:
                        print(f"Early stopping: {reason}")
                        break

        return self.history

    def _emit_event(self, event: TrainingEvent) -> None:
        """Helper to emit events if monitor is available."""
        if self.monitor is not None:
            self.monitor.on_event(event)
```

---

#### 8. `demos/constitutional_ai_real_training_demo.py`
**Purpose:** Update demo to use monitoring system

**Changes Needed:**

```python
from src.training.monitoring import (
    TrainingMonitor,
    VerbosityLevel
)
from src.training.monitoring.terminal_display import TerminalDisplay
from src.training.monitoring.sample_comparator import SampleComparator
from src.training.monitoring.quality_analyzer import QualityAnalyzer
from src.training.monitoring.ppo_metrics_tracker import PPOMetricsTracker
from src.training.monitoring.plot_manager import PlotManager
from src.training.monitoring.report_generator import ReportGenerator

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--verbose', action='store_true',
                       help='Enable verbose monitoring')
    parser.add_argument('--output-dir', type=str,
                       default='output/training_monitor',
                       help='Output directory for plots and reports')
    args = parser.parse_args()

    # Setup monitoring
    verbosity = VerbosityLevel.VERBOSE if args.verbose else VerbosityLevel.SIMPLE
    monitor = TrainingMonitor(
        verbosity=verbosity,
        output_dir=args.output_dir
    )

    # Register callbacks
    terminal = TerminalDisplay(verbosity=verbosity)
    monitor.register_callback(terminal)

    comparator = SampleComparator(sample_size=5, comparison_frequency=20)
    monitor.register_callback(comparator)

    analyzer = QualityAnalyzer(monitor, reward_hack_threshold=0.3)
    monitor.register_callback(analyzer)

    if args.verbose:
        ppo_tracker = PPOMetricsTracker(verbosity=verbosity)
        monitor.register_callback(ppo_tracker)

    # Link metrics store to terminal
    terminal.set_metrics_store(monitor.get_metrics_store())

    # Train with monitoring
    trainer = RLAIFTrainer(
        policy_model=model,
        ref_policy_model=ref_model,
        value_model=value_model,
        reward_model=reward_model,
        tokenizer=tokenizer,
        device=device,
        monitor=monitor  # Pass monitor to trainer
    )

    print("Starting training with monitoring...")
    trainer.train(prompts=prompts, num_iterations=50, batch_size=4)

    # Generate reports
    print("\nGenerating plots and reports...")
    plot_manager = PlotManager(output_dir=Path(args.output_dir))
    plot_manager.create_all_plots(
        metrics_store=monitor.get_metrics_store(),
        comparator=comparator,
        analyzer=analyzer
    )

    report_gen = ReportGenerator(output_dir=Path(args.output_dir))
    report_gen.generate_markdown_report(
        monitor=monitor,
        comparator=comparator,
        analyzer=analyzer,
        ppo_tracker=ppo_tracker if args.verbose else None
    )
    report_gen.generate_json_report(monitor, comparator, analyzer)

    print(f"\nReports saved to: {args.output_dir}")
    print("  - training_report.md")
    print("  - training_report.json")
    print("  - *.png (plots)")

if __name__ == "__main__":
    main()
```

**Usage:**
```bash
# Simple mode
python demos/constitutional_ai_real_training_demo.py

# Verbose mode with all details
python demos/constitutional_ai_real_training_demo.py --verbose

# Custom output directory
python demos/constitutional_ai_real_training_demo.py --verbose --output-dir results/run_001
```

---

## 🚧 TODO: Phase 7 - Testing & Finalization (Day 12)

### Tasks:

1. **Write integration tests** (`test_integration.py`)
   - Test full monitoring workflow end-to-end
   - Test all callbacks working together
   - Test report generation

2. **Performance benchmarking**
   - Measure monitoring overhead (should be < 1% of training time)
   - Test with large-scale training (1000+ iterations)
   - Verify memory bounds (ring buffers working)

3. **Documentation**
   - Update __init__.py docstrings with usage examples
   - Add README.md to monitoring module
   - Document all public APIs

4. **Final testing**
   - Run full test suite: `pytest tests/training/monitoring/ -v`
   - Run demo with verbose mode
   - Verify all outputs (terminal, plots, reports)

---

## File Structure Summary

```
src/training/monitoring/
├── __init__.py                 # ✅ Public API
├── verbosity.py                # ✅ VerbosityLevel enum
├── events.py                   # ✅ TrainingEvent, TrainingPhase
├── callbacks.py                # ✅ Protocol, CallbackManager
├── metrics.py                  # ✅ MetricsBuffer, MetricsStore
├── training_monitor.py         # ✅ Main orchestrator
├── terminal_display.py         # ✅ Rich terminal UI
├── sample_comparator.py        # 🚧 TODO: Before/after tracking
├── quality_analyzer.py         # 🚧 TODO: Degradation detection
├── ppo_metrics_tracker.py      # 🚧 TODO: PPO mechanics
├── plot_manager.py             # 🚧 TODO: Matplotlib plots
└── report_generator.py         # 🚧 TODO: Markdown/JSON reports

tests/training/monitoring/
├── __init__.py                 # ✅ Test suite
├── test_verbosity.py           # ✅ 6 tests
├── test_events.py              # ✅ 12 tests
├── test_callbacks.py           # ✅ 17 tests
├── test_metrics.py             # ✅ 44 tests
├── test_training_monitor.py    # ✅ 25 tests
├── test_terminal_display.py    # 🚧 TODO
├── test_sample_comparator.py   # 🚧 TODO
├── test_quality_analyzer.py    # 🚧 TODO
├── test_ppo_metrics_tracker.py # 🚧 TODO
├── test_plot_manager.py        # 🚧 TODO
├── test_report_generator.py    # 🚧 TODO
└── test_integration.py         # 🚧 TODO: End-to-end tests
```

---

## Key Design Decisions

1. **Event-Driven Architecture**
   - Loose coupling via Protocol-based callbacks
   - Zero-copy event sharing (frozen dataclasses)
   - Easy to add new callbacks without modifying existing code

2. **Memory Efficiency**
   - Ring buffers with bounded capacity (10K values per metric)
   - __slots__ to reduce memory footprint
   - Lazy NumPy conversion with caching

3. **Rate Limiting**
   - Max 2 Hz display updates to avoid flickering
   - Cache invalidation on new data

4. **Adaptive Verbosity**
   - SIMPLE: One-line status updates
   - VERBOSE: Detailed tables, trends, PPO mechanics

5. **Quality Monitoring**
   - Sample comparison for catastrophic forgetting
   - Reward hacking detection
   - KL divergence bounds
   - Automatic early stopping

---

## Next Steps for Continuation

1. Start with `sample_comparator.py` - foundational for quality analysis
2. Then `quality_analyzer.py` - depends on comparator
3. Then `ppo_metrics_tracker.py` - independent, can be done in parallel
4. Then `plot_manager.py` and `report_generator.py` - depend on all callbacks
5. Finally integration into trainer and demo
6. End with comprehensive testing

**Estimated Remaining Time:** 10 days (Days 3-12)
**Current Progress:** Days 1-2 complete (Core Infrastructure + Terminal Display)
