# Migration Guide: Modular Multimodal Trainer

This guide helps you migrate from the monolithic `multimodal_trainer.py` to the new modular architecture.

## Table of Contents

1. [Overview](#overview)
2. [What Changed](#what-changed)
3. [Step-by-Step Migration](#step-by-step-migration)
4. [API Changes](#api-changes)
5. [Common Patterns](#common-patterns)
6. [Troubleshooting](#troubleshooting)

---

## Overview

The original `MultimodalTrainer` class (2,927 lines) has been decomposed into focused modules:

- **CheckpointManager**: Handles checkpoint operations
- **MetricsCollector**: Tracks and visualizes metrics
- **TrainingLoop**: Core training execution (coming soon)
- **Evaluator**: Evaluation logic (coming soon)
- **DataHandler**: Data loading (coming soon)
- **MultimodalTrainer**: Main coordinator (refactored)

## What Changed

### ✅ Backward Compatible

The public API remains the same:

```python
# This still works exactly as before
trainer = MultimodalTrainer(
    model=model,
    train_dataloader=train_loader,
    val_dataloader=val_loader,
    optimizer=optimizer
)
trainer.train(num_epochs=10)
```

### 🔄 Internal Changes

Internally, the trainer now uses modular components:

```python
# Old: Everything in one class
class MultimodalTrainer:
    def save_checkpoint(self, path): ...
    def load_checkpoint(self, path): ...
    def _log_metrics(self, metrics): ...
    def plot_history(self): ...
    # ... 2,900+ more lines

# New: Focused modules
class MultimodalTrainer:
    def __init__(self, ...):
        self.checkpoint_mgr = CheckpointManager(...)
        self.metrics = MetricsCollector(...)
        # ... other components

    def save_checkpoint(self, path):
        # Delegates to component
        self.checkpoint_mgr.save_checkpoint(path)
```

---

## Step-by-Step Migration

### Phase 1: CheckpointManager & MetricsCollector (Complete)

**Status**: ✅ Complete

**What to Update**: Nothing if using public API

**Optional**: Access new components directly

```python
# Old way (still works)
trainer.save_checkpoint(path)
trainer.history['train_loss']

# New way (more explicit)
trainer.checkpoint_mgr.save_checkpoint(path)
trainer.metrics.get_metric('train_loss')
```

### Phase 2: TrainingLoop (Coming Soon)

**Impact**: Low (internal refactoring)

**What to Update**: Custom training loops that override `train_epoch()`

**Before**:
```python
class CustomTrainer(MultimodalTrainer):
    def train_epoch(self):
        # Custom training logic
        ...
```

**After**:
```python
from src.training.trainers.multimodal import TrainingLoop

class CustomTrainingLoop(TrainingLoop):
    def train_step(self, batch, optimizer):
        # Custom step logic
        ...

class CustomTrainer(MultimodalTrainer):
    def __init__(self, ...):
        super().__init__(...)
        self.training_loop = CustomTrainingLoop(...)
```

### Phase 3: Evaluator (Coming Soon)

**Impact**: Low (internal refactoring)

**What to Update**: Custom evaluation that overrides `evaluate()`

**Before**:
```python
class CustomTrainer(MultimodalTrainer):
    def evaluate(self, dataloader):
        # Custom evaluation
        ...
```

**After**:
```python
from src.training.trainers.multimodal import Evaluator

class CustomEvaluator(Evaluator):
    def compute_metrics(self, outputs, labels):
        # Custom metrics
        ...

class CustomTrainer(MultimodalTrainer):
    def __init__(self, ...):
        super().__init__(...)
        self.evaluator = CustomEvaluator(...)
```

---

## API Changes

### CheckpointManager

#### Save Checkpoint

**Old**:
```python
trainer.save_checkpoint(path)
```

**New (recommended)**:
```python
# Option 1: Through trainer (backward compatible)
trainer.save_checkpoint(path)

# Option 2: Direct access (more control)
trainer.checkpoint_mgr.save_checkpoint(
    path=path,
    current_epoch=trainer.current_epoch,
    global_step=trainer.global_step
)

# Option 3: Save best model
trainer.checkpoint_mgr.save_best_checkpoint(
    metric_value=0.95,
    current_epoch=10,
    global_step=1000,
    history=trainer.metrics.to_dict()
)
```

#### Load Checkpoint

**Old**:
```python
trainer.load_checkpoint(path)
```

**New (recommended)**:
```python
# Option 1: Through trainer (backward compatible)
trainer.load_checkpoint(path)

# Option 2: Direct access (returns state dict)
state = trainer.checkpoint_mgr.load_checkpoint(path)
print(f"Resuming from epoch {state['current_epoch']}")

# Option 3: Load latest checkpoint
latest = trainer.checkpoint_mgr.get_latest_checkpoint()
if latest:
    trainer.checkpoint_mgr.load_checkpoint(latest)
```

### MetricsCollector

#### Track Metrics

**Old**:
```python
# Internal to trainer
trainer.history['train_loss'].append(0.5)
```

**New (recommended)**:
```python
# Option 1: Through trainer
trainer.metrics.update({'loss': 0.5, 'accuracy': 0.85}, prefix='train')

# Option 2: Log and update
trainer.metrics.log_metrics({'loss': 0.5}, prefix='train')

# Option 3: Get metric history
losses = trainer.metrics.get_metric('train_loss')
latest_loss = trainer.metrics.get_latest('train_loss')
```

#### Visualize Metrics

**Old**:
```python
trainer.plot_history(save_dir='./plots')
```

**New (recommended)**:
```python
# Option 1: Through trainer (backward compatible)
trainer.plot_history(save_dir='./plots')

# Option 2: Direct access (more control)
trainer.metrics.plot_history(save_dir='./plots')

# Option 3: Get summary
summary = trainer.metrics.get_summary()
print(f"Best loss: {summary['train_loss']['best']}")
```

#### Diagnose Training Issues

**New Feature**:
```python
# Analyze metrics for issues
diagnosis = trainer.metrics.diagnose_training_issues()
print(diagnosis)
# Output: "Training issues detected:
#   - Loss is not decreasing (plateau detected)
#   - Loss is unstable (high variance)"
```

---

## Common Patterns

### Pattern 1: Custom Checkpoint Logic

**Use Case**: Save additional state with checkpoints

```python
from src.training.trainers.multimodal import CheckpointManager

class CustomCheckpointManager(CheckpointManager):
    def __init__(self, *args, custom_state=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.custom_state = custom_state or {}

    def save_checkpoint(self, path, **kwargs):
        # Add custom state
        kwargs['custom_state'] = self.custom_state
        super().save_checkpoint(path, **kwargs)

    def load_checkpoint(self, path):
        state = super().load_checkpoint(path)
        if 'custom_state' in state:
            self.custom_state = state['custom_state']
        return state

# Use custom checkpoint manager
trainer = MultimodalTrainer(...)
trainer.checkpoint_mgr = CustomCheckpointManager(
    model=trainer.model,
    optimizer=trainer.optimizer,
    checkpoint_dir=trainer.checkpoint_dir,
    custom_state={'experiment_id': 'exp123'}
)
```

### Pattern 2: Custom Metrics Visualization

**Use Case**: Add domain-specific visualizations

```python
from src.training.trainers.multimodal import MetricsCollector
import matplotlib.pyplot as plt

class CustomMetricsCollector(MetricsCollector):
    def plot_retrieval_metrics(self, save_dir=None):
        """Plot retrieval-specific metrics."""
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))

        # Plot text-to-image retrieval
        recalls_t2i = [
            self.get_metric('val_recalls.t2i_R1'),
            self.get_metric('val_recalls.t2i_R5'),
            self.get_metric('val_recalls.t2i_R10')
        ]

        # ... custom plotting logic

        if save_dir:
            plt.savefig(f"{save_dir}/retrieval_metrics.png")
        plt.close()

# Use custom metrics collector
trainer = MultimodalTrainer(...)
trainer.metrics = CustomMetricsCollector()

# After training
trainer.metrics.plot_retrieval_metrics(save_dir='./plots')
```

### Pattern 3: Early Stopping with Custom Logic

**Use Case**: Implement custom early stopping criteria

```python
class EarlyStoppingTrainer(MultimodalTrainer):
    def __init__(self, *args, patience=5, min_delta=0.001, **kwargs):
        super().__init__(*args, **kwargs)
        self.patience = patience
        self.min_delta = min_delta

    def should_stop(self):
        """Check if training should stop early."""
        val_losses = self.metrics.get_metric('val_loss')

        if len(val_losses) < self.patience + 1:
            return False

        recent_losses = val_losses[-(self.patience + 1):]
        best_loss = min(recent_losses)
        current_loss = recent_losses[-1]

        # Stop if no improvement beyond min_delta
        return (current_loss - best_loss) > self.min_delta

    def train(self, num_epochs):
        for epoch in range(num_epochs):
            # ... training logic

            if self.should_stop():
                print(f"Early stopping at epoch {epoch}")
                break
```

### Pattern 4: Checkpoint Every N Epochs

**Use Case**: Save checkpoints at regular intervals

```python
def train_with_checkpoints(trainer, num_epochs, checkpoint_every=5):
    """Train with periodic checkpointing."""
    for epoch in range(num_epochs):
        # Train one epoch
        train_metrics = trainer.train_epoch()

        # Evaluate
        val_metrics = trainer.evaluate(trainer.val_dataloader)

        # Update metrics
        trainer.metrics.log_metrics(train_metrics, prefix='train')
        trainer.metrics.log_metrics(val_metrics, prefix='val')

        # Checkpoint periodically
        if (epoch + 1) % checkpoint_every == 0:
            path = trainer.checkpoint_mgr.get_checkpoint_path(
                epoch=epoch,
                metric_value=val_metrics.get('loss')
            )
            trainer.checkpoint_mgr.update_state(
                current_epoch=epoch,
                global_step=trainer.global_step,
                best_val_metric=val_metrics.get('loss')
            )
            trainer.checkpoint_mgr.save_checkpoint(path)
            print(f"Checkpoint saved: {path}")
```

### Pattern 5: Resuming Training

**Use Case**: Resume interrupted training

```python
def train_with_resume(trainer, num_epochs, resume_from=None):
    """Train with automatic resume capability."""
    start_epoch = 0

    # Try to resume
    if resume_from:
        state = trainer.checkpoint_mgr.load_checkpoint(resume_from)
        start_epoch = state['current_epoch']
        print(f"Resuming from epoch {start_epoch}")
    else:
        # Try to find latest checkpoint
        latest = trainer.checkpoint_mgr.get_latest_checkpoint()
        if latest:
            state = trainer.checkpoint_mgr.load_checkpoint(latest)
            start_epoch = state['current_epoch']
            print(f"Resuming from latest checkpoint at epoch {start_epoch}")

    # Continue training
    for epoch in range(start_epoch, num_epochs):
        # ... training logic
        pass

# Usage
train_with_resume(trainer, num_epochs=100, resume_from='./checkpoints/checkpoint_epoch_50.pt')
```

---

## Troubleshooting

### Issue: `AttributeError: 'MultimodalTrainer' object has no attribute 'checkpoint_mgr'`

**Cause**: Using old version of MultimodalTrainer

**Solution**:
1. Update to latest code
2. Or access via old API: `trainer.save_checkpoint()` instead of `trainer.checkpoint_mgr.save_checkpoint()`

### Issue: Checkpoint loading fails with `KeyError`

**Cause**: Checkpoint format mismatch

**Solution**:
```python
# Add error handling
try:
    state = trainer.checkpoint_mgr.load_checkpoint(path)
except KeyError as e:
    print(f"Checkpoint format error: {e}")
    print("This checkpoint may be from an older version")
    # Fall back to manual loading
    checkpoint = torch.load(path, map_location='cpu')
    trainer.model.load_state_dict(checkpoint['model_state_dict'])
```

### Issue: Metrics not showing in plots

**Cause**: No metrics collected yet, or metrics are dictionaries

**Solution**:
```python
# Check if metrics exist
if trainer.metrics.history:
    trainer.metrics.plot_history(save_dir='./plots')
else:
    print("No metrics to plot yet")

# Check metric format
for key, values in trainer.metrics.history.items():
    print(f"{key}: {type(values[0]) if values else 'empty'}")
```

### Issue: Memory usage grows over training

**Cause**: Metrics history accumulates

**Solution**:
```python
# Option 1: Clear metrics periodically
if epoch % 100 == 0:
    trainer.metrics.clear()

# Option 2: Save and clear
if epoch % 100 == 0:
    history = trainer.metrics.to_dict()
    torch.save(history, f'metrics_epoch_{epoch}.pt')
    trainer.metrics.clear()

# Option 3: Limit history size
class LimitedMetricsCollector(MetricsCollector):
    def update(self, metrics, prefix=''):
        super().update(metrics, prefix)
        # Keep only last 1000 entries
        for key in self.history:
            if len(self.history[key]) > 1000:
                self.history[key] = self.history[key][-1000:]
```

---

## Testing Your Migration

### Checklist

- [ ] Training starts successfully
- [ ] Checkpoints save correctly
- [ ] Checkpoints load and resume correctly
- [ ] Metrics are tracked properly
- [ ] Plots are generated
- [ ] No performance regression
- [ ] Custom logic still works

### Validation Script

```python
def validate_migration(trainer, test_loader):
    """Validate that migration works correctly."""
    import tempfile
    import os

    print("Testing checkpoint save/load...")
    with tempfile.TemporaryDirectory() as tmpdir:
        # Save checkpoint
        checkpoint_path = os.path.join(tmpdir, 'test.pt')
        trainer.checkpoint_mgr.update_state(current_epoch=5, global_step=100)
        trainer.checkpoint_mgr.save_checkpoint(checkpoint_path)
        assert os.path.exists(checkpoint_path), "Checkpoint not saved"

        # Load checkpoint
        state = trainer.checkpoint_mgr.load_checkpoint(checkpoint_path)
        assert state['current_epoch'] == 6, "Epoch not restored correctly"  # +1 for resume
        print("✓ Checkpoint save/load works")

    print("Testing metrics collection...")
    trainer.metrics.update({'loss': 0.5}, prefix='train')
    assert len(trainer.metrics.get_metric('train_loss')) > 0, "Metrics not collected"
    print("✓ Metrics collection works")

    print("Testing metrics visualization...")
    with tempfile.TemporaryDirectory() as tmpdir:
        trainer.metrics.plot_history(save_dir=tmpdir)
        # Should not raise error even with minimal data
        print("✓ Metrics visualization works")

    print("\n✅ Migration validated successfully!")

# Run validation
validate_migration(trainer, test_loader)
```

---

## Getting Help

- **Documentation**: See `src/training/trainers/multimodal/README.md`
- **Examples**: Check `examples/` directory (coming soon)
- **Issues**: Report at [GitHub Issues](https://github.com/apassuello/multimodal_insight_engine/issues)
- **Architecture Docs**: `docs/improvement-plan/2-architecture-refactoring/`

---

## Timeline

- **Phase 1** (Week 3): ✅ CheckpointManager, MetricsCollector
- **Phase 2** (Week 3-4): TrainingLoop, Evaluator, DataHandler
- **Phase 3** (Week 4): Refactor main trainer.py
- **Phase 4** (Week 4): Update all imports, integration tests

---

**Last Updated**: 2025-11-10
**Status**: Phase 1 Complete
