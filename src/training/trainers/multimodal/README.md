# Multimodal Trainer Components

This package contains the decomposed multimodal trainer implementation, split into focused modules following the Single Responsibility Principle. This refactoring addresses the God object anti-pattern in the original 2,927-line `multimodal_trainer.py`.

## Architecture Overview

The multimodal trainer has been decomposed into six focused modules:

```
src/training/trainers/multimodal/
├── __init__.py              # Package exports
├── checkpoint_manager.py    # Checkpoint save/load/resume (~260 lines)
├── metrics_collector.py     # Metrics tracking and visualization (~330 lines)
├── training_loop.py         # Core training execution (~500 lines) [TODO]
├── evaluation.py            # Evaluation logic (~400 lines) [TODO]
├── data_handler.py          # Data loading and batching (~300 lines) [TODO]
└── trainer.py               # Main coordinator (~400 lines) [TODO]
```

## Module Documentation

### CheckpointManager

**Purpose**: Manages model checkpointing and training state persistence.

**Responsibilities**:
- Save and load model checkpoints
- Persist optimizer and scheduler states
- Track training progress (epoch, step, best metric, patience counter)
- Resume training from checkpoints
- Find and manage multiple checkpoints

**Example Usage**:

```python
from src.training.trainers.multimodal import CheckpointManager

# Initialize
checkpoint_mgr = CheckpointManager(
    model=model,
    optimizer=optimizer,
    checkpoint_dir="./checkpoints",
    scheduler=scheduler,
    device=torch.device("cuda")
)

# Update training state
checkpoint_mgr.update_state(
    current_epoch=5,
    global_step=1000,
    best_val_metric=0.85
)

# Save checkpoint
checkpoint_path = checkpoint_mgr.get_checkpoint_path(epoch=5, metric_value=0.85)
checkpoint_mgr.save_checkpoint(checkpoint_path)

# Load checkpoint
state = checkpoint_mgr.load_checkpoint(checkpoint_path)
print(f"Resuming from epoch {state['current_epoch']}")

# Find latest checkpoint
latest = checkpoint_mgr.get_latest_checkpoint()
if latest:
    checkpoint_mgr.load_checkpoint(latest)
```

**Key Features**:
- Automatic directory creation
- Support for learning rate scheduler persistence
- Best model tracking and saving
- Flexible state updates (partial or complete)
- Safe checkpoint loading with validation

**Location**: `src/training/trainers/multimodal/checkpoint_manager.py`

**Tests**: `tests/test_checkpoint_manager.py` (18 test cases)

---

### MetricsCollector

**Purpose**: Collects, tracks, and visualizes training metrics.

**Responsibilities**:
- Collect and store metrics history
- Support nested metrics (e.g., recalls.top1, recalls.top5)
- Log metrics to console
- Visualize training progress with matplotlib
- Diagnose training issues (plateau, exploding loss, etc.)
- Track alignment-specific metrics for multimodal models

**Example Usage**:

```python
from src.training.trainers.multimodal import MetricsCollector

# Initialize
metrics = MetricsCollector()

# Update simple metrics
metrics.update({"loss": 0.5, "accuracy": 0.85}, prefix="train")

# Update nested metrics
metrics.update({
    "loss": 0.4,
    "recalls": {"top1": 0.8, "top5": 0.95}
}, prefix="val")

# Log metrics to console
metrics.log_metrics({"loss": 0.3}, prefix="test")

# Get metric history
loss_history = metrics.get_metric("train_loss")
latest_loss = metrics.get_latest("train_loss")

# Diagnose training issues
diagnosis = metrics.diagnose_training_issues()
print(diagnosis)

# Visualize metrics
metrics.plot_history(save_dir="./plots")

# Get summary statistics
summary = metrics.get_summary()
print(f"Best train loss: {summary['train_loss']['best']}")

# Update alignment metrics (for multimodal training)
metrics.update_alignment_metrics(
    step=1000,
    diag_mean=0.8,
    sim_mean=0.5,
    alignment_gap=0.3,
    alignment_snr=2.5
)

# Save/load metrics
history_dict = metrics.to_dict()
metrics.from_dict(history_dict)
```

**Key Features**:
- Support for both scalar and nested dictionary metrics
- Automatic PyTorch tensor to scalar conversion
- Training diagnostics (plateau detection, exploding loss, unstable training)
- Visualization with automatic grouping by metric type
- Alignment metrics tracking for multimodal models
- Summary statistics (latest, best, mean, std)

**Location**: `src/training/trainers/multimodal/metrics_collector.py`

**Tests**: `tests/test_metrics_collector.py` (26 test cases)

---

## Migration Guide

### For Users of `multimodal_trainer.py`

The original `MultimodalTrainer` class will be refactored to use these new modules. During the transition period:

**Current API** (will remain compatible):
```python
from src.training.trainers.multimodal_trainer import MultimodalTrainer

trainer = MultimodalTrainer(
    model=model,
    train_dataloader=train_loader,
    val_dataloader=val_loader,
    optimizer=optimizer,
    checkpoint_dir="./checkpoints"
)

trainer.train(num_epochs=10)
```

**New Modular API** (after refactoring):
```python
from src.training.trainers.multimodal import MultimodalTrainer

# Same interface, but internally uses modular components
trainer = MultimodalTrainer(
    model=model,
    train_dataloader=train_loader,
    val_dataloader=val_loader,
    optimizer=optimizer,
    checkpoint_dir="./checkpoints"
)

trainer.train(num_epochs=10)

# Access components directly if needed
trainer.checkpoint_mgr.save_checkpoint(path)
trainer.metrics.plot_history(save_dir="./plots")
```

### For Developers Extending the Trainer

You can now extend individual components:

```python
from src.training.trainers.multimodal import CheckpointManager, MetricsCollector

# Custom checkpoint manager with additional state
class CustomCheckpointManager(CheckpointManager):
    def save_checkpoint(self, path, **kwargs):
        # Add custom state before saving
        kwargs['custom_data'] = self.custom_data
        super().save_checkpoint(path, **kwargs)

# Custom metrics collector with additional visualizations
class CustomMetricsCollector(MetricsCollector):
    def plot_custom_metric(self, save_dir):
        # Add custom visualization
        pass
```

---

## Design Decisions

### Why Split the God Object?

**Problems with Original Design**:
1. Single file with 2,927 lines violates Single Responsibility Principle
2. `train_epoch()` method had cyclomatic complexity of 93
3. Difficult to test individual components
4. High coupling between concerns
5. Hard to extend or modify

**Benefits of New Design**:
1. Each module has a single, clear responsibility
2. Easier to test in isolation (44 test cases added)
3. Lower coupling, higher cohesion
4. Can extend/replace individual components
5. Easier to understand and maintain

### Module Boundaries

We identified natural boundaries based on:
- **Data access patterns**: What data each module needs
- **Cohesion**: Related functionality stays together
- **Independence**: Modules can be tested/used independently
- **Size**: Target 200-500 lines per module

### State Management

**Shared State**: Training state is managed by CheckpointManager:
- `current_epoch`, `global_step`
- `best_val_metric`, `patience_counter`
- Training history

**Component State**: Each component manages its own:
- CheckpointManager: File paths, device
- MetricsCollector: Metrics history, alignment data

---

## Testing

### Running Tests

```bash
# Run all tests for new modules
python -m pytest tests/test_checkpoint_manager.py tests/test_metrics_collector.py -v

# Run with coverage
python -m pytest tests/test_checkpoint_manager.py tests/test_metrics_collector.py --cov=src/training/trainers/multimodal --cov-report=html

# Run specific test
python -m pytest tests/test_checkpoint_manager.py::TestCheckpointManager::test_save_checkpoint -v
```

### Test Coverage

**CheckpointManager** (18 tests):
- Initialization and directory creation
- Save/load checkpoint operations
- State persistence and resumption
- Scheduler state handling
- Best model tracking
- Latest checkpoint finding
- Model weight preservation
- Error handling

**MetricsCollector** (26 tests):
- Simple and nested metrics
- Metric history tracking
- Console logging
- Alignment metrics
- Training diagnostics
- Visualization
- Summary statistics
- Tensor handling
- Serialization (to_dict/from_dict)

---

## Performance Considerations

### Memory

- **CheckpointManager**: Minimal overhead (only stores references)
- **MetricsCollector**: O(n) memory where n = number of epochs × number of metrics
  - History stored in `defaultdict(list)`
  - Can be cleared with `metrics.clear()` if needed

### Disk I/O

- **CheckpointManager**: Uses PyTorch's optimized checkpoint format
  - Checkpoints are typically 100-500MB depending on model size
  - Use `save_best_checkpoint()` to avoid saving all epochs

### Computational

- **MetricsCollector**:
  - Metric updates: O(1) per metric
  - Plotting: O(n) where n = number of epochs
  - Only plot when needed (not every epoch)

---

## Future Enhancements

### Planned Features

1. **Distributed Training Support**
   - CheckpointManager: Handle rank-specific checkpoints
   - MetricsCollector: Aggregate metrics across ranks

2. **Cloud Storage Integration**
   - Save checkpoints to S3/GCS
   - Stream metrics to cloud logging services

3. **Advanced Diagnostics**
   - Gradient statistics tracking
   - Learning rate schedule visualization
   - Model parameter histograms

4. **Experiment Tracking**
   - Integration with MLflow/Weights & Biases
   - Hyperparameter logging
   - Model versioning

---

## References

- **Refactoring Strategy**: `docs/improvement-plan/2-architecture-refactoring/refactoring-strategy.md`
- **Architecture Review**: `docs/improvement-plan/2-architecture-refactoring/architecture-review.md`
- **Original Code**: `src/training/trainers/multimodal_trainer.py`

---

## Contributing

When contributing to these modules:

1. **Follow PEP 8** style guidelines
2. **Add type hints** to all function signatures
3. **Write docstrings** in Google style
4. **Add tests** for new functionality (maintain >90% coverage)
5. **Update this README** with new features or changes
6. **Keep modules focused** (200-500 lines recommended)

---

## Questions?

See the main project documentation or contact the architecture team.
