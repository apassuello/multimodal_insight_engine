# Refactoring Strategy

## Overview

This document provides detailed step-by-step strategies for refactoring the key architectural components identified in [architecture-review.md](architecture-review.md).

See [README.md](README.md) for the overall plan and timeline.

---

## Strategy 1: Decompose multimodal_trainer.py

**Goal**: Split 2,927-line God object into 6 focused modules
**Timeline**: Weeks 3-4 (10 working days)
**Effort**: 40-50 hours

### Current State

File: `src/training/trainers/multimodal_trainer.py` (2,927 lines)

Key problems:
- `train_epoch()` method: Complexity 93 (should be <10)
- Single Responsibility Principle violated
- Difficult to test
- Difficult to extend

### Target Structure

```
src/training/trainers/multimodal/
├── __init__.py                    # Package exports
├── trainer.py                     # Main coordinator (~400 lines)
├── training_loop.py               # Training execution (~500 lines)
├── evaluation.py                  # Evaluation logic (~400 lines)
├── checkpoint_manager.py          # Save/load/resume (~200 lines)
├── metrics_collector.py           # Metrics tracking (~300 lines)
└── data_handler.py                # Data loading (~300 lines)
```

### Migration Steps

#### Day 1: Extract CheckpointManager

**Why start here**: Simplest extraction, clear boundaries

1. Create file:
```bash
touch src/training/trainers/multimodal/checkpoint_manager.py
```

2. Extract all checkpoint-related methods:
```python
# src/training/trainers/multimodal/checkpoint_manager.py
class CheckpointManager:
    """Manages model checkpointing."""

    def __init__(self, model, optimizer, save_dir):
        self.model = model
        self.optimizer = optimizer
        self.save_dir = save_dir

    def save_checkpoint(self, epoch, metrics):
        """Save model checkpoint."""
        checkpoint = {
            'epoch': epoch,
            'model_state': self.model.state_dict(),
            'optimizer_state': self.optimizer.state_dict(),
            'metrics': metrics
        }
        path = f"{self.save_dir}/checkpoint_epoch_{epoch}.pt"
        torch.save(checkpoint, path, _use_new_zipfile_serialization=True)

    def load_checkpoint(self, path):
        """Load model checkpoint."""
        checkpoint = torch.load(path, map_location='cpu', weights_only=False)
        self.model.load_state_dict(checkpoint['model_state'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state'])
        return checkpoint['epoch'], checkpoint['metrics']

    def get_latest_checkpoint(self):
        """Find the most recent checkpoint."""
        # Implementation
        pass
```

3. Update main trainer to use it:
```python
# In trainer.py
from .checkpoint_manager import CheckpointManager

class MultimodalTrainer:
    def __init__(self, ...):
        ...
        self.checkpoint_mgr = CheckpointManager(model, optimizer, save_dir)

    def save_checkpoint(self, epoch, metrics):
        self.checkpoint_mgr.save_checkpoint(epoch, metrics)
```

4. Test:
```bash
pytest tests/test_checkpoint_manager.py -v
```

#### Day 2: Extract MetricsCollector

**Why next**: Also clear boundaries, helps simplify trainer

```python
# src/training/trainers/multimodal/metrics_collector.py
class MetricsCollector:
    """Collects and tracks training metrics."""

    def __init__(self):
        self.metrics = defaultdict(list)
        self.current_batch_metrics = {}

    def update(self, **kwargs):
        """Update metrics for current batch."""
        self.current_batch_metrics.update(kwargs)

    def end_epoch(self):
        """Aggregate epoch metrics."""
        for key, value in self.current_batch_metrics.items():
            self.metrics[key].append(value)
        self.current_batch_metrics = {}
        return {k: v[-1] for k, v in self.metrics.items()}

    def get_history(self, metric_name):
        """Get full history of a metric."""
        return self.metrics[metric_name]
```

#### Days 3-4: Extract TrainingLoop

**Most complex extraction** - requires careful planning

Strategy:
1. Create skeleton with all method signatures
2. Move logic method by method
3. Test after each method migration
4. Keep old file alongside new temporarily

```python
# src/training/trainers/multimodal/training_loop.py
class TrainingLoop:
    """Handles the core training loop execution."""

    def __init__(self, model, loss_fn, device):
        self.model = model
        self.loss_fn = loss_fn
        self.device = device

    def train_epoch(self, dataloader, optimizer, epoch):
        """Train for one epoch."""
        self.model.train()
        epoch_loss = 0.0

        for batch_idx, batch in enumerate(dataloader):
            loss = self.train_step(batch, optimizer)
            epoch_loss += loss.item()

            if batch_idx % 100 == 0:
                self._log_progress(epoch, batch_idx, loss)

        return epoch_loss / len(dataloader)

    def train_step(self, batch, optimizer):
        """Single training step."""
        # Move batch to device
        batch = self._move_to_device(batch)

        # Forward pass
        optimizer.zero_grad()
        outputs = self.model(**batch)
        loss = self.loss_fn(outputs, batch['labels'])

        # Backward pass
        loss.backward()
        optimizer.step()

        return loss

    def _move_to_device(self, batch):
        """Move batch tensors to device."""
        return {k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                for k, v in batch.items()}

    def _log_progress(self, epoch, batch_idx, loss):
        """Log training progress."""
        logger.info(f"Epoch {epoch}, Batch {batch_idx}, Loss: {loss.item():.4f}")
```

Break down the complex `train_epoch()` using this strategy:
- Extract batch processing → `train_step()`
- Extract device management → `_move_to_device()`
- Extract logging → `_log_progress()`
- Extract validation calls → separate method

#### Days 5-6: Remaining Extractions

- Day 5: Extract `Evaluation` module
- Day 6: Extract `DataHandler` module

#### Days 7-8: Refactor Main Trainer

Now `trainer.py` becomes a clean coordinator:

```python
# src/training/trainers/multimodal/trainer.py
class MultimodalTrainer:
    """Main multimodal trainer - coordinates training workflow."""

    def __init__(self, model, config):
        self.model = model
        self.config = config

        # Initialize components
        self.training_loop = TrainingLoop(model, config.loss_fn, config.device)
        self.evaluator = Evaluator(model, config.eval_metrics, config.device)
        self.checkpoint_mgr = CheckpointManager(model, config.optimizer, config.save_dir)
        self.metrics = MetricsCollector()
        self.data_handler = DataHandler(config)

    def train(self, num_epochs):
        """Main training method - simple coordinator."""
        for epoch in range(num_epochs):
            # Training
            train_loss = self.training_loop.train_epoch(
                self.data_handler.train_loader,
                self.config.optimizer,
                epoch
            )

            # Evaluation
            eval_metrics = self.evaluator.evaluate(self.data_handler.val_loader)

            # Collect metrics
            self.metrics.update(train_loss=train_loss, **eval_metrics)

            # Checkpoint
            if epoch % self.config.checkpoint_every == 0:
                self.checkpoint_mgr.save_checkpoint(epoch, self.metrics.end_epoch())

        return self.metrics.get_history('train_loss')
```

#### Days 9-10: Testing & Documentation

- Write tests for each new module
- Update existing tests to use new structure
- Add documentation
- Create migration guide for contributors

### Testing Strategy

**For each extracted module**:

1. Unit tests:
```python
# tests/test_checkpoint_manager.py
def test_save_and_load_checkpoint():
    # Create manager
    mgr = CheckpointManager(model, optimizer, tmpdir)

    # Save
    mgr.save_checkpoint(epoch=1, metrics={'loss': 0.5})

    # Load
    epoch, metrics = mgr.load_checkpoint(f"{tmpdir}/checkpoint_epoch_1.pt")

    assert epoch == 1
    assert metrics['loss'] == 0.5
```

2. Integration tests:
```python
# tests/test_multimodal_trainer.py
def test_full_training_loop():
    trainer = MultimodalTrainer(model, config)
    metrics = trainer.train(num_epochs=2)
    assert len(metrics) == 2
    assert metrics[-1] < metrics[0]  # Loss should decrease
```

### Rollout Strategy

**Gradual migration** to minimize risk:

1. **Week 3**: Create new modules alongside old file
2. **Week 3-4**: Migrate functionality, keep both versions
3. **Week 4**: Switch to new version, deprecate old
4. **Week 5**: Remove old file after validation

Use feature flags:
```python
# config.py
USE_REFACTORED_TRAINER = os.getenv('USE_REFACTORED_TRAINER', 'true') == 'true'

# In code:
if USE_REFACTORED_TRAINER:
    from .multimodal import MultimodalTrainer
else:
    from .multimodal_trainer import MultimodalTrainer  # Old version
```

---

## Strategy 2: Consolidate Loss Functions

**Goal**: Reduce 21 loss files (35% duplication) → 8-10 well-designed classes
**Timeline**: Weeks 3-4 (parallel with trainer refactoring)
**Effort**: 40-50 hours

### Current State

21 loss function files with extensive code duplication:
- 10+ files implement feature normalization identically
- 8+ files implement temperature scaling identically
- Similar contrastive loss patterns repeated

### Target Structure

```
src/training/losses/
├── __init__.py
├── loss_registry.py               # Factory + registration
├── base/
│   ├── __init__.py
│   ├── base_contrastive.py       # Shared contrastive logic
│   ├── base_supervised.py        # Supervised patterns
│   └── mixins.py                 # Temperature, normalization mixins
├── contrastive/
│   ├── __init__.py
│   ├── clip_loss.py              # CLIP-style contrastive
│   ├── vicreg_loss.py            # VICReg
│   ├── barlow_twins_loss.py      # Barlow Twins
│   └── simclr_loss.py            # SimCLR-style
├── multimodal/
│   ├── __init__.py
│   ├── cross_modal_loss.py
│   └── fusion_loss.py
└── supervised/
    ├── __init__.py
    ├── cross_entropy_loss.py
    └── supervised_contrastive_loss.py
```

### Step-by-Step Migration

#### Step 1: Create Base Classes (Day 1)

Extract common patterns into base class:

```python
# src/training/losses/base/base_contrastive.py
class BaseContrastiveLoss(nn.Module):
    """Base class for all contrastive losses."""

    def __init__(self, temperature=0.07, normalize=True):
        super().__init__()
        self.temperature = temperature
        self.normalize = normalize

    def normalize_features(self, features):
        """L2 normalization of features."""
        if self.normalize:
            return F.normalize(features, p=2, dim=1)
        return features

    def compute_similarity(self, features1, features2):
        """Compute cosine similarity matrix."""
        features1 = self.normalize_features(features1)
        features2 = self.normalize_features(features2)
        return torch.matmul(features1, features2.T) / self.temperature

    def forward(self, features1, features2, labels=None):
        """Override in subclasses."""
        raise NotImplementedError
```

####Step 2: Create Mixins for Shared Functionality (Day 1)

```python
# src/training/losses/base/mixins.py
class TemperatureScalingMixin:
    """Mixin for temperature-scaled losses."""

    def __init__(self, *args, temperature=0.07, learnable_temp=False, **kwargs):
        super().__init__(*args, **kwargs)
        if learnable_temp:
            self.temperature = nn.Parameter(torch.tensor(temperature))
        else:
            self.temperature = temperature

class ProjectionMixin:
    """Mixin for feature projection."""

    def __init__(self, *args, input_dim, projection_dim, **kwargs):
        super().__init__(*args, **kwargs)
        self.projection = nn.Sequential(
            nn.Linear(input_dim, projection_dim),
            nn.ReLU(),
            nn.Linear(projection_dim, projection_dim)
        )

    def project_features(self, features):
        return self.projection(features)
```

#### Step 3: Migrate Individual Losses (Days 2-8)

Migrate 3 losses per day, starting with simplest:

**Day 2**: CLIP, SimCLR (simple contrastive)
**Day 3**: VICReg, Barlow Twins
**Day 4**: Supervised contrastive, Cross-modal
**Day 5**: Complex variants (MoCo, queue-based)
**Day 6**: Multimodal fusion losses
**Day 7**: Remaining losses
**Day 8**: Testing & validation

Example migration:
```python
# src/training/losses/contrastive/clip_loss.py
from ..base import BaseContrastiveLoss

class CLIPLoss(BaseContrastiveLoss):
    """CLIP-style contrastive loss."""

    def forward(self, image_features, text_features):
        # Normalize
        image_features = self.normalize_features(image_features)
        text_features = self.normalize_features(text_features)

        # Compute similarity
        logits = self.compute_similarity(image_features, text_features)

        # Contrastive loss (both directions)
        labels = torch.arange(len(logits), device=logits.device)
        loss_i2t = F.cross_entropy(logits, labels)
        loss_t2i = F.cross_entropy(logits.T, labels)

        return (loss_i2t + loss_t2i) / 2
```

#### Step 4: Loss Registry (Day 9)

```python
# src/training/losses/loss_registry.py
_LOSS_REGISTRY = {}

def register_loss(name):
    """Decorator to register loss functions."""
    def decorator(cls):
        _LOSS_REGISTRY[name] = cls
        return cls
    return decorator

def get_loss(name, **kwargs):
    """Factory function to get loss by name."""
    if name not in _LOSS_REGISTRY:
        raise ValueError(f"Unknown loss: {name}")
    return _LOSS_REGISTRY[name](**kwargs)

# Usage:
@register_loss('clip')
class CLIPLoss(BaseContrastiveLoss):
    pass
```

See [code-patterns.md](code-patterns.md) for complete pattern examples.

---

## Strategy 3: Establish BaseTrainer Pattern

**Goal**: Eliminate 60% duplication across 8 trainer types
**Timeline**: Weeks 5-6
**Effort**: 30-40 hours

Covered in detail in [code-patterns.md](code-patterns.md) - Template Method Pattern section.

---

## Risk Mitigation

### Risk 1: Breaking Existing Functionality

**Mitigation**:
1. Keep old code alongside new (parallel development)
2. Comprehensive tests before migration
3. Feature flags for gradual rollout
4. Integration tests for each component

### Risk 2: Merge Conflicts

**Mitigation**:
1. Work in feature branches
2. Frequent small PRs
3. Clear code ownership
4. Daily standup coordination

### Risk 3: Performance Regression

**Mitigation**:
1. Benchmark before/after
2. Profile hot paths
3. Optimize if needed
4. Monitor in production

---

## Success Criteria

After refactoring:

✅ No file >800 lines
✅ All functions complexity <15
✅ Code duplication <5%
✅ All tests passing
✅ Performance maintained or improved
✅ Documentation complete

---

For code examples and patterns, see [code-patterns.md](code-patterns.md).
For overall plan, see [README.md](README.md).
