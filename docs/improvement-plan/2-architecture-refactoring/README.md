# Axis 2: Architecture Refactoring

**Timeline**: Weeks 3-6
**Effort**: 234-293 hours
**Priority**: 🟠 HIGH

## Overview

Refactor the core architecture to eliminate God objects, reduce code duplication, and establish maintainable patterns. This builds on the stable foundation from Axis 1.

## Current State

- **Architecture Score**: 5.5/10 (Needs Refactoring)
- **Largest File**: 2,927 lines (`multimodal_trainer.py`)
- **Loss Function Files**: 21 with 35% duplication
- **Trainer Duplication**: 60% across 8 classes
- **Configuration Approaches**: 4 different inconsistent patterns

## Target State (After Week 6)

- **Architecture Score**: 7.5/10 ✅
- **Largest File**: <800 lines ✅
- **Loss Function Files**: 8-10 well-designed classes ✅
- **Trainer Duplication**: <10% (using BaseTrainer) ✅
- **Configuration**: Single Pydantic-based system ✅

---

## Week 3-4: Split God Objects (80-100 hours)

### 1. Decompose multimodal_trainer.py (40-50 hours)

**Problem**: 2,927 lines, complexity 93 in `train_epoch()`, violates Single Responsibility Principle

**Strategy**: Extract into 6 focused modules

**New Structure**:
```
src/training/trainers/multimodal/
├── __init__.py
├── trainer.py                  (~400 lines - main coordinator)
├── training_loop.py            (~500 lines - epoch/batch training)
├── evaluation.py               (~400 lines - evaluation logic)
├── checkpoint_manager.py       (~200 lines - save/load/resume)
├── metrics_collector.py        (~300 lines - metrics tracking)
└── data_handler.py             (~300 lines - data loading/batching)
```

**Week 3 Focus**:
- [ ] Extract `checkpoint_manager.py` (simplest, 1 day)
- [ ] Extract `metrics_collector.py` (1 day)
- [ ] Create `training_loop.py` skeleton (2 days)

**Week 4 Focus**:
- [ ] Complete `training_loop.py` (3 days)
- [ ] Extract `evaluation.py` (2 days)
- [ ] Refactor main `trainer.py` to use extracted modules (2 days)

**Success Criteria**:
- No file >800 lines
- All tests pass
- Coverage maintained or improved

**See**: `refactoring-strategy.md` for detailed migration plan

---

### 2. Consolidate Loss Functions (40-50 hours)

**Problem**: 21 loss files with 35% code duplication

**Strategy**: Create hierarchy with shared base classes

**Target Structure**:
```
src/training/losses/
├── __init__.py
├── loss_registry.py            (factory + registration)
├── base/
│   ├── base_contrastive.py    (shared contrastive logic)
│   ├── base_supervised.py     (supervised patterns)
│   └── mixins.py              (temperature, normalization, etc.)
├── contrastive/
│   ├── clip_loss.py           (CLIP-style)
│   ├── vicreg_loss.py         (VICReg)
│   ├── barlow_twins_loss.py   (Barlow Twins)
│   └── simclr_loss.py         (SimCLR-style)
├── multimodal/
│   ├── cross_modal_loss.py
│   └── fusion_loss.py
└── supervised/
    ├── cross_entropy_loss.py
    └── supervised_contrastive_loss.py
```

**Week 3-4 Tasks**:
- [ ] Create `base_contrastive.py` with shared logic (3 days)
- [ ] Migrate 5 loss functions to new structure (4 days)
- [ ] Update loss registry (1 day)
- [ ] Deprecate old loss files (don't delete yet) (1 day)
- [ ] Update all imports in trainers (1 day)

**Reduction**: 21 files → 8-10 files, 35% duplication → <5%

**See**: `refactoring-strategy.md` section 2

---

## Week 5-6: Establish Patterns (154-193 hours)

### 3. Implement BaseTrainer Pattern (30-40 hours)

**Problem**: 60% code duplication across 8 trainer types

**Solution**: Template method pattern with hooks

**BaseTrainer Implementation**:
```python
class BaseTrainer:
    """Base trainer with template method pattern."""

    def train_epoch(self):
        """Template method - same for all trainers."""
        self.on_epoch_start()

        for batch in self.dataloader:
            loss = self.training_step(batch)  # Hook - override in subclass
            self.backward(loss)
            self.optimizer_step()

        metrics = self.on_epoch_end()
        return metrics

    def training_step(self, batch):
        """Hook method - override in subclasses."""
        raise NotImplementedError

    def on_epoch_start(self):
        """Hook for subclass customization."""
        pass
```

**Migration Plan**:
- [ ] Week 5: Implement BaseTrainer with all common methods
- [ ] Week 5: Migrate MultimodalTrainer to use BaseTrainer
- [ ] Week 6: Migrate remaining 7 trainers
- [ ] Week 6: Remove duplicated code

**Reduction**: 60% duplication → <10%

**See**: `code-patterns.md` for template method pattern

---

### 4. Unify Configuration Management (20-25 hours)

**Problem**: 4 different configuration approaches:
- Dataclasses in `src/configs/`
- Argparse in demo scripts
- Dict-based configs
- Hard-coded values

**Solution**: Single Pydantic-based configuration system

**New Structure**:
```python
# src/configs/base_config.py
from pydantic import BaseModel, Field

class BaseConfig(BaseModel):
    """Base configuration with validation."""

    class Config:
        extra = "forbid"  # Reject unknown fields
        validate_assignment = True

class ModelConfig(BaseConfig):
    hidden_dim: int = Field(768, ge=64, le=4096)
    num_layers: int = Field(12, ge=1, le=48)
    dropout: float = Field(0.1, ge=0.0, le=1.0)

class TrainingConfig(BaseConfig):
    batch_size: int = Field(32, ge=1)
    learning_rate: float = Field(1e-4, gt=0.0)
    epochs: int = Field(10, ge=1)

class Config(BaseConfig):
    """Main configuration."""
    model: ModelConfig = Field(default_factory=ModelConfig)
    training: TrainingConfig = Field(default_factory=TrainingConfig)
```

**Migration Tasks**:
- [ ] Create Pydantic config classes (2 days)
- [ ] Migrate all hardcoded values to config (2 days)
- [ ] Update trainers to use unified config (2 days)
- [ ] Update demos to use unified config (1 day)
- [ ] Add config validation tests (1 day)

**Benefits**:
- Type validation
- Default values
- Environment variable support
- JSON/YAML loading
- Documentation from code

**See**: `code-patterns.md` section 4

---

### 5. Improve Test Coverage (64-83 hours)

**Target**: 45% → 65% coverage

**Focus Areas**:

**Loss Functions** (30 hours):
- [ ] Test remaining 15 loss functions (10 already done in Axis 1)
- [ ] Test edge cases (empty batches, single samples)
- [ ] Test gradient flow
- [ ] Property-based tests for invariants

**Trainers** (20 hours):
- [ ] Test MultimodalTrainer (refactored version)
- [ ] Test BaseTrainer hooks
- [ ] Integration tests for full training loop
- [ ] Test checkpoint save/load

**Data Pipeline** (14 hours):
- [ ] Test augmentation pipeline
- [ ] Test dataset loading
- [ ] Test tokenization edge cases

**See**: `../3-testing-and-quality/coverage-roadmap.md`

---

## Week 6: Polish & Integration

### Code Quality Improvements
- [ ] Fix all PEP 8 violations (flake8)
- [ ] Add type hints to new code (100% coverage)
- [ ] Run mypy strict mode
- [ ] Update docstrings

### Documentation
- [ ] Document new architecture
- [ ] Create architecture diagrams
- [ ] Update CLAUDE.md with patterns
- [ ] Add migration guide for contributors

### Integration Testing
- [ ] Full training run on toy dataset
- [ ] Verify all trainers work with new config
- [ ] Verify all loss functions work
- [ ] Performance benchmarks (ensure no regression)

---

## Success Metrics

After completing Axis 2, you should have:

✅ **Architecture Score**: 5.5/10 → 7.5/10
✅ **Largest File**: 2,927 → <800 lines
✅ **Loss Files**: 21 → 8-10 with clear hierarchy
✅ **Code Duplication**: 35% → <5% (losses), 60% → <10% (trainers)
✅ **Test Coverage**: 45% → 65%
✅ **Configuration**: Unified Pydantic-based system
✅ **Development Velocity**: 30-40% faster (measured by PR merge time)

---

## Risk Mitigation

**Risk**: Breaking existing functionality during refactoring

**Mitigation**:
1. ✅ Comprehensive tests from Axis 1
2. Keep old code alongside new (deprecate, don't delete)
3. Feature flags for gradual rollout
4. Integration tests at each step
5. Daily smoke tests on toy dataset

**Risk**: Merge conflicts if multiple people working

**Mitigation**:
1. Work in feature branches
2. Frequent small merges
3. Clear ownership per module
4. Daily standups to coordinate

---

## Next Steps

Once Axis 2 is complete, proceed to:
- **Axis 3**: Testing & Quality (Weeks 7-10)

---

## Documents in This Axis

- **README.md** (this file) - Overview and action items
- **architecture-review.md** - Detailed architecture analysis with specific issues
- **refactoring-strategy.md** - Step-by-step migration guide for each component
- **code-patterns.md** - Reusable patterns (template method, strategy, factory, etc.)

## Related Documentation

- `../1-security-and-stability/` - Foundation work (must complete first)
- `../3-testing-and-quality/` - Testing patterns for validation
- `../diagrams/` - Visual architecture diagrams

---

**Questions?** See `refactoring-strategy.md` for detailed step-by-step migration plans with code examples.
