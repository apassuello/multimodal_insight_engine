# Coverage Improvement Roadmap

**Project**: Multimodal Insight Engine
**Generated**: 2025-12-18
**Current Coverage**: 35% (421 passing tests, 12 skipped)
**Target Coverage**: 55-60% (by Week 4)
**Ultimate Goal**: 70%+ (by Month 3)

---

## Table of Contents

1. [Executive Summary](#executive-summary)
2. [Coverage Gap Analysis](#coverage-gap-analysis)
3. [4-Week Sprint Plan](#4-week-sprint-plan)
4. [Test Templates](#test-templates)
5. [Refactoring Guides](#refactoring-guides)
6. [CI/CD Integration](#cicd-integration)
7. [Success Metrics](#success-metrics)

---

## Executive Summary

### Current State

**Coverage Analysis** (from CI report):
- **Total Lines**: ~12,000 lines of code
- **Covered**: 4,200 lines (35%)
- **Uncovered**: 7,800 lines (65%)
- **Files with < 30% coverage**: 79 files
- **Files with 0% coverage**: 24 files

**Test Health**:
- ✅ 421 tests passing
- ⚠️ 12 tests skipped (recently reduced from 55+)
- 🔴 55 total skip scenarios identified across all test files

**Risk Assessment**:
- 🔴 **Critical**: Training loop (6% coverage) - bugs corrupt model weights
- 🔴 **Critical**: Loss factory (27% coverage) - silent learning failures
- 🟡 **High**: Model factory (7% coverage) - initialization bugs
- 🟡 **High**: 3 core utilities (0% coverage) - gradient handling, LR scheduling, metrics

### The Plan

**Strategy**: Focus on **ROI** (Return on Investment) = (Impact × Risk) / Effort

**4-Week Phased Approach**:
- **Week 1**: Quick wins (+10% coverage, 55 tests)
- **Week 2-3**: High-value infrastructure (+12% coverage, 60 tests)
- **Week 4**: Begin refactoring critical components (+6% coverage)
- **Month 2-3**: Complete refactoring, integration tests

**Expected Outcomes**:
- **Coverage**: 35% → 55-60% (+20-25% improvement)
- **Tests**: 421 → 600+ tests (+180 tests)
- **Risk**: Critical untested code reduced by 70%
- **Architecture**: 3 God objects refactored into 9 focused classes

---

## Coverage Gap Analysis

### Critical Files Requiring Immediate Attention

| Component | File | Coverage | Risk | Why Critical |
|-----------|------|----------|------|--------------|
| **Training Loop** | transformer_trainer.py | 6% | 🔴 Critical | Bugs silently corrupt model weights, training metrics |
| **Loss Creation** | loss_factory.py | 27% | 🔴 Critical | Wrong loss = models train but never learn (silent failure) |
| **Model Creation** | model_factory.py | 7% | 🔴 Critical | Initialization bugs cause immediate visible failures |
| **LR Scheduling** | learningrate_scheduler.py | 0% | 🟡 High | Wrong LR destroys training convergence |
| **Gradient Handling** | gradient_handler.py | 0% | 🟡 High | Gradient explosions, modality imbalance |
| **Metrics Tracking** | metrics_tracker.py | 0% | 🟡 High | Wrong metrics mislead training decisions |

### Testing Blockers Identified

**Blocker 1: Test File Not Running**
- **File**: `tests/test_wmt_dataloader.py` (170 lines)
- **Status**: Test exists but coverage shows 0%
- **Impact**: Immediate +5% coverage when fixed
- **Investigation needed**: pytest discovery, import paths, skip decorators

**Blocker 2: Architectural Anti-Patterns**
- **God Objects**: Classes with 20+ responsibilities (transformer_trainer.py)
- **No Dependency Injection**: Hardcoded dependencies (timm, transformers)
- **Args Mutation**: Functions mutate input args as side effect
- **Mixed Concerns**: Business logic + I/O + visualization in same class

**Blocker 3: External Dependencies**
- **HuggingFace datasets**: Integration tests require network/downloads
- **GPU requirements**: 6 tests require CUDA (quantization, mixed precision)
- **Optional libraries**: TensorFlow for wikipedia_dataset.py

### Dead Code Identified

**Immediate Archive Candidates**:
1. `src/data/image_dataset.py` (177 lines) - No imports found
2. `src/utils/feature_attribution.py` (586 lines) - Interpretability, unused
3. Potentially: Old training scripts (flickr_multistage, joint_bpe) if demos aren't used

**Action**: Move to `archived/` directory, update documentation

---

## 4-Week Sprint Plan

### Week 1: Quick Wins (Target: +10% coverage)

**Goal**: Test high-ROI files (ROI > 8, Effort ≤ 2)
**Coverage Target**: 35% → 45%
**Tests Added**: 55 tests

#### Day 1: Fix Existing Test + Easy Pure Functions

**Morning (4h)**:
```bash
# TASK 1.1: Investigate wmt_dataloader test failure (CRITICAL)
python -m pytest tests/test_wmt_dataloader.py -v --tb=short
# Expected: Test runs but shows 0% coverage OR doesn't discover tests
# If import error: Fix import paths
# If skip: Remove skip decorator/condition
# Expected gain: +5% coverage (170 lines of tests exist!)

# TASK 1.2: Start learningrate_scheduler tests (pure math, easy)
# Create tests/test_learningrate_scheduler.py
```

**Test Plan for learningrate_scheduler.py**:
```python
# tests/test_learningrate_scheduler.py (10 tests)
# 1. test_warmup_cosine_scheduler_warmup_phase (LR increases linearly)
# 2. test_warmup_cosine_scheduler_cosine_phase (LR decreases with cosine)
# 3. test_warmup_cosine_scheduler_min_lr (doesn't go below min_lr)
# 4. test_warmup_linear_scheduler_warmup_phase
# 5. test_warmup_linear_scheduler_linear_phase
# 6. test_constant_scheduler (LR stays constant)
# 7. test_inverse_sqrt_scheduler (follows 1/sqrt formula)
# 8. test_scheduler_step_updates_lr
# 9. test_scheduler_get_last_lr
# 10. test_scheduler_state_dict_save_load
```

**Afternoon (4h)**:
```bash
# TASK 1.3: Complete learningrate_scheduler tests
# Run: python -m pytest tests/test_learningrate_scheduler.py -v
# Expected gain: +3% coverage (356 lines tested)
```

**Deliverables**:
- ✅ wmt_dataloader.py test fixed (+5%)
- ✅ learningrate_scheduler.py fully tested (+3%)
- **Day 1 Total**: +8% coverage

---

#### Day 2-3: Gradient Handler + Dataset Utilities

**Day 2 (8h)**:
```bash
# TASK 2.1: Test gradient_handler.py (469 lines)
# Create tests/test_gradient_handler.py (15 tests)
```

**Test Plan for gradient_handler.py**:
```python
# tests/test_gradient_handler.py (15 tests)
# 1. test_clip_gradients_by_norm (gradient clipping)
# 2. test_clip_gradients_by_value
# 3. test_compute_gradient_norm (norm calculation)
# 4. test_gradient_monitoring_detects_explosion
# 5. test_gradient_monitoring_detects_vanishing
# 6. test_modality_balancing_vision_text
# 7. test_modality_balancing_weights_sum_to_one
# 8. test_gradient_handler_with_none_gradients (edge case)
# 9. test_gradient_handler_mixed_precision_compatibility
# 10. test_log_gradient_statistics
# 11. test_gradient_accumulation_steps
# 12. test_zero_gradients_handling
# 13. test_gradient_checkpointing_compatibility
# 14. test_multiple_optimizers_support
# 15. test_gradient_handler_state_persistence
```

**Mocking Strategy**:
```python
# Mock PyTorch optimizer and model
import torch.nn as nn
from unittest.mock import Mock, MagicMock

@pytest.fixture
def mock_model():
    model = Mock(spec=nn.Module)
    model.parameters.return_value = [
        torch.nn.Parameter(torch.randn(10, 10)),
        torch.nn.Parameter(torch.randn(10))
    ]
    return model
```

**Day 3 (8h)**:
```bash
# TASK 3.1: Test dataset_wrapper.py (165 lines, 8 tests)
# TASK 3.2: Test fixed_semantic_sampler.py (310 lines, 12 tests)
```

**Deliverables**:
- ✅ gradient_handler.py tested (+4%)
- ✅ dataset_wrapper.py tested (+1%)
- ✅ fixed_semantic_sampler.py tested (+2%)
- **Days 2-3 Total**: +7% coverage

---

#### Day 4-5: Contrastive Learning Utils

**Day 4-5 (16h)**:
```bash
# TASK 4.1: Test contrastive_learning.py (229 lines, 10 tests)
# Already at 7%, bring to 75%+
```

**Test Plan for contrastive_learning.py**:
```python
# tests/test_contrastive_learning_utils.py (10 tests)
# 1. test_nt_xent_loss_basic (NT-Xent loss computation)
# 2. test_nt_xent_loss_temperature_sensitivity
# 3. test_nt_xent_loss_batch_size_invariance
# 4. test_nt_xent_loss_gradient_flow
# 5. test_supervised_contrastive_loss_basic
# 6. test_supervised_contrastive_loss_with_labels
# 7. test_supervised_contrastive_loss_label_propagation
# 8. test_compute_recall_at_k_basic
# 9. test_compute_recall_at_k_multiple_k_values
# 10. test_compute_recall_at_k_edge_cases (all correct, none correct)
```

**Deliverables**:
- ✅ contrastive_learning.py tested to 75%+ (+2%)
- **Week 1 Total**: +10% coverage (35% → 45%)

---

### Week 2: High-Value Infrastructure (Target: +6%)

**Goal**: Test core factories and utilities
**Coverage Target**: 45% → 51%
**Tests Added**: 30 tests

#### Day 6-7: Easy Wins First

**Tasks**:
```bash
# TASK 6.1: Test argument_configs.py (70 lines, 4 tests)
# Very simple, pure configuration
# Expected: +0.5% coverage

# TASK 6.2: Test joint_bpe_training.py (131 lines, 5 tests)
# Thin wrapper around BPETokenizer
# Expected: +1% coverage

# TASK 6.3: Start metrics_tracker.py (663 lines, 15 tests)
# Mock file I/O (save_metrics, plot_metrics)
# Expected: +5% coverage
```

**Deliverables**:
- ✅ argument_configs.py tested (+0.5%)
- ✅ joint_bpe_training.py tested (+1%)
- ✅ metrics_tracker.py tested (+5%)
- **Days 6-7 Total**: +6.5% coverage

---

#### Day 8-10: Factory Refactoring + Testing

**Day 8: Refactor loss_factory.py (8h)**

**Current Problem**:
```python
# src/training/losses/loss_factory.py (470 lines in one function!)
def create_loss_function(args, dataset_size, train_loader):
    # Args mutation (BAD!)
    args.fusion_dim = fusion_dim

    # 12+ loss type branches in one function
    if args.loss_type == "barlow_twins":
        # 20 lines
    elif args.loss_type == "vicreg":
        # 100+ lines with curriculum logic
    elif args.loss_type == "memory_queue":
        # 30 lines
    # ... 9 more branches
```

**Refactoring Strategy**:
```python
# src/training/losses/loss_factory.py (refactored)
from abc import ABC, abstractmethod
from dataclasses import dataclass

@dataclass
class LossConfig:
    """Immutable loss configuration (no args mutation!)"""
    loss_type: str
    temperature: float
    fusion_dim: int
    # ... other params

class LossFactory:
    """Factory with strategy pattern"""

    def __init__(self):
        self._strategies = {
            "barlow_twins": BarlowTwinsStrategy(),
            "vicreg": VICRegStrategy(),
            "memory_queue": MemoryQueueStrategy(),
            # ...
        }

    def create(self, config: LossConfig) -> nn.Module:
        strategy = self._strategies.get(config.loss_type)
        if not strategy:
            raise ValueError(f"Unknown loss type: {config.loss_type}")
        return strategy.create_loss(config)

class LossStrategy(ABC):
    @abstractmethod
    def create_loss(self, config: LossConfig) -> nn.Module:
        pass

class BarlowTwinsStrategy(LossStrategy):
    def create_loss(self, config: LossConfig) -> nn.Module:
        return BarlowTwinsLoss(
            dim=config.fusion_dim,
            lambd=config.barlow_lambda
        )

# Similar for other 11 loss types...
```

**Benefits**:
- ✅ No args mutation
- ✅ Each strategy is independently testable
- ✅ Open/Closed principle (easy to add new loss types)
- ✅ Single Responsibility (each strategy creates one loss type)

**Day 9-10: Test refactored factory (16h)**

**Test Plan**:
```python
# tests/test_loss_factory.py (20+ tests)
# 1. test_factory_creates_barlow_twins_loss
# 2. test_factory_creates_vicreg_loss
# 3. test_factory_creates_vicreg_with_curriculum
# ... (one test per loss type)
# 15. test_factory_raises_on_unknown_loss_type
# 16. test_loss_config_immutability
# 17. test_strategy_pattern_extensibility
# 18. test_all_strategies_registered
# 19. test_loss_creation_with_minimal_config
# 20. test_loss_creation_with_full_config
```

**Deliverables**:
- ✅ loss_factory.py refactored (Day 8)
- ✅ loss_factory.py tested to 75%+ (Days 9-10, +4%)
- **Week 2 Total**: +10.5% coverage (45% → 55.5%)

---

### Week 3: Model Factory + Remaining High-Value (Target: +4%)

**Goal**: Complete Tier 2, prepare for refactoring
**Coverage Target**: 55.5% → 59.5%
**Tests Added**: 30 tests

#### Day 11-13: Model Factory with Mocking

**Mocking Strategy for External Dependencies**:
```python
# tests/test_model_factory.py

import pytest
from unittest.mock import Mock, patch, MagicMock

@pytest.fixture
def mock_timm():
    """Mock timm library to avoid downloading pretrained weights"""
    with patch('src.models.model_factory.timm') as mock:
        mock.create_model.return_value = Mock()
        yield mock

@pytest.fixture
def mock_transformers():
    """Mock HuggingFace transformers"""
    with patch('src.models.model_factory.transformers') as mock:
        mock.AutoModel.from_pretrained.return_value = Mock()
        yield mock

def test_create_vision_model_timm(mock_timm):
    """Test vision model creation using timm"""
    from src.models.model_factory import create_multimodal_model

    config = ModelConfig(
        vision_model_name="vit_base_patch16_224",
        text_model_name="albert-base-v2",
        fusion_dim=512
    )

    model = create_multimodal_model(config)

    mock_timm.create_model.assert_called_once_with(
        "vit_base_patch16_224",
        pretrained=True
    )
    assert model is not None
```

**Test Plan for model_factory.py** (15 tests):
```python
# 1. test_create_vision_model_timm
# 2. test_create_vision_model_custom
# 3. test_create_text_model_transformers
# 4. test_create_text_model_custom
# 5. test_create_fusion_module_concatenation
# 6. test_create_fusion_module_cross_attention
# 7. test_create_multimodal_model_full
# 8. test_model_device_compatibility_cpu
# 9. test_model_device_compatibility_mps
# 10. test_model_handles_missing_timm
# 11. test_model_handles_missing_transformers
# 12. test_model_config_validation
# 13. test_model_parameter_initialization
# 14. test_model_forward_pass_shape
# 15. test_model_gradient_flow
```

**Deliverables**:
- ✅ model_factory.py tested to 70%+ (+4%)
- **Week 3 Total**: +4% coverage (55.5% → 59.5%)

---

### Week 4: Begin Critical Refactoring (Target: +6%)

**Goal**: Extract components from transformer_trainer.py
**Coverage Target**: 59.5% → 65%
**Refactoring**: Extract 3 classes

#### Day 14-16: Extract CheckpointManager

**Current Problem**:
```python
# transformer_trainer.py (367 lines, God object)
class TransformerTrainer:
    def save_checkpoint(self, epoch, is_best=False):
        # 99 lines of checkpoint logic
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            # ... 15 more fields
        }
        torch.save(checkpoint, path)
        if is_best:
            shutil.copy(path, best_path)
        # ... more logic

    def load_checkpoint(self, path):
        # 82 lines of restoration logic
```

**Refactored Design**:
```python
# src/training/checkpoint_manager.py (NEW FILE)
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Any, Optional
import torch
import shutil

@dataclass
class CheckpointConfig:
    """Configuration for checkpoint management"""
    checkpoint_dir: Path
    save_frequency: int = 1
    keep_last_n: int = 3
    save_best: bool = True
    metric_for_best: str = "val_loss"
    metric_mode: str = "min"  # "min" or "max"

class CheckpointManager:
    """
    Manages model checkpoints with automatic cleanup and best model tracking.

    Single Responsibility: Checkpoint persistence and retrieval.
    """

    def __init__(self, config: CheckpointConfig):
        self.config = config
        self.config.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.best_metric_value = float('inf') if config.metric_mode == 'min' else float('-inf')

    def save(
        self,
        epoch: int,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        metrics: Dict[str, float],
        scheduler: Optional[Any] = None,
        **extra_state
    ) -> Path:
        """Save checkpoint and handle best model tracking."""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'metrics': metrics,
            **extra_state
        }

        if scheduler is not None:
            checkpoint['scheduler_state_dict'] = scheduler.state_dict()

        # Save regular checkpoint
        checkpoint_path = self.config.checkpoint_dir / f"checkpoint_epoch_{epoch}.pt"
        torch.save(checkpoint, checkpoint_path)

        # Check if best model
        metric_value = metrics.get(self.config.metric_for_best)
        if metric_value is not None and self._is_better(metric_value):
            self.best_metric_value = metric_value
            best_path = self.config.checkpoint_dir / "best_model.pt"
            shutil.copy(checkpoint_path, best_path)

        # Cleanup old checkpoints
        self._cleanup_old_checkpoints()

        return checkpoint_path

    def load(self, checkpoint_path: Path) -> Dict[str, Any]:
        """Load checkpoint from path."""
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        return checkpoint

    def load_best(self) -> Dict[str, Any]:
        """Load best model checkpoint."""
        best_path = self.config.checkpoint_dir / "best_model.pt"
        return self.load(best_path)

    def _is_better(self, metric_value: float) -> bool:
        """Check if metric value is better than current best."""
        if self.config.metric_mode == 'min':
            return metric_value < self.best_metric_value
        else:
            return metric_value > self.best_metric_value

    def _cleanup_old_checkpoints(self):
        """Keep only last N checkpoints."""
        checkpoints = sorted(
            self.config.checkpoint_dir.glob("checkpoint_epoch_*.pt"),
            key=lambda p: p.stat().st_mtime
        )

        # Remove old checkpoints (keep last N)
        if len(checkpoints) > self.config.keep_last_n:
            for checkpoint in checkpoints[:-self.config.keep_last_n]:
                checkpoint.unlink()
```

**Test Plan for checkpoint_manager.py** (12 tests):
```python
# tests/test_checkpoint_manager.py
# 1. test_checkpoint_manager_save_creates_file
# 2. test_checkpoint_manager_load_restores_state
# 3. test_checkpoint_manager_saves_best_model_min_mode
# 4. test_checkpoint_manager_saves_best_model_max_mode
# 5. test_checkpoint_manager_cleanup_old_checkpoints
# 6. test_checkpoint_manager_keeps_last_n_checkpoints
# 7. test_checkpoint_manager_handles_missing_checkpoint
# 8. test_checkpoint_manager_scheduler_state_optional
# 9. test_checkpoint_manager_extra_state_preservation
# 10. test_checkpoint_manager_best_metric_tracking
# 11. test_checkpoint_manager_creates_directory_if_missing
# 12. test_checkpoint_manager_concurrent_saves
```

**Benefits**:
- ✅ 181 lines extracted from transformer_trainer.py
- ✅ Independently testable (12 tests, ~90% coverage)
- ✅ Reusable across different trainers
- ✅ Single Responsibility Principle

**Day 17-18: Extract DeviceManager**

```python
# src/training/device_manager.py (NEW FILE)
import torch
from typing import Union, List, Optional
from enum import Enum

class DeviceType(Enum):
    """Supported device types"""
    CPU = "cpu"
    CUDA = "cuda"
    MPS = "mps"
    AUTO = "auto"

class DeviceManager:
    """
    Manages device placement and multi-GPU support.

    Single Responsibility: Device detection and tensor movement.
    """

    def __init__(self, device: Union[str, DeviceType] = DeviceType.AUTO):
        if isinstance(device, str):
            device = DeviceType(device)

        self.device = self._resolve_device(device)
        self.is_distributed = torch.cuda.device_count() > 1

    def _resolve_device(self, device_type: DeviceType) -> torch.device:
        """Auto-detect best available device."""
        if device_type == DeviceType.AUTO:
            if torch.cuda.is_available():
                return torch.device("cuda")
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                return torch.device("mps")
            else:
                return torch.device("cpu")
        else:
            return torch.device(device_type.value)

    def move_to_device(
        self,
        obj: Union[torch.nn.Module, torch.Tensor, List, tuple]
    ) -> Union[torch.nn.Module, torch.Tensor, List, tuple]:
        """Move model, tensor, or collection to device."""
        if isinstance(obj, (list, tuple)):
            return type(obj)(self.move_to_device(item) for item in obj)
        elif hasattr(obj, 'to'):
            return obj.to(self.device)
        else:
            return obj

    def wrap_for_distributed(self, model: torch.nn.Module) -> torch.nn.Module:
        """Wrap model for distributed training if applicable."""
        if self.is_distributed and self.device.type == "cuda":
            return torch.nn.DataParallel(model)
        return model

    def synchronize(self):
        """Synchronize device operations."""
        if self.device.type == "cuda":
            torch.cuda.synchronize()
```

**Deliverables**:
- ✅ CheckpointManager extracted and tested (+2%, Days 14-16)
- ✅ DeviceManager extracted and tested (+1%, Days 17-18)
- **Week 4 Total**: +3% coverage (59.5% → 62.5%)

---

### Summary of 4-Week Plan

| Week | Focus | Tests Added | Coverage Gain | New Coverage |
|------|-------|-------------|---------------|--------------|
| 1 | Quick wins (pure functions, easy mocks) | 55 | +10% | 45% |
| 2 | Factories + utilities | 30 | +10.5% | 55.5% |
| 3 | Model factory with mocking | 30 | +4% | 59.5% |
| 4 | Begin refactoring (extract classes) | 25 | +3% | 62.5% |
| **Total** | **4 weeks** | **140** | **+27.5%** | **62.5%** |

**Post Week 4**:
- Week 5-6: Extract TrainerVisualizer, complete transformer_trainer.py refactoring
- Week 7-8: Integration tests for HuggingFace dataset loaders
- Month 3: Reach 70%+ coverage

---

## Test Templates

### Template 1: Pure Function Testing (learningrate_scheduler.py)

```python
# tests/test_learningrate_scheduler.py
"""
Tests for learning rate schedulers.

Target: 90%+ coverage (pure math, no I/O)
Estimated tests: 10
"""

import pytest
import torch
import torch.optim as optim
from src.utils.learningrate_scheduler import (
    WarmupCosineScheduler,
    WarmupLinearScheduler,
    ConstantScheduler,
    InverseSqrtScheduler
)


class TestWarmupCosineScheduler:
    """Test warmup + cosine annealing scheduler."""

    @pytest.fixture
    def optimizer(self):
        """Create dummy optimizer."""
        model = torch.nn.Linear(10, 10)
        return optim.Adam(model.parameters(), lr=1e-3)

    def test_warmup_phase_increases_lr(self, optimizer):
        """Test that LR increases linearly during warmup."""
        scheduler = WarmupCosineScheduler(
            optimizer,
            warmup_steps=100,
            total_steps=1000,
            min_lr=0.0
        )

        # Collect LRs during warmup
        lrs = []
        for _ in range(100):
            lrs.append(scheduler.get_last_lr()[0])
            scheduler.step()

        # Should increase monotonically
        assert all(lrs[i] < lrs[i+1] for i in range(len(lrs)-1))

    def test_cosine_phase_decreases_lr(self, optimizer):
        """Test that LR decreases with cosine after warmup."""
        scheduler = WarmupCosineScheduler(
            optimizer,
            warmup_steps=10,
            total_steps=100,
            min_lr=0.0
        )

        # Skip warmup
        for _ in range(10):
            scheduler.step()

        # Collect LRs during cosine phase
        lrs = []
        for _ in range(50):
            lrs.append(scheduler.get_last_lr()[0])
            scheduler.step()

        # Should decrease (mostly) - cosine can have small bumps
        decreasing_count = sum(1 for i in range(len(lrs)-1) if lrs[i] >= lrs[i+1])
        assert decreasing_count / len(lrs) > 0.9  # 90% of steps should decrease

    def test_min_lr_not_violated(self, optimizer):
        """Test that LR never goes below min_lr."""
        min_lr = 1e-6
        scheduler = WarmupCosineScheduler(
            optimizer,
            warmup_steps=10,
            total_steps=100,
            min_lr=min_lr
        )

        # Run entire schedule
        for _ in range(100):
            scheduler.step()
            current_lr = scheduler.get_last_lr()[0]
            assert current_lr >= min_lr

    def test_state_dict_save_and_load(self, optimizer):
        """Test scheduler state can be saved and restored."""
        scheduler = WarmupCosineScheduler(
            optimizer,
            warmup_steps=10,
            total_steps=100,
            min_lr=0.0
        )

        # Run 50 steps
        for _ in range(50):
            scheduler.step()

        # Save state
        state = scheduler.state_dict()
        lr_at_50 = scheduler.get_last_lr()[0]

        # Create new scheduler and load state
        scheduler2 = WarmupCosineScheduler(
            optimizer,
            warmup_steps=10,
            total_steps=100,
            min_lr=0.0
        )
        scheduler2.load_state_dict(state)

        # LR should match
        assert scheduler2.get_last_lr()[0] == pytest.approx(lr_at_50)


# Similar test classes for:
# - TestWarmupLinearScheduler
# - TestConstantScheduler
# - TestInverseSqrtScheduler
```

---

### Template 2: Mocking External Dependencies (model_factory.py)

```python
# tests/test_model_factory.py
"""
Tests for model factory with mocked external dependencies.

Target: 70%+ coverage
Estimated tests: 15
"""

import pytest
import torch
from unittest.mock import Mock, patch, MagicMock
from src.models.model_factory import (
    create_multimodal_model,
    ModelConfig
)


@pytest.fixture
def mock_timm():
    """Mock timm library to avoid downloading models."""
    with patch('src.models.model_factory.timm') as mock:
        # Mock vision model
        vision_model = Mock(spec=torch.nn.Module)
        vision_model.num_features = 768
        mock.create_model.return_value = vision_model
        yield mock


@pytest.fixture
def mock_transformers():
    """Mock HuggingFace transformers library."""
    with patch('src.models.model_factory.transformers') as mock:
        # Mock text model
        text_model = Mock(spec=torch.nn.Module)
        text_model.config.hidden_size = 768
        mock.AutoModel.from_pretrained.return_value = text_model
        yield mock


class TestModelFactory:
    """Test model factory functionality."""

    def test_create_vision_model_with_timm(self, mock_timm):
        """Test creating vision model using timm."""
        config = ModelConfig(
            vision_model_name="vit_base_patch16_224",
            text_model_name="albert-base-v2",
            fusion_dim=512,
            use_pretrained=True
        )

        model = create_multimodal_model(config)

        # Verify timm was called correctly
        mock_timm.create_model.assert_called_once_with(
            "vit_base_patch16_224",
            pretrained=True
        )

        assert model is not None

    def test_create_text_model_with_transformers(self, mock_transformers):
        """Test creating text model using transformers."""
        config = ModelConfig(
            vision_model_name="custom",
            text_model_name="albert-base-v2",
            fusion_dim=512,
            use_pretrained=True
        )

        model = create_multimodal_model(config)

        # Verify transformers was called
        mock_transformers.AutoModel.from_pretrained.assert_called_once_with(
            "albert-base-v2"
        )

        assert model is not None

    def test_model_handles_missing_timm(self, mock_transformers):
        """Test graceful fallback when timm is not available."""
        with patch('src.models.model_factory.timm', None):
            config = ModelConfig(
                vision_model_name="vit_base_patch16_224",
                text_model_name="albert-base-v2",
                fusion_dim=512
            )

            # Should either raise or fall back to custom model
            try:
                model = create_multimodal_model(config)
                # If it succeeds, verify it's using fallback
                assert model is not None
            except ImportError as e:
                # Or it should raise with helpful message
                assert "timm" in str(e).lower()

    def test_fusion_module_creation(self, mock_timm, mock_transformers):
        """Test that fusion module is created correctly."""
        config = ModelConfig(
            vision_model_name="vit_base_patch16_224",
            text_model_name="albert-base-v2",
            fusion_dim=512,
            fusion_type="concatenation"
        )

        model = create_multimodal_model(config)

        # Check that model has fusion module
        assert hasattr(model, 'fusion') or hasattr(model, 'fusion_module')

        # Verify output dimension
        dummy_vision = torch.randn(2, 768)
        dummy_text = torch.randn(2, 768)

        with torch.no_grad():
            fused = model.fusion(dummy_vision, dummy_text)

        assert fused.shape[-1] == config.fusion_dim
```

---

### Template 3: Integration Test with Fixtures (wmt_dataloader.py)

```python
# tests/test_wmt_dataloader.py
"""
Integration tests for WMT dataloader.

Note: This file exists with 170 lines but shows 0% coverage.
PRIORITY: Fix pytest discovery/import issues.
"""

import pytest
import torch
from pathlib import Path
from unittest.mock import Mock, patch
from src.data.wmt_dataloader import WMTDataLoader


@pytest.fixture
def sample_wmt_data(tmp_path):
    """Create mock WMT dataset files."""
    data_dir = tmp_path / "wmt_data"
    data_dir.mkdir()

    # Create mock train file
    train_file = data_dir / "train.txt"
    train_file.write_text(
        "Hello world.\tBonjour le monde.\n"
        "How are you?\tComment allez-vous?\n"
    )

    return data_dir


class TestWMTDataLoader:
    """Test WMT data loading functionality."""

    def test_dataloader_initialization(self, sample_wmt_data):
        """Test dataloader initializes correctly."""
        loader = WMTDataLoader(
            data_dir=sample_wmt_data,
            batch_size=2,
            language_pair=("en", "fr")
        )

        assert loader is not None
        assert loader.batch_size == 2

    def test_dataloader_iteration(self, sample_wmt_data):
        """Test iterating through batches."""
        loader = WMTDataLoader(
            data_dir=sample_wmt_data,
            batch_size=1,
            language_pair=("en", "fr")
        )

        batches = list(loader)

        assert len(batches) > 0
        # Verify batch structure
        batch = batches[0]
        assert "src" in batch
        assert "tgt" in batch

    def test_dataloader_batch_size(self, sample_wmt_data):
        """Test batch size is respected."""
        batch_size = 2
        loader = WMTDataLoader(
            data_dir=sample_wmt_data,
            batch_size=batch_size,
            language_pair=("en", "fr")
        )

        for batch in loader:
            # Last batch might be smaller
            assert len(batch["src"]) <= batch_size
            assert len(batch["tgt"]) <= batch_size
```

---

## Refactoring Guides

### Guide 1: Extracting CheckpointManager from TransformerTrainer

**Before** (transformer_trainer.py):
```python
class TransformerTrainer:
    def __init__(self, model, optimizer, ...):
        self.model = model
        self.optimizer = optimizer
        self.checkpoint_dir = checkpoint_dir
        # ... 15 more parameters

    def save_checkpoint(self, epoch, is_best=False):
        # 99 lines of checkpoint logic
        pass

    def load_checkpoint(self, path):
        # 82 lines of restoration logic
        pass

    def train_epoch(self, dataloader):
        # 145 lines of training logic
        pass
```

**After**:
```python
# transformer_trainer.py (simplified)
from src.training.checkpoint_manager import CheckpointManager, CheckpointConfig

class TransformerTrainer:
    def __init__(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        checkpoint_config: CheckpointConfig,  # Single config object
        device_manager: DeviceManager,
        ...
    ):
        self.model = model
        self.optimizer = optimizer
        self.checkpoint_mgr = CheckpointManager(checkpoint_config)
        self.device_mgr = device_manager
        # ... fewer parameters

    def save_checkpoint(self, epoch: int, metrics: Dict[str, float]):
        """Simplified - delegates to CheckpointManager."""
        return self.checkpoint_mgr.save(
            epoch=epoch,
            model=self.model,
            optimizer=self.optimizer,
            metrics=metrics
        )

    def load_checkpoint(self, path: Path):
        """Simplified - delegates to CheckpointManager."""
        checkpoint = self.checkpoint_mgr.load(path)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        return checkpoint
```

**Migration Steps**:
1. Create `src/training/checkpoint_manager.py` with CheckpointManager class
2. Write tests for CheckpointManager (12 tests, target 90% coverage)
3. Update TransformerTrainer to use CheckpointManager
4. Update all training scripts to pass CheckpointConfig
5. Run full test suite to verify no regressions
6. Remove old checkpoint code from TransformerTrainer

---

### Guide 2: Refactoring loss_factory.py to Strategy Pattern

**Current Anti-Pattern**:
```python
# One giant function with 12+ branches
def create_loss_function(args, dataset_size, train_loader):
    # PROBLEM 1: Args mutation
    args.fusion_dim = fusion_dim

    # PROBLEM 2: 12+ loss type branches
    if args.loss_type == "barlow_twins":
        # 20 lines
        loss = BarlowTwinsLoss(...)
    elif args.loss_type == "vicreg":
        # 100+ lines with curriculum
        if args.vicreg_use_curriculum:
            # ...
        loss = VICRegLoss(...)
    # ... 10 more branches

    return loss
```

**Refactored with Strategy Pattern**:

**Step 1: Create immutable config**
```python
# src/training/losses/loss_config.py
from dataclasses import dataclass
from typing import Optional

@dataclass(frozen=True)  # Immutable!
class LossConfig:
    """Immutable loss configuration."""
    loss_type: str
    temperature: float = 0.07
    fusion_dim: int = 512

    # Barlow Twins specific
    barlow_lambda: float = 0.005

    # VICReg specific
    vicreg_sim_weight: float = 25.0
    vicreg_var_weight: float = 25.0
    vicreg_cov_weight: float = 1.0
    vicreg_use_curriculum: bool = False
    vicreg_curriculum_start: int = 0
    vicreg_curriculum_end: int = 10

    # Contrastive specific
    use_hard_negatives: bool = False
    memory_bank_size: Optional[int] = None

    # ... other loss-specific configs

    @classmethod
    def from_args(cls, args) -> 'LossConfig':
        """Create config from argparse Namespace (no mutation!)."""
        return cls(
            loss_type=args.loss_type,
            temperature=getattr(args, 'temperature', 0.07),
            fusion_dim=getattr(args, 'fusion_dim', 512),
            # ... extract all fields
        )
```

**Step 2: Create strategy interface**
```python
# src/training/losses/loss_strategy.py
from abc import ABC, abstractmethod
import torch.nn as nn
from .loss_config import LossConfig

class LossStrategy(ABC):
    """Abstract base class for loss creation strategies."""

    @abstractmethod
    def create_loss(self, config: LossConfig) -> nn.Module:
        """Create loss function from config."""
        pass

    @abstractmethod
    def get_supported_type(self) -> str:
        """Return the loss type this strategy handles."""
        pass
```

**Step 3: Implement concrete strategies**
```python
# src/training/losses/strategies/barlow_twins_strategy.py
from ..loss_strategy import LossStrategy
from ..loss_config import LossConfig
from ..self_supervised import BarlowTwinsLoss

class BarlowTwinsStrategy(LossStrategy):
    """Strategy for creating Barlow Twins loss."""

    def get_supported_type(self) -> str:
        return "barlow_twins"

    def create_loss(self, config: LossConfig) -> BarlowTwinsLoss:
        return BarlowTwinsLoss(
            dim=config.fusion_dim,
            lambd=config.barlow_lambda
        )


# src/training/losses/strategies/vicreg_strategy.py
from ..loss_strategy import LossStrategy
from ..loss_config import LossConfig
from ..self_supervised import VICRegLoss
from ..wrappers import CurriculumLossWrapper

class VICRegStrategy(LossStrategy):
    """Strategy for creating VICReg loss with optional curriculum."""

    def get_supported_type(self) -> str:
        return "vicreg"

    def create_loss(self, config: LossConfig) -> nn.Module:
        base_loss = VICRegLoss(
            sim_weight=config.vicreg_sim_weight,
            var_weight=config.vicreg_var_weight,
            cov_weight=config.vicreg_cov_weight
        )

        if config.vicreg_use_curriculum:
            return CurriculumLossWrapper(
                loss=base_loss,
                start_epoch=config.vicreg_curriculum_start,
                end_epoch=config.vicreg_curriculum_end
            )

        return base_loss

# Similar strategies for:
# - ContrastiveStrategy (SimCLR, NT-Xent)
# - CLIPStrategy
# - MoCoStrategy (memory queue)
# - etc. (12 total)
```

**Step 4: Create factory with registry**
```python
# src/training/losses/loss_factory.py (refactored)
from typing import Dict
from .loss_strategy import LossStrategy
from .loss_config import LossConfig
from .strategies import (
    BarlowTwinsStrategy,
    VICRegStrategy,
    ContrastiveStrategy,
    CLIPStrategy,
    MoCoStrategy,
    # ... import all 12 strategies
)

class LossFactory:
    """Factory for creating loss functions using strategy pattern."""

    def __init__(self):
        """Initialize with all available strategies."""
        self._strategies: Dict[str, LossStrategy] = {}
        self._register_default_strategies()

    def _register_default_strategies(self):
        """Register all built-in strategies."""
        strategies = [
            BarlowTwinsStrategy(),
            VICRegStrategy(),
            ContrastiveStrategy(),
            CLIPStrategy(),
            MoCoStrategy(),
            # ... all 12 strategies
        ]

        for strategy in strategies:
            self.register_strategy(strategy)

    def register_strategy(self, strategy: LossStrategy):
        """Register a new loss strategy (Open/Closed Principle)."""
        loss_type = strategy.get_supported_type()
        self._strategies[loss_type] = strategy

    def create(self, config: LossConfig) -> nn.Module:
        """Create loss function from config."""
        strategy = self._strategies.get(config.loss_type)

        if strategy is None:
            available = ', '.join(self._strategies.keys())
            raise ValueError(
                f"Unknown loss type: {config.loss_type}. "
                f"Available types: {available}"
            )

        return strategy.create_loss(config)


# Convenience function (backward compatibility)
def create_loss_function(args, dataset_size=None, train_loader=None) -> nn.Module:
    """
    Create loss function from args (legacy interface).

    NOTE: No args mutation! Creates immutable config instead.
    """
    config = LossConfig.from_args(args)
    factory = LossFactory()
    return factory.create(config)
```

**Benefits**:
- ✅ No args mutation (immutable config)
- ✅ Each strategy is independently testable
- ✅ Open/Closed: Can add new loss types without modifying factory
- ✅ Single Responsibility: Each strategy creates one loss type
- ✅ Easy to test: 12 strategies × 2 tests each = 24 simple unit tests

**Testing Strategy**:
```python
# tests/test_loss_factory.py (24 tests)
class TestLossFactory:
    def test_barlow_twins_creation(self):
        config = LossConfig(loss_type="barlow_twins", fusion_dim=512)
        factory = LossFactory()
        loss = factory.create(config)
        assert isinstance(loss, BarlowTwinsLoss)

    def test_vicreg_without_curriculum(self):
        config = LossConfig(loss_type="vicreg", vicreg_use_curriculum=False)
        factory = LossFactory()
        loss = factory.create(config)
        assert isinstance(loss, VICRegLoss)

    def test_vicreg_with_curriculum(self):
        config = LossConfig(loss_type="vicreg", vicreg_use_curriculum=True)
        factory = LossFactory()
        loss = factory.create(config)
        assert isinstance(loss, CurriculumLossWrapper)

    # ... 21 more tests (one per loss type + variations)

    def test_unknown_loss_type_raises(self):
        config = LossConfig(loss_type="unknown_loss")
        factory = LossFactory()
        with pytest.raises(ValueError, match="Unknown loss type"):
            factory.create(config)

    def test_config_immutability(self):
        config = LossConfig(loss_type="barlow_twins")
        with pytest.raises(AttributeError):
            config.fusion_dim = 1024  # Should raise - frozen dataclass
```

---

## CI/CD Integration

### GitHub Actions Workflow

Create `.github/workflows/coverage.yml`:

```yaml
name: Test Coverage CI

on:
  push:
    branches: [ main, develop ]
  pull_request:
    branches: [ main ]

jobs:
  test:
    runs-on: ubuntu-latest

    steps:
    - uses: actions/checkout@v3

    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: '3.10'

    - name: Cache dependencies
      uses: actions/cache@v3
      with:
        path: ~/.cache/pip
        key: ${{ runner.os }}-pip-${{ hashFiles('**/requirements.txt') }}

    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install -r requirements.txt
        pip install pytest pytest-cov pytest-xdist

    - name: Run tests with coverage
      run: |
        pytest --cov=src --cov-report=xml --cov-report=term -n auto

    - name: Upload coverage to Codecov
      uses: codecov/codecov-action@v3
      with:
        file: ./coverage.xml
        flags: unittests
        name: codecov-umbrella

    - name: Coverage comment
      uses: py-cov-action/python-coverage-comment-action@v3
      with:
        GITHUB_TOKEN: ${{ github.token }}
        MINIMUM_GREEN: 70
        MINIMUM_ORANGE: 50

    - name: Check coverage threshold
      run: |
        coverage report --fail-under=50
```

### Pre-commit Hook

Create `.pre-commit-config.yaml`:

```yaml
repos:
  - repo: local
    hooks:
      - id: pytest-coverage
        name: Run pytest with coverage check
        entry: bash -c 'pytest --cov=src --cov-fail-under=50 -q'
        language: system
        pass_filenames: false
        always_run: true
```

### Coverage Badges

Add to `README.md`:

```markdown
# Multimodal Insight Engine

[![Coverage](https://codecov.io/gh/your-org/multimodal_insight_engine/branch/main/graph/badge.svg)](https://codecov.io/gh/your-org/multimodal_insight_engine)
[![Tests](https://github.com/your-org/multimodal_insight_engine/workflows/Test%20Coverage%20CI/badge.svg)](https://github.com/your-org/multimodal_insight_engine/actions)

## Coverage Progress

**Current Coverage**: 35% → **Target**: 70%

| Week | Coverage | Tests | Status |
|------|----------|-------|--------|
| Baseline | 35% | 421 | ⚪️ Starting |
| Week 1 | 45% | 476 | 🟢 Quick Wins |
| Week 2 | 55% | 506 | 🟢 Core Infrastructure |
| Week 3 | 59% | 536 | 🟡 Factories |
| Week 4 | 62% | 561 | 🟡 Refactoring |
| Month 2 | 70% | 650+ | 🎯 Target |
```

---

## Success Metrics

### Weekly Targets

**Week 1 Success Criteria**:
- ✅ Coverage: 35% → 45% (+10%)
- ✅ Tests added: 55
- ✅ wmt_dataloader.py test fixed (0% → 80%)
- ✅ learningrate_scheduler.py tested (0% → 90%)
- ✅ gradient_handler.py tested (0% → 80%)
- ✅ All tests passing (no new failures)

**Week 2 Success Criteria**:
- ✅ Coverage: 45% → 55% (+10%)
- ✅ Tests added: 30
- ✅ loss_factory.py refactored and tested (27% → 75%)
- ✅ metrics_tracker.py tested (0% → 75%)
- ✅ All Tier 1 complete

**Week 3 Success Criteria**:
- ✅ Coverage: 55% → 59% (+4%)
- ✅ Tests added: 30
- ✅ model_factory.py tested (7% → 70%)
- ✅ All Tier 2 complete

**Week 4 Success Criteria**:
- ✅ Coverage: 59% → 62% (+3%)
- ✅ CheckpointManager extracted and tested (90% coverage)
- ✅ DeviceManager extracted and tested (85% coverage)
- ✅ transformer_trainer.py reduced from 367 → ~200 lines
- ✅ Architecture document updated

### Risk Reduction Metrics

**Critical Risk Reduction** (Week 1-2):
- Training loop bugs: 🔴 → 🟡 (refactoring in progress)
- Loss function bugs: 🔴 → 🟢 (tested after refactoring)
- LR scheduling bugs: 🔴 → 🟢 (fully tested)
- Gradient handling bugs: 🔴 → 🟢 (fully tested)

**Code Quality Metrics**:
- Cyclomatic complexity: Reduced by extracting God objects
- Class responsibilities: 20+ → 5-7 per class
- Test maintainability: Isolated tests, no shared state
- Code duplication: Strategies replace duplicated factory logic

---

## Appendix: File Coverage Reference

### Tier 1 Files (Quick Wins)
1. ✅ `src/utils/learningrate_scheduler.py` (356 lines) - 0% → 90%
2. ✅ `src/data/wmt_dataloader.py` (114 lines) - 0% → 80%
3. ✅ `src/utils/gradient_handler.py` (469 lines) - 0% → 80%
4. ✅ `src/data/dataset_wrapper.py` (165 lines) - 0% → 85%
5. ✅ `src/training/losses/contrastive_learning.py` (229 lines) - 7% → 75%
6. ✅ `src/data/fixed_semantic_sampler.py` (310 lines) - 0% → 70%

### Tier 2 Files (High Value)
7. ✅ `src/models/model_factory.py` (141 lines) - 7% → 70%
8. ✅ `src/training/losses/loss_factory.py` (207 lines) - 27% → 75%
9. ✅ `src/utils/metrics_tracker.py` (663 lines) - 0% → 75%
10. ✅ `src/training/joint_bpe_training.py` (131 lines) - 0% → 70%
11. ✅ `src/utils/argument_configs.py` (70 lines) - 4% → 80%

### Tier 3 Files (Refactoring)
12. ⏳ `src/training/trainers/transformer_trainer.py` (367 lines) - 6% → 60%
13. ⏳ `src/training/flickr_multistage_training.py` (784 lines) - 0% → 40%

### Tier 4 Files (Archive/Defer)
14. 🗄️ `src/data/image_dataset.py` (177 lines) - ARCHIVE (no imports)
15. ⏸️ `src/utils/feature_attribution.py` (586 lines) - Defer (unused)
16. ⏸️ `src/utils/profiling.py` (1214 lines) - Defer (debug tool)
17. ⏸️ Integration test datasets (combined_wmt, iwslt, wmt) - Month 2

---

## Next Actions

1. **Immediate** (Today):
   - Investigate `tests/test_wmt_dataloader.py` coverage issue
   - Run: `python -m pytest tests/test_wmt_dataloader.py -v --tb=short`
   - Expected: Either fix imports or remove skip conditions

2. **Week 1 Start** (Day 1):
   - Create `tests/test_learningrate_scheduler.py`
   - Implement 10 tests for scheduler functions
   - Target: 90% coverage of learningrate_scheduler.py

3. **Continuous**:
   - Update TODO list daily with completed items
   - Run coverage after each test file: `pytest --cov=src --cov-report=term`
   - Commit when each file reaches target coverage

---

**End of Coverage Improvement Roadmap**
**Generated**: 2025-12-18
**For questions or updates, see**: `COVERAGE_PRIORITY_MATRIX.md`
