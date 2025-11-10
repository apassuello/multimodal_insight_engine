# Architecture Quick Fixes - Action Plan

**For**: Development Team
**Purpose**: Immediate actionable fixes based on architecture review
**Timeline**: Next 2 weeks

---

## 🔴 CRITICAL: Fix Today (< 2 hours)

### 1. Remove Duplicate DecoupledContrastiveLoss

**Problem**: Same class exists in TWO files
- `/home/user/multimodal_insight_engine/src/training/losses/contrastive_learning.py`
- `/home/user/multimodal_insight_engine/src/training/losses/decoupled_contrastive_loss.py`

**Solution**:
```bash
# Keep the standalone file, remove from contrastive_learning.py
cd /home/user/multimodal_insight_engine

# 1. Check which version is imported
grep -r "from.*decoupled_contrastive_loss import" src/

# 2. Remove duplicate from contrastive_learning.py
# Edit src/training/losses/contrastive_learning.py
# Delete the DecoupledContrastiveLoss class (lines ~200-300)

# 3. Verify all imports point to standalone file
# Update any imports to:
from src.training.losses.decoupled_contrastive_loss import DecoupledContrastiveLoss
```

**Time**: 30 minutes
**Risk**: Low
**Impact**: Prevents import bugs

---

### 2. Extract SimpleContrastiveLoss from Factory

**Problem**: 187 lines of loss implementation inside `loss_factory.py`

**Solution**:
```bash
# Create new file
touch src/training/losses/simple_contrastive_loss.py
```

**Move this code**:
```python
# FROM: src/training/losses/loss_factory.py (lines 26-213)
# TO: src/training/losses/simple_contrastive_loss.py

"""MODULE: simple_contrastive_loss.py
PURPOSE: Simple InfoNCE-style contrastive loss for basic multimodal models
"""

import torch
import torch.nn as nn
from typing import Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)

class SimpleContrastiveLoss(nn.Module):
    # ... move 187 lines here ...
```

**Update factory**:
```python
# src/training/losses/loss_factory.py
from .simple_contrastive_loss import SimpleContrastiveLoss

# Remove class definition, keep factory function
```

**Time**: 1 hour
**Risk**: Low
**Impact**: Proper separation of concerns

---

## 🟡 HIGH PRIORITY: Fix This Week

### 3. Create BaseTrainer Class

**File**: `/home/user/multimodal_insight_engine/src/training/trainers/base_trainer.py`

**Create**:
```python
"""MODULE: base_trainer.py
PURPOSE: Base class for all trainer implementations with shared functionality
"""

import torch
import torch.nn as nn
from abc import ABC, abstractmethod
from typing import Dict, Optional, Any, List
import logging
import os

logger = logging.getLogger(__name__)


class BaseTrainer(ABC):
    """
    Abstract base class for all trainers.

    Implements template method pattern for training loop with
    extension points for subclass-specific behavior.
    """

    def __init__(
        self,
        model: nn.Module,
        train_dataloader: torch.utils.data.DataLoader,
        val_dataloader: Optional[torch.utils.data.DataLoader] = None,
        test_dataloader: Optional[torch.utils.data.DataLoader] = None,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[Any] = None,
        device: Optional[torch.device] = None,
        checkpoint_dir: str = "checkpoints",
        **kwargs
    ):
        """Initialize base trainer."""
        self.model = model
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.test_dataloader = test_dataloader
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.checkpoint_dir = checkpoint_dir

        # Move model to device
        self.model.to(self.device)

        # Create checkpoint directory
        os.makedirs(checkpoint_dir, exist_ok=True)

        # Training state
        self.current_epoch = 0
        self.global_step = 0
        self.best_val_loss = float('inf')

    def train(self, num_epochs: int) -> Dict[str, List[float]]:
        """
        Template method for training loop.

        Defines the overall training flow while allowing subclasses
        to customize specific steps.
        """
        self.on_train_begin()

        history = {
            'train_loss': [],
            'val_loss': [],
        }

        for epoch in range(num_epochs):
            self.current_epoch = epoch
            self.on_epoch_begin(epoch)

            # Training phase (implemented by subclass)
            train_metrics = self.train_epoch()
            history['train_loss'].append(train_metrics['loss'])

            # Validation phase (implemented by subclass)
            if self.val_dataloader is not None:
                val_metrics = self.validate_epoch()
                history['val_loss'].append(val_metrics['loss'])
            else:
                val_metrics = None

            # Epoch end hook
            self.on_epoch_end(epoch, train_metrics, val_metrics)

            # Check early stopping
            if self.should_stop_early(val_metrics):
                logger.info(f"Early stopping triggered at epoch {epoch}")
                break

        self.on_train_end()
        return history

    @abstractmethod
    def train_epoch(self) -> Dict[str, float]:
        """
        Train for one epoch.

        Returns:
            Dictionary of training metrics
        """
        pass

    @abstractmethod
    def validate_epoch(self) -> Dict[str, float]:
        """
        Validate for one epoch.

        Returns:
            Dictionary of validation metrics
        """
        pass

    def on_train_begin(self):
        """Hook called at the start of training."""
        logger.info("Starting training...")
        self.model.train()

    def on_epoch_begin(self, epoch: int):
        """Hook called at the start of each epoch."""
        logger.info(f"Epoch {epoch + 1}/{self.current_epoch} starting...")

    def on_epoch_end(self, epoch: int, train_metrics: Dict, val_metrics: Optional[Dict]):
        """Hook called at the end of each epoch."""
        msg = f"Epoch {epoch + 1} - Train Loss: {train_metrics['loss']:.4f}"
        if val_metrics is not None:
            msg += f", Val Loss: {val_metrics['loss']:.4f}"
        logger.info(msg)

        # Update learning rate
        if self.scheduler is not None:
            self.scheduler.step()

        # Save checkpoint if best
        if val_metrics is not None and val_metrics['loss'] < self.best_val_loss:
            self.best_val_loss = val_metrics['loss']
            self.save_checkpoint(is_best=True)

    def on_train_end(self):
        """Hook called at the end of training."""
        logger.info("Training completed!")

    def should_stop_early(self, val_metrics: Optional[Dict]) -> bool:
        """
        Determine if training should stop early.

        Default implementation: no early stopping.
        Override in subclass for custom logic.
        """
        return False

    def save_checkpoint(self, is_best: bool = False):
        """Save model checkpoint."""
        checkpoint = {
            'epoch': self.current_epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict() if self.optimizer else None,
            'best_val_loss': self.best_val_loss,
        }

        path = os.path.join(self.checkpoint_dir, 'checkpoint.pt')
        torch.save(checkpoint, path)

        if is_best:
            best_path = os.path.join(self.checkpoint_dir, 'best_model.pt')
            torch.save(checkpoint, best_path)
            logger.info(f"Saved best model to {best_path}")

    def load_checkpoint(self, checkpoint_path: str):
        """Load model checkpoint."""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])

        if self.optimizer and checkpoint['optimizer_state_dict']:
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        self.current_epoch = checkpoint.get('epoch', 0)
        self.best_val_loss = checkpoint.get('best_val_loss', float('inf'))

        logger.info(f"Loaded checkpoint from {checkpoint_path}")
```

**Update existing trainers**:
```python
# Example: src/training/trainers/multimodal_trainer.py
from .base_trainer import BaseTrainer

class MultimodalTrainer(BaseTrainer):  # Now inherits!

    def __init__(self, ...):
        super().__init__(...)  # Call parent init
        # Multimodal-specific init

    def train_epoch(self) -> Dict[str, float]:
        """Implement multimodal training loop"""
        # ... existing logic ...

    def validate_epoch(self) -> Dict[str, float]:
        """Implement multimodal validation loop"""
        # ... existing logic ...
```

**Time**: 1 day (create base) + 2 days (update subclasses)
**Risk**: Medium (requires careful testing)
**Impact**: Massive reduction in duplication

---

### 4. Standardize Configuration Naming

**Problem**: Multiple names for same concept
- `fusion_dim` vs `projection_dim` vs `model_dim` vs `input_dim`

**Solution**: Create glossary and refactor

**Glossary** (`docs/ARCHITECTURE_GLOSSARY.md`):
```markdown
# Architecture Glossary

## Dimension Terminology

| Term | Usage | Example |
|------|-------|---------|
| `embedding_dim` | Size of feature embeddings | 768 for BERT-base |
| `hidden_dim` | Size of hidden layers | 512 for custom layers |
| `projection_dim` | Size after projection layer | 256 for contrastive |
| `vocab_size` | Tokenizer vocabulary size | 30522 for BERT |

## Deprecated Terms (Do Not Use)
- ❌ `fusion_dim` → Use `embedding_dim`
- ❌ `model_dim` → Use `embedding_dim`
- ❌ `input_dim` → Use `embedding_dim`
- ❌ `dim` → Too vague, use specific term
```

**Refactoring Script**:
```python
# scripts/refactor_dimension_names.py
import os
import re

def refactor_file(filepath):
    """Replace deprecated dimension names with standard terms."""
    with open(filepath, 'r') as f:
        content = f.read()

    # Replacements
    replacements = {
        r'\bfusion_dim\b': 'embedding_dim',
        r'\bmodel_dim\b': 'embedding_dim',
        r'\binput_dim\b': 'embedding_dim',
    }

    for pattern, replacement in replacements.items():
        content = re.sub(pattern, replacement, content)

    with open(filepath, 'w') as f:
        f.write(content)

# Run on all Python files
for root, dirs, files in os.walk('src'):
    for file in files:
        if file.endswith('.py'):
            filepath = os.path.join(root, file)
            refactor_file(filepath)
```

**Time**: 2 days (create glossary + careful refactoring)
**Risk**: Medium (requires thorough testing)
**Impact**: Reduced confusion, better maintainability

---

## 🟢 MEDIUM PRIORITY: Fix Next Week

### 5. Add Architecture Decision Records

**Setup**:
```bash
mkdir -p docs/adr
```

**Template** (`docs/adr/template.md`):
```markdown
# ADR-NNN: [Title]

**Date**: YYYY-MM-DD
**Status**: [Proposed | Accepted | Deprecated | Superseded]
**Deciders**: [List of people involved]

## Context
[What is the issue that we're seeing that is motivating this decision?]

## Decision
[What is the change that we're proposing and/or doing?]

## Consequences

### Positive
- [Positive consequence 1]
- [Positive consequence 2]

### Negative
- [Negative consequence 1]
- [Negative consequence 2]

### Risks
- [Risk 1]
- [Risk 2]

## Alternatives Considered
1. **[Alternative 1]**: [Why not chosen]
2. **[Alternative 2]**: [Why not chosen]

## Implementation Notes
[Any technical details about implementation]
```

**First ADRs to Create**:

1. `docs/adr/001-loss-function-architecture.md`
```markdown
# ADR-001: Loss Function Architecture Refactoring

**Date**: 2025-11-07
**Status**: Accepted

## Context
Currently have 21 loss classes with ~35% code duplication.

## Decision
Create inheritance hierarchy with:
- BaseLoss (abstract)
- ContrastiveLossBase (shared contrastive logic)
- Specialized variants (InfoNCE, VICReg, BarlowTwins)

## Consequences
### Positive
- 67% code reduction
- Consistent behavior
- Easier testing

### Negative
- 2-week refactoring effort
- Temporary instability

## Implementation Notes
See ARCHITECTURE_REVIEW.md Section 6.1
```

2. `docs/adr/002-trainer-base-class.md`
3. `docs/adr/003-configuration-management.md`

**Time**: 1 day
**Risk**: Low
**Impact**: Long-term knowledge preservation

---

## Testing Checklist for Refactoring

Before making ANY changes:

```bash
# 1. Run full test suite
./run_tests.sh

# 2. Record baseline metrics
python -m pytest tests/ --cov=src --cov-report=html
# Save coverage report

# 3. Create characterization tests
# tests/test_characterization.py
def test_multimodal_trainer_baseline():
    """Lock in current behavior"""
    trainer = create_test_trainer()
    result = trainer.train(epochs=1)
    # Save result to baseline.json
```

After changes:

```bash
# 1. Verify tests still pass
./run_tests.sh

# 2. Check coverage didn't decrease
python -m pytest tests/ --cov=src --cov-report=html

# 3. Run characterization tests
pytest tests/test_characterization.py

# 4. Manual smoke test
python demos/multimodal_training_demo.py
```

---

## Code Review Checklist

When reviewing refactored code:

- [ ] Does it follow SRP (Single Responsibility Principle)?
- [ ] Are classes < 500 lines?
- [ ] Are methods < 50 lines?
- [ ] Is there a base class for shared logic?
- [ ] Are configuration parameters documented?
- [ ] Are there unit tests for new classes?
- [ ] Is the change documented in an ADR?
- [ ] Does it reduce or eliminate duplication?
- [ ] Are deprecated patterns removed?
- [ ] Is naming consistent with glossary?

---

## Gradual Migration Strategy

**DON'T**: Rewrite everything at once
**DO**: Migrate incrementally

### Week 1: Foundation
```
Day 1-2: Remove duplicates (fixes 1-2)
Day 3-5: Create BaseTrainer (fix 3)
```

### Week 2: Standardization
```
Day 1-3: Standardize naming (fix 4)
Day 4-5: Add ADRs (fix 5)
```

### Week 3-4: Major Refactoring
```
Week 3: Loss function refactoring
Week 4: Trainer decomposition
```

### Week 5-8: Enhancement
```
Configuration unification
Observer pattern
Repository pattern
```

---

## Success Criteria

After fixes:

✅ **No duplicate classes**
✅ **No business logic in factories**
✅ **BaseTrainer exists and is used**
✅ **Consistent dimension naming**
✅ **All trainers < 1000 lines**
✅ **Test coverage > 75%**
✅ **ADRs document major decisions**

---

## Questions?

See:
- `ARCHITECTURE_REVIEW.md` - Full architectural analysis
- `CLAUDE.md` - Project guidelines
- `docs/adr/` - Architecture decisions

Contact: Architecture review available for consultation
