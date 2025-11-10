# Axis 1: Security & Stability

**Timeline**: Weeks 1-2
**Effort**: 44-54 hours
**Priority**: 🔴 CRITICAL

## Overview

This axis addresses **critical security vulnerabilities** and **immediate stability issues** that must be fixed before any refactoring work can begin safely.

## Current State

- **Security Score**: 5.5/10 (Moderate Risk)
- **Critical Vulnerabilities**: 4
- **Test Infrastructure**: Broken (pytest not in requirements)
- **Code Duplication**: Dangerous overlaps causing import bugs

## Target State (After Week 2)

- **Security Score**: 8.0/10 ✅
- **Critical Vulnerabilities**: 0 ✅
- **Test Infrastructure**: Working ✅
- **Dangerous Duplications**: Removed ✅

---

## Week 1: Critical Security Fixes (24-27 hours)

### 1. Fix Pickle Deserialization Vulnerability 🔴 CRITICAL
**Risk**: Remote Code Execution (RCE)
**Time**: 4-6 hours
**Priority**: P0

**Problem**: 8+ instances of `pickle.load()` in:
- `src/data/multimodal_dataset.py` (lines 1026, 1048, 1329)
- `src/data/tokenization/turbo_bpe_preprocessor.py` (lines 86, 110)

**Fix**: Replace pickle with safe serialization:
```python
# BEFORE (UNSAFE):
import pickle
with open('cache.pkl', 'rb') as f:
    data = pickle.load(f)  # Can execute arbitrary code!

# AFTER (SAFE):
import json
with open('cache.json', 'r') as f:
    data = json.load(f)  # Safe - only loads data
```

**Action Items**:
- [ ] Replace pickle in `multimodal_dataset.py` with JSON/HDF5
- [ ] Replace pickle in `turbo_bpe_preprocessor.py` with JSON
- [ ] Update cache loading in all data modules
- [ ] Test that caching still works correctly

**See**: `security-audit.md` for full details

---

### 2. Remove exec() Code Injection 🔴 CRITICAL
**Risk**: Arbitrary code execution
**Time**: 2-3 hours
**Priority**: P0

**Problem**: `compile_metadata.py` line 99 uses `exec()` to execute Python file contents:
```python
exec(file_content)  # Executes ANY code in the file!
```

**Fix**: Use AST parsing instead:
```python
import ast

# Parse Python file safely
tree = ast.parse(file_content)
# Extract metadata without executing code
for node in ast.walk(tree):
    if isinstance(node, ast.Assign):
        # Extract variable assignments safely
        pass
```

**Action Items**:
- [ ] Replace `exec()` with AST parsing in `compile_metadata.py`
- [ ] Test metadata compilation still works
- [ ] Add validation for parsed metadata

**See**: `security-audit.md` section 2

---

### 3. Fix Unsafe torch.load() Calls 🔴 CRITICAL
**Risk**: Malicious model files can execute code
**Time**: 2-3 hours
**Priority**: P0

**Problem**: 30+ instances of `torch.load()` without `weights_only=True`

**Fix**: Add safety parameter:
```python
# BEFORE (UNSAFE):
model = torch.load('model.pt')

# AFTER (SAFE):
model = torch.load('model.pt', weights_only=True)
```

**Action Items**:
- [ ] Search all `.py` files for `torch.load(`
- [ ] Add `weights_only=True` to all instances
- [ ] Test model loading still works
- [ ] Add to code review checklist

**See**: `security-audit.md` section 3

---

### 4. Fix Subprocess Command Injection 🟠 HIGH
**Risk**: OS command injection
**Time**: 30 minutes
**Priority**: P1

**Problem**: `setup_test/test_gpu.py` uses `shell=True`:
```python
subprocess.run(cmd, shell=True)  # Vulnerable to injection!
```

**Fix**: Remove shell=True:
```python
subprocess.run(['nvidia-smi', '--query'], shell=False)
```

**Action Items**:
- [ ] Fix `setup_test/test_gpu.py` lines 36, 45
- [ ] Test GPU detection still works

**See**: `security-audit.md` section 4

---

### 5. Fix Test Infrastructure ⚠️ CRITICAL
**Risk**: Cannot run tests, cannot validate changes
**Time**: 30 minutes
**Priority**: P0

**Problem**: `pytest` not in `requirements.txt`, tests don't run

**Fix**: Add test dependencies:
```bash
# Add to requirements.txt:
pytest>=8.3.5
pytest-cov>=4.0.0
pytest-xdist>=3.3.0  # parallel test execution
```

**Action Items**:
- [ ] Add pytest to `requirements.txt`
- [ ] Run `pip install pytest pytest-cov`
- [ ] Verify tests run: `pytest tests/ -v`
- [ ] Document test commands in README

**See**: `immediate-actions.md` section 5

---

### 6. Add Merge Validation Tests ⚠️ HIGH
**Risk**: Future merge bugs (like commit ccd463c)
**Time**: 2-3 hours
**Priority**: P1

**Problem**: Recent merge caused bugs, no validation tests exist

**Fix**: Create merge validation test suite:
```python
# tests/test_merge_validation.py
def test_no_duplicate_loss_classes():
    """Ensure DecoupledContrastiveLoss exists in only one file."""
    # Implementation provided in immediate-actions.md

def test_loss_function_registry():
    """Ensure all loss functions are properly registered."""
    pass

def test_trainer_instantiation():
    """Ensure all trainers can be instantiated."""
    pass
```

**Action Items**:
- [ ] Create `tests/test_merge_validation.py`
- [ ] Add duplicate class detection test
- [ ] Add loss registry validation test
- [ ] Run tests and ensure they pass

**See**: `immediate-actions.md` section 6

---

### 7. Start Testing Loss Functions 🟡 MEDIUM
**Risk**: 9,000+ lines of untested code
**Time**: 4-6 hours
**Priority**: P2

**Problem**: 18 out of 20 loss functions have ZERO tests

**Fix**: Test top 5 critical loss functions:
1. `ContrastiveLoss` (1,098 lines)
2. `VICRegLoss` (273 lines)
3. `CLIPStyleLoss` (complex)
4. `MultiModalMixedContrastiveLoss` (complex)
5. `SupervisedContrastiveLoss` (complex)

**Action Items**:
- [ ] Create `tests/test_contrastive_loss.py`
- [ ] Test forward pass with sample inputs
- [ ] Test gradient flow
- [ ] Test edge cases (batch_size=1, empty tensors)
- [ ] Achieve >60% coverage for these 5 losses

**See**: `../3-testing-and-quality/testing-patterns.md` for templates

---

### 8. Remove DecoupledContrastiveLoss Duplication 🟡 MEDIUM
**Risk**: Import conflicts, maintenance nightmare
**Time**: 1 hour
**Priority**: P2

**Problem**: `DecoupledContrastiveLoss` exists in TWO files:
- `src/training/losses/decoupled_contrastive_loss.py`
- `src/training/losses/contrastive_learning.py`

**Fix**: Keep standalone file, remove from `contrastive_learning.py`

**Action Items**:
- [ ] Remove class from `contrastive_learning.py`
- [ ] Update imports to use standalone file
- [ ] Run all tests to ensure nothing breaks
- [ ] Search codebase for any direct imports

**See**: `immediate-actions.md` section 8

---

### 9. Extract SimpleContrastiveLoss from Factory 🟡 MEDIUM
**Risk**: 187 lines of loss code inside factory file (anti-pattern)
**Time**: 2 hours
**Priority**: P2

**Problem**: `SimpleContrastiveLoss` defined inside `loss_factory.py` (lines 26-213)

**Fix**: Move to new file:
```bash
# Create new file
touch src/training/losses/simple_contrastive_loss.py

# Move class (187 lines)
# Update factory to import it
```

**Action Items**:
- [ ] Create `src/training/losses/simple_contrastive_loss.py`
- [ ] Move class definition (lines 26-213)
- [ ] Update `loss_factory.py` to import it
- [ ] Test loss factory still works

**See**: `immediate-actions.md` section 9

---

### 10. Create BaseTrainer Skeleton 🟡 MEDIUM
**Risk**: 60% code duplication across 8 trainers
**Time**: 3-4 hours
**Priority**: P2

**Problem**: 8 trainer classes with massive duplication, no base class

**Fix**: Create minimal base trainer with common functionality:
```python
# src/training/trainers/base_trainer.py
class BaseTrainer:
    """Base class for all trainers."""

    def __init__(self, model, optimizer, device):
        self.model = model
        self.optimizer = optimizer
        self.device = device

    def _move_to_device(self, batch):
        """Move batch to device."""
        pass

    def save_checkpoint(self, path):
        """Save model checkpoint."""
        pass

    def load_checkpoint(self, path):
        """Load model checkpoint."""
        pass
```

**Action Items**:
- [ ] Create `src/training/trainers/base_trainer.py`
- [ ] Implement checkpoint save/load
- [ ] Implement device management
- [ ] Add tests for base trainer
- [ ] Don't migrate existing trainers yet (do in Axis 2)

**See**: `../2-architecture-refactoring/refactoring-strategy.md`

---

## Week 2: Stabilization (20-27 hours)

### Verification & Testing
- [ ] All security fixes verified with tests
- [ ] No security vulnerabilities in automated scans
- [ ] Test suite runs cleanly
- [ ] All merge validation tests pass
- [ ] No dangerous code duplications remain

### Documentation
- [ ] Update CRITICAL_README.md with security notes
- [ ] Document new testing commands
- [ ] Add security section to CLAUDE.md

### Automation
- [ ] Add pre-commit hook to prevent `pickle.load()`
- [ ] Add pre-commit hook to prevent `torch.load()` without safety
- [ ] Add pre-commit hook to run tests

---

## Success Metrics

After completing Axis 1, you should have:

✅ **Security Score**: 5.5/10 → 8.0/10
✅ **Critical Vulnerabilities**: 4 → 0
✅ **Test Infrastructure**: Working
✅ **Top 5 Loss Functions**: Tested (>60% coverage)
✅ **Dangerous Duplications**: Removed
✅ **Merge Validation**: Automated
✅ **Development Velocity**: Can now safely refactor

---

## Next Steps

Once Axis 1 is complete, proceed to:
- **Axis 2**: Architecture Refactoring (Weeks 3-6)

---

## Documents in This Axis

- **README.md** (this file) - Overview and action items
- **security-audit.md** - Detailed security vulnerability analysis
- **immediate-actions.md** - Complete Week 1 implementation guide with code examples
- **quick-wins.md** - Additional fast improvements (logging, configuration)

## Related Documentation

- `../3-testing-and-quality/testing-patterns.md` - Test templates
- `../2-architecture-refactoring/code-patterns.md` - Refactoring patterns
- Root: `IMPROVEMENT_PLAN.md` - Overall plan

---

**Questions?** See `immediate-actions.md` for detailed implementation guidance with copy-paste code examples.
