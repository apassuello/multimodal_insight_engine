# Axis 3: Testing & Quality

**Timeline**: Weeks 7-10
**Effort**: 208-261 hours
**Priority**: 🟡 MEDIUM-HIGH

## Overview

Complete comprehensive test coverage, establish quality standards, and implement automated quality gates. This solidifies the improvements from Axes 1 & 2.

## Current State (After Axis 2)

- **Test Coverage**: 65% (improved from 45%)
- **Untested Components**: Optimization, safety modules, utilities
- **Test Quality**: Basic functionality tests only
- **CI/CD**: No automated testing pipeline

## Target State (After Week 10)

- **Test Coverage**: 75% ✅
- **Critical Components**: 100% coverage ✅
- **Test Quality**: Unit + integration + property-based ✅
- **CI/CD**: Fully automated with quality gates ✅

---

## Week 7-8: Complete Core Coverage (110-140 hours)

### 1. Test Remaining Loss Functions (25-30 hours)

**Status**: 5 tested, 15-16 remaining

**Priority Loss Functions**:
- [ ] `HybridPretrainVICRegLoss` (complex, 22K lines file)
- [ ] `MultiModalMixedContrastiveLoss` (22K lines file)
- [ ] `MemoryQueueContrastiveLoss` (16K lines)
- [ ] `EMAMoCoLoss` (16K lines)
- [ ] `FeatureConsistencyLoss` (18K lines)

**Test Template**:
```python
class TestLossFunction:
    def test_forward_pass(self):
        """Test basic forward pass."""

    def test_backward_pass(self):
        """Test gradient computation."""

    def test_batch_sizes(self):
        """Test with various batch sizes."""

    def test_edge_cases(self):
        """Test empty, single sample, etc."""

    def test_loss_properties(self):
        """Test that loss >= 0, symmetric, etc."""
```

**See**: `testing-patterns.md` for complete templates

---

### 2. Test Safety Module (35-45 hours)

**Current Coverage**: 0%
**Target Coverage**: 80%

**Components to Test**:

**Constitutional AI** (20 hours):
- [ ] `src/safety/constitutional/ppo_trainer.py` (820 lines)
- [ ] `src/safety/constitutional/reward_model.py` (23K lines)
- [ ] `src/safety/constitutional/evaluator.py` (13K lines)
- [ ] `src/safety/constitutional/filter.py` (13K lines)

**Red Teaming** (15 hours):
- [ ] `src/safety/red_teaming/evaluator.py` (13K lines)
- [ ] `src/safety/red_teaming/prompt_injection.py` (22K lines)
- [ ] `src/safety/red_teaming/generators.py` (14K lines)

**Test Focus**:
- Principle evaluation correctness
- Filter effectiveness (false positives/negatives)
- Reward model ranking consistency
- Prompt injection detection

---

### 3. Test Optimization Module (20-25 hours)

**Current Coverage**: 20%
**Target Coverage**: 70%

**Components**:
- [ ] `src/optimization/quantization.py` - INT8/INT4 quantization
- [ ] `src/optimization/pruning.py` - Magnitude & structured pruning
- [ ] `src/optimization/mixed_precision.py` - FP16 training
- [ ] `src/optimization/benchmarking.py` - Performance analysis

**Test Requirements**:
- Verify quantization doesn't break models
- Test pruning maintains accuracy
- Benchmark speed improvements
- Test memory reduction

---

### 4. Test Utils Module (15-20 hours)

**Current Coverage**: 0%
**Target Coverage**: 75%

**Components** (13 files, 5,900 LOC):
- [ ] `profiling.py` (1,181 lines)
- [ ] `metrics_tracker.py` (676 lines)
- [ ] `visualization.py` (608 lines)
- [ ] `gradient_handler.py` (476 lines)
- [ ] `feature_attribution.py` (588 lines)
- [ ] Configuration and logging utilities

**Test Focus**:
- Profiling accuracy
- Metrics calculation correctness
- Gradient handling edge cases
- Feature attribution validity

---

### 5. Integration Tests (15-20 hours)

**End-to-End Workflows**:
- [ ] Full training loop (text → tokens → batches → training → checkpoint)
- [ ] Multimodal pipeline (text + images → features → fusion → training)
- [ ] Constitutional AI training (base → critique → revision → reward → PPO)
- [ ] Model optimization pipeline (train → quantize → prune → benchmark)

**Integration Test Template**:
```python
def test_full_training_pipeline():
    """Test complete training workflow."""
    # 1. Load data
    dataset = create_toy_dataset()

    # 2. Create model
    model = create_toy_model()

    # 3. Train for 2 epochs
    trainer = Trainer(model, dataset)
    trainer.train(epochs=2)

    # 4. Verify
    assert trainer.metrics['loss'][-1] < trainer.metrics['loss'][0]
    assert os.path.exists('checkpoint.pt')
```

---

## Week 9: Quality Infrastructure (48-61 hours)

### 6. CI/CD Pipeline Setup (20-25 hours)

**GitHub Actions Workflow**:

```yaml
# .github/workflows/ci.yml
name: CI

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.9'
      - name: Install dependencies
        run: pip install -r requirements.txt
      - name: Run tests
        run: pytest --cov=src --cov-report=xml
      - name: Upload coverage
        uses: codecov/codecov-action@v3

  lint:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - name: Run flake8
        run: flake8 src/ tests/
      - name: Run mypy
        run: mypy src/

  security:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - name: Run bandit
        run: bandit -r src/
```

**Tasks**:
- [ ] Create `.github/workflows/ci.yml`
- [ ] Setup code coverage reporting (Codecov/Coveralls)
- [ ] Add status badges to README
- [ ] Configure branch protection rules

---

### 7. Pre-commit Hooks (8-10 hours)

**`.pre-commit-config.yaml`**:
```yaml
repos:
  - repo: https://github.com/psf/black
    rev: 23.3.0
    hooks:
      - id: black

  - repo: https://github.com/PyCQA/flake8
    rev: 6.0.0
    hooks:
      - id: flake8

  - repo: https://github.com/pre-commit/mirrors-mypy
    rev: v1.3.0
    hooks:
      - id: mypy

  - repo: local
    hooks:
      - id: pytest-fast
        name: Fast tests
        entry: pytest tests/ -k "not slow"
        language: system
        pass_filenames: false
```

**Tasks**:
- [ ] Create pre-commit config
- [ ] Install pre-commit: `pre-commit install`
- [ ] Test hooks work
- [ ] Document in CONTRIBUTING.md

---

### 8. Property-Based Testing (20-25 hours)

**Using Hypothesis for ML invariants**:

```python
from hypothesis import given, strategies as st
import hypothesis.extra.numpy as npst

class TestLossProperties:
    @given(
        features=npst.arrays(
            dtype=np.float32,
            shape=st.tuples(st.integers(1, 64), st.integers(128, 2048)),
        )
    )
    def test_loss_non_negative(self, features):
        """Loss should always be non-negative."""
        loss = compute_loss(features)
        assert loss >= 0

    @given(
        features=npst.arrays(dtype=np.float32, shape=(32, 768))
    )
    def test_loss_finite(self, features):
        """Loss should never be inf or nan."""
        loss = compute_loss(features)
        assert torch.isfinite(loss).all()
```

**Focus Areas**:
- [ ] Loss function invariants (non-negative, finite, etc.)
- [ ] Model output shapes
- [ ] Tokenization round-trip
- [ ] Data augmentation preservation

**See**: `testing-patterns.md` section on property-based testing

---

## Week 10: Polish & Documentation (50-60 hours)

### 9. Performance Tests (15-20 hours)

**Benchmark Suite**:
```python
import pytest

@pytest.mark.slow
def test_training_speed():
    """Ensure training speed meets baseline."""
    # Train for 100 steps
    elapsed = trainer.train(steps=100)
    # Should complete in <5 minutes
    assert elapsed < 300

@pytest.mark.slow
def test_inference_latency():
    """Ensure inference latency acceptable."""
    latency = measure_inference_latency(model, batch_size=32)
    # Should be <100ms per batch
    assert latency < 0.1
```

**Tasks**:
- [ ] Create performance benchmark suite
- [ ] Establish baseline metrics
- [ ] Add to CI (run nightly)
- [ ] Document expected performance

---

### 10. Test Documentation (10-15 hours)

**Create**:
- [ ] `docs/testing/TESTING_GUIDE.md` - How to write tests
- [ ] `docs/testing/COVERAGE_REPORT.md` - Current coverage status
- [ ] `docs/testing/CI_CD_GUIDE.md` - CI/CD pipeline documentation
- [ ] Update CLAUDE.md with testing standards

**Testing Standards**:
- All new features require tests
- Minimum 80% coverage for new code
- Integration tests for workflows
- Property-based tests for invariants

---

### 11. Mutation Testing (15-20 hours)

**Using mutmut to test test quality**:

```bash
# Install mutation testing
pip install mutmut

# Run mutation testing
mutmut run

# View results
mutmut results
mutmut show
```

**What it does**: Mutates your code (change `+` to `-`, etc.) and checks if tests catch it.

**Tasks**:
- [ ] Setup mutmut configuration
- [ ] Run mutation testing on core modules
- [ ] Improve tests that miss mutations
- [ ] Target: >70% mutation score

---

### 12. Final Verification (10 hours)

**Checklist**:
- [ ] All tests pass on CI
- [ ] Coverage ≥75% overall
- [ ] Coverage ≥90% on critical modules (losses, trainers)
- [ ] No flaky tests (run 10x to verify)
- [ ] Pre-commit hooks work
- [ ] Documentation complete
- [ ] Performance benchmarks pass
- [ ] Mutation score >70%

---

## Success Metrics

After completing Axis 3, you should have:

✅ **Test Coverage**: 65% → 75% overall
✅ **Critical Coverage**: 90%+ on losses, trainers, safety
✅ **Test Quality**: Unit + integration + property-based + mutation
✅ **CI/CD**: Fully automated with quality gates
✅ **Pre-commit Hooks**: Catch issues before commit
✅ **Performance Tests**: Ensure no regression
✅ **Documentation**: Complete testing guides
✅ **Mutation Score**: >70% (tests actually catch bugs)

---

## Coverage Breakdown Target

| Module | Current | Target | Status |
|--------|---------|--------|--------|
| **Data** | 95% | 95% | ✅ Maintain |
| **Models** | 67% | 85% | 📈 Improve |
| **Training** | 65% | 90% | 📈 Improve |
| **Losses** | 40% | 95% | 📈 Critical |
| **Safety** | 0% | 80% | 📈 Critical |
| **Optimization** | 20% | 70% | 📈 Improve |
| **Utils** | 0% | 75% | 📈 Improve |
| **OVERALL** | 65% | 75% | 📈 Target |

---

## Risk Mitigation

**Risk**: Tests too slow, developers skip them

**Mitigation**:
- Separate fast (<1s) and slow (>1s) tests
- Run fast tests in pre-commit
- Run slow tests in CI only
- Parallelize test execution (pytest-xdist)

**Risk**: Flaky tests

**Mitigation**:
- Use fixed random seeds
- Mock external dependencies
- Isolate test state
- Run tests multiple times in CI

---

## Next Steps

Once Axis 3 is complete, proceed to:
- **Axis 4**: Repository Structure (Ongoing)

---

## Documents in This Axis

- **README.md** (this file) - Overview and action items
- **testing-assessment.md** - Detailed analysis of current test state
- **coverage-roadmap.md** - Module-by-module testing plan
- **testing-patterns.md** - Reusable test templates and patterns

## Related Documentation

- `../1-security-and-stability/` - Security tests from Axis 1
- `../2-architecture-refactoring/` - Tests for refactored code
- `../diagrams/` - Testing coverage visualization

---

**Questions?** See `testing-patterns.md` for ready-to-use test templates and `coverage-roadmap.md` for detailed module-by-module plans.
