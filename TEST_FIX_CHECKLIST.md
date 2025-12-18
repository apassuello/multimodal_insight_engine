# Test Failure Fix Checklist

**Generated:** 2025-12-15
**Status:** 147 failures, 695 passing (77.6%)
**Target:** >90% passing, >45% coverage

---

## Critical Path (Fix First - Resolves 73.5% of Failures)

### Priority 1: Logger Initialization (91 tests)
- [ ] Investigate logger setup in `conftest.py`
- [ ] Check module `__init__.py` files for logger initialization
- [ ] Add logger fixture to test setup if missing
- [ ] Verify logger is not None before returning from fixtures
- [ ] Run affected test files to verify fix:
  - [ ] `pytest tests/test_cai_integration.py -v`
  - [ ] `pytest tests/test_principles.py -v`
  - [ ] `pytest tests/test_evaluator.py -v`
  - [ ] `pytest tests/test_filter.py -v`

**Files to check:**
- `tests/conftest.py`
- `src/cai/__init__.py`
- `src/evaluator.py`
- `src/filter.py`
- `src/principles.py`

### Priority 2: Training Data Format - 'chosen' Key (12 tests)
- [ ] Locate reward model test fixtures
- [ ] Update test data format to include 'chosen' and 'rejected' keys
- [ ] Verify format matches: `{'prompt': str, 'chosen': str, 'rejected': str}`
- [ ] Run reward model tests:
  - [ ] `pytest tests/test_reward_model.py -v`
  - [ ] `pytest tests/test_cai_training_integration.py::TestPhase2Training -v`

**Files to check:**
- `tests/fixtures/training_data.py` or similar
- `tests/conftest.py` (reward model fixtures)
- `src/reward_model.py` (data loading)

### Priority 3: Training Data Validation (5 tests)
- [ ] Check Phase 1 training data validation logic
- [ ] Review what makes training examples "invalid"
- [ ] Fix validation criteria or update test data
- [ ] Add verbose validation error messages
- [ ] Run Phase 1 tests:
  - [ ] `pytest tests/test_cai_training_integration.py::TestPhase1Training -v`
  - [ ] `pytest tests/test_cai_training_integration.py::TestEndToEndPipeline -v`

**Files to check:**
- `src/training_pipeline.py`
- `src/data_validation.py`
- Training data fixtures

**Checkpoint:** After fixing these 3 issues, run full test suite:
```bash
pytest tests/ -v --tb=short | tee test_results_phase1.log
```
**Expected:** ~803/895 passing (89.7%)

---

## High Priority (Fix Next - Gets to 91%)

### Priority 4: Logger API Misuse (10 tests)

#### Pattern A: Invalid 'end' parameter (6 tests)
- [ ] Search for: `grep -r "logger.*end=" src/`
- [ ] Replace `logger.info(..., end='')` with proper logger call
- [ ] Alternative: Use `print()` if progress indicator needed
- [ ] Run: `pytest tests/test_training.py -v`

**Files to fix:**
- `src/training.py` (likely culprit)

#### Pattern B: 'level' parameter conflict (4 tests)
- [ ] Search for: `grep -r "Logger._log" src/`
- [ ] Fix calls to use `logger.info()`, `logger.debug()`, etc.
- [ ] Don't pass 'level' explicitly to _log()
- [ ] Run: `pytest tests/test_principles.py::TestJSONParsing -v`

**Files to fix:**
- `src/principles.py` (JSON parsing functions)

### Priority 5: Matrix Dimension Mismatch (4 tests)
- [ ] Check VICReg loss input expectations (should be 768-dim)
- [ ] Update test to provide correct input dimensions
- [ ] Or make loss accept flexible dimensions
- [ ] Add dimension validation with clear error message
- [ ] Run: `pytest tests/test_selfsupervised_losses.py::TestHybridPretrainVICRegLoss -v`

**Files to fix:**
- `tests/test_selfsupervised_losses.py` (test setup)
- `src/selfsupervised_losses.py` (if architecture needs update)

**Checkpoint:** After fixing priorities 4-5:
```bash
pytest tests/ -v --tb=short | tee test_results_phase2.log
```
**Expected:** ~817/895 passing (91.3%)

---

## Medium Priority (Clean Up - Gets to 93%)

### Priority 6: Mock Configuration (3 tests)
- [ ] Update mock evaluator to include 'model' attribute
- [ ] Review what attributes CritiqueRevisionPipeline expects
- [ ] Use `spec=` when creating mocks to auto-detect missing attrs
- [ ] Run: `pytest tests/test_critique_revision.py::TestCritiqueRevisionPipeline -v`

**Files to fix:**
- `tests/test_critique_revision.py`

### Priority 7: Generation Config (2 tests)
- [ ] Change `config['max_length']` to `config.get('max_length', 100)`
- [ ] Define default config dictionary
- [ ] Run: `pytest tests/test_model_utils.py::TestGenerateText -v`

**Files to fix:**
- `src/model_utils.py`

### Priority 8: Assertion Failures (9 tests)

#### Consequence Analysis (3 tests)
- [ ] Check if returned strings are semantically correct but different wording
- [ ] Update test expectations to match actual output
- [ ] Or fix consequence analysis logic if output is wrong
- [ ] Run: `pytest tests/test_principles.py::TestAnalyzePotentialConsequences -v`

#### Metrics Logging (2 tests)
- [ ] Debug why metrics output is empty string
- [ ] Check if output capture is working correctly
- [ ] Run: `pytest tests/test_metrics_collector.py -v -s`

#### Config Values (2 tests)
- [ ] Fix default value initialization (None vs 100)
- [ ] Fix config override logic (not applying 0.9)
- [ ] Run: `pytest tests/test_model_utils.py::TestGenerationConfig -v`

#### Other (2 tests)
- [ ] Fix case-insensitive detection in evaluator
- [ ] Fix plot alignment metrics (check matplotlib/file creation)
- [ ] Run individually and debug

---

## Low Priority (Polish)

### Priority 9: Tokenizer Preprocessing (1 test)
- [ ] Decide if '_space_' replacement is intentional
- [ ] Update test or fix tokenizer accordingly
- [ ] Run: `pytest tests/test_optimized_bpe_tokenizer.py::test_preprocess -v`

### Priority 10: Miscellaneous (5 tests)
- [ ] Europarl data: Add fixture or skip test
- [ ] Pickle error: Add numpy to safe globals
- [ ] Gradient error: Enable requires_grad
- [ ] Type error: Add type validation
- [ ] BLEU score: Fix calculation logic

---

## Coverage Improvement Tasks

### After Test Fixes
- [ ] Generate coverage report by file:
  ```bash
  pytest --cov=src --cov-report=html --cov-report=term-missing
  ```
- [ ] Identify files with <30% coverage
- [ ] Prioritize core modules:
  - [ ] CAI principles
  - [ ] Evaluator
  - [ ] Filter
  - [ ] Training pipeline
  - [ ] Data loading

### Add Tests For
- [ ] Uncovered conditional branches
- [ ] Error handling paths
- [ ] Edge cases in core functions
- [ ] Integration tests for full pipelines

**Target:** 45% coverage (currently 36.38%)
**Gap:** Need ~120 additional test assertions in uncovered areas

---

## Verification Commands

### Quick Smoke Test (Fast)
```bash
pytest tests/test_data.py tests/test_comparison_engine.py -v
```
Should pass - these are working.

### Test Each Priority After Fix
```bash
# After P1: Logger
pytest tests/test_cai_integration.py tests/test_principles.py -v

# After P2: Training data
pytest tests/test_reward_model.py -v

# After P3: Validation
pytest tests/test_cai_training_integration.py -v

# After P4: Logger API
pytest tests/test_training.py tests/test_principles.py::TestJSONParsing -v

# After P5: Dimensions
pytest tests/test_selfsupervised_losses.py -v
```

### Full Test Run
```bash
pytest tests/ -v --tb=short --cov=src --cov-report=term-missing
```

### Parallel Test Run (Faster)
```bash
pytest tests/ -n 14 -v --tb=short
```

---

## Success Criteria

- [ ] **Phase 1 Complete:** >89% tests passing (803/895)
- [ ] **Phase 2 Complete:** >91% tests passing (817/895)
- [ ] **Phase 3 Complete:** >93% tests passing (839/895)
- [ ] **Coverage Target:** >45% code coverage
- [ ] **CI Green:** All critical tests pass in CI
- [ ] **No Regressions:** Previously passing tests still pass

---

## Notes

- Focus on **logger initialization first** - biggest impact
- **Don't skip ahead** - earlier fixes may resolve later issues
- **Run tests incrementally** - verify each fix before moving on
- **Document changes** - especially data format changes
- **Update this checklist** as you progress

---

## Quick Reference: Test Counts by Module

| Module | Total | Passed | Failed | Skip | Pass % |
|--------|-------|--------|--------|------|--------|
| test_cai_integration.py | ~30 | ~4 | 26 | 0 | 13% |
| test_principles.py | ~60 | ~13 | 47 | 0 | 22% |
| test_reward_model.py | ~15 | ~3 | 12 | 0 | 20% |
| test_evaluator.py | ~15 | ~6 | 9 | 0 | 40% |
| test_filter.py | ~15 | ~4 | 11 | 0 | 27% |
| test_training.py | ~10 | ~4 | 6 | 0 | 40% |
| test_cai_training_integration.py | ~12 | ~5 | 7 | 0 | 42% |
| test_selfsupervised_losses.py | ~8 | ~4 | 4 | 0 | 50% |
| Others | ~731 | ~652 | ~25 | ~54 | 89% |

**Observation:** Core CAI modules have lowest pass rates - focus there first.
