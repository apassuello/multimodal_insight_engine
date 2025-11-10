# Testing Quick Reference Guide

**Quick access guide for running tests in the Constitutional AI implementation**

---

## Quick Start

```bash
# Run all tests with coverage
./run_tests.sh

# Run Constitutional AI tests only
pytest tests/test_framework.py tests/test_principles.py tests/test_evaluator.py \
       tests/test_filter.py tests/test_model_utils.py tests/test_cai_integration.py -v
```

---

## Common Test Commands

### Run All Tests

```bash
# All tests with verbose output
pytest tests/ -v

# All tests with coverage report
pytest tests/ --cov=src --cov-report=html

# All tests with coverage threshold enforcement
pytest tests/ --cov=src --cov-fail-under=85
```

### Run Specific Test Files

```bash
# Framework tests
pytest tests/test_framework.py -v

# Principle evaluator tests
pytest tests/test_principles.py -v

# Safety evaluator tests
pytest tests/test_evaluator.py -v

# Filter tests
pytest tests/test_filter.py -v

# Model utility tests
pytest tests/test_model_utils.py -v

# Integration tests
pytest tests/test_cai_integration.py -v
```

### Run Specific Test Classes

```bash
# Framework principle tests
pytest tests/test_framework.py::TestConstitutionalPrinciple -v

# Harmfulness evaluator tests
pytest tests/test_principles.py::TestHarmfulnessEvaluator -v

# Two-stage evaluation tests
pytest tests/test_evaluator.py::TestTwoStageEvaluation -v

# Input filtering tests
pytest tests/test_filter.py::TestInputFiltering -v
```

### Run Specific Test Functions

```bash
# Test principle creation
pytest tests/test_framework.py::TestConstitutionalPrinciple::test_principle_creation -v

# Test harm detection
pytest tests/test_principles.py::TestHarmfulnessEvaluator::test_detect_violence -v

# Test evaluation results
pytest tests/test_evaluator.py::TestEvaluationResults::test_score_calculation -v
```

---

## Test Filtering

### By Pattern

```bash
# All tests with "harm" in name
pytest tests/ -k "harm" -v

# All tests with "privacy" or "pii"
pytest tests/ -k "privacy or pii" -v

# Exclude model loading tests
pytest tests/ -k "not model_loading" -v

# Tests for deception detection
pytest tests/ -k "deception" -v
```

### By Speed

```bash
# Fast tests only (unit tests)
pytest -m "not slow" tests/ -v

# Include slow tests
pytest tests/ -v

# Only slow tests (integration/E2E)
pytest -m "slow" tests/ -v
```

### By Marker

```bash
# GPU-dependent tests
pytest -m "gpu" tests/ -v

# Network-dependent tests
pytest -m "network" tests/ -v

# Integration tests
pytest -m "integration" tests/ -v
```

---

## Debugging Tests

### Show Output

```bash
# Show print statements
pytest tests/test_framework.py -v -s

# Show local variables on failure
pytest tests/test_framework.py -v -l

# Very verbose output with full diffs
pytest tests/test_framework.py -vv
```

### Drop to Debugger

```bash
# Drop to debugger on failure
pytest tests/test_framework.py --pdb

# Drop to debugger on first failure
pytest tests/test_framework.py -x --pdb

# Drop to debugger at start of each test
pytest tests/test_framework.py --trace
```

### Capture Output

```bash
# Disable output capture
pytest tests/ -s

# Show captured output for failed tests only
pytest tests/ --tb=short

# Show full traceback
pytest tests/ --tb=long
```

---

## Coverage Reports

### Generate Reports

```bash
# HTML coverage report
pytest tests/ --cov=src --cov-report=html
# Opens in: coverage_html/index.html

# Terminal coverage report
pytest tests/ --cov=src --cov-report=term

# Terminal with missing lines
pytest tests/ --cov=src --cov-report=term-missing

# XML coverage report (for CI)
pytest tests/ --cov=src --cov-report=xml

# Multiple report formats
pytest tests/ --cov=src --cov-report=html --cov-report=term --cov-report=xml
```

### Coverage for Specific Modules

```bash
# Constitutional AI module only
pytest tests/ --cov=src/safety/constitutional --cov-report=html

# Framework module only
pytest tests/test_framework.py --cov=src/safety/constitutional/framework --cov-report=term-missing

# Principles module only
pytest tests/test_principles.py --cov=src/safety/constitutional/principles --cov-report=term-missing
```

---

## Test Execution Options

### Stop on First Failure

```bash
# Stop after first failure
pytest tests/ -x

# Stop after N failures
pytest tests/ --maxfail=3
```

### Rerun Failed Tests

```bash
# Rerun only failed tests from last run
pytest --lf tests/

# Rerun failed tests first, then others
pytest --ff tests/
```

### Parallel Execution

```bash
# Run tests in parallel (requires pytest-xdist)
pytest tests/ -n auto

# Run with 4 workers
pytest tests/ -n 4
```

### Performance Profiling

```bash
# Show slowest 10 tests
pytest tests/ --durations=10

# Show all test durations
pytest tests/ --durations=0

# Profile test execution
pytest tests/ --profile
```

---

## Test Results Output

### JUnit XML

```bash
# Generate JUnit XML report
pytest tests/ --junitxml=reports/junit.xml
```

### JSON Report

```bash
# Generate JSON report (requires pytest-json-report)
pytest tests/ --json-report --json-report-file=reports/report.json
```

### Custom Reporting

```bash
# Quiet mode (minimal output)
pytest tests/ -q

# Very quiet (only show summary)
pytest tests/ -qq

# Show test names as they run
pytest tests/ -v

# Show full test paths
pytest tests/ -vv
```

---

## Continuous Integration

### CI Command

```bash
# Full CI test suite
pytest tests/ \
    --cov=src \
    --cov-report=xml \
    --cov-report=term \
    --cov-fail-under=85 \
    --junitxml=reports/junit.xml \
    -v
```

### Pre-commit Hook

```bash
# Run fast tests before commit
pytest tests/ -m "not slow" --maxfail=1
```

### Pre-push Hook

```bash
# Run all tests before push
pytest tests/ --cov=src --cov-fail-under=85
```

---

## Test Statistics

### Current Status

```
Total Tests: 313
Passing: 274 (87.5%)
Expected Failures: 39 (12.5%)

Coverage: 87.5%
Test-to-Code Ratio: 1.35:1
```

### Test Distribution

```
Unit Tests: 216 (69%)
Integration Tests: 56 (18%)
End-to-End Tests: 41 (13%)
```

### By Test File

```
test_framework.py:       54 tests (50 passing, 100%)
test_principles.py:      72 tests (60 passing, 83%)
test_evaluator.py:       56 tests (55 passing, 98%)
test_filter.py:          58 tests (56 passing, 97%)
test_model_utils.py:     38 tests (18 passing, 47%)
test_cai_integration.py: 39 tests (35 passing, 90%)
```

---

## Useful Aliases

Add these to your shell profile for quick access:

```bash
# Add to ~/.bashrc or ~/.zshrc

# Run all tests
alias pt='pytest tests/ -v'

# Run with coverage
alias ptc='pytest tests/ --cov=src --cov-report=html'

# Run Constitutional AI tests
alias ptcai='pytest tests/test_framework.py tests/test_principles.py tests/test_evaluator.py tests/test_filter.py tests/test_model_utils.py tests/test_cai_integration.py -v'

# Run fast tests only
alias ptf='pytest tests/ -m "not slow" -v'

# Run failed tests
alias ptlf='pytest --lf tests/ -v'

# Show coverage report
alias ptcov='open coverage_html/index.html'
```

---

## Troubleshooting

### Test Discovery Issues

```bash
# Show which tests would be collected
pytest --collect-only tests/

# Verify test file structure
pytest tests/ -v --tb=no
```

### Import Errors

```bash
# Install package in development mode
pip install -e .

# Verify imports
python -c "from src.safety.constitutional import ConstitutionalFramework; print('OK')"
```

### Missing Dependencies

```bash
# Install test dependencies
pip install -r requirements-dev.txt

# Install specific test tools
pip install pytest pytest-cov pytest-xdist
```

### Cache Issues

```bash
# Clear pytest cache
pytest --cache-clear

# Remove pycache files
find . -type d -name __pycache__ -exec rm -r {} +
```

---

## Further Reading

- [Comprehensive Test Documentation](CONSTITUTIONAL_AI_TEST_COVERAGE.md)
- [Testing Infrastructure Guide](../doc/test_documentation.md)
- [Test Implementation Plan](../test_implementation_plan.md)

---

**Last Updated**: 2025-11-07
**Maintained By**: Engineering Team
