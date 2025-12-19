---
description: Run pytest with common configurations
---

# Test Command

Run tests with various configurations based on arguments.

## Arguments

Parse `$ARGUMENTS` to determine test mode:

- **(empty)** or **"all"** - Run all tests with standard output
- **"coverage"** - Run with coverage report
- **"quick"** or **"q"** - Quick test run (minimal output)
- **"verbose"** or **"v"** - Verbose output with full tracebacks
- **{module_name}** - Run specific test module (e.g., "framework", "trainer", "principles")

## Instructions

Based on the arguments provided:

### Default / All Tests
```bash
pytest tests/ -v
```

### Coverage Mode
```bash
pytest tests/ --cov=src --cov-report=term-missing
```

### Quick Mode
```bash
pytest tests/ -q
```

### Verbose Mode
```bash
pytest tests/ -v --tb=long
```

### Specific Module
If argument is a module name (e.g., "framework"):
```bash
pytest tests/test_{module}.py -v
```

## Common Patterns

**Full test suite with coverage** (what CI runs):
```bash
pytest tests/ -v --cov=src --cov-report=term-missing
```

**Parallel execution** (faster):
```bash
pytest tests/ -n auto
```

**Stop on first failure**:
```bash
pytest tests/ -x
```

**Run only failed tests from last run**:
```bash
pytest tests/ --lf
```

## Current Test Status

- **Total tests**: 658
- **Status**: ~100% passing (after recent fixes)
- **Coverage**: ~45% (expected for ML research code)
- **Test files**: 14 modules in tests/

## Examples

- `/test` → Run all tests
- `/test coverage` → Run with coverage report
- `/test framework` → Run only test_framework.py
- `/test quick` → Quick run with minimal output
- `/test verbose` → Full output with detailed tracebacks
