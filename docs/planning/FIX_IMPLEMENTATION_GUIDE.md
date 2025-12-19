# Fix Implementation Guide

**Quick Start Guide for Fixing Top 3 Test Failure Categories**

---

## Overview

This guide provides step-by-step instructions to fix 108 failing tests (73.5% of all failures).

**Time Required:** 60-80 minutes
**Difficulty:** Low
**Risk:** Very Low

---

## Pre-Flight Checklist

Before starting, verify:

- [ ] Current test results: Run `./run_tests.sh` and confirm 695 tests passing
- [ ] Git status clean or committed: `git status`
- [ ] Create branch: `git checkout -b fix/top-3-test-failures`
- [ ] Python environment active: `python --version` shows 3.10+

---

## Fix 1: Training Data Format (Easiest First)

**Time:** 10-15 minutes | **Tests Fixed:** 12 | **Risk:** None

### Step 1.1: Open the test file
```bash
code tests/test_reward_model.py
# Or use your preferred editor
```

### Step 1.2: Find the fixture (around line 61)
```python
@pytest.fixture
def sample_preference_data():
    """Create sample preference data for testing."""
    return [
        {
            "prompt": "What is photosynthesis?",
            "response_chosen": "Photosynthesis is...",  # ← Change this
            "response_rejected": "Plants make food.",   # ← Change this
        },
        # ... 3 more examples
    ]
```

### Step 1.3: Replace key names in all 4 examples
Use find/replace or manual editing:

**Find:** `"response_chosen"`
**Replace:** `"chosen"`

**Find:** `"response_rejected"`
**Replace:** `"rejected"`

**After changes:**
```python
@pytest.fixture
def sample_preference_data():
    """Create sample preference data for testing."""
    return [
        {
            "prompt": "What is photosynthesis?",
            "chosen": "Photosynthesis is...",     # ← Fixed
            "rejected": "Plants make food.",      # ← Fixed
        },
        # ... 3 more examples (fix all of them)
    ]
```

### Step 1.4: Check if test_cai_training_integration.py needs fixes
```bash
grep -n "response_chosen\|response_rejected" tests/test_cai_training_integration.py
# If nothing found, you're done. Otherwise, apply same changes.
```

### Step 1.5: Test the fix
```bash
python -m pytest tests/test_reward_model.py -v

# Expected output:
# ✓ 10 tests should now pass
# Look for: test_training_completes, test_train_method, etc.
```

### Step 1.6: Commit progress
```bash
git add tests/test_reward_model.py
git commit -m "fix: Update reward model test fixtures to use correct key names (chosen/rejected)"
```

**Checkpoint:** 12 tests fixed ✓

---

## Fix 2: Training Validation (Next Priority)

**Time:** 20 minutes | **Tests Fixed:** 5 | **Risk:** Low

### Step 2.1: Open the source file
```bash
code src/safety/constitutional/critique_revision.py
```

### Step 2.2: Find data generation function (around line 340)
Search for: `generate_critiques_and_revisions`

Look for where training examples are created:
```python
example = {
    "prompt": prompt,
    "initial_response": initial_response,
    "critique": critique_text,
    "revised_response": revised_response,  # ← Find this line
    "evaluation": evaluation,
    "revision_history": revision_history,
}
```

### Step 2.3: Add response alias
Add one line after `"revised_response"`:
```python
example = {
    "prompt": prompt,
    "initial_response": initial_response,
    "critique": critique_text,
    "revised_response": revised_response,
    "response": revised_response,  # ← ADD THIS LINE (alias for training)
    "evaluation": evaluation,
    "revision_history": revision_history,
}
```

### Step 2.4: Find validation function (around line 570)
Search for: `supervised_fine_tune_on_revised`

Find the validation loop:
```python
# Filter out invalid training examples
valid_data = []
for idx, item in enumerate(training_data):
    # Check if required fields exist and are non-empty
    if "prompt" not in item or "response" not in item:  # ← Find this
        _logger.info(f"Warning: Skipping training example {idx}: missing prompt or response")
        continue

    prompt = item.get("prompt", "").strip()
    response = item.get("response", "").strip()  # ← And this
```

### Step 2.5: Update validation to accept both keys
Replace the validation section with:
```python
# Filter out invalid training examples
valid_data = []
for idx, item in enumerate(training_data):
    # Check if required fields exist and are non-empty
    if "prompt" not in item:
        _logger.info(f"Warning: Skipping training example {idx}: missing prompt")
        continue

    # Accept either 'response' or 'revised_response'
    if "response" not in item and "revised_response" not in item:
        _logger.info(f"Warning: Skipping training example {idx}: missing response/revised_response")
        continue

    prompt = item.get("prompt", "").strip()
    # Prefer 'response', fall back to 'revised_response'
    response = item.get("response") or item.get("revised_response", "")
    response = response.strip()

    if not prompt or not response:
        _logger.info(f"Warning: Skipping training example {idx}: empty prompt or response")
        continue

    # Check for NaN or None values
    if prompt == "nan" or response == "nan" or prompt == "None" or response == "None":
        _logger.info(f"Warning: Skipping training example {idx}: NaN or None value detected")
        continue

    # Normalize to 'response' key for dataset
    normalized_item = dict(item)
    normalized_item["response"] = response
    valid_data.append(normalized_item)
```

### Step 2.6: Test the fix
```bash
python -m pytest tests/test_cai_training_integration.py::TestPhase1Training -v

# Expected output:
# ✓ test_phase1_pipeline_runs - should pass
# ✓ test_phase1_checkpoint_save_load - should pass
```

### Step 2.7: Test end-to-end pipeline
```bash
python -m pytest tests/test_cai_training_integration.py::TestEndToEndPipeline -v

# Expected: 3 more tests pass
```

### Step 2.8: Commit progress
```bash
git add src/safety/constitutional/critique_revision.py
git commit -m "fix: Add response alias and flexible validation for training data"
```

**Checkpoint:** 17 tests fixed (12 + 5) ✓

---

## Fix 3: Logger Initialization (Biggest Impact)

**Time:** 30-45 minutes | **Tests Fixed:** 91 | **Risk:** Very Low

### Step 3.1: Open the source file
```bash
code src/safety/constitutional/principles.py
```

### Step 3.2: Understand the pattern
The file has functions that accept `logger=None` parameter but call logger methods without checking:

**BROKEN:**
```python
def some_function(text, logger=None):
    logger.info("message")  # ← Crashes if logger is None!
```

**FIXED:**
```python
def some_function(text, logger=None):
    if logger:
        logger.info("message")  # ← Safe, only calls if logger exists
```

### Step 3.3: Find all unguarded logger calls
```bash
# Search for logger calls that aren't guarded
grep -n "^\s*logger\.[a-z_]*(" src/safety/constitutional/principles.py | \
    grep -v "if logger:"

# This will show all lines that need fixing
```

### Step 3.4: Fix each function systematically

**Functions to fix (in order):**

#### 3.4.1: `_evaluate_harm_with_ai()` (around line 418)
Find these unguarded calls:
```python
logger.info("Evaluating HARM with AI...", level=2, prefix="EVAL")
logger.info(f"Text: {text[:100]}...", level=2, prefix="EVAL")
logger.info(f"Response ({len(response)} chars)", level=2, prefix="EVAL")
logger.info(f"AI eval failed: {e}, using regex", level=1, prefix="EVAL")
```

Wrap each with `if logger:`:
```python
if logger:
    logger.info("Evaluating HARM with AI...", level=2, prefix="EVAL")
    logger.info(f"Text: {text[:100]}...", level=2, prefix="EVAL")

# ... in try block ...
if logger:
    logger.info(f"Response ({len(response)} chars)", level=2, prefix="EVAL")

# ... in except block ...
if logger:
    logger.info(f"AI eval failed: {e}, using regex", level=1, prefix="EVAL")
```

#### 3.4.2: `evaluate_harm_potential()` (around line 565)
Find these unguarded calls:
```python
logger.info("Regex detected harm - trusting regex", level=1, prefix="HARM")
logger.info("Regex found nothing, trying AI...", level=2, prefix="HARM")
logger.info(f"AI found harm (regex missed it)", level=2, prefix="HARM")
```

Wrap each:
```python
if logger:
    logger.info("Regex detected harm - trusting regex", level=1, prefix="HARM")

if logger:
    logger.info("Regex found nothing, trying AI...", level=2, prefix="HARM")

if logger:
    logger.info(f"AI found harm (regex missed it)", level=2, prefix="HARM")
```

#### 3.4.3: Repeat for other functions
Apply the same pattern to:
- `_evaluate_truthfulness_with_ai()` (around line 728)
- `evaluate_truthfulness()` (around line 850)
- `_evaluate_fairness_with_ai()` (around line 976)
- `evaluate_fairness()` (around line 1100)
- `_evaluate_autonomy_with_ai()` (around line 1157)
- `evaluate_autonomy_respect()` (around line 1280)

**Tip:** Use editor's multi-cursor or find/replace to speed this up:
1. Select `logger.info(` pattern
2. Add `if logger:\n        ` before it
3. Indent the logger call

### Step 3.5: Test incrementally
After fixing each function, test it:

```bash
# Test harm evaluation
python -m pytest tests/test_principles.py::TestEvaluateHarmPotential -v

# Test truthfulness evaluation
python -m pytest tests/test_principles.py::TestEvaluateTruthfulness -v

# Test all principles
python -m pytest tests/test_principles.py -v
```

### Step 3.6: Test integration
```bash
# Test CAI integration
python -m pytest tests/test_cai_integration.py -v

# Test evaluator
python -m pytest tests/test_evaluator.py -v

# Test filter
python -m pytest tests/test_filter.py -v
```

### Step 3.7: Verify all logger calls are guarded
```bash
# Should return no results (or only false positives)
grep -n "^\s*logger\.[a-z_]*(" src/safety/constitutional/principles.py | \
    grep -v "if logger:" | \
    grep -v "logger = get_logger"

# Manually check any results to confirm they're safe
```

### Step 3.8: Commit progress
```bash
git add src/safety/constitutional/principles.py
git commit -m "fix: Add null checks for optional logger parameter in all principle evaluators"
```

**Checkpoint:** 108 tests fixed (17 + 91) ✓

---

## Final Verification

### Run full test suite
```bash
./run_tests.sh

# Watch the output carefully
```

### Expected results:
```
Tests passing: 803/895 (89.7%)  ← Up from 695 (77.6%)
Tests failing: 39/895 (4.4%)    ← Down from 147 (16.4%)
Coverage: ~40-42%               ← Up from 36.38%
```

### Verify specific numbers:
```bash
# Count passing tests
pytest --co -q | wc -l  # Total tests
pytest --tb=no -q | grep "passed" | head -1  # Passing tests
```

### Check for regressions
```bash
# Make sure no previously passing tests now fail
# If any new failures appear, investigate immediately
```

---

## Troubleshooting

### If Fix 1 tests still fail
```bash
# Check you changed ALL 4 examples in the fixture
grep -A 2 '"prompt":' tests/test_reward_model.py | grep -E "chosen|rejected"

# Should show "chosen" and "rejected", not "response_chosen" or "response_rejected"
```

### If Fix 2 tests still fail
```bash
# Verify both changes were made:
# 1. Check data generation has 'response' alias
grep -A 5 '"revised_response": revised_response' src/safety/constitutional/critique_revision.py | grep '"response"'

# 2. Check validation accepts both keys
grep -A 3 'if "response" not in item' src/safety/constitutional/critique_revision.py | grep "revised_response"
```

### If Fix 3 tests still fail
```bash
# Find any remaining unguarded logger calls
grep -n "^\s*logger\.[a-z_]*(" src/safety/constitutional/principles.py | \
    grep -v "if logger:" | \
    grep -v "logger = get_logger"

# Each result needs an 'if logger:' guard added
```

### If tests pass locally but fail in CI
```bash
# Check Python version
python --version  # Should match CI version

# Run with same pytest version as CI
pip list | grep pytest

# Check for environment-specific issues
env | grep -E "PYTHON|TEST|LOG"
```

---

## Commit & Push

### Review changes
```bash
git status
git diff --cached
```

### Final commit (if needed)
```bash
# If you made additional fixes
git add .
git commit -m "fix: Final adjustments for test failures"
```

### Push branch
```bash
git push origin fix/top-3-test-failures
```

### Create PR
```bash
# Use GitHub CLI (if installed)
gh pr create --title "Fix top 3 test failure categories (108 tests)" \
    --body "Fixes 73.5% of all test failures:
- Logger initialization (91 tests)
- Training data format (12 tests)
- Training validation (5 tests)

Pass rate increases from 77.6% to 89.7%"
```

Or create PR manually on GitHub.

---

## Success Criteria Checklist

- [ ] Fix 1: 12 reward model tests pass
- [ ] Fix 2: 5 Phase 1 training tests pass
- [ ] Fix 3: 91 logger-related tests pass
- [ ] Total: 803/895 tests passing (89.7%)
- [ ] No new test failures introduced
- [ ] All changes committed with clear messages
- [ ] Branch pushed to remote
- [ ] PR created (optional)

---

## Next Steps

After these fixes are merged:

1. **Address remaining 39 failures** - Now clearly visible
2. **Increase coverage** - Target 45% (currently ~40-42%)
3. **Run mutation testing** - Ensure test quality
4. **Performance optimization** - Address any slow tests

---

## Time Log Template

Track your progress:

```
Start time: ____:____
Fix 1 complete: ____:____ (Duration: ____ min)
Fix 2 complete: ____:____ (Duration: ____ min)
Fix 3 complete: ____:____ (Duration: ____ min)
Verification complete: ____:____ (Duration: ____ min)
Total time: ____ minutes

Notes:
- Challenges encountered: ________________
- Surprises: ________________
- Additional fixes needed: ________________
```

---

## Quick Reference

### Key Files Modified
1. `tests/test_reward_model.py` - 4 key renames
2. `tests/test_cai_training_integration.py` - Mock data (if needed)
3. `src/safety/constitutional/critique_revision.py` - 11 lines
4. `src/safety/constitutional/principles.py` - ~50 guards

### Test Commands
```bash
# Fix 1
pytest tests/test_reward_model.py -v

# Fix 2
pytest tests/test_cai_training_integration.py::TestPhase1Training -v

# Fix 3
pytest tests/test_principles.py tests/test_cai_integration.py -v

# All
./run_tests.sh
```

### Git Commands
```bash
git checkout -b fix/top-3-test-failures
git add <file>
git commit -m "fix: <message>"
git push origin fix/top-3-test-failures
```

---

**Good luck! The fixes are straightforward and will have a massive impact on test suite health.**
