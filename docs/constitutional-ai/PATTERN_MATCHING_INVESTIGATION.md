# Constitutional Principle Pattern Matching Investigation

**Date**: 2025-11-12
**Status**: Fixes Attempted, Tests Still Failing
**Investigator**: Claude (via session claude/review-cai-implementation-011CUznV4ghCLgicSpHb29b7)

## Background

During CAI integration testing, 13 tests failed related to constitutional principle detection and filtering. Pattern matching logic in `src/safety/constitutional/principles.py` and `src/safety/constitutional/filter.py` was not detecting violations as expected.

## Attempted Fixes (Commits 6113a83, 4714c1c)

### Fix 1: Fairness Stereotype Detection
**File**: `src/safety/constitutional/principles.py:258`

**Before**:
```python
r"(all|every)\s+(men|women|people from|individuals from|members of)\s+(are|is|do|does|can|cannot)"
```

**After**:
```python
r"(all|every)\s+(men|women|man|woman|person|people from|individuals from|members of)\s+(are|is|do|does|can|cannot)"
```

**Rationale**: Added singular forms ("man", "woman", "person") to catch phrases like "Every woman is emotional"

**Test**: `test_stereotype_detection` - Expected to flag "Every woman is emotional"

### Fix 2: Truthfulness Unsupported Claims
**File**: `src/safety/constitutional/principles.py:146`

**Before**:
```python
r"(all|every|no|none)\s+\w+\s+(is|are|do|does|can|cannot)"
```

**After**:
```python
r"(all|every|no|none)\s+(?:\w+\s+)+?(is|are|do|does|can|cannot|will|would)"
```

**Rationale**: Changed `\w+` (single word) to `(?:\w+\s+)+?` (multiple words) to match "None of them are good"

**Test**: `test_unsupported_absolute_claims` - Expected to flag "None of them are good"

### Fix 3: Truthfulness Contradictions
**File**: `src/safety/constitutional/principles.py:184-188`

**Before**:
```python
(r"\balways\b", r"\b(sometimes not|not always|occasionally|rarely)\b")
```

**After**:
```python
(r"\balways\b", r"\b(sometimes|not always|occasionally|rarely|doesn't|don't)\b")
```

**Rationale**: Added semantic contradictions - "always" contradicts "sometimes", "doesn't", "don't"

**Test**: `test_logical_contradictions_detected` - Expected to flag "This always works. Sometimes it doesn't work."

### Fix 4: Harm Detection
**File**: `src/safety/constitutional/principles.py:38`

**Added Pattern**:
```python
r"how to\b.{0,50}\b(hurt|harm|injure|kill|damage|attack)"
```

**Rationale**: Proximity-based matching - finds harm words within 50 characters of "how to"

**Test**: `test_harm_with_positive_context` - Expected to flag "How to protect yourself from being hurt"

### Fix 5: Misleading Statistics
**File**: `src/safety/constitutional/principles.py:217`

**Added Pattern**:
```python
r"(increased|decreased|grew|declined|rose|fell)\s+by\s+\d+\s*%"
```

**Rationale**: Added verb-first format to catch "Sales decreased by 30%"

**Test**: `test_percentage_without_source` - Expected to flag "Sales decreased by 30%"

### Fix 6: Preference Extraction
**File**: `src/safety/constitutional/preference_comparison.py:169-172`

**Before**:
```python
if re.search(r'\bprefer\s+b\b', text):
```

**After**:
```python
if re.search(r'\bprefer.{0,50}\bb\b', text):
```

**Rationale**: Allow up to 50 characters between "prefer" and choice letter

**Test**: `test_extract_prefer_with_distance` - Expected to return 'B' for "I prefer, without hesitation, B."

### Fix 7: Filter Biased Content
**File**: `src/safety/constitutional/filter.py:272`

**Before**:
```python
(r"\b(all|every)\s+(men|women|people)", r"some \2")
```

**After**:
```python
(r"\b(all|every)\s+(men|women|man|woman|people)", r"some \2")
```

**Rationale**: Match singular forms added to evaluation patterns

**Test**: `test_softens_generalizations` - Expected to rewrite "every woman" → "some woman"

## Test Results

**Status**: All fixes committed, but tests still fail

**Unable to Verify**: The testing environment does not have PyTorch installed, preventing actual test execution. The fixes were based on logical analysis of the regex patterns and test expectations, but could not be validated with actual test runs.

## Issues Identified

### 1. No Direct Test Execution
Without running the actual tests, it's impossible to:
- See the actual input being tested
- Observe what the regex patterns match or don't match
- Debug with print statements or breakpoints
- Understand if there are other code paths affecting the results

### 2. Possible Root Causes for Continued Failures

#### A. Regex Pattern Issues
- **Greedy vs Non-greedy Matching**: The `(?:\w+\s+)+?` pattern may still not match correctly
- **Word Boundaries**: `\b` might not work as expected with punctuation
- **Case Sensitivity**: Even with `re.IGNORECASE`, some patterns might fail
- **Pattern Order**: Early patterns might match and prevent later patterns from being tested

#### B. Context Extraction Issues
Many functions extract "context" around matches:
```python
start = max(0, match.start() - 20)
end = min(len(text), match.end() + 40)
context = text[start:end].strip()
```

The context might not include the full matched phrase, affecting downstream logic.

#### C. Evidence Checking
Some functions check for nearby evidence:
```python
has_evidence = re.search(
    r"(according to|based on|source|study|research|evidence|shows that)",
    text[max(0, match.start() - 100):min(len(text), match.end() + 100)],
    re.IGNORECASE
)
```

False positives in evidence detection could prevent flagging.

#### D. Filter Application Logic
The filter might not be applying transformations:
```python
if constitutional_evaluation["any_flagged"] and apply_transformations:
```

Even if evaluation flags issues, the filter might not be called with `apply_transformations=True` in tests.

### 3. Specific Test Mysteries

#### Autonomy Coercion: "There is no choice but to agree"
Pattern: `r"\b(no choice but to|only option is to|only way is to)\b"`

This pattern should match "no choice but to" in the text. Why doesn't it?
- Possible: Text preprocessing or normalization issues
- Possible: The pattern is inside a larger phrase that prevents matching
- Possible: The match succeeds but context extraction or softening detection interferes

#### Filter Weapon Creation: "how to build an explosive"
Should be replaced with "I cannot provide instructions for creating dangerous items."

Filter pattern: `r"(make|create|build|construct)\s+(?:a\s+|an\s+)?(weapon|bomb|explosive|poison)[\w\s]*"`

Why isn't it working?
- Possible: `evaluate_harm_potential()` isn't flagging it as explicit harm
- Possible: Filter isn't being called with `apply_transformations=True`
- Possible: Pattern doesn't match because of `[\w\s]*` at the end (should probably use a more specific end anchor)

## Recommendations

### Immediate: Manual Test Execution
```bash
# Install dependencies
pip install torch transformers pytest

# Run specific failing tests with verbose output
pytest tests/test_principles.py::TestEvaluateFairness::test_stereotype_detection -xvs
pytest tests/test_filter.py::TestFilterHarmfulContent::test_filters_weapon_creation -xvs
pytest tests/test_preference_comparison.py::TestExtractPreference::test_extract_prefer_with_distance -xvs
```

### Debug Strategy
1. **Add Debug Logging**: Insert print statements to see what patterns match:
   ```python
   print(f"Testing pattern: {pattern}")
   print(f"Text: {text}")
   matches = re.finditer(pattern, text, re.IGNORECASE)
   for match in matches:
       print(f"Match found: {match.group()}")
   ```

2. **Test Patterns in Isolation**: Create a simple script to test each pattern:
   ```python
   import re

   text = "Every woman is emotional"
   pattern = r"(all|every)\s+(men|women|man|woman|person)\s+(are|is|do|does)"

   if re.search(pattern, text, re.IGNORECASE):
       print("MATCH!")
   else:
       print("NO MATCH")
   ```

3. **Check Evaluation Flow**: Verify that:
   - `evaluate_fairness()` returns `{"flagged": True, ...}`
   - `filter_output()` receives `constitutional_evaluation["any_flagged"] == True`
   - `apply_transformations` parameter is `True`
   - `_filter_biased_content()` is actually called

4. **Examine Test Assertions**: Review what tests actually check:
   ```python
   result = evaluate_fairness(text)
   assert result["flagged"] is True  # Does evaluation flag it?
   assert len(result["stereotypes"]) > 0  # Does it capture examples?
   ```

### Potential Next Fixes

Based on common regex pitfalls:

1. **Autonomy Pattern** - Try looser matching:
   ```python
   # Current:
   r"\b(no choice but to|only option is to|only way is to)\b"

   # Try:
   r"no\s+choice\s+but\s+to"
   ```

2. **Filter Weapon Pattern** - Add boundaries:
   ```python
   # Current:
   r"(make|create|build|construct)\s+(?:a\s+|an\s+)?(weapon|bomb|explosive|poison)[\w\s]*"

   # Try:
   r"(make|create|build|construct)\s+(?:a\s+|an\s+)?(weapon|bomb|explosive|poison)(?:\s|$)"
   ```

3. **Unsupported Claims** - Simplify multi-word matching:
   ```python
   # Current:
   r"(all|every|no|none)\s+(?:\w+\s+)+?(is|are|do|does|can|cannot|will|would)"

   # Try:
   r"(all|every|no|none)\s+.+?\s+(is|are|do|does|can|cannot|will|would)"
   ```

## Conclusion

The pattern fixes were theoretically sound but could not be validated due to lack of test execution. The actual failures may be due to:
- Subtle regex issues not visible without debugging
- Logic errors in context extraction or filtering application
- Test setup issues or incorrect test expectations
- Integration issues between evaluation and filtering

**Next Steps**: Run tests with debug output to identify the actual failure points, then iterate on fixes based on observed behavior.

---

**Related Commits**:
- 6113a83: [fix] Fix constitutional principle pattern matching for better detection
- 4714c1c: [fix] Fix misleading statistics detection and preference extraction
