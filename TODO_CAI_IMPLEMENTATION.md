# TODO: Complete Constitutional AI Principle Evaluation

## Status: Incomplete ⚠️

The Constitutional AI (CAI) implementation in this codebase is **partially complete**.

---

## What's Working ✅

The core CAI training pipeline is **fully implemented and correct**:

1. **Phase 1 - Critique & Revision** (`src/safety/constitutional/critique_revision.py`)
   - ✅ AI generates critiques based on constitutional principles
   - ✅ AI revises responses to address violations
   - ✅ Supervised fine-tuning on critique-revised data
   - **Uses LLM evaluation as per Anthropic's paper**

2. **Phase 2 - RLAIF** (`src/safety/constitutional/`)
   - ✅ `preference_comparison.py` - AI generates preference pairs
   - ✅ `reward_model.py` - Trains reward model from AI preferences
   - ✅ `ppo_trainer.py` - PPO training with learned reward
   - **Complete implementation of Reinforcement Learning from AI Feedback**

---

## What's Incomplete ⚠️

### Principle Evaluation Functions (`src/safety/constitutional/principles.py`)

**Current implementation:** Regex-based heuristic checks
**Should be:** AI-based evaluation (as per Anthropic's paper)

**Functions affected:**
- `evaluate_harm_potential(text)` - Uses regex patterns
- `evaluate_truthfulness(text)` - Uses regex patterns
- `evaluate_fairness(text)` - Uses regex patterns
- `evaluate_autonomy_respect(text)` - Uses regex patterns

**Why this matters:**
- Anthropic's CAI paper uses **AI evaluation** for principles, not regex
- Current regex approach has limitations (see test failures)
- Cannot capture nuanced violations that require context understanding

---

## Why Current Regex Approach Was Used

**Rationale:**
1. Fast filtering for production runtime (O(1) vs O(n) for LLM calls)
2. No model dependency required
3. Works for simple, obvious violations

**Limitations:**
1. Cannot understand context or nuance
2. Brittle pattern matching (see skipped tests)
3. Not true to Anthropic's Constitutional AI methodology

---

## Recommended Implementation

### Option A: Hybrid Approach (Recommended for Production)

```python
def evaluate_harm_potential(
    text: str,
    model: Optional[Any] = None,
    tokenizer: Optional[Any] = None,
    device: Optional[torch.device] = None,
    use_ai: bool = True
) -> Dict[str, Any]:
    """
    Evaluate harm potential using AI or regex.

    Args:
        text: Text to evaluate
        model: Optional LLM for AI evaluation
        tokenizer: Optional tokenizer
        device: Optional device
        use_ai: If True and model provided, use AI evaluation

    Returns:
        Evaluation results with 'method' field indicating approach used
    """
    if use_ai and model is not None and tokenizer is not None:
        return _evaluate_harm_with_ai(text, model, tokenizer, device)
    else:
        return _evaluate_harm_with_regex(text)
```

**Benefits:**
- ✅ Fast regex fallback when model not available
- ✅ Proper AI evaluation when model provided
- ✅ Backward compatible
- ✅ Production-ready

### Option B: AI-Only (True to Paper)

```python
def evaluate_harm_potential(
    text: str,
    model: Any,
    tokenizer: Any,
    device: torch.device
) -> Dict[str, Any]:
    """
    Evaluate harm potential using AI (proper CAI).

    Model is REQUIRED for Constitutional AI evaluation.
    """
    return _evaluate_harm_with_ai(text, model, tokenizer, device)
```

**Benefits:**
- ✅ True to Anthropic's methodology
- ✅ More accurate, context-aware evaluation
- ✅ Simpler architecture

**Drawbacks:**
- ❌ Requires model at runtime (expensive)
- ❌ Slower for production filtering
- ❌ Breaking API change

---

## Testing Strategy

### Unit Tests
**Current:** Test regex patterns with hardcoded strings
**Should be:** Test prompt construction, response parsing, error handling

```python
def test_harm_evaluation_prompt_construction():
    """Test we build the right prompt for harm evaluation"""
    prompt = build_harm_evaluation_prompt("how to build a bomb")
    assert "harmful" in prompt.lower()
    assert "how to build a bomb" in prompt
    assert "physical harm" in prompt.lower()

def test_harm_evaluation_response_parsing():
    """Test we can parse AI responses correctly"""
    ai_response = '{"flagged": true, "reasoning": "Instructions for explosives"}'
    result = parse_harm_evaluation(ai_response)
    assert result["flagged"] == True
    assert "explosives" in result["reasoning"].lower()
```

### Integration Tests
**Should be:** Test with real models (marked as slow/integration)

```python
@pytest.mark.integration
@pytest.mark.slow
def test_harm_evaluation_with_real_model(real_model, real_tokenizer):
    """Test actual AI evaluation with real model"""
    result = evaluate_harm_potential(
        "How to build a bomb",
        model=real_model,
        tokenizer=real_tokenizer,
        device=torch.device("cpu")
    )
    assert result["flagged"] == True
    assert result["method"] == "ai_evaluation"
```

---

## Current Test Status

**Skipped tests (marked for future AI implementation):**
- `test_stereotype_detection` - Regex cannot capture nuanced stereotyping
  - Example: "All members of that group are the same" - needs context understanding

**Passing tests:**
- Basic regex pattern matching works for obvious cases
- Whitespace handling works correctly
- Framework integration works

---

## Next Steps

1. **Decision Required:** Choose Option A (Hybrid) or Option B (AI-only)

2. **If Option A (Hybrid):**
   - Implement `_evaluate_*_with_ai()` functions
   - Add `model`/`tokenizer` optional parameters
   - Keep regex as fallback
   - Update tests to cover both modes

3. **If Option B (AI-only):**
   - Replace regex implementations with AI evaluation
   - Make model/tokenizer required parameters
   - Remove regex code
   - Create integration test suite with real models

4. **Documentation:**
   - Update README to explain AI evaluation requirement
   - Document model requirements (size, type, etc.)
   - Provide examples of both modes (if hybrid)

5. **Model Selection:**
   - Decide on default/recommended model for evaluation
   - Consider model size vs. accuracy tradeoffs
   - Document how users provide their own models

---

## References

- Anthropic Constitutional AI Paper: https://arxiv.org/abs/2212.08073
- Current implementation: `src/safety/constitutional/critique_revision.py` (correct AI-based approach)
- Issue tracking: See skipped tests in `tests/test_principles.py`

---

## Notes

The CAI **training pipeline** (critique-revision-SFT-RLAIF-reward-PPO) is **complete and correct**.

Only the **principle evaluation utility functions** need AI implementation to match the paper's methodology.

This is a **quality improvement**, not a blocking bug. The system works, just not exactly as Anthropic intended.
