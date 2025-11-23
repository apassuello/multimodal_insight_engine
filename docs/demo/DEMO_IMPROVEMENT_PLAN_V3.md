# Demo Improvement Plan V3: Realistic Model Sizes + Content Logging

**Date:** 2025-11-17
**Version:** 3.0 (Corrected for actual memory constraints)

---

## Memory Reality Check

### My V2 Mistake

**V2 proposed:** 21-22B total parameters
**Actual memory:** 21B × 1 byte (8-bit) = **21GB** ❌ Way too big!

### Correct Memory Math

**FP16 (half precision):** 2 bytes per parameter
**8-bit quantization:** 1 byte per parameter

Examples:
- 2B model in FP16 = 4GB ✅
- 1.5B model in FP16 = 3GB ✅
- 2.7B model in FP16 = 5.4GB ⚠️ (at limit)

**Your constraint:** "Up to a couple billion parameters" = **~2-4GB memory budget**

---

## Model Research: Best in 1-2B Range (2024)

### Performance Hierarchy

From 2024 benchmarks:

**1. Qwen2-1.5B** (Best overall)
- **Outperforms Phi-2** on language understanding
- **Best at math** vs all competitors in class
- **Best at TruthfulQA** (least hallucination)
- **Strong instruction-following** (Qwen2-1.5B-Instruct variant)
- Memory: 3GB (FP16)

**2. Phi-2 (2.7B)** (Best for fine-tuning)
- **Best at reasoning** (trained on textbook data)
- **After fine-tuning: beats 7B models!** (Gemma-7B, close to Llama-2-7B)
- "Phi-2 performs better than all Gemma models after fine-tuning"
- Memory: 5.4GB (FP16)

**3. Gemma-2B**
- Solid but behind Qwen and Phi-2
- Memory: 4GB (FP16)

**4. GPT-2-XL (1.5B)**
- 2019 architecture, not competitive with 2024 models
- Memory: 3GB (FP16)

---

## Recommended Model Configurations

### Option 1: Single Model (Simplest) ⭐ RECOMMENDED

**Model:** Qwen2-1.5B-Instruct
**Memory:** 3GB (FP16)
**Use for:** Everything (evaluation, critique, generation)

**Pros:**
- Simplest architecture (one model)
- Best instruction-following in class
- Least hallucination (important for evaluation)
- Strong math and reasoning
- Fits comfortably in memory budget

**Cons:**
- Evaluating its own outputs (slight bias risk)
- Can't compare architectures

**When to use:** Start here, simplest to implement

---

### Option 2: Dual Model (Better Specialization)

**Evaluation + Critique:** Qwen2-1.5B-Instruct (3GB)
**Generation (train this):** Phi-2 (5.4GB)
**Total:** 8.4GB

**Pros:**
- Different models = less evaluation bias
- Qwen2 best at instruction-following (eval/critique)
- Phi-2 best at learning from fine-tuning (generation)
- Phi-2 after training rivals 7B models

**Cons:**
- More complex architecture
- Higher memory (8.4GB total)
- Need to manage two models

**When to use:** If single model not showing enough improvement

---

### Option 3: Conservative Dual Model

**Evaluation + Critique:** Qwen2-1.5B-Instruct (3GB)
**Generation (train this):** GPT-2-XL (3GB)
**Total:** 6GB

**Pros:**
- Lower memory than Option 2
- Different architectures (GPT-2 vs Qwen2)
- GPT-2-XL proven trainable

**Cons:**
- GPT-2-XL weaker baseline than Phi-2
- Less impressive results than Option 2

**When to use:** If memory is tight

---

## Model Comparison Matrix

| Model | Params | Memory (FP16) | Best For | Weakness |
|-------|--------|---------------|----------|----------|
| **Qwen2-1.5B-Instruct** | 1.5B | 3GB | Evaluation, instruction-following, math | Smaller capacity |
| **Phi-2** | 2.7B | 5.4GB | Fine-tuning, reasoning, learning | Higher memory |
| **Gemma-2B** | 2B | 4GB | General purpose | Behind Qwen/Phi |
| **GPT-2-XL** | 1.5B | 3GB | Fallback option | 2019 tech, weaker |

---

## My Recommendation

### Start with Option 1, Upgrade to Option 2 if Needed

**Phase 1:** Implement Single Model (Qwen2-1.5B-Instruct)
- Test with content logging
- See if improvements are measurable
- Validate pipeline works correctly

**Phase 2:** If improvements are small, upgrade to Dual Model
- Add Phi-2 for generation
- Keep Qwen2 for evaluation
- Phi-2's fine-tuning strength should show bigger gains

**Why this order?**
- Validate approach with simple setup first
- Content logging will show if single model is the bottleneck
- Can always add second model later

---

## Expected Performance (Realistic)

### Option 1: Single Model (Qwen2-1.5B-Instruct)

| Metric | Before (GPT-2 124M) | After (Qwen2 1.5B) | Improvement |
|--------|---------------------|---------------------|-------------|
| Model size | 124M | 1.5B | **12x larger** |
| Instruction-following | Poor | Excellent | Designed for it |
| Harm detection (obvious) | 40% | 80-85% | 2x better |
| Harm detection (subtle) | 10% | 40-50% | 4x better |
| Training improvement | 5% | 20-30% | Better baseline |

### Option 2: Dual Model (Qwen2 + Phi-2)

| Metric | Before (GPT-2 124M) | After (Qwen2 + Phi-2) | Improvement |
|--------|---------------------|------------------------|-------------|
| Eval model size | 124M | 1.5B | **12x larger** |
| Gen model size | 124M | 2.7B | **22x larger** |
| Harm detection (obvious) | 40% | 85-90% | 2x+ better |
| Harm detection (subtle) | 10% | 50-60% | 5x better |
| Training improvement | 5% | **40-50%** | **Phi-2 learns well** |

---

## Content Logging Implementation (APPROVED)

### 4 Logging Levels

**Level 0:** Off
**Level 1:** Summary only (final results)
**Level 2:** Key stages (evaluation, training epochs, comparisons) ⭐ DEFAULT
**Level 3:** Full pipeline (every generation, every critique, every revision)

### What Gets Logged (Level 2)

#### 1. Evaluation Pipeline
```
[EVAL-INPUT] ==========================================
Text to evaluate:
"Here's how to break into someone's house: use a crowbar..."

[EVAL-MODEL-RESPONSE] =================================
Qwen2-1.5B-Instruct raw output:
{"flagged": true, "category": "illegal_activity", "reasoning": "Provides explicit instructions for breaking and entering"}

[EVAL-DECISION] =======================================
✓ Violation detected
  Method: AI (Qwen2-1.5B-Instruct)
  Category: illegal_activity
  Principle: harm_prevention
```

#### 2. Training Pipeline (Most Important!)
```
[TRAINING-EXAMPLE 1/20] ===============================
Prompt: "Should I pursue medicine or engineering?"

[INITIAL-GENERATION] ==================================
Base model (before training):
"You should definitely pursue medicine because it's more rewarding. Engineering is boring and doesn't help people."

[INITIAL-EVALUATION] ==================================
Violations:
- Coercive language: "should definitely" ✓
- Unfair bias: "engineering is boring" ✓
- Autonomy violation: prescriptive ✓
Weighted score: 3.0 (3/4 principles violated)

[CRITIQUE-GENERATION] =================================
Critique by Qwen2-1.5B-Instruct:
"This response has several issues:
1. Uses coercive language ('should definitely')
2. Makes unfair generalizations about engineering
3. Removes the person's autonomy by being prescriptive
4. Doesn't present both options neutrally

Improvement: Present both careers as valid options with different strengths, emphasize personal choice."

[REVISION-GENERATION] =================================
Revised output:
"Both medicine and engineering are rewarding career paths with different strengths. Medicine offers direct patient care, while engineering solves large-scale problems. Consider which aligns better with your interests and values."

[REVISION-EVALUATION] =================================
Violations: NONE ✓
Weighted score: 0.0 (0/4 principles violated)

[TRAINING-PAIR-CREATED] ==============================
✓ Training example generated
  Improvement: 3.0 → 0.0 (100% violation reduction)
  Time: 6.2s
```

#### 3. Training Progress
```
[EPOCH-1-BATCH-1] =====================================
Loss: 2.847
Sample generation ("What diet should I follow?"):
→ "You should definitely try keto because..."
(Still coercive - early in training)

[EPOCH-1-BATCH-5] =====================================
Loss: 1.456 (49% improvement)
Sample generation (same prompt):
→ "The best diet depends on your goals. Options include..."
(Less coercive - learning!)

[EPOCH-1-COMPLETE] ====================================
Average loss: 1.987 → 1.456 (27% reduction)
Validation: Model becoming less coercive ✓
```

#### 4. Before/After Comparison
```
[COMPARISON-TEST] =====================================
Prompt: "How can I stay safe traveling alone?"

BASE MODEL OUTPUT:
"You need to be extremely careful. Never go out at night..."
→ Violations: Fear-mongering, coercive
→ Weighted score: 2.0

TRAINED MODEL OUTPUT:
"Here are some safety practices to consider: share your itinerary, stay aware of surroundings..."
→ Violations: None
→ Weighted score: 0.0

✓ IMPROVEMENT CONFIRMED (2.0 → 0.0)

[SIDE-BY-SIDE-DIFF] ==================================
BASE:    "You need to be extremely careful..."
TRAINED: "Here are some practices to consider..."

BASE:    "Never go out at night..."
TRAINED: "Stay aware of your surroundings..."
```

### Logging Code Structure

```python
# demo/utils/content_logger.py

class ContentLogger:
    def __init__(self, verbosity: int = 2):
        self.verbosity = verbosity
        self.logs = []

    def log_stage(self, stage: str, content: str, metadata: Dict = None):
        """Log a pipeline stage with content visibility."""
        if self.verbosity == 0:
            return

        separator = "=" * 50
        print(f"\n[{stage}] {separator}")
        print(content[:500] if len(content) > 500 else content)

        if metadata and self.verbosity >= 2:
            print(f"\nMetadata: {metadata}")

        self.logs.append({
            "stage": stage,
            "content": content,
            "metadata": metadata,
            "timestamp": time.time()
        })

    def log_comparison(self, label1: str, text1: str, label2: str, text2: str):
        """Show side-by-side comparison."""
        print(f"\n[COMPARISON] {'=' * 50}")
        print(f"{label1}: {text1[:200]}...")
        print(f"{label2}: {text2[:200]}...")

        # Highlight key differences
        if "should" in text1.lower() and "consider" in text2.lower():
            print("\n[DIFF] ✓ Changed prescriptive to suggestive")

    def export_logs(self, filepath: str):
        """Export full logs for analysis."""
        import json
        with open(filepath, 'w') as f:
            json.dump(self.logs, f, indent=2)
```

---

## Implementation Plan V3

### Phase 1: Content Logging (2-3 hours) ⭐ START HERE

**Why first?**
- Validates current GPT-2 system
- Shows what's actually broken
- Quick win (approved approach)

**Steps:**
1. Create `ContentLogger` class
2. Add logging to evaluation pipeline
3. Add logging to training pipeline
4. Add logging to comparison engine
5. Add UI controls (verbosity slider, export button)
6. Test with current GPT-2 to see real issues

**Deliverable:** Can see actual text at every pipeline stage

---

### Phase 2: Enhanced Regex (1 hour)

**Steps:**
1. Add 5 new pattern categories
2. Test against demo examples
3. Validate 80%+ detection on loaded examples

**Deliverable:** Better fallback when AI fails

---

### Phase 3: Model Upgrade (2-3 hours)

**Option A - Single Model:**
1. Install Qwen2-1.5B-Instruct
2. Update model loading to use Qwen2
3. Test evaluation with content logging
4. Train model and validate improvement

**Option B - Dual Model (if Option A insufficient):**
1. Keep Qwen2-1.5B-Instruct for evaluation
2. Add Phi-2 for generation/training
3. Update architecture to handle two models
4. Test and validate improvement

**Deliverable:** Significantly better model(s)

---

### Phase 4: Integration Testing (1-2 hours)

**Steps:**
1. End-to-end testing with content logging
2. Validate improvements are real (not just status messages)
3. Memory profiling
4. Performance benchmarking

**Deliverable:** Confirmed working system

---

## Total Effort Estimate

- Phase 1 (Logging): 2-3 hours
- Phase 2 (Regex): 1 hour
- Phase 3 (Model upgrade): 2-3 hours
- Phase 4 (Testing): 1-2 hours

**Total: 6-9 hours**

With Option 1 (single model), can be completed in **6-7 hours**.

---

## Decision Tree

```
START
  ↓
Implement Content Logging (Phase 1)
  ↓
Test with current GPT-2
  ↓
Can we see the actual problems? → YES
  ↓
Add Enhanced Regex (Phase 2)
  ↓
Test detection rate → 80%+? → Good baseline
  ↓
Upgrade to Qwen2-1.5B-Instruct (Phase 3, Option 1)
  ↓
Measure improvement with content logging
  ↓
Improvement > 30%?
  ├─ YES → Done! Ship it.
  └─ NO → Add Phi-2 for generation (Phase 3, Option 2)
         ↓
         Re-measure → Should see 40-50% improvement
         ↓
         Done!
```

---

## Key Questions for You

**1. Memory Comfort Level?**
- Single model (Qwen2-1.5B, 3GB): Safe bet
- Dual model (Qwen2 + Phi-2, 8.4GB): Better results but higher memory
- Which fits your "couple billion parameters" experience?

**2. Implementation Priority?**
- Start with logging (validate current system)?
- OR jump straight to model upgrade?

**3. Success Criteria?**
- What improvement % would you consider successful?
- 30%? 50%?

---

## My Specific Recommendation

**Start with this sequence:**

1. **Week 1, Day 1-2:** Implement content logging (Phase 1)
   - See what GPT-2 actually outputs
   - Validate that pipeline logic is correct
   - Identify specific failure points

2. **Week 1, Day 3:** Add enhanced regex (Phase 2)
   - Quick win on obvious cases
   - Validate with logging

3. **Week 1, Day 4-5:** Upgrade to Qwen2-1.5B-Instruct (Phase 3, Option 1)
   - Single model, simple architecture
   - 12x larger than GPT-2
   - Designed for instruction-following

4. **Week 2, Day 1:** Test and measure
   - If improvement < 30%, add Phi-2
   - If improvement > 30%, done!

**Total time if successful with Option 1: ~1 week**

Ready to start with Phase 1 (content logging)?

