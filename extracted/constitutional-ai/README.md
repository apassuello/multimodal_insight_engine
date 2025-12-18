# Constitutional AI - Extracted Component

**Status:** Extracted from MultiModal Insight Engine
**Date:** December 2025
**Standalone Repository:** `/Users/apa/ml_projects/constitutional-ai`

---

## Purpose of This Archive

This directory contains Constitutional AI code extracted from the main MultiModal Insight Engine repository.

### Why Extracted?

- **Reusable component** suitable for multiple projects
- **Independent development** and testing cycles
- **Cleaner statistics** for MultiModal Insight Engine (LOC, coverage, complexity)
- **Clear separation** of concerns (multimodal learning vs. safety frameworks)

---

## ⚠️ Do NOT Use From This Location

**This is an archive for reference only.**

### Use the Standalone Repository Instead:

```bash
cd /Users/apa/ml_projects/constitutional-ai
```

The standalone repository has:
- ✅ Complete, functional implementation
- ✅ All improvements and bug fixes ported
- ✅ Full documentation
- ✅ Active development

---

## Contents

### Source Code (`src/`)
- **constitutional/** - Main Constitutional AI module (13 files, ~7,000 LOC)
  - `principles.py` - Core evaluation principles
  - `evaluator.py` - Safety evaluation engine
  - `filter.py` - Content filtering
  - `framework.py` - Constitutional framework
  - `critique_revision.py` - Critique and revision loop
  - `preference_comparison.py` - Preference model
  - `reward_model.py` - Reward modeling
  - `trainer.py` - RLAIF trainer
  - `ppo_trainer.py` - PPO implementation
  - `pipeline.py` - Training pipeline
  - `hf_api_evaluator.py` - HuggingFace API evaluator
  - `model_utils.py` - Model utilities
- **constitutional_trainer.py** - Training orchestration
- **constitutional_dataset.py** - Dataset handling
- **constitutional_training_config.py** - Configuration

### Tests (`tests/`)
13 comprehensive test files:
- Core: `test_principles.py`, `test_evaluator.py`, `test_filter.py`, `test_framework.py`
- Training: `test_ppo_trainer.py`, `test_preference_comparison.py`, `test_critique_revision.py`
- Models: `test_reward_model.py`, `test_model_utils.py`
- Integration: `test_cai_integration.py`, `test_cai_ml_integration.py`, `test_cai_training_integration.py`
- Utils: `test_comparison_engine.py`

### Demos (`demos/`)
- `demo_constitutional_ai.py` - Main Gradio demo
- `verify_constitutional_ai.py` - Verification script
- `train_constitutional_ai_production.py` - Production training
- `constitutional_ai_demo.py` - Simplified demo
- `constitutional_ai_real_training_demo.py` - Training demonstration
- `validate_cai_improvements.py` - Validation script

### Demo Infrastructure (`demo/managers/`)
- `evaluation_manager.py` - Evaluation orchestration
- `training_manager.py` - Training orchestration
- `comparison_engine.py` - Before/after comparison
- `model_manager.py` - Model lifecycle
- `multi_model_manager.py` - Multi-model support

### Examples (`examples/`)
- `ppo_training_example.py` - PPO training example
- `reward_model_example.py` - Reward model example
- `quick_start_demo.py` - Quick start guide

### Scripts (`scripts/`)
- `generate_constitutional_prompts.py` - Prompt generation

### Documentation (`docs/`)
- `constitutional-ai/` - Comprehensive documentation (7 files)
- `CONSTITUTIONAL_AI_IMPLEMENTATION.md` - Implementation guide

### Literature (`lit/`)
- `constitutional_ai_harmlessness_from_AI_feedback.pdf` - Anthropic research paper

---

## Statistics

### Code Volume
- **Source:** 16 files, ~7,250 LOC
- **Tests:** 13 files, ~6,500 LOC
- **Demos/Examples:** 12 files, ~8,800 LOC
- **Total:** ~22,600 LOC

### Impact on MultiModal Insight Engine
- **LOC Reduction:** 22,600+ lines
- **File Reduction:** 44 Python files
- **Coverage Focus:** Now on multimodal components
- **GitHub Stats:** Marked as vendored (excluded)

---

## Extraction Timeline

- **Phase 1 (Dec 2025):** ✅ Archived to this directory
- **Phase 2 (Q1 2026):** Planned full removal after verification period
- **Phase 3 (Q2 2026):** Delete archive, preserve git history only

---

## Migration to Standalone Repository

The extracted code has been enhanced and is fully functional in the standalone repository.

### Recent Improvements (Dec 2025)
- ✅ **+6.7% accuracy** improvement (86.7% → 93.3%)
- ✅ **1.37x performance** increase
- ✅ **Critical bug fix**: Website hacking now properly detected
- ✅ **Zero false positive** increase

See `/Users/apa/ml_projects/constitutional-ai/INVESTIGATION_FINDINGS.md` for details.

---

## License

Same as MultiModal Insight Engine (see root LICENSE file).

---

**For Questions:** This is an archived component. For active development, use the standalone repository.
