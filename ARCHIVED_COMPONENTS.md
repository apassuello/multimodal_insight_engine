# Archived Components

This document tracks components that have been extracted from this repository to standalone projects.

---

## Constitutional AI

**Status:** ✅ Extracted
**Date:** December 18, 2025
**Reason:** Standalone, reusable component for safe AI systems

### Locations

- **Archive:** `extracted/constitutional-ai/` (reference only, excluded from statistics)
- **Standalone Repository:** `/Users/apa/ml_projects/constitutional-ai`
- **Git Branch:** `enhancement/port-cai-improvements`

### What Was Extracted

#### Source Code (16 files, ~7,250 LOC)
- Complete Constitutional AI implementation
- RLAIF (Reinforcement Learning from AI Feedback) framework
- Critique & revision loops
- Preference modeling
- Reward model training
- PPO trainer implementation

#### Tests (13 files, ~6,500 LOC)
- Comprehensive test suite with 100+ tests
- Integration tests
- Performance benchmarks

#### Demo Infrastructure (12 files, ~8,800 LOC)
- Gradio web interface
- Model managers
- Training orchestration
- Evaluation engines
- Before/after comparison tools

#### Documentation
- 8 comprehensive markdown files
- Research paper (Anthropic)
- Implementation guides
- API reference

### Statistics Impact

#### Before Extraction
- **Total LOC:** ~75,400
- **Constitutional AI:** ~22,600 LOC (30% of codebase)
- **Test Coverage:** Mixed (multimodal + constitutional)

#### After Extraction
- **Total LOC:** ~52,800 (30% reduction)
- **Coverage Focus:** Multimodal learning only
- **GitHub Stats:** Constitutional AI marked as vendored (excluded)
- **Clarity:** Clear focus on multimodal vision-language models

### Migration Guide

#### For Users

**DO NOT import from archived location:**
```python
# ❌ Wrong - This won't work
from src.safety.constitutional import ConstitutionalFramework
```

**Use the standalone repository:**
```bash
cd /Users/apa/ml_projects/constitutional-ai

# Or install as package (when published)
pip install constitutional-ai
```

#### For Developers

**Standalone repository benefits:**
- Independent versioning
- Faster CI/CD (no multimodal dependencies)
- Clearer contribution guidelines
- Reusable across projects

**Location:**
```bash
/Users/apa/ml_projects/constitutional-ai
```

### Improvements Made During Extraction

The standalone repository includes recent enhancements:

#### Performance (Dec 2025)
- **Accuracy:** +6.7% improvement (86.7% → 93.3%)
- **Speed:** 1.37x faster
- **False Positives:** 0% (no increase)

#### Bug Fixes (Dec 2025)
- ✅ Fixed: "How to hack a website" now properly detected
- ✅ Enhanced: Cyber vs. physical threat differentiation
- ✅ Improved: Normalized consequence messages

See investigation findings:
```
/Users/apa/ml_projects/constitutional-ai/INVESTIGATION_FINDINGS.md
/Users/apa/ml_projects/constitutional-ai/MIGRATION_PLAN_PORT_TO_ORIGINAL.md
```

### Removal Timeline

#### Phase 1: Archival (Dec 2025) ✅ Complete
- Moved to `extracted/constitutional-ai/`
- Updated statistics exclusions
- Removed from main codebase imports
- Created standalone repository

#### Phase 2: Verification (Q1 2026)
- Monitor for issues
- Verify standalone repository stability
- Update any remaining references

#### Phase 3: Full Removal (Q2 2026)
- Delete `extracted/constitutional-ai/` archive
- Preserve git history
- Update documentation to point only to standalone repo

---

## Dead Code (Removed from Coverage)

**Status:** 🗑️ Archived (Unused Code)
**Date:** December 18, 2025
**Reason:** Zero imports found in codebase, superseded by other implementations, or redundant

### Files Archived

#### 1. `archived/utils/feature_attribution.py` (585 lines)
- **Why removed:** Interpretability/explainability code that was never used
- **Contains:** GradCAM, IntegratedGradients, SaliencyMap, AttributionVisualizer classes
- **Imports found:** 0 (completely unused)
- **Recommendation:** If interpretability is needed in future, consider modern libraries like Captum

#### 2. `archived/models/activations.py` (81 lines)
- **Why removed:** Redundant wrapper around PyTorch's built-in GELU
- **Contains:** GELU activation class
- **Imports found:** 0 (completely unused)
- **Recommendation:** Use `torch.nn.functional.gelu()` directly

#### 3. `archived/data/augmentation.py` (88 lines)
- **Why removed:** Superseded by `src/data/augmentation_pipeline.py` (759 lines)
- **Contains:** Early/simple version of multimodal augmentation
- **Imports found:** 0 (completely unused)
- **Recommendation:** Use `MultimodalAugmentationPipeline` from augmentation_pipeline.py

#### 4. `archived/data/image_dataset.py` (177 lines)
- **Why removed:** No imports found anywhere
- **Contains:** Image dataset loader
- **Imports found:** 0 (completely unused)
- **Recommendation:** Use modern dataset loaders from `src/data/multimodal_dataset.py`

#### 5. `archived/evaluation/translation_metrics.py` (146 lines)
- **Why removed:** Functions (`calculate_bleu`, `calculate_ter`) are redefined in demos, never imported
- **Contains:** Translation evaluation metrics
- **Imports found:** 0 (referenced but never imported)
- **Recommendation:** Demo scripts have their own implementations

### Impact

**Coverage Cleanup:**
- **Total dead code lines:** 1,077 lines
- **Files archived:** 5 files
- **Coverage impact:** Reduces false-positive 0% coverage by ~24-33%
- **Benefit:** Clearer picture of actual coverage gaps vs. dead code

**Before Archival:**
- Files with 0% coverage: 24 files
- Dead code contributing to 0% metric: 5 files (20% of 0% files)

**After Archival:**
- Active files with 0% coverage: 19 files (legitimate gaps needing tests)
- Dead code removed from statistics

### Discovery Process

Dead code identified through systematic import analysis:
1. Searched for `from src.path.module import` patterns across all Python files
2. Checked test files, demos, and documentation
3. Verified __init__.py references
4. Categorized as DEAD_CODE only if zero imports found

**Tool:** AI agent-powered codebase exploration (Explore subagent)

---

## Future Extractions

This section will track any future components extracted to standalone repositories.

### Criteria for Extraction

A component is a good candidate for extraction if:
- ✅ **Reusable:** Applicable to multiple projects
- ✅ **Independent:** Minimal dependencies on core codebase
- ✅ **Self-contained:** Complete functionality in isolation
- ✅ **Mature:** Stable API and well-tested
- ✅ **Separable:** Clear boundaries from other components

### Process

1. Create standalone repository
2. Copy code to standalone repo
3. Enhance/fix issues
4. Port improvements back to original
5. Move to `extracted/` in original repo
6. Exclude from statistics
7. Update all references
8. Verify stability
9. Remove archive after verification period

---

## Configuration

### Coverage Exclusion
```toml
# pyproject.toml
[tool.coverage.run]
omit = [
    "extracted/*",  # Archived components
]
```

### GitHub Statistics
```.gitattributes
extracted/* linguist-vendored
extracted/* linguist-documentation=false
```

### LOC Counting
```.tokeignore
extracted/
```

---

## Questions?

For questions about:
- **Extracted components:** See standalone repositories
- **Extraction process:** Contact maintainers
- **Using archived code:** Don't - use standalone repos instead

---

**Last Updated:** December 18, 2025
