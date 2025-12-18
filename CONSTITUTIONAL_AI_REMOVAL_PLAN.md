# Constitutional AI Removal Plan

**Date:** 2025-12-18
**Purpose:** Remove Constitutional AI from codebase statistics after extraction to standalone repo
**Impact:** ~8,500 LOC removal

---

## Executive Summary

Constitutional AI has been extracted to standalone repository at `/Users/apa/ml_projects/constitutional-ai`. This plan removes it from the MultiModal Insight Engine codebase to clean up statistics (LOC, coverage, complexity).

**Current Impact:**
- **Source Code:** 16 Python files, ~7,254 LOC
- **Tests:** 11+ test files
- **Demos:** 6+ demo/example files
- **Documentation:** 8+ doc files
- **Total LOC Impact:** ~8,549 lines

---

## Inventory: What Will Be Removed/Archived

### 1. Source Code (16 files, ~8,549 LOC)

**Primary Constitutional AI Module:**
```
src/safety/constitutional/
├── __init__.py
├── principles.py              (~740 LOC)
├── evaluator.py              (~650 LOC)
├── filter.py                 (~400 LOC)
├── framework.py              (~850 LOC)
├── critique_revision.py      (~600 LOC)
├── preference_comparison.py  (~500 LOC)
├── reward_model.py           (~1,200 LOC)
├── trainer.py                (~900 LOC)
├── ppo_trainer.py            (~800 LOC)
├── pipeline.py               (~450 LOC)
├── hf_api_evaluator.py       (~350 LOC)
└── model_utils.py            (~314 LOC)
```

**Supporting Files:**
```
src/training/trainers/constitutional_trainer.py  (~176 LOC)
src/data/constitutional_dataset.py               (~178 LOC)
src/configs/constitutional_training_config.py    (~66 LOC)
```

### 2. Tests (11 files)

```
tests/
├── test_principles.py                (~1,500 LOC - 108 tests!)
├── test_evaluator.py                 (~800 LOC)
├── test_filter.py                    (~600 LOC)
├── test_framework.py                 (~700 LOC)
├── test_ppo_trainer.py              (~500 LOC)
├── test_preference_comparison.py     (~450 LOC)
├── test_critique_revision.py         (~400 LOC)
├── test_reward_model.py             (~650 LOC)
├── test_model_utils.py              (~300 LOC)
├── test_cai_integration.py          (~550 LOC)
├── test_cai_ml_integration.py       (~500 LOC)
├── test_cai_training_integration.py (~450 LOC)
└── test_comparison_engine.py        (~200 LOC)
```

### 3. Demos & Examples (6+ files)

**Root Level:**
```
demo_constitutional_ai.py
verify_constitutional_ai.py
train_constitutional_ai_production.py
validate_cai_improvements.py  (just created today)
```

**Demo Directory:**
```
demos/constitutional_ai_demo.py
demos/constitutional_ai_real_training_demo.py
```

**Examples:**
```
examples/ppo_training_example.py
examples/quick_start_demo.py
examples/reward_model_example.py
```

**Scripts:**
```
scripts/generate_constitutional_prompts.py
```

### 4. Documentation (8+ files)

**Dedicated Directory:**
```
docs/constitutional-ai/
├── CONSTITUTIONAL_AI_ARCHITECTURE.md
├── CONSTITUTIONAL_AI_IMPLEMENTATION_SPEC.md
├── CONSTITUTIONAL_AI_TEST_COVERAGE.md
├── HF_API_EVALUATOR_GUIDE.md
├── PPO_IMPLEMENTATION.md
├── PROMPT_GENERATION_GUIDE.md
└── REWARD_MODEL_IMPLEMENTATION_SUMMARY.md
```

**Other Docs:**
```
docs/CONSTITUTIONAL_AI_IMPLEMENTATION.md
```

**References in:**
- README.md
- CLAUDE.md
- docs/ARCHITECTURE.md
- docs/API_REFERENCE.md
- docs/USER_GUIDE.md
- Multiple audit/assessment docs

### 5. Literature

```
lit/constitutional_ai_harmlessness_from_AI_feedback.pdf
```

### 6. Demo Infrastructure Dependencies

**Files that import/use Constitutional AI:**
```
demo/managers/evaluation_manager.py
demo/managers/training_manager.py
demo/managers/comparison_engine.py
demo/data/test_examples.py
demo/main.py
```

---

## Recommended Approach: Safe Archival

### Option A: Move to Excluded Directory (RECOMMENDED) ✅

**Structure:**
```
extracted/
├── README.md  ← Explains extraction, links to new repo
├── constitutional-ai/
│   ├── src/
│   │   └── [all constitutional source]
│   ├── tests/
│   │   └── [all constitutional tests]
│   ├── demos/
│   │   └── [all demos]
│   ├── examples/
│   │   └── [all examples]
│   ├── scripts/
│   │   └── [all scripts]
│   ├── docs/
│   │   └── [all docs]
│   └── lit/
│       └── [paper]
```

**Why this approach:**
- ✅ Safe: Nothing deleted, can verify no breakage
- ✅ Statistics: Excluded from LOC/coverage
- ✅ Reference: Available if needed
- ✅ Git history: Preserved
- ✅ Reversible: Easy to truly delete later

---

## Implementation Plan

### Phase 1: Prepare Exclusion Configuration (5 min)

**Step 1.1: Update pyproject.toml coverage exclusions**
```toml
[tool.coverage.run]
omit = [
    "*/tests/*",
    "*/extracted/*",  # NEW: Exclude extracted code
    "setup.py",
]
```

**Step 1.2: Create .tokeignore for LOC counting**
```bash
cat > .tokeignore << 'EOF'
extracted/
coverage_html/
.git/
__pycache__/
*.pyc
EOF
```

**Step 1.3: Update .gitattributes (optional)**
```bash
echo "extracted/* linguist-vendored" >> .gitattributes
```

### Phase 2: Create Archive Structure (10 min)

**Step 2.1: Create directories**
```bash
mkdir -p extracted/constitutional-ai/{src,tests,demos,examples,scripts,docs,lit}
```

**Step 2.2: Create README**
```bash
cat > extracted/README.md << 'EOF'
# Extracted Components

This directory contains code that has been extracted to standalone repositories.

## Constitutional AI

**Status:** Extracted to standalone repository
**Location:** `/Users/apa/ml_projects/constitutional-ai`
**Date Extracted:** December 2025
**Reason:** Reusable component, independent development

### What Was Extracted

- Complete Constitutional AI implementation
- All tests and demos
- Full documentation
- Training infrastructure

### Why It's Here

This code remains in the repository for:
1. **Reference:** Easy comparison with original
2. **Verification:** Confirm nothing breaks
3. **Git History:** Preserved but separated
4. **Statistics:** Excluded from LOC/coverage/complexity metrics

### Using Constitutional AI

For new projects, use the standalone repository:
```bash
git clone /path/to/constitutional-ai
```

Or install as package (when published):
```bash
pip install constitutional-ai
```

### Removal Timeline

- **Phase 1 (Dec 2025):** Archived to this directory
- **Phase 2 (Q1 2026):** Full removal after verification period
- **Phase 3 (Q2 2026):** Delete archive, keep git history only
EOF
```

### Phase 3: Move Files (20 min)

**Step 3.1: Move source code**
```bash
# Constitutional AI module
mv src/safety/constitutional extracted/constitutional-ai/src/

# Supporting files
mv src/training/trainers/constitutional_trainer.py extracted/constitutional-ai/src/
mv src/data/constitutional_dataset.py extracted/constitutional-ai/src/
mv src/configs/constitutional_training_config.py extracted/constitutional-ai/src/
```

**Step 3.2: Move tests**
```bash
mv tests/test_principles.py extracted/constitutional-ai/tests/
mv tests/test_evaluator.py extracted/constitutional-ai/tests/
mv tests/test_filter.py extracted/constitutional-ai/tests/
mv tests/test_framework.py extracted/constitutional-ai/tests/
mv tests/test_ppo_trainer.py extracted/constitutional-ai/tests/
mv tests/test_preference_comparison.py extracted/constitutional-ai/tests/
mv tests/test_critique_revision.py extracted/constitutional-ai/tests/
mv tests/test_reward_model.py extracted/constitutional-ai/tests/
mv tests/test_model_utils.py extracted/constitutional-ai/tests/
mv tests/test_cai_*.py extracted/constitutional-ai/tests/
mv tests/test_comparison_engine.py extracted/constitutional-ai/tests/
```

**Step 3.3: Move demos & examples**
```bash
mv demo_constitutional_ai.py extracted/constitutional-ai/demos/
mv verify_constitutional_ai.py extracted/constitutional-ai/demos/
mv train_constitutional_ai_production.py extracted/constitutional-ai/demos/
mv validate_cai_improvements.py extracted/constitutional-ai/demos/
mv demos/constitutional_ai*.py extracted/constitutional-ai/demos/
mv examples/*ppo*.py examples/*reward*.py extracted/constitutional-ai/examples/ 2>/dev/null
mv scripts/generate_constitutional_prompts.py extracted/constitutional-ai/scripts/
```

**Step 3.4: Move documentation**
```bash
mv docs/constitutional-ai extracted/constitutional-ai/docs/
mv docs/CONSTITUTIONAL_AI_IMPLEMENTATION.md extracted/constitutional-ai/docs/
mv lit/constitutional_ai_harmlessness_from_AI_feedback.pdf extracted/constitutional-ai/lit/
```

### Phase 4: Update Import References (30 min)

**Step 4.1: Find all imports**
```bash
grep -r "from src.safety.constitutional" --include="*.py" . > /tmp/cai_imports.txt
grep -r "import src.safety.constitutional" --include="*.py" . >> /tmp/cai_imports.txt
```

**Step 4.2: Options for handling imports:**

**Option A: Remove dependencies entirely**
- Delete files that import Constitutional AI
- Update demos to reference extracted repo

**Option B: Create import shims with warnings**
```python
# src/safety/constitutional/__init__.py
import warnings
warnings.warn(
    "Constitutional AI has been extracted to standalone repository. "
    "This import will be removed in future versions. "
    "Use: from extracted.constitutional_ai import *",
    DeprecationWarning,
    stacklevel=2
)
from extracted.constitutional_ai.src import *
```

**Option C: Comment out imports with instructions**
```python
# Constitutional AI extracted - see extracted/constitutional-ai/
# from src.safety.constitutional import evaluate_harm_potential
# Use the standalone repository instead
```

### Phase 5: Update Documentation (20 min)

**Step 5.1: Update README.md**
```markdown
## Note on Constitutional AI

Constitutional AI has been extracted to a standalone repository for reusability.

**Extracted Repository:** `/Users/apa/ml_projects/constitutional-ai`
**Archived Code:** `extracted/constitutional-ai/` (for reference only)
**Reason:** Independent component suitable for multiple projects

This repository now focuses on multimodal learning and vision-language integration.
```

**Step 5.2: Update CLAUDE.md**
Remove Constitutional AI references from project priorities.

**Step 5.3: Update docs/ARCHITECTURE.md**
Remove Constitutional AI from architecture diagrams and descriptions.

**Step 5.4: Update docs/API_REFERENCE.md**
Remove or mark Constitutional AI APIs as extracted.

### Phase 6: Clean Up References (15 min)

**Step 6.1: Remove from imports in __init__.py files**
```bash
# Check what imports Constitutional AI
grep -r "constitutional" src/**/__init__.py
```

**Step 6.2: Update demo configurations**
```bash
# Update demo/main.py to not load Constitutional AI by default
```

**Step 6.3: Remove from requirements.txt** (if any specific CAI deps)

### Phase 7: Verification (20 min)

**Step 7.1: Run remaining tests**
```bash
python -m pytest tests/ -v --no-cov
```

**Step 7.2: Check coverage excludes working**
```bash
python -m pytest tests/ --cov=src --cov-report=term-missing
# Should NOT include extracted/
```

**Step 7.3: Count LOC**
```bash
# Before
find src -name "*.py" | xargs wc -l | tail -1

# After (should be ~8,500 less)
```

**Step 7.4: Verify demos work** (or document what's broken)
```bash
python run_demo.py  # Should work or show clear extraction message
```

### Phase 8: Commit Changes (10 min)

```bash
git add extracted/
git add -u  # Stage deletions
git status  # Review what's being committed

git commit -m "$(cat <<'EOF'
refactor: Extract Constitutional AI to standalone repository

Move Constitutional AI implementation to extracted/constitutional-ai/
for archival and statistics exclusion.

## Changes

### Extracted (~8,500 LOC)
- src/safety/constitutional/* (13 files)
- src/training/trainers/constitutional_trainer.py
- src/data/constitutional_dataset.py
- src/configs/constitutional_training_config.py
- 11 test files
- 6+ demo/example files
- 8+ documentation files

### Archive Location
- extracted/constitutional-ai/ (excluded from coverage/LOC stats)

### Configuration Updates
- pyproject.toml: Exclude extracted/ from coverage
- .tokeignore: Exclude from LOC counting
- .gitattributes: Mark as vendored code

### Documentation Updates
- README.md: Note extraction
- ARCHITECTURE.md: Remove Constitutional AI sections
- Added extracted/README.md with extraction details

## Rationale

Constitutional AI extracted to standalone repository for:
1. Reusability across projects
2. Independent development/testing
3. Cleaner statistics for MultiModal Insight Engine
4. Clear separation of concerns

## Standalone Repository

Location: /Users/apa/ml_projects/constitutional-ai
Status: Fully functional with improvements ported back

## Impact

- ✅ LOC reduced by ~8,500 lines
- ✅ Test coverage focused on multimodal components
- ✅ Statistics reflect true MultiModal Insight Engine scope
- ✅ Constitutional AI still available in extracted/ for reference

🤖 Generated with [Claude Code](https://claude.com/claude-code)

Co-Authored-By: Claude <noreply@anthropic.com>
EOF
)"
```

---

## Verification Checklist

After implementation, verify:

- [ ] `extracted/constitutional-ai/` exists with all files
- [ ] `src/safety/constitutional/` removed or empty shim
- [ ] Constitutional AI tests moved to `extracted/`
- [ ] Demos/examples moved to `extracted/`
- [ ] Documentation moved to `extracted/`
- [ ] Coverage excludes `extracted/` (check pyproject.toml)
- [ ] LOC count reduced by ~8,500 lines
- [ ] Remaining tests pass (or expected failures documented)
- [ ] README updated with extraction notice
- [ ] ARCHITECTURE.md updated
- [ ] Git commit created with comprehensive message

---

## Alternative: Complete Deletion

If you prefer complete deletion instead of archival:

```bash
# Delete instead of moving
rm -rf src/safety/constitutional/
rm src/training/trainers/constitutional_trainer.py
rm src/data/constitutional_dataset.py
# ... etc

git add -u
git commit -m "chore: Remove Constitutional AI (extracted to standalone repo)"
```

**Pros:**
- Cleanest approach
- No lingering code

**Cons:**
- Harder to verify nothing breaks
- Can't easily reference original
- More disruptive

---

## Rollback Plan

If issues arise:

```bash
# Restore from git
git revert <commit-hash>

# Or move files back
mv extracted/constitutional-ai/src/constitutional src/safety/
mv extracted/constitutional-ai/tests/* tests/
# ... etc
```

---

## Timeline Estimate

- **Phase 1:** 5 minutes (configuration)
- **Phase 2:** 10 minutes (create structure)
- **Phase 3:** 20 minutes (move files)
- **Phase 4:** 30 minutes (update imports)
- **Phase 5:** 20 minutes (update docs)
- **Phase 6:** 15 minutes (clean references)
- **Phase 7:** 20 minutes (verification)
- **Phase 8:** 10 minutes (commit)

**Total:** ~2 hours (with careful verification)

---

## Success Metrics

After completion:

- **LOC:** Reduced by ~8,500 lines
- **Test Coverage:** Excludes Constitutional AI code
- **Complexity:** Reduced by Constitutional AI contribution
- **Clarity:** Clear this is a MultiModal Insight Engine project
- **Usability:** Constitutional AI still usable from extracted repo

---

**Next Steps:** Review this plan, then execute phase by phase with verification at each step.
