# Coverage Priority Matrix

**Generated**: 2025-12-18
**Project**: Multimodal Insight Engine
**Current Coverage**: 35%
**Files Analyzed**: 79 files with < 30% coverage

---

## Executive Summary

This priority matrix ranks all low-coverage files by **ROI Score** = (Business Impact × Risk Level) / Effort.

**Key Findings**:
- **6 Quick Wins** (ROI > 8, Effort ≤ 2): Can add ~10% coverage in 1 week
- **5 High-Value files** (ROI 5-8): Core infrastructure needing tests
- **3 Refactoring candidates**: God objects blocking testability
- **7 Archive candidates**: Dead code (0 usage)

**Recommended 4-Week Plan**: Focus on Tier 1 + Tier 2 to reach 45-50% coverage.

---

## Priority Rankings (Sorted by ROI Score)

| Rank | File Path | LOC | Coverage | Impact | Risk | Effort | Usage | ROI | Category | Recommended Action |
|------|-----------|-----|----------|--------|------|--------|-------|-----|----------|-------------------|
| 1 | `src/utils/learningrate_scheduler.py` | 356 | 0% | 5 | 4 | 1 | 5 | **20.0** | Quick Win | Pure math functions. Create 8-10 unit tests for warmup/cosine/linear schedulers. Target: 90% coverage. |
| 2 | `src/data/wmt_dataloader.py` | 114 | 0% | 4 | 3 | 1 | 3 | **12.0** | Quick Win | **Test file exists!** Fix pytest discovery or import issues. 170-line test should give 80%+ coverage immediately. |
| 3 | `src/utils/gradient_handler.py` | 469 | 0% | 5 | 4 | 2 | 5 | **10.0** | Quick Win | Mock training context. Test gradient clipping, monitoring, modality balancing. Target: 80% coverage with 12-15 tests. |
| 4 | `src/data/dataset_wrapper.py` | 165 | 0% | 3 | 3 | 1 | 3 | **9.0** | Quick Win | Small utility wrapper. Create 6-8 unit tests. Target: 85% coverage. |
| 5 | `src/training/losses/contrastive_learning.py` | 229 | 7% | 4 | 4 | 2 | 4 | **8.0** | Quick Win | Pure utility functions. Test nt_xent_loss, supervised_contrastive_loss, compute_recall_at_k. Target: 75% coverage with 10 tests. |
| 6 | `src/data/fixed_semantic_sampler.py` | 310 | 0% | 4 | 4 | 2 | 4 | **8.0** | Quick Win | Critical for VICReg. Mock PyTorch dataloader. Test semantic batch sampling logic. Target: 70% with 10-12 tests. |
| 7 | `src/models/model_factory.py` | 141 | 7% | 5 | 4 | 3 | 5 | **6.67** | High Value | Mock timm/transformers. Test all model creation branches. Consider Adapter pattern refactoring. Target: 70% with 15 tests. |
| 8 | `src/training/losses/loss_factory.py` | 207 | 27% | 5 | 5 | 4 | 5 | **6.25** | High Value | Test factory for 12+ loss types. Fix args mutation side effect. Needs Strategy pattern refactoring. Target: 75% with 20+ tests. |
| 9 | `src/utils/metrics_tracker.py` | 663 | 0% | 4 | 3 | 2 | 4 | **6.0** | High Value | Mock I/O (save_metrics, plot_metrics). Test update, should_stop_early, metric tracking. Target: 75% with 15 tests. |
| 10 | `src/training/trainers/transformer_trainer.py` | 367 | 6% | 5 | 5 | 5 | 5 | **5.0** | Refactor Needed | God object (20+ responsibilities). Needs major refactoring: extract DeviceManager, CheckpointManager, TrainerVisualizer. Then test incrementally. Target: 60% after refactoring. |
| 11 | `src/training/joint_bpe_training.py` | 131 | 0% | 3 | 3 | 2 | 3 | **4.5** | High Value | Thin wrapper around BPETokenizer. Create 4-5 unit tests for joint training. Target: 70% coverage. |
| 12 | `src/utils/argument_configs.py` | 70 | 4% | 2 | 2 | 1 | 3 | **4.0** | High Value | Pure configuration. Test get_multimodal_training_args parser. Target: 80% with 3-4 tests. |
| 13 | `src/training/flickr_multistage_training.py` | 784 | 0% | 4 | 4 | 5 | 3 | **3.2** | Refactor Needed | 784 lines, 73 branches. Needs integration tests for full pipeline + unit tests for key methods (_create_component_optimizer, _freeze_layers). Target: 40% with mixed approach. |
| 14 | `src/data/combined_wmt_translation_dataset.py` | ~250 | 0% | 3 | 3 | 4 | 3 | **2.25** | Integration Test | Requires HuggingFace datasets. Create 2-3 integration tests. Used in 3 demos. Target: 50% coverage. |
| 15 | `src/data/wmt_dataset.py` | 335 | 0% | 3 | 3 | 4 | 3 | **2.25** | Integration Test | WMT dataset loading. Integration tests with mock downloads. Target: 50% coverage with 5-6 tests. |
| 16 | `src/data/iwslt_dataset.py` | 628 | 0% | 3 | 3 | 5 | 4 | **1.8** | Integration Test | Complex fallback logic (628 lines). Integration tests with mock downloads. Used in 4 demos. Target: 40% with 8-10 tests. |
| 17 | `src/data/wikipedia_dataset.py` | 312 | 0% | 2 | 2 | 4 | 1 | **1.0** | Low Priority | TFRecord parsing, requires TensorFlow. Optional feature. Target: 30% if needed. |
| 18 | `src/utils/list_models.py` | 131 | 0% | 1 | 1 | 2 | 1 | **0.5** | Low Priority | CLI utility, not library code. Low value for unit tests. Consider manual testing. |
| 19 | `src/utils/feature_attribution.py` | 586 | 0% | 1 | 1 | 3 | 0 | **0.33** | Archive | Interpretability feature, no imports found. Archive or document as optional. |
| 20 | `src/utils/profiling.py` | 1214 | 0% | 1 | 1 | 4 | 1 | **0.25** | Low Priority | Debug tool (1214 lines). Manual testing acceptable. Very low ROI for unit tests. |
| 21 | `src/data/image_dataset.py` | 177 | 0% | 0 | 0 | - | 0 | **N/A** | **ARCHIVE** | **No imports found. Dead code. Move to archived/ directory.** |

---

## Work Tiers

### Tier 1: Quick Wins (Target: 1 week, +10% coverage)

**ROI > 8, Effort ≤ 2**

| File | LOC | Current | Estimated Tests | Target Coverage | Est. Hours |
|------|-----|---------|-----------------|-----------------|------------|
| `learningrate_scheduler.py` | 356 | 0% | 10 tests | 90% | 8h |
| `wmt_dataloader.py` | 114 | 0% | 0 (exists!) | 80% | 2h |
| `gradient_handler.py` | 469 | 0% | 15 tests | 80% | 12h |
| `dataset_wrapper.py` | 165 | 0% | 8 tests | 85% | 6h |
| `contrastive_learning.py` | 229 | 7% | 10 tests | 75% | 8h |
| `fixed_semantic_sampler.py` | 310 | 0% | 12 tests | 70% | 10h |

**Total**: 1,643 lines | **55 tests** | **Coverage gain**: +8-10% | **Time**: 46 hours (1 week)

**Key Actions**:
1. **Day 1**: Fix wmt_dataloader.py test discovery (2h) - Immediate win
2. **Day 2-3**: Test learningrate_scheduler.py (pure math, easiest)
3. **Day 4-5**: Test gradient_handler.py and dataset_wrapper.py (mock dependencies)

---

### Tier 2: High Value Core Infrastructure (Target: 2-3 weeks, +12% coverage)

**ROI 4-8, Core functionality**

| File | LOC | Current | Complexity | Target Coverage | Est. Hours |
|------|-----|---------|------------|-----------------|------------|
| `model_factory.py` | 141 | 7% | Medium | 70% | 16h |
| `loss_factory.py` | 207 | 27% | High | 75% | 24h |
| `metrics_tracker.py` | 663 | 0% | Medium | 75% | 20h |
| `joint_bpe_training.py` | 131 | 0% | Low | 70% | 8h |
| `argument_configs.py` | 70 | 4% | Low | 80% | 4h |

**Total**: 1,212 lines | **~60 tests** | **Coverage gain**: +10-12% | **Time**: 72 hours (2 weeks)

**Dependencies**:
- `loss_factory.py` needs Strategy pattern refactoring (8h) before testing
- `model_factory.py` needs timm/transformers mocking strategy

**Key Actions**:
1. **Week 2**: Start with easiest (argument_configs, joint_bpe_training)
2. **Week 3**: Tackle factories (mock external dependencies)
3. **Week 3**: Test metrics_tracker.py (mock file I/O)

---

### Tier 3: Refactoring Required (Target: 4+ weeks, +8% coverage)

**ROI 3-6, Effort > 3, Architectural changes needed**

| File | LOC | Current | Refactoring Needed | Target Coverage | Est. Hours |
|------|-----|---------|-------------------|-----------------|------------|
| `transformer_trainer.py` | 367 | 6% | Extract 3 classes | 60% | 40h (refactor: 24h, test: 16h) |
| `flickr_multistage_training.py` | 784 | 0% | Integration tests | 40% | 32h |

**Total**: 1,151 lines | **Coverage gain**: +6-8% | **Time**: 72 hours (refactoring-heavy)

**Refactoring Plan for transformer_trainer.py**:
```python
# Before: God object (367 lines, 20+ responsibilities)
class TransformerTrainer:
    __init__(17 parameters)  # Too many!
    train_epoch(145 lines)
    save_checkpoint(99 lines)
    load_checkpoint(82 lines)
    # + visualization, early stopping, device management...

# After: Separated concerns
class TransformerTrainer:
    __init__(trainer_config: TrainerConfig)  # 1 parameter
    train_epoch() -> Metrics

class DeviceManager:
    move_to_device(model, device)

class CheckpointManager:
    save(model, optimizer, path)
    load(path) -> dict

class TrainerVisualizer:
    plot_metrics(history)
```

**Recommended Approach**:
1. Extract CheckpointManager first (week 4)
2. Test CheckpointManager in isolation (week 4)
3. Extract DeviceManager (week 5)
4. Extract TrainerVisualizer (week 5)
5. Test refactored TransformerTrainer (week 6)

---

### Tier 4: Low Priority / Archive (Deferred or Archive)

**ROI < 3 or Usage = 0**

| File | LOC | Current | Category | Action |
|------|-----|---------|----------|--------|
| `image_dataset.py` | 177 | 0% | Dead Code | **ARCHIVE** - No imports found |
| `feature_attribution.py` | 586 | 0% | Unused | Archive or mark as optional |
| `profiling.py` | 1214 | 0% | Debug Tool | Manual testing only |
| `list_models.py` | 131 | 0% | CLI Utility | Manual testing acceptable |
| `wikipedia_dataset.py` | 312 | 0% | Optional | Test only if TensorFlow becomes required |
| `combined_wmt_translation_dataset.py` | ~250 | 0% | Integration | Defer to month 2 |
| `wmt_dataset.py` | 335 | 0% | Integration | Defer to month 2 |
| `iwslt_dataset.py` | 628 | 0% | Integration | Defer to month 2 |

**Recommended Actions**:
- **Immediate**: Move `image_dataset.py` to `archived/` directory
- **Document**: Mark `feature_attribution.py` as "optional interpretability feature"
- **Defer**: Integration tests for HuggingFace datasets to Month 2

---

## Impact Projections

### Current State (Baseline)
- **Total Tests**: 421 passing, 12 skipped
- **Coverage**: 35%
- **High-Risk Untested Code**: 5 critical files (training loop, factories)

### After Tier 1 (Week 1)
- **Tests Added**: +55 tests
- **Coverage**: 43-45% (+8-10%)
- **Risk Reduction**: Gradient handling, LR scheduling, data sampling now tested

### After Tier 1 + Tier 2 (Week 3)
- **Tests Added**: +115 tests total
- **Coverage**: 53-57% (+18-22%)
- **Risk Reduction**: All factories tested, metrics tracking validated

### After All Tiers (Week 6)
- **Tests Added**: +175 tests total
- **Coverage**: 59-65% (+24-30%)
- **Risk Reduction**: Training loop refactored and tested
- **Architectural Improvements**: 3 extracted classes with clear responsibilities

---

## Testing Blockers Resolution

### Priority 1: Fix Existing Tests Not Running
- **File**: `tests/test_wmt_dataloader.py` (170 lines, 0% coverage shown)
- **Issue**: Test file exists but coverage shows 0%
- **Diagnosis Needed**:
  - Check pytest discovery: `python -m pytest tests/test_wmt_dataloader.py -v`
  - Check imports: Does test file import src.data.wmt_dataloader correctly?
  - Check for skip decorators or try/except skip blocks
- **Expected Outcome**: Should immediately add 5-8% coverage when fixed

### Priority 2: Fix 27 Skipped Augmentation Tests
- **File**: `tests/test_augmentation_pipeline.py`
- **Issue**: Interface mismatch between test expectations and implementation
- **Root Cause**: Tests use `pipeline({"image": img, "text": text})` but implementation may have changed
- **Resolution**: Update test calls to match current API (already partially done in recent fixes)

### Priority 3: Document GPU-Required Tests
- **Count**: 6 tests require CUDA
- **Files**: test_quantization.py (5 tests), test_trainer.py (1 test)
- **Action**: Add clear markers and CI skip with reason: `@pytest.mark.gpu`
- **Future**: Consider CPU fallback implementations or separate GPU test suite

---

## Scoring Criteria Reference

### Business Impact (1-5)
- **5**: Core infrastructure (training loop, loss/model factories, schedulers)
- **4**: Important features (data loading, metrics, sampling)
- **3**: Supporting features (dataset wrappers, tokenizers)
- **2**: Optional features (profiling, interpretability)
- **1**: Debug tools, CLI utilities
- **0**: Dead code

### Risk Level (1-5)
- **5**: Silent failures that corrupt results (loss functions, training loop)
- **4**: High-impact bugs (wrong gradients, wrong LR, wrong sampling)
- **3**: Medium-impact bugs (data loading issues, metric errors)
- **2**: Low-impact bugs (configuration issues)
- **1**: Minimal impact (optional features)
- **0**: No risk (dead code)

### Effort (1-5)
- **1**: Trivial (test exists, pure math, < 150 lines)
- **2**: Easy (can mock, pure functions, 150-300 lines)
- **3**: Moderate (some mocking needed, 300-500 lines)
- **4**: Hard (complex mocking, integration tests, 500-800 lines)
- **5**: Very hard (refactoring required, > 800 lines, God objects)

### Usage Frequency (0-5)
- **5**: Used by 4+ files across project
- **4**: Used by 2-3 files
- **3**: Used by 1 file or in multiple demos
- **2**: Used occasionally
- **1**: Rarely used (CLI, debug)
- **0**: No imports found (dead code)

---

## Dependencies Between Items

```
Tier 1 (Parallel - No dependencies)
├─ learningrate_scheduler.py  ← Independent
├─ wmt_dataloader.py          ← Independent (fix existing test)
├─ gradient_handler.py        ← Independent
├─ dataset_wrapper.py         ← Independent
├─ contrastive_learning.py    ← Independent
└─ fixed_semantic_sampler.py  ← Independent

Tier 2 (Some dependencies)
├─ argument_configs.py        ← Independent
├─ joint_bpe_training.py      ← Independent
├─ metrics_tracker.py         ← Independent
├─ loss_factory.py            ← Needs refactoring first (8h)
└─ model_factory.py           ← Depends on mocking strategy

Tier 3 (Sequential dependencies)
├─ transformer_trainer.py
│   └─ Requires: Extract CheckpointManager → DeviceManager → TrainerVisualizer
│       └─ Then: Test each extracted class
│           └─ Finally: Test refactored trainer
└─ flickr_multistage_training.py
    └─ Depends on: transformer_trainer.py refactoring
```

**Recommended Execution Order**:
1. **Week 1**: All Tier 1 items in parallel
2. **Week 2**: Start Tier 2 (argument_configs, joint_bpe_training, metrics_tracker)
3. **Week 2 Mid**: Refactor loss_factory.py (8h), then test it
4. **Week 3**: Complete Tier 2 (model_factory.py with mocking)
5. **Week 4**: Begin Tier 3 refactoring (extract CheckpointManager)
6. **Week 5-6**: Complete transformer_trainer.py refactoring and testing

---

## Next Steps

1. **Approve this priority matrix** to proceed to Phase 4 (Actionable Roadmap)
2. **Phase 4 will generate**:
   - Detailed 4-week sprint plan with day-by-day tasks
   - Test file templates for Tier 1 quick wins
   - Refactoring design documents for Tier 3
   - CI/CD integration recommendations
   - Final consolidated report

3. **Immediate action** (can start today):
   - Investigate why `tests/test_wmt_dataloader.py` shows 0% coverage
   - Run: `python -m pytest tests/test_wmt_dataloader.py -v --tb=short`
