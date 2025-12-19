# C901 Complexity Assessment

**Total Functions with Complexity > 10:** 144

## Summary

The codebase has widespread complexity issues that indicate a need for systematic refactoring. While some complexity is inherent in ML code, many functions can be decomposed for better testability, maintainability, and reliability.

---

## Priority 1: Critical - Complexity ≥ 30 (5 functions)

These functions are **extremely difficult to test and maintain**. They should be refactored immediately.

### 🚨 `src/models/model_factory.py::create_multimodal_model` (46)
- **Issue**: Factory function with 46 branches is essentially unmaintainable
- **Impact**: High - core model creation logic
- **Refactoring Strategy**:
  - Extract builder classes for each model type
  - Use strategy pattern or registry pattern
  - Create separate factory functions per model family

### 🚨 `src/training/losses/loss_factory.py::create_loss_function` (39)
- **Issue**: Another factory with too many branches
- **Impact**: High - core loss creation logic
- **Refactoring Strategy**:
  - Loss registry pattern
  - Builder classes for complex loss configurations
  - Separate validation from construction

### 🚨 `src/data/multimodal_dataset.py::EnhancedMultimodalDataset.__init__` (39)
- **Issue**: Constructor doing too much initialization logic
- **Impact**: High - core dataset class
- **Refactoring Strategy**:
  - Extract initialization into smaller helper methods
  - Use builder pattern for complex configurations
  - Separate validation, path setup, and data loading phases

### 🚨 `src/models/multimodal/multimodal_integration.py::forward` (33)
- **Issue**: Forward pass with too many conditional paths
- **Impact**: High - core model forward pass
- **Refactoring Strategy**:
  - Extract feature extraction to separate methods
  - Use composition for different forward pass strategies
  - Reduce conditional logic with polymorphism

### 🚨 `src/training/losses/feature_consistency_loss.py::forward` (33)
- **Issue**: Loss computation with excessive branching
- **Impact**: Medium-High - affects training stability
- **Refactoring Strategy**:
  - Decompose into feature extraction + similarity computation + loss aggregation
  - Extract conditional blocks into named helper methods

---

## Priority 2: High - Complexity 20-29 (9 functions)

These functions are **very difficult to understand and test**. Should be refactored soon.

### `src/optimization/benchmarking.py::_generate_recommendations` (29)
- **Refactoring**: Extract recommendation generators for each category

### `src/training/losses/decorrelation_loss.py::forward` (27)
- **Refactoring**: Extract correlation computation, normalization, and loss calculation

### `src/models/multimodal/vicreg_multimodal_model.py::forward` (27)
- **Refactoring**: Extract variance/invariance/covariance calculations

### `src/safety/red_teaming/model_loader.py::_load_huggingface_model` (25)
- **Refactoring**: Extract model type detection, configuration, and loading phases

### `src/training/trainers/multimodal/training_loop.py::_analyze_gradients` (23)
- **Refactoring**: Extract per-component analysis into separate methods

### `src/data/tokenization/optimized_bpe_tokenizer.py::train` (22)
- **Refactoring**: Extract corpus processing, merge operations, and vocabulary building

### `src/models/multimodal/multimodal_decoder_generation.py::generate` (22)
- **Refactoring**: Extract beam search, sampling, and decoding strategies

### `src/training/trainers/transformer_trainer.py::restore_from_checkpoint` (21)
- **Refactoring**: Extract state restoration for model, optimizer, scheduler separately

### `src/training/trainers/trainer.py::train_model` (21)
- **Refactoring**: Extract epoch logic, validation logic, checkpointing into methods

---

## Priority 3: Medium - Complexity 15-19 (15 functions)

These should be refactored during normal maintenance or when modifying the code.

- `src/data/multimodal_dataset.py::_load_flickr30k` (20)
- `src/utils/gradient_handler.py::analyze_gradients` (20)
- `src/training/trainers/transformer_trainer.py::load_checkpoint` (19)
- `src/models/pretrained/huggingface_wrapper.py::encode` (21)
- `src/data/tokenization/optimized_bpe_tokenizer.py::_tokenize_word_optimized` (18)
- `src/data/wmt_dataset.py::load_data` (18)
- `src/models/multimodal/multimodal_integration.py::extract_text_features` (18)
- `src/training/losses/contrastive_learning.py::forward` (18)
- `src/optimization/quantization.py::_fuse_modules` (18)
- `src/data/multimodal_dataset.py::__getitem__` (17)
- `src/data/tokenization/optimized_bpe_tokenizer.py::_process_batch` (17)
- `src/models/multimodal/vicreg_multimodal_model.py::_extract_features` (17)
- `src/safety/harness.py::evaluate_model` (17)
- `src/optimization/quantization.py::_fuse_modules` (17) [duplicate entry?]
- `src/text_generation.py::generate` (16)

---

## Priority 4: Low - Complexity 11-14 (115 functions)

These are moderately complex but manageable. Address opportunistically.

**Common patterns:**
- Dataset loading functions (11-14 complexity)
- Metric calculation functions (12-15 complexity)
- Batch preparation logic (11-12 complexity)
- Initialization methods (13-15 complexity)

**Recommendation**:
- When modifying these functions, look for opportunities to extract helper methods
- Apply single responsibility principle
- Don't suppress warnings - fix incrementally

---

## Refactoring Strategy Recommendations

### 1. **Factory Pattern Refactoring** (Highest ROI)
Both `create_multimodal_model` (46) and `create_loss_function` (39) are factories that should use:
```python
# Before: Giant if/elif chain
if model_type == "clip":
    # 50 lines of CLIP setup
elif model_type == "vicreg":
    # 50 lines of VICReg setup
# ... 10 more types

# After: Registry pattern
class ModelRegistry:
    _builders = {}

    @classmethod
    def register(cls, name):
        def decorator(builder_fn):
            cls._builders[name] = builder_fn
            return builder_fn
        return decorator

    @classmethod
    def create(cls, model_type, **kwargs):
        builder = cls._builders.get(model_type)
        if not builder:
            raise ValueError(f"Unknown model type: {model_type}")
        return builder(**kwargs)

@ModelRegistry.register("clip")
def build_clip_model(**kwargs):
    # Focused CLIP building logic
    ...

# Usage
model = ModelRegistry.create(model_type, **config)
```

### 2. **Extract Method Refactoring** (Medium ROI)
For complex `__init__`, `forward`, and `load_data` methods:
```python
# Before: __init__ with 39 complexity
def __init__(self, ...):
    # Validation (5 branches)
    # Path setup (8 branches)
    # Config loading (12 branches)
    # Data loading (10 branches)
    # Preprocessing setup (4 branches)

# After: Decomposed
def __init__(self, ...):
    self._validate_config()
    self._setup_paths()
    self._load_config()
    self._load_data()
    self._setup_preprocessing()

def _validate_config(self):
    # 5 branches, but focused and testable

def _setup_paths(self):
    # 8 branches, but focused and testable
```

### 3. **Strategy Pattern** (for forward passes with many modes)
```python
# Before: forward with 33 complexity
def forward(self, x, mode="train", use_cache=False, ...):
    if mode == "train":
        if use_cache:
            # ...
        else:
            # ...
    elif mode == "eval":
        # ...

# After: Strategy pattern
class TrainingForwardStrategy:
    def __call__(self, model, x, use_cache=False):
        # Focused training logic

class EvalForwardStrategy:
    def __call__(self, model, x):
        # Focused eval logic

class MultimodalModel:
    def __init__(self):
        self._strategies = {
            "train": TrainingForwardStrategy(),
            "eval": EvalForwardStrategy(),
        }

    def forward(self, x, mode="train", **kwargs):
        strategy = self._strategies[mode]
        return strategy(self, x, **kwargs)
```

---

## Impact Analysis

### Testing Impact
- **Critical (≥30 complexity)**: Likely untestable or tests have low coverage
- **High (20-29)**: Tests probably miss edge cases
- **Medium (15-19)**: Tests exist but may be brittle
- **Low (11-14)**: Tests probably adequate

### Maintenance Impact
- **Critical functions**: Changes are risky and time-consuming
- **High functions**: Require significant cognitive load to modify
- **Medium functions**: Can be modified with care
- **Low functions**: Normal maintenance difficulty

### Bug Density
Functions with complexity > 20 statistically have 2-3x more bugs than simpler functions.

---

## Recommended Approach

### Phase 1: Address Critical Issues (Week 1-2)
1. Refactor `create_multimodal_model` (46) using registry pattern
2. Refactor `create_loss_function` (39) using registry pattern
3. Decompose `EnhancedMultimodalDataset.__init__` (39)

### Phase 2: High Priority (Week 3-4)
4. Refactor complex forward passes (33, 27, 27)
5. Decompose training loop complexity (23, 22, 21)

### Phase 3: Medium Priority (Ongoing)
6. Address complexity 15-19 during normal maintenance
7. Extract methods opportunistically

### Phase 4: Continuous Improvement
8. Set C901 limit to 15 (not 10, which may be too strict for ML)
9. Require refactoring before adding to functions > 15
10. Regular complexity audits

---

## Should We Ignore C901?

### ❌ **NO** - Don't blanket ignore C901

**Reasoning:**
1. **Technical Debt**: Currently have 144 complex functions - this is unsustainable
2. **Bug Density**: Complex functions have higher bug rates (we've seen this in the test failures)
3. **Maintainability**: Hard to onboard new developers when core functions are 30-46 complexity
4. **Testability**: Functions with 30+ complexity are nearly impossible to test thoroughly

### ✅ **Better Approach:**

1. **Remove C901 from global ignore list**
2. **Set a higher threshold** (complexity 15 instead of 10):
```toml
[tool.ruff.lint.mccabe]
max-complexity = 15  # More reasonable for ML code
```

3. **Suppress only the ones we'll fix later**:
```python
def create_multimodal_model(...):  # noqa: C901 - TODO: Refactor to registry pattern (Issue #XX)
    # Complexity 46 - scheduled for refactoring
```

4. **Track refactoring in issues**:
   - Create issues for complexity > 20
   - Schedule systematic refactoring
   - Don't just hide the problem

---

## Conclusion

**Current State:** 144 functions exceeding complexity threshold indicates systemic design issues.

**Recommendation:**
- Do NOT ignore C901 globally
- Raise threshold to 15 (from 10)
- Systematically refactor the worst offenders (>= 20)
- Use `# noqa: C901` with TODO comments only as temporary measure
- Track technical debt explicitly

**Estimated Effort:**
- Phase 1 (Critical): ~40 hours
- Phase 2 (High): ~60 hours
- Phase 3 (Medium): ~40 hours (spread over time)
- **Total**: ~140 hours of refactoring work

**Benefits:**
- Improved testability (easier to hit 80%+ coverage)
- Reduced bug density
- Better maintainability
- Easier onboarding for new developers
- More reliable codebase
