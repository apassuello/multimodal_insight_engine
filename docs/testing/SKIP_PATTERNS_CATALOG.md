# Skip Patterns Catalog - All 55 Instances

**Purpose**: Comprehensive catalog of every skip statement for systematic resolution
**Date**: 2025-12-18

---

## Category 1: Interface Mismatch (32 total)

### test_augmentation_pipeline.py (27 instances)

**Pattern**: Try-except with skip on exception

**Line 118**:
```python
try:
    augmented = pipeline({"image": sample_image, "text": "test caption"})
    if isinstance(augmented, tuple):
        aug_image, aug_text = augmented
    else:
        aug_image = augmented.get("image", augmented.get("pixel_values"))
    assert isinstance(aug_image, (torch.Tensor, Image.Image))
except Exception as e:
    pytest.skip(f"Pipeline has different interface: {e}")
```

**All 27 Lines with this pattern**:
- 118, 130, 156, 174, 184, 194, 218, 235, 246, 267, 279, 291, 312, 329, 350, 362, 374, 386, 398, 410, 429, 439, 458, 468, 478, 493, 514, 537

**Fix Strategy**:
1. Run test once to capture actual exception
2. Update test to match actual pipeline return format
3. Remove try-except, use direct assertions

---

### test_specialized_losses.py (5 instances)

**Lines 508, 530**: CombinedLoss interface
```python
try:
    loss = combined_loss(predictions, targets)
    assert loss is not None
except (TypeError, AttributeError):
    pytest.skip("CombinedLoss has different interface")
```

**Lines 556, 574, 587**: Loss factory interface
```python
try:
    loss = create_loss_function(args)
    assert loss is not None
except (TypeError, AttributeError):
    pytest.skip("Loss factory has different interface")
```

**Lines 627, 660**: FeatureConsistencyLoss interface
```python
try:
    loss = feature_loss(features1, features2)
    assert loss is not None
except (TypeError, AttributeError):
    pytest.skip("FeatureConsistencyLoss has different interface")
```

**Fix Strategy**:
1. Import and inspect actual function signature
2. Update test calls to match signature
3. Remove defensive try-except

---

## Category 2: Import Failures (8 total)

### test_specialized_losses.py (6 instances)

**Lines 22-51**: Conditional imports with None fallback
```python
try:
    from src.training.losses import DecorrelationLoss
except ImportError:
    DecorrelationLoss = None

try:
    from src.training.losses import MultitaskLoss
except ImportError:
    MultitaskLoss = None

try:
    from src.training.losses import CLIPLoss
except ImportError:
    CLIPLoss = None

try:
    from src.training.losses import CombinedLoss
except ImportError:
    CombinedLoss = None

try:
    from src.training.losses import create_loss_function
except ImportError:
    create_loss_function = None

try:
    from src.training.losses import FeatureConsistencyLoss
except ImportError:
    FeatureConsistencyLoss = None
```

**Lines 152, 278, 384, 487, 538, 595**: Skipif decorators
```python
@pytest.mark.skipif(DecorrelationLoss is None, reason="DecorrelationLoss not available")
@pytest.mark.skipif(MultitaskLoss is None, reason="MultitaskLoss not available")
@pytest.mark.skipif(CLIPLoss is None, reason="CLIPLoss not available")
@pytest.mark.skipif(CombinedLoss is None, reason="CombinedLoss not available")
@pytest.mark.skipif(create_loss_function is None, reason="Loss factory not available")
@pytest.mark.skipif(FeatureConsistencyLoss is None, reason="FeatureConsistencyLoss not available")
```

**Fix Strategy**:
1. Verify imports work: `python -c "from src.training.losses import DecorrelationLoss"`
2. If they work, remove conditional imports entirely
3. If they don't work, fix the import path or add missing module

---

### test_wmt_bpe_tokenizer.py (1 instance)

**Line 9**: Module-level skip
```python
try:
    from src.data.tokenization.wmt_bpe_tokenizer import WMTBPETokenizer
except ImportError:
    pytest.skip("WMTBPETokenizer not available", allow_module_level=True)
```

**Impact**: Entire test module (11 tests) skipped

**Fix Strategy**:
1. Check if file exists: `/Users/apa/ml_projects/claud_code/multimodal_insight_engine/src/data/tokenization/wmt_bpe_tokenizer.py`
2. If missing, create it or update import path
3. If present, check dependencies

---

### test_quantization.py (1 instance)

**Line 335**: torch.quantization check
```python
try:
    import torch.quantization
except ImportError:
    pytest.skip("torch.quantization not available")
```

**Fix Strategy**: Verify PyTorch version supports quantization (1.3+)

---

## Category 3: Hardware Dependencies (6 total)

### test_quantization.py (3 instances)

**Lines 187, 237, 268**: CUDA requirement
```python
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_quantized_model_cuda():
    ...
```

**Fix Strategy**: Add `@pytest.mark.cuda` marker, keep skip (valid hardware dependency)

---

### test_trainer.py (1 instance)

**Line 390**: Mixed precision CUDA check
```python
if not torch.cuda.is_available():
    pytest.skip("CUDA not available, skipping mixed precision test")
```

**Fix Strategy**: Keep skip, but add CPU-equivalent test for mixed precision API

---

### test_quantization.py (2 instances)

**Lines 304, 359, 362, 398**: Quantization runtime failures
```python
try:
    quantized_model = torch.quantization.quantize_dynamic(model, ...)
except Exception as e:
    pytest.skip(f"Quantization failed: {str(e)}")
```

**Fix Strategy**:
- Separate "quantization not supported" from "quantization failed"
- Make test more robust with proper model setup
- Keep skip only for unsupported architectures

---

## Category 4: Architecture-Specific (4 total)

### test_transformer.py (2 instances)

**Line 194**: Encoder limitation
```python
def test_encoder_with_preembedded_inputs(self):
    pytest.skip("Encoder doesn't support pre-embedded inputs explicitly")
```

**Line 224**: Decoder limitation
```python
def test_decoder_with_preembedded_inputs(self):
    pytest.skip("Decoder doesn't support pre-embedded inputs explicitly")
```

**Fix Strategy**:
- If feature is not needed, remove test
- If feature is needed, implement it and remove skip
- Document as known limitation if intentional

---

### test_security_fixes.py (2 instances)

**Line 123**: Missing file check
```python
compile_metadata_path = Path("compile_metadata.py")
if not compile_metadata_path.exists():
    pytest.skip("compile_metadata.py not found")
```

**Line 200**: Missing file check
```python
test_gpu_path = Path("setup_test/test_gpu.py")
if not test_gpu_path.exists():
    pytest.skip("setup_test/test_gpu.py not found")
```

**Fix Strategy**:
- Create missing files if needed for tests
- Or update test to use correct paths
- Or remove tests if files are obsolete

---

## Summary by File

| File | Skip Count | Type |
|------|-----------|------|
| test_augmentation_pipeline.py | 27 | Interface mismatch |
| test_specialized_losses.py | 13 | Import failures (6) + Interface (7) |
| test_quantization.py | 7 | Hardware (3) + Runtime (4) |
| test_transformer.py | 2 | Architecture limitations |
| test_security_fixes.py | 2 | Missing files |
| test_trainer.py | 1 | CUDA dependency |
| test_wmt_bpe_tokenizer.py | 1 | Module-level import |
| **TOTAL** | **55** | |

---

## Resolution Priority

### Priority 1 (Week 1): Import Failures
**Lines to fix**: test_specialized_losses.py (22-51, 152, 278, 384, 487, 538, 595)
**Expected gain**: 13 skips eliminated, +8% coverage

### Priority 2 (Week 1): Interface Mismatches
**Lines to fix**: test_augmentation_pipeline.py (all 27 instances)
**Expected gain**: 27 skips eliminated, +5% coverage

### Priority 3 (Week 2): Missing Files
**Lines to fix**: test_wmt_bpe_tokenizer.py (9), test_security_fixes.py (123, 200)
**Expected gain**: 3 skips eliminated, +2% coverage

### Priority 4 (Week 2): Hardware Tests
**Lines to fix**: Add CPU equivalents for CUDA tests
**Expected gain**: Better test coverage on all platforms

### Priority 5 (Week 3): Architecture Limitations
**Lines to fix**: test_transformer.py (194, 224)
**Expected gain**: Either implement feature or remove test

---

## Verification Commands

```bash
# Count current skips
grep -r "pytest.skip" tests/ --include="*.py" | wc -l
# Should be: 55

# After Priority 1 fixes
grep -r "pytest.skip" tests/ --include="*.py" | wc -l
# Should be: 42

# After Priority 2 fixes
grep -r "pytest.skip" tests/ --include="*.py" | wc -l
# Should be: 15

# After Priority 3 fixes
grep -r "pytest.skip" tests/ --include="*.py" | wc -l
# Should be: 12

# Run specific test file
pytest tests/test_specialized_losses.py -v --tb=short

# Run with skip reasons
pytest tests/ -v --tb=short -rs
```

---

## Code Inspection Commands

For each skip, run this to understand the actual interface:

```python
# Example: Inspect pipeline interface
from src.data.augmentation_pipeline import MultimodalAugmentationPipeline
import inspect

pipeline = MultimodalAugmentationPipeline()
print(inspect.signature(pipeline.__call__))
print(pipeline.__call__.__doc__)

# Example: Inspect loss factory
from src.training.losses.loss_factory import create_loss_function
print(inspect.signature(create_loss_function))
print(create_loss_function.__doc__)

# Example: Test actual import
try:
    from src.training.losses import DecorrelationLoss
    print(f"DecorrelationLoss imported successfully: {DecorrelationLoss}")
except ImportError as e:
    print(f"Import failed: {e}")
```

---

## Final Checklist

After fixes, verify:

- [ ] No skip statements with "interface different" (should be 0)
- [ ] No skip statements with "not available" for core modules (should be 0)
- [ ] CUDA skips have `@pytest.mark.cuda` marker
- [ ] Quantization skips are only for unsupported architectures
- [ ] All imports in test files succeed
- [ ] Overall skip count reduced from 55 to <15
