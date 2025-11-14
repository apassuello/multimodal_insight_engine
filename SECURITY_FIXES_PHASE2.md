# Security Fixes - Phase 2 Quick Reference

This document provides actionable code fixes for security issues identified in the Phase 2 audit.

---

## CRITICAL: Fix Immediately

### CRIT-01: Remove Full Traceback Exposure

**File:** `demo/main.py` line 587-591

**Current Code:**
```python
except Exception as e:
    error_msg = f"✗ Comparison failed: {str(e)}"
    import traceback
    error_msg += f"\n\nTraceback:\n{traceback.format_exc()}"
    return error_msg, "", ""
```

**Fixed Code:**
```python
except Exception as e:
    # Log full traceback server-side for debugging
    import logging
    import traceback
    logger = logging.getLogger(__name__)
    logger.error(f"Comparison failed: {str(e)}", exc_info=True)

    # Return user-friendly message without sensitive details
    error_msg = "✗ Comparison failed due to an internal error. Please check your model configuration and try again."
    return error_msg, "", ""
```

---

## HIGH PRIORITY: Fix Before Production

### HIGH-01: Add Test Suite Size Limits

**File:** `demo/managers/comparison_engine.py` line 80-91

**Add constants at top of file:**
```python
# Security limits
MAX_TEST_SUITE_SIZE = 100
GENERATION_TIMEOUT_SECONDS = 30
```

**Add validation in compare_models():**
```python
def compare_models(
    self,
    base_model,
    base_tokenizer,
    trained_model,
    trained_tokenizer,
    test_suite: List[str],
    device: torch.device,
    generation_config: GenerationConfig,
    test_suite_name: str = "Test Suite",
    progress_callback: Optional[Callable[[int, int, str], None]] = None
) -> ComparisonResult:
    """Compare base and trained models on test suite."""

    # SECURITY: Validate test suite size
    if len(test_suite) > MAX_TEST_SUITE_SIZE:
        raise ValueError(
            f"Test suite too large ({len(test_suite)} prompts). "
            f"Maximum allowed: {MAX_TEST_SUITE_SIZE}"
        )

    result = ComparisonResult(
        test_suite_name=test_suite_name,
        num_prompts=len(test_suite)
    )
    # ... rest of implementation
```

---

### HIGH-02: Validate Generation Parameters

**File:** `src/safety/constitutional/model_utils.py` line 16-27

**Current Code:**
```python
@dataclass
class GenerationConfig:
    """Configuration for text generation."""
    max_length: int = 100
    temperature: float = 1.0
    top_p: float = 0.9
    top_k: int = 50
    num_return_sequences: int = 1
    do_sample: bool = True
    pad_token_id: Optional[int] = None
    eos_token_id: Optional[int] = None
```

**Fixed Code:**
```python
@dataclass
class GenerationConfig:
    """Configuration for text generation."""
    max_length: int = 100
    temperature: float = 1.0
    top_p: float = 0.9
    top_k: int = 50
    num_return_sequences: int = 1
    do_sample: bool = True
    pad_token_id: Optional[int] = None
    eos_token_id: Optional[int] = None

    def __post_init__(self):
        """Validate parameters after initialization."""
        if not 0.01 <= self.temperature <= 5.0:
            raise ValueError(
                f"Temperature must be in [0.01, 5.0], got {self.temperature}"
            )
        if not 1 <= self.max_length <= 2048:
            raise ValueError(
                f"max_length must be in [1, 2048], got {self.max_length}"
            )
        if not 0.0 <= self.top_p <= 1.0:
            raise ValueError(
                f"top_p must be in [0.0, 1.0], got {self.top_p}"
            )
        if self.top_k < 0:
            raise ValueError(
                f"top_k must be non-negative, got {self.top_k}"
            )
        if self.num_return_sequences < 1:
            raise ValueError(
                f"num_return_sequences must be positive, got {self.num_return_sequences}"
            )
```

---

### HIGH-03: Sanitize Error Messages

**File:** `demo/managers/comparison_engine.py` line 205-217

**Add at top of file:**
```python
import logging

logger = logging.getLogger(__name__)
```

**Current Code:**
```python
except Exception as e:
    # Log error and continue
    error_msg = f"Error processing prompt {idx + 1}: {str(e)}"
    result.errors.append(error_msg)
    result.skipped_prompts += 1

    # Report error in progress
    if progress_callback:
        progress_callback(
            idx + 1,
            len(test_suite),
            f"⚠ Error on prompt {idx + 1}: {str(e)[:50]}"
        )

    continue
```

**Fixed Code:**
```python
except Exception as e:
    # Log detailed error server-side
    logger.error(
        f"Error processing prompt {idx + 1}: {type(e).__name__}: {str(e)}"
    )

    # Store generic error for user display
    error_msg = f"Error processing prompt {idx + 1}: Generation failed"
    result.errors.append(error_msg)
    result.skipped_prompts += 1

    # Report error in progress
    if progress_callback:
        progress_callback(
            idx + 1,
            len(test_suite),
            f"⚠ Error on prompt {idx + 1}: Generation failed"
        )

    continue
```

---

## MEDIUM PRIORITY: Should Address

### MED-01: Add Checkpoint Path Validation

**File:** `demo/managers/model_manager.py` line 210-244

**Current Code:**
```python
def load_checkpoint(
    self,
    checkpoint_path: Path,
    device: Optional[torch.device] = None
) -> Tuple[Optional[Any], Optional[Any], bool, str]:
    """Load model and tokenizer from checkpoint."""
    try:
        if not checkpoint_path.exists():
            return None, None, False, f"✗ Checkpoint not found: {checkpoint_path}"
        # ... rest of implementation
```

**Fixed Code:**
```python
def load_checkpoint(
    self,
    checkpoint_path: Path,
    device: Optional[torch.device] = None
) -> Tuple[Optional[Any], Optional[Any], bool, str]:
    """Load model and tokenizer from checkpoint."""
    try:
        # SECURITY: Validate path stays within checkpoint directory
        try:
            resolved_path = checkpoint_path.resolve()
            checkpoint_base = self.checkpoint_dir.resolve()

            # Check if resolved path is within checkpoint directory
            if not str(resolved_path).startswith(str(checkpoint_base)):
                return None, None, False, "✗ Invalid checkpoint path: outside checkpoint directory"
        except Exception:
            return None, None, False, "✗ Invalid checkpoint path"

        if not checkpoint_path.exists():
            return None, None, False, "✗ Checkpoint not found"

        # ... rest of implementation
```

---

### MED-03: Add Rate Limiting (For Multi-User Deployments)

**File:** `demo/main.py` at top of file

**Add at top:**
```python
from threading import Lock

# Global lock for comparison operations
comparison_lock = Lock()
```

**Modify run_comparison_handler:**
```python
def run_comparison_handler(
    test_suite_name: str,
    temperature: float,
    max_length: int,
    progress=gr.Progress()
) -> Tuple[str, str, str]:
    """Run comparison between base and trained models on selected test suite."""

    # SECURITY: Prevent concurrent comparisons
    if not comparison_lock.acquire(blocking=False):
        return "✗ Comparison already in progress. Please wait.", "", ""

    try:
        if not model_manager.can_compare():
            error_msg = "✗ Cannot run comparison: Need both base and trained model checkpoints.\n"
            error_msg += "Please train a model first in the Training tab."
            return error_msg, "", ""

        # ... existing implementation ...

    finally:
        comparison_lock.release()
```

---

## LOW PRIORITY: Code Quality Improvements

### LOW-01: Fix Silent Exception Handling

**File:** `demo/main.py` line 574-580

**Current Code:**
```python
try:
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    if hasattr(torch, 'mps') and hasattr(torch.mps, 'empty_cache'):
        torch.mps.empty_cache()
except:
    pass
```

**Fixed Code:**
```python
import logging
logger = logging.getLogger(__name__)

try:
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    if hasattr(torch, 'mps') and hasattr(torch.mps, 'empty_cache'):
        torch.mps.empty_cache()
except Exception as e:
    # Log but don't fail on cleanup errors
    logger.debug(f"Failed to clear GPU cache: {e}")
```

---

### LOW-02: Add Structured Logging

**File:** Create new file `demo/logging_config.py`

```python
"""Logging configuration for Constitutional AI demo."""

import logging
import sys
from pathlib import Path

def setup_logging(log_level: str = "INFO", log_file: str = "demo/logs/app.log"):
    """
    Configure structured logging for the application.

    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Path to log file
    """
    # Create logs directory
    log_path = Path(log_file)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    # Configure logging
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )

    logger = logging.getLogger(__name__)
    logger.info(f"Logging configured: level={log_level}, file={log_file}")
```

**File:** `demo/main.py` at top

```python
from demo.logging_config import setup_logging

# Setup logging at application start
setup_logging(log_level="INFO")
logger = logging.getLogger(__name__)
```

---

## Testing Security Fixes

### Test CRIT-01: Error Message Sanitization

```python
def test_error_message_sanitization():
    """Test that error messages don't leak sensitive information."""
    # Trigger an error condition (e.g., invalid checkpoint)
    # Verify:
    # - Error message is user-friendly
    # - No file paths exposed
    # - No stack traces in UI
    # - Full details logged server-side
```

### Test HIGH-01: Test Suite Size Limits

```python
def test_test_suite_size_limits():
    """Test that large test suites are rejected."""
    # Create test suite with 150 prompts
    large_suite = ["test prompt"] * 150

    # Attempt comparison
    # Verify: ValueError raised with appropriate message
```

### Test HIGH-02: Parameter Validation

```python
def test_generation_config_validation():
    """Test that invalid generation parameters are rejected."""
    # Test invalid temperature
    with pytest.raises(ValueError):
        GenerationConfig(temperature=-1.0)

    with pytest.raises(ValueError):
        GenerationConfig(temperature=10.0)

    # Test invalid max_length
    with pytest.raises(ValueError):
        GenerationConfig(max_length=0)

    with pytest.raises(ValueError):
        GenerationConfig(max_length=10000)
```

### Test MED-01: Path Traversal Protection

```python
def test_checkpoint_path_validation():
    """Test that path traversal is prevented."""
    from pathlib import Path

    manager = ModelManager()

    # Attempt path traversal
    malicious_path = Path("../../../etc/passwd")

    model, tokenizer, success, msg = manager.load_checkpoint(malicious_path)

    # Verify:
    # - success is False
    # - msg indicates invalid path
    # - No file outside checkpoint_dir was accessed
```

---

## Deployment Checklist

Before deploying Phase 2:

- [ ] **CRIT-01:** Remove full traceback exposure
- [ ] **HIGH-01:** Add test suite size limits
- [ ] **HIGH-02:** Add generation parameter validation
- [ ] **HIGH-03:** Sanitize error messages
- [ ] **MED-01:** Add checkpoint path validation
- [ ] Add structured logging
- [ ] Document security limitations in README
- [ ] Add warning about adversarial content in UI
- [ ] Test all security fixes
- [ ] Review deployment context (local vs multi-user)

For multi-user deployments, additionally:

- [ ] **MED-03:** Add rate limiting/concurrency control
- [ ] Implement authentication/authorization
- [ ] Set up monitoring and alerting
- [ ] Deploy in containerized environment
- [ ] Configure reverse proxy with security headers
- [ ] Set up audit logging

---

## Quick Command Reference

```bash
# Run security tests
python -m pytest tests/test_security.py -v

# Check for vulnerable dependencies
pip-audit

# Run static security analysis
bandit -r src/ demo/ -ll

# Check for secrets in code
trufflehog filesystem .

# Lint security issues
flake8 src/ demo/ --select=S

# Type check (helps catch security issues)
mypy src/ demo/
```

---

## Security Contact

For security concerns or to report vulnerabilities:
- Create a private security advisory on GitHub
- Do not publicly disclose vulnerabilities
- Allow 90 days for fixes before disclosure

---

**Document Version:** 1.0
**Last Updated:** 2025-11-14
**Next Review:** After fixes implemented
