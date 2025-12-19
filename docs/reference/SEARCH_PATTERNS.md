# Search Patterns for Test Fixes

Quick reference for finding and fixing issues in the codebase.

---

## Priority 1: Logger Initialization (91 tests)

### Find logger usage without initialization
```bash
# Find all logger.info/debug/warning calls
grep -r "logger\.\(info\|debug\|warning\|error\)" src/ --include="*.py"

# Find logger assignments
grep -r "logger\s*=" src/ --include="*.py"

# Check if logger is imported but not initialized
grep -r "import.*logging" src/ --include="*.py" | head -20
```

### Check test fixtures
```bash
# Find logger fixtures in tests
grep -r "@pytest.fixture.*logger" tests/ --include="*.py"

# Check conftest.py for logger setup
cat tests/conftest.py | grep -A10 -B2 "logger"

# Find logger initialization in CAI modules
grep -r "logging.getLogger" src/cai/ --include="*.py"
```

### Expected pattern (correct)
```python
import logging

logger = logging.getLogger(__name__)
# NOT: logger = None
# NOT: logger = logging.getLogger() without assignment check
```

### Common bugs
```python
# Bug 1: Logger not initialized
logger = None  # Later code calls logger.info() → AttributeError

# Bug 2: Conditional initialization
if some_condition:
    logger = logging.getLogger(__name__)
# If condition false, logger undefined

# Bug 3: Delayed initialization
class MyClass:
    def __init__(self):
        pass  # logger not set

    def method(self):
        self.logger.info("...")  # AttributeError if logger not set
```

---

## Priority 2: Training Data 'chosen' Key (12 tests)

### Find reward model data loading
```bash
# Find where reward model expects 'chosen' key
grep -r "chosen" src/ --include="*.py" | grep -v ".pyc"

# Find reward model training data creation
grep -r "def.*reward.*data" tests/ --include="*.py"

# Find preference data fixtures
grep -r "preference.*data" tests/ --include="*.py"
```

### Expected data format
```python
# Correct format for reward model training
training_data = [
    {
        'prompt': 'What is 2+2?',
        'chosen': '2+2 equals 4',      # Preferred response
        'rejected': '2+2 equals 5'      # Rejected response
    },
    # ... more examples
]
```

### Find test data creation
```bash
# Check fixtures
grep -r "@pytest.fixture" tests/test_reward_model.py -A20

# Find data generation in conftest
grep -r "chosen\|rejected\|preference" tests/conftest.py
```

---

## Priority 3: Training Data Validation (5 tests)

### Find validation logic
```bash
# Find where "invalid" error message comes from
grep -r "training examples are invalid" src/ --include="*.py"

# Find validation functions
grep -r "def.*validate.*data" src/ --include="*.py"

# Find Phase 1 training data creation
grep -r "phase.*1\|phase1" src/training*.py --include="*.py"
```

### Check validation criteria
```bash
# Find what makes an example "valid"
grep -r "is_valid\|validate.*example" src/ --include="*.py" -A5
```

---

## Priority 4A: Logger API 'end' Parameter (6 tests)

### Find incorrect logger calls
```bash
# Find logger calls with 'end' parameter
grep -r "logger.*end\s*=" src/ --include="*.py"

# More specific pattern
grep -rE "logger\.(info|debug|warning|error).*end\s*=" src/ --include="*.py"

# Check training.py specifically
grep -n "logger.*end=" src/training.py
```

### Expected issues
```python
# WRONG - logger doesn't accept 'end' parameter
logger.info("Training epoch", end='')
logger.info(f"Loss: {loss}", end='\r')

# CORRECT - use print for progress indicators
print("Training epoch", end='')
# or
logger.info("Training epoch")
```

---

## Priority 4B: Logger API 'level' Parameter (4 tests)

### Find Logger._log misuse
```bash
# Find direct _log calls
grep -r "Logger\._log\|logger\._log" src/ --include="*.py"

# Find logging with explicit level parameter issues
grep -r "\.log(.*level" src/ --include="*.py"

# Check principles.py specifically
grep -n "_log\|\.log(.*level" src/principles.py
```

### Expected issues
```python
# WRONG - conflicting level parameters
logger._log(logging.INFO, msg, level=logging.DEBUG)
Logger._log(self, logging.INFO, msg)  # level passed twice

# CORRECT - use specific methods
logger.info(msg)
logger.debug(msg)
logger.warning(msg)
```

---

## Priority 5: Matrix Dimension Mismatch (4 tests)

### Find VICReg loss usage
```bash
# Find VICReg implementation
grep -r "VICReg\|vicreg" src/ --include="*.py"

# Find dimension specifications
grep -r "768\|512\|128" src/selfsupervised_losses.py

# Check test setup
grep -r "HybridPretrainVICRegLoss" tests/ --include="*.py" -A10
```

### Check tensor dimensions
```python
# Error shows: mat1 (16x128) @ mat2 (768x512)
# Input is 128-dim but expected 768-dim

# Find where embeddings are created
grep -r "embedding.*dim\|hidden.*size" tests/test_selfsupervised_losses.py
```

---

## Priority 6: Mock Configuration (3 tests)

### Find mock setup for critique revision
```bash
# Find mock evaluator creation
grep -r "Mock.*evaluator\|mock_evaluator" tests/test_critique_revision.py -B5 -A10

# Find what attributes are accessed
grep -r "evaluator\.model" src/ --include="*.py"
```

### Expected fix
```python
# WRONG - mock without required attributes
mock_evaluator = Mock()

# CORRECT - mock with model attribute
mock_evaluator = Mock()
mock_evaluator.model = Mock()  # Add missing attribute

# BETTER - use spec to auto-detect
from src.evaluator import Evaluator
mock_evaluator = Mock(spec=Evaluator)
```

---

## Priority 7: Generation Config KeyError (2 tests)

### Find config access
```bash
# Find max_length access
grep -r "config\['max_length'\]" src/ --include="*.py"

# Find generation config usage
grep -r "generation.*config\|gen.*config" src/model_utils.py
```

### Expected fix
```python
# WRONG - direct dictionary access
max_length = config['max_length']  # KeyError if missing

# CORRECT - safe access with default
max_length = config.get('max_length', 100)

# BETTER - validate config first
def validate_config(config):
    required = ['max_length', 'temperature', 'top_p']
    for key in required:
        if key not in config:
            config[key] = DEFAULTS[key]
    return config
```

---

## Priority 8: Assertion Failures

### Consequence Analysis (3 tests)
```bash
# Find consequence analysis function
grep -r "analyze.*consequence\|potential.*consequence" src/principles.py -A20

# Check test expectations
grep -r "unauthorized access\|property damage" tests/test_principles.py
```

### Metrics Logging (2 tests)
```bash
# Find metrics logging
grep -r "def.*log_metrics" src/ --include="*.py" -A15

# Check output capture in tests
grep -r "capsys\|caplog" tests/test_metrics_collector.py
```

### Config Values (2 tests)
```bash
# Find default config initialization
grep -r "class.*GenerationConfig\|def.*generation.*config" src/model_utils.py -A20
```

---

## Quick Verification Commands

### After each fix, verify specific tests
```bash
# Logger fix
pytest tests/test_cai_integration.py::TestFrameworkToEvaluatorIntegration -v

# Training data fix
pytest tests/test_reward_model.py::TestTrainRewardModel::test_training_completes -v

# Logger API fix
pytest tests/test_training.py::test_basic_training -v
pytest tests/test_principles.py::TestJSONParsing::test_parse_valid_json -v

# VICReg fix
pytest tests/test_selfsupervised_losses.py::TestHybridPretrainVICRegLoss::test_basic_forward -v
```

### Run entire module after fix
```bash
pytest tests/test_cai_integration.py -v --tb=short
pytest tests/test_reward_model.py -v --tb=short
pytest tests/test_training.py -v --tb=short
```

---

## Common Code Patterns to Check

### Logger initialization pattern (should be at module level)
```python
import logging

logger = logging.getLogger(__name__)

# NOT inside __init__ or conditional
```

### Test fixture pattern
```python
@pytest.fixture
def logger():
    """Provide configured logger for tests."""
    import logging
    logger = logging.getLogger('test_logger')
    logger.setLevel(logging.DEBUG)
    return logger  # Make sure this returns logger, not None
```

### Safe dictionary access
```python
# WRONG
value = dict['key']

# CORRECT
value = dict.get('key', default_value)

# OR with validation
if 'key' not in dict:
    raise ValueError(f"Missing required key: 'key'")
value = dict['key']
```

---

## Find All Error Messages in Code

```bash
# Find where each error message originates
grep -r "Could not find Europarl data" src/ --include="*.py"
grep -r "All.*training examples are invalid" src/ --include="*.py"
grep -r "Generation failed" src/ --include="*.py"

# Find assertion messages in tests
grep -r "assert.*dangerous devices" tests/ --include="*.py"
grep -r "assert.*'Train:'" tests/ --include="*.py"
```

---

## Useful One-Liners

### Count logger issues
```bash
grep -r "logger\.info\|logger\.debug" src/ --include="*.py" | wc -l
```

### Find all None assignments that might break
```bash
grep -r "logger.*=.*None" src/ --include="*.py"
```

### Find all dictionary access (potential KeyErrors)
```bash
grep -rE "\w+\['[^']+'\]" src/ --include="*.py" | grep -v "\.get("
```

### Check for Mock usage in tests
```bash
grep -r "Mock()" tests/ --include="*.py" | grep -v "spec="
```

---

## Files to Check (Priority Order)

1. `/Users/apa/ml_projects/claud_code/multimodal_insight_engine/tests/conftest.py`
2. `/Users/apa/ml_projects/claud_code/multimodal_insight_engine/src/cai/__init__.py`
3. `/Users/apa/ml_projects/claud_code/multimodal_insight_engine/src/evaluator.py`
4. `/Users/apa/ml_projects/claud_code/multimodal_insight_engine/src/filter.py`
5. `/Users/apa/ml_projects/claud_code/multimodal_insight_engine/src/principles.py`
6. `/Users/apa/ml_projects/claud_code/multimodal_insight_engine/src/training.py`
7. `/Users/apa/ml_projects/claud_code/multimodal_insight_engine/src/reward_model.py`
8. `/Users/apa/ml_projects/claud_code/multimodal_insight_engine/src/model_utils.py`
9. `/Users/apa/ml_projects/claud_code/multimodal_insight_engine/tests/test_critique_revision.py`
10. `/Users/apa/ml_projects/claud_code/multimodal_insight_engine/tests/test_reward_model.py`

---

## Debugging Tips

### To understand logger issue
```bash
# Add to test to debug
def test_something(logger):
    print(f"Logger type: {type(logger)}")
    print(f"Logger value: {logger}")
    assert logger is not None
```

### To understand data format
```bash
# Add to test to debug
def test_training(training_data):
    print(f"Data keys: {training_data[0].keys()}")
    print(f"Data sample: {training_data[0]}")
    assert 'chosen' in training_data[0]
```

### To understand tensor shapes
```bash
# Add to test to debug
def test_loss(input_tensor):
    print(f"Input shape: {input_tensor.shape}")
    print(f"Expected: batch_size x 768")
```
