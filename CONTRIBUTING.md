# Contributing Guide

Thank you for contributing to the MultiModal Insight Engine! This guide ensures consistent code quality and smooth collaboration.

## Development Workflow

### 1. Create a Feature Branch
```bash
git checkout -b feature/your-feature-name
# or
git checkout -b fix/issue-number-description
# or
git checkout -b docs/documentation-topic
```

Branch naming conventions:
- `feature/` - New functionality
- `fix/` - Bug fixes
- `docs/` - Documentation updates
- `refactor/` - Code refactoring
- `test/` - Test additions/improvements

### 2. Make Your Changes
- Write code following the style guide (see below)
- Add tests for new functionality
- Update documentation as needed

### 3. Run Local Checks
Before committing, run these checks:

```bash
# Lint check
make lint

# Format code
make format

# Type check
make type-check

# Run tests
make test-fast     # Quick test
make test          # Full test suite
```

Or run all at once:
```bash
make check         # Runs lint, format, type-check
```

### 4. Commit Your Changes
```bash
git add .
git commit -m "Clear, descriptive commit message"
```

Pre-commit hooks will automatically:
- Format code with black and isort
- Run flake8 linter
- Check for security issues with bandit
- Verify type hints with mypy

If hooks fail, fix the issues and try committing again.

### 5. Push and Create Pull Request
```bash
git push origin feature/your-feature-name
```

Then create a pull request on GitHub with:
- Clear title describing the change
- Description of what changed and why
- Reference to any related issues
- Results of local testing

### 6. Review and Merge
- Address feedback from code reviewers
- Run CI checks pass (automated on GitHub)
- Squash commits if needed
- Merge to main branch

---

## Code Style Guide

### Python Style
- Follow **PEP 8** strictly
- Max line length: **100 characters** (enforced by black)
- Use 4-space indentation

### Type Hints
Every function must have type hints:

```python
# GOOD
def process_tensor(
    data: torch.Tensor,
    batch_size: int,
) -> Tuple[torch.Tensor, np.ndarray]:
    """Process tensor data.

    Args:
        data: Input tensor of shape (N, D)
        batch_size: Batch size for processing

    Returns:
        Processed tensor and metadata array
    """
    # Implementation
    return processed, metadata

# BAD - Missing type hints
def process_tensor(data, batch_size):
    return processed, metadata
```

### Docstrings
Use **Google-style docstrings** with all sections:

```python
def create_model(
    vocab_size: int,
    hidden_dim: int,
    num_heads: int,
    dropout: float = 0.1,
) -> nn.Module:
    """Create a transformer model.

    Args:
        vocab_size: Size of vocabulary
        hidden_dim: Hidden dimension size
        num_heads: Number of attention heads
        dropout: Dropout probability. Defaults to 0.1.

    Returns:
        Initialized transformer model

    Raises:
        ValueError: If vocab_size < 1

    Example:
        >>> model = create_model(10000, 512, 8)
        >>> output = model(input_ids)
    """
    if vocab_size < 1:
        raise ValueError(f"vocab_size must be >= 1, got {vocab_size}")
    # Implementation
    return model
```

### Imports
Group and sort imports:
```python
# 1. Standard library
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# 2. Third-party
import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam

# 3. Local/relative
from src.models.transformer import Transformer
from src.utils.logging import get_logger
```

Use `isort` to automatically organize:
```bash
make format  # Runs isort
```

### Class and Function Naming
```python
# Classes: PascalCase
class MultiHeadAttention(nn.Module):
    pass

class SafetyEvaluator:
    pass

# Functions and methods: snake_case
def forward_pass(model, data):
    pass

def evaluate_model_safety():
    pass

# Constants: UPPER_CASE
MAX_SEQUENCE_LENGTH = 512
DEFAULT_DROPOUT = 0.1
```

### Error Handling
Always use specific exceptions:

```python
# GOOD
if not isinstance(data, torch.Tensor):
    raise TypeError(f"Expected Tensor, got {type(data)}")

if data.shape[0] == 0:
    raise ValueError("data cannot be empty")

# BAD - Too generic
if not isinstance(data, torch.Tensor):
    raise Exception("Invalid type")
```

---

## Testing Requirements

### 1. Write Tests for All New Code
Every new module/function needs tests:

```bash
# For src/models/new_model.py
# Write tests/test_new_model.py
```

### 2. Test Structure
```python
import pytest
from src.models.new_model import NewModel

class TestNewModel:
    """Test cases for NewModel class."""

    def setup_method(self):
        """Setup test fixtures."""
        self.model = NewModel(hidden_dim=128)

    def test_initialization(self):
        """Test model initializes correctly."""
        assert self.model is not None
        assert self.model.hidden_dim == 128

    def test_forward_pass(self):
        """Test forward pass works."""
        input_data = torch.randn(2, 10)
        output = self.model(input_data)
        assert output.shape == (2, 10)

    @pytest.mark.slow
    def test_training_step(self):
        """Test training step (slow test)."""
        # Slow tests marked with @pytest.mark.slow
        pass
```

### 3. Run Tests Before Committing
```bash
make test-fast     # Quick validation (<30 sec)
make test          # Full test suite with coverage
```

### 4. Coverage Requirements
- Target: **90%+ coverage**
- Check coverage report: `coverage_html/index.html`
- Mark untestable code with `# pragma: no cover`

---

## Documentation Requirements

### 1. Module-Level Docstring
Every module must have a docstring at the top:

```python
"""
Models for neural network architectures.

This module provides transformer-based model implementations including
attention mechanisms, positional encodings, and layer implementations.

Key Components:
    - Transformer: Main encoder-decoder architecture
    - MultiHeadAttention: Multi-head attention mechanism
    - FeedForward: Position-wise feed-forward network
"""
```

### 2. Function Documentation
Every function must be documented:
```python
def tokenize(
    text: str,
    max_length: int = 512,
) -> List[int]:
    """Convert text to token IDs.

    Args:
        text: Input text to tokenize
        max_length: Maximum number of tokens. Defaults to 512.

    Returns:
        List of token IDs

    Raises:
        ValueError: If text is empty
    """
```

### 3. Update README if Needed
If your change affects usage, update `README.md` and relevant docs.

### 4. Add Comments for Complex Logic
```python
# GOOD - Explains why, not what
# Use residual connection to enable training of very deep networks
x = x + self.feed_forward(self.layer_norm(x))

# BAD - Explains what is obvious from code
# Add x to feed_forward output
x = x + self.feed_forward(x)
```

---

## Common Commands

```bash
# Format and check code
make format      # Auto-format code
make lint        # Check for style violations
make type-check  # Run type checker
make check       # Run all checks

# Testing
make test        # Full test suite
make test-fast   # Quick unit tests only
make test-verbose # Detailed output

# Setup
make setup       # Create venv and install everything
make install-dev # Install dev dependencies

# Cleanup
make clean       # Remove artifacts
```

---

## Commit Message Guidelines

Write clear, descriptive commit messages:

```bash
# GOOD
git commit -m "Add attention mechanism visualization

- Implement AttentionVisualizer class
- Add heatmap plotting for attention weights
- Include tests for edge cases (zero weights, single head)
- Update docs with usage example"

# Also acceptable for simple changes
git commit -m "Fix typo in docstring"
git commit -m "Add missing type hint to forward_pass"

# BAD
git commit -m "fix stuff"
git commit -m "update code"
git commit -m "work in progress"
```

---

## Pull Request Checklist

Before submitting a PR, ensure:

- [ ] Code follows style guide (run `make check`)
- [ ] All tests pass (run `make test`)
- [ ] New code has tests (>80% coverage)
- [ ] Docstrings are complete (Google style)
- [ ] Type hints are present on all functions
- [ ] No commented-out code left
- [ ] No debug print statements
- [ ] Branch is up-to-date with main
- [ ] Commit messages are clear and descriptive
- [ ] PR description explains the change and why

---

## Getting Help

- **Questions about code style?** Check `CLAUDE.md`
- **Need debugging help?** See `docs/DEBUGGING.md`
- **Want to understand architecture?** Read `docs/ARCHITECTURE.md`
- **Looking for examples?** Check `demos/` directory
- **Need test tips?** See `docs/TESTING_QUICK_REFERENCE.md`

---

## Recognition

Thank you for helping improve the MultiModal Insight Engine! Contributors will be recognized in the project documentation.

Happy contributing!
