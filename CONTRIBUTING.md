# Contributing to MultiModal Insight Engine

Thank you for your interest in contributing to the MultiModal Insight Engine! This document provides guidelines for contributing to the project.

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [How to Contribute](#how-to-contribute)
- [Development Workflow](#development-workflow)
- [Code Style Guidelines](#code-style-guidelines)
- [Testing Requirements](#testing-requirements)
- [Commit Message Conventions](#commit-message-conventions)
- [Pull Request Process](#pull-request-process)

## Code of Conduct

This project adheres to a code of conduct that all contributors are expected to follow. Please be respectful and constructive in all interactions.

## How to Contribute

### Reporting Bugs

- Check if the bug has already been reported in the Issues section
- If not, create a new issue with a clear title and description
- Include steps to reproduce, expected behavior, and actual behavior
- Add relevant labels and provide system information (Python version, OS, etc.)

### Suggesting Enhancements

- Check if the enhancement has already been suggested
- Create a new issue with a clear description of the proposed feature
- Explain the use case and potential benefits
- Be open to discussion and feedback

### Code Contributions

1. **Fork the repository**
2. **Create a feature branch** from `main`
   ```bash
   git checkout -b feature/your-feature-name
   ```
3. **Make your changes** following the code style guidelines
4. **Write or update tests** to cover your changes
5. **Run the test suite** to ensure everything passes
6. **Commit your changes** with clear, descriptive commit messages
7. **Push to your fork** and submit a pull request

## Development Workflow

### Setting Up Your Development Environment

```bash
# Clone your fork
git clone https://github.com/your-username/multimodal_insight_engine.git
cd multimodal_insight_engine

# Create and activate a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install development dependencies
pip install -r requirements-dev.txt  # If available

# Install the package in editable mode
pip install -e .

# Verify installation
python verify_install.py
```

### Running Tests

```bash
# Run all tests with coverage
./run_tests.sh

# Run specific test file
python -m pytest tests/test_file.py -v

# Run specific test function
python -m pytest tests/test_file.py::test_function -v

# Run with coverage report
pytest tests/ --cov=src --cov=demo --cov-report=html
```

### Code Quality Checks

```bash
# Run linting
flake8 src/ tests/ demo/

# Run type checking
mypy src/ tests/ demo/

# Format code (if using black)
black src/ tests/ demo/
```

## Code Style Guidelines

This project follows [PEP 8](https://www.python.org/dev/peps/pep-0008/) style guidelines. Key points:

### Python Style

- **Indentation**: 4 spaces (no tabs)
- **Line Length**: Maximum 79 characters for code, 72 for docstrings/comments
- **Imports**: Group in order: standard library → third-party → local
- **Naming Conventions**:
  - Classes: `CamelCase`
  - Functions/variables: `snake_case`
  - Constants: `UPPER_CASE`
  - Private members: `_leading_underscore`

### Type Hints

Use type hints for all function parameters and return values:

```python
def process_data(input_text: str, max_length: int = 100) -> List[str]:
    """Process input text and return tokens.

    Args:
        input_text: The text to process
        max_length: Maximum token length (default: 100)

    Returns:
        List of processed tokens
    """
    # Implementation
    return tokens
```

### Docstrings

Use Google-style docstrings:

```python
def example_function(param1: int, param2: str) -> bool:
    """Short description of the function.

    Longer description with more details about what the function does,
    its behavior, and any important notes.

    Args:
        param1: Description of param1
        param2: Description of param2

    Returns:
        Description of return value

    Raises:
        ValueError: When param1 is negative
        TypeError: When param2 is not a string

    Example:
        >>> example_function(42, "test")
        True
    """
    # Implementation
```

### Module Headers

Include module-level docstrings:

```python
"""Module description.

PURPOSE:
    Brief description of the module's purpose

KEY COMPONENTS:
    - Component1: Description
    - Component2: Description
"""
```

## Testing Requirements

### Test Coverage

- All new code must include tests
- Maintain minimum 80% code coverage
- Aim for comprehensive test coverage of critical paths

### Test Structure

- **Unit Tests**: Test individual functions/methods in isolation
- **Integration Tests**: Test component interactions
- **End-to-End Tests**: Test complete workflows

### Writing Tests

```python
import pytest
from src.module import YourClass

class TestYourClass:
    """Test suite for YourClass."""

    @pytest.fixture
    def instance(self):
        """Create test instance."""
        return YourClass()

    def test_basic_functionality(self, instance):
        """Test basic functionality."""
        result = instance.method()
        assert result == expected_value

    def test_error_handling(self, instance):
        """Test error handling."""
        with pytest.raises(ValueError):
            instance.method(invalid_input)
```

### Test Naming

- Test files: `test_<module_name>.py`
- Test classes: `Test<ClassName>`
- Test methods: `test_<description_of_what_is_tested>`

## Commit Message Conventions

Follow the conventional commit format:

```
[type] Brief description (max 72 characters)

Detailed explanation of the changes (optional):
- What was changed
- Why it was changed
- Any breaking changes or migration notes

Closes #123
```

### Commit Types

- `[feat]` - New feature
- `[fix]` - Bug fix
- `[docs]` - Documentation changes
- `[test]` - Test additions or modifications
- `[refactor]` - Code refactoring (no functional changes)
- `[perf]` - Performance improvements
- `[style]` - Code style changes (formatting, etc.)
- `[build]` - Build system or dependency changes
- `[ci]` - CI/CD configuration changes
- `[chore]` - Other changes that don't modify src/test files

### Examples

```
[feat] Add support for custom tokenizer vocabularies

- Implement vocabulary loading from file
- Add validation for custom vocabularies
- Update documentation with examples

Closes #45
```

```
[fix] Fix memory leak in attention mechanism

The attention weights were not being properly released
after computation, causing memory to accumulate during
long training runs.

Fixes #78
```

## Pull Request Process

### Before Submitting

1. **Update documentation** - Document new features or API changes
2. **Add tests** - Ensure your code is well-tested
3. **Run the test suite** - All tests must pass
4. **Check code style** - Run linting and fix any issues
5. **Update CHANGELOG.md** - Add entry for your changes
6. **Rebase on latest main** - Ensure your branch is up-to-date

### Pull Request Template

When creating a pull request, include:

```markdown
## Description
Brief description of what this PR does

## Type of Change
- [ ] Bug fix (non-breaking change which fixes an issue)
- [ ] New feature (non-breaking change which adds functionality)
- [ ] Breaking change (fix or feature that would cause existing functionality to not work as expected)
- [ ] Documentation update

## How Has This Been Tested?
Describe the tests you ran and how to reproduce them

## Checklist
- [ ] My code follows the style guidelines of this project
- [ ] I have performed a self-review of my own code
- [ ] I have commented my code, particularly in hard-to-understand areas
- [ ] I have made corresponding changes to the documentation
- [ ] My changes generate no new warnings
- [ ] I have added tests that prove my fix is effective or that my feature works
- [ ] New and existing unit tests pass locally with my changes
- [ ] Any dependent changes have been merged and published
```

### Review Process

1. **Automated checks** - CI/CD pipeline runs tests and linting
2. **Code review** - At least one maintainer reviews the code
3. **Address feedback** - Make requested changes
4. **Approval** - Once approved, the PR can be merged
5. **Merge** - Maintainer will merge using squash or rebase

### After Merging

- Delete your feature branch
- Pull the latest changes from main
- Update your local repository

## Development Best Practices

### Error Handling

- Use specific exception types
- Provide clear error messages
- Validate inputs with assertions or explicit checks
- Log errors appropriately

### Performance Considerations

- Profile code before optimizing
- Use appropriate data structures
- Avoid premature optimization
- Document performance-critical sections

### Security

- Never commit secrets or credentials
- Validate all user inputs
- Follow security best practices (see SECURITY.md)
- Report security vulnerabilities privately

### Documentation

- Keep documentation up-to-date with code changes
- Include docstrings for all public functions/classes
- Add examples for complex functionality
- Update README.md for user-facing changes

## Getting Help

If you need help or have questions:

- Check existing documentation
- Search through existing issues
- Create a new issue with the "question" label
- Join community discussions (if applicable)

## Recognition

Contributors will be recognized in:
- AUTHORS.md file
- Release notes
- Project documentation

Thank you for contributing to MultiModal Insight Engine!
