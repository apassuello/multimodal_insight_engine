# MultiModal Insight Engine Guidelines

## Build, Test, & Lint Commands
```bash
# Run all tests with coverage
./run_tests.sh

# Run a single test file
python -m pytest tests/test_file.py -v

# Run a specific test function
python -m pytest tests/test_file.py::test_function -v

# Run lint checks
flake8 src/ tests/

# Type checking
mypy src/ tests/
```

## Style Guidelines
- **Code Style**: Follow PEP 8 (4-space indentation, 79-char line limit)
- **Imports**: Group standard lib → third-party → local; alphabetically sorted in each group
- **Typing**: Use type hints for all function parameters and returns
- **Docstrings**: Google-style docstrings with Args/Returns sections
- **Classes**: CamelCase for classes, snake_case for functions/variables
- **Error Handling**: Validate inputs with assertions, use specific exception types
- **Testing**: Every module should have corresponding tests in tests/ directory
- **Module Headers**: Include module-level docstrings with PURPOSE and KEY COMPONENTS
- **Naming**: Descriptive variable names that indicate purpose and type