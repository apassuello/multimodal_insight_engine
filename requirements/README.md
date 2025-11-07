# Dependency Management

This directory contains organized dependency files for the MultiModal Insight Engine.

## Files

- **`base.txt`** - Core dependencies needed for running the project
- **`dev.txt`** - Development tools (testing, linting, formatting, type checking)
- **`docs.txt`** - Documentation generation tools (optional)
- **`all.txt`** - Everything combined (for CI/exhaustive testing)

## Installation

### For Development (Recommended)
```bash
pip install -r requirements/base.txt -r requirements/dev.txt
```

Or use the Makefile:
```bash
make install-dev
```

### For Production
```bash
pip install -r requirements/base.txt
```

### For Everything
```bash
pip install -r requirements/all.txt
```

## Updating Dependencies

### Add a new dependency
1. Determine which file it belongs in
2. Add the package name to that file
3. Reinstall: `pip install -r requirements/<file>.txt`

### Freeze current state
```bash
pip freeze > requirements/current-state.txt
```

### Update a specific package
```bash
pip install --upgrade package-name
pip freeze | grep package-name >> requirements/<file>.txt
```

## Dependency Organization

### Core (base.txt)
Libraries needed to run the project:
- PyTorch and related (torch, torchvision, torchaudio)
- Data processing (numpy, pandas, datasets)
- ML frameworks (transformers, lightning)
- Utilities (tqdm, pydantic, pyyaml)

### Development (dev.txt)
Tools for development, testing, and quality:
- Testing (pytest, pytest-cov, pytest-xdist)
- Linting (flake8, flake8-docstrings, flake8-bugbear)
- Type checking (mypy)
- Code formatting (black, isort)
- Security (bandit)
- Pre-commit framework (pre-commit)

### Documentation (docs.txt)
Optional tools for generating documentation:
- Sphinx and extensions
- Theme packages
- Autodoc plugins

## Best Practices

1. **Pin major versions** in base.txt for reproducibility
2. **Use ranges** for dev tools (they're more flexible)
3. **Keep dev.txt minimal** - only essential tools
4. **Test before committing** - ensure new dependencies work
5. **Document why** - add comments for unusual dependencies

## Troubleshooting

### Conflicting dependencies
```bash
pip install --dry-run -r requirements/all.txt
# Check output for conflicts before actually installing
```

### Clean reinstall
```bash
pip uninstall -r requirements/all.txt -y
pip install -r requirements/base.txt -r requirements/dev.txt
```

### Check what's installed
```bash
pip list
pip freeze
```

### Update all packages safely
```bash
pip list --outdated
# Review changes, then update one by one:
pip install --upgrade package-name
```
