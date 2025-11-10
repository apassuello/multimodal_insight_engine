# Getting Started with MultiModal Insight Engine

**Estimated Time: 20-30 minutes**

This guide will have you up and running with the MultiModal Insight Engine in just a few steps.

## Prerequisites

- Python 3.8 or newer (`python --version`)
- Git
- pip (Python package manager)
- ~2GB disk space for dependencies

## Quick Setup (5 Steps)

### Step 1: Clone the Repository
```bash
git clone https://github.com/yourusername/multimodal_insight_engine.git
cd multimodal_insight_engine
```

### Step 2: Create a Virtual Environment
This isolates project dependencies from your system Python.

```bash
# Create virtual environment
python -m venv venv

# Activate it
# On Linux/macOS:
source venv/bin/activate
# On Windows:
# venv\Scripts\activate
```

**Expected output**: Your shell prompt should show `(venv)` prefix.

### Step 3: Install Dependencies
```bash
# Upgrade pip first
pip install --upgrade pip

# Install project dependencies (takes 3-5 minutes)
pip install -r requirements.txt

# Install the project itself in development mode
pip install -e .
```

**Note**: First-time installation takes 3-5 minutes due to 331 dependencies. Subsequent installs are faster.

### Step 4: Verify Installation
```bash
# Run verification script
python verify_install.py
```

**Expected output**:
```
✓ Core imports successful
✓ PyTorch available
✓ Required dependencies installed
✓ Tests can be discovered

Installation verified! You're ready to go.
```

### Step 5: Run Your First Test
```bash
# Run all tests
make test

# Or manually:
pytest tests/ -v --tb=short
```

**Expected output**: Should see tests running with pass/fail counts.

---

## What's Next?

### To Run Demos:
```bash
cd demos/
python language_model_demo.py --help

# Try the training monitoring demo
python ppo_monitoring_demo.py --quick
```

See [demos/README.md](demos/README.md) and [demos/MONITORING_USER_GUIDE.md](demos/MONITORING_USER_GUIDE.md) for more information.

### To Understand the Code:
1. Read [docs/INDEX.md](docs/INDEX.md) for documentation guide
2. Check [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for code structure
3. Browse `src/` - each module has docstrings

### To Develop:
```bash
# Install development tools
pip install -r requirements-dev.txt

# Run linting and type checks
make lint

# Format code
make format

# Run all quality checks
make check
```

### To Train a Model:
See [docs/TRAINING.md](docs/TRAINING.md) for detailed training guides.

---

## Common Issues & Solutions

### "No module named pytest"
```bash
# Solution: Install test dependencies
pip install pytest pytest-cov

# Or reinstall the project
pip install -e .
```

### "ModuleNotFoundError: No module named 'torch'"
```bash
# Solution: Install PyTorch (included in requirements.txt)
pip install torch

# Or reinstall all dependencies
pip install -r requirements.txt
```

### "Python version 3.8 required"
```bash
# Check your Python version
python --version

# If you have multiple Python versions, specify which to use
python3.9 -m venv venv
```

### Virtual Environment Not Activating
```bash
# Make sure you're in the project directory
cd multimodal_insight_engine

# Try activating again
source venv/bin/activate  # Linux/macOS
# or
venv\Scripts\activate     # Windows
```

### "pip: command not found"
```bash
# Use python -m pip instead
python -m pip install -r requirements.txt
```

---

## Development Workflow

### Recommended Daily Workflow:
```bash
# 1. Activate environment
source venv/bin/activate

# 2. Pull latest changes
git pull

# 3. Run all checks before committing
make check

# 4. Commit changes
git commit -m "your message"

# 5. Push to remote
git push
```

### Code Quality Checks:
```bash
# Run tests with coverage
make test

# Run fast tests only
make test-fast

# Run monitoring system tests
make test-monitoring

# Check code style (PEP 8)
make lint

# Type checking
mypy src/ tests/

# Auto-format code
make format

# Run training monitoring demo
make demo-monitoring
```

---

## Project Structure

```
multimodal_insight_engine/
├── src/                    # Source code (your main code)
│   ├── models/            # Transformer models
│   ├── data/              # Data processing & tokenization
│   ├── training/          # Training loops & utilities
│   │   └── monitoring/    # Training monitoring system (202 tests)
│   ├── safety/            # Safety & constitutional AI
│   ├── optimization/      # Model optimization
│   ├── evaluation/        # Evaluation metrics
│   ├── configs/           # Configuration files
│   └── utils/             # Utilities (logging, visualization)
├── tests/                  # Test files (mirror src/ structure)
│   └── training/monitoring/ # Monitoring system tests
├── demos/                  # Demo scripts (usage examples)
│   ├── ppo_monitoring_demo.py  # Training monitoring demo
│   └── MONITORING_USER_GUIDE.md # Comprehensive monitoring guide
├── docs/                   # Documentation
├── scripts/                # Utility scripts
├── requirements.txt        # All dependencies
├── setup.py               # Python package setup
├── Makefile               # Common commands
└── README.md              # Project overview
```

---

## Key Files to Understand

### For Setup & Configuration:
- `requirements.txt` - All dependencies
- `setup.py` - Python package configuration
- `.coveragerc` - Test coverage settings
- `.vscode/settings.json` - VS Code configuration

### For Development:
- `CLAUDE.md` - Development guidelines & style
- `run_tests.sh` - Test runner script
- `Makefile` - Common development tasks

### For Documentation:
- `README.md` - Project overview
- `docs/` - Detailed documentation
- `CRITICAL_README.md` - Important notes on Constitutional AI

---

## Getting Help

### Available Resources:
1. **This file** - Getting started guide
2. **docs/INDEX.md** - Documentation navigation
3. **CLAUDE.md** - Development guidelines
4. **CRITICAL_README.md** - Project-specific clarifications
5. **Source code docstrings** - Each module documents itself

### Common Questions:

**Q: How do I run a specific test?**
```bash
pytest tests/test_models.py::test_transformer -v
```

**Q: How do I train a model?**
```bash
# See docs/TRAINING.md for detailed guides
python demos/language_model_demo.py --dataset wikitext
```

**Q: How do I use Constitutional AI?**
```bash
# See docs/DEMO_GUIDE.md
python demos/constitutional_ai_demo.py --help
```

**Q: How do I use the training monitoring system?**
```bash
# Quick demo
python demos/ppo_monitoring_demo.py --quick

# See comprehensive guide
cat demos/MONITORING_USER_GUIDE.md
```

**Q: How do I profile code for performance?**
```bash
# See src/utils/profiling.py for utilities
python -m cProfile -o profile.stats your_script.py
```

---

## Next Steps

1. ✅ Complete all 5 setup steps above
2. ✅ Run `python verify_install.py` to verify
3. ✅ Run `make test` to confirm tests work
4. ✅ Read `docs/INDEX.md` to navigate documentation
5. ✅ Start with a [demo script](demos/README.md)
6. ✅ Try the [training monitoring demo](demos/MONITORING_USER_GUIDE.md)

---

## Troubleshooting Checklist

- [ ] Virtual environment is activated (check for `(venv)` in prompt)
- [ ] Python version is 3.8+ (`python --version`)
- [ ] Dependencies installed (`pip list | grep torch`)
- [ ] Tests can be discovered (`pytest --collect-only`)
- [ ] Verification script passes (`python verify_install.py`)

If any step fails, check the "Common Issues & Solutions" section above.

---

## Time Estimates

| Step | Time |
|------|------|
| Clone repo | 1 min |
| Virtual environment | 1 min |
| Install dependencies | 3-5 min |
| Verify installation | 2 min |
| First test run | 2 min |
| **TOTAL** | **~10 min** |

*(First installation takes longer due to dependency downloads. Subsequent installations are faster.)*

---

## Quick Reference

| Task | Command |
|------|---------|
| Install | `make install-dev` |
| Test | `make test` |
| Fast test | `make test-fast` |
| Monitoring tests | `make test-monitoring` |
| Monitoring demo | `make demo-monitoring` |
| Lint | `make lint` |
| Format | `make format` |
| Type check | `make type-check` |
| Clean | `make clean` |
| Help | `make help` |

---

**Still stuck?** Check your activation status and dependency installation:
```bash
# Confirm environment is active
which python  # Should show path in venv/

# Verify dependencies
pip list | head -20

# Try verification again
python verify_install.py
```

Good luck! Welcome to the MultiModal Insight Engine!
