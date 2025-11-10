# Getting Started with MultiModal Insight Engine

A quick guide to get the project running in under 5 minutes.

## Prerequisites
- Python 3.8+
- pip and venv
- Git

## 1. Clone and Setup (2 minutes)

```bash
# Clone the repository
git clone <repository-url>
cd multimodal_insight_engine

# Create virtual environment
python -m venv venv

# Activate it
source venv/bin/activate          # On Linux/Mac
# OR
venv\Scripts\activate              # On Windows
```

## 2. Install Dependencies (1-2 minutes)

```bash
# Use the convenient make command (if you have make installed)
make install-dev

# OR install manually
pip install -r requirements/base.txt -r requirements/dev.txt
```

## 3. Verify Installation (1 minute)

```bash
# Run a quick sanity check
python -c "import torch; print(f'PyTorch {torch.__version__} ready')"

# List available tests
pytest --co -q | head -20
```

## 4. Run Your First Test (1 minute)

```bash
# Run fast unit tests only
make test-fast

# OR run all tests
make test
```

## 5. Try the Monitoring Demo (Optional, 30 seconds)

```bash
# Run the training monitoring demo
python demos/ppo_monitoring_demo.py --quick

# Check the generated reports
ls training_outputs/
cat training_outputs/training_report.md
```

Done! You're ready to develop.

---

## Common Commands

### Development Commands

```bash
# Auto-format code
make format

# Run linter
make lint

# Type check
make type-check

# Run all checks at once
make check

# Run tests with coverage
make test

# Run only fast tests
make test-fast

# Clean build artifacts
make clean
```

### Helpful Info

```bash
# See all available commands
make help

# Check your environment
python -c "import src; print('Import successful')"

# List tests by file
pytest --collect-only -q

# Run specific test
pytest tests/test_file.py::test_function -v
```

---

## Project Structure

```
multimodal_insight_engine/
├── src/                 # Main source code
│   ├── models/         # Model implementations
│   ├── data/           # Data processing
│   ├── training/       # Training utilities (including monitoring system)
│   ├── safety/         # Safety evaluation (Constitutional AI)
│   └── utils/          # Utility functions
├── tests/              # Test suite (202 monitoring tests)
├── docs/               # Documentation
├── demos/              # Demo scripts (including monitoring demo)
│   ├── ppo_monitoring_demo.py           # Training monitoring demo
│   └── MONITORING_USER_GUIDE.md         # Comprehensive monitoring guide
├── requirements/       # Dependency files
├── Makefile           # Common tasks
└── README.md          # Full documentation
```

---

## Next Steps

1. **Read the documentation**: See `docs/ARCHITECTURE.md` for system overview
2. **Look at examples**: Check `demos/` for usage examples
3. **Try the monitoring system**: See `demos/MONITORING_USER_GUIDE.md` for training monitoring
4. **Run tests**: `make test` to ensure everything works (202 monitoring tests included)
5. **See code style**: Read `CONTRIBUTING.md` for development guidelines
6. **Debug issues**: Check `docs/DEBUGGING.md` if something breaks

---

## Troubleshooting

### Python not found
```bash
# Ensure Python 3.8+ is installed
python --version

# If not available, install or use python3
python3 -m venv venv
```

### Virtual environment not activating
```bash
# Check you're in the right directory
pwd

# Try the full path
source /path/to/project/venv/bin/activate
```

### Import errors
```bash
# Ensure you're in the venv and dependencies are installed
which python  # Should point to venv/bin/python
pip list      # Should show all dependencies

# Reinstall if needed
pip install -r requirements/base.txt
```

### Tests failing
```bash
# Check your environment
python -c "import torch; import pytest"

# Run with verbose output to see what's wrong
pytest tests/ -v --tb=long

# Check one test in detail
pytest tests/test_file.py::test_function -vv
```

### Permission denied on venv/bin/activate
```bash
# Make sure the activate script is executable
chmod +x venv/bin/activate
```

---

## Need More Help?

- **Architecture overview**: See `docs/ARCHITECTURE.md`
- **How to contribute**: See `CONTRIBUTING.md`
- **Debugging issues**: See `docs/DEBUGGING.md`
- **Full documentation**: See `README.md`

---

## Quick Reference

| Task | Command |
|------|---------|
| Install | `make install-dev` |
| Test | `make test` |
| Fast test | `make test-fast` |
| Lint | `make lint` |
| Format | `make format` |
| Type check | `make type-check` |
| Clean | `make clean` |
| Help | `make help` |

---

Happy coding! If you have questions, check the `docs/` directory or open an issue.
