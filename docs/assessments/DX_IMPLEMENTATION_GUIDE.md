# DX Improvement Implementation Guide

Quick reference for implementing the recommendations in `DX_ASSESSMENT.md`.

## Phase 1: Immediate Wins (Do Today! ~1.5 hours)

These are low-effort, high-impact changes you can do right now.

### Step 1: Fix Configuration Files (15 min)

#### 1a. Update VSCode Settings
```bash
# Replace hardcoded paths with workspace-relative ones
# File: .vscode/settings.json
# Before: "python.defaultInterpreterPath": "/Users/apa/miniconda3/envs/me/bin/python"
# After:  "python.defaultInterpreterPath": "${workspaceFolder}/venv/bin/python"
```

#### 1b. Update Pyright Config
```bash
# File: pyrightconfig.json
# Should use environment-agnostic configuration
```

See the improved config templates in this repo.

### Step 2: Create Environment Template (5 min)
```bash
# Create .env.example with needed variables
cat > .env.example << 'EOF'
PYTHONPATH=.
DEBUG=false
CUDA_VISIBLE_DEVICES=0
SEED=42
EOF
```

### Step 3: Review and Commit New Files (10 min)

We've created these files for you:
- `Makefile` - Common task automation
- `.pre-commit-config.yaml` - Git hook configuration
- `GETTING_STARTED.md` - Quick start guide
- `CONTRIBUTING.md` - Development guidelines
- `requirements/base.txt` - Core dependencies
- `requirements/dev.txt` - Development tools
- `requirements/all.txt` - Everything combined

Review them and commit:
```bash
git add Makefile .pre-commit-config.yaml GETTING_STARTED.md CONTRIBUTING.md requirements/
git commit -m "chore: Add DX improvements (Makefile, pre-commit, documentation)"
```

### Step 4: Update .gitignore (10 min)
```bash
# Backup current
cp .gitignore .gitignore.backup

# Replace with improved version
cp .gitignore.improved .gitignore

# Verify changes look good
git diff .gitignore | head -50
```

**Checkpoint: Stop here if short on time. You've already improved DX significantly!**

---

## Phase 2: Automation Setup (Next hour ~1.5 hours)

Install and configure tooling for automatic quality checks.

### Step 5: Install Pre-commit Hooks (20 min)
```bash
# Install pre-commit framework
pip install pre-commit

# Install hooks
pre-commit install

# Run on all files to verify
pre-commit run --all-files

# This will format and check your code
# Fix any issues and commit
git add .
git commit -m "style: Auto-format code with black and isort"
```

### Step 6: Set Up Makefile (Already Done!)
```bash
# Just use it
make help              # See all available commands
make test-fast         # Run fast tests
make lint              # Check style
make format            # Auto-format code
```

### Step 7: Reorganize Dependencies (20 min)
```bash
# We've created a requirements/ directory with:
# - base.txt      (core dependencies)
# - dev.txt       (development tools)
# - all.txt       (everything)

# Verify the files work
pip install -r requirements/base.txt      # Minimal install
pip install -r requirements/dev.txt       # Add dev tools
pip install -r requirements/all.txt       # Everything

# Commit changes
git add requirements/
git commit -m "refactor: Organize dependencies into base/dev/all"

# Update any documentation referencing old requirements.txt
# Update GETTING_STARTED.md if needed
```

### Step 8: Run Quality Checks (30 min)
```bash
# Now that everything is set up, run the checks
make lint              # Should pass or give style errors
make format            # Auto-fix style issues
make check             # Run all checks
make test-fast         # Quick test to verify

# Commit improvements
git add .
git commit -m "chore: Apply linting and formatting improvements"
```

**Checkpoint: Your local development is now much better! Pre-commit will prevent bad commits.**

---

## Phase 3: Documentation (30 min)

Improve project documentation.

### Step 9: Consolidate Root-Level Docs (30 min)
```bash
# Create docs/archive/ for old docs
mkdir -p docs/archive

# Move old assessment docs
mv AUDIT_FINDINGS.md docs/archive/
mv MERGE_READINESS_ASSESSMENT.md docs/archive/
mv code_quality_assessment.md docs/archive/
# (Keep CRITICAL_README.md if it's important, otherwise archive it too)

# Verify new docs are in place
ls -la GETTING_STARTED.md CONTRIBUTING.md README.md

# Commit cleanup
git add .
git commit -m "docs: Organize documentation, move legacy docs to archive"
```

**Checkpoint: Root directory is much cleaner!**

---

## Phase 4: CI/CD Pipeline (Optional - takes 2 hours)

### Step 10: Set Up GitHub Actions
Create `.github/workflows/ci.yml`:

```yaml
name: CI

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        python-version: ['3.8', '3.9', '3.10']

    steps:
      - uses: actions/checkout@v3
      - uses: actions/setup-python@v4
        with:
          python-version: ${{ matrix.python-version }}

      - name: Install dependencies
        run: |
          pip install -r requirements/base.txt -r requirements/dev.txt

      - name: Lint
        run: flake8 src/ tests/

      - name: Type check
        run: mypy src/

      - name: Run tests
        run: pytest tests/ --cov=src --cov-report=xml

      - name: Upload coverage
        uses: codecov/codecov-action@v3
```

Then:
```bash
git add .github/
git commit -m "ci: Add GitHub Actions CI pipeline"
git push origin your-branch
```

---

## Verification Checklist

After implementing, verify these work:

- [ ] `make help` shows available commands
- [ ] `make test-fast` runs in <30 seconds
- [ ] `make lint` shows no errors (or expected issues)
- [ ] `make format` auto-formats code
- [ ] `make check` runs all checks
- [ ] Pre-commit hooks run on commit
- [ ] New developers can follow GETTING_STARTED.md
- [ ] Repository root is organized (no clutter)
- [ ] Dependencies are clearly separated
- [ ] Documentation is organized in docs/

## DX Improvement Checklist

Check off as you complete each phase:

### Phase 1: Immediate Wins
- [ ] Fixed VSCode and Pyright configs
- [ ] Created .env.example
- [ ] Reviewed new files (Makefile, pre-commit, docs)
- [ ] Updated .gitignore
- [ ] Initial commit

### Phase 2: Automation
- [ ] Installed pre-commit hooks
- [ ] Verified Makefile works
- [ ] Reorganized dependencies
- [ ] Ran quality checks
- [ ] Committed improvements

### Phase 3: Documentation
- [ ] Consolidated root-level docs
- [ ] Archived old docs
- [ ] Updated any references
- [ ] Verified docs structure

### Phase 4: CI/CD (Optional)
- [ ] Created GitHub Actions workflow
- [ ] Tested on a feature branch
- [ ] Verified tests run on push

---

## If You Get Stuck

### "Pre-commit hooks failing"
```bash
# See what's wrong
pre-commit run --all-files -v

# Fix issues and try again
pre-commit run --all-files
```

### "Make command not found"
```bash
# On Mac/Linux
brew install make
# or
apt-get install build-essential

# Then it should work
make help
```

### "Import errors after reorganizing"
```bash
# Clear Python cache
make clean

# Reinstall
pip install -r requirements/dev.txt

# Try again
make test-fast
```

### "Tests still failing"
```bash
# Run with verbose output
make test-verbose

# Check specific test
pytest tests/test_file.py::test_function -vv
```

---

## Success Indicators

You'll know you've succeeded when:

1. **Faster onboarding**: New developers can go from clone to running tests in <5 min
2. **No bad commits**: Pre-commit hooks catch issues before they enter the repo
3. **Clear expectations**: CONTRIBUTING.md shows developers how to work
4. **Organized codebase**: Root directory is clean, logical structure
5. **Automated checks**: `make check` finds problems automatically
6. **Fast feedback**: `make test-fast` gives results in <30 seconds

---

## What's Next?

After completing these phases, consider:

1. Add project-specific documentation
2. Create contributing guidelines for external contributors
3. Set up GitHub Pages for documentation
4. Add security scanning (already configured with bandit)
5. Add performance benchmarking
6. Create contribution templates

See `DX_ASSESSMENT.md` for detailed recommendations.

---

## Quick Reference

```bash
# Most important commands
make help              # See all commands
make install-dev       # Set up environment
make test-fast         # Quick validation
make lint              # Check code style
make format            # Auto-fix formatting
make check             # Run all quality checks

# Git workflow
pre-commit install     # Set up hooks
git add .
git commit -m "message"  # Pre-commit runs automatically
git push
```

Good luck! Feel free to reach out if you have questions.
