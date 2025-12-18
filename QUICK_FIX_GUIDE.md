# Quick Fix Guide - CI Failures

**Estimated Time**: 15-30 minutes
**Risk Level**: LOW
**Testing Required**: YES (validate against all 3 Python versions)

---

## The Problem (30-second version)

```
✗ Python 3.10/3.11: fastapi 0.115.0 needs starlette>=0.37.2 but you have 0.22.0
✗ Python 3.12: numpy 1.24.3 can't build (uses deprecated pkgutil.ImpImporter)
✗ All builds: yaml module missing (in requirements.txt but not setup.py)
```

---

## The Solution (3 changes)

### Change 1: Fix Starlette (1 line in requirements.txt)

```diff
# Line 260 in requirements.txt
- starlette==0.22.0
+ starlette==0.38.0
```

**Why**: FastAPI 0.115.0 requires starlette>=0.37.2, your version is 0.22.0 (13 versions behind)

---

### Change 2: Fix NumPy for Python 3.12 (1 line in requirements.txt)

```diff
# Line 173 in requirements.txt
- numpy==1.24.3
+ numpy==1.26.4
```

**Why**: NumPy 1.24.3 doesn't have Python 3.12 wheels; 1.26.4 is the LTS version for 3.10-3.12

---

### Change 3: Fix PyYAML Missing from Build (setup.py)

```diff
# Lines 7-10 in setup.py
  install_requires=[
-     "pytest>=7.0",
-     "pytest-cov>=4.0",
+     # Core runtime dependencies
+     "PyYAML>=6.0",
+     "pydantic>=2.0.0",
+     "fastapi>=0.100.0",
+     "starlette>=0.37.2",
+     "uvicorn>=0.20.0",
+     "torch>=1.11.0",
+     "transformers>=4.30.0",
+     "lightning>=2.0.0",
+
+     # Testing
+     "pytest>=7.0",
+     "pytest-cov>=4.0",
  ]
```

**Why**: setup.py defines what pip installs; your code imports yaml but it's not declared here. The build process uses setup.py, not requirements.txt.

---

## Bonus Fixes (Optional but Recommended)

### Bonus 1: Fix Python Version Mismatch (1 line in setup.py)

```diff
# Line 11 in setup.py
- python_requires=">=3.8",
+ python_requires=">=3.10",
```

**Why**: Your actual minimum is 3.10 (pyproject.toml says this, numpy 1.24 doesn't support 3.8/3.9 anyway)

---

### Bonus 2: Remove Duplicate Package (1 line in requirements.txt)

```bash
# Line 52 in requirements.txt - remove this line entirely
- lightning==2.0.0
  # Keep pytorch-lightning==2.0.0 instead (they're the same package)
```

**Why**: lightning and pytorch-lightning are duplicates; pip gets confused

---

### Bonus 3: Remove Old fsspec Pin (1 line in requirements.txt)

```bash
# Line 71 in requirements.txt - remove this line entirely
- fsspec==2023.12.2
# Let torch/lightning pull the version they need
```

**Why**: This was causing issues in previous commits (see git history). Let the dependency resolver pick a compatible version.

---

## Step-by-Step Execution

### Step 1: Verify Current State

```bash
cd /path/to/multimodal_insight_engine

# Check current versions
grep -n "starlette==\|numpy==\|lightning==\|fsspec==" requirements.txt
grep "install_requires" setup.py -A 5

# Expected output:
# 52:lightning==2.0.0  (duplicate)
# 71:fsspec==2023.12.2 (old)
# 173:numpy==1.24.3    (py3.12 incompatible)
# 260:starlette==0.22.0 (fastapi conflict)
```

---

### Step 2: Apply Changes

**Option A: Manual Edit**

```bash
# Edit 1: requirements.txt line 173
# nano requirements.txt  # Or your editor
# Change: numpy==1.24.3 → numpy==1.26.4

# Edit 2: requirements.txt line 260
# Change: starlette==0.22.0 → starlette==0.38.0

# Edit 3: setup.py lines 7-10
# Replace install_requires list with version from "Change 3" above

# Edit 4 (optional): setup.py line 11
# Change: python_requires=">=3.8" → python_requires=">=3.10"

# Edit 5 (optional): requirements.txt line 52
# Delete: lightning==2.0.0

# Edit 6 (optional): requirements.txt line 71
# Delete: fsspec==2023.12.2
```

**Option B: Command Line (Unix/Mac)**

```bash
# Make Changes 1 & 2
sed -i '' 's/numpy==1.24.3/numpy==1.26.4/' requirements.txt
sed -i '' 's/starlette==0.22.0/starlette==0.38.0/' requirements.txt

# Make Bonus 1 (optional)
sed -i '' 's/python_requires=">=3.8"/python_requires=">=3.10"/' setup.py

# Make Bonus 2 & 3 (optional) - remove lines
sed -i '' '/^lightning==2.0.0$/d' requirements.txt
sed -i '' '/^fsspec==2023.12.2$/d' requirements.txt

# For setup.py (Change 3), use your editor - it's a multi-line change
# Copy the install_requires block from "Change 3" above
```

---

### Step 3: Verify Changes

```bash
# Quick check
echo "=== Verification ==="
echo "NumPy version:"
grep "^numpy==" requirements.txt

echo "Starlette version:"
grep "^starlette==" requirements.txt

echo "setup.py first install_requires line:"
grep -A 2 "install_requires" setup.py | head -3

# Should show:
# numpy==1.26.4
# starlette==0.38.0
# And setup.py with PyYAML in list
```

---

### Step 4: Test Python 3.10

```bash
# Create isolated test environment
python3.10 -m venv test-py310

# Activate
source test-py310/bin/activate  # Mac/Linux
# or: test-py310\Scripts\activate  # Windows

# Install updated dependencies
pip install -r requirements.txt

# Test imports
python << 'EOF'
try:
    import fastapi
    import starlette
    import numpy
    import yaml
    import torch
    import transformers
    print("✓ Python 3.10: All imports successful")
    print(f"  fastapi: {fastapi.__version__}")
    print(f"  starlette: {starlette.__version__}")
    print(f"  numpy: {numpy.__version__}")
except Exception as e:
    print(f"✗ Python 3.10: Import failed: {e}")
    exit(1)
EOF

# If that worked, continue:
python -m pytest tests/ -v --tb=short -x  # Run subset of tests

# Clean up
deactivate
rm -rf test-py310
```

---

### Step 5: Test Python 3.11

```bash
# Same as Step 4, but with python3.11
python3.11 -m venv test-py311
source test-py311/bin/activate
pip install -r requirements.txt

python << 'EOF'
import fastapi, starlette, numpy, yaml
print("✓ Python 3.11: All imports successful")
EOF

python -m pytest tests/ -v --tb=short -x

deactivate
rm -rf test-py311
```

---

### Step 6: Test Python 3.12 (The Critical One)

```bash
# This is where the original failure occurred
python3.12 -m venv test-py312
source test-py312/bin/activate
pip install -r requirements.txt

# Watch for the numpy build - should use wheels now, not source build
# You should see: "Using cached numpy-1.26.4-cp312-..." (wheel, fast)
# NOT: "Building wheel for numpy..." (source, slow)

python << 'EOF'
import fastapi, starlette, numpy, yaml
print("✓ Python 3.12: All imports successful")
print(f"  numpy: {numpy.__version__}")  # Should be 1.26.4
EOF

python -m pytest tests/ -v --tb=short -x

deactivate
rm -rf test-py312
```

---

### Step 7: Test Package Build

```bash
# This mimics what GitHub Actions does
pip install build
python -m build

# Check if wheel was created
ls -lh dist/

# Install from wheel
pip install dist/multimodal_insight_engine-0.1.0-py3-none-any.whl

# Test package import (was failing before)
python -c "import src; print('✓ Package installs and imports successfully')"
```

---

### Step 8: Commit Changes

```bash
# Stage changes
git add requirements.txt setup.py

# Review what changed
git diff --staged

# Commit with message
git commit -m "fix: Resolve CI failures across all Python versions (3.10, 3.11, 3.12)

- Update starlette from 0.22.0 to 0.38.0 (fixes fastapi 0.115.0 compatibility)
- Update numpy from 1.24.3 to 1.26.4 (fixes Python 3.12 build failure)
- Add core runtime dependencies to setup.py install_requires
- Remove duplicate 'lightning' package (was 'pytorch-lightning')
- Remove old fsspec pin to allow dependency resolver to work
- Update python_requires from >=3.8 to >=3.10 (matches actual minimum)

Fixes #XX (if you have an issue number)"

# Verify commit
git log -1 --stat
```

---

## What Each Fix Solves

| Fix | Breaks | Affects Python | Solution |
|-----|--------|-----------------|----------|
| starlette 0.38.0 | fastapi conflict | 3.10, 3.11 | Line 260 of requirements.txt |
| numpy 1.26.4 | py3.12 build | 3.12 | Line 173 of requirements.txt |
| setup.py core deps | yaml missing | All | setup.py lines 7-10 |
| python_requires >=3.10 | Mismatch | Documentation | setup.py line 11 |
| Remove duplicate lightning | Confusion | All | requirements.txt line 52 |
| Remove fsspec pin | Version conflict | All | requirements.txt line 71 |

---

## Expected CI Result After Fixes

```
Before:
├─ Python 3.10: ✗ FAILED (starlette conflict)
├─ Python 3.11: ✗ FAILED (starlette conflict)
├─ Python 3.12: ✗ FAILED (numpy build error)
└─ Build: ✗ FAILED (yaml missing)

After:
├─ Python 3.10: ✓ PASSED
├─ Python 3.11: ✓ PASSED
├─ Python 3.12: ✓ PASSED (THIS IS THE CRITICAL WIN)
└─ Build: ✓ PASSED
```

---

## Troubleshooting

### "ERROR: Cannot build wheel for numpy"

**Symptom**: `ERROR: Failed to build 'numpy' when getting requirements to build wheel`

**Cause**: requirements.txt still has numpy==1.24.3

**Fix**:
```bash
grep "^numpy==" requirements.txt  # Should show 1.26.4
sed -i '' 's/numpy==1.24.3/numpy==1.26.4/' requirements.txt
```

---

### "ModuleNotFoundError: No module named 'yaml'"

**Symptom**: `python -c "import src"` fails with yaml missing

**Cause**: setup.py not updated with core dependencies

**Fix**: Ensure your setup.py has PyYAML in install_requires
```bash
grep "PyYAML" setup.py  # Should exist
```

---

### "starlette<0.39.0,>=0.37.2 constraint not satisfied"

**Symptom**: Pip resolver says starlette conflict

**Cause**: requirements.txt still has starlette==0.22.0

**Fix**:
```bash
grep "^starlette==" requirements.txt  # Should show 0.38.0
sed -i '' 's/starlette==0.22.0/starlette==0.38.0/' requirements.txt
```

---

### Testing Hangs on Python 3.12

**Symptom**: Tests start but seem to hang indefinitely

**Cause**: Likely torch/GPU initialization issue (not your dependency fix)

**Fix**: Run with CPU-only torch
```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

---

## Rollback (If Needed)

If something goes wrong and you need to revert:

```bash
# Option 1: Undo last commit
git reset --hard HEAD~1

# Option 2: Manual rollback
git checkout requirements.txt setup.py
```

But this is very unlikely - the changes are straightforward upgrades.

---

## Double-Check Checklist

Before considering the fixes complete:

- [ ] Line 173: `numpy==1.26.4` (was 1.24.3)
- [ ] Line 260: `starlette==0.38.0` (was 0.22.0)
- [ ] setup.py: Has PyYAML in install_requires
- [ ] setup.py: Has pydantic, fastapi, starlette, torch, transformers, lightning
- [ ] Line 52: `lightning==2.0.0` removed (duplicate)
- [ ] Line 71: `fsspec==2023.12.2` removed (optional but recommended)
- [ ] Line 11 setup.py: `python_requires=">=3.10"` (optional but recommended)
- [ ] Python 3.10 test: ✓ Passes
- [ ] Python 3.11 test: ✓ Passes
- [ ] Python 3.12 test: ✓ Passes
- [ ] Package build: ✓ Passes
- [ ] Package import: ✓ Works

---

## Next Steps

1. **Apply the fixes** (Steps 1-3 above)
2. **Test locally** (Steps 4-6 above)
3. **Test package build** (Step 7 above)
4. **Commit changes** (Step 8 above)
5. **Push and watch CI pass** (GitHub Actions should all turn green)

---

## Support

For any issues:
1. Re-read the "Troubleshooting" section above
2. Review the detailed analysis in `CI_FAILURE_ANALYSIS.md`
3. Check compatibility details in `DEPENDENCY_VERIFICATION.md`

The three documents form a complete troubleshooting reference:
- `CI_FAILURE_ANALYSIS.md` - Why failures happened
- `DEPENDENCY_VERIFICATION.md` - Why proposed versions are safe
- `QUICK_FIX_GUIDE.md` - How to apply fixes (this file)
