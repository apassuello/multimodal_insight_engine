# Getting Started with MultiModal Insight Engine

Welcome! This guide will help you set up and start using the MultiModal Insight Engine, including the Constitutional AI Interactive Demo.

## Table of Contents

- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Verification](#verification)
- [Quick Start](#quick-start)
- [Running the Constitutional AI Demo](#running-the-constitutional-ai-demo)
- [Running Tests](#running-tests)
- [Common Issues](#common-issues)
- [Next Steps](#next-steps)

## Prerequisites

Before you begin, ensure you have the following installed:

### Required

- **Python 3.8 or higher** (3.10+ recommended)
  ```bash
  python --version  # Check your Python version
  ```

- **pip** (Python package installer)
  ```bash
  pip --version
  ```

- **Git** (for cloning the repository)
  ```bash
  git --version
  ```

### Recommended

- **CUDA Toolkit** (for GPU acceleration, optional)
  - CUDA 11.8 or higher for PyTorch 2.0+
  - Check compatibility: https://pytorch.org/get-started/locally/

- **8GB+ RAM** (16GB recommended for training)
- **10GB+ free disk space**

### Operating Systems

- Linux (Ubuntu 20.04+, tested)
- macOS (10.15+, with Metal GPU support)
- Windows 10/11 (with WSL2 recommended)

## Installation

### Step 1: Clone the Repository

```bash
# Clone the repository
git clone https://github.com/yourusername/multimodal_insight_engine.git
cd multimodal_insight_engine
```

### Step 2: Create a Virtual Environment

**Linux/macOS:**
```bash
python -m venv venv
source venv/bin/activate
```

**Windows:**
```cmd
python -m venv venv
venv\Scripts\activate
```

**Windows (PowerShell):**
```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

You should see `(venv)` in your terminal prompt indicating the virtual environment is active.

### Step 3: Install Dependencies

```bash
# Upgrade pip to latest version
pip install --upgrade pip

# Install project dependencies
pip install -r requirements.txt

# Install the package in development mode
pip install -e .
```

**GPU Support (Optional):**

If you have a CUDA-compatible GPU:

```bash
# Verify CUDA availability
python -c "import torch; print(torch.cuda.is_available())"

# If False, you may need to reinstall PyTorch with CUDA support:
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

**Apple Silicon (M1/M2/M3) Support:**

PyTorch has Metal Performance Shaders (MPS) backend support:

```bash
# Verify MPS availability
python -c "import torch; print(torch.backends.mps.is_available())"
```

## Verification

### Verify Installation

Run the installation verification script:

```bash
python verify_install.py
```

**Expected output:**
```
✓ Python version: 3.10.x (OK)
✓ PyTorch installed: 2.x.x
✓ GPU available: True (CUDA 11.8) / False (CPU only)
✓ All core modules imported successfully
✓ Installation complete!
```

### Quick Functionality Check

Test that the core components work:

```python
python -c "
from src.safety.constitutional.framework import ConstitutionalFramework
from demo.managers.model_manager import ModelManager

print('✓ Core imports successful')
print('✓ Installation verified')
"
```

## Quick Start

### Option 1: Interactive Demo (Recommended for Beginners)

Launch the Constitutional AI Interactive Demo:

```bash
# From the project root directory
python demo/main.py
```

This will:
1. Start a Gradio web interface
2. Open your browser to `http://localhost:7860`
3. Provide a visual interface for all features

See the [User Guide](docs/USER_GUIDE.md) for detailed instructions on using the demo.

### Option 2: Python API

Use the framework programmatically:

```python
from src.safety.constitutional.framework import ConstitutionalFramework

# Initialize the framework
framework = ConstitutionalFramework()

# Evaluate text against principles
result = framework.evaluate(
    "Your input text here",
    principles=["harm_prevention", "fairness", "truthfulness"]
)

print(f"Alignment score: {result['alignment_score']:.2f}")
print(f"Flagged principles: {result['flagged_principles']}")
```

### Option 3: Command-Line Demos

Run specific demo scripts:

```bash
# Language modeling demo
python demos/language_model_demo.py --dataset wikitext --model_config small

# Translation demo
python demos/translation_example.py --src_lang de --tgt_lang en

# Red teaming demo
python demos/red_teaming_demo.py --model phi-2 --verbose

# Safety evaluation demo
python demos/demo_safety.py --model_name phi-2

# Model optimization demo
python demos/model_optimization_demo.py --technique pruning
```

## Running the Constitutional AI Demo

The Constitutional AI Interactive Demo is the main interface for exploring constitutional AI concepts.

### Starting the Demo

```bash
python demo/main.py
```

**Command-line options:**

```bash
# Specify server port
python demo/main.py --port 7860

# Run on all network interfaces (accessible from other devices)
python demo/main.py --server_name 0.0.0.0

# Enable debug mode
python demo/main.py --debug
```

### Demo Workflow

The demo has 5 main tabs:

1. **Setup Tab**: Load base and trained models, select device (CPU/GPU/MPS)
2. **Training Tab**: Configure and run Constitutional AI training
3. **Evaluation Tab**: Test single prompts against principles
4. **Impact Tab**: Compare model performance on test suites
5. **Architecture Tab**: Explore system documentation and examples

**Typical workflow:**
1. Load models in Setup tab
2. (Optional) Train a model in Training tab
3. Evaluate responses in Evaluation tab
4. Analyze impact with test suites in Impact tab

For detailed instructions, see the [User Guide](docs/USER_GUIDE.md).

## Running Tests

### Run All Tests

```bash
# Run complete test suite with coverage
./run_tests.sh
```

### Run Specific Tests

```bash
# Run Constitutional AI tests
pytest tests/test_framework.py tests/test_evaluator.py -v

# Run comparison engine tests
pytest tests/test_comparison_engine.py -v

# Run with coverage report
pytest tests/ --cov=src --cov=demo --cov-report=html

# View coverage report
open htmlcov/index.html  # macOS
xdg-open htmlcov/index.html  # Linux
start htmlcov/index.html  # Windows
```

### Linting and Type Checking

```bash
# Run linting
flake8 src/ tests/ demo/

# Run type checking
mypy src/ tests/ demo/
```

## Common Issues

### Issue: Import Errors

**Problem:**
```python
ModuleNotFoundError: No module named 'src'
```

**Solution:**
```bash
# Make sure you installed in development mode
pip install -e .

# Or add project root to PYTHONPATH
export PYTHONPATH="${PYTHONPATH}:$(pwd)"  # Linux/macOS
set PYTHONPATH=%PYTHONPATH%;%CD%  # Windows
```

### Issue: CUDA Out of Memory

**Problem:**
```
RuntimeError: CUDA out of memory
```

**Solutions:**
1. **Use smaller model:** Select a smaller model in the demo (e.g., `gpt2` instead of `gpt2-large`)
2. **Reduce batch size:** Lower the batch size in training configuration
3. **Use CPU:** Switch device to CPU in the Setup tab
4. **Clear cache:**
   ```python
   import torch
   torch.cuda.empty_cache()
   ```

### Issue: Gradio Demo Won't Start

**Problem:**
```
OSError: [Errno 48] Address already in use
```

**Solutions:**
```bash
# Port 7860 is already in use, try a different port
python demo/main.py --port 7861

# Or kill the process using port 7860
lsof -ti:7860 | xargs kill -9  # macOS/Linux
netstat -ano | findstr :7860  # Windows (then kill by PID)
```

### Issue: Slow Model Loading

**Problem:** Model loading takes several minutes

**Solutions:**
1. **First-time download:** Models are downloaded from HuggingFace on first use (normal)
2. **Check cache:** Models are cached in `~/.cache/huggingface/`
3. **Use smaller model:** Try `gpt2` (124M params) instead of `gpt2-large` (774M params)

### Issue: PyTorch Not Found

**Problem:**
```
ModuleNotFoundError: No module named 'torch'
```

**Solution:**
```bash
# Reinstall PyTorch
pip install torch torchvision torchaudio

# For GPU support (CUDA 11.8):
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### Issue: Permission Denied (Linux/macOS)

**Problem:**
```bash
./run_tests.sh: Permission denied
```

**Solution:**
```bash
# Make the script executable
chmod +x run_tests.sh

# Then run it
./run_tests.sh
```

## Next Steps

Now that you have the system running, explore:

### 📖 Documentation

- **[User Guide](docs/USER_GUIDE.md)** - Comprehensive demo walkthrough
- **[Architecture](ARCHITECTURE.md)** - System design and components
- **[API Reference](docs/API_REFERENCE.md)** - Developer documentation
- **[Contributing](CONTRIBUTING.md)** - How to contribute

### 🚀 Tutorials

- **[Constitutional AI Concepts](docs/constitutional-ai/README.md)** - Learn CAI principles
- **[Training Guide](docs/constitutional-ai/PPO_IMPLEMENTATION_GUIDE.md)** - Advanced training
- **[Demo Guide](docs/reference/DEMO_GUIDE.md)** - Demo features

### 🔬 Experiments

Try these experiments:

1. **Train a model** on harmful content test suite
2. **Compare performance** before/after training
3. **Evaluate custom prompts** against principles
4. **Export results** to CSV for analysis
5. **Explore red teaming** techniques

### 🤝 Community

- **Report issues:** [GitHub Issues](https://github.com/yourusername/multimodal_insight_engine/issues)
- **Contribute:** See [CONTRIBUTING.md](CONTRIBUTING.md)
- **Discuss:** Create a discussion on GitHub

### 📊 Example Projects

Build on this framework:

1. **Custom principles:** Define your own constitutional principles
2. **Fine-tuning:** Train models on domain-specific data
3. **Evaluation:** Create custom test suites
4. **Integration:** Integrate into your applications

## Troubleshooting Resources

If you encounter issues not covered here:

1. **Check documentation:** [docs/](docs/) directory
2. **Search issues:** [GitHub Issues](https://github.com/yourusername/multimodal_insight_engine/issues)
3. **Create an issue:** Provide system info, error messages, and steps to reproduce
4. **Security issues:** See [SECURITY.md](SECURITY.md) for private reporting

## System Requirements Summary

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| Python | 3.8+ | 3.10+ |
| RAM | 8GB | 16GB+ |
| Disk Space | 5GB | 10GB+ |
| GPU VRAM | N/A (CPU) | 8GB+ |
| OS | Ubuntu 20.04, macOS 10.15, Windows 10 | Ubuntu 22.04, macOS 13+, Windows 11 |

## Quick Reference Commands

```bash
# Activate virtual environment
source venv/bin/activate  # Linux/macOS
venv\Scripts\activate  # Windows

# Run demo
python demo/main.py

# Run tests
./run_tests.sh

# Run linting
flake8 src/ tests/ demo/

# Check coverage
pytest tests/ --cov=src --cov=demo --cov-report=html

# Deactivate virtual environment
deactivate
```

---

**Ready to get started?** Launch the demo:

```bash
python demo/main.py
```

Then open your browser to `http://localhost:7860` and explore!

For detailed usage instructions, see the [User Guide](docs/USER_GUIDE.md).
