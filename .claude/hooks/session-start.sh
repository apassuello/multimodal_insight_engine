#!/bin/bash
set -euo pipefail

# Only run in Claude Code Web environment
if [ "${CLAUDE_CODE_REMOTE:-}" != "true" ]; then
  echo "Not running in Claude Code Web, skipping setup"
  exit 0
fi

echo "🚀 Setting up MultiModal Insight Engine environment..."

# Change to project directory
cd "$CLAUDE_PROJECT_DIR" || exit 1

# Set Python path for imports
echo "📝 Setting PYTHONPATH..."
echo 'export PYTHONPATH="$CLAUDE_PROJECT_DIR:$PYTHONPATH"' >> "$CLAUDE_ENV_FILE"

# Install Python dependencies
echo "📦 Installing Python dependencies (this may take a few minutes)..."
if [ -f requirements.txt ]; then
    # Create a filtered requirements file for Claude Code Web
    # Remove ROCm-specific packages and use standard versions
    grep -v "rocm" requirements.txt | \
    sed 's/torch==2.1.0.*/torch>=2.0.0/' | \
    sed 's/torchaudio==.*/torchaudio>=2.0.0/' | \
    sed 's/torchvision==.*/torchvision>=0.15.0/' | \
    sed 's/tensorflow-rocm.*/tensorflow>=2.14.0/' > /tmp/requirements_web.txt

    # Install core dependencies first (needed for testing and linting)
    echo "Installing core development dependencies..."
    python -m pip install --user pytest pytest-cov flake8 mypy --quiet 2>/dev/null || echo "Core deps partially installed"

    # Install transformers and essential ML packages
    echo "Installing ML/NLP dependencies..."
    python -m pip install --user transformers torch numpy pandas --quiet 2>/dev/null || echo "ML deps partially installed"

    # Try to install remaining dependencies (best effort)
    echo "Installing remaining dependencies (best effort)..."
    python -m pip install --user -r /tmp/requirements_web.txt --quiet 2>/dev/null || {
        echo "Some packages skipped (expected for ROCm-specific or GPU packages)"
    }

    rm -f /tmp/requirements_web.txt
    echo "✅ Dependencies installation completed"
else
    echo "⚠️  Warning: requirements.txt not found"
fi

# Create reports directory for test output
mkdir -p reports

# Verify critical packages are installed
echo "🔍 Verifying installation..."
python -c "import pytest; import torch; import transformers; print('✅ Core packages verified')" 2>/dev/null || {
    echo "⚠️  Warning: Some core packages may not be properly installed"
}

echo "✨ Environment setup complete!"
echo ""
echo "Available commands:"
echo "  - Run tests: python -m pytest tests/ -v"
echo "  - Run linter: flake8 src/ tests/"
echo "  - Type check: mypy src/ tests/"
echo "  - Full test suite: ./run_tests.sh"
