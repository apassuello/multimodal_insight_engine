#!/bin/bash
# Startup script for Constitutional AI Demo
# Includes health checks and graceful error handling

set -e

echo "========================================"
echo "Constitutional AI Interactive Demo"
echo "========================================"
echo ""

# Function to check if required directories exist
check_directories() {
    echo "Checking required directories..."
    for dir in demo/checkpoints demo/logs demo/exports; do
        if [ ! -d "$dir" ]; then
            echo "Creating directory: $dir"
            mkdir -p "$dir"
        fi
    done
    echo "✓ Directories ready"
    echo ""
}

# Function to check Python dependencies
check_dependencies() {
    echo "Checking Python dependencies..."
    python -c "import gradio; import torch; import transformers" 2>/dev/null
    if [ $? -eq 0 ]; then
        echo "✓ Dependencies installed"
    else
        echo "✗ Missing dependencies"
        exit 1
    fi
    echo ""
}

# Function to check available device
check_device() {
    echo "Checking available compute device..."
    python -c "
import torch
if torch.cuda.is_available():
    print('✓ CUDA available:', torch.cuda.get_device_name(0))
elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
    print('✓ MPS available (Apple Silicon)')
else:
    print('✓ CPU only (CUDA/MPS not available)')
"
    echo ""
}

# Function to display configuration
show_config() {
    echo "Configuration:"
    echo "  Server: ${GRADIO_SERVER_NAME:-0.0.0.0}:${GRADIO_SERVER_PORT:-7860}"
    echo "  Device: ${DEVICE_PREFERENCE:-auto}"
    echo "  Model: ${DEFAULT_MODEL:-gpt2}"
    echo ""
}

# Main startup sequence
main() {
    check_directories
    check_dependencies
    check_device
    show_config

    echo "Starting Gradio application..."
    echo "========================================"
    echo ""

    # Start the demo with error handling
    exec python -m demo.main \
        --server-name "${GRADIO_SERVER_NAME:-0.0.0.0}" \
        --server-port "${GRADIO_SERVER_PORT:-7860}" \
        --share false
}

# Handle SIGTERM gracefully
trap 'echo "Shutting down gracefully..."; exit 0' SIGTERM SIGINT

# Run main function
main
