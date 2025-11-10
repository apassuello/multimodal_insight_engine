# Save as gpu_diagnostic.py
import sys
import subprocess
import platform

print(f"Python version: {platform.python_version()}")
print(f"System: {platform.system()} {platform.release()}")

# Try to import torch
try:
    import torch
    print(f"PyTorch version: {torch.__version__}")
    
    # Check if CUDA/ROCm is available
    if torch.cuda.is_available():
        print("CUDA is available")
        print(f"CUDA version: {torch.version.cuda}")
        print(f"Number of devices: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            print(f"Device {i}: {torch.cuda.get_device_name(i)}")
    else:
        print("CUDA is not available")
    
    # Check specifically for ROCm
    if hasattr(torch.version, 'hip') and torch.version.hip:
        print(f"ROCm version: {torch.version.hip}")
    else:
        print("ROCm version information not found")

except ImportError as e:
    print(f"Error importing PyTorch: {e}")

# Try to get GPU information from system
try:
    print("\nSystem GPU Information:")
    # SECURITY FIX: Don't use shell=True - use Python filtering instead
    lspci_result = subprocess.run(['lspci'],
                                  capture_output=True,
                                  text=True,
                                  check=False,
                                  timeout=10)
    # Filter for VGA in Python instead of using shell pipes
    gpu_lines = [line for line in lspci_result.stdout.splitlines()
                 if 'vga' in line.lower()]
    print('\n'.join(gpu_lines) if gpu_lines else "No VGA devices found")
except Exception as e:
    print(f"Error getting GPU info: {e}")

# Check ROCm installation
try:
    print("\nROCm Information:")
    # SECURITY FIX: Remove shell=True to prevent command injection
    rocm_info = subprocess.run(['rocminfo'],
                               capture_output=True,
                               text=True,
                               check=False,
                               timeout=10)
    # Print just a summary to avoid overwhelming output
    if rocm_info.stdout:
        print("rocminfo available - first few lines:")
        print('\n'.join(rocm_info.stdout.split('\n')[:10]))
    else:
        print("rocminfo returned no output")
    print(f"ROCm error output: {rocm_info.stderr}")
except Exception as e:
    print(f"Error running rocminfo: {e}")