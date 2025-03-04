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
    gpu_info = subprocess.run(['lspci', '|', 'grep', '-i', 'vga'], 
                             shell=True, text=True, capture_output=True)
    print(gpu_info.stdout)
except Exception as e:
    print(f"Error getting GPU info: {e}")

# Check ROCm installation
try:
    print("\nROCm Information:")
    rocm_info = subprocess.run(['rocminfo'], shell=True, text=True, capture_output=True)
    # Print just a summary to avoid overwhelming output
    if rocm_info.stdout:
        print("rocminfo available - first few lines:")
        print('\n'.join(rocm_info.stdout.split('\n')[:10]))
    else:
        print("rocminfo returned no output")
    print(f"ROCm error output: {rocm_info.stderr}")
except Exception as e:
    print(f"Error running rocminfo: {e}")