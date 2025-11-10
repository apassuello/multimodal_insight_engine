#!/usr/bin/env python3
"""
Script to add weights_only=True to all torch.load() calls for security.

SECURITY FIX: Prevents arbitrary code execution from malicious model files.
"""

import re
import sys
from pathlib import Path


def fix_torch_load_in_file(file_path):
    """
    Add weights_only=True to all torch.load() calls in a file.

    Args:
        file_path: Path to the Python file

    Returns:
        bool: True if file was modified, False otherwise
    """
    with open(file_path, 'r') as f:
        content = f.read()

    original_content = content

    # Pattern 1: torch.load with map_location
    pattern1 = r'torch\.load\(([^,]+),\s*map_location=([^)]+)\)(?!\s*,\s*weights_only)'
    replacement1 = r'torch.load(\1, map_location=\2, weights_only=True)'
    content = re.sub(pattern1, replacement1, content)

    # Pattern 2: torch.load with just path (no map_location)
    pattern2 = r'torch\.load\(([^)]+)\)(?!\s*,\s*weights_only)'
    # Only replace if it doesn't already have weights_only
    def replace_simple(match):
        arg = match.group(1)
        # Check if this is already a full call with weights_only
        if 'weights_only' in arg:
            return match.group(0)
        # Check if it already has map_location (already handled by pattern1)
        if 'map_location' in arg:
            return match.group(0)
        return f'torch.load({arg}, weights_only=True)'

    content = re.sub(pattern2, replace_simple, content)

    # Write back if modified
    if content != original_content:
        with open(file_path, 'w') as f:
            f.write(content)
        return True

    return False


def main():
    """Find and fix all torch.load() calls in the codebase."""
    src_dir = Path('src')

    if not src_dir.exists():
        print("Error: src/ directory not found")
        sys.exit(1)

    # Find all Python files
    python_files = list(src_dir.rglob('*.py'))

    modified_files = []
    for file_path in python_files:
        try:
            if fix_torch_load_in_file(file_path):
                modified_files.append(file_path)
                print(f"✅ Fixed: {file_path}")
        except Exception as e:
            print(f"❌ Error fixing {file_path}: {e}")

    print(f"\nFixed {len(modified_files)} files")

    # Verify no unsafe torch.load() calls remain
    import subprocess
    result = subprocess.run(
        ['grep', '-rn', 'torch.load', 'src/', '--include=*.py'],
        capture_output=True,
        text=True
    )

    if result.returncode == 0:
        lines = result.stdout.strip().split('\n')
        unsafe = [line for line in lines if 'weights_only' not in line]

        if unsafe:
            print(f"\n⚠️  WARNING: Found {len(unsafe)} unsafe torch.load() calls:")
            for line in unsafe:
                print(f"  {line}")
        else:
            print("\n✅ All torch.load() calls are now safe!")


if __name__ == '__main__':
    main()
