#!/usr/bin/env python
"""
Installation Verification Script

This script verifies that the MultiModal Insight Engine is correctly installed
and all dependencies are available.

Usage:
    python verify_install.py

Exit codes:
    0 - All checks passed
    1 - Some checks failed
"""

import sys
import importlib
from pathlib import Path
from typing import List, Tuple, Optional

# Color codes for terminal output
class Colors:
    GREEN = '\033[92m'
    RED = '\033[91m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    RESET = '\033[0m'
    BOLD = '\033[1m'

def colored(text: str, color: str) -> str:
    """Add color to text for terminal output."""
    return f"{color}{text}{Colors.RESET}"

def check_import(module_name: str, package_name: Optional[str] = None) -> Tuple[bool, str]:
    """
    Check if a module can be imported.

    Args:
        module_name: The module to import (e.g., 'torch')
        package_name: Optional display name (e.g., 'PyTorch'). Defaults to module_name.

    Returns:
        Tuple of (success: bool, message: str)
    """
    display_name = package_name or module_name
    try:
        mod = importlib.import_module(module_name)
        version = getattr(mod, '__version__', 'unknown')
        return True, f"{display_name} ({version})"
    except ImportError as e:
        return False, f"{display_name} - {str(e)}"

def check_file(path: Path, description: str) -> Tuple[bool, str]:
    """
    Check if a file exists.

    Args:
        path: Path to check
        description: Description of the file

    Returns:
        Tuple of (success: bool, message: str)
    """
    if path.exists():
        size = path.stat().st_size
        return True, f"{description} ({size} bytes)"
    else:
        return False, f"{description} - NOT FOUND at {path}"

def check_directory(path: Path, description: str) -> Tuple[bool, str]:
    """
    Check if a directory exists.

    Args:
        path: Path to check
        description: Description of the directory

    Returns:
        Tuple of (success: bool, message: str)
    """
    if path.is_dir():
        file_count = len(list(path.glob('*.py')))
        return True, f"{description} ({file_count} Python files)"
    else:
        return False, f"{description} - NOT FOUND at {path}"

def print_check(success: bool, message: str) -> None:
    """Print a check result with appropriate coloring."""
    if success:
        print(f"  {colored('✓', Colors.GREEN)} {message}")
    else:
        print(f"  {colored('✗', Colors.RED)} {message}")

def print_section(title: str) -> None:
    """Print a section header."""
    print(f"\n{colored(title, Colors.BOLD + Colors.BLUE)}")
    print(f"  {'-' * 50}")

def main() -> int:
    """
    Run all installation checks.

    Returns:
        0 if all checks pass, 1 if any fail
    """
    print(f"\n{colored('MultiModal Insight Engine - Installation Verification', Colors.BOLD)}")
    print(f"{colored('=' * 60, Colors.BLUE)}\n")

    checks_passed = 0
    checks_failed = 0
    all_results = []

    # Section 1: Python Version
    print_section("1. Python Environment")

    version_info = sys.version_info
    version_string = f"{version_info.major}.{version_info.minor}.{version_info.micro}"
    required_version = (3, 8)

    if version_info >= required_version:
        print_check(True, f"Python {version_string}")
        checks_passed += 1
    else:
        print_check(False, f"Python {version_string} (requires {required_version[0]}.{required_version[1]}+)")
        checks_failed += 1

    python_path = Path(sys.executable)
    print_check(python_path.exists(), f"Python executable: {sys.executable}")
    if python_path.exists():
        checks_passed += 1
    else:
        checks_failed += 1

    # Section 2: Core Dependencies
    print_section("2. Core Dependencies")

    core_imports = [
        ('torch', 'PyTorch'),
        ('transformers', 'Transformers'),
        ('numpy', 'NumPy'),
        ('pandas', 'Pandas'),
    ]

    for module, name in core_imports:
        success, msg = check_import(module, name)
        print_check(success, msg)
        if success:
            checks_passed += 1
        else:
            checks_failed += 1

    # Section 3: Development Dependencies
    print_section("3. Development Dependencies")

    dev_imports = [
        ('pytest', 'pytest'),
        ('pytest_cov', 'pytest-cov'),
        ('mypy', 'mypy'),
        ('flake8', 'flake8'),
    ]

    for module, name in dev_imports:
        success, msg = check_import(module, name)
        print_check(success, msg)
        if success:
            checks_passed += 1
        else:
            checks_failed += 1

    # Section 4: Project Structure
    print_section("4. Project Structure")

    repo_root = Path(__file__).parent

    directories = [
        (repo_root / 'src', 'src/ directory'),
        (repo_root / 'tests', 'tests/ directory'),
        (repo_root / 'demos', 'demos/ directory'),
        (repo_root / 'docs', 'docs/ directory'),
    ]

    for path, description in directories:
        success, msg = check_directory(path, description)
        print_check(success, msg)
        if success:
            checks_passed += 1
        else:
            checks_failed += 1

    # Section 5: Key Files
    print_section("5. Configuration Files")

    files = [
        (repo_root / 'setup.py', 'setup.py'),
        (repo_root / 'requirements.txt', 'requirements.txt'),
        (repo_root / '.coveragerc', '.coveragerc'),
        (repo_root / 'run_tests.sh', 'run_tests.sh'),
    ]

    for path, description in files:
        success, msg = check_file(path, description)
        print_check(success, msg)
        if success:
            checks_passed += 1
        else:
            checks_failed += 1

    # Section 6: Project Imports
    print_section("6. Project Imports")

    # Add repo to path for imports
    sys.path.insert(0, str(repo_root))

    project_imports = [
        ('src.models', 'Models module'),
        ('src.data', 'Data module'),
        ('src.training', 'Training module'),
        ('src.safety', 'Safety module'),
        ('src.utils', 'Utils module'),
    ]

    for module, name in project_imports:
        success, msg = check_import(module, name)
        print_check(success, msg)
        if success:
            checks_passed += 1
        else:
            checks_failed += 1

    # Section 7: Test Discovery
    print_section("7. Test Infrastructure")

    test_dir = repo_root / 'tests'
    test_files = list(test_dir.glob('test_*.py'))

    if test_files:
        print_check(True, f"Found {len(test_files)} test files")
        checks_passed += 1
    else:
        print_check(False, "No test files found")
        checks_failed += 1

    # Check if pytest can discover tests
    try:
        import pytest
        success = True
        msg = "pytest test discovery available"
    except ImportError:
        success = False
        msg = "pytest not available (needed for testing)"

    print_check(success, msg)
    if success:
        checks_passed += 1
    else:
        checks_failed += 1

    # Summary Section
    print_section("SUMMARY")

    total = checks_passed + checks_failed
    percentage = (checks_passed / total * 100) if total > 0 else 0

    print(f"\n  Total Checks: {total}")
    print(f"  Passed: {colored(str(checks_passed), Colors.GREEN)}")
    print(f"  Failed: {colored(str(checks_failed), Colors.RED)}")
    print(f"  Success Rate: {percentage:.1f}%")

    # Final verdict
    print()
    if checks_failed == 0:
        print(colored(
            "✓ Installation verified! All checks passed.",
            Colors.GREEN + Colors.BOLD
        ))
        print(colored(
            "  You're ready to start using MultiModal Insight Engine!",
            Colors.GREEN
        ))
        status = 0
    else:
        print(colored(
            f"✗ {checks_failed} check(s) failed. Please review above.",
            Colors.RED + Colors.BOLD
        ))
        print(colored(
            "  See GETTING_STARTED.md for help resolving issues.",
            Colors.YELLOW
        ))
        status = 1

    print()
    return status

if __name__ == '__main__':
    sys.exit(main())
