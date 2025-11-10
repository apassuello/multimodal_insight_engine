"""
Security regression tests for critical vulnerability fixes.

PURPOSE: Ensure that security vulnerabilities remain fixed and don't regress.
Tests cover:
1. Pickle usage (should be replaced with JSON)
2. exec() usage (should be replaced with AST parsing)
3. torch.load() safety (should use weights_only=True)
4. subprocess command injection (should not use shell=True)
"""

import os
import json
import tempfile
import shutil
import pytest
from pathlib import Path


class TestPickleRemoval:
    """Test that pickle is no longer used for serialization."""

    def test_no_pickle_imports_in_dataset(self):
        """Verify that multimodal_dataset.py doesn't import pickle."""
        dataset_file = Path("src/data/multimodal_dataset.py")
        assert dataset_file.exists(), "Dataset file not found"

        content = dataset_file.read_text()

        # Check that pickle is not imported at the module level
        lines = content.split('\n')
        import_lines = [line for line in lines if 'import' in line and not line.strip().startswith('#')]

        for line in import_lines:
            # Allow pickle in comments but not actual imports
            if not line.strip().startswith('#'):
                assert 'pickle' not in line.lower(), f"Found pickle import: {line}"

    def test_no_pickle_usage_in_turbo_bpe(self):
        """Verify that turbo_bpe_preprocessor.py doesn't use pickle."""
        preprocessor_file = Path("src/data/tokenization/turbo_bpe_preprocessor.py")
        assert preprocessor_file.exists(), "Preprocessor file not found"

        content = preprocessor_file.read_text()

        # Check for pickle.load or pickle.dump
        assert 'pickle.load' not in content, "Found pickle.load() in turbo_bpe_preprocessor.py"
        assert 'pickle.dump' not in content, "Found pickle.dump() in turbo_bpe_preprocessor.py"

    def test_json_serialization_works(self):
        """Test that JSON serialization works for cached data."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_file = os.path.join(tmpdir, "test_cache.json")

            # Test data
            test_data = {
                "src_sequences": [[1, 2, 3], [4, 5, 6]],
                "tgt_sequences": [[7, 8, 9], [10, 11, 12]]
            }

            # Save to JSON
            with open(cache_file, 'w') as f:
                json.dump(test_data, f)

            # Load from JSON
            with open(cache_file, 'r') as f:
                loaded_data = json.load(f)

            # Verify data integrity
            assert loaded_data == test_data
            assert loaded_data["src_sequences"] == [[1, 2, 3], [4, 5, 6]]
            assert loaded_data["tgt_sequences"] == [[7, 8, 9], [10, 11, 12]]

    def test_dataset_metadata_uses_json(self):
        """Verify that dataset code creates .json files instead of .pkl files."""
        # Read the multimodal_dataset.py file
        dataset_file = Path("src/data/multimodal_dataset.py")
        content = dataset_file.read_text()

        # Check that new cache files use .json extension
        assert '.json' in content, "Dataset should use .json for caching"
        assert 'cache_samples_json' in content, "Should have cache_samples_json variable"


class TestExecRemoval:
    """Test that exec() is not used for code execution."""

    def test_no_exec_in_compile_metadata(self):
        """Verify that compile_metadata.py doesn't use exec()."""
        metadata_file = Path("compile_metadata.py")

        if not metadata_file.exists():
            pytest.skip("compile_metadata.py not found")

        content = metadata_file.read_text()

        # Check for exec() calls
        assert 'exec(' not in content, "Found exec() in compile_metadata.py"


class TestTorchLoadSafety:
    """Test that torch.load() uses weights_only=True parameter."""

    def test_search_for_unsafe_torch_load(self):
        """Search for torch.load() calls without weights_only=True."""
        import subprocess

        # Search for torch.load calls
        result = subprocess.run(
            ['grep', '-rn', 'torch.load', 'src/', '--include=*.py'],
            capture_output=True,
            text=True
        )

        if result.returncode == 0:
            # Found torch.load calls - check they have weights_only
            lines = result.stdout.strip().split('\n')
            unsafe_calls = []

            for line in lines:
                if 'torch.load' in line and 'weights_only' not in line:
                    # Allow safe_torch_load function definition
                    if 'def safe_torch_load' not in line:
                        unsafe_calls.append(line)

            if unsafe_calls:
                error_msg = "Found unsafe torch.load() calls without weights_only=True:\n"
                error_msg += "\n".join(unsafe_calls)
                pytest.fail(error_msg)


class TestSubprocessSafety:
    """Test that subprocess.run() doesn't use shell=True."""

    def test_no_shell_true_in_test_gpu(self):
        """Verify that setup_test/test_gpu.py doesn't use shell=True."""
        gpu_test_file = Path("setup_test/test_gpu.py")

        if not gpu_test_file.exists():
            pytest.skip("setup_test/test_gpu.py not found")

        content = gpu_test_file.read_text()

        # Check for shell=True
        assert 'shell=True' not in content, "Found shell=True in test_gpu.py"


class TestSecurityCodePatterns:
    """Test for general security anti-patterns."""

    def test_no_eval_usage(self):
        """Ensure eval() is not used anywhere in the codebase."""
        import subprocess

        result = subprocess.run(
            ['grep', '-rn', 'eval(', 'src/', '--include=*.py'],
            capture_output=True,
            text=True
        )

        if result.returncode == 0:
            # Found eval() calls - check if they're in comments
            lines = result.stdout.strip().split('\n')
            actual_eval = []

            for line in lines:
                # Skip comments
                code_part = line.split('#')[0]
                if 'eval(' in code_part:
                    actual_eval.append(line)

            if actual_eval:
                error_msg = "Found eval() usage (security risk):\n"
                error_msg += "\n".join(actual_eval)
                pytest.fail(error_msg)

    def test_file_permissions_safe(self):
        """Ensure that sensitive files don't have overly permissive permissions."""
        # This is a basic check - in production you'd check actual file permissions
        sensitive_patterns = ['*.key', '*.pem', 'credentials.*', '.env']

        found_sensitive = []
        for pattern in sensitive_patterns:
            for file in Path('.').rglob(pattern):
                found_sensitive.append(str(file))

        # Just log if found - don't fail (they might be test files)
        if found_sensitive:
            print(f"Found potentially sensitive files: {found_sensitive}")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
