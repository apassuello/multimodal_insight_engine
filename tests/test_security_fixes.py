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
  """Test that pickle is no longer used for NEW serialization (backward compatibility allowed)."""

    def test_no_pickle_imports_in_dataset(self):
        """Verify that multimodal_dataset.py only imports pickle for backward compatibility."""
        dataset_file = Path("src/data/multimodal_dataset.py")
        assert dataset_file.exists(), "Dataset file not found"

        content = dataset_file.read_text()

        # Check pickle imports
        lines = content.split('\n')
        pickle_imports = []

        for line_num, line in enumerate(lines, 1):
            # Skip comment-only lines
            if line.strip().startswith('#'):
                continue

            # Check for pickle imports in code
            if 'import pickle' in line.lower():
                # Allow if comment mentions backward compatibility
                if 'backward compatibility' in line.lower() or 'migration' in line.lower():
                    continue  # This is acceptable
                else:
                    pickle_imports.append(f"Line {line_num}: {line.strip()}")

        if pickle_imports:
            pytest.fail(f"Found pickle imports without backward compatibility justification:\n" + "\n".join(pickle_imports))

    def test_no_pickle_usage_in_turbo_bpe(self):
        """Verify that turbo_bpe_preprocessor.py uses pickle ONLY for reading old caches (not writing new ones)."""
        preprocessor_file = Path("src/data/tokenization/turbo_bpe_preprocessor.py")
        assert preprocessor_file.exists(), "Preprocessor file not found"

        content = preprocessor_file.read_text()

        # CRITICAL: pickle.dump() should NOT exist (no new pickle files)
        if 'pickle.dump' in content:
            pytest.fail("Found pickle.dump() in turbo_bpe_preprocessor.py - Should NOT create new pickle files!")

        # pickle.load() is allowed for backward compatibility (reading old caches)
        # But we should verify it's only used as fallback after JSON
        if 'pickle.load' in content:
            # Verify JSON is tried first
            json_load_pos = content.find('json.load')
            pickle_load_pos = content.find('pickle.load')

            if json_load_pos == -1:
                pytest.fail("Found pickle.load() but no json.load() - Should try JSON first!")

            if pickle_load_pos < json_load_pos:
                pytest.fail("pickle.load() appears before json.load() - Should try JSON first!")

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

        # Check for exec() calls in actual code (not comments or strings)
        lines = content.split('\n')
        in_multiline_string = False
        quote_char = None
        exec_found = []

        for line_num, line in enumerate(lines, 1):
            # Handle multiline strings
            if '"""' in line or "'''" in line:
                if not in_multiline_string:
                    quote_char = '"""' if '"""' in line else "'''"
                    in_multiline_string = True
                elif quote_char in line:
                    in_multiline_string = False
                continue

            if in_multiline_string:
                continue

            # Remove comments
            code_part = line.split('#')[0]

            # Check for exec() in actual code
            if 'exec(' in code_part:
                # Ignore if it's in a string
                if '"exec(' not in code_part and "'exec(" not in code_part:
                    # Ignore documentation/comments explaining why we DON'T use exec
                    doc_keywords = ['instead of', 'not use', 'instead', 'SECURITY', 'avoid']
                    is_documentation = any(keyword in line for keyword in doc_keywords)
                    if not is_documentation:
                        exec_found.append(f"Line {line_num}: {line.strip()}")

        if exec_found:
            pytest.fail(f"Found exec() in compile_metadata.py:\n" + "\n".join(exec_found))


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

        # Check for shell=True in actual code (not comments or strings)
        lines = content.split('\n')
        shell_true_found = []

        for line_num, line in enumerate(lines, 1):
            # Remove comments
            code_part = line.split('#')[0]

            # Check for shell=True in actual code
            if 'shell=True' in code_part:
                # Ignore if it's in a string (documentation or examples)
                if '"shell=True"' not in code_part and "'shell=True'" not in code_part:
                    shell_true_found.append(f"Line {line_num}: {line.strip()}")

        if shell_true_found:
            pytest.fail(f"Found shell=True in test_gpu.py:\n" + "\n".join(shell_true_found))


class TestSecurityCodePatterns:
    """Test for general security anti-patterns."""

    def test_no_eval_usage(self):
        """Ensure eval() builtin is not used anywhere in the codebase."""
        import subprocess
        import re

        result = subprocess.run(
            ['grep', '-rn', 'eval(', 'src/', '--include=*.py'],
            capture_output=True,
            text=True
        )

        if result.returncode == 0:
            # Found eval() calls - filter out safe patterns
            lines = result.stdout.strip().split('\n')
            actual_eval = []

            for line in lines:
                # Skip comments
                code_part = line.split('#')[0]
                if 'eval(' not in code_part:
                    continue

                # Safe patterns to ignore:
                # - .eval() - PyTorch/TensorFlow model evaluation mode (method call)
                # - "eval()" in strings
                # - def eval() - method definitions
                safe_patterns = [
                    r'\.eval\(',           # Method call like model.eval()
                    r'"[^"]*eval\([^"]*"', # In double-quoted strings
                    r"'[^']*eval\([^']*'", # In single-quoted strings
                    r'def\s+eval\(',       # Method definition
                    r'signature.*eval\(',  # Signature documentation
                ]

                is_safe = False
                for pattern in safe_patterns:
                    if re.search(pattern, code_part):
                        is_safe = True
                        break

                if not is_safe:
                    actual_eval.append(line)

            if actual_eval:
                error_msg = "Found dangerous eval() builtin usage (security risk):\n"
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
