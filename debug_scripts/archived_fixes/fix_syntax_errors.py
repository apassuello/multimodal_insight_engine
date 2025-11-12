#!/usr/bin/env python3
"""Fix syntax errors where multiple statements got merged onto single lines."""

import re

def fix_syntax_errors(file_path):
    """Fix syntax errors in test files."""
    with open(file_path, 'r') as f:
        content = f.read()

    # Pattern 1: extract_loss(result)        result = ...
    content = re.sub(
        r'(extract_loss\(result\))(\s{2,})(result\s*=)',
        r'\1\n        \3',
        content
    )

    # Pattern 2: extract_loss(result)        assert
    content = re.sub(
        r'(extract_loss\(result\))(\s{2,})(assert\s)',
        r'\1\n        \3',
        content
    )

    # Pattern 3: loss = extract_loss(result)        loss.backward()
    content = re.sub(
        r'(loss\s*=\s*extract_loss\(result\))(\s{2,})(loss\.backward\(\))',
        r'\1\n        \3',
        content
    )

    # Pattern 4: loss = extract_loss(result)        assert
    content = re.sub(
        r'(loss\s*=\s*extract_loss\(result\))(\s{2,})(assert\s)',
        r'\1\n        \3',
        content
    )

    # Pattern 5: More general patterns with various statements
    # loss_low = extract_loss(result)        result = loss_fn_high
    content = re.sub(
        r'(loss_\w+\s*=\s*extract_loss\(result\))(\s{2,})(result\s*=)',
        r'\1\n        \3',
        content
    )

    # Pattern 6: loss_high = extract_loss(result)        result = loss_fn_low
    content = re.sub(
        r'(loss_\w+\s*=\s*extract_loss\(result_\w+\))(\s{2,})(result\w*\s*=)',
        r'\1\n        \3',
        content
    )

    with open(file_path, 'w') as f:
        f.write(content)

    print(f"✓ Fixed {file_path}")

if __name__ == '__main__':
    files = [
        'tests/test_contrastive_losses.py',
        'tests/test_selfsupervised_losses.py',
        'tests/test_specialized_losses.py',
    ]

    for file_path in files:
        try:
            fix_syntax_errors(file_path)
        except Exception as e:
            print(f"✗ Error fixing {file_path}: {e}")
