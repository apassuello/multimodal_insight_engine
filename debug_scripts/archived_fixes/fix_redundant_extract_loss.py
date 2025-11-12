#!/usr/bin/env python3
"""
Fix redundant extract_loss() calls in test files.

Pattern:  assert not torch.isnan(extract_loss(loss_var) if not isinstance(loss_var, torch.Tensor) else loss_var)
Should be: assert not torch.isnan(loss_var)

This happens when extract_loss() was already called to create loss_var.
"""

import re

def fix_redundant_extract_loss(file_path):
    """Remove redundant extract_loss calls from assertions."""
    with open(file_path, 'r') as f:
        content = f.read()

    # Pattern: assert not torch.isnan(extract_loss(loss_X) if not isinstance(loss_X, torch.Tensor) else loss_X)
    # Replace with: assert not torch.isnan(loss_X)
    pattern = r'assert not torch\.isnan\(extract_loss\((loss_\w+)\) if not isinstance\(\1, torch\.Tensor\) else \1\)'
    replacement = r'assert not torch.isnan(\1)'

    modified = re.sub(pattern, replacement, content)

    # Also handle the pattern for allclose with extract_loss on both sides
    # Pattern: assert not torch.isnan(extract_loss(loss_high) if ...)
    #          assert not torch.isnan(extract_loss(loss_low) if ...)

    with open(file_path, 'w') as f:
        f.write(modified)

    changes = content.count('extract_loss') - modified.count('extract_loss')
    if changes > 0:
        print(f"✓ Fixed {changes} redundant extract_loss() calls in {file_path}")
    else:
        print(f"  No redundant extract_loss() found in {file_path}")

if __name__ == '__main__':
    fix_redundant_extract_loss('tests/test_selfsupervised_losses.py')
    fix_redundant_extract_loss('tests/test_contrastive_losses.py')
    fix_redundant_extract_loss('tests/test_specialized_losses.py')
    print("\n✅ All redundant extract_loss() calls fixed")
