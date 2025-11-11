#!/usr/bin/env python3
"""
Fix all loss function test API signatures and return type handling.
"""

import re
from pathlib import Path

# Helper function template to add to files
HELPER_FUNCTION = '''
# ============================================================================
# Helper Functions
# ============================================================================


def extract_loss(result):
    """
    Extract loss tensor from various return types.

    Handles:
    - dict: returns result['loss'] or result['total_loss']
    - tuple: returns first element
    - tensor: returns as-is
    """
    if isinstance(result, dict):
        return result.get('loss', result.get('total_loss', result.get('contrastive_loss')))
    elif isinstance(result, tuple):
        return result[0]
    else:
        return result
'''

def fix_file(filepath):
    """Fix a single test file."""
    with open(filepath, 'r') as f:
        content = f.read()

    original = content

    # Add helper function if not present
    if 'def extract_loss(result):' not in content:
        # Find insertion point (after imports, before fixtures)
        fixture_match = re.search(r'(@pytest\.fixture|# =+ Fixtures|class Test)', content)
        if fixture_match:
            pos = fixture_match.start()
            content = content[:pos] + HELPER_FUNCTION + '\n' + content[pos:]

    # Fix API signatures
    content = re.sub(
        r'MemoryQueueContrastiveLoss\(\s*temperature=([^,]+),\s*queue_size=([^,]+),\s*embed_dim=([^)]+)\)',
        r'MemoryQueueContrastiveLoss(temperature=\1, queue_size=\2, dim=\3)',
        content
    )

    content = re.sub(
        r'HardNegativeMiningContrastiveLoss\(\s*temperature=([^,]+),\s*num_hard_negatives=([^)]+)\)',
        r'HardNegativeMiningContrastiveLoss(temperature=\1)',
        content
    )

    content = re.sub(
        r'DynamicTemperatureContrastiveLoss\(\s*initial_temperature=([^,]+)',
        r'DynamicTemperatureContrastiveLoss(base_temperature=\1',
        content
    )

    content = re.sub(
        r'DecoupledContrastiveLoss\(\s*temperature=([^,]+),\s*decouple_factor=([^)]+)\)',
        r'DecoupledContrastiveLoss(temperature=\1)',
        content
    )

    # Fix forward calls that need match_ids
    # HardNegativeMiningContrastiveLoss and DecoupledContrastiveLoss need match_ids

    # Fix loss assignments - convert to result + extract_loss pattern
    # Pattern: loss = loss_fn(...) -> result = loss_fn(...); loss = extract_loss(result)
    def add_extract_loss(match):
        indent = match.group(1)
        var_name = match.group(2) if match.group(2) else 'loss'
        call = match.group(3)

        # Don't modify if already using extract_loss
        if 'extract_loss' in call:
            return match.group(0)

        return f'{indent}result = {call}\n{indent}{var_name} = extract_loss(result)'

    # Match: whitespace + loss_var = loss_fn_call(...)
    content = re.sub(
        r'(\s+)(loss(?:_\w+)?)\s*=\s*(loss_fn[^\n]+)\n',
        add_extract_loss,
        content
    )

    # Also fix direct assertions on loss that might be dict/tuple
    # Pattern: assert not torch.isnan(loss_xxx)
    content = re.sub(
        r'(assert not torch\.isnan\()(loss_\w+)\)',
        r'\1extract_loss(\2) if not isinstance(\2, torch.Tensor) else \2)',
        content
    )

    # Pattern: assert not torch.isinf(loss_xxx)
    content = re.sub(
        r'(assert not torch\.isinf\()(loss_\w+)\)',
        r'\1extract_loss(\2) if not isinstance(\2, torch.Tensor) else \2)',
        content
    )

    # Save if changed
    if content != original:
        with open(filepath, 'w') as f:
            f.write(content)
        return True
    return False

# Fix test files
test_files = [
    'tests/test_contrastive_losses.py',
    'tests/test_selfsupervised_losses.py',
    'tests/test_specialized_losses.py',
]

for test_file in test_files:
    filepath = Path(test_file)
    if filepath.exists():
        if fix_file(filepath):
            print(f'✓ Fixed {test_file}')
        else:
            print(f'- No changes needed for {test_file}')
    else:
        print(f'✗ File not found: {test_file}')

print('\\nDone!')
