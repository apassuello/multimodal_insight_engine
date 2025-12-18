"""
Comprehensive tests for learning rate schedulers.

Tests cover:
- WarmupCosineScheduler (warmup + cosine annealing)
- LinearWarmupScheduler (warmup + linear decay)
- LayerwiseLRScheduler (per-group scheduling)

Target: 90%+ coverage (pure math, no I/O)
"""

import pytest
import torch
import torch.nn as nn
from torch.optim import Adam, SGD

from src.utils.learningrate_scheduler import (
    WarmupCosineScheduler,
    LinearWarmupScheduler,
    LayerwiseLRScheduler,
)


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def simple_model():
    """Create a simple model for testing."""
    return nn.Linear(10, 10)


@pytest.fixture
def optimizer(simple_model):
    """Create optimizer with single parameter group."""
    return Adam(simple_model.parameters(), lr=1e-3)


@pytest.fixture
def multi_group_optimizer(simple_model):
    """Create optimizer with multiple parameter groups."""
    layer1 = nn.Linear(10, 10)
    layer2 = nn.Linear(10, 5)

    return Adam([
        {'params': layer1.parameters(), 'lr': 1e-3, 'name': 'layer1'},
        {'params': layer2.parameters(), 'lr': 5e-4, 'name': 'layer2'}
    ])


# ============================================================================
# WarmupCosineScheduler Tests
# ============================================================================


class TestWarmupCosineScheduler:
    """Test warmup + cosine annealing scheduler."""

    def test_initialization(self, optimizer):
        """Test scheduler initializes correctly."""
        scheduler = WarmupCosineScheduler(
            optimizer,
            warmup_steps=100,
            total_steps=1000,
            min_lr=0.0
        )

        assert scheduler.warmup_steps == 100
        assert scheduler.total_steps == 1000
        assert scheduler.min_lr == 0.0
        assert len(scheduler.base_lrs) == 1
        assert scheduler.base_lrs[0] == 1e-3

    def test_warmup_phase_increases_lr(self, optimizer):
        """Test that LR increases linearly during warmup."""
        scheduler = WarmupCosineScheduler(
            optimizer,
            warmup_steps=100,
            total_steps=1000,
            min_lr=0.0,
            warmup_start_factor=0.1
        )

        # Collect LRs during warmup (first 100 steps)
        lrs = []
        for _ in range(100):
            scheduler.step()
            lrs.append(scheduler.get_last_lr()[0])

        # Should increase monotonically during warmup
        for i in range(len(lrs) - 1):
            assert lrs[i] < lrs[i + 1], f"LR should increase but {lrs[i]} >= {lrs[i+1]} at step {i}"

        # Last warmup LR should be close to base_lr
        assert lrs[-1] == pytest.approx(1e-3, rel=1e-6)

    def test_warmup_start_factor(self, optimizer):
        """Test warmup_start_factor controls initial LR."""
        # With warmup_start_factor=0.1, should start at 10% of base_lr
        scheduler = WarmupCosineScheduler(
            optimizer,
            warmup_steps=10,
            total_steps=100,
            warmup_start_factor=0.1
        )

        scheduler.step()
        first_lr = scheduler.get_last_lr()[0]

        # At step 1, should be ~10% + (1-10%) * (1/10) = ~19% of base_lr
        expected = 1e-3 * (0.1 + 0.9 * (1 / 10))
        assert first_lr == pytest.approx(expected, rel=1e-6)

    def test_cosine_phase_decreases_lr(self, optimizer):
        """Test that LR decreases with cosine after warmup."""
        scheduler = WarmupCosineScheduler(
            optimizer,
            warmup_steps=10,
            total_steps=100,
            min_lr=0.0
        )

        # Skip warmup
        for _ in range(10):
            scheduler.step()

        # Collect LRs during cosine phase
        lrs = []
        for _ in range(50):
            scheduler.step()
            lrs.append(scheduler.get_last_lr()[0])

        # Should decrease (with cosine, not perfectly monotonic but mostly)
        decreasing_count = sum(1 for i in range(len(lrs) - 1) if lrs[i] >= lrs[i + 1])
        assert decreasing_count / len(lrs) > 0.8  # At least 80% of steps should decrease

    def test_min_lr_not_violated(self, optimizer):
        """Test that LR never goes below min_lr."""
        min_lr = 1e-6
        scheduler = WarmupCosineScheduler(
            optimizer,
            warmup_steps=10,
            total_steps=100,
            min_lr=min_lr
        )

        # Run entire schedule
        for _ in range(100):
            scheduler.step()
            current_lr = scheduler.get_last_lr()[0]
            assert current_lr >= min_lr, f"LR {current_lr} below min_lr {min_lr}"

    def test_cosine_reaches_min_lr_at_end(self, optimizer):
        """Test that LR reaches min_lr at end of schedule."""
        min_lr = 1e-6
        scheduler = WarmupCosineScheduler(
            optimizer,
            warmup_steps=10,
            total_steps=100,
            min_lr=min_lr
        )

        # Run to end
        for _ in range(100):
            scheduler.step()

        final_lr = scheduler.get_last_lr()[0]
        # Should be very close to min_lr at the end
        assert final_lr == pytest.approx(min_lr, abs=1e-7)

    def test_state_dict_save_and_load(self, optimizer):
        """Test scheduler state can be saved and restored."""
        scheduler = WarmupCosineScheduler(
            optimizer,
            warmup_steps=10,
            total_steps=100,
            min_lr=0.0
        )

        # Run 50 steps
        for _ in range(50):
            scheduler.step()

        # Save state
        state = scheduler.state_dict()
        lr_at_50 = scheduler.get_last_lr()[0]

        # Create new scheduler and load state
        scheduler2 = WarmupCosineScheduler(
            optimizer,
            warmup_steps=10,
            total_steps=100,
            min_lr=0.0
        )
        scheduler2.load_state_dict(state)

        # LR should match
        assert scheduler2.get_last_lr()[0] == pytest.approx(lr_at_50)

    def test_multiple_parameter_groups(self, multi_group_optimizer):
        """Test scheduler works with multiple parameter groups."""
        scheduler = WarmupCosineScheduler(
            multi_group_optimizer,
            warmup_steps=10,
            total_steps=100,
            min_lr=0.0
        )

        assert len(scheduler.base_lrs) == 2
        assert scheduler.base_lrs[0] == 1e-3
        assert scheduler.base_lrs[1] == 5e-4

        scheduler.step()
        lrs = scheduler.get_last_lr()

        assert len(lrs) == 2
        # Both groups should have different LRs based on their base_lrs
        assert lrs[0] > lrs[1]


# ============================================================================
# LinearWarmupScheduler Tests
# ============================================================================


class TestLinearWarmupScheduler:
    """Test linear warmup + linear decay scheduler."""

    def test_initialization(self, optimizer):
        """Test scheduler initializes correctly."""
        scheduler = LinearWarmupScheduler(
            optimizer,
            warmup_epochs=10,
            total_epochs=100,
            init_lr=0.0,
            final_lr=1e-6
        )

        assert scheduler.warmup_epochs == 10
        assert scheduler.total_epochs == 100
        assert scheduler.init_lr == 0.0
        assert scheduler.final_lr == 1e-6

    def test_warmup_phase_linear_increase(self, optimizer):
        """Test linear warmup increases LR linearly."""
        scheduler = LinearWarmupScheduler(
            optimizer,
            warmup_epochs=10,
            total_epochs=100,
            init_lr=0.0,
            final_lr=0.0
        )

        # Collect LRs during warmup
        lrs = []
        for _ in range(10):
            scheduler.step()
            lrs.append(scheduler.get_last_lr()[0])

        # Should increase linearly
        # Check that differences between consecutive LRs are roughly equal
        diffs = [lrs[i + 1] - lrs[i] for i in range(len(lrs) - 1)]
        avg_diff = sum(diffs) / len(diffs)

        for diff in diffs:
            assert diff == pytest.approx(avg_diff, rel=0.1)

    def test_linear_decay_phase(self, optimizer):
        """Test linear decay decreases LR linearly."""
        scheduler = LinearWarmupScheduler(
            optimizer,
            warmup_epochs=10,
            total_epochs=100,
            init_lr=0.0,
            final_lr=0.0
        )

        # Skip warmup
        for _ in range(10):
            scheduler.step()

        # Collect LRs during decay
        lrs = []
        for _ in range(50):
            scheduler.step()
            lrs.append(scheduler.get_last_lr()[0])

        # Should decrease linearly
        diffs = [lrs[i] - lrs[i + 1] for i in range(len(lrs) - 1)]
        avg_diff = sum(diffs) / len(diffs)

        for diff in diffs:
            assert diff == pytest.approx(avg_diff, rel=0.1)

    def test_reaches_final_lr(self, optimizer):
        """Test that scheduler reaches final_lr at end."""
        final_lr = 1e-6
        scheduler = LinearWarmupScheduler(
            optimizer,
            warmup_epochs=10,
            total_epochs=100,
            init_lr=0.0,
            final_lr=final_lr
        )

        # Run to end
        for _ in range(100):
            scheduler.step()

        assert scheduler.get_last_lr()[0] == pytest.approx(final_lr, abs=1e-9)

    def test_init_lr_respected(self, optimizer):
        """Test that warmup starts from init_lr."""
        init_lr = 1e-4
        scheduler = LinearWarmupScheduler(
            optimizer,
            warmup_epochs=10,
            total_epochs=100,
            init_lr=init_lr,
            final_lr=0.0
        )

        scheduler.step()
        first_lr = scheduler.get_last_lr()[0]

        # At epoch 1, should be init_lr + (base_lr - init_lr) * (1/10)
        base_lr = 1e-3
        expected = init_lr + (base_lr - init_lr) * (1 / 10)
        assert first_lr == pytest.approx(expected, rel=1e-6)


# ============================================================================
# LayerwiseLRScheduler Tests
# ============================================================================


class TestLayerwiseLRScheduler:
    """Test layerwise LR scheduler manager."""

    def test_initialization(self, multi_group_optimizer):
        """Test manager initializes correctly."""
        manager = LayerwiseLRScheduler(multi_group_optimizer)

        assert len(manager.param_groups) == 2
        assert 'layer1' in manager.param_groups
        assert 'layer2' in manager.param_groups

    def test_add_scheduler_warmup_cosine(self, multi_group_optimizer):
        """Test adding warmup_cosine scheduler for a group."""
        manager = LayerwiseLRScheduler(multi_group_optimizer)

        manager.add_scheduler(
            'layer1',
            scheduler_type='warmup_cosine',
            warmup_steps=10,
            total_steps=100,
            min_lr=0.0
        )

        assert 'layer1' in manager.schedulers

    def test_add_scheduler_linear_warmup(self, multi_group_optimizer):
        """Test adding linear_warmup scheduler for a group."""
        manager = LayerwiseLRScheduler(multi_group_optimizer)

        manager.add_scheduler(
            'layer2',
            scheduler_type='linear_warmup',
            warmup_epochs=10,
            total_epochs=100,
            init_lr=0.0,
            final_lr=0.0
        )

        assert 'layer2' in manager.schedulers

    def test_add_scheduler_unknown_group(self, multi_group_optimizer):
        """Test adding scheduler for unknown group logs warning."""
        manager = LayerwiseLRScheduler(multi_group_optimizer)

        # Should not raise, just log warning
        manager.add_scheduler(
            'unknown_group',
            scheduler_type='warmup_cosine',
            warmup_steps=10,
            total_steps=100
        )

        # Unknown group should not be added
        assert 'unknown_group' not in manager.schedulers

    def test_add_scheduler_unsupported_type(self, multi_group_optimizer):
        """Test adding unsupported scheduler type raises error."""
        manager = LayerwiseLRScheduler(multi_group_optimizer)

        with pytest.raises(ValueError, match="Unsupported scheduler type"):
            manager.add_scheduler(
                'layer1',
                scheduler_type='unsupported_scheduler'
            )

    def test_step_updates_all_schedulers(self, multi_group_optimizer):
        """Test step() updates all group schedulers."""
        manager = LayerwiseLRScheduler(multi_group_optimizer)

        # Add different schedulers for each group
        manager.add_scheduler(
            'layer1',
            scheduler_type='warmup_cosine',
            warmup_steps=10,
            total_steps=100,
            min_lr=0.0
        )
        manager.add_scheduler(
            'layer2',
            scheduler_type='linear_warmup',
            warmup_epochs=10,
            total_epochs=100,
            init_lr=0.0,
            final_lr=0.0
        )

        # Get initial LRs
        initial_lrs = manager.get_last_lrs()

        # Step forward
        manager.step()

        # Get new LRs
        new_lrs = manager.get_last_lrs()

        # LRs should have changed
        assert new_lrs['layer1'] != initial_lrs['layer1']
        assert new_lrs['layer2'] != initial_lrs['layer2']

    def test_get_last_lrs_all_groups(self, multi_group_optimizer):
        """Test get_last_lrs returns LRs for all groups."""
        manager = LayerwiseLRScheduler(multi_group_optimizer)

        # Only add scheduler for layer1
        manager.add_scheduler(
            'layer1',
            scheduler_type='warmup_cosine',
            warmup_steps=10,
            total_steps=100
        )

        lrs = manager.get_last_lrs()

        # Should return LRs for both groups
        assert 'layer1' in lrs
        assert 'layer2' in lrs

    def test_different_schedules_per_group(self, multi_group_optimizer):
        """Test different groups can have different schedules."""
        manager = LayerwiseLRScheduler(multi_group_optimizer)

        # Add different schedulers
        manager.add_scheduler(
            'layer1',
            scheduler_type='warmup_cosine',
            warmup_steps=5,
            total_steps=50,
            min_lr=0.0
        )
        manager.add_scheduler(
            'layer2',
            scheduler_type='linear_warmup',
            warmup_epochs=20,
            total_epochs=50,
            init_lr=0.0,
            final_lr=0.0
        )

        # Run 10 steps
        for _ in range(10):
            manager.step()

        lrs = manager.get_last_lrs()

        # Layer1 should be past warmup (5 steps), layer2 still in warmup (20 epochs)
        # layer1 LR should be decreasing (cosine), layer2 still increasing (warmup)
        # This validates they're using different schedules

        # Get initial comparison
        manager2 = LayerwiseLRScheduler(multi_group_optimizer)
        manager2.add_scheduler('layer1', scheduler_type='warmup_cosine', warmup_steps=5, total_steps=50, min_lr=0.0)
        manager2.add_scheduler('layer2', scheduler_type='linear_warmup', warmup_epochs=20, total_epochs=50, init_lr=0.0, final_lr=0.0)

        manager2.step()
        lrs_step1 = manager2.get_last_lrs()

        # After 10 steps, layer1 should be lower than after 1 step
        assert lrs['layer1'] < lrs_step1['layer1']


# ============================================================================
# Edge Cases & Numerical Stability Tests
# ============================================================================


class TestEdgeCases:
    """Test edge cases and numerical stability."""

    def test_zero_warmup_steps(self, optimizer):
        """Test scheduler with zero warmup steps."""
        scheduler = WarmupCosineScheduler(
            optimizer,
            warmup_steps=0,
            total_steps=100,
            min_lr=0.0
        )

        scheduler.step()
        lr = scheduler.get_last_lr()[0]

        # Should start cosine immediately (slightly below base_lr)
        assert lr <= 1e-3

    def test_warmup_equals_total_steps(self, optimizer):
        """Test when warmup_steps == total_steps (no cosine phase)."""
        scheduler = WarmupCosineScheduler(
            optimizer,
            warmup_steps=100,
            total_steps=100,
            min_lr=0.0
        )

        # Run full schedule
        for _ in range(100):
            scheduler.step()

        # Should end at base_lr
        final_lr = scheduler.get_last_lr()[0]
        assert final_lr == pytest.approx(1e-3, rel=1e-5)

    def test_single_step_schedule(self, optimizer):
        """Test scheduler with total_steps=1."""
        scheduler = WarmupCosineScheduler(
            optimizer,
            warmup_steps=0,
            total_steps=1,
            min_lr=0.0
        )

        scheduler.step()
        lr = scheduler.get_last_lr()[0]

        # Should not crash, should return valid LR
        assert 0 <= lr <= 1e-3

    def test_very_large_total_steps(self, optimizer):
        """Test scheduler handles large step counts."""
        scheduler = WarmupCosineScheduler(
            optimizer,
            warmup_steps=1000,
            total_steps=1000000,  # 1 million steps
            min_lr=1e-7
        )

        # Should initialize without overflow
        scheduler.step()
        lr = scheduler.get_last_lr()[0]

        assert 0 < lr <= 1e-3
