"""MODULE: test_metrics.py
PURPOSE: Unit tests for metrics storage system
KEY COMPONENTS:
- Test MetricsBuffer ring buffer
- Test MetricsStore collection
DEPENDENCIES: pytest, numpy, src.training.monitoring.metrics
"""

import pytest
import numpy as np
from src.training.monitoring.metrics import MetricsBuffer, MetricsStore


class TestMetricsBuffer:
    """Test MetricsBuffer ring buffer."""

    def test_initialization(self):
        """Test buffer initialization."""
        buffer = MetricsBuffer(capacity=100)
        assert len(buffer) == 0
        assert buffer.capacity == 100
        assert not buffer.is_full

    def test_append_single_value(self):
        """Test appending a single value."""
        buffer = MetricsBuffer(capacity=10)
        buffer.append(1.5)
        assert len(buffer) == 1

    def test_append_multiple_values(self):
        """Test appending multiple values."""
        buffer = MetricsBuffer(capacity=10)
        for i in range(5):
            buffer.append(float(i))
        assert len(buffer) == 5

    def test_capacity_enforcement(self):
        """Test that buffer enforces capacity (ring buffer behavior)."""
        buffer = MetricsBuffer(capacity=3)
        for i in range(10):
            buffer.append(float(i))

        # Should only keep last 3 values
        assert len(buffer) == 3
        arr = buffer.to_array()
        np.testing.assert_array_equal(arr, np.array([7.0, 8.0, 9.0], dtype=np.float32))

    def test_is_full(self):
        """Test is_full property."""
        buffer = MetricsBuffer(capacity=3)
        assert not buffer.is_full

        buffer.append(1.0)
        buffer.append(2.0)
        assert not buffer.is_full

        buffer.append(3.0)
        assert buffer.is_full

    def test_to_array_empty(self):
        """Test converting empty buffer to array."""
        buffer = MetricsBuffer()
        arr = buffer.to_array()
        assert isinstance(arr, np.ndarray)
        assert len(arr) == 0
        assert arr.dtype == np.float32

    def test_to_array_with_values(self):
        """Test converting buffer with values to array."""
        buffer = MetricsBuffer()
        values = [1.0, 2.5, 3.7, 4.2]
        for val in values:
            buffer.append(val)

        arr = buffer.to_array()
        assert isinstance(arr, np.ndarray)
        assert len(arr) == len(values)
        np.testing.assert_array_almost_equal(arr, np.array(values, dtype=np.float32))

    def test_to_array_caching(self):
        """Test that to_array caches result."""
        buffer = MetricsBuffer()
        buffer.append(1.0)
        buffer.append(2.0)

        arr1 = buffer.to_array()
        arr2 = buffer.to_array()

        # Should return same cached array
        assert arr1 is arr2

    def test_to_array_cache_invalidation(self):
        """Test that cache is invalidated on new append."""
        buffer = MetricsBuffer()
        buffer.append(1.0)
        arr1 = buffer.to_array()

        buffer.append(2.0)
        arr2 = buffer.to_array()

        # Should return different array after append
        assert arr1 is not arr2
        assert len(arr1) == 1
        assert len(arr2) == 2

    def test_statistics_empty(self):
        """Test statistics on empty buffer."""
        buffer = MetricsBuffer()
        stats = buffer.statistics()

        assert stats['mean'] == 0.0
        assert stats['std'] == 0.0
        assert stats['min'] == 0.0
        assert stats['max'] == 0.0
        assert stats['median'] == 0.0
        assert stats['count'] == 0

    def test_statistics_with_values(self):
        """Test statistics computation."""
        buffer = MetricsBuffer()
        values = [1.0, 2.0, 3.0, 4.0, 5.0]
        for val in values:
            buffer.append(val)

        stats = buffer.statistics()
        assert stats['mean'] == pytest.approx(3.0)
        assert stats['std'] == pytest.approx(np.std(values))
        assert stats['min'] == 1.0
        assert stats['max'] == 5.0
        assert stats['median'] == 3.0
        assert stats['count'] == 5

    def test_get_recent_less_than_n(self):
        """Test get_recent when buffer has fewer than n values."""
        buffer = MetricsBuffer()
        buffer.append(1.0)
        buffer.append(2.0)

        recent = buffer.get_recent(10)
        assert len(recent) == 2
        np.testing.assert_array_equal(recent, np.array([1.0, 2.0], dtype=np.float32))

    def test_get_recent_more_than_n(self):
        """Test get_recent when buffer has more than n values."""
        buffer = MetricsBuffer()
        for i in range(10):
            buffer.append(float(i))

        recent = buffer.get_recent(3)
        assert len(recent) == 3
        np.testing.assert_array_equal(recent, np.array([7.0, 8.0, 9.0], dtype=np.float32))

    def test_compute_trend_insufficient_data(self):
        """Test trend computation with insufficient data."""
        buffer = MetricsBuffer()
        assert buffer.compute_trend() == 0.0

        buffer.append(1.0)
        assert buffer.compute_trend() == 0.0

    def test_compute_trend_increasing(self):
        """Test trend computation for increasing values."""
        buffer = MetricsBuffer()
        for i in range(10):
            buffer.append(float(i))

        trend = buffer.compute_trend(window=10)
        assert trend > 0.0  # Positive slope for increasing values

    def test_compute_trend_decreasing(self):
        """Test trend computation for decreasing values."""
        buffer = MetricsBuffer()
        for i in range(10, 0, -1):
            buffer.append(float(i))

        trend = buffer.compute_trend(window=10)
        assert trend < 0.0  # Negative slope for decreasing values

    def test_compute_trend_constant(self):
        """Test trend computation for constant values."""
        buffer = MetricsBuffer()
        for _ in range(10):
            buffer.append(5.0)

        trend = buffer.compute_trend(window=10)
        assert abs(trend) < 1e-5  # Near-zero slope for constant values

    def test_compute_trend_window(self):
        """Test trend computation with specific window size."""
        buffer = MetricsBuffer()
        # Add 10 decreasing values, then 5 increasing values
        for i in range(10, 0, -1):
            buffer.append(float(i))
        for i in range(1, 6):
            buffer.append(float(i))

        # Recent window should show increasing trend
        trend = buffer.compute_trend(window=5)
        assert trend > 0.0

    def test_slots_memory_efficiency(self):
        """Test that MetricsBuffer uses slots."""
        buffer = MetricsBuffer()
        assert hasattr(buffer, '__slots__')
        assert not hasattr(buffer, '__dict__')

    def test_custom_dtype(self):
        """Test buffer with custom dtype."""
        buffer = MetricsBuffer(capacity=10, dtype=np.float64)
        buffer.append(1.5)
        arr = buffer.to_array()
        assert arr.dtype == np.float64

    def test_large_capacity(self):
        """Test buffer with large capacity."""
        buffer = MetricsBuffer(capacity=10_000)
        for i in range(1000):
            buffer.append(float(i))
        assert len(buffer) == 1000
        assert not buffer.is_full


class TestMetricsStore:
    """Test MetricsStore collection."""

    def test_initialization(self):
        """Test store initialization."""
        store = MetricsStore(capacity=1000)
        assert len(store) == 0

    def test_record_single_metric(self):
        """Test recording a single metric."""
        store = MetricsStore()
        store.record('loss', 2.5)
        assert len(store) == 1
        assert store.has_metric('loss')

    def test_record_multiple_metrics(self):
        """Test recording multiple different metrics."""
        store = MetricsStore()
        store.record('loss', 2.5)
        store.record('reward', 0.7)
        store.record('kl_div', 0.01)

        assert len(store) == 3
        assert store.has_metric('loss')
        assert store.has_metric('reward')
        assert store.has_metric('kl_div')

    def test_record_multiple_values_same_metric(self):
        """Test recording multiple values for same metric."""
        store = MetricsStore()
        for i in range(10):
            store.record('loss', float(i))

        assert len(store) == 1  # Still only 1 metric
        buffer = store.get_buffer('loss')
        assert len(buffer) == 10

    def test_get_buffer_existing(self):
        """Test getting buffer for existing metric."""
        store = MetricsStore()
        store.record('loss', 1.5)

        buffer = store.get_buffer('loss')
        assert buffer is not None
        assert isinstance(buffer, MetricsBuffer)
        assert len(buffer) == 1

    def test_get_buffer_nonexistent(self):
        """Test getting buffer for nonexistent metric."""
        store = MetricsStore()
        buffer = store.get_buffer('nonexistent')
        assert buffer is None

    def test_get_statistics_existing(self):
        """Test getting statistics for existing metric."""
        store = MetricsStore()
        values = [1.0, 2.0, 3.0, 4.0, 5.0]
        for val in values:
            store.record('loss', val)

        stats = store.get_statistics('loss')
        assert stats['mean'] == pytest.approx(3.0)
        assert stats['count'] == 5

    def test_get_statistics_nonexistent(self):
        """Test getting statistics for nonexistent metric."""
        store = MetricsStore()
        stats = store.get_statistics('nonexistent')
        assert stats == {}

    def test_get_recent_existing(self):
        """Test getting recent values for existing metric."""
        store = MetricsStore()
        for i in range(10):
            store.record('loss', float(i))

        recent = store.get_recent('loss', n=3)
        assert len(recent) == 3
        np.testing.assert_array_equal(recent, np.array([7.0, 8.0, 9.0], dtype=np.float32))

    def test_get_recent_nonexistent(self):
        """Test getting recent values for nonexistent metric."""
        store = MetricsStore()
        recent = store.get_recent('nonexistent', n=5)
        assert len(recent) == 0

    def test_compute_trend_existing(self):
        """Test computing trend for existing metric."""
        store = MetricsStore()
        for i in range(10):
            store.record('loss', float(i))

        trend = store.compute_trend('loss', window=10)
        assert trend > 0.0  # Increasing trend

    def test_compute_trend_nonexistent(self):
        """Test computing trend for nonexistent metric."""
        store = MetricsStore()
        trend = store.compute_trend('nonexistent')
        assert trend == 0.0

    def test_get_all_statistics(self):
        """Test getting statistics for all metrics."""
        store = MetricsStore()
        store.record('loss', 1.0)
        store.record('loss', 2.0)
        store.record('reward', 0.5)
        store.record('reward', 0.7)

        all_stats = store.get_all_statistics()
        assert 'loss' in all_stats
        assert 'reward' in all_stats
        assert all_stats['loss']['count'] == 2
        assert all_stats['reward']['count'] == 2

    def test_has_metric(self):
        """Test checking if metric exists."""
        store = MetricsStore()
        assert not store.has_metric('loss')

        store.record('loss', 1.5)
        assert store.has_metric('loss')
        assert not store.has_metric('reward')

    def test_get_metric_names(self):
        """Test getting all metric names."""
        store = MetricsStore()
        assert store.get_metric_names() == []

        store.record('loss', 1.0)
        store.record('reward', 0.5)
        store.record('kl_div', 0.01)

        names = store.get_metric_names()
        assert len(names) == 3
        assert 'loss' in names
        assert 'reward' in names
        assert 'kl_div' in names

    def test_clear(self):
        """Test clearing all metrics."""
        store = MetricsStore()
        store.record('loss', 1.0)
        store.record('reward', 0.5)
        assert len(store) == 2

        store.clear()
        assert len(store) == 0
        assert not store.has_metric('loss')
        assert not store.has_metric('reward')

    def test_capacity_propagation(self):
        """Test that capacity is propagated to buffers."""
        store = MetricsStore(capacity=5)
        for i in range(10):
            store.record('loss', float(i))

        buffer = store.get_buffer('loss')
        assert buffer.capacity == 5
        assert len(buffer) == 5  # Should only keep last 5

    def test_automatic_buffer_creation(self):
        """Test that buffers are created automatically."""
        store = MetricsStore()
        assert not store.has_metric('new_metric')

        store.record('new_metric', 42.0)
        assert store.has_metric('new_metric')

        buffer = store.get_buffer('new_metric')
        assert buffer is not None
        assert len(buffer) == 1

    def test_slots_memory_efficiency(self):
        """Test that MetricsStore uses slots."""
        store = MetricsStore()
        assert hasattr(store, '__slots__')
        assert not hasattr(store, '__dict__')

    def test_multiple_metrics_independence(self):
        """Test that different metrics don't interfere."""
        store = MetricsStore()

        # Record different patterns for different metrics
        for i in range(5):
            store.record('increasing', float(i))
            store.record('decreasing', float(5 - i))

        inc_trend = store.compute_trend('increasing', window=5)
        dec_trend = store.compute_trend('decreasing', window=5)

        assert inc_trend > 0.0
        assert dec_trend < 0.0

    def test_large_number_of_metrics(self):
        """Test store with many different metrics."""
        store = MetricsStore()
        num_metrics = 100

        for i in range(num_metrics):
            store.record(f'metric_{i}', float(i))

        assert len(store) == num_metrics
        names = store.get_metric_names()
        assert len(names) == num_metrics
