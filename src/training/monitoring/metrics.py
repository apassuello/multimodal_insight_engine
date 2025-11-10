"""MODULE: metrics.py
PURPOSE: Efficient metrics storage for training monitoring
KEY COMPONENTS:
- MetricsBuffer: Ring buffer with O(1) appends and bounded memory
- MetricsStore: Collection of named metric buffers
DEPENDENCIES: collections, numpy, typing
SPECIAL NOTES: Uses deque for ring buffer, lazy NumPy conversion for efficiency
"""

from collections import deque
from typing import Dict, Optional
import numpy as np
from numpy.typing import NDArray


class MetricsBuffer:
    """
    Memory-efficient ring buffer for metric values.

    Design:
    - Uses deque with maxlen for O(1) append and automatic eviction
    - Lazy conversion to NumPy arrays (only when needed)
    - Cache invalidation on new appends
    - Bounded memory: max ~80KB per metric (10K float32 values)

    Example:
        buffer = MetricsBuffer(capacity=1000)
        for i in range(5000):
            buffer.append(float(i))

        # Only last 1000 values retained
        arr = buffer.to_array()
        stats = buffer.statistics()
    """
    __slots__ = ('_buffer', '_capacity', '_dtype', '_aggregated')

    def __init__(self, capacity: int = 10_000, dtype: np.dtype = np.float32):
        """
        Initialize metrics buffer.

        Args:
            capacity: Maximum number of values to store
            dtype: NumPy dtype for array conversion (float32 for memory efficiency)
        """
        self._buffer: deque[float] = deque(maxlen=capacity)
        self._capacity = capacity
        self._dtype = dtype
        self._aggregated: Optional[NDArray[np.float32]] = None

    def append(self, value: float) -> None:
        """
        Append a value to the buffer.

        O(1) operation. Oldest value automatically evicted when at capacity.

        Args:
            value: Metric value to append
        """
        self._buffer.append(value)
        self._aggregated = None  # Invalidate cache

    def to_array(self) -> NDArray[np.float32]:
        """
        Convert buffer to NumPy array (lazy, cached).

        Returns:
            NumPy array of buffered values
        """
        if self._aggregated is None:
            self._aggregated = np.array(list(self._buffer), dtype=self._dtype)
        return self._aggregated

    def statistics(self) -> Dict[str, float]:
        """
        Compute statistics efficiently using NumPy.

        Returns:
            Dict with mean, std, min, max, median
        """
        if len(self._buffer) == 0:
            return {
                'mean': 0.0,
                'std': 0.0,
                'min': 0.0,
                'max': 0.0,
                'median': 0.0,
                'count': 0
            }

        arr = self.to_array()
        return {
            'mean': float(np.mean(arr)),
            'std': float(np.std(arr)),
            'min': float(np.min(arr)),
            'max': float(np.max(arr)),
            'median': float(np.median(arr)),
            'count': len(arr)
        }

    def get_recent(self, n: int = 100) -> NDArray[np.float32]:
        """
        Get the last N values efficiently.

        Args:
            n: Number of recent values to return

        Returns:
            NumPy array of last n values
        """
        arr = self.to_array()
        return arr[-n:] if len(arr) > n else arr

    def compute_trend(self, window: int = 10) -> float:
        """
        Compute trend (slope) over recent window using linear regression.

        Positive trend = increasing, negative = decreasing.

        Args:
            window: Number of recent values to analyze

        Returns:
            Trend slope (change per iteration)
        """
        if len(self._buffer) < 2:
            return 0.0

        recent = self.get_recent(window)
        if len(recent) < 2:
            return 0.0

        # Simple linear regression: y = mx + b
        x = np.arange(len(recent), dtype=np.float32)
        slope = float(np.polyfit(x, recent, 1)[0])
        return slope

    def __len__(self) -> int:
        """Get number of values in buffer."""
        return len(self._buffer)

    @property
    def capacity(self) -> int:
        """Get buffer capacity."""
        return self._capacity

    @property
    def is_full(self) -> bool:
        """Check if buffer is at capacity."""
        return len(self._buffer) >= self._capacity


class MetricsStore:
    """
    Collection of named metric buffers.

    Manages multiple metrics with automatic buffer creation and efficient access.

    Example:
        store = MetricsStore(capacity=1000)
        store.record('loss', 2.34)
        store.record('reward', 0.65)

        loss_stats = store.get_statistics('loss')
        recent_rewards = store.get_recent('reward', n=10)
    """
    __slots__ = ('_metrics', '_capacity')

    def __init__(self, capacity: int = 10_000):
        """
        Initialize metrics store.

        Args:
            capacity: Maximum values per metric buffer
        """
        self._metrics: Dict[str, MetricsBuffer] = {}
        self._capacity = capacity

    def record(self, name: str, value: float) -> None:
        """
        Record a metric value.

        Automatically creates buffer if metric is new.

        Args:
            name: Metric name
            value: Metric value
        """
        if name not in self._metrics:
            self._metrics[name] = MetricsBuffer(self._capacity)
        self._metrics[name].append(value)

    def get_buffer(self, name: str) -> Optional[MetricsBuffer]:
        """
        Get buffer for a metric.

        Args:
            name: Metric name

        Returns:
            MetricsBuffer or None if metric doesn't exist
        """
        return self._metrics.get(name)

    def get_statistics(self, name: str) -> Dict[str, float]:
        """
        Get statistics for a metric.

        Args:
            name: Metric name

        Returns:
            Statistics dict or empty dict if metric doesn't exist
        """
        buffer = self.get_buffer(name)
        if buffer is None:
            return {}
        return buffer.statistics()

    def get_recent(self, name: str, n: int = 100) -> NDArray[np.float32]:
        """
        Get recent values for a metric.

        Args:
            name: Metric name
            n: Number of recent values

        Returns:
            NumPy array of recent values (empty if metric doesn't exist)
        """
        buffer = self.get_buffer(name)
        if buffer is None:
            return np.array([], dtype=np.float32)
        return buffer.get_recent(n)

    def compute_trend(self, name: str, window: int = 10) -> float:
        """
        Compute trend for a metric.

        Args:
            name: Metric name
            window: Analysis window size

        Returns:
            Trend slope (0.0 if metric doesn't exist)
        """
        buffer = self.get_buffer(name)
        if buffer is None:
            return 0.0
        return buffer.compute_trend(window)

    def get_all_statistics(self) -> Dict[str, Dict[str, float]]:
        """
        Get statistics for all metrics.

        Returns:
            Dict mapping metric name to statistics
        """
        return {
            name: buffer.statistics()
            for name, buffer in self._metrics.items()
        }

    def has_metric(self, name: str) -> bool:
        """Check if a metric exists."""
        return name in self._metrics

    def get_metric_names(self) -> list[str]:
        """Get list of all metric names."""
        return list(self._metrics.keys())

    def clear(self) -> None:
        """Clear all metrics."""
        self._metrics.clear()

    def __len__(self) -> int:
        """Get number of tracked metrics."""
        return len(self._metrics)
