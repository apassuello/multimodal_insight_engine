"""MODULE: monitoring
PURPOSE: Training monitoring system for Constitutional AI
KEY COMPONENTS:
- TrainingMonitor: Main orchestrator
- TrainingCallback: Protocol for implementing callbacks
- CallbackManager: Event dispatcher
- TrainingEvent, TrainingPhase: Event data structures
- MetricsBuffer, MetricsStore: Efficient metrics storage
- VerbosityLevel: Control display detail level
- SampleComparator: Track response quality changes
- QualityAnalyzer: Detect quality degradation and trigger early stopping
DEPENDENCIES: See individual modules
SPECIAL NOTES: Public API for training monitoring system
"""

# Main orchestrator
from .training_monitor import TrainingMonitor

# Core callback system
from .callbacks import TrainingCallback, CallbackManager

# Event system
from .events import TrainingEvent, TrainingPhase

# Metrics storage
from .metrics import MetricsBuffer, MetricsStore

# Verbosity control
from .verbosity import VerbosityLevel

# Display callbacks
from .terminal_display import TerminalDisplay

# Quality monitoring
from .sample_comparator import SampleComparator, SampleComparison
from .quality_analyzer import QualityAnalyzer, QualityAlert, AlertSeverity

# PPO-specific tracking
from .ppo_metrics_tracker import PPOMetricsTracker, PPOSnapshot

__all__ = [
    # Main orchestrator
    'TrainingMonitor',

    # Callback system
    'TrainingCallback',
    'CallbackManager',

    # Events
    'TrainingEvent',
    'TrainingPhase',

    # Metrics
    'MetricsBuffer',
    'MetricsStore',

    # Verbosity
    'VerbosityLevel',

    # Display
    'TerminalDisplay',

    # Quality monitoring
    'SampleComparator',
    'SampleComparison',
    'QualityAnalyzer',
    'QualityAlert',
    'AlertSeverity',

    # PPO tracking
    'PPOMetricsTracker',
    'PPOSnapshot',
]

__version__ = '0.1.0'
