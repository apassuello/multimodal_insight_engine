"""MODULE: verbosity.py
PURPOSE: Verbosity level control for training monitoring
KEY COMPONENTS:
- VerbosityLevel: Enum for controlling display detail level
DEPENDENCIES: enum
SPECIAL NOTES: Simple/verbose modes control how much detail is shown
"""

from enum import Enum, auto


class VerbosityLevel(Enum):
    """
    Verbosity level for training monitoring.

    SIMPLE: Minimal output, one-line summaries per iteration
    VERBOSE: Detailed output with full metrics, explanations, and diffs
    """
    SIMPLE = auto()
    VERBOSE = auto()

    def is_verbose(self) -> bool:
        """Check if verbosity is set to verbose mode."""
        return self == VerbosityLevel.VERBOSE

    def is_simple(self) -> bool:
        """Check if verbosity is set to simple mode."""
        return self == VerbosityLevel.SIMPLE
