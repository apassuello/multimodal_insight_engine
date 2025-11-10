"""MODULE: test_verbosity.py
PURPOSE: Unit tests for VerbosityLevel enum
KEY COMPONENTS:
- Test enum values and comparisons
DEPENDENCIES: pytest, src.training.monitoring.verbosity
"""

import pytest
from src.training.monitoring.verbosity import VerbosityLevel


class TestVerbosityLevel:
    """Test VerbosityLevel enum."""

    def test_enum_values(self):
        """Test that all expected enum values exist."""
        assert hasattr(VerbosityLevel, 'SIMPLE')
        assert hasattr(VerbosityLevel, 'VERBOSE')

    def test_enum_distinct(self):
        """Test that enum values are distinct."""
        assert VerbosityLevel.SIMPLE != VerbosityLevel.VERBOSE

    def test_enum_comparison(self):
        """Test enum value comparisons."""
        simple = VerbosityLevel.SIMPLE
        verbose = VerbosityLevel.VERBOSE

        assert simple == VerbosityLevel.SIMPLE
        assert verbose == VerbosityLevel.VERBOSE
        assert simple != verbose

    def test_enum_string_representation(self):
        """Test string representation."""
        assert 'SIMPLE' in str(VerbosityLevel.SIMPLE)
        assert 'VERBOSE' in str(VerbosityLevel.VERBOSE)

    def test_enum_name_access(self):
        """Test accessing enum by name."""
        assert VerbosityLevel['SIMPLE'] == VerbosityLevel.SIMPLE
        assert VerbosityLevel['VERBOSE'] == VerbosityLevel.VERBOSE

    def test_enum_iteration(self):
        """Test iterating over enum values."""
        levels = list(VerbosityLevel)
        assert len(levels) == 2
        assert VerbosityLevel.SIMPLE in levels
        assert VerbosityLevel.VERBOSE in levels
