import pytest

def pytest_configure(config):
    """Register custom marks."""
    config.addinivalue_line(
        "markers",
        "no_test: mark a class as not being a test class"
    ) 