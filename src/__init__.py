"""
MultiModal Insight Engine

A framework for multimodal learning and understanding.
"""

# Lazy imports to avoid pulling in heavy dependencies during testing
# Use __getattr__ for PEP 562 lazy module loading
def __getattr__(name):
    """Lazy import modules on first access."""
    if name in ("configs", "data", "evaluation", "models", "optimization", "safety", "training", "utils"):
        import importlib
        module = importlib.import_module(f".{name}", __package__)
        globals()[name] = module
        return module
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = [
    "models",
    "data",
    "training",
    "utils",
    "configs",
    "evaluation",
    "optimization",
    "safety",
]
