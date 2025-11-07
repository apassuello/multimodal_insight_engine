# src/safety/__init__.py

from .evaluator import SafetyEvaluator
from .filter import SafetyFilter
from .harness import SafetyTestHarness
from .integration import SafetyAugmentedModel

# Optional constitutional AI components
try:
    from .constitutional import (
        ConstitutionalFramework,
        ConstitutionalPrinciple,
        ConstitutionalSafetyEvaluator,
        ConstitutionalSafetyFilter,
        RLAIFTrainer,
        setup_default_framework,
    )
    CONSTITUTIONAL_AI_AVAILABLE = True
except ImportError:
    CONSTITUTIONAL_AI_AVAILABLE = False
    ConstitutionalPrinciple = None
    ConstitutionalFramework = None
    ConstitutionalSafetyEvaluator = None
    ConstitutionalSafetyFilter = None
    RLAIFTrainer = None
    setup_default_framework = None

__all__ = [
    "SafetyEvaluator",
    "SafetyFilter",
    "SafetyTestHarness",
    "SafetyAugmentedModel",
    # Constitutional AI (if available)
    "ConstitutionalPrinciple",
    "ConstitutionalFramework",
    "ConstitutionalSafetyEvaluator",
    "ConstitutionalSafetyFilter",
    "RLAIFTrainer",
    "setup_default_framework",
    "CONSTITUTIONAL_AI_AVAILABLE",
]
