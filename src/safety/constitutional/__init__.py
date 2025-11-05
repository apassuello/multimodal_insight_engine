"""
Constitutional AI Module

This module implements Constitutional AI principles for safer, more aligned
model training and evaluation. Based on Anthropic's Constitutional AI research.

Components:
- ConstitutionalPrinciple: Individual principle with evaluation logic
- ConstitutionalFramework: Collection of principles for comprehensive evaluation
- ConstitutionalSafetyEvaluator: Two-stage evaluator with self-critique
- ConstitutionalSafetyFilter: Input/output filtering with constitutional principles
- RLAIFTrainer: Reinforcement Learning from AI Feedback trainer

Key Features:
- Four core principles: harm prevention, truthfulness, fairness, autonomy respect
- Two-stage evaluation: direct checks + model self-critique
- RLAIF: Scalable training with AI-generated feedback
- Flexible: Enable/disable principles, adjust weights, extend with custom principles
"""

from .framework import ConstitutionalPrinciple, ConstitutionalFramework
from .principles import (
    evaluate_harm_potential,
    evaluate_truthfulness,
    evaluate_fairness,
    evaluate_autonomy_respect,
    setup_default_framework
)
from .evaluator import ConstitutionalSafetyEvaluator, critique_indicates_issues
from .filter import ConstitutionalSafetyFilter
from .trainer import RLAIFTrainer

__all__ = [
    # Core framework
    "ConstitutionalPrinciple",
    "ConstitutionalFramework",

    # Principle evaluators
    "evaluate_harm_potential",
    "evaluate_truthfulness",
    "evaluate_fairness",
    "evaluate_autonomy_respect",
    "setup_default_framework",

    # Evaluator and filter
    "ConstitutionalSafetyEvaluator",
    "ConstitutionalSafetyFilter",
    "critique_indicates_issues",

    # Training
    "RLAIFTrainer",
]

__version__ = "0.1.0"
