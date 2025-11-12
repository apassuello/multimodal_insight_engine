"""MODULE: principles.py
PURPOSE: Hybrid constitutional principle evaluators for AI safety
KEY COMPONENTS:
- evaluate_harm_potential: AI-based evaluation with regex fallback
- evaluate_truthfulness: AI-based evaluation with regex fallback
- evaluate_fairness: AI-based evaluation with regex fallback
- evaluate_autonomy_respect: AI-based evaluation with regex fallback
- setup_default_framework: Creates framework with all principles
DEPENDENCIES: re, numpy, typing, torch (optional), framework, ai_evaluator, principles_regex_fallback
SPECIAL NOTES: Implements proper Constitutional AI using LLM evaluation (when model provided)
              Falls back to regex-based evaluation for fast checks (when model not provided)
"""

import torch
from typing import Dict, List, Any, Optional

from .framework import ConstitutionalPrinciple, ConstitutionalFramework

# Import AI evaluator for proper CAI implementation
try:
    from .ai_evaluator import (
        ai_evaluate_harm_potential,
        ai_evaluate_truthfulness,
        ai_evaluate_fairness,
        ai_evaluate_autonomy_respect
    )
    HAS_AI_EVALUATOR = True
except ImportError:
    HAS_AI_EVALUATOR = False

# Import regex fallback evaluators
from .principles_regex_fallback import (
    evaluate_harm_potential as regex_evaluate_harm,
    evaluate_truthfulness as regex_evaluate_truthfulness,
    evaluate_fairness as regex_evaluate_fairness,
    evaluate_autonomy_respect as regex_evaluate_autonomy
)


def evaluate_harm_potential(
    text: str,
    model: Optional[Any] = None,
    tokenizer: Optional[Any] = None,
    device: Optional[torch.device] = None
) -> Dict[str, Any]:
    """
    Evaluate potential for physical, psychological or social harm.
    Uses AI-based evaluation when model is provided (proper CAI implementation).
    Falls back to regex-based evaluation when model is not provided (fast checks).

    Args:
        text: Text to evaluate
        model: Optional language model for AI evaluation
        tokenizer: Optional model tokenizer for AI evaluation
        device: Optional computation device

    Returns:
        Dictionary with evaluation results including:
        - flagged: Whether harm was detected
        - reasoning/details: Explanation of the evaluation
        - method: "ai_evaluation", "regex_fallback", or "error"
    """
    # Prefer AI evaluation when model is available (proper CAI)
    if model is not None and tokenizer is not None and HAS_AI_EVALUATOR:
        result = ai_evaluate_harm_potential(text, model, tokenizer, device)
        # AI evaluation succeeded
        if result.get("method") != "none" and result.get("method") != "error":
            return result

    # Fall back to regex evaluation for fast checks
    regex_result = regex_evaluate_harm(text)
    regex_result["method"] = "regex_fallback"
    return regex_result


def evaluate_truthfulness(
    text: str,
    model: Optional[Any] = None,
    tokenizer: Optional[Any] = None,
    device: Optional[torch.device] = None
) -> Dict[str, Any]:
    """
    Evaluate whether content is misleading or deceptive.
    Uses AI-based evaluation when model is provided (proper CAI implementation).
    Falls back to regex-based evaluation when model is not provided (fast checks).

    Args:
        text: Text to evaluate
        model: Optional language model for AI evaluation
        tokenizer: Optional model tokenizer for AI evaluation
        device: Optional computation device

    Returns:
        Dictionary with evaluation results
    """
    # Prefer AI evaluation when model is available (proper CAI)
    if model is not None and tokenizer is not None and HAS_AI_EVALUATOR:
        result = ai_evaluate_truthfulness(text, model, tokenizer, device)
        if result.get("method") != "none" and result.get("method") != "error":
            return result

    # Fall back to regex evaluation
    regex_result = regex_evaluate_truthfulness(text)
    regex_result["method"] = "regex_fallback"
    return regex_result


def evaluate_fairness(
    text: str,
    model: Optional[Any] = None,
    tokenizer: Optional[Any] = None,
    device: Optional[torch.device] = None
) -> Dict[str, Any]:
    """
    Evaluate whether content treats individuals and groups fairly.
    Uses AI-based evaluation when model is provided (proper CAI implementation).
    Falls back to regex-based evaluation when model is not provided (fast checks).

    Args:
        text: Text to evaluate
        model: Optional language model for AI evaluation
        tokenizer: Optional model tokenizer for AI evaluation
        device: Optional computation device

    Returns:
        Dictionary with evaluation results
    """
    # Prefer AI evaluation when model is available (proper CAI)
    if model is not None and tokenizer is not None and HAS_AI_EVALUATOR:
        result = ai_evaluate_fairness(text, model, tokenizer, device)
        if result.get("method") != "none" and result.get("method") != "error":
            return result

    # Fall back to regex evaluation
    regex_result = regex_evaluate_fairness(text)
    regex_result["method"] = "regex_fallback"
    return regex_result


def evaluate_autonomy_respect(
    text: str,
    model: Optional[Any] = None,
    tokenizer: Optional[Any] = None,
    device: Optional[torch.device] = None
) -> Dict[str, Any]:
    """
    Evaluate whether content respects human autonomy and decision-making.
    Uses AI-based evaluation when model is provided (proper CAI implementation).
    Falls back to regex-based evaluation when model is not provided (fast checks).

    Args:
        text: Text to evaluate
        model: Optional language model for AI evaluation
        tokenizer: Optional model tokenizer for AI evaluation
        device: Optional computation device

    Returns:
        Dictionary with evaluation results
    """
    # Prefer AI evaluation when model is available (proper CAI)
    if model is not None and tokenizer is not None and HAS_AI_EVALUATOR:
        result = ai_evaluate_autonomy_respect(text, model, tokenizer, device)
        if result.get("method") != "none" and result.get("method") != "error":
            return result

    # Fall back to regex evaluation
    regex_result = regex_evaluate_autonomy(text)
    regex_result["method"] = "regex_fallback"
    return regex_result


def setup_default_framework(
    model: Optional[Any] = None,
    tokenizer: Optional[Any] = None,
    device: Optional[torch.device] = None,
    use_ai_evaluation: bool = True
) -> ConstitutionalFramework:
    """
    Setup a constitutional framework with all four core principles.

    Args:
        model: Optional language model for AI evaluation
        tokenizer: Optional model tokenizer for AI evaluation
        device: Optional computation device
        use_ai_evaluation: Whether to use AI evaluation (when model provided)

    Returns:
        ConstitutionalFramework with harm prevention, truthfulness,
        fairness, and autonomy respect principles
    """
    framework = ConstitutionalFramework(name="default_constitutional_framework")

    # Create evaluation functions with model binding if provided
    if use_ai_evaluation and model is not None and tokenizer is not None:
        # Create closures that bind the model/tokenizer
        def harm_eval(text):
            return evaluate_harm_potential(text, model, tokenizer, device)

        def truth_eval(text):
            return evaluate_truthfulness(text, model, tokenizer, device)

        def fair_eval(text):
            return evaluate_fairness(text, model, tokenizer, device)

        def autonomy_eval(text):
            return evaluate_autonomy_respect(text, model, tokenizer, device)
    else:
        # Use regex-only fallback
        harm_eval = lambda text: evaluate_harm_potential(text)
        truth_eval = lambda text: evaluate_truthfulness(text)
        fair_eval = lambda text: evaluate_fairness(text)
        autonomy_eval = lambda text: evaluate_autonomy_respect(text)

    # Add core principles inspired by Anthropic's Constitutional AI
    framework.add_principle(
        ConstitutionalPrinciple(
            name="harm_prevention",
            description="The AI should not help users plan or execute harmful activities",
            evaluation_fn=harm_eval,
            weight=2.0  # Higher weight for harm prevention
        )
    )

    framework.add_principle(
        ConstitutionalPrinciple(
            name="truthfulness",
            description="The AI should not provide misleading or deceptive information",
            evaluation_fn=truth_eval,
            weight=1.5
        )
    )

    framework.add_principle(
        ConstitutionalPrinciple(
            name="fairness",
            description="The AI should treat individuals and groups fairly and without bias",
            evaluation_fn=fair_eval,
            weight=1.0
        )
    )

    framework.add_principle(
        ConstitutionalPrinciple(
            name="autonomy_respect",
            description="The AI should respect human autonomy and decision-making",
            evaluation_fn=autonomy_eval,
            weight=1.0
        )
    )

    return framework
