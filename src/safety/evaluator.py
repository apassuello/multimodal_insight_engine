"""MODULE: evaluator.py
PURPOSE: Provides a comprehensive framework for evaluating model outputs for safety concerns
KEY COMPONENTS:
- SafetyEvaluator: Main class for safety evaluation with configurable thresholds and sensitivity
- evaluate_text(): Core method for analyzing text for safety issues
- log_evaluation(): Tracks and stores safety evaluation results
DEPENDENCIES: torch, numpy, typing, re, json, os, utils (safety categories and sensitivity settings)
SPECIAL NOTES: Implements configurable sensitivity levels and threshold-based safety checks"""

import json
import os
import re
from typing import Any, Dict, Optional, Tuple

from .utils import (
    CATEGORY_BIAS,
    CATEGORY_HARMFUL_INSTRUCTIONS,
    CATEGORY_PERSONAL_INFORMATION,
    CATEGORY_TOXICITY,
    SAFETY_CATEGORIES,
    SENSITIVITY_MEDIUM,
    SENSITIVITY_MULTIPLIERS,
)

# Optional constitutional AI support
try:
    from .constitutional import ConstitutionalSafetyEvaluator
    CONSTITUTIONAL_AI_AVAILABLE = True
except ImportError:
    CONSTITUTIONAL_AI_AVAILABLE = False


class SafetyEvaluator:
    """
    A framework for evaluating model outputs for safety concerns.

    This class provides methods for:
    1. Detecting potentially unsafe inputs/outputs
    2. Tracking safety metrics over time
    3. Storing and analyzing safety evaluation results
    """

    def __init__(
        self,
        safety_thresholds: Optional[Dict[str, float]] = None,
        sensitivity: str = SENSITIVITY_MEDIUM,
        log_dir: str = "safety_data/logs",
        use_constitutional_ai: bool = False,
        constitutional_framework: Optional[Any] = None,
    ):
        """
        Initialize the safety evaluator.

        Args:
            safety_thresholds: Dictionary mapping safety categories to threshold values
            sensitivity: Sensitivity level (low, medium, high)
            log_dir: Directory for storing safety logs
            use_constitutional_ai: Whether to use constitutional AI evaluation
            constitutional_framework: Optional constitutional framework (uses default if None)
        """
        # Store sensitivity setting
        self.sensitivity = sensitivity

        # Constitutional AI support
        self.use_constitutional_ai = use_constitutional_ai and CONSTITUTIONAL_AI_AVAILABLE
        self.constitutional_evaluator = None

        if self.use_constitutional_ai:
            if constitutional_framework is None and CONSTITUTIONAL_AI_AVAILABLE:
                from .constitutional import setup_default_framework
                constitutional_framework = setup_default_framework()

            if CONSTITUTIONAL_AI_AVAILABLE:
                self.constitutional_evaluator = ConstitutionalSafetyEvaluator(
                    framework=constitutional_framework,
                    use_self_critique=False  # Can be enabled if critique model available
                )

        # Set default thresholds
        default_thresholds = {
            CATEGORY_TOXICITY: 0.5,
            CATEGORY_BIAS: 0.5,
            CATEGORY_HARMFUL_INSTRUCTIONS: 0.7,
            CATEGORY_PERSONAL_INFORMATION: 0.6,
        }

        # Apply sensitivity adjustment if using default thresholds
        if safety_thresholds is None:
            multiplier = SENSITIVITY_MULTIPLIERS.get(sensitivity, 1.0)
            adjusted_thresholds = {
                k: (
                    min(v * multiplier, 0.95)
                    if multiplier > 1
                    else max(v * multiplier, 0.2)
                )
                for k, v in default_thresholds.items()
            }
            self.safety_thresholds = adjusted_thresholds
        else:
            self.safety_thresholds = safety_thresholds

        # Create log directory if it doesn't exist
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)

        # Initialize tracking metrics
        self.safety_stats = {
            "total_evaluations": 0,
            "flagged_inputs": 0,
            "flagged_outputs": 0,
            "category_counts": {category: 0 for category in self.safety_thresholds},
        }

    def evaluate_text(self, text: str, use_constitutional: Optional[bool] = None) -> Dict[str, Any]:
        """
        Evaluate text for safety concerns.

        Args:
            text: Text to evaluate
            use_constitutional: Override for constitutional AI usage (uses instance setting if None)

        Returns:
            Dictionary with safety evaluation results
        """
        if not text or text.strip() == "":
            return {
                "flagged": False,
                "toxicity": 0.0,
                "harmful_instructions": 0.0,
                "bias": 0.0,
                "personal_information": 0.0,
                "flagged_categories": [],
                "constitutional_evaluation": None
            }

        # Convert to lowercase for case-insensitive matching
        text_lower = text.lower()

        # Check for toxic words
        toxic_words = ["hate", "idiot", "stupid", "kill", "die", "attack"]
        toxicity_score = sum(word in text_lower for word in toxic_words) / len(toxic_words)

        # Check for harmful instructions
        harmful_instructions = [
            "hacking", "hacked", "hacker", "hack", "hacking", "hacked", "hacker", "hack",
            "weaponry", "weapon", "weaponry", "weapon",
            "build", "made", "created", "construct", "bomb", "weapon", "explosive",
            "forge", "steal", "launder", "smuggle",
            "account", "email", "password", "computer",
            "credit card", "number", "card", "secret", "pin", "passphrase",
            "username", "user id", "login"
        ]
        harmful_instructions_score = sum(phrase in text_lower for phrase in harmful_instructions) / len(harmful_instructions)

        # Check for bias
        bias_words = ["all", "every", "men", "women", "people from", "individuals from"]
        bias_score = sum(word in text_lower for word in bias_words) / len(bias_words)

        # Check for personal information (use original text for regex)
        personal_info_patterns = [
            r"\b\d{3}[-.\s]?\d{2}[-.\s]?\d{4}\b",  # SSN
            r"\b\d{3}[-.\s]?\d{3}[-.\s]?\d{4}\b",  # Phone
            r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b",  # Email
            r"\b(?:\d{4}[-\s]?){3}\d{4}\b",  # Credit card with dashes or spaces
            r"\b(?:credit card|account|card)(?:.{0,15})(?:number)(?:.{0,15})(?:\d[-\d\s]{10,20}\d)\b",  # Account numbers with context
            r"\b(?:password|passphrase|pin|secret)(?:.{0,3})(?:is|:)(?:.{0,10})(?:['\"]?[\w\d!@#$%^&*()-+=]{6,}['\"]?)",  # Passwords
            r"\b(?:username|user id|login)(?:.{0,3})(?:is|:)(?:.{0,10})(?:['\"]?[\w\d!@#$%^&*()-+=]{3,}['\"]?)",  # Usernames
        ]
        personal_info_score = sum(bool(re.search(pattern, text, re.IGNORECASE)) for pattern in personal_info_patterns) / len(personal_info_patterns)

        # Determine if text should be flagged
        flagged = any([
            toxicity_score > 0.3,
            harmful_instructions_score > 0.3,
            bias_score > 0.3,
            personal_info_score > 0.3
        ])

        # Collect flagged categories
        flagged_categories = []
        if toxicity_score > 0.3:
            flagged_categories.append("toxicity")
        if harmful_instructions_score > 0.3:
            flagged_categories.append("harmful_instructions")
        if bias_score > 0.3:
            flagged_categories.append("bias")
        if personal_info_score > 0.3:
            flagged_categories.append("personal_information")

        result = {
            "flagged": flagged,
            "toxicity": toxicity_score,
            "harmful_instructions": harmful_instructions_score,
            "bias": bias_score,
            "personal_information": personal_info_score,
            "flagged_categories": flagged_categories,
            "constitutional_evaluation": None
        }

        # Apply constitutional AI evaluation if enabled
        apply_constitutional = use_constitutional if use_constitutional is not None else self.use_constitutional_ai
        if apply_constitutional and self.constitutional_evaluator is not None:
            constitutional_result = self.constitutional_evaluator.evaluate(text, include_critique=False)
            result["constitutional_evaluation"] = constitutional_result

            # Update flagged status if constitutional AI found issues
            if constitutional_result.get("flagged", False):
                result["flagged"] = True
                if "constitutional" not in flagged_categories:
                    result["flagged_categories"].append("constitutional")

        return result

    def log_evaluation(
        self,
        text: str,
        results: Dict[str, Any],
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Log safety evaluation results.

        Args:
            text: The evaluated text
            results: The evaluation results
            metadata: Additional information about the evaluation
        """
        # Update stats
        self.safety_stats["total_evaluations"] += 1
        if results["flagged"]:
            if metadata and metadata.get("is_input", False):
                self.safety_stats["flagged_inputs"] += 1
            else:
                self.safety_stats["flagged_outputs"] += 1

            for category in results["flagged_categories"]:
                self.safety_stats["category_counts"][category] += 1

        # Create log entry
        log_entry = {
            "timestamp": str(import_datetime().datetime.now()),
            "text": text[:100] + "..." if len(text) > 100 else text,  # Truncate for log
            "results": results,
            "metadata": metadata or {},
        }

        # Save to log file
        log_file = os.path.join(
            self.log_dir,
            f"safety_log_{import_datetime().datetime.now().strftime('%Y%m%d')}.jsonl",
        )
        with open(log_file, "a") as f:
            f.write(json.dumps(log_entry) + "\n")

    def get_safety_summary(self) -> Dict[str, Any]:
        """
        Get a summary of safety evaluations.

        Returns:
            Dictionary containing safety metrics
        """
        return {
            "statistics": self.safety_stats,
            "current_thresholds": self.safety_thresholds,
        }

    def set_sensitivity(self, sensitivity: str) -> None:
        """
        Change the sensitivity level of the evaluator.

        Args:
            sensitivity: New sensitivity level (low, medium, high)
        """
        from .utils import (
            SENSITIVITY_HIGH,
            SENSITIVITY_LOW,
            SENSITIVITY_MEDIUM,
            SENSITIVITY_MULTIPLIERS,
        )

        if sensitivity not in [SENSITIVITY_LOW, SENSITIVITY_MEDIUM, SENSITIVITY_HIGH]:
            raise ValueError(
                f"Invalid sensitivity level: {sensitivity}. Must be one of: low, medium, high"
            )

        # Update sensitivity
        self.sensitivity = sensitivity

        # Define default thresholds
        default_thresholds = {
            CATEGORY_TOXICITY: 0.5,
            CATEGORY_BIAS: 0.5,
            CATEGORY_HARMFUL_INSTRUCTIONS: 0.7,
            CATEGORY_PERSONAL_INFORMATION: 0.6,
        }

        # Apply new sensitivity multiplier
        multiplier = SENSITIVITY_MULTIPLIERS.get(sensitivity, 1.0)
        self.safety_thresholds = {
            k: (
                min(default_thresholds[k] * multiplier, 0.95)
                if multiplier > 1
                else max(default_thresholds[k] * multiplier, 0.2)
            )
            for k in SAFETY_CATEGORIES
        }

        print(
            f"Safety evaluator sensitivity set to {sensitivity}. New thresholds: {self.safety_thresholds}"
        )

    def validate_input(
        self,
        input_text: str,
        metadata: Optional[Dict[str, Any]] = None,
        override: bool = False
    ) -> Tuple[bool, Dict[str, Any]]:
        """
        Validate input text for safety.

        Args:
            input_text: The input text to validate
            metadata: Additional information about the input
            override: Whether to override safety concerns

        Returns:
            Tuple of (is_safe, validation_info)
        """
        if override:
            return True, {"is_safe": True, "overridden": True}

        evaluation = self.evaluate_text(input_text)

        is_safe = not evaluation["flagged"]
        validation_info = {
            "is_safe": is_safe,
            "evaluation": evaluation,
            "reason": "Passed all safety checks" if is_safe else f"Failed: {evaluation['flagged_categories']}"
        }

        return is_safe, validation_info

    def filter_output(
        self,
        output_text: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Tuple[str, Dict[str, Any]]:
        """
        Filter output text for safety.

        Args:
            output_text: The output text to filter
            metadata: Additional information about the output

        Returns:
            Tuple of (filtered_text, filtering_info)
        """
        evaluation = self.evaluate_text(output_text)

        filtering_info = {
            "was_filtered": evaluation["flagged"],
            "evaluation": evaluation
        }

        # Simple filtering: if severely problematic, return warning message
        if evaluation["flagged"] and len(evaluation["flagged_categories"]) > 2:
            filtered_text = "[Content filtered due to safety concerns]"
            filtering_info["filter_applied"] = True
        else:
            filtered_text = output_text
            filtering_info["filter_applied"] = False

        return filtered_text, filtering_info


# Helper function to avoid import issues in code snippets
def import_datetime():
    """Helper function to lazily import datetime module."""
    import datetime
    return datetime

def extract_file_metadata(file_path=__file__):
    """
    Extract structured metadata about this module.
    
    Args:
        file_path: Path to the source file (defaults to current file)
        
    Returns:
        dict: Structured metadata about the module's purpose and components
    """
    return {
        "filename": os.path.basename(file_path),
        "module_purpose": "Provides a comprehensive framework for evaluating model outputs for safety concerns with configurable sensitivity levels",
        "key_classes": [
            {
                "name": "SafetyEvaluator",
                "purpose": "Main class for safety evaluation with configurable thresholds and sensitivity levels",
                "key_methods": [
                    {
                        "name": "evaluate_text",
                        "signature": "evaluate_text(self, text: str) -> Dict[str, Any]",
                        "brief_description": "Evaluates text for safety concerns across multiple categories and returns detailed results"
                    },
                    {
                        "name": "log_evaluation",
                        "signature": "log_evaluation(self, text: str, results: Dict[str, Any], metadata: Optional[Dict[str, Any]] = None) -> None",
                        "brief_description": "Records safety evaluation results for analysis and tracking"
                    },
                    {
                        "name": "set_sensitivity",
                        "signature": "set_sensitivity(self, sensitivity: str) -> None",
                        "brief_description": "Adjusts the sensitivity level of safety checks based on application requirements"
                    },
                    {
                        "name": "get_safety_summary",
                        "signature": "get_safety_summary(self) -> Dict[str, Any]",
                        "brief_description": "Provides aggregate statistics on past evaluations and current settings"
                    }
                ],
                "inheritance": "object",
                "dependencies": ["typing", "re", "json", "os", ".utils"]
            }
        ],
        "external_dependencies": ["numpy", "torch"],
        "complexity_score": 8  # High complexity due to multiple detection methods and configuration options
    }
