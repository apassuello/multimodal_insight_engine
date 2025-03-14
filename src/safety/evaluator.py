import torch
import numpy as np
from typing import Dict, List, Union, Optional, Any, Tuple
import re
import json
import os

from .utils import (
    CATEGORY_TOXICITY,
    CATEGORY_BIAS,
    CATEGORY_HARMFUL_INSTRUCTIONS,
    CATEGORY_PERSONAL_INFORMATION,
    SAFETY_CATEGORIES,
    SENSITIVITY_MEDIUM,
    SENSITIVITY_MULTIPLIERS,
)


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
    ):
        """
        Initialize the safety evaluator.

        Args:
            safety_thresholds: Dictionary mapping safety categories to threshold values
            sensitivity: Sensitivity level (low, medium, high)
            log_dir: Directory for storing safety logs
        """
        # Store sensitivity setting
        self.sensitivity = sensitivity

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

    def evaluate_text(self, text: str) -> Dict[str, Union[float, bool]]:
        """
        Evaluate text for potential safety concerns.

        Args:
            text: The text to evaluate

        Returns:
            Dictionary containing safety scores and flags
        """
        # In a real implementation, this would use more sophisticated methods
        # For now, we'll use simple keyword and pattern matching

        results = {}

        # Simple toxicity detection
        toxic_words = ["hate", "idiot", "stupid", "kill", "die", "attack"]
        toxicity_score = sum(word in text.lower() for word in toxic_words) / len(
            text.split()
        )
        results["toxicity"] = min(
            toxicity_score * 10, 1.0
        )  # Scale up for demonstration

        # Simple bias detection (very simplified example)
        bias_patterns = [
            r"\b(all|every)\s+(men|women|people from|individuals from)\b",
            r"\b(men|women) (always|never)\b",
        ]
        bias_matches = sum(
            len(re.findall(pattern, text.lower())) for pattern in bias_patterns
        )
        results["bias"] = min(bias_matches / max(len(text.split()) / 10, 1), 1.0)

        # Harmful instructions detection
        # Enhanced harmful instructions detection
        harmful_patterns = [
            # More flexible pattern for hacking-related content
            r"(?:how|steps|instructions|guide|ways)(?:.{0,20})(?:hack|break in|access|compromise)",
            # For weaponry and dangerous items
            r"(?:how|steps|instructions|guide|ways)(?:.{0,20})(?:build|make|create|construct)(?:.{0,20})(?:bomb|weapon|explosive)",
            # For general illegal activities
            r"(?:how|steps|instructions|guide|ways)(?:.{0,20})(?:forge|steal|launder|smuggle)",
            # For specific account hacking
            r"(?:hack|access|break into)(?:.{0,20})(?:account|email|password|computer)",
        ]
        harmful_matches = sum(
            len(re.findall(pattern, text.lower())) for pattern in harmful_patterns
        )
        results["harmful_instructions"] = min(harmful_matches, 1.0)

        # Personal information detection
        pii_patterns = [
            r"\b\d{3}[-.\s]?\d{2}[-.\s]?\d{4}\b",  # SSN
            r"\b\d{3}[-.\s]?\d{3}[-.\s]?\d{4}\b",  # Phone
            r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b",  # Email
            r"\b(?:\d{4}[-\s]?){3}\d{4}\b",  # Credit card with dashes or spaces
            r"\b(?:credit card|account|card)(?:.{0,15})(?:number)(?:.{0,15})(?:\d[-\d\s]{10,20}\d)\b",  # Account numbers with context
            r"\b(?:password|passphrase|pin|secret)(?:.{0,3})(?:is|:)(?:.{0,10})(?:['\"]?[\w\d!@#$%^&*()-+=]{6,}['\"]?)",  # Passwords
            r"\b(?:username|user id|login)(?:.{0,3})(?:is|:)(?:.{0,10})(?:['\"]?[\w\d!@#$%^&*()-+=]{3,}['\"]?)",  # Usernames
        ]
        pii_matches = sum(len(re.findall(pattern, text)) for pattern in pii_patterns)
        results["personal_information"] = min(pii_matches, 1.0)

        # Set flags based on thresholds
        results["flagged"] = any(
            results[category] >= self.safety_thresholds[category]
            for category in self.safety_thresholds
        )

        # Determine which categories were flagged
        results["flagged_categories"] = [
            category
            for category in self.safety_thresholds
            if results[category] >= self.safety_thresholds[category]
        ]

        return results

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
            SENSITIVITY_LOW,
            SENSITIVITY_MEDIUM,
            SENSITIVITY_HIGH,
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


# Helper function to avoid import issues in code snippets
def import_datetime():
    """Import datetime module dynamically."""
    import datetime

    return datetime
