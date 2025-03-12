import torch
import numpy as np
from typing import Dict, List, Union, Optional, Any, Tuple
import re
from .evaluator import SafetyEvaluator  # Adjust the import path as necessary


class SafetyFilter:
    """
    A safety filter for processing model inputs and outputs.

    This class provides methods for:
    1. Validating model inputs
    2. Filtering model outputs
    3. Providing safe fallbacks when necessary
    """

    def __init__(self, safety_evaluator: SafetyEvaluator, allow_override: bool = False):
        """
        Initialize the safety filter.

        Args:
            safety_evaluator: Instance of SafetyEvaluator for content analysis
            allow_override: Whether to allow overriding the filter in certain cases
        """
        self.safety_evaluator = safety_evaluator
        self.allow_override = allow_override

    def validate_input(
        self,
        input_text: str,
        metadata: Optional[Dict[str, Any]] = None,
        override: bool = False,
    ) -> Tuple[bool, Dict[str, Any]]:
        """
        Validate input text for safety concerns.

        Args:
            input_text: The input text to validate
            metadata: Additional information about the input
            override: Whether to override safety concerns

        Returns:
            Tuple containing:
            - Boolean indicating if input is safe
            - Dictionary with validation details
        """
        # Add input flag to metadata
        meta = metadata or {}
        meta["is_input"] = True

        # Evaluate the input text
        evaluation = self.safety_evaluator.evaluate_text(input_text)

        # Log the evaluation
        self.safety_evaluator.log_evaluation(input_text, evaluation, meta)

        # Determine if input should be accepted
        is_safe = not evaluation["flagged"] or (override and self.allow_override)

        return is_safe, {
            "is_safe": is_safe,
            "evaluation": evaluation,
            "reason": (
                "Override applied"
                if (not is_safe and override and self.allow_override)
                else ""
            ),
        }

    def filter_output(
        self, output_text: str, metadata: Optional[Dict[str, Any]] = None
    ) -> Tuple[str, Dict[str, Any]]:
        """
        Filter output text to ensure safety.

        Args:
            output_text: The output text to filter
            metadata: Additional information about the output

        Returns:
            Tuple containing:
            - Filtered (or original) output text
            - Dictionary with filtering details
        """
        # Add output flag to metadata
        meta = metadata or {}
        meta["is_input"] = False

        # Evaluate the output text
        evaluation = self.safety_evaluator.evaluate_text(output_text)

        # Log the evaluation
        self.safety_evaluator.log_evaluation(output_text, evaluation, meta)

        # If output is flagged, apply appropriate filtering
        if evaluation["flagged"]:
            filtered_output = self._redact_unsafe_content(output_text, evaluation)

            return filtered_output, {
                "was_filtered": True,
                "evaluation": evaluation,
                "original_length": len(output_text),
                "filtered_length": len(filtered_output),
            }

        return output_text, {"was_filtered": False, "evaluation": evaluation}

    def _redact_unsafe_content(self, text: str, evaluation: Dict[str, Any]) -> str:
        """
        Redact unsafe content from text.

        Args:
            text: The text to redact
            evaluation: Safety evaluation results

        Returns:
            Redacted text
        """
        redacted_text = text

        # For each flagged category, apply appropriate redaction
        for category in evaluation["flagged_categories"]:
            if category == "personal_information":
                # Redact potential PII
                pii_patterns = [
                    r"\b\d{3}[-.\s]?\d{2}[-.\s]?\d{4}\b",  # SSN
                    r"\b\d{3}[-.\s]?\d{3}[-.\s]?\d{4}\b",  # Phone
                    r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b",  # Email
                ]
                for pattern in pii_patterns:
                    redacted_text = re.sub(pattern, "[REDACTED]", redacted_text)

            elif category == "toxicity":
                # Replace toxic words with asterisks
                toxic_words = ["hate", "idiot", "stupid", "kill", "die", "attack"]
                for word in toxic_words:
                    redacted_text = re.sub(
                        r"\b" + word + r"\b",
                        "*" * len(word),
                        redacted_text,
                        flags=re.IGNORECASE,
                    )

            elif category in ["harmful_instructions", "bias"]:
                # More complex categories might require more sophisticated handling
                # For now, add a disclaimer
                disclaimer = (
                    "\n\n[Note: This content has been identified as potentially "
                    + category.replace("_", " ")
                    + ".]"
                )
                if disclaimer not in redacted_text:
                    redacted_text += disclaimer

        return redacted_text
