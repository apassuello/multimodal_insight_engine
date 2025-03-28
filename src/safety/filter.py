# src/safety/filter.py

import re
from typing import Dict, List, Union, Optional, Any, Tuple
from .evaluator import SafetyEvaluator
from .utils import (
    CATEGORY_TOXICITY,
    CATEGORY_BIAS,
    CATEGORY_HARMFUL_INSTRUCTIONS,
    CATEGORY_PERSONAL_INFORMATION,
    PII_PATTERNS,
    TOXICITY_PATTERNS,
)
import os

"""MODULE: filter.py
PURPOSE: Implements safety filtering mechanisms for model inputs and outputs
KEY COMPONENTS:
- SafetyFilter: Main class for validating inputs and filtering outputs
- validate_input(): Checks input text for safety concerns
- filter_output(): Filters output text to ensure safety
DEPENDENCIES: re, typing, evaluator (SafetyEvaluator), utils (safety categories and patterns)
SPECIAL NOTES: Implements content redaction and filtering based on safety evaluation results"""


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
            if category == CATEGORY_PERSONAL_INFORMATION:
                # Redact potential PII
                for pattern_name, pattern in PII_PATTERNS.items():
                    redacted_text = re.sub(
                        pattern, "[REDACTED PII]", redacted_text, flags=re.IGNORECASE
                    )

            elif category == CATEGORY_TOXICITY:
                # Replace toxic words with asterisks
                for pattern_name, pattern in TOXICITY_PATTERNS.items():
                    # Find all matches
                    matches = re.finditer(pattern, redacted_text, re.IGNORECASE)

                    # Create a list of replacements (to avoid modifying string while iterating)
                    replacements: List[Tuple[int, int, str]] = []
                    for match in matches:
                        word = match.group(0)
                        replacements.append(
                            (match.start(), match.end(), "*" * len(word))
                        )

                    # Apply replacements in reverse order to maintain indices
                    for start, end, replacement in sorted(replacements, reverse=True):
                        redacted_text = (
                            redacted_text[:start] + replacement + redacted_text[end:]
                        )

            elif category in [CATEGORY_HARMFUL_INSTRUCTIONS, CATEGORY_BIAS]:
                # More complex categories might require more sophisticated handling
                # For now, add a disclaimer
                disclaimer = f"\n\n[Note: This content has been identified as potentially containing {category.replace('_', ' ')}.]"
                if disclaimer not in redacted_text:
                    redacted_text += disclaimer

        return redacted_text

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
        "module_purpose": "Implements safety filtering mechanisms for validating model inputs and filtering outputs based on safety evaluation results",
        "key_classes": [
            {
                "name": "SafetyFilter",
                "purpose": "Main class for validating inputs and filtering outputs based on safety concerns",
                "key_methods": [
                    {
                        "name": "validate_input",
                        "signature": "def validate_input(self, input_text: str, metadata: Optional[Dict[str, Any]] = None, override: bool = False) -> Tuple[bool, Dict[str, Any]]",
                        "brief_description": "Validates input text for safety concerns with optional override"
                    },
                    {
                        "name": "filter_output",
                        "signature": "def filter_output(self, output_text: str, metadata: Optional[Dict[str, Any]] = None) -> Tuple[str, Dict[str, Any]]",
                        "brief_description": "Filters output text to ensure safety and remove unsafe content"
                    },
                    {
                        "name": "_redact_unsafe_content",
                        "signature": "def _redact_unsafe_content(self, text: str, evaluation: Dict[str, Any]) -> str",
                        "brief_description": "Redacts unsafe content from text based on safety evaluation"
                    }
                ],
                "inheritance": "object",
                "dependencies": ["re", "typing", "evaluator", "utils"]
            }
        ],
        "external_dependencies": [],
        "complexity_score": 8,  # Complex due to content filtering, redaction, and safety evaluation integration
    }
