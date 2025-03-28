# src/safety/integration.py

from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime
from .filter import SafetyFilter
import os

"""MODULE: integration.py
PURPOSE: Provides integration layer for augmenting models with safety mechanisms
KEY COMPONENTS:
- SafetyAugmentedModel: Wrapper class that adds safety checks to base models
- predict(): Main method for safe model inference
- _generate_rejection_message(): Handles safety rejection responses
DEPENDENCIES: typing, datetime, filter (SafetyFilter)
SPECIAL NOTES: Implements three-stage safety pipeline: input validation, model inference, output filtering"""


class SafetyAugmentedModel:
    """
    A wrapper for augmenting a model with safety mechanisms.

    This class integrates safety evaluation, input validation,
    and output filtering with an underlying model.
    """

    def __init__(
        self,
        base_model: Any,
        safety_filter: SafetyFilter,
        enable_input_validation: bool = True,
        enable_output_filtering: bool = True,
        safe_mode: bool = True,
    ):
        """
        Initialize the safety-augmented model.

        Args:
            base_model: The underlying model to augment
            safety_filter: Instance of SafetyFilter
            enable_input_validation: Whether to validate inputs
            enable_output_filtering: Whether to filter outputs
            safe_mode: Whether to operate in safe mode
        """
        self.base_model = base_model
        self.safety_filter = safety_filter
        self.enable_input_validation = enable_input_validation
        self.enable_output_filtering = enable_output_filtering
        self.safe_mode = safe_mode

        # Store safety events for monitoring
        self.safety_events = []

    def predict(
        self, input_text: str, metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Generate a prediction with safety guardrails.

        Args:
            input_text: The input text
            metadata: Additional information about the request

        Returns:
            Dictionary containing output and safety information
        """
        meta = metadata or {}
        result = {"input": input_text, "safety_info": {}}

        # Step 1: Input validation
        if self.enable_input_validation:
            is_safe, validation_info = self.safety_filter.validate_input(
                input_text, meta
            )
            result["safety_info"]["input_validation"] = validation_info

            if not is_safe and self.safe_mode:
                # Return early with safety rejection
                result["output"] = self._generate_rejection_message(validation_info)
                result["was_rejected"] = True

                # Log safety event
                self._log_safety_event("input_rejected", input_text, result)

                return result

        # Step 2: Model inference
        try:
            # Call the base model to generate output
            if hasattr(self.base_model, "predict"):
                output_text = self.base_model.predict(input_text)
            else:
                # Assume the model itself is callable
                output_text = self.base_model(input_text)

            # Store raw output for reference
            result["raw_output"] = output_text

        except Exception as e:
            # Handle model errors
            result["error"] = str(e)
            result["output"] = f"An error occurred: {str(e)}"
            return result

        # Step 3: Output filtering
        if self.enable_output_filtering:
            filtered_output, filtering_info = self.safety_filter.filter_output(
                output_text, meta
            )
            result["safety_info"]["output_filtering"] = filtering_info

            if filtering_info["was_filtered"]:
                # Log safety event
                self._log_safety_event("output_filtered", output_text, filtering_info)

            result["output"] = filtered_output
        else:
            result["output"] = output_text

        # Set rejection flag
        result["was_rejected"] = False

        return result

    def _generate_rejection_message(self, validation_info: Dict[str, Any]) -> str:
        """
        Generate an appropriate rejection message based on validation info.

        Args:
            validation_info: Information from input validation

        Returns:
            Rejection message
        """
        categories = validation_info["evaluation"].get("flagged_categories", [])

        if "harmful_instructions" in categories:
            return "I'm unable to provide assistance with that request as it appears to involve potentially harmful instructions."
        elif "toxicity" in categories:
            return "I'm unable to process that input as it contains language that violates our content policy."
        elif "bias" in categories:
            return "I'm unable to engage with that input as it contains potentially biased statements."
        elif "personal_information" in categories:
            return "I've detected what appears to be personal information in your input. For privacy and security reasons, please remove sensitive information before proceeding."
        else:
            return "I'm unable to process that input due to safety concerns. Please try rephrasing your request."

    def _log_safety_event(
        self, event_type: str, content: str, details: Dict[str, Any]
    ) -> None:
        """
        Log a safety event for monitoring.

        Args:
            event_type: Type of safety event
            content: The content that triggered the event
            details: Additional details about the event
        """
        event = {
            "timestamp": str(datetime.now()),
            "event_type": event_type,
            "content_snippet": content[:100] + "..." if len(content) > 100 else content,
            "details": details,
        }

        self.safety_events.append(event)

        # In a production system, you might want to send these events to a monitoring service

    def get_safety_events(self) -> List[Dict[str, Any]]:
        """
        Get the list of safety events.

        Returns:
            List of safety events
        """
        return self.safety_events

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
        "module_purpose": "Provides integration layer for augmenting models with safety mechanisms, including input validation, output filtering, and safety event logging",
        "key_classes": [
            {
                "name": "SafetyAugmentedModel",
                "purpose": "Wrapper class that adds safety checks and filtering to base models",
                "key_methods": [
                    {
                        "name": "predict",
                        "signature": "def predict(self, input_text: str, metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]",
                        "brief_description": "Main method for safe model inference with input validation and output filtering"
                    },
                    {
                        "name": "_generate_rejection_message",
                        "signature": "def _generate_rejection_message(self, validation_info: Dict[str, Any]) -> str",
                        "brief_description": "Generates appropriate rejection messages based on safety violations"
                    },
                    {
                        "name": "_log_safety_event",
                        "signature": "def _log_safety_event(self, event_type: str, content: str, details: Dict[str, Any]) -> None",
                        "brief_description": "Logs safety-related events for monitoring"
                    }
                ],
                "inheritance": "object",
                "dependencies": ["typing", "datetime", "filter"]
            }
        ],
        "external_dependencies": [],
        "complexity_score": 8,  # Complex due to multi-stage safety pipeline, error handling, and event logging
    }
