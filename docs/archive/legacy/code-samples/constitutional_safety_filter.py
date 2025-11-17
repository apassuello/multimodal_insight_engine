from typing import Dict, Any, Optional, Tuple
import re

class ConstitutionalSafetyFilter:
    """
    Safety filter enhanced with constitutional principles from Anthropic's
    Constitutional AI approach. Extends standard safety filtering with
    principled evaluation of inputs and outputs.
    """
    def __init__(self, safety_evaluator, constitutional_framework):
        """
        Initialize the constitutional safety filter.
        
        Args:
            safety_evaluator: Base safety evaluator
            constitutional_framework: Framework of constitutional principles
        """
        self.safety_evaluator = safety_evaluator
        self.constitutional_framework = constitutional_framework
        
    def validate_input(self, input_text: str, metadata: Optional[Dict[str, Any]] = None, 
                      override: bool = False) -> Tuple[bool, Dict[str, Any]]:
        """
        Validate input text using both standard safety checks and
        constitutional principles.
        
        Args:
            input_text: The input text to validate
            metadata: Additional information about the input
            override: Whether to override safety concerns
            
        Returns:
            Tuple containing:
            - Boolean indicating if input is safe
            - Dictionary with validation details
        """
        # First, perform standard safety validation
        is_safe, validation_info = self.safety_evaluator.validate_input(
            input_text, metadata, override
        )
        
        # If passed standard validation, apply constitutional evaluation
        if is_safe:
            constitutional_evaluation = self.constitutional_framework.evaluate_text(input_text)
            constitutional_issues = any(
                result.get("flagged", False) 
                for result in constitutional_evaluation.values()
            )
            
            if constitutional_issues:
                is_safe = False
                validation_info["constitutional_evaluation"] = constitutional_evaluation
                validation_info["is_safe"] = False
                validation_info["reason"] = "Failed constitutional principles"
                
                # Add detailed information about which principles were violated
                flagged_principles = [
                    principle 
                    for principle, result in constitutional_evaluation.items() 
                    if result.get("flagged", False)
                ]
                validation_info["flagged_principles"] = flagged_principles
        
        return is_safe, validation_info
    
    def filter_output(self, output_text: str, metadata: Optional[Dict[str, Any]] = None) -> Tuple[str, Dict[str, Any]]:
        """
        Filter output text using both standard safety filters and 
        constitutional principles.
        
        Args:
            output_text: The output text to filter
            metadata: Additional information about the output
            
        Returns:
            Tuple containing:
            - Filtered output text
            - Dictionary with filtering details
        """
        # First apply standard safety filtering
        filtered_text, filtering_info = self.safety_evaluator.filter_output(
            output_text, metadata
        )
        
        # Apply constitutional evaluation
        constitutional_evaluation = self.constitutional_framework.evaluate_text(filtered_text)
        constitutional_issues = any(
            result.get("flagged", False) 
            for result in constitutional_evaluation.values()
        )
        
        # If constitutional issues remain after standard filtering,
        # apply additional constitutional filtering
        if constitutional_issues:
            filtering_info["constitutional_evaluation"] = constitutional_evaluation
            filtering_info["was_filtered"] = True
            
            # Apply constitutional filtering
            constitutionally_filtered_text = self._apply_constitutional_filtering(
                filtered_text, constitutional_evaluation
            )
            
            # Update filtering info
            filtering_info["constitutional_filtering_applied"] = True
            filtering_info["original_length"] = len(output_text)
            filtering_info["standard_filtered_length"] = len(filtered_text)
            filtering_info["final_length"] = len(constitutionally_filtered_text)
            
            return constitutionally_filtered_text, filtering_info
        
        return filtered_text, filtering_info
    
    def _apply_constitutional_filtering(self, text: str, 
                                      evaluation: Dict[str, Any]) -> str:
        """
        Apply constitutional principles to filter text.
        
        Args:
            text: Text to filter
            evaluation: Constitutional evaluation results
            
        Returns:
            Filtered text
        """
        filtered_text = text
        
        # Check each principle and apply appropriate filtering
        for principle, result in evaluation.items():
            if result.get("flagged", False):
                if principle == "harm_prevention":
                    # Replace harmful instructions or content
                    filtered_text = self._filter_harmful_content(
                        filtered_text, result
                    )
                    
                elif principle == "truthfulness":
                    # Add qualification to potentially misleading content
                    filtered_text = self._add_truthfulness_qualifiers(
                        filtered_text, result
                    )
                    
                elif principle == "fairness":
                    # Modify biased or unfair language
                    filtered_text = self._filter_biased_content(
                        filtered_text, result
                    )
                    
                elif principle == "autonomy_respect":
                    # Modify coercive or manipulative language
                    filtered_text = self._filter_autonomy_violations(
                        filtered_text, result
                    )
        
        return filtered_text
    
    def _filter_harmful_content(self, text: str, evaluation: Dict[str, Any]) -> str:
        """Filter content that could cause harm"""
        if "explicit_harm_detected" in evaluation and evaluation["explicit_harm_detected"]:
            # Replace explicit harmful instructions
            harmful_patterns = [
                r"how to (harm|hurt|injure|kill|damage)",
                r"ways to (harm|hurt|injure|kill|damage)",
                r"instructions for (harming|hurting|injuring|killing)"
            ]
            
            for pattern in harmful_patterns:
                text = re.sub(
                    pattern, 
                    "I cannot provide harmful instructions. Instead, I'd suggest ", 
                    text, 
                    flags=re.IGNORECASE
                )
        
        return text
    
    def _add_truthfulness_qualifiers(self, text: str, evaluation: Dict[str, Any]) -> str:
        """Add qualifiers to potentially misleading content"""
        # Add disclaimer for content with many unsupported claims
        if "unsupported_claims" in evaluation and len(evaluation["unsupported_claims"]) > 2:
            disclaimer = "\n\nNote: Some statements above represent opinions or claims that may require additional verification."
            if disclaimer not in text:
                text += disclaimer
        
        # Add specific qualifications for contradictions
        if "contradictions" in evaluation and evaluation["contradictions"]:
            contradiction_note = "\n\nPlease note there may be some inconsistencies in the information provided."
            if contradiction_note not in text:
                text += contradiction_note
        
        return text
    
    def _filter_biased_content(self, text: str, evaluation: Dict[str, Any]) -> str:
        """Filter biased or unfair language"""
        # Replace stereotyping language with more nuanced phrasing
        if "stereotypes" in evaluation:
            for stereotype in evaluation["stereotypes"]:
                # This is a simplified example - real implementation would be more sophisticated
                text = text.replace(
                    stereotype, 
                    "individuals vary widely and generalizations should be avoided"
                )
        
        return text
    
    def _filter_autonomy_violations(self, text: str, evaluation: Dict[str, Any]) -> str:
        """Filter content that may violate autonomy"""
        # Replace coercive language with more respectful alternatives
        if "coercive_language" in evaluation:
            # This is a simplified example - real implementation would be more sophisticated
            coercive_patterns = [
                (r"(must|have to|need to|required to)\s+", "may consider "),
                (r"(no choice but to|only option is to)", "one option might be to"),
                (r"(cannot|must not|should not)\s+", "might want to reconsider ")
            ]
            
            for pattern, replacement in coercive_patterns:
                text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
        
        return text