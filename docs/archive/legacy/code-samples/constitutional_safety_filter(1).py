class ConstitutionalSafetyFilter:
    """
    Safety filter enhanced with constitutional principles.
    Extends the base safety filter with constitutional evaluation.
    """
    def __init__(self, safety_evaluator, constitutional_framework):
        """
        Initialize the constitutional safety filter.
        
        Args:
            safety_evaluator: Base safety evaluator
            constitutional_framework: Framework with principles
        """
        self.safety_evaluator = safety_evaluator
        self.constitutional_framework = constitutional_framework
        
    def validate_input(self, input_text, metadata=None, override=False):
        """
        Validate input using both standard safety and constitutional principles.
        
        Args:
            input_text: Text to validate
            metadata: Additional context information
            override: Whether to override safety checks
            
        Returns:
            Tuple of (is_safe, validation_info)
        """
        # First, perform standard safety validation
        is_safe, validation_info = self.safety_evaluator.validate_input(
            input_text, metadata, override
        )
        
        # If passed standard validation, apply constitutional evaluation
        if is_safe:
            constitutional_evaluation = self.constitutional_framework.evaluate_text(input_text)
            constitutional_issues = any(constitutional_evaluation.values())
            
            if constitutional_issues:
                is_safe = False
                validation_info["constitutional_evaluation"] = constitutional_evaluation
                validation_info["is_safe"] = False
                validation_info["reason"] = "Failed constitutional principles"
        
        return is_safe, validation_info
        
    def filter_output(self, output_text, metadata=None):
        """
        Filter output using both standard safety and constitutional principles.
        
        Args:
            output_text: Text to filter
            metadata: Additional context information
            
        Returns:
            Tuple of (filtered_text, filtering_info)
        """
        # First apply standard output filtering
        filtered_text, filtering_info = self.safety_evaluator.filter_output(
            output_text, metadata
        )
        
        # If not already filtered, check constitutional principles
        if not filtering_info.get("was_filtered", False):
            constitutional_evaluation = self.constitutional_framework.evaluate_text(filtered_text)
            constitutional_issues = any(constitutional_evaluation.values())
            
            if constitutional_issues:
                # Apply constitutional filtering
                filtered_text = self._apply_constitutional_filtering(
                    filtered_text, constitutional_evaluation
                )
                
                # Update filtering info
                filtering_info["was_filtered"] = True
                filtering_info["constitutional_evaluation"] = constitutional_evaluation
                filtering_info["filtering_reason"] = "Constitutional principles"
        
        return filtered_text, filtering_info
    
    def _apply_constitutional_filtering(self, text, evaluation):
        """
        Apply filtering based on constitutional evaluation.
        
        Args:
            text: Text to filter
            evaluation: Constitutional evaluation results
            
        Returns:
            Filtered text
        """
        # This would implement sophisticated filtering based on specific principles
        # Simple implementation for illustration
        if "harm_prevention" in evaluation:
            # Remove harmful content
            pass
            
        if "truthfulness" in evaluation:
            # Add disclaimers for potentially misleading content
            text += "\n\nNote: Some claims in this content may require additional verification."
            
        # Return filtered text (in this simplified example, mostly unchanged)
        return text