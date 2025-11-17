import torch
import json
import os
import logging
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime

# Import relevant safety components
# These would be imported from their respective modules in practice
# from .constitutional_framework import ConstitutionalFramework
# from .safety_filter import SafetyFilter
# from .two_stage_evaluator import TwoStageEvaluator
# from .self_improving_safety_system import SelfImprovingSafetySystem

class CombinedSafetySystem:
    """
    A comprehensive safety system that integrates multiple safety components
    following Anthropic's layered approach to AI safety.
    
    Features:
    - Constitutional AI principles integration
    - Standard input/output filtering
    - Two-stage evaluation
    - Self-improvement mechanisms
    - Logging and monitoring
    - Explainable rejections
    """
    def __init__(
        self,
        model,
        constitutional_framework,
        safety_filter,
        enable_two_stage: bool = True,
        enable_self_improvement: bool = True,
        log_dir: str = "safety_logs",
        sensitivity: str = "medium"
    ):
        """
        Initialize the combined safety system.
        
        Args:
            model: The underlying model for generation and critique
            constitutional_framework: Framework of constitutional principles
            safety_filter: Base safety filter
            enable_two_stage: Whether to enable two-stage evaluation
            enable_self_improvement: Whether to enable self-improvement
            log_dir: Directory for safety logs
            sensitivity: Safety sensitivity level (low, medium, high)
        """
        self.model = model
        self.constitutional_framework = constitutional_framework
        self.safety_filter = safety_filter
        self.sensitivity = sensitivity
        
        # Initialize logging
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)
        self._setup_logging()
        
        # Initialize two-stage evaluation if enabled
        self.two_stage_evaluator = None
        if enable_two_stage:
            self.two_stage_evaluator = TwoStageEvaluator(
                model=model,
                constitutional_framework=constitutional_framework
            )
        
        # Initialize self-improvement if enabled
        self.self_improvement_system = None
        if enable_self_improvement:
            from datetime import datetime
            feedback_dataset_path = os.path.join(
                log_dir, f"feedback_dataset_{datetime.now().strftime('%Y%m%d')}"
            )
            self.self_improvement_system = SelfImprovingSafetySystem(
                safety_filter=safety_filter,
                model=model,
                feedback_dataset=FeedbackDataset(save_dir=feedback_dataset_path)
            )
        
        # Statistics for monitoring
        self.stats = {
            "total_requests": 0,
            "filtered_inputs": 0,
            "filtered_outputs": 0,
            "two_stage_improvements": 0,
            "constitutional_violations": {},
        }
        
        self.logger.info(f"Combined safety system initialized with sensitivity: {sensitivity}")
    
    def _setup_logging(self):
        """Set up logging for the safety system"""
        self.logger = logging.getLogger("CombinedSafety")
        self.logger.setLevel(logging.INFO)
        
        # File handler
        log_file = os.path.join(
            self.log_dir, f"safety_log_{datetime.now().strftime('%Y%m%d')}.log"
        )
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.INFO)
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.WARNING)
        
        # Create formatter and add to handlers
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)
        
        # Add handlers to logger
        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)
    
    def process_request(self, input_text: str, metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Process a request with full safety pipeline.
        
        Args:
            input_text: The input text to process
            metadata: Additional context or information
            
        Returns:
            Dictionary with processed results
        """
        # Update statistics
        self.stats["total_requests"] += 1
        
        # Add processing timestamp
        processing_info = {
            "timestamp": str(datetime.now()),
            "input_length": len(input_text),
            "safety_applied": True,
        }
        
        # 1. Input validation
        is_safe, validation_info = self.safety_filter.validate_input(input_text, metadata)
        processing_info["input_validation"] = validation_info
        
        # Log unsafe inputs
        if not is_safe:
            self.stats["filtered_inputs"] += 1
            self._log_safety_event("input_filtered", input_text, validation_info)
            
            # Generate rejection response
            return {
                "status": "rejected",
                "reason": validation_info.get("reason", "Safety concerns"),
                "suggestion": self._generate_alternative_suggestion(input_text, validation_info),
                "processing_info": processing_info
            }
        
        # 2. Generate initial response
        try:
            initial_response = self.model.generate(input_text)
            processing_info["initial_response_length"] = len(initial_response)
        except Exception as e:
            self.logger.error(f"Error generating response: {str(e)}")
            return {
                "status": "error",
                "message": f"Error generating response: {str(e)}",
                "processing_info": processing_info
            }
        
        # 3. Apply two-stage evaluation if enabled
        final_response = initial_response
        if self.two_stage_evaluator is not None:
            evaluation = self.two_stage_evaluator.evaluate_response(
                input_text, initial_response
            )
            processing_info["two_stage_evaluation"] = evaluation
            
            # Use revised response if evaluation indicates issues
            if evaluation.get("needs_revision", False):
                self.stats["two_stage_improvements"] += 1
                final_response = evaluation.get("revised_response", initial_response)
                processing_info["response_revised"] = True
        
        # 4. Output filtering
        filtered_response, filtering_info = self.safety_filter.filter_output(
            final_response, metadata
        )
        processing_info["output_filtering"] = filtering_info
        
        # Log filtered outputs
        if filtering_info.get("was_filtered", False):
            self.stats["filtered_outputs"] += 1
            self._log_safety_event("output_filtered", final_response, filtering_info)
        
        # 5. Collect feedback for self-improvement if enabled
        if self.self_improvement_system is not None:
            self.self_improvement_system.evaluate_and_collect_feedback(input_text)
        
        # 6. Update constitutional violation statistics
        self._update_constitutional_stats(validation_info, filtering_info)
        
        # Return the final processed result
        return {
            "status": "success",
            "response": filtered_response,
            "was_filtered": filtering_info.get("was_filtered", False),
            "processing_info": processing_info
        }
    
    def _generate_alternative_suggestion(self, input_text: str, validation_info: Dict[str, Any]) -> str:
        """
        Generate an alternative suggestion for rejected input.
        
        Args:
            input_text: The rejected input text
            validation_info: Validation information
            
        Returns:
            Alternative suggestion text
        """
        # Get flagged categories
        flagged_categories = []
        if "evaluation" in validation_info:
            flagged_categories = validation_info["evaluation"].get("flagged_categories", [])
        
        flagged_principles = validation_info.get("flagged_principles", [])
        
        # Generate customized suggestion based on issues
        if "harm_prevention" in flagged_principles or "harmful_instructions" in flagged_categories:
            return "Consider asking about this topic in a more general, educational context without focusing on harmful applications."
            
        elif "truthfulness" in flagged_principles:
            return "Consider requesting factual information from reliable sources on this topic."
            
        elif "fairness" in flagged_principles or "bias" in flagged_categories:
            return "Consider rephrasing to avoid generalizations about groups of people."
            
        elif "personal_information" in flagged_categories:
            return "Please avoid sharing sensitive personal information in your requests."
            
        # Generic fallback suggestion
        return "Please rephrase your request in a way that adheres to safety guidelines."
    
    def _log_safety_event(self, event_type: str, content: str, details: Dict[str, Any]) -> None:
        """
        Log safety events for monitoring and analysis.
        
        Args:
            event_type: Type of safety event
            content: Content that triggered the event
            details: Event details
        """
        # Create event record
        event = {
            "timestamp": str(datetime.now()),
            "event_type": event_type,
            "content_snippet": content[:100] + "..." if len(content) > 100 else content,
            "details": details
        }
        
        # Log to file
        log_file = os.path.join(
            self.log_dir, 
            f"safety_events_{datetime.now().strftime('%Y%m%d')}.jsonl"
        )
        
        with open(log_file, "a") as f:
            f.write(json.dumps(event) + "\n")
        
        # Log to logger
        self.logger.info(f"Safety event: {event_type} - {event['content_snippet']}")
    
    def _update_constitutional_stats(self, 
                                   validation_info: Dict[str, Any],
                                   filtering_info: Dict[str, Any]) -> None:
        """
        Update statistics on constitutional principle violations.
        
        Args:
            validation_info: Input validation information
            filtering_info: Output filtering information
        """
        # Check for constitutional violations in input
        if "constitutional_evaluation" in validation_info:
            for principle, result in validation_info["constitutional_evaluation"].items():
                if result.get("flagged", False):
                    self.stats["constitutional_violations"][principle] = self.stats["constitutional_violations"].get(principle, 0) + 1
        
        # Check for constitutional violations in output
        if "constitutional_evaluation" in filtering_info:
            for principle, result in filtering_info["constitutional_evaluation"].items():
                if result.get("flagged", False):
                    self.stats["constitutional_violations"][principle] = self.stats["constitutional_violations"].get(principle, 0) + 1
    
    def get_safety_stats(self) -> Dict[str, Any]:
        """
        Get current safety statistics.
        
        Returns:
            Dictionary with safety statistics
        """
        # Calculate percentages
        stats = self.stats.copy()
        
        if stats["total_requests"] > 0:
            stats["input_filtering_rate"] = stats["filtered_inputs"] / stats["total_requests"]
            stats["output_filtering_rate"] = stats["filtered_outputs"] / stats["total_requests"]
            
            if "two_stage_improvements" in stats:
                stats["improvement_rate"] = stats["two_stage_improvements"] / stats["total_requests"]
        
        return stats
    
    def improve_system(self, iterations: int = 10) -> Dict[str, Any]:
        """
        Trigger system improvement cycle.
        
        Args:
            iterations: Number of improvement iterations
            
        Returns:
            Results of the improvement process
        """
        if self.self_improvement_system is None:
            return {"status": "error", "message": "Self-improvement not enabled"}
        
        # Run improvement cycle
        try:
            self.self_improvement_system.improve_system(iterations)
            return {
                "status": "success", 
                "message": f"Completed {iterations} improvement iterations"
            }
        except Exception as e:
            self.logger.error(f"Error during system improvement: {str(e)}")
            return {"status": "error", "message": str(e)}
    
    def set_sensitivity(self, sensitivity: str) -> Dict[str, Any]:
        """
        Adjust safety sensitivity level.
        
        Args:
            sensitivity: Sensitivity level (low, medium, high)
            
        Returns:
            Status of sensitivity change
        """
        valid_levels = ["low", "medium", "high"]
        if sensitivity not in valid_levels:
            return {
                "status": "error", 
                "message": f"Invalid sensitivity level. Choose from: {valid_levels}"
            }
        
        # Update sensitivity in all components
        self.sensitivity = sensitivity
        
        # Update safety evaluator sensitivity if available
        if hasattr(self.safety_filter, "safety_evaluator") and hasattr(self.safety_filter.safety_evaluator, "set_sensitivity"):
            self.safety_filter.safety_evaluator.set_sensitivity(sensitivity)
        
        self.logger.info(f"Safety sensitivity set to: {sensitivity}")
        
        return {"status": "success", "message": f"Sensitivity set to {sensitivity}"}


# Helper classes that would typically be imported
class TwoStageEvaluator:
    """Placeholder for the actual TwoStageEvaluator implementation"""
    def __init__(self, model, constitutional_framework):
        self.model = model
        self.framework = constitutional_framework
    
    def evaluate_response(self, prompt, initial_response):
        """Simplified placeholder implementation"""
        return {"needs_revision": False, "revised_response": initial_response}


class FeedbackDataset:
    """Placeholder for the actual FeedbackDataset implementation"""
    def __init__(self, save_dir):
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
    
    def add_entry(self, entry):
        """Simplified placeholder implementation"""
        pass


class SelfImprovingSafetySystem:
    """Placeholder for the actual SelfImprovingSafetySystem implementation"""
    def __init__(self, safety_filter, model, feedback_dataset):
        self.safety_filter = safety_filter
        self.model = model
        self.feedback_dataset = feedback_dataset
    
    def evaluate_and_collect_feedback(self, input_text):
        """Simplified placeholder implementation"""
        return {"is_safe": True}
    
    def improve_system(self, iterations):
        """Simplified placeholder implementation"""
        pass