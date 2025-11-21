import torch
import json
import numpy as np
import os
from datetime import datetime
from typing import Dict, List, Any, Optional

class FeedbackDataset:
    """
    Dataset to store and manage feedback for safety system improvement.
    Collects examples of safety decisions for future learning.
    """
    def __init__(self, save_dir: str = "safety_feedback"):
        """
        Initialize the feedback dataset.
        
        Args:
            save_dir: Directory to save feedback data
        """
        self.save_dir = save_dir
        self.data = []
        os.makedirs(save_dir, exist_ok=True)
        
    def add_entry(self, entry: Dict[str, Any]):
        """
        Add a new feedback entry to the dataset.
        
        Args:
            entry: Dictionary with feedback data
        """
        # Add timestamp
        entry["timestamp"] = str(datetime.now())
        
        # Add to memory
        self.data.append(entry)
        
        # Save to disk
        self._save_entry(entry)
        
    def _save_entry(self, entry: Dict[str, Any]):
        """
        Save a single entry to disk.
        
        Args:
            entry: Dictionary with feedback data
        """
        # Create filename based on timestamp
        filename = os.path.join(
            self.save_dir, 
            f"feedback_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        )
        
        # Save as JSON
        with open(filename, "w") as f:
            json.dump(entry, f, indent=2)
            
    def get_batch(self, batch_size: int = 16) -> List[Dict[str, Any]]:
        """
        Get a batch of feedback entries.
        
        Args:
            batch_size: Number of entries to retrieve
            
        Returns:
            List of feedback entries
        """
        if len(self.data) == 0:
            return []
            
        # Sample randomly from available data
        indices = np.random.choice(len(self.data), min(batch_size, len(self.data)), replace=False)
        return [self.data[i] for i in indices]
        
    def load_from_disk(self):
        """Load all feedback entries from disk."""
        self.data = []
        
        # Find all JSON files in the save directory
        for filename in os.listdir(self.save_dir):
            if filename.endswith(".json") and filename.startswith("feedback_"):
                filepath = os.path.join(self.save_dir, filename)
                
                try:
                    with open(filepath, "r") as f:
                        entry = json.load(f)
                        self.data.append(entry)
                except Exception as e:
                    print(f"Error loading {filepath}: {e}")
        
        print(f"Loaded {len(self.data)} feedback entries from disk")

class SelfImprovingSafetySystem:
    """
    Safety system that learns from its own evaluations,
    implementing a key aspect of Constitutional AI's approach.
    """
    def __init__(self, safety_filter, model, feedback_dataset: Optional[FeedbackDataset] = None):
        """
        Initialize the self-improving safety system.
        
        Args:
            safety_filter: Filter for safety evaluation
            model: Model to generate alternatives and critiques
            feedback_dataset: Dataset for storing feedback
        """
        self.safety_filter = safety_filter
        self.model = model
        self.feedback_dataset = feedback_dataset or FeedbackDataset()
        
    def evaluate_and_collect_feedback(self, input_text: str):
        """
        Evaluate input and collect feedback for improvement.
        
        Args:
            input_text: Text to evaluate
            
        Returns:
            Evaluation results and optional safe alternative
        """
        # Standard evaluation
        is_safe, validation_info = self.safety_filter.validate_input(input_text)
        
        # Generate alternative safe response for comparison if not safe
        safe_alternative = None
        if not is_safe:
            safe_alternative = self.generate_safe_alternative(input_text, validation_info)
            
            # Store the pair for learning
            self.feedback_dataset.add_entry({
                "original_input": input_text,
                "safety_evaluation": validation_info,
                "safe_alternative": safe_alternative
            })
            
        return {
            "is_safe": is_safe,
            "validation_info": validation_info,
            "safe_alternative": safe_alternative
        }
        
    def generate_safe_alternative(self, input_text: str, validation_info: Dict[str, Any]) -> str:
        """
        Generate a safe alternative to potentially unsafe input.
        
        Args:
            input_text: Original potentially unsafe input
            validation_info: Safety evaluation information
            
        Returns:
            Safe alternative text
        """
        # Determine which principles were violated
        flagged_categories = []
        if "evaluation" in validation_info:
            if "flagged_categories" in validation_info["evaluation"]:
                flagged_categories = validation_info["evaluation"]["flagged_categories"]
        
        # Construct a prompt for the model to generate a safe alternative
        principles = ", ".join(flagged_categories) if flagged_categories else "safety guidelines"
        
        prompt = f"""
        The following input may violate {principles}.
        
        Original input: "{input_text}"
        
        Please provide a revised version that addresses any safety concerns while 
        preserving the core intention where possible:
        """
        
        # Generate alternative using the model
        # In practice, this would use a proper call to the LLM
        # Simplified implementation for illustration
        safe_alternative = self._mock_model_generate(prompt, input_text, flagged_categories)
        
        return safe_alternative
        
    def _mock_model_generate(self, prompt: str, original_text: str, flagged_categories: List[str]) -> str:
        """
        Mock implementation of model generation.
        In practice, this would call a real LLM.
        
        Args:
            prompt: Prompt for generation
            original_text: Original text
            flagged_categories: Categories that were flagged
            
        Returns:
            Generated text
        """
        # Simple rule-based mock for illustration
        if "harm_prevention" in flagged_categories:
            return f"I understand you're interested in learning about this topic. Instead of specific harm-causing instructions, I can provide information about the general subject in a safe and educational manner."
            
        if "truthfulness" in flagged_categories:
            return f"I'd be happy to discuss this topic, but I want to make sure I provide accurate information. Here's what I know based on reliable sources..."
            
        if "fairness" in flagged_categories:
            return f"I'd be happy to discuss this topic while considering diverse perspectives and avoiding generalizations about groups of people."
            
        # Default safe alternative
        return f"I'd be happy to help with a similar request that adheres to our content guidelines. Could you rephrase your request?"
        
    def improve_system(self, iterations: int = 10):
        """
        Use collected feedback to improve safety system.
        
        Args:
            iterations: Number of improvement iterations
        """
        print(f"Starting system improvement with {iterations} iterations")
        
        # In practice, this would train models and update thresholds
        # based on the collected feedback
        for i in range(iterations):
            # Get a batch of feedback examples
            batch = self.feedback_dataset.get_batch()
            if not batch:
                print("No feedback data available for improvement")
                break
                
            print(f"Iteration {i+1}/{iterations}: Processing {len(batch)} feedback examples")
            
            # Analyze patterns in feedback
            categories = {}
            for entry in batch:
                # Count frequency of flagged categories
                if "safety_evaluation" in entry and "evaluation" in entry["safety_evaluation"]:
                    eval_data = entry["safety_evaluation"]["evaluation"]
                    if "flagged_categories" in eval_data:
                        for category in eval_data["flagged_categories"]:
                            categories[category] = categories.get(category, 0) + 1
            
            # In a real system, would use this analysis to:
            # 1. Tune category-specific thresholds
            # 2. Update detection patterns
            # 3. Fine-tune models
            
            print(f"Category distribution in feedback: {categories}")
            
            # In practice, would update the safety filter based on analysis
            # self.safety_filter.update_thresholds(...)
        
        print("System improvement completed")