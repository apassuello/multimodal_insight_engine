import torch
from PIL import Image
import numpy as np

class MultimodalSafetyEvaluator:
    """
    Safety evaluator extended to handle multimodal content.
    Evaluates both text and images for safety concerns.
    """
    def __init__(self, text_safety_evaluator, image_safety_model=None):
        """
        Initialize the multimodal safety evaluator.
        
        Args:
            text_safety_evaluator: Evaluator for text content
            image_safety_model: Optional model for image content evaluation
        """
        self.text_safety_evaluator = text_safety_evaluator
        self.image_safety_model = image_safety_model
        
    def evaluate_multimodal_content(self, text, image):
        """
        Evaluate both text and image content for safety.
        
        Args:
            text: Text content to evaluate
            image: Image content to evaluate (PIL Image or tensor)
            
        Returns:
            Dictionary with combined safety evaluation results
        """
        # Evaluate text safety
        text_evaluation = self.text_safety_evaluator.evaluate_text(text)
        
        # Evaluate image safety
        image_evaluation = self.evaluate_image(image)
        
        # Evaluate text-image relationship for additional risks
        # (e.g., text that seems innocent but refers to harmful image content)
        relationship_evaluation = self.evaluate_text_image_relationship(text, image)
        
        # Combined evaluation
        combined_evaluation = {
            "text_evaluation": text_evaluation,
            "image_evaluation": image_evaluation,
            "relationship_evaluation": relationship_evaluation,
            "flagged": (text_evaluation.get("flagged", False) or 
                       image_evaluation.get("flagged", False) or
                       relationship_evaluation.get("flagged", False)),
            "flagged_categories": list(set(
                text_evaluation.get("flagged_categories", []) + 
                image_evaluation.get("flagged_categories", []) +
                relationship_evaluation.get("flagged_categories", [])
            ))
        }
        
        return combined_evaluation
        
    def evaluate_image(self, image):
        """
        Evaluate image for safety concerns.
        
        Args:
            image: Image to evaluate (PIL Image, path, or tensor)
            
        Returns:
            Dictionary with image safety evaluation results
        """
        # Handle different image input types
        if isinstance(image, str):
            image = Image.open(image).convert('RGB')
            
        if self.image_safety_model is None:
            # Fallback to a simple evaluation if no model is provided
            return {"flagged": False, "flagged_categories": []}
            
        # Use the image safety model for evaluation
        return self.image_safety_model(image)
        
    def evaluate_text_image_relationship(self, text, image):
        """
        Evaluate the relationship between text and image for potential risks.
        
        Args:
            text: Text content
            image: Image content
            
        Returns:
            Dictionary with relationship safety evaluation
        """
        # This would analyze how text and image interact
        # For example, text might ask for analysis of harmful content in an image
        # In a full implementation, this would use a multimodal model to analyze
        # the relationship between text and image
        
        # Simple implementation for illustration
        text_lower = text.lower()
        suspicious_phrases = [
            "explain this image",
            "what does this show",
            "can you describe",
            "what do you see",
            "analyze this picture"
        ]
        
        # Check if text contains suspicious phrases that might be asking
        # for description of harmful content
        potentially_suspicious = any(phrase in text_lower for phrase in suspicious_phrases)
        
        return {
            "flagged": False,  # Would be determined by a proper model
            "potentially_suspicious": potentially_suspicious,
            "flagged_categories": []
        }
        
    def get_safe_response(self, evaluation, original_prompt=None):
        """
        Generate a safe response based on evaluation results.
        
        Args:
            evaluation: Safety evaluation results
            original_prompt: Optional original user prompt
            
        Returns:
            Safe response or explanation for rejection
        """
        if not evaluation["flagged"]:
            return None  # No safety issues, return None to indicate original response can be used
            
        # Create appropriate response based on flagged categories
        categories = evaluation["flagged_categories"]
        
        if any(cat in ["violence", "graphic", "explicit", "nsfw"] for cat in categories):
            return "I'm unable to process this content as it appears to contain material that violates content policies."
            
        if "harmful_instructions" in categories:
            return "I'm unable to assist with this request as it appears to involve potentially harmful instructions."
            
        if "personal_information" in categories:
            return "I've detected what appears to be personal information. For privacy and security reasons, I cannot process this type of content."
            
        # Generic response for other cases
        return "I'm unable to process this request due to content safety guidelines."