# src/models/pretrained/vision_transformer.py
from .base_wrapper import PretrainedModelWrapper
from transformers import ViTModel, ViTConfig
import torch
import os

class VisionTransformerWrapper(PretrainedModelWrapper):
    """Wrapper for Hugging Face Vision Transformer models."""
    
    def __init__(self, model_name: str = "google/vit-base-patch16-224"):
        super().__init__(model_name=model_name)
        self.load_model(model_name)
        
    def load_model(self, model_name: str) -> None:
        """Load a pretrained Vision Transformer."""
        # Map shortened model names to full Hugging Face identifiers
        model_mapping = {
            "vit-base": "google/vit-base-patch16-224",
            "vit-large": "google/vit-large-patch16-224",
            "vit-base-384": "google/vit-base-patch16-384",
            "vit-large-384": "google/vit-large-patch16-384",
            "vit-huge": "google/vit-huge-patch14-224-in21k",
        }
        
        # Use the mapping if available, otherwise use the original name
        full_model_name = model_mapping.get(model_name, model_name)
        print(f"Loading vision model: {full_model_name} (from '{model_name}')")
        
        self.pretrained_model = ViTModel.from_pretrained(full_model_name)
        self.config = self.pretrained_model.config.to_dict()
        
    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the Vision Transformer.
        
        Args:
            pixel_values: Tensor of shape [batch_size, channels, height, width]
            
        Returns:
            Image embeddings of shape [batch_size, sequence_length, hidden_size]
        """
        outputs = self.pretrained_model(pixel_values=pixel_values)
        return outputs.last_hidden_state

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
        "module_purpose": "Provides a wrapper for Hugging Face Vision Transformer models with standardized interface",
        "key_classes": [
            {
                "name": "VisionTransformerWrapper",
                "purpose": "Wrapper for Hugging Face Vision Transformer models with simplified interface",
                "key_methods": [
                    {
                        "name": "__init__",
                        "signature": "__init__(self, model_name: str = \"google/vit-base-patch16-224\")",
                        "brief_description": "Initialize with specific ViT model"
                    },
                    {
                        "name": "load_model",
                        "signature": "load_model(self, model_name: str) -> None",
                        "brief_description": "Load a pretrained Vision Transformer model"
                    },
                    {
                        "name": "forward",
                        "signature": "forward(self, pixel_values: torch.Tensor) -> torch.Tensor",
                        "brief_description": "Process images through the Vision Transformer"
                    }
                ],
                "inheritance": "PretrainedModelWrapper",
                "dependencies": ["torch", "transformers"]
            }
        ],
        "external_dependencies": ["torch", "transformers"],
        "complexity_score": 3  # Low complexity
    }