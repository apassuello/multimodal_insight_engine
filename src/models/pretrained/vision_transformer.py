# src/models/pretrained/vision_transformer.py
from .base_wrapper import PretrainedModelWrapper
from transformers import ViTModel, ViTConfig
import torch

class VisionTransformerWrapper(PretrainedModelWrapper):
    """Wrapper for Hugging Face Vision Transformer models."""
    
    def __init__(self, model_name: str = "google/vit-base-patch16-224"):
        super().__init__(model_name=model_name)
        self.load_model(model_name)
        
    def load_model(self, model_name: str) -> None:
        """Load a pretrained Vision Transformer."""
        self.pretrained_model = ViTModel.from_pretrained(model_name)
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