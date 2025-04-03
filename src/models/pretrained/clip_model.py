# src/models/pretrained/clip_model.py
from .base_wrapper import PretrainedModelWrapper
import torch
import torch.nn as nn
import open_clip
import os

class CLIPModelWrapper(PretrainedModelWrapper):
    """Wrapper for OpenAI CLIP models."""
    
    def __init__(self, model_name: str = "ViT-B-32", pretrained: str = "laion2b_s34b_b79k"):
        super().__init__(model_name=model_name)
        self.pretrained = pretrained
        self.load_model(model_name)
        
    def load_model(self, model_name: str) -> None:
        """Load a pretrained CLIP model."""
        self.pretrained_model, _, self.preprocess = open_clip.create_model_and_transforms(
            model_name, 
            pretrained=self.pretrained
        )
        self.tokenizer = open_clip.get_tokenizer(model_name)
        
    def encode_image(self, image: torch.Tensor) -> torch.Tensor:
        """Encode images to the multimodal embedding space."""
        return self.pretrained_model.encode_image(image)
        
    def encode_text(self, text: list) -> torch.Tensor:
        """Encode text to the multimodal embedding space."""
        tokenized_text = self.tokenizer(text)
        return self.pretrained_model.encode_text(tokenized_text)
    
    def forward(self, images: torch.Tensor = None, texts: list = None) -> dict:
        """
        Forward pass through CLIP.
        
        Args:
            images: Optional tensor of shape [batch_size, channels, height, width]
            texts: Optional list of strings
            
        Returns:
            Dictionary with image and/or text embeddings
        """
        results = {}
        
        if images is not None:
            results['image_embeddings'] = self.encode_image(images)
            
        if texts is not None:
            results['text_embeddings'] = self.encode_text(texts)
            
        if images is not None and texts is not None:
            # Calculate similarity scores
            image_embeddings = results['image_embeddings']
            text_embeddings = results['text_embeddings']
            
            # Normalize embeddings
            image_embeddings = image_embeddings / image_embeddings.norm(dim=-1, keepdim=True)
            text_embeddings = text_embeddings / text_embeddings.norm(dim=-1, keepdim=True)
            
            # Calculate cosine similarity
            similarity = (100.0 * image_embeddings @ text_embeddings.T)
            results['similarity'] = similarity
            
        return results

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
        "module_purpose": "Provides a wrapper for OpenAI CLIP multimodal models with standardized interface",
        "key_classes": [
            {
                "name": "CLIPModelWrapper",
                "purpose": "Wrapper for OpenAI CLIP models with image-text similarity functionality",
                "key_methods": [
                    {
                        "name": "__init__",
                        "signature": "__init__(self, model_name: str = \"ViT-B-32\", pretrained: str = \"laion2b_s34b_b79k\")",
                        "brief_description": "Initialize with specific CLIP model variant and weights"
                    },
                    {
                        "name": "load_model",
                        "signature": "load_model(self, model_name: str) -> None",
                        "brief_description": "Load a pretrained CLIP model with transforms and tokenizer"
                    },
                    {
                        "name": "encode_image",
                        "signature": "encode_image(self, image: torch.Tensor) -> torch.Tensor",
                        "brief_description": "Encode images to the multimodal embedding space"
                    },
                    {
                        "name": "encode_text",
                        "signature": "encode_text(self, text: list) -> torch.Tensor",
                        "brief_description": "Encode text to the multimodal embedding space"
                    },
                    {
                        "name": "forward",
                        "signature": "forward(self, images: torch.Tensor = None, texts: list = None) -> dict",
                        "brief_description": "Process images and/or text through CLIP and compute similarities"
                    }
                ],
                "inheritance": "PretrainedModelWrapper",
                "dependencies": ["torch", "torch.nn", "open_clip"]
            }
        ],
        "external_dependencies": ["torch", "open_clip"],
        "complexity_score": 5  # Moderate complexity
    }