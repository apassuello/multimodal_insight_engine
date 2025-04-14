# src/models/vision/multimodal_integration.py
import torch
import torch.nn as nn
from typing import Dict, Optional, Tuple

from ..base_model import BaseModel
from ..transformer import EncoderDecoderTransformer
from .vision_transformer import VisionTransformer


class MultiModalTransformer(BaseModel):
    """
    MultiModal Transformer combining vision and text capabilities.

    This is a simple integration that processes images and text separately,
    and will be enhanced in future weeks for deeper integration.
    """

    def __init__(
        self,
        vision_model: VisionTransformer,
        text_model: EncoderDecoderTransformer,
        projection_dim: int = 512,
        dropout: float = 0.1,
    ):
        """
        Initialize the multimodal transformer.

        Args:
            vision_model: Vision Transformer model
            text_model: Text Transformer model
            projection_dim: Dimension for feature projection
            dropout: Dropout probability
        """
        super().__init__()

        self.vision_model = vision_model
        self.text_model = text_model

        # Get embedding dimensions from models
        vision_dim = vision_model.embed_dim
        text_dim = text_model.d_model

        # Projection layers to common space
        self.vision_projection = nn.Sequential(
            nn.Linear(vision_dim, projection_dim),
            nn.LayerNorm(projection_dim),
            nn.Dropout(dropout),
        )

        self.text_projection = nn.Sequential(
            nn.Linear(text_dim, projection_dim),
            nn.LayerNorm(projection_dim),
            nn.Dropout(dropout),
        )

        # Initialize projection layers
        nn.init.xavier_uniform_(self.vision_projection[0].weight)
        nn.init.zeros_(self.vision_projection[0].bias)
        nn.init.xavier_uniform_(self.text_projection[0].weight)
        nn.init.zeros_(self.text_projection[0].bias)

    def encode_image(self, image: torch.Tensor) -> torch.Tensor:
        """
        Encode an image to the multimodal embedding space.

        Args:
            image: Image tensor of shape [B, C, H, W]

        Returns:
            Image features of shape [B, projection_dim]
        """
        # Pass image through vision model
        with torch.no_grad():
            vision_features = self.vision_model.extract_features(image)

        # Project to common space
        return self.vision_projection(vision_features)

    def encode_text(self, text: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Encode text to the multimodal embedding space.

        Args:
            text: Dictionary containing 'src' and optionally 'src_mask'

        Returns:
            Text features of shape [B, projection_dim]
        """
        # Pass text through encoder
        with torch.no_grad():
            # Use the encoder part of the text model
            text_encoding = self.text_model.encode(
                text["src"], src_mask=text.get("src_mask", None)
            )

            # Use the final hidden state of the first token (assumed to be [CLS] or similar)
            text_features = text_encoding[:, 0, :]

        # Project to common space
        return self.text_projection(text_features)

    def forward(
        self,
        image: Optional[torch.Tensor] = None,
        text: Optional[Dict[str, torch.Tensor]] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass of the multimodal transformer.

        Args:
            image: Optional image tensor of shape [B, C, H, W]
            text: Optional text dictionary containing 'src' and optionally 'src_mask'

        Returns:
            Dictionary containing:
            - 'image_features': Image features if image is provided
            - 'text_features': Text features if text is provided
            - 'similarity': Cosine similarity between image and text features if both are provided
        """
        results = {}

        if image is not None:
            results["image_features"] = self.encode_image(image)

        if text is not None:
            results["text_features"] = self.encode_text(text)

        # If both modalities are present, compute similarity
        if image is not None and text is not None:
            image_features = results["image_features"]
            text_features = results["text_features"]

            # Normalize features
            image_features = F.normalize(image_features, p=2, dim=1)
            text_features = F.normalize(text_features, p=2, dim=1)

            # Compute cosine similarity
            similarity = torch.matmul(image_features, text_features.transpose(0, 1))
            results["similarity"] = similarity

        return results
