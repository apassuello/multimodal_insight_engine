# src/models/vision/multimodal_integration.py
"""MODULE: multimodal_integration.py
PURPOSE: Implements models for combining vision and text modalities in a unified architecture
KEY COMPONENTS:
- MultiModalTransformer: Combines vision and text transformers with projection layers
DEPENDENCIES: torch, torch.nn, typing, ..base_model, ..transformer, .vision_transformer
SPECIAL NOTES: Current implementation uses a simple integration approach that will be enhanced in future versions
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple
import os

from ..base_model import BaseModel
from ..transformer import EncoderDecoderTransformer
from .vision_transformer import VisionTransformer
from .cross_modal_attention import CoAttentionFusion, BidirectionalCrossAttention


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
        # Removed torch.no_grad() to allow gradient flow
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
        # Removed torch.no_grad() to allow gradient flow
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


class EnhancedMultiModalTransformer(BaseModel):
    """
    Enhanced MultiModal Transformer with advanced cross-modal integration.

    This implementation uses bidirectional cross-attention and co-attention fusion
    to create rich interactions between vision and text modalities.
    """

    def __init__(
        self,
        vision_model: VisionTransformer,
        text_model: EncoderDecoderTransformer,
        fusion_dim: int = 512,
        num_fusion_layers: int = 2,
        num_heads: int = 8,
        dropout: float = 0.1,
        fusion_type: str = "co_attention",  # "co_attention" or "bidirectional"
    ):
        """
        Initialize the enhanced multimodal transformer.

        Args:
            vision_model: Vision Transformer model
            text_model: Text Transformer model
            fusion_dim: Dimension for multimodal fusion
            num_fusion_layers: Number of fusion layers
            num_heads: Number of attention heads in fusion layers
            dropout: Dropout probability
            fusion_type: Type of fusion mechanism to use
        """
        super().__init__()

        self.vision_model = vision_model
        self.text_model = text_model
        self.fusion_type = fusion_type

        # Get embedding dimensions from models
        vision_dim = vision_model.embed_dim
        text_dim = text_model.d_model

        # Create fusion module based on specified type
        if fusion_type == "co_attention":
            self.fusion_module = CoAttentionFusion(
                vision_dim=vision_dim,
                text_dim=text_dim,
                fusion_dim=fusion_dim,
                num_heads=num_heads,
                num_layers=num_fusion_layers,
                dropout=dropout,
            )
        elif fusion_type == "bidirectional":
            self.fusion_module = BidirectionalCrossAttention(
                vision_dim=vision_dim,
                text_dim=text_dim,
                fusion_dim=fusion_dim,
                num_heads=num_heads,
                dropout=dropout,
            )
        else:
            raise ValueError(f"Unsupported fusion type: {fusion_type}")

        # Final projection for classification and similarity tasks
        self.classifier = nn.Sequential(
            nn.LayerNorm(fusion_dim),
            nn.Linear(fusion_dim, fusion_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(fusion_dim, 2),  # Binary classification (match/no-match)
        )

        # Initialize weights
        self.apply(self._init_weights)

    def _init_weights(self, module):
        """Initialize the weights."""
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def extract_vision_features(self, images: torch.Tensor) -> torch.Tensor:
        """
        Extract features from images using the vision model.

        Args:
            images: Input images [batch_size, channels, height, width]

        Returns:
            Vision features [batch_size, num_patches+1, vision_dim]
        """
        # Use the vision transformer to extract patch features
        # Note: VisionTransformer.forward_features() should return sequence of patch features
        # including the class token, not just the pooled representation
        # Removed torch.no_grad() to allow gradient flow through the vision model
        # For a standard ViT, this would extract features before the final pooling

        # Modify this based on your specific implementation to get patch features
        if hasattr(self.vision_model, "extract_patch_features"):
            vision_features = self.vision_model.extract_patch_features(images)
        else:
            # Fallback approach - this would need to be adapted to your implementation
            # You might need to add a method to your VisionTransformer class
            vision_features = self._extract_patch_features(images)

        return vision_features

    def _extract_patch_features(self, images: torch.Tensor) -> torch.Tensor:
        """
        Fallback method to extract patch features from the vision model.

        This is a provisional implementation and should be replaced with a proper
        method in your VisionTransformer class.

        Args:
            images: Input images [batch_size, channels, height, width]

        Returns:
            Vision features [batch_size, num_patches+1, vision_dim]
        """
        # Get batch size
        B = images.shape[0]

        # Access the internal components of the vision model
        # 1. Extract patch embeddings
        x = self.vision_model.patch_embed(images)

        # 2. Add class token if the model uses it
        if self.vision_model.pool == "cls" and hasattr(self.vision_model, "cls_token"):
            cls_tokens = self.vision_model.cls_token.expand(B, -1, -1)
            x = torch.cat((cls_tokens, x), dim=1)

        # 3. Add positional embeddings
        x = x + self.vision_model.pos_embed
        x = self.vision_model.pos_drop(x)

        # 4. Pass through transformer blocks
        for blk in self.vision_model.blocks:
            x = blk(x)

        # 5. Apply final normalization
        x = self.vision_model.norm(x)

        return x

    def extract_text_features(self, text_data: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Extract features from text using the text model.

        Args:
            text_data: Dictionary with text input tensors including 'src' and optionally 'src_mask'

        Returns:
            Text features [batch_size, seq_length, text_dim]
        """
        # Removed torch.no_grad() to allow gradient flow through the text model
        # Use the encoder part of the text model to get sequence representations
        text_features = self.text_model.encode(
            text_data["src"], src_mask=text_data.get("src_mask", None)
        )

        return text_features

    def forward(
        self,
        images: Optional[torch.Tensor] = None,
        text_data: Optional[Dict[str, torch.Tensor]] = None,
        return_attention: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass for the enhanced multimodal transformer.

        Args:
            images: Optional image tensor [batch_size, channels, height, width]
            text_data: Optional text dictionary containing 'src' and optionally 'src_mask'
            return_attention: Whether to return attention maps for visualization

        Returns:
            Dictionary with various outputs depending on the inputs:
            - 'vision_features': Vision features if images provided
            - 'text_features': Text features if text provided
            - 'fusion_features': Fused representation if both provided
            - 'similarity': Text-image similarity scores if both provided
            - 'classification': Match/no-match prediction if both provided
            - 'attention_maps': Attention visualizations if return_attention=True
        """
        results = {}

        # Process vision input if provided
        vision_features = None
        if images is not None:
            vision_features = self.extract_vision_features(images)
            results["vision_features"] = vision_features

            # Create vision mask (all tokens are valid)
            vision_mask = torch.ones(
                vision_features.shape[0],
                vision_features.shape[1],
                device=vision_features.device,
                dtype=torch.bool,
            )
        else:
            vision_mask = None

        # Process text input if provided
        text_features = None
        if text_data is not None:
            text_features = self.extract_text_features(text_data)
            results["text_features"] = text_features

            # Create text mask from padding if available
            if "src_mask" in text_data:
                text_mask = text_data["src_mask"]
            else:
                # If no mask provided, assume all tokens are valid
                text_mask = torch.ones(
                    text_features.shape[0],
                    text_features.shape[1],
                    device=text_features.device,
                    dtype=torch.bool,
                )
        else:
            text_mask = None

        # Apply fusion if both modalities are present
        if vision_features is not None and text_features is not None:
            # Apply cross-modal fusion
            fusion_outputs = self.fusion_module(
                vision_features=vision_features,
                text_features=text_features,
                vision_mask=vision_mask,
                text_mask=text_mask,
            )

            # Store updated features
            results["vision_features_enhanced"] = fusion_outputs["vision_features"]
            results["text_features_enhanced"] = fusion_outputs["text_features"]

            # Get fused representation
            if "fusion_features" in fusion_outputs:
                results["fusion_features"] = fusion_outputs["fusion_features"]

                # Calculate similarity using normalized fused features
                fused_vision = F.normalize(
                    fusion_outputs["fusion_features"], p=2, dim=1
                )
                fused_text = F.normalize(fusion_outputs["fusion_features"], p=2, dim=1)
                results["similarity"] = torch.matmul(
                    fused_vision, fused_text.transpose(0, 1)
                )

                # Classification prediction (match/no-match)
                if hasattr(self, "classifier"):
                    results["classification"] = self.classifier(
                        fusion_outputs["fusion_features"]
                    )

            # Use pooled fusion if available (from co-attention)
            if "pooled_fusion" in fusion_outputs:
                results["pooled_fusion"] = fusion_outputs["pooled_fusion"]

            # Include attention maps if requested
            if return_attention and "attention_maps" in fusion_outputs:
                results["attention_maps"] = fusion_outputs["attention_maps"]

        # Calculate raw similarity if requested (even without fusion)
        if vision_features is not None and text_features is not None:
            # Get global representations for both modalities
            # For MPS compatibility, use simpler pooling approach that avoids device issues

            # For vision features, prefer class token if available, otherwise use mean pooling
            # For both vision and text, either use CLS tokens for both or mean pooling for both
            if (
                hasattr(self.vision_model, "pool")
                and self.vision_model.pool == "cls"
                and hasattr(text_features, "cls_token")
            ):
                vision_global = vision_features[:, 0]  # CLS token
                text_global = text_features[:, 0]  # CLS token
            else:
                # Use mean pooling for both
                vision_global = vision_features.mean(dim=1)
                text_global = text_features.mean(dim=1)

            # Ensure features have compatible dimensions
            # Check if dimensions match before computing similarity
            if vision_global.shape[1] == text_global.shape[1]:
                # Normalize features
                vision_global = F.normalize(vision_global, p=2, dim=1)
                text_global = F.normalize(text_global, p=2, dim=1)

                # Compute raw similarity
                results["raw_similarity"] = torch.matmul(
                    vision_global, text_global.transpose(0, 1)
                )
            else:
                # Handle case where dimensions don't match - skip similarity calculation
                # and log the issue
                print(
                    f"Warning: Feature dimensions don't match for similarity calculation: "
                    f"vision_global shape={vision_global.shape}, text_global shape={text_global.shape}"
                )

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
        "module_purpose": "Implements models for combining vision and text modalities in a unified architecture",
        "key_classes": [
            {
                "name": "MultiModalTransformer",
                "purpose": "Combines vision and text transformer models with projection layers to a common embedding space",
                "key_methods": [
                    {
                        "name": "encode_image",
                        "signature": "encode_image(self, image: torch.Tensor) -> torch.Tensor",
                        "brief_description": "Projects image features to multimodal embedding space",
                    },
                    {
                        "name": "encode_text",
                        "signature": "encode_text(self, text: Dict[str, torch.Tensor]) -> torch.Tensor",
                        "brief_description": "Projects text features to multimodal embedding space",
                    },
                    {
                        "name": "forward",
                        "signature": "forward(self, image: Optional[torch.Tensor] = None, text: Optional[Dict[str, torch.Tensor]] = None) -> Dict[str, torch.Tensor]",
                        "brief_description": "Processes image and/or text inputs and computes similarity if both are provided",
                    },
                ],
                "inheritance": "BaseModel",
                "dependencies": [
                    "torch",
                    "torch.nn",
                    "torch.nn.functional",
                    "..base_model",
                    "..transformer",
                    ".vision_transformer",
                ],
            }
        ],
        "external_dependencies": ["torch"],
        "complexity_score": 7,  # High complexity due to multimodal integration
    }
