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

        # Important: Check the shape of positional embeddings to determine if we need a class token
        expected_seq_len = self.vision_model.pos_embed.shape[1]

        # 2. Add class token if needed (ensuring compatibility with pos_embed shape)
        if hasattr(self.vision_model, "cls_token"):
            cls_tokens = self.vision_model.cls_token.expand(B, -1, -1)
            x = torch.cat((cls_tokens, x), dim=1)
        elif expected_seq_len > x.shape[1]:
            # If we don't have cls_token but pos_embed expects one more token
            # Create a learnable cls token on the fly
            cls_dim = x.shape[-1]
            cls_tokens = torch.zeros(B, 1, cls_dim, device=x.device)
            x = torch.cat((cls_tokens, x), dim=1)

        # Check that shapes match before adding positional embeddings
        if x.shape[1] != expected_seq_len:
            raise ValueError(
                f"Sequence length mismatch: features shape={x.shape}, "
                f"positional embeddings expect length {expected_seq_len}"
            )

        # 3. Add positional embeddings
        x = x + self.vision_model.pos_embed

        # Apply dropout if available
        if hasattr(self.vision_model, "pos_drop"):
            x = self.vision_model.pos_drop(x)

        # 4. Pass through transformer blocks
        for blk in self.vision_model.blocks:
            x = blk(x)

        # 5. Apply final normalization
        if hasattr(self.vision_model, "norm"):
            x = self.vision_model.norm(x)

        return x

    def extract_text_features(self, text_data: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Extract features from text using the text model.
        Using a hybrid approach for HuggingFace models on MPS: compute on CPU, return on MPS.

        Args:
            text_data: Dictionary with text input tensors including 'src' and optionally 'src_mask'

        Returns:
            Text features [batch_size, seq_length, text_dim]
        """
        # Get original device for returning output
        original_device = text_data["src"].device
        
        # Check if this is a HuggingFace model
        is_huggingface = hasattr(self.text_model, 'encoder') and 'bert' in str(type(self.text_model)).lower()
        is_mps = original_device.type == "mps"
        
        # First attempt: Try processing directly on the original device
        # This works for non-HuggingFace models and some HuggingFace models not on MPS
        try:
            if hasattr(self.text_model, 'encode'):
                text_features = self.text_model.encode(
                    text_data["src"], src_mask=text_data.get("src_mask", None)
                )
                return text_features  # Already on correct device
            else:
                # Try forward method as fallback
                return self.text_model(
                    text_data["src"], src_mask=text_data.get("src_mask", None)
                )
        except Exception as e:
            # If we're on MPS with a HuggingFace model, provide a more informative message
            if is_huggingface and is_mps:
                print(f"Using hybrid CPU-MPS approach for HuggingFace model: {str(e)}")
            else:
                print(f"Error in direct text processing: {str(e)}")
                print("Attempting CPU processing...")
        
        # Hybrid approach for HuggingFace models on MPS:
        # 1. Process on CPU
        # 2. Return results on original device (MPS)
        # This allows computation to happen where it's most compatible while 
        # keeping tensor operations on the faster device
        
        # Save original model device for later restoration
        original_model_device = next(self.text_model.parameters()).device
        
        # Move inputs to CPU
        cpu_src = text_data["src"].to('cpu')
        cpu_mask = text_data.get("src_mask", None)
        if cpu_mask is not None:
            cpu_mask = cpu_mask.to('cpu')
        
        # Create CPU copy of model for processing
        try:
            # Don't move the actual model, create a copy on CPU
            with torch.no_grad():
                # Process on CPU
                if hasattr(self.text_model, 'encoder') and is_huggingface:
                    # For HuggingFace models, try using just the encoder component on CPU
                    # This is more efficient and avoids some compatibility issues
                    encoder_cpu = self.text_model.encoder.to('cpu')
                    
                    # Handle out-of-range token indices
                    if hasattr(encoder_cpu, "embeddings") and hasattr(encoder_cpu.embeddings, "word_embeddings"):
                        vocab_size = encoder_cpu.embeddings.word_embeddings.weight.size(0)
                        if torch.max(cpu_src) >= vocab_size:
                            print(f"Clipping token indices that exceed vocabulary size ({vocab_size})")
                            cpu_src = torch.clamp(cpu_src, max=vocab_size - 1)
                    
                    outputs = encoder_cpu(input_ids=cpu_src, attention_mask=cpu_mask)
                    features = outputs.last_hidden_state
                    # Move encoder back if needed
                    if original_model_device != torch.device('cpu'):
                        self.text_model.encoder = self.text_model.encoder.to(original_model_device)
                elif hasattr(self.text_model, 'encode'):
                    # Use full model on CPU with encode method
                    text_model_cpu = self.text_model.to('cpu')
                    
                    # Check for vocab size issues if possible
                    if hasattr(text_model_cpu, "embeddings") and hasattr(text_model_cpu.embeddings, "word_embeddings"):
                        vocab_size = text_model_cpu.embeddings.word_embeddings.weight.size(0)
                        if torch.max(cpu_src) >= vocab_size:
                            print(f"Clipping token indices that exceed vocabulary size ({vocab_size})")
                            cpu_src = torch.clamp(cpu_src, max=vocab_size - 1)
                            
                    features = text_model_cpu.encode(cpu_src, src_mask=cpu_mask)
                    # Move model back
                    if original_model_device != torch.device('cpu'):
                        self.text_model = self.text_model.to(original_model_device)
                else:
                    # Use full model on CPU with forward method
                    text_model_cpu = self.text_model.to('cpu')
                    
                    # Check for vocab size issues if possible
                    if hasattr(text_model_cpu, "embeddings") and hasattr(text_model_cpu.embeddings, "word_embeddings"):
                        vocab_size = text_model_cpu.embeddings.word_embeddings.weight.size(0)
                        if torch.max(cpu_src) >= vocab_size:
                            print(f"Clipping token indices that exceed vocabulary size ({vocab_size})")
                            cpu_src = torch.clamp(cpu_src, max=vocab_size - 1)
                            
                    features = text_model_cpu(cpu_src, src_mask=cpu_mask)
                    # Move model back
                    if original_model_device != torch.device('cpu'):
                        self.text_model = self.text_model.to(original_model_device)
                
                # Move features back to original device
                return features.to(original_device)
                
        except Exception as cpu_err:
            print(f"CPU processing failed with error: {str(cpu_err)}")
            # Ensure model is moved back if needed
            if next(self.text_model.parameters()).device != original_model_device:
                self.text_model = self.text_model.to(original_model_device)
            
            # Since training can't proceed without text features, use zeros as absolute last resort
            print("WARNING: Returning zeros as absolute last resort. Training will be severely affected.")
            print("Consider using the --device cpu option if this problem persists.")
            batch_size, seq_length = text_data["src"].shape
            d_model = getattr(self.text_model, 'd_model', 768)
            return torch.zeros(batch_size, seq_length, d_model, device=original_device)

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

        # First determine which device we should use consistently
        # This helps prevent mixed device operations
        target_device = None
        if images is not None:
            target_device = images.device
        elif text_data is not None and "src" in text_data:
            target_device = text_data["src"].device
        else:
            # Fallback to model's device
            target_device = next(self.parameters()).device
            
        # Log the device we're using - just once per run for clarity
        if not hasattr(self, '_logged_device') or self._logged_device != target_device:
            # Format the device name to be more user-friendly (strip ":0" from device name)
            device_name = str(target_device).split(':')[0]
            print(f"MultiModal model using device: {device_name}")
            self._logged_device = target_device

        # Process vision input if provided
        vision_features = None
        if images is not None:
            # Ensure images are on the target device
            if images.device != target_device:
                images = images.to(target_device)
                
            # Extract vision features
            vision_features = self.extract_vision_features(images)
            
            # Ensure vision features are on the target device
            if vision_features.device != target_device:
                vision_features = vision_features.to(target_device)
                
            results["vision_features"] = vision_features

            # Create vision mask (all tokens are valid) directly on target device
            vision_mask = torch.ones(
                vision_features.shape[0],
                vision_features.shape[1],
                device=target_device,
                dtype=torch.bool,
            )
        else:
            vision_mask = None

        # Process text input if provided
        text_features = None
        if text_data is not None:
            # Ensure text data is on the target device
            if "src" in text_data and text_data["src"].device != target_device:
                text_data["src"] = text_data["src"].to(target_device)
            if "src_mask" in text_data and text_data["src_mask"] is not None and text_data["src_mask"].device != target_device:
                text_data["src_mask"] = text_data["src_mask"].to(target_device)
                
            # Extract text features
            text_features = self.extract_text_features(text_data)
            
            # Ensure text features are on the target device
            if text_features.device != target_device:
                text_features = text_features.to(target_device)
                
            results["text_features"] = text_features

            # Create text mask from padding if available
            if "src_mask" in text_data and text_data["src_mask"] is not None:
                text_mask = text_data["src_mask"]
                # Ensure mask is on the target device
                if text_mask.device != target_device:
                    text_mask = text_mask.to(target_device)
            else:
                # If no mask provided, assume all tokens are valid
                text_mask = torch.ones(
                    text_features.shape[0],
                    text_features.shape[1],
                    device=target_device,
                    dtype=torch.bool,
                )
        else:
            text_mask = None

        # Apply fusion if both modalities are present
        if vision_features is not None and text_features is not None:
            # Double check device consistency before fusion
            if vision_features.device != target_device:
                vision_features = vision_features.to(target_device)
            if text_features.device != target_device:
                text_features = text_features.to(target_device)
            if vision_mask is not None and vision_mask.device != target_device:
                vision_mask = vision_mask.to(target_device)
            if text_mask is not None and text_mask.device != target_device:
                text_mask = text_mask.to(target_device)
                
            # Apply cross-modal fusion
            fusion_outputs = self.fusion_module(
                vision_features=vision_features,
                text_features=text_features,
                vision_mask=vision_mask,
                text_mask=text_mask,
            )

            # Ensure fusion outputs are on the target device
            for key, value in fusion_outputs.items():
                if isinstance(value, torch.Tensor) and value.device != target_device:
                    fusion_outputs[key] = value.to(target_device)

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
                    # Ensure classifier is on the target device
                    if next(self.classifier.parameters()).device != target_device:
                        self.classifier = self.classifier.to(target_device)
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

            # Ensure consistent device for global features
            if vision_global.device != target_device:
                vision_global = vision_global.to(target_device)
            if text_global.device != target_device:
                text_global = text_global.to(target_device)

            # Handle dimension mismatch with on-the-fly projection
            vision_dim = vision_global.shape[1]
            text_dim = text_global.shape[1]
            
            if vision_dim != text_dim:
                # Create a simple projection layer to match dimensions on the target device
                if vision_dim > text_dim:
                    # Project vision features to text dimension
                    projection = nn.Linear(vision_dim, text_dim).to(target_device)
                    vision_global = projection(vision_global)
                else:
                    # Project text features to vision dimension
                    projection = nn.Linear(text_dim, vision_dim).to(target_device)
                    text_global = projection(text_global)
                
                print(f"Created projection layer to match vision dim={vision_dim} with text dim={text_dim} on {target_device}")
            
            # Normalize features (after projection if needed)
            vision_global = F.normalize(vision_global, p=2, dim=1)
            text_global = F.normalize(text_global, p=2, dim=1)

            # Compute raw similarity
            results["raw_similarity"] = torch.matmul(
                vision_global, text_global.transpose(0, 1)
            )

        # Final device consistency check
        for key, value in results.items():
            if isinstance(value, torch.Tensor) and value.device != target_device:
                results[key] = value.to(target_device)

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
