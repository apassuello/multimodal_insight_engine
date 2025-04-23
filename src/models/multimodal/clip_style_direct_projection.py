import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Optional, Tuple, Union, List, Any
import logging

logger = logging.getLogger(__name__)

class CLIPStyleDirectProjection(nn.Module):
    """
    CLIP-style model with direct projection between modalities.
    
    This model implements an architecture similar to CLIP (Contrastive Language-Image Pre-training)
    with direct projection from each modality to a shared embedding space, without
    cross-attention or other complex interaction mechanisms.
    
    It focuses on learning strong aligned representations through contrastive learning,
    with careful design choices to prevent representation collapse.
    
    Reference: Radford et al., "Learning Transferable Visual Models From Natural Language Supervision"
    https://arxiv.org/abs/2103.00020
    """
    
    def __init__(
        self,
        vision_model: nn.Module,
        text_model: nn.Module,
        projection_dim: int = 512,
        dropout: float = 0.1,
        use_multi_head: bool = False,
        num_projection_heads: int = 4,
        use_layernorm: bool = True,
        prompt_pooling: str = "mean",
        initial_temperature: float = 0.07,
        feature_dropout: float = 0.0,
    ):
        """
        Initialize the CLIP-style model.
        
        Args:
            vision_model: Vision encoder model
            text_model: Text encoder model
            projection_dim: Dimension of the joint embedding space
            dropout: Dropout rate in projection layers
            use_multi_head: Whether to use multiple projection heads
            num_projection_heads: Number of projection heads if use_multi_head is True
            use_layernorm: Whether to use layer normalization in projection layers
            prompt_pooling: How to pool text representations ("mean", "cls", or "last")
            initial_temperature: Initial value for the learnable temperature parameter
            feature_dropout: Dropout rate applied directly to features (for regularization)
        """
        super().__init__()
        self.vision_model = vision_model
        self.text_model = text_model
        self.prompt_pooling = prompt_pooling
        self.use_multi_head = use_multi_head
        self.feature_dropout = feature_dropout
        
        # Determine model dimensions
        self.vision_dim = self._get_model_dimension(vision_model)
        self.text_dim = self._get_model_dimension(text_model)
        
        logger.info(f"Vision model dimension: {self.vision_dim}")
        logger.info(f"Text model dimension: {self.text_dim}")
        
        # Create vision projection
        if use_multi_head and num_projection_heads > 1:
            # Multi-head projection
            self.vision_proj = MultiHeadProjection(
                input_dim=self.vision_dim,
                output_dim=projection_dim,
                num_heads=num_projection_heads,
                dropout=dropout,
                use_layernorm=use_layernorm
            )
        else:
            # Standard two-layer projection with optional layer norm
            vision_layers = []
            vision_layers.append(nn.Linear(self.vision_dim, self.vision_dim))
            
            if use_layernorm:
                vision_layers.append(nn.LayerNorm(self.vision_dim))
            
            vision_layers.extend([
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(self.vision_dim, projection_dim)
            ])
            
            self.vision_proj = nn.Sequential(*vision_layers)
        
        # Create text projection with identical structure
        if use_multi_head and num_projection_heads > 1:
            # Multi-head projection
            self.text_proj = MultiHeadProjection(
                input_dim=self.text_dim,
                output_dim=projection_dim,
                num_heads=num_projection_heads,
                dropout=dropout,
                use_layernorm=use_layernorm
            )
        else:
            # Standard two-layer projection with optional layer norm
            text_layers = []
            text_layers.append(nn.Linear(self.text_dim, self.text_dim))
            
            if use_layernorm:
                text_layers.append(nn.LayerNorm(self.text_dim))
            
            text_layers.extend([
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(self.text_dim, projection_dim)
            ])
            
            self.text_proj = nn.Sequential(*text_layers)
        
        # Learnable temperature parameter
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / initial_temperature))
        
        # Initialize weights for better training dynamics
        self._init_parameters()
        
        # Track feature statistics for debugging
        self.register_buffer("running_vision_var", torch.ones(1) * 0.5)
        self.register_buffer("running_text_var", torch.ones(1) * 0.5)
        self.register_buffer("iter_count", torch.zeros(1))
    
    def _init_parameters(self):
        """Initialize model parameters for better training dynamics."""
        # Use orthogonal initialization for projection layers
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=1.0)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def _get_model_dimension(self, model: nn.Module) -> int:
        """
        Extract the output dimension from a model.
        
        Args:
            model: Neural network model
            
        Returns:
            The dimension of the model's output embeddings
        """
        # Try various common attribute names for dimensions
        if hasattr(model, "num_features"):
            return model.num_features
        elif hasattr(model, "embed_dim"):
            return model.embed_dim
        elif hasattr(model, "d_model"):
            return model.d_model
        elif hasattr(model, "hidden_size"):
            return model.hidden_size
        elif hasattr(model, "config") and hasattr(model.config, "hidden_size"):
            return model.config.hidden_size
        elif hasattr(model, "config") and hasattr(model.config, "d_model"):
            return model.config.d_model
        
        # If no dimension found, use a common default but log a warning
        logger.warning(f"Could not determine dimension for model {type(model).__name__}, using default of 768")
        return 768
    
    def extract_text_features(
        self, text_data: Union[Dict[str, torch.Tensor], torch.Tensor]
    ) -> torch.Tensor:
        """
        Extract features from the text model.
        
        Args:
            text_data: Input text data (could be a tensor or dict with 'src' and 'src_mask')
            
        Returns:
            Text features tensor
        """
        # Handle different input formats
        if isinstance(text_data, dict):
            # Input is a dict with 'src' and 'src_mask'
            text_outputs = self.text_model(**text_data)
        else:
            # Input is a tensor
            text_outputs = self.text_model(text_data)
        
        # Handle different output formats
        if isinstance(text_outputs, tuple):
            # Some models return multiple tensors
            text_features = text_outputs[0]
        elif isinstance(text_outputs, dict) and "last_hidden_state" in text_outputs:
            # HuggingFace style outputs
            text_features = text_outputs["last_hidden_state"]
        elif isinstance(text_outputs, torch.Tensor):
            # Direct tensor output
            text_features = text_outputs
        else:
            raise ValueError(f"Unexpected output format from text model: {type(text_outputs)}")
        
        # Pool sequence dimension if needed
        if text_features.dim() == 3:  # [batch_size, seq_len, hidden_dim]
            if self.prompt_pooling == "cls":
                # Use CLS token (first token)
                text_features = text_features[:, 0]
            elif self.prompt_pooling == "last":
                # Use last token
                if "src_mask" in text_data:
                    # Get the last token for each sequence based on mask
                    mask = text_data["src_mask"]
                    seq_lengths = mask.sum(dim=1) - 1  # -1 because index starts at 0
                    batch_indices = torch.arange(mask.size(0), device=mask.device)
                    text_features = text_features[batch_indices, seq_lengths]
                else:
                    # Just use the last token if no mask provided
                    text_features = text_features[:, -1]
            else:
                # Default to mean pooling (across non-padded tokens if mask provided)
                if "src_mask" in text_data:
                    # Apply mask for proper mean calculation
                    mask = text_data["src_mask"].unsqueeze(-1)
                    text_features = (text_features * mask).sum(dim=1) / mask.sum(dim=1)
                else:
                    # Simple mean pooling
                    text_features = text_features.mean(dim=1)
        
        return text_features
    
    def extract_vision_features(self, images: torch.Tensor) -> torch.Tensor:
        """
        Extract features from the vision model.
        
        Args:
            images: Input image tensor [batch_size, channels, height, width]
            
        Returns:
            Vision features tensor
        """
        # Handle different vision model types
        if hasattr(self.vision_model, "forward_features"):
            # timm models
            vision_features = self.vision_model.forward_features(images)
        elif hasattr(self.vision_model, "extract_features"):
            # Some custom implementations
            vision_features = self.vision_model.extract_features(images)
        else:
            # Standard forward
            vision_features = self.vision_model(images)
        
        # Handle different output formats
        if isinstance(vision_features, tuple):
            vision_features = vision_features[0]
        elif isinstance(vision_features, dict) and "pooler_output" in vision_features:
            vision_features = vision_features["pooler_output"]
        
        # Pool if necessary (for feature maps)
        if vision_features.dim() > 2:
            # If features are [batch, spatial_dim, spatial_dim, channels]
            if vision_features.dim() == 4:
                # Handle channel-last format
                if vision_features.shape[-1] < vision_features.shape[1]:
                    # Channel is last dimension - rearrange to [batch, channels, spatial, spatial]
                    vision_features = vision_features.permute(0, 3, 1, 2)
            
            # Global average pooling for features with spatial dimensions
            if vision_features.dim() > 2:
                vision_features = F.adaptive_avg_pool2d(vision_features, (1, 1)).squeeze(-1).squeeze(-1)
        
        return vision_features
    
    def forward(
        self, 
        images: Optional[torch.Tensor] = None, 
        text_data: Optional[Union[Dict[str, torch.Tensor], torch.Tensor]] = None
    ) -> Dict[str, Any]:
        """
        Forward pass for the CLIP-style model.
        
        Args:
            images: Optional image inputs [batch_size, channels, height, width]
            text_data: Optional text inputs (tensor or dict with 'src' and 'src_mask')
            
        Returns:
            Dictionary with vision_features, text_features, and similarity if both are provided
        """
        outputs = {}
        
        # Process vision if provided
        if images is not None:
            # Extract features
            vision_features = self.extract_vision_features(images)
            
            # Apply feature dropout if enabled (during training only)
            if self.training and self.feature_dropout > 0:
                vision_features = F.dropout(vision_features, p=self.feature_dropout)
            
            # Apply projection
            vision_proj = self.vision_proj(vision_features)
            
            # Track variance statistics during training
            if self.training:
                with torch.no_grad():
                    vision_var = torch.var(vision_proj, dim=1).mean()
                    momentum = 0.9  # Exponential moving average
                    self.running_vision_var = momentum * self.running_vision_var + (1 - momentum) * vision_var
                    
                    # Log occasionally
                    if self.iter_count % 100 == 0:
                        logger.debug(f"Vision feature variance: {vision_var.item():.4f}, running: {self.running_vision_var.item():.4f}")
            
            outputs["vision_features"] = vision_proj
        
        # Process text if provided
        if text_data is not None:
            # Extract features
            text_features = self.extract_text_features(text_data)
            
            # Apply feature dropout if enabled (during training only)
            if self.training and self.feature_dropout > 0:
                text_features = F.dropout(text_features, p=self.feature_dropout)
            
            # Apply projection
            text_proj = self.text_proj(text_features)
            
            # Track variance statistics during training
            if self.training:
                with torch.no_grad():
                    text_var = torch.var(text_proj, dim=1).mean()
                    momentum = 0.9  # Exponential moving average
                    self.running_text_var = momentum * self.running_text_var + (1 - momentum) * text_var
                    
                    # Log occasionally
                    if self.iter_count % 100 == 0:
                        logger.debug(f"Text feature variance: {text_var.item():.4f}, running: {self.running_text_var.item():.4f}")
            
            outputs["text_features"] = text_proj
        
        # Compute similarity if both modalities present
        if images is not None and text_data is not None:
            # Normalize features for cosine similarity
            vision_norm = F.normalize(outputs["vision_features"], p=2, dim=1)
            text_norm = F.normalize(outputs["text_features"], p=2, dim=1)
            
            # Scale logits by temperature
            logit_scale = torch.clamp(self.logit_scale.exp(), min=1.0, max=100.0)
            
            # Compute scaled cosine similarity [batch_size_img, batch_size_text]
            similarity = logit_scale * torch.matmul(vision_norm, text_norm.T)
            
            outputs["similarity"] = similarity
            outputs["logit_scale"] = logit_scale.item()
            
            # Increment iteration counter (for logging)
            if self.training:
                self.iter_count += 1
        
        return outputs


class MultiHeadProjection(nn.Module):
    """
    Multi-head projection module that projects a feature vector to multiple heads
    and then combines them for more expressiveness.
    """
    
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        num_heads: int = 4,
        dropout: float = 0.1,
        use_layernorm: bool = True
    ):
        """
        Initialize multi-head projection.
        
        Args:
            input_dim: Input feature dimension
            output_dim: Output feature dimension
            num_heads: Number of projection heads
            dropout: Dropout rate
            use_layernorm: Whether to use layer normalization
        """
        super().__init__()
        self.num_heads = num_heads
        
        # Head dimension must divide output_dim evenly
        assert output_dim % num_heads == 0, f"Output dimension ({output_dim}) must be divisible by num_heads ({num_heads})"
        self.head_dim = output_dim // num_heads
        
        # Create a projection for each head
        self.heads = nn.ModuleList()
        for _ in range(num_heads):
            head_layers = []
            head_layers.append(nn.Linear(input_dim, input_dim))
            
            if use_layernorm:
                head_layers.append(nn.LayerNorm(input_dim))
            
            head_layers.extend([
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(input_dim, self.head_dim)
            ])
            
            self.heads.append(nn.Sequential(*head_layers))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Project input features through multiple heads and concatenate.
        
        Args:
            x: Input features [batch_size, input_dim]
            
        Returns:
            Projected features [batch_size, output_dim]
        """
        # Process each head
        head_outputs = [head(x) for head in self.heads]
        
        # Concatenate along feature dimension
        return torch.cat(head_outputs, dim=1)


def extract_file_metadata(file_path=__file__):
    """
    Extract structured metadata about this module.
    
    Args:
        file_path: Path to the source file (defaults to current file)
        
    Returns:
        dict: Structured metadata about the module's purpose and components
    """
    import os
    
    return {
        "filename": os.path.basename(file_path),
        "module_purpose": "Implements a CLIP-style model with direct projection between modalities",
        "key_classes": [
            {
                "name": "CLIPStyleDirectProjection",
                "purpose": "CLIP-style model that directly projects vision and text to a shared space",
                "key_methods": [
                    {
                        "name": "__init__",
                        "signature": "__init__(self, vision_model: nn.Module, text_model: nn.Module, projection_dim: int = 512, dropout: float = 0.1, use_multi_head: bool = False, num_projection_heads: int = 4, use_layernorm: bool = True, prompt_pooling: str = 'mean', initial_temperature: float = 0.07, feature_dropout: float = 0.0)",
                        "brief_description": "Initialize CLIP-style model with projection options"
                    },
                    {
                        "name": "extract_text_features",
                        "signature": "extract_text_features(self, text_data: Union[Dict[str, torch.Tensor], torch.Tensor]) -> torch.Tensor",
                        "brief_description": "Extract features from text model with pooling"
                    },
                    {
                        "name": "extract_vision_features",
                        "signature": "extract_vision_features(self, images: torch.Tensor) -> torch.Tensor",
                        "brief_description": "Extract features from vision model with pooling"
                    },
                    {
                        "name": "forward",
                        "signature": "forward(self, images: Optional[torch.Tensor] = None, text_data: Optional[Union[Dict[str, torch.Tensor], torch.Tensor]] = None) -> Dict[str, Any]",
                        "brief_description": "Process images and text, compute similarity"
                    }
                ],
                "inheritance": "nn.Module",
                "dependencies": ["torch", "torch.nn", "numpy"]
            },
            {
                "name": "MultiHeadProjection",
                "purpose": "Multi-head projection module for more expressive projections",
                "key_methods": [
                    {
                        "name": "forward",
                        "signature": "forward(self, x: torch.Tensor) -> torch.Tensor",
                        "brief_description": "Project features through multiple heads and concatenate"
                    }
                ],
                "inheritance": "nn.Module",
                "dependencies": ["torch", "torch.nn"]
            }
        ],
        "external_dependencies": ["torch", "numpy", "logging"],
        "complexity_score": 7  # Higher complexity due to many options and handling different model types
    }