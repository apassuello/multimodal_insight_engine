"""MODULE: dual_encoder.py
PURPOSE: Implements a dual encoder architecture for vision-text multimodal learning with specialized projection layers.

KEY COMPONENTS:
- DualEncoder: Core class that handles dual encoding of vision and text inputs with projection layers
- Feature extraction and normalization for both modalities
- Learned temperature scaling for similarity computation
- Variance preservation techniques during training

DEPENDENCIES:
- PyTorch (torch, torch.nn)
- NumPy (for initialization)

SPECIAL NOTES:
- Includes dynamic scaling based on feature variance for improved training stability
- Provides asymmetric projection paths for vision and text modalities
- Monitors feature diversity metrics during training
"""

import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class DualEncoder(nn.Module):
    def __init__(self, vision_model, text_model, projection_dim=512):
        super().__init__()
        self.vision_model = vision_model
        self.text_model = text_model

        # Get dimensions from models
        vision_dim = self._get_model_dimension(vision_model)
        text_dim = self._get_model_dimension(text_model)

        # Asymmetric vision projection with batch normalization
        self.vision_proj = nn.Sequential(
            nn.Linear(vision_dim, vision_dim),
            nn.LayerNorm(vision_dim),
            nn.GELU(),
            nn.Linear(vision_dim, projection_dim),
            nn.BatchNorm1d(projection_dim, affine=True),
        )

        # Text projection with different structure to ensure asymmetry
        self.text_proj = nn.Sequential(
            nn.Linear(text_dim, text_dim),
            nn.LayerNorm(text_dim),
            nn.GELU(),
            nn.Linear(text_dim, projection_dim),
            nn.BatchNorm1d(projection_dim, affine=True),
        )

        # Learnable temperature parameter
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

        # Initialize with orthogonal weights for diversity
        self._init_parameters()

    def _init_parameters(self):
        # Orthogonal initialization for better feature diversity
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=1.2)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def _get_model_dimension(self, model):
        # Extract dimension from various model types
        if hasattr(model, "num_features"):
            return model.num_features
        elif hasattr(model, "embed_dim"):
            return model.embed_dim
        elif hasattr(model, "d_model"):
            return model.d_model
        elif hasattr(model, "config") and hasattr(model.config, "hidden_size"):
            return model.config.hidden_size
        else:
            return 768  # Default fallback

    def forward(self, images=None, text_data=None):
        outputs = {}

        # Process images if provided
        if images is not None:
            vision_features = self._extract_features(self.vision_model, images)
            # Apply projection with variance-preserving scaling during training
            vision_proj = self.vision_proj(vision_features)
            if self.training:
                # Scale features to increase variance explicitly
                vision_variance = torch.var(vision_proj, dim=0).mean()
                scaling_factor = torch.clamp(
                    1.0 / torch.sqrt(vision_variance), min=1.0, max=10.0
                )
                vision_proj = vision_proj * scaling_factor

            outputs["vision_features"] = vision_proj

        # Process text if provided
        if text_data is not None:
            text_features = self._extract_features(self.text_model, text_data)
            # Apply projection with different scaling for asymmetry
            text_proj = self.text_proj(text_features)
            if self.training:
                # Different scaling for text to create asymmetry
                text_variance = torch.var(text_proj, dim=0).mean()
                scaling_factor = torch.clamp(
                    1.0 / torch.sqrt(text_variance), min=1.0, max=8.0
                )
                text_proj = text_proj * scaling_factor

            outputs["text_features"] = text_proj

        # Compute similarity if both modalities present
        if images is not None and text_data is not None:
            # Normalize for similarity computation
            vision_norm = F.normalize(outputs["vision_features"], p=2, dim=1)
            text_norm = F.normalize(outputs["text_features"], p=2, dim=1)

            # Compute similarity with learned temperature
            logit_scale = torch.clamp(self.logit_scale.exp(), min=1.0, max=100.0)
            similarity = logit_scale * torch.matmul(vision_norm, text_norm.T)

            outputs["similarity"] = similarity

            # Calculate diversity metrics for monitoring
            if self.training:
                with torch.no_grad():
                    # Track feature statistics
                    vision_var = torch.var(vision_norm, dim=0).mean().item()
                    text_var = torch.var(text_norm, dim=0).mean().item()

                    # Count low-variance dimensions (<0.01)
                    v_low_var_dims = (torch.var(vision_norm, dim=0) < 0.01).sum().item()
                    t_low_var_dims = (torch.var(text_norm, dim=0) < 0.01).sum().item()

                    outputs["vision_var"] = vision_var
                    outputs["text_var"] = text_var
                    outputs["low_var_dims"] = v_low_var_dims + t_low_var_dims

        return outputs

    def _extract_features(self, model, inputs):
        """Extract features from various model types with unified interface"""
        if hasattr(model, "forward_features"):
            features = model.forward_features(inputs)
        elif hasattr(model, "encode"):
            features = model.encode(**inputs if isinstance(inputs, dict) else inputs)
        elif hasattr(model, "extract_features"):
            features = model.extract_features(inputs)
        else:
            # Generic forward pass
            features = model(inputs)

        # Handle various output formats
        if isinstance(features, dict) and "pooler_output" in features:
            return features["pooler_output"]
        elif isinstance(features, tuple):
            return features[0]
        elif isinstance(features, torch.Tensor) and features.dim() == 3:
            # Sequence outputs, use CLS token or mean pooling
            if hasattr(model, "pooling") and model.pooling == "mean":
                return torch.mean(features, dim=1)
            else:
                return features[:, 0]  # CLS token

        return features


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
        "module_purpose": "Implements a dual encoder architecture for vision-text multimodal learning with specialized projection layers",
        "key_classes": [
            {
                "name": "DualEncoder",
                "purpose": "Core class that processes both vision and text inputs through modality-specific encoders and projections",
                "key_methods": [
                    {
                        "name": "__init__",
                        "signature": "__init__(self, vision_model, text_model, projection_dim=512)",
                        "brief_description": "Initialize with vision and text encoders and configurable projection dimension",
                    },
                    {
                        "name": "forward",
                        "signature": "forward(self, images=None, text_data=None)",
                        "brief_description": "Process vision and text inputs, computing aligned features and similarity scores",
                    },
                    {
                        "name": "_extract_features",
                        "signature": "_extract_features(self, model, inputs)",
                        "brief_description": "Extract features from various model types with unified interface",
                    },
                    {
                        "name": "_get_model_dimension",
                        "signature": "_get_model_dimension(self, model)",
                        "brief_description": "Determine output dimension from different model architectures",
                    },
                ],
                "inheritance": "nn.Module",
                "dependencies": ["torch", "torch.nn", "numpy"],
            }
        ],
        "external_dependencies": ["torch", "torch.nn", "numpy"],
        "complexity_score": 7,  # High complexity due to dynamic scaling and diverse model support
    }
