import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Optional, Union
import logging

logger = logging.getLogger(__name__)

class BarlowTwinsLoss(nn.Module):
    """
    Implements Barlow Twins loss for multimodal learning.
    
    Barlow Twins loss creates embeddings that are invariant to distortions
    by minimizing redundancy between the components of the embedding vectors
    through a cross-correlation matrix that is pushed to the identity.
    
    This implementation supports multimodal applications where the two views 
    come from different modalities (vision and text) rather than augmentations
    of the same modality as in the original paper.
    
    Reference: J. Zbontar et al., "Barlow Twins: Self-Supervised Learning via Redundancy Reduction"
    https://arxiv.org/abs/2103.03230
    """
    
    def __init__(
        self,
        lambda_coeff: float = 0.005,
        batch_norm_last_layer: bool = True,
        correlation_mode: str = "cross_modal", 
        add_projection: bool = False,
        projection_dim: int = 8192,
        input_dim: Optional[int] = None,
        normalize_embeddings: bool = True,
    ):
        """
        Initialize the Barlow Twins loss module.
        
        Args:
            lambda_coeff: Coefficient for the off-diagonal terms in the loss
            batch_norm_last_layer: Whether to use batch normalization in the final projection layer
            correlation_mode: How to compute correlations ('cross_modal' or 'within_batch')
            add_projection: Whether to add MLP projection heads for embeddings
            projection_dim: Dimension of projection space (if add_projection is True)
            input_dim: Input dimension for projection heads (required if add_projection is True)
            normalize_embeddings: Whether to L2-normalize embeddings before computing loss
        """
        super().__init__()
        self.lambda_coeff = lambda_coeff
        self.normalize_embeddings = normalize_embeddings
        self.correlation_mode = correlation_mode
        self.batch_norm_last_layer = batch_norm_last_layer
        
        # Create projection heads if specified
        self.add_projection = add_projection
        if add_projection:
            assert input_dim is not None, "input_dim must be specified when add_projection=True"
            
            # Vision projection head - follows the paper's recommendation for a 3-layer MLP
            # with batch normalization at each layer and no ReLU after the final layer
            vision_layers = [
                nn.Linear(input_dim, input_dim),
                nn.BatchNorm1d(input_dim),
                nn.ReLU(inplace=True),
                nn.Linear(input_dim, input_dim),
                nn.BatchNorm1d(input_dim),
                nn.ReLU(inplace=True),
                nn.Linear(input_dim, projection_dim),
            ]
            
            # Add batch norm to final layer if specified
            if batch_norm_last_layer:
                vision_layers.append(nn.BatchNorm1d(projection_dim, affine=False))
                
            self.vision_projection = nn.Sequential(*vision_layers)
            
            # Text projection head - same architecture as vision
            text_layers = [
                nn.Linear(input_dim, input_dim),
                nn.BatchNorm1d(input_dim),
                nn.ReLU(inplace=True),
                nn.Linear(input_dim, input_dim),
                nn.BatchNorm1d(input_dim),
                nn.ReLU(inplace=True),
                nn.Linear(input_dim, projection_dim),
            ]
            
            # Add batch norm to final layer if specified
            if batch_norm_last_layer:
                text_layers.append(nn.BatchNorm1d(projection_dim, affine=False))
                
            self.text_projection = nn.Sequential(*text_layers)
            
            # Ensure projection heads are on the same device as the rest of the model
            device = next(self.parameters()).device if list(self.parameters()) else None
            if device:
                self.vision_projection = self.vision_projection.to(device)
                self.text_projection = self.text_projection.to(device)
    
    def project(
        self, vision_features: torch.Tensor, text_features: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply projection heads to features if enabled.
        
        Args:
            vision_features: Vision features [batch_size, vision_dim]
            text_features: Text features [batch_size, text_dim]
            
        Returns:
            Tuple of (projected_vision_features, projected_text_features)
        """
        if self.add_projection:
            # Make sure projection heads are on the same device as the input tensors
            device = vision_features.device
            if list(self.vision_projection.parameters()) and next(self.vision_projection.parameters()).device != device:
                self.vision_projection = self.vision_projection.to(device)
                self.text_projection = self.text_projection.to(device)
            
            # Apply projections
            vision_features = self.vision_projection(vision_features)
            text_features = self.text_projection(text_features)
        
        # Normalize features if specified
        if self.normalize_embeddings:
            vision_features = F.normalize(vision_features, p=2, dim=1)
            text_features = F.normalize(text_features, p=2, dim=1)
        
        return vision_features, text_features
    
    def forward(
        self,
        vision_features: torch.Tensor,
        text_features: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute Barlow Twins loss between vision and text features.
        
        Args:
            vision_features: Vision features [batch_size, vision_dim]
            text_features: Text features [batch_size, text_dim]
            
        Returns:
            Tuple containing:
            - Loss tensor
            - Dictionary with loss components (diagonal_loss, off_diagonal_loss)
        """
        # Get batch size and validate inputs
        batch_size = vision_features.shape[0]
        
        # Safety check for empty batch
        if batch_size == 0:
            return torch.tensor(0.0, device=vision_features.device), {"diagonal_loss": 0.0, "off_diagonal_loss": 0.0}
        
        # Apply projection (and normalization if enabled)
        vision_features, text_features = self.project(vision_features, text_features)
        
        # Center the features along the batch dimension
        vision_features = vision_features - vision_features.mean(dim=0, keepdim=True)
        text_features = text_features - text_features.mean(dim=0, keepdim=True)
        
        # Computing the cross-correlation matrix
        # [d, d] where d is the feature dimension
        if self.correlation_mode == "cross_modal":
            # Cross-modal correlation: vision-text correlation matrix
            # Scale by batch_size to normalize properly
            cross_correlation = torch.matmul(vision_features.T, text_features) / batch_size
        else:
            # Within-batch correlation: average of vision-vision and text-text correlations
            # This mode computes correlation within each modality and averages them
            vision_correlation = torch.matmul(vision_features.T, vision_features) / batch_size
            text_correlation = torch.matmul(text_features.T, text_features) / batch_size
            cross_correlation = (vision_correlation + text_correlation) / 2.0
        
        # Loss computation
        # Diagonal terms: push cross-correlation of corresponding features to 1
        diagonal_loss = torch.mean((torch.diagonal(cross_correlation) - 1)**2)
        
        # Off-diagonal terms: push cross-correlation of different features to 0
        off_diagonal_mask = 1 - torch.eye(cross_correlation.shape[0], device=cross_correlation.device)
        off_diagonal_loss = torch.mean((cross_correlation * off_diagonal_mask)**2)
        
        # Combine the losses with lambda coefficient for off-diagonal terms
        loss = diagonal_loss + self.lambda_coeff * off_diagonal_loss
        
        components = {
            "diagonal_loss": diagonal_loss.item(),
            "off_diagonal_loss": off_diagonal_loss.item(),
        }
        
        return loss, components
    
    def train(self, mode: bool = True):
        """
        Set the module in training mode.
        
        Args:
            mode: Whether to set training mode (True) or evaluation mode (False)
            
        Returns:
            self
        """
        if self.add_projection:
            self.vision_projection.train(mode)
            self.text_projection.train(mode)
        return super().train(mode)
    
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
        "module_purpose": "Implements Barlow Twins loss for multimodal learning through redundancy reduction",
        "key_classes": [
            {
                "name": "BarlowTwinsLoss",
                "purpose": "Loss function that creates embeddings with minimal redundancy between their components",
                "key_methods": [
                    {
                        "name": "__init__",
                        "signature": "__init__(self, lambda_coeff: float = 0.005, batch_norm_last_layer: bool = True, correlation_mode: str = 'cross_modal', add_projection: bool = False, projection_dim: int = 8192, input_dim: Optional[int] = None, normalize_embeddings: bool = True)",
                        "brief_description": "Initialize the Barlow Twins loss module with configurable parameters"
                    },
                    {
                        "name": "project",
                        "signature": "project(self, vision_features: torch.Tensor, text_features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]",
                        "brief_description": "Apply projection heads to features if enabled"
                    },
                    {
                        "name": "forward",
                        "signature": "forward(self, vision_features: torch.Tensor, text_features: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, float]]",
                        "brief_description": "Compute Barlow Twins loss between vision and text features"
                    }
                ],
                "inheritance": "nn.Module",
                "dependencies": ["torch", "torch.nn", "typing"]
            }
        ],
        "external_dependencies": ["torch"],
        "complexity_score": 6  # Moderate complexity
    }