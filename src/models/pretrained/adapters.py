# src/models/pretrained/adapters.py
import torch
import torch.nn as nn

class ModelAdapter(nn.Module):
    """
    Adapter that adds a small trainable component to a frozen pretrained model.
    """
    
    def __init__(self, base_model: nn.Module, adapter_dim: int = 64):
        super().__init__()
        self.base_model = base_model
        
        # Freeze the base model
        for param in self.base_model.parameters():
            param.requires_grad = False
            
        # Create adapter layers
        base_output_dim = self._get_output_dim()
        self.down_projection = nn.Linear(base_output_dim, adapter_dim)
        self.up_projection = nn.Linear(adapter_dim, base_output_dim)
        self.activation = nn.GELU()
        
    def _get_output_dim(self) -> int:
        """Get the output dimension of the base model."""
        # This would need to be implemented based on your specific model
        # For example, for ViT:
        return self.base_model.pretrained_model.config.hidden_size
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with residual adapter connection.
        
        Args:
            x: Input tensor
            
        Returns:
            Adapted output tensor
        """
        # Get base model output
        base_output = self.base_model(x)
        
        # Apply adapter
        adapter_output = self.down_projection(base_output)
        adapter_output = self.activation(adapter_output)
        adapter_output = self.up_projection(adapter_output)
        
        # Residual connection
        return base_output + adapter_output