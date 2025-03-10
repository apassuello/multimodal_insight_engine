# Structure outline
import torch
import torch.nn as nn
from .base_model import BaseModel
from .layers import LinearLayer, FeedForwardBlock

class FeedForwardNN(BaseModel):
    """
    A configurable feed-forward neural network.
    """
    
    def __init__(self, input_size, hidden_sizes, output_size, activation='relu', dropout=0.0):
        """
        Initialize the feed-forward neural network.
        
        Args:
            input_size (int): Size of the input features
            hidden_sizes (list): List of hidden layer sizes
            output_size (int): Size of the output features
            activation (str): Activation function to use
            dropout (float): Dropout probability
        """
        # We'll implement this together
    
    def forward(self, x):
        """Forward pass of the network."""
        # We'll implement this together

class MultiLayerPerceptron(nn.Module):
    """
    A multi-layer perceptron (MLP) consisting of multiple feed-forward blocks.
    """
    
    def __init__(self, input_dim: int, hidden_dims: List[int], output_dim: int, 
                 activation: str = 'relu', dropout: float = 0.0, 
                 use_layer_norm: bool = False, use_residual: bool = False):
        """
        Initialize the multi-layer perceptron.
        
        Args:
            input_dim: Dimension of input features
            hidden_dims: List of hidden layer dimensions
            output_dim: Dimension of output features
            activation: Activation function
            dropout: Dropout probability
            use_layer_norm: Whether to apply layer normalization
            use_residual: Whether to use residual connections
        """
        super().__init__()
        
        layer_dims = [input_dim] + hidden_dims + [output_dim]
        layers = []
        
        # Create feed-forward blocks for each layer
        for i in range(len(layer_dims) - 1):
            # For all but the last layer
            if i < len(layer_dims) - 2:
                layers.append(
                    FeedForwardBlock(
                        input_dim=layer_dims[i],
                        output_dim=layer_dims[i+1],
                        activation=activation,
                        dropout=dropout,
                        use_layer_norm=use_layer_norm,
                        use_residual=use_residual and layer_dims[i] == layer_dims[i+1]
                    )
                )
            # For the last layer, no dropout or layer norm
            else:
                layers.append(
                    FeedForwardBlock(
                        input_dim=layer_dims[i],
                        output_dim=layer_dims[i+1],
                        activation=activation,
                        dropout=0.0,
                        use_layer_norm=False,
                        use_residual=False
                    )
                )
        
        # Store the layers in a ModuleList
        self.layers = nn.ModuleList(layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the multi-layer perceptron.
        
        Args:
            x: Input tensor
            
        Returns:
            Output tensor
        """
        for layer in self.layers:
            x = layer(x)
        return x