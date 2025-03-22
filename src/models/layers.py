import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional, Union, Literal

class LinearLayer(nn.Module):
    """A linear layer with initialization, dropout, and normalization options."""
    
    def __init__(self, in_features: int, out_features: int, bias: bool = True, 
                 init_type: str = 'kaiming_uniform', dropout: float = 0.0,
                 use_layer_norm: bool = False):
        """
        Initialize the linear layer.
        
        Args:
            in_features: Size of each input sample
            out_features: Size of each output sample
            bias: If set to False, the layer will not learn an additive bias
            init_type: Weight initialization method
            dropout: Dropout probability (0.0 means no dropout)
            use_layer_norm: Whether to apply layer normalization
        """
        super().__init__()
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        self._init_weights(init_type)
        
        # Dropout for regularization
        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else None
        
        # Layer normalization
        self.layer_norm = nn.LayerNorm(out_features) if use_layer_norm else None
    
    def _init_weights(self, init_type: str) -> None:
        """Initialize the weights using the specified method."""
        if init_type == 'kaiming_uniform':
            nn.init.kaiming_uniform_(self.linear.weight, nonlinearity='relu')
        elif init_type == 'kaiming_normal':
            nn.init.kaiming_normal_(self.linear.weight, nonlinearity='relu')
        elif init_type == 'xavier_uniform':
            nn.init.xavier_uniform_(self.linear.weight)
        elif init_type == 'xavier_normal':
            nn.init.xavier_normal_(self.linear.weight)
        else:
            raise ValueError(f"Unknown initialization type: {init_type}")
            
        if self.linear.bias is not None:
            nn.init.zeros_(self.linear.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the linear layer.
        
        Args:
            x: Input tensor
            
        Returns:
            Transformed output tensor
        """
        # Move input to device
        x = x.to(next(self.parameters()).device)
        
        x = self.linear(x)
        
        # Apply layer normalization if specified
        if self.layer_norm is not None:
            x = self.layer_norm(x)
            
        # Apply dropout if specified
        if self.dropout is not None:
            x = self.dropout(x)
            
        return x


class FeedForwardBlock(nn.Module):
    """
    A feed-forward block with optional residual connection.
    """
    
    def __init__(self, input_dim: int, hidden_dim: Optional[int] = None, 
                 output_dim: Optional[int] = None, 
                 activation: Literal['relu', 'gelu', 'tanh', 'sigmoid'] = 'relu', 
                 dropout: float = 0.0, use_layer_norm: bool = False,
                 use_residual: bool = False):
        """
        Initialize the feed-forward block.
        
        Args:
            input_dim: Dimension of input features
            hidden_dim: Dimension of hidden layer (if None, uses 4*input_dim)
            output_dim: Dimension of output features (if None, uses input_dim)
            activation: Activation function
            dropout: Dropout probability
            use_layer_norm: Whether to apply layer normalization
            use_residual: Whether to use a residual (skip) connection
        """
        super().__init__()
        
        # Set default dimensions if not provided
        hidden_dim = hidden_dim if hidden_dim is not None else 4 * input_dim
        output_dim = output_dim if output_dim is not None else input_dim
        
        # Check if residual connection is possible
        self.use_residual = use_residual and input_dim == output_dim
        if use_residual and input_dim != output_dim:
            print(f"Warning: Cannot use residual connection when input_dim ({input_dim}) "
                  f"!= output_dim ({output_dim}). Disabling residual connection.")
        
        # First linear layer
        self.linear1 = LinearLayer(
            input_dim, 
            hidden_dim, 
            init_type='kaiming_uniform' if activation == 'relu' else 'xavier_uniform',
            dropout=0.0,
            use_layer_norm=False
        )
        
        # Second linear layer
        self.linear2 = LinearLayer(
            hidden_dim, 
            output_dim, 
            init_type='xavier_uniform',
            dropout=dropout,
            use_layer_norm=use_layer_norm
        )
        
        # Store the activation type
        self.activation = activation
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the feed-forward block.
        
        Args:
            x: Input tensor
            
        Returns:
            Transformed output tensor
        """
        # Move input to device
        x = x.to(next(self.parameters()).device)
        
        # Store input for residual connection
        residual = x
        
        # First linear layer
        x = self.linear1(x)
        
        # Apply activation function
        if self.activation == 'relu':
            x = F.relu(x)
        elif self.activation == 'gelu':
            x = torch.nn.GELU()(x)
        elif self.activation == 'tanh':
            x = torch.tanh(x)
        elif self.activation == 'sigmoid':
            x = torch.sigmoid(x)
        else:
            raise ValueError(f"Unknown activation function: {self.activation}")
        
        # Second linear layer
        x = self.linear2(x)
        
        # Apply residual connection if specified
        if self.use_residual:
            x = x + residual
        
        return x
