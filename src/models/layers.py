import torch
import torch.nn as nn
import torch.nn.init as init

class LinearLayer(nn.Module):
    """A linear layer with customizable initialization."""
    
    def __init__(self, in_features, out_features, bias=True, 
                 init_type='kaiming_uniform'):
        """
        Initialize the linear layer.
        
        Args:
            in_features: Size of each input sample
            out_features: Size of each output sample
            bias: If set to False, the layer will not learn an additive bias
            init_type: Weight initialization method ('kaiming_uniform', 
                       'kaiming_normal', 'xavier_uniform', 'xavier_normal')
        """
        super().__init__()
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        self._init_weights(init_type)
    
    def _init_weights(self, init_type):
        """
        Initialize the weights using the specified method.
        
        Args:
            init_type: Weight initialization method
        """
        if init_type == 'kaiming_uniform':
            init.kaiming_uniform_(self.linear.weight, nonlinearity='relu')
        elif init_type == 'kaiming_normal':
            init.kaiming_normal_(self.linear.weight, nonlinearity='relu')
        elif init_type == 'xavier_uniform':
            init.xavier_uniform_(self.linear.weight)
        elif init_type == 'xavier_normal':
            init.xavier_normal_(self.linear.weight)
        else:
            raise ValueError(f"Unknown initialization type: {init_type}")
            
        if self.linear.bias is not None:
            init.zeros_(self.linear.bias)
    
    def forward(self, x):
        """
        Forward pass of the linear layer.
        
        Args:
            x: Input tensor of shape [batch_size, in_features]
            
        Returns:
            Output tensor of shape [batch_size, out_features]
        """
        return self.linear(x)
class FeedForwardBlock(nn.Module):
    """Feed-forward block with configurable activation."""
    # We'll implement this together