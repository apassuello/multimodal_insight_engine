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