"""MODULE: feed_forward.py
PURPOSE: Implements various feed-forward neural network architectures with modern features
KEY COMPONENTS:
- FeedForwardNN: Base class for configurable feed-forward networks
- FeedForwardClassifier: Specialized classifier with training utilities
- MultiLayerPerceptron: Traditional MLP implementation with modern features
DEPENDENCIES: torch, torch.nn, torch.nn.functional, typing, .base_model, .layers
SPECIAL NOTES: Provides flexible architectures with options for layer normalization, residual connections, and dropout"""

import os
from typing import Dict, List, Literal

import torch
import torch.nn as nn
import torch.nn.functional as F

from .base_model import BaseModel
from .layers import FeedForwardBlock, LinearLayer


class FeedForwardNN(BaseModel):
    """
    A configurable feed-forward neural network model.
    """

    def __init__(
        self,
        input_size: int,
        hidden_sizes: List[int],
        output_size: int,
        activation: Literal['relu', 'gelu', 'tanh', 'sigmoid'] = "relu",
        dropout: float = 0.0,
        use_layer_norm: bool = False,
        use_residual: bool = False,
    ):
        """
        Initialize the feed-forward neural network.

        Args:
            input_size: Size of the input features
            hidden_sizes: List of hidden layer sizes
            output_size: Size of the output features
            activation: Activation function to use ('relu', 'gelu', 'tanh', 'sigmoid')
            dropout: Dropout probability
            use_layer_norm: Whether to use layer normalization
            use_residual: Whether to use residual connections
        """
        super().__init__()

        # Store model hyperparameters
        self.input_size = input_size
        self.hidden_sizes = hidden_sizes
        self.output_size = output_size
        self.activation = activation

        # Create the network layers
        layer_dims = [input_size] + hidden_sizes
        layers = []

        # Create hidden layers
        for i in range(len(layer_dims) - 1):
            layers.append(
                FeedForwardBlock(
                    input_dim=layer_dims[i],
                    output_dim=layer_dims[i + 1],
                    activation=activation,
                    dropout=dropout,
                    use_layer_norm=use_layer_norm,
                    use_residual=use_residual and layer_dims[i] == layer_dims[i + 1],
                )
            )

        self.hidden_layers = nn.ModuleList(layers)

        # Output layer (no activation, dropout, or layer norm in output layer)
        self.output_layer = LinearLayer(
            hidden_sizes[-1] if hidden_sizes else input_size, output_size
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the network.

        Args:
            x: Input tensor of shape [batch_size, input_size]

        Returns:
            Output tensor of shape [batch_size, output_size]
        """
        # Pass input through hidden layers
        for layer in self.hidden_layers:
            x = layer(x)

        # Pass through output layer
        x = self.output_layer(x)

        return x


class FeedForwardClassifier(FeedForwardNN):
    """
    A feed-forward neural network classifier.
    """

    def __init__(
        self,
        input_size: int,
        hidden_sizes: List[int],
        num_classes: int,
        activation: Literal['relu', 'gelu', 'tanh', 'sigmoid'] = "relu",
        dropout: float = 0.0,
        use_layer_norm: bool = False,
        use_residual: bool = False,
    ):
        """
        Initialize the feed-forward classifier.

        Args:
            input_size: Size of the input features
            hidden_sizes: List of hidden layer sizes
            num_classes: Number of classes
            activation: Activation function to use
            dropout: Dropout probability
            use_layer_norm: Whether to use layer normalization
            use_residual: Whether to use residual connections
        """
        super().__init__(
            input_size=input_size,
            hidden_sizes=hidden_sizes,
            output_size=num_classes,
            activation=activation,
            dropout=dropout,
            use_layer_norm=use_layer_norm,
            use_residual=use_residual,
        )

        self.num_classes = num_classes

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the classifier.

        Args:
            x: Input tensor of shape [batch_size, input_size]

        Returns:
            Logits tensor of shape [batch_size, num_classes]
        """
        # Use the parent class forward method
        return super().forward(x)

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """
        Make predictions on input data.

        Args:
            x: Input tensor of shape [batch_size, input_size]

        Returns:
            Predicted class indices of shape [batch_size]
        """
        with torch.no_grad():
            logits = self.forward(x)
            return torch.argmax(logits, dim=1)

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """
        Get class probabilities for input data.

        Args:
            x: Input tensor of shape [batch_size, input_size]

        Returns:
            Class probabilities of shape [batch_size, num_classes]
        """
        with torch.no_grad():
            logits = self.forward(x)
            return F.softmax(logits, dim=1)

    def training_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Perform a single training step.

        Args:
            batch: Dictionary containing 'inputs' and 'targets'

        Returns:
            Dictionary with loss and other metrics
        """
        inputs, targets = batch["inputs"], batch["targets"]

        # Forward pass
        logits = self.forward(inputs)

        # Calculate loss
        loss = F.cross_entropy(logits, targets)

        # Calculate accuracy
        predictions = torch.argmax(logits, dim=1)
        accuracy = (predictions == targets).float().mean()

        return {"loss": loss, "accuracy": accuracy, "predictions": predictions}

    def validation_step(
        self, batch: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """
        Perform a single validation step.

        Args:
            batch: Dictionary containing 'inputs' and 'targets'

        Returns:
            Dictionary with loss and other metrics
        """
        # Validation is similar to training, but without gradient tracking
        with torch.no_grad():
            return self.training_step(batch)

    def configure_optimizers(self, lr: float = 0.001) -> torch.optim.Optimizer:
        """
        Configure the optimizer for training.

        Args:
            lr: Learning rate

        Returns:
            Configured optimizer
        """
        return torch.optim.Adam(self.parameters(), lr=lr)


class MultiLayerPerceptron(nn.Module):
    """
    A multi-layer perceptron (MLP) consisting of multiple feed-forward blocks.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int],
        output_dim: int,
        activation: Literal['relu', 'gelu', 'tanh', 'sigmoid'] = "relu",
        dropout: float = 0.0,
        use_layer_norm: bool = False,
        use_residual: bool = False,
    ):
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
                        output_dim=layer_dims[i + 1],
                        activation=activation,
                        dropout=dropout,
                        use_layer_norm=use_layer_norm,
                        use_residual=use_residual
                        and layer_dims[i] == layer_dims[i + 1],
                    )
                )
            # For the last layer, no dropout or layer norm
            else:
                layers.append(
                    FeedForwardBlock(
                        input_dim=layer_dims[i],
                        output_dim=layer_dims[i + 1],
                        activation=activation,
                        dropout=0.0,
                        use_layer_norm=False,
                        use_residual=False,
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
        "module_purpose": "Implements various feed-forward neural network architectures with modern features for flexible model building",
        "key_classes": [
            {
                "name": "FeedForwardNN",
                "purpose": "Base class for configurable feed-forward neural networks with optional layer normalization and residual connections",
                "key_methods": [
                    {
                        "name": "__init__",
                        "signature": "__init__(self, input_size: int, hidden_sizes: List[int], output_size: int, activation: Literal['relu', 'gelu', 'tanh', 'sigmoid'] = 'relu', dropout: float = 0.0, use_layer_norm: bool = False, use_residual: bool = False)",
                        "brief_description": "Initializes a configurable feed-forward neural network with given architecture"
                    },
                    {
                        "name": "forward",
                        "signature": "forward(self, x: torch.Tensor) -> torch.Tensor",
                        "brief_description": "Performs forward pass through the network layers"
                    }
                ],
                "inheritance": "BaseModel",
                "dependencies": ["torch", "torch.nn", ".base_model", ".layers"]
            },
            {
                "name": "FeedForwardClassifier",
                "purpose": "Specialized feed-forward classifier with training utilities and prediction methods",
                "key_methods": [
                    {
                        "name": "predict",
                        "signature": "predict(self, x: torch.Tensor) -> torch.Tensor",
                        "brief_description": "Makes class predictions by selecting highest probability class"
                    },
                    {
                        "name": "predict_proba",
                        "signature": "predict_proba(self, x: torch.Tensor) -> torch.Tensor",
                        "brief_description": "Returns class probabilities using softmax on logits"
                    },
                    {
                        "name": "training_step",
                        "signature": "training_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]",
                        "brief_description": "Performs a single training step with loss calculation and metrics"
                    }
                ],
                "inheritance": "FeedForwardNN",
                "dependencies": ["torch", "torch.nn.functional"]
            },
            {
                "name": "MultiLayerPerceptron",
                "purpose": "Traditional MLP implementation with modern features like layer normalization and residual connections",
                "key_methods": [
                    {
                        "name": "forward",
                        "signature": "forward(self, x: torch.Tensor) -> torch.Tensor",
                        "brief_description": "Passes input through all network layers with optional skip connections"
                    }
                ],
                "inheritance": "nn.Module",
                "dependencies": ["torch", "torch.nn"]
            }
        ],
        "external_dependencies": ["torch"],
        "complexity_score": 6  # Moderate complexity due to configuration options and training utilities
    }
