"""MODULE: activations.py
PURPOSE: Implements various activation functions used in the transformer architecture
KEY COMPONENTS:
- GELU: Gaussian Error Linear Unit activation function, commonly used in transformer models
DEPENDENCIES: torch, torch.nn, torch.nn.functional
SPECIAL NOTES: This module provides activation functions optimized for transformer architectures"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import os

class GELU(nn.Module):
    """
    Gaussian Error Linear Unit activation function.
    
    GELU is a smooth, non-linear activation function that is commonly used in transformer
    architectures. It approximates the cumulative distribution function of the normal
    distribution and has been shown to work well in deep neural networks.
    
    The implementation uses the approximate form: x * 0.5 * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x^3)))
    """
    
    def __init__(self):
        """Initialize the GELU activation layer."""
        super().__init__()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply the GELU activation function.
        
        Args:
            x: Input tensor of any shape
            
        Returns:
            Tensor of the same shape as input with GELU activation applied
        """
        return F.gelu(x)

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
        "module_purpose": "Implements various activation functions used in the transformer architecture",
        "key_classes": [
            {
                "name": "GELU",
                "purpose": "Gaussian Error Linear Unit activation function for transformer architectures",
                "key_methods": [
                    {
                        "name": "__init__",
                        "signature": "__init__(self)",
                        "brief_description": "Initialize the GELU activation layer"
                    },
                    {
                        "name": "forward",
                        "signature": "forward(self, x: torch.Tensor) -> torch.Tensor",
                        "brief_description": "Apply the GELU activation function to the input tensor"
                    }
                ],
                "inheritance": "nn.Module",
                "dependencies": ["torch", "torch.nn", "torch.nn.functional"]
            }
        ],
        "external_dependencies": ["torch"],
        "complexity_score": 1,  # Very simple module with a single activation function
    }

# Additional activations as needed