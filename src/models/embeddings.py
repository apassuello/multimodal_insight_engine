import math
import os

import torch
import torch.nn as nn

"""MODULE: embeddings.py
PURPOSE: Implements token embedding layers for transformer models with proper initialization and scaling
KEY COMPONENTS:
- TokenEmbedding: Neural network layer that converts token indices to dense vector representations
DEPENDENCIES: torch, torch.nn, math, typing
SPECIAL NOTES: Implements the embedding scaling factor of sqrt(d_model) as per the original transformer paper"""

class TokenEmbedding(nn.Module):
    """
    Token embedding layer for transformer models.
    
    This layer converts token indices to dense vector representations.
    It also scales the embeddings by sqrt(d_model) as per the original transformer paper.
    """

    def __init__(self, vocab_size: int, d_model: int):
        """
        Initialize the token embedding layer.
        
        Args:
            vocab_size: Size of the vocabulary
            d_model: Dimension of the embeddings
        """
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, d_model)
        self.d_model = d_model

        # Initialize weights using Xavier uniform initialization
        nn.init.xavier_uniform_(self.embedding.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Convert token indices to embeddings.
        
        Args:
            x: Token indices of shape [batch_size, seq_length]
            
        Returns:
            Token embeddings of shape [batch_size, seq_length, d_model]
        """
        # Apply embedding and scale by sqrt(d_model)
        embeddings = self.embedding(x)

        # Ensure embeddings have the correct shape and dimension
        batch_size, seq_length = x.shape
        embeddings = embeddings.view(batch_size, seq_length, self.d_model)

        # Scale embeddings
        return embeddings * math.sqrt(self.d_model)

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
        "module_purpose": "Implements token embedding layers for transformer models with proper initialization and scaling",
        "key_classes": [
            {
                "name": "TokenEmbedding",
                "purpose": "Neural network layer that converts token indices to dense vector representations with proper scaling",
                "key_methods": [
                    {
                        "name": "__init__",
                        "signature": "__init__(self, vocab_size: int, d_model: int)",
                        "brief_description": "Initialize the embedding layer with Xavier uniform initialization"
                    },
                    {
                        "name": "forward",
                        "signature": "forward(self, x: torch.Tensor) -> torch.Tensor",
                        "brief_description": "Convert token indices to scaled embeddings"
                    }
                ],
                "inheritance": "nn.Module",
                "dependencies": ["torch", "torch.nn", "math"]
            }
        ],
        "external_dependencies": ["torch"],
        "complexity_score": 2,  # Simple module with straightforward embedding functionality
    }
