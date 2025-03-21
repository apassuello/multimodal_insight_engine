import torch
import torch.nn as nn
import math
from typing import Optional

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