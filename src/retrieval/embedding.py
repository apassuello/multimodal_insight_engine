# src/retrieval/embedding.py
from typing import List, Union, Dict, Any

import torch
import torch.nn as nn

class EmbeddingModel:
    def __init__(self, model_name_or_path="sentence-transformers/all-MiniLM-L6-v2"):
        """Initialize embedding model for RAG.
        
        Args:
            model_name_or_path: HuggingFace model name or path to local model
        """
        # Import here to avoid dependencies if not using this module
        from sentence_transformers import SentenceTransformer
        
        self.model = SentenceTransformer(model_name_or_path)
        self.embedding_dim = self.model.get_sentence_embedding_dimension()
        
    def embed_texts(self, texts: List[str]) -> torch.Tensor:
        """Embed a list of texts.
        
        Args:
            texts: List of texts to embed
            
        Returns:
            Tensor of embeddings with shape (n_texts, embedding_dim)
        """
        embeddings = self.model.encode(texts, convert_to_tensor=True)
        return embeddings
    
    def embed_query(self, query: str) -> torch.Tensor:
        """Embed a single query text.
        
        Args:
            query: Query text to embed
            
        Returns:
            Tensor of query embedding with shape (1, embedding_dim)
        """
        query_embedding = self.model.encode([query], convert_to_tensor=True)
        return query_embedding

# For your custom model once it's trained:
class TransformerEmbedding:
    def __init__(self, model: nn.Module, tokenizer=None, max_length: int = 512):
        """Use your own transformer model for embeddings.
        
        Args:
            model: Your transformer model
            tokenizer: Your tokenizer
            max_length: Maximum sequence length
        """
        self.model = model
        self.tokenizer = tokenizer
        self.max_length = max_length
        
    def embed_texts(self, texts: List[str]) -> torch.Tensor:
        """Generate embeddings using your transformer."""
        # Implementation depends on your model's specifics
        # This is a placeholder
        embeddings = []
        for text in texts:
            tokens = self.tokenizer.encode(text)[:self.max_length]
            with torch.no_grad():
                output = self.model(torch.tensor([tokens]))
                # Assuming your model outputs hidden states or has an embedding method
                embedding = output.mean(dim=1)  # Example pooling strategy
                embeddings.append(embedding)
        return torch.cat(embeddings, dim=0)