# src/retrieval/vector_store.py
from typing import List, Dict, Any, Optional
import pickle

import numpy as np
import faiss

class VectorStore:
    def __init__(self, embedding_dim: int = 768):
        """Initialize a vector store for RAG retrieval.
        
        Args:
            embedding_dim: Dimension of the embeddings used
        """
        self.embedding_dim = embedding_dim
        self.index = faiss.IndexFlatL2(embedding_dim)  # L2 distance index
        self.documents = []
        self.document_metadata = []
        
    def add_documents(self, 
                     documents: List[str], 
                     embeddings: np.ndarray,
                     metadata: Optional[List[Dict[str, Any]]] = None):
        """Add documents and their embeddings to the vector store.
        
        Args:
            documents: List of document texts
            embeddings: Numpy array of embeddings with shape (n_docs, embedding_dim)
            metadata: Optional list of metadata dictionaries for each document
        """
        # Validate inputs
        assert len(documents) == embeddings.shape[0], "Number of documents must match number of embeddings"
        if metadata is not None:
            assert len(documents) == len(metadata), "Number of documents must match number of metadata entries"
        else:
            metadata = [{} for _ in range(len(documents))]
            
        # Store documents and metadata
        start_idx = len(self.documents)
        self.documents.extend(documents)
        self.document_metadata.extend(metadata)
        
        # Add embeddings to the index
        embeddings = embeddings.astype(np.float32)  # FAISS requires float32
        self.index.add(embeddings)  # noqa: E call-arg
        
        return start_idx, start_idx + len(documents) - 1
    
    def search(self, query_embedding: np.ndarray, k: int = 5) -> List[Dict[str, Any]]:
        """Search for similar documents using a query embedding.
        
        Args:
            query_embedding: Embedding of the query with shape (embedding_dim,)
            k: Number of results to return
            
        Returns:
            List of dictionaries containing retrieved documents, scores, and metadata
        """
        # Reshape query if needed
        if len(query_embedding.shape) == 1:
            query_embedding = query_embedding.reshape(1, -1)
        
        # Ensure float32 type
        query_embedding = query_embedding.astype(np.float32)
        
        # Search the index
        distances, indices = self.index.search(query_embedding, k)  # noqa: E call-arg
        
        # Prepare results
        results = []
        for i, (distance, idx) in enumerate(zip(distances[0], indices[0])):
            if idx >= 0 and idx < len(self.documents):  # Valid index
                results.append({
                    "document": self.documents[idx],
                    "score": float(distance),
                    "metadata": self.document_metadata[idx]
                })
        
        return results
    
    def save(self, path: str):
        """Save the vector store to disk."""
        # Save the index
        faiss.write_index(self.index, f"{path}_index.faiss")
        
        # Save documents and metadata
        with open(f"{path}_data.pkl", "wb") as f:
            pickle.dump({
                "documents": self.documents,
                "document_metadata": self.document_metadata,
                "embedding_dim": self.embedding_dim
            }, f)
    
    @classmethod
    def load(cls, path: str) -> "VectorStore":
        """Load a vector store from disk."""
        # Load data
        with open(f"{path}_data.pkl", "rb") as f:
            data = pickle.load(f)
        
        # Create instance
        instance = cls(embedding_dim=data["embedding_dim"])
        instance.documents = data["documents"]
        instance.document_metadata = data["document_metadata"]
        
        # Load index
        instance.index = faiss.read_index(f"{path}_index.faiss")
        
        return instance