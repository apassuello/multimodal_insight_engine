# src/retrieval/rag.py
from typing import List, Dict, Any, Optional

import torch
import numpy as np
from src.retrieval.vector_store import VectorStore
from src.retrieval.embedding import EmbeddingModel
from src.models.text_generation import TextGenerator

class RAGOrchestrator:
    def __init__(self, 
                vector_store: VectorStore,
                embedding_model: EmbeddingModel,
                generator: TextGenerator,
                num_retrieved_docs: int = 5,
                include_docs_in_response: bool = False):
        """Initialize the RAG orchestrator.
        
        Args:
            vector_store: Vector store containing documents
            embedding_model: Model for embedding queries
            generator: Text generation model
            num_retrieved_docs: Number of documents to retrieve
            include_docs_in_response: Whether to include retrieved docs in response
        """
        self.vector_store = vector_store
        self.embedding_model = embedding_model
        self.generator = generator
        self.num_retrieved_docs = num_retrieved_docs
        self.include_docs_in_response = include_docs_in_response
        
    def process_query(self, query: str, max_new_tokens: int = 50) -> Dict[str, Any]:
        """Process a query through the RAG pipeline.
        
        Args:
            query: User query string
            max_new_tokens: Maximum tokens to generate
            
        Returns:
            Dict containing the response and metadata
        """
        # Step 1: Embed the query
        query_embedding = self.embedding_model.embed_query(query)
        
        # Step 2: Retrieve relevant documents
        retrieved_docs = self.vector_store.search(
            query_embedding.cpu().numpy(),
            k=self.num_retrieved_docs
        )
        
        # Step 3: Construct augmented prompt
        context = "\n\n".join([doc["document"] for doc in retrieved_docs])
        augmented_prompt = f"Context information:\n{context}\n\nQuestion: {query}\n\nAnswer:"
        
        # Step 4: Generate response
        response = self.generator.generate(
            prompt=augmented_prompt,
            max_new_tokens=max_new_tokens
        )
        
        # Prepare result
        result = {
            "query": query,
            "response": response,
        }
        
        if self.include_docs_in_response:
            result["retrieved_documents"] = retrieved_docs
            
        return result
    
    def add_documents(self, documents: List[str], metadata: Optional[List[Dict[str, Any]]] = None):
        """Add documents to the RAG system.
        
        Args:
            documents: List of documents to add
            metadata: Optional metadata for each document
        """
        # Embed documents
        embeddings = self.embedding_model.embed_texts(documents)
        
        # Add to vector store
        self.vector_store.add_documents(
            documents=documents,
            embeddings=embeddings.cpu().numpy(),
            metadata=metadata
        )
        
        return len(documents)