# src/retrieval/multimodal_rag.py

import torch
import numpy as np
from typing import List, Dict, Any, Optional, Union, Tuple

from src.retrieval.rag import RAGOrchestrator
from src.models.pretrained.clip_model import CLIPModelWrapper

class MultimodalRAG(RAGOrchestrator):
    def __init__(self, 
                vector_store,
                text_embedding_model,
                image_embedding_model,
                generator,
                num_retrieved_docs: int = 5,
                include_docs_in_response: bool = False):
        """Initialize multimodal RAG.
        
        Args:
            vector_store: Vector store for document retrieval
            text_embedding_model: Model for text embeddings
            image_embedding_model: Model for image embeddings
            generator: Text generation model
            num_retrieved_docs: Number of documents to retrieve
            include_docs_in_response: Whether to include retrieved docs in response
        """
        super().__init__(
            vector_store=vector_store,
            embedding_model=text_embedding_model,
            generator=generator,
            num_retrieved_docs=num_retrieved_docs,
            include_docs_in_response=include_docs_in_response
        )
        self.image_embedding_model = image_embedding_model
        
    def process_image_query(self, 
                           image: torch.Tensor, 
                           text_query: Optional[str] = None,
                           max_new_tokens: int = 50) -> Dict[str, Any]:
        """Process a query with an image through the multimodal RAG pipeline.
        
        Args:
            image: Image tensor
            text_query: Optional text query to accompany the image
            max_new_tokens: Maximum tokens to generate
            
        Returns:
            Dict containing the response and metadata
        """
        # Get image embedding
        image_embedding = self.image_embedding_model.encode_image(image)
        
        # Get text embedding if provided
        if text_query:
            text_embedding = self.embedding_model.embed_query(text_query)
            # Combine embeddings (simple average as an example)
            combined_embedding = (image_embedding + text_embedding) / 2
            query_embedding = combined_embedding
        else:
            query_embedding = image_embedding
            
        # Retrieve documents
        retrieved_docs = self.vector_store.search(
            query_embedding.cpu().numpy(),
            k=self.num_retrieved_docs
        )
        
        # Construct prompt
        context = "\n\n".join([doc["document"] for doc in retrieved_docs])
        
        if text_query:
            prompt = f"Context information:\n{context}\n\nQuestion about the image: {text_query}\n\nAnswer:"
        else:
            prompt = f"Context information:\n{context}\n\nDescribe the image based on the retrieved information:\n\nAnswer:"
            
        # Generate response
        response = self.generator.generate(
            prompt=prompt,
            max_new_tokens=max_new_tokens
        )
        
        # Prepare result
        result = {
            "image_query": True,
            "text_query": text_query if text_query else None,
            "response": response,
        }
        
        if self.include_docs_in_response:
            result["retrieved_documents"] = retrieved_docs
            
        return result
    
    def index_image_text_pairs(self, 
                              images: List[torch.Tensor], 
                              texts: List[str],
                              metadata: Optional[List[Dict[str, Any]]] = None):
        """Index image-text pairs for multimodal retrieval.
        
        Args:
            images: List of image tensors
            texts: Corresponding text descriptions
            metadata: Optional metadata for each pair
        """
        # Generate embeddings
        image_embeddings = torch.stack([self.image_embedding_model.encode_image(img) for img in images])
        text_embeddings = self.embedding_model.embed_texts(texts)
        
        # Create combined embeddings (simple average as an example)
        combined_embeddings = (image_embeddings + text_embeddings) / 2
        
        # Add to vector store
        self.vector_store.add_documents(
            documents=texts,
            embeddings=combined_embeddings.cpu().numpy(),
            metadata=metadata
        )
        
        return len(texts)