# src/retrieval/claude_rag.py
from typing import List, Dict, Any, Optional

import subprocess
import json
import os

class ClaudeRAGAssistant:
    def __init__(self, 
                vector_store_path: str,
                embedding_model=None,
                temperature: float = 0.7,
                model: str = "claude-3-sonnet-20240229"):
        """Initialize Claude RAG Assistant.
        
        Args:
            vector_store_path: Path to vector store
            embedding_model: Optional embedding model (uses Claude if None)
            temperature: Temperature for generation
            model: Claude model to use
        """
        self.vector_store_path = vector_store_path
        self.embedding_model = embedding_model  # Use for query embedding
        self.temperature = temperature
        self.model = model
        
        # Load vector store for direct access
        from src.retrieval.vector_store import VectorStore
        self.vector_store = VectorStore.load(vector_store_path)
        
    def query(self, user_query: str, num_docs: int = 5) -> str:
        """Process RAG query using Claude Code.
        
        Args:
            user_query: User's question
            num_docs: Number of documents to retrieve
            
        Returns:
            Claude's response
        """
        # If we have an embedding model, use it
        if self.embedding_model:
            query_embedding = self.embedding_model.embed_query(user_query).cpu().numpy()
            retrieved_docs = self.vector_store.search(query_embedding, k=num_docs)
            context = "\n\n".join([doc["document"] for doc in retrieved_docs])
        else:
            # We'll let Claude handle retrieval (less optimal)
            context = "Please use your knowledge to answer."
            
        # Prepare the prompt
        prompt = f"""
        Context information:
        {context}
        
        Question: {user_query}
        
        Please provide a detailed, accurate answer based on the context information provided.
        If the context doesn't contain relevant information, please say so rather than making up an answer.
        """
        
        # Call Claude Code CLI
        try:
            result = subprocess.run(
                ["claude", "chat", 
                 "--model", self.model,
                 "--temperature", str(self.temperature)],
                input=prompt.encode(),
                capture_output=True,
                text=True
            )
            response = result.stdout.strip()
            return response
        except Exception as e:
            return f"Error querying Claude: {str(e)}"
    
    def batch_index_with_claude(self, files_path: str, chunk_size: int = 1000):
        """Use Claude to help process and index documents.
        
        Args:
            files_path: Path to documents directory
            chunk_size: Size of text chunks
        """
        files = [os.path.join(files_path, f) for f in os.listdir(files_path) 
                if os.path.isfile(os.path.join(files_path, f))]
        
        all_chunks = []
        all_metadata = []
        
        for file_path in files:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # Use Claude to intelligently chunk the document
                chunk_prompt = f"""
                I have a document that I need to split into semantically meaningful chunks for a RAG system.
                Please split the following document into chunks of approximately {chunk_size} characters.
                Make sure each chunk is a coherent semantic unit.
                
                Document:
                {content[:10000]}  # Only process first 10K chars as example
                
                Return the result as a JSON array where each element is a chunk.
                """
                
                # Call Claude Code CLI
                result = subprocess.run(
                    ["claude", "chat", "--model", "claude-3-haiku-20240307"],
                    input=chunk_prompt.encode(),
                    capture_output=True,
                    text=True
                )
                
                # Extract JSON array from Claude's response
                claude_response = result.stdout
                # Find JSON array in response
                import re
                json_match = re.search(r'\[[\s\S]*\]', claude_response)
                if json_match:
                    chunks = json.loads(json_match.group(0))
                else:
                    chunks = [content[:chunk_size]]  # Fallback
                
                # Prepare metadata
                for chunk in chunks:
                    all_chunks.append(chunk)
                    all_metadata.append({
                        "source": file_path,
                        "chunk_size": len(chunk)
                    })
                    
            except Exception as e:
                print(f"Error processing {file_path}: {e}")
        
        # Now index these chunks
        if self.embedding_model:
            embeddings = self.embedding_model.embed_texts(all_chunks)
            self.vector_store.add_documents(
                documents=all_chunks,
                embeddings=embeddings.cpu().numpy(),
                metadata=all_metadata
            )
            self.vector_store.save(self.vector_store_path)
            return len(all_chunks)
        else:
            return "No embedding model provided for indexing"