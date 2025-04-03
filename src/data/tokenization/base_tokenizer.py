# src/data/tokenization/base_tokenizer.py
from abc import ABC, abstractmethod
from typing import List, Dict
import os

class BaseTokenizer(ABC):
    """
    Abstract base class for all tokenizers.
    
    This defines the standard interface that all tokenizer implementations
    should follow to ensure interoperability with the transformer model.
    """
    
    @abstractmethod
    def tokenize(self, text: str) -> List[str]:
        """
        Convert a text string into a list of tokens.
        
        Args:
            text: The input text to tokenize
            
        Returns:
            A list of tokens
        """
        pass
    
    @abstractmethod
    def encode(self, text: str) -> List[int]:
        """
        Convert a text string into a list of token indices.
        
        Args:
            text: The input text to encode
            
        Returns:
            A list of token indices
        """
        pass
    
    @abstractmethod
    def decode(self, token_ids: List[int]) -> str:
        """
        Convert a list of token indices back into a text string.
        
        Args:
            token_ids: The token indices to decode
            
        Returns:
            The reconstructed text
        """
        pass
    
    @abstractmethod
    def batch_encode(self, texts: List[str]) -> List[List[int]]:
        """
        Encode a batch of texts into token indices.
        
        Args:
            texts: List of input texts to encode
            
        Returns:
            List of token index lists
        """
        pass
    
    @property
    @abstractmethod
    def vocab_size(self) -> int:
        """
        Get the vocabulary size of this tokenizer.
        
        Returns:
            The number of tokens in the vocabulary
        """
        pass
    
    @property
    @abstractmethod
    def special_tokens(self) -> Dict[str, int]:
        """
        Get the special tokens used by this tokenizer.
        
        Returns:
            Dictionary mapping special token names to their indices
        """
        pass
        
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
        "module_purpose": "Defines the abstract base class for all tokenizers in the system with standard interface",
        "key_classes": [
            {
                "name": "BaseTokenizer",
                "purpose": "Abstract base class that defines the standard interface for all tokenizer implementations",
                "key_methods": [
                    {
                        "name": "tokenize",
                        "signature": "tokenize(self, text: str) -> List[str]",
                        "brief_description": "Convert text into tokens"
                    },
                    {
                        "name": "encode",
                        "signature": "encode(self, text: str) -> List[int]",
                        "brief_description": "Convert text to token indices"
                    },
                    {
                        "name": "decode",
                        "signature": "decode(self, token_ids: List[int]) -> str",
                        "brief_description": "Convert token indices back to text"
                    },
                    {
                        "name": "batch_encode",
                        "signature": "batch_encode(self, texts: List[str]) -> List[List[int]]",
                        "brief_description": "Encode multiple texts efficiently"
                    },
                    {
                        "name": "vocab_size",
                        "signature": "vocab_size(self) -> int",
                        "brief_description": "Get the size of the vocabulary"
                    },
                    {
                        "name": "special_tokens",
                        "signature": "special_tokens(self) -> Dict[str, int]",
                        "brief_description": "Get the special tokens used by this tokenizer"
                    }
                ],
                "inheritance": "ABC",
                "dependencies": ["abc.ABC"]
            }
        ],
        "external_dependencies": ["abc"],
        "complexity_score": 2  # Low complexity as it's just an interface definition
    }