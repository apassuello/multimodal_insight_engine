# src/data/tokenization/vocabulary.py
from typing import Dict, List, Optional, Any, Union
from collections import Counter
import json
import os
import logging

logger = logging.getLogger(__name__)

class Vocabulary:
    """
    Manages the mapping between tokens and their indices.
    
    This class handles the creation and lookup of token-to-index and
    index-to-token mappings, as well as special tokens.
    """
    
    def __init__(
        self,
        tokens: Optional[List[str]] = None,
        pad_token: str = "<pad>",
        unk_token: str = "<unk>",
        bos_token: str = "<bos>",
        eos_token: str = "<eos>",
        mask_token: str = "<mask>",
    ):
        """
        Initialize the vocabulary.
        
        Args:
            tokens: Optional list of tokens to initially populate the vocabulary
            pad_token: Padding token string
            unk_token: Unknown token string
            bos_token: Beginning of sequence token string
            eos_token: End of sequence token string
            mask_token: Mask token string for masked language modeling
        """
        # Initialize mappings
        self.token_to_idx: Dict[str, int] = {}
        self.idx_to_token: List[str] = []
        
        # Store special tokens
        self.pad_token = pad_token
        self.unk_token = unk_token
        self.bos_token = bos_token
        self.eos_token = eos_token
        self.mask_token = mask_token
        
        # Add special tokens to vocabulary
        self._add_special_tokens()
        
        # Add provided tokens if any
        if tokens:
            for token in tokens:
                self.add_token(token)
    
    def _add_special_tokens(self) -> None:
        """Add special tokens to the vocabulary."""
        for token in [
            self.pad_token,
            self.unk_token,
            self.bos_token,
            self.eos_token,
            self.mask_token,
        ]:
            self.add_token(token)
    
    def add_token(self, token: str) -> int:
        """
        Add a token to the vocabulary if it doesn't already exist.
        
        Args:
            token: The token to add
            
        Returns:
            The index of the token
            
        Raises:
            TypeError: If token is not a string
            ValueError: If token is empty
        """
        # Validate input
        if not isinstance(token, str):
            raise TypeError(f"Token must be a string, got {type(token).__name__}")
        
        if not token:
            raise ValueError("Cannot add empty token to vocabulary")
            
        # Add token if it doesn't exist
        if token not in self.token_to_idx:
            idx = len(self.idx_to_token)
            self.token_to_idx[token] = idx
            self.idx_to_token.append(token)
            return idx
            
        return self.token_to_idx[token]
    
    def token_to_index(self, token: str) -> int:
        """
        Convert a token to its index with validation.
        
        Args:
            token: The token to look up
            
        Returns:
            The index of the token, or the unknown token index if not found
        
        Raises:
            TypeError: If token is not a string
        """
        # Validate input
        if not isinstance(token, str):
            raise TypeError(f"Token must be a string, got {type(token).__name__}")
            
        return self.token_to_idx.get(token, self.token_to_idx[self.unk_token])
    
    def index_to_token(self, idx: int) -> str:
        """
        Convert an index to its token with validation.
        
        Args:
            idx: The index to look up
            
        Returns:
            The token at the given index, or the unknown token if index is out of range
            
        Raises:
            TypeError: If idx is not an integer
        """
        # Validate input
        if not isinstance(idx, int):
            raise TypeError(f"Index must be an integer, got {type(idx).__name__}")
            
        # Look up token
        if 0 <= idx < len(self.idx_to_token):
            return self.idx_to_token[idx]
            
        # Return unknown token for out-of-range indices
        logger.warning(f"Index {idx} out of range for vocabulary size {len(self.idx_to_token)}")
        return self.unk_token
    
    def tokens_to_indices(self, tokens: List[str]) -> List[int]:
        """
        Convert a list of tokens to their indices with validation.
        
        Args:
            tokens: List of tokens to convert
            
        Returns:
            List of corresponding indices
            
        Raises:
            TypeError: If tokens is not a list or contains non-string elements
        """
        # Validate input
        if not isinstance(tokens, list):
            raise TypeError(f"Expected list of strings, got {type(tokens).__name__}")
            
        # Check elements if list is not empty
        if tokens and not all(isinstance(token, str) for token in tokens):
            raise TypeError("All elements in tokens list must be strings")
            
        # Handle empty list
        if not tokens:
            return []
            
        return [self.token_to_index(token) for token in tokens]
    
    def indices_to_tokens(self, indices: List[int]) -> List[str]:
        """
        Convert a list of indices to their tokens with validation.
        
        Args:
            indices: List of indices to convert
            
        Returns:
            List of corresponding tokens
            
        Raises:
            TypeError: If indices is not a list or contains non-integer elements
        """
        # Validate input
        if not isinstance(indices, list):
            raise TypeError(f"Expected list of integers, got {type(indices).__name__}")
            
        # Check elements if list is not empty
        if indices and not all(isinstance(idx, int) for idx in indices):
            raise TypeError("All elements in indices list must be integers")
            
        # Handle empty list
        if not indices:
            return []
            
        return [self.index_to_token(idx) for idx in indices]
    
    def save(self, path: str) -> None:
        """
        Save the vocabulary to a file with validation.
        
        Args:
            path: Path to save the vocabulary
            
        Raises:
            ValueError: If path is empty
            OSError: If unable to create directory or write file
        """
        # Validate path
        if not path:
            raise ValueError("Path cannot be empty")
            
        try:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(path), exist_ok=True)
            
            # Prepare data to save
            vocab_data = {
                "token_to_idx": self.token_to_idx,
                "special_tokens": {
                    "pad_token": self.pad_token,
                    "unk_token": self.unk_token,
                    "bos_token": self.bos_token,
                    "eos_token": self.eos_token,
                    "mask_token": self.mask_token,
                },
                "vocab_size": len(self.idx_to_token)
            }
            
            # Save to file
            with open(path, "w", encoding="utf-8") as f:
                json.dump(vocab_data, f, ensure_ascii=False, indent=2)
                
            logger.info(f"Vocabulary with {len(self)} tokens saved to {path}")
                
        except (OSError, IOError) as e:
            raise OSError(f"Failed to save vocabulary to {path}: {str(e)}")
    
    @classmethod
    def load(cls, path: str) -> "Vocabulary":
        """
        Load a vocabulary from a file with validation.
        
        Args:
            path: Path to load the vocabulary from
            
        Returns:
            Loaded Vocabulary instance
            
        Raises:
            FileNotFoundError: If file doesn't exist
            ValueError: If file format is invalid
            json.JSONDecodeError: If file contains invalid JSON
        """
        # Validate path
        if not path:
            raise ValueError("Path cannot be empty")
            
        if not os.path.exists(path):
            raise FileNotFoundError(f"Vocabulary file not found: {path}")
            
        try:
            with open(path, "r", encoding="utf-8") as f:
                vocab_data = json.load(f)
                
            # Validate required keys
            required_keys = ["token_to_idx", "special_tokens"]
            for key in required_keys:
                if key not in vocab_data:
                    raise ValueError(f"Invalid vocabulary file: missing '{key}' key")
                    
            # Validate special tokens structure
            special_token_keys = ["pad_token", "unk_token", "bos_token", "eos_token", "mask_token"]
            for key in special_token_keys:
                if key not in vocab_data["special_tokens"]:
                    raise ValueError(f"Invalid vocabulary file: missing '{key}' in special_tokens")
            
            # Create a new vocabulary with the special tokens
            special_tokens = vocab_data["special_tokens"]
            vocab = cls(
                pad_token=special_tokens["pad_token"],
                unk_token=special_tokens["unk_token"],
                bos_token=special_tokens["bos_token"],
                eos_token=special_tokens["eos_token"],
                mask_token=special_tokens["mask_token"],
            )
            
            # Override token_to_idx and rebuild idx_to_token
            vocab.token_to_idx = {k: int(v) for k, v in vocab_data["token_to_idx"].items()}
            vocab.idx_to_token = [""] * len(vocab.token_to_idx)
            for token, idx in vocab.token_to_idx.items():
                vocab.idx_to_token[idx] = token
            
            # Validate that idx_to_token has no gaps
            if "" in vocab.idx_to_token:
                raise ValueError("Invalid vocabulary file: index mapping has gaps")
                
            logger.info(f"Loaded vocabulary with {len(vocab)} tokens from {path}")
            
            return vocab
            
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in vocabulary file {path}: {str(e)}")
        except Exception as e:
            raise ValueError(f"Failed to load vocabulary from {path}: {str(e)}")
    
    @classmethod
    def build_from_texts(
        cls,
        texts: List[str],
        tokenizer,
        max_vocab_size: Optional[int] = None,
        min_freq: int = 1,
        **kwargs,
    ) -> "Vocabulary":
        """
        Build a vocabulary from a list of texts with validation.
        
        Args:
            texts: List of texts to build vocabulary from
            tokenizer: Tokenizer function to use
            max_vocab_size: Maximum vocabulary size (None means no limit)
            min_freq: Minimum frequency for a token to be included
            **kwargs: Additional arguments to pass to the Vocabulary constructor
            
        Returns:
            Built Vocabulary instance
            
        Raises:
            TypeError: If texts is not a list of strings or tokenizer is not callable
            ValueError: If min_freq < 1 or max_vocab_size < 1
        """
        # Validate inputs
        if not isinstance(texts, list):
            raise TypeError(f"Expected list of strings, got {type(texts).__name__}")
            
        if not callable(tokenizer):
            raise TypeError(f"Tokenizer must be callable, got {type(tokenizer).__name__}")
            
        if min_freq < 1:
            raise ValueError(f"min_freq must be at least 1, got {min_freq}")
            
        if max_vocab_size is not None and max_vocab_size < 1:
            raise ValueError(f"max_vocab_size must be at least 1, got {max_vocab_size}")
            
        if not texts:
            logger.warning("Empty texts list provided to build_from_texts")
            return cls(**kwargs)
        
        # Count token frequencies
        counter = Counter()
        for text in texts:
            if not isinstance(text, str):
                raise TypeError(f"All elements in texts must be strings, got {type(text).__name__}")
                
            tokens = tokenizer(text)
            if not isinstance(tokens, list):
                raise TypeError(f"Tokenizer must return a list, got {type(tokens).__name__}")
                
            counter.update(tokens)
        
        # Filter by frequency and limit vocabulary size
        filtered_tokens = [
            token for token, count in counter.most_common(max_vocab_size)
            if count >= min_freq
        ]
        
        # Log vocabulary statistics
        total_unique = len(counter)
        included = len(filtered_tokens)
        logger.info(f"Built vocabulary with {included} tokens (from {total_unique} unique tokens)")
        
        # Create vocabulary
        return cls(tokens=filtered_tokens, **kwargs)
    
    def __len__(self) -> int:
        """Get the size of the vocabulary."""
        return len(self.idx_to_token)
    
    def __contains__(self, token: str) -> bool:
        """Check if a token is in the vocabulary."""
        return token in self.token_to_idx
    
    @property
    def special_token_indices(self) -> Dict[str, int]:
        """
        Get a dictionary mapping special token names to their indices.
        
        Returns:
            Dictionary of special token indices
        """
        return {
            "pad_token_idx": self.token_to_idx[self.pad_token],
            "unk_token_idx": self.token_to_idx[self.unk_token],
            "bos_token_idx": self.token_to_idx[self.bos_token],
            "eos_token_idx": self.token_to_idx[self.eos_token],
            "mask_token_idx": self.token_to_idx[self.mask_token],
        }