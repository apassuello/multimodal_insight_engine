# src/data/tokenization/vocabulary.py
from typing import Dict, List, Optional
from collections import Counter
import json
import os

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
        """
        if token not in self.token_to_idx:
            idx = len(self.idx_to_token)
            self.token_to_idx[token] = idx
            self.idx_to_token.append(token)
            return idx
        return self.token_to_idx[token]
    
    def token_to_index(self, token: str) -> int:
        """
        Convert a token to its index.
        
        Args:
            token: The token to look up
            
        Returns:
            The index of the token, or the unknown token index if not found
        """
        return self.token_to_idx.get(token, self.token_to_idx[self.unk_token])
    
    def index_to_token(self, idx: int) -> str:
        """
        Convert an index to its token.
        
        Args:
            idx: The index to look up
            
        Returns:
            The token at the given index, or the unknown token if index is out of range
        """
        if 0 <= idx < len(self.idx_to_token):
            return self.idx_to_token[idx]
        return self.unk_token
    
    def tokens_to_indices(self, tokens: List[str]) -> List[int]:
        """
        Convert a list of tokens to their indices.
        
        Args:
            tokens: List of tokens to convert
            
        Returns:
            List of corresponding indices
        """
        return [self.token_to_index(token) for token in tokens]
    
    def indices_to_tokens(self, indices: List[int]) -> List[str]:
        """
        Convert a list of indices to their tokens.
        
        Args:
            indices: List of indices to convert
            
        Returns:
            List of corresponding tokens
        """
        return [self.index_to_token(idx) for idx in indices]
    
    def save(self, path: str) -> None:
        """
        Save the vocabulary to a file.
        
        Args:
            path: Path to save the vocabulary
        """
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
            }
        }
        
        # Save to file
        with open(path, "w", encoding="utf-8") as f:
            json.dump(vocab_data, f, ensure_ascii=False, indent=2)
    
    @classmethod
    def load(cls, path: str) -> "Vocabulary":
        """
        Load a vocabulary from a file.
        
        Args:
            path: Path to load the vocabulary from
            
        Returns:
            Loaded Vocabulary instance
        """
        with open(path, "r", encoding="utf-8") as f:
            vocab_data = json.load(f)
        
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
        vocab.token_to_idx = vocab_data["token_to_idx"]
        vocab.idx_to_token = [""] * len(vocab.token_to_idx)
        for token, idx in vocab.token_to_idx.items():
            vocab.idx_to_token[idx] = token
        
        return vocab
    
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
        Build a vocabulary from a list of texts.
        
        Args:
            texts: List of texts to build vocabulary from
            tokenizer: Tokenizer function to use
            max_vocab_size: Maximum vocabulary size (None means no limit)
            min_freq: Minimum frequency for a token to be included
            **kwargs: Additional arguments to pass to the Vocabulary constructor
            
        Returns:
            Built Vocabulary instance
        """
        # Count token frequencies
        counter = Counter()
        for text in texts:
            tokens = tokenizer(text)
            counter.update(tokens)
        
        # Filter by frequency and limit vocabulary size
        filtered_tokens = [
            token for token, count in counter.most_common(max_vocab_size)
            if count >= min_freq
        ]
        
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