# src/data/tokenization/simple_tokenizer.py
from typing import List, Dict, Optional

from .base_tokenizer import BaseTokenizer
from .vocabulary import Vocabulary
from .preprocessing import clean_text, segment_on_punc

class WhitespaceTokenizer(BaseTokenizer):
    """
    A simple tokenizer that splits text on whitespace.
    
    This tokenizer provides a baseline implementation that splits text on
    whitespace after optional preprocessing.
    """
    
    def __init__(
        self,
        vocab: Optional[Vocabulary] = None,
        split_on_punct: bool = True,
        lower_case: bool = True,
    ):
        """
        Initialize the whitespace tokenizer.
        
        Args:
            vocab: Optional vocabulary to use
            split_on_punct: Whether to add spaces around punctuation
            lower_case: Whether to convert text to lowercase
        """
        self.split_on_punct = split_on_punct
        self.lower_case = lower_case
        
        # Create or use provided vocabulary
        if vocab is None:
            self.vocab = Vocabulary()
        else:
            self.vocab = vocab
    
    def preprocess(self, text: str) -> str:
        """
        Preprocess text before tokenization.
        
        Args:
            text: Input text
            
        Returns:
            Preprocessed text
        """
        # Clean text
        text = clean_text(text, lower=self.lower_case)
        
        # Segment on punctuation if requested
        if self.split_on_punct:
            text = segment_on_punc(text)
        
        return text
    
    def tokenize(self, text: str) -> List[str]:
        """
        Convert a text string into a list of tokens.
        
        Args:
            text: The input text to tokenize
            
        Returns:
            A list of tokens
        """
        # Preprocess text
        text = self.preprocess(text)
        
        # Split on whitespace
        tokens = text.split()
        
        return tokens
    
    def encode(self, text: str) -> List[int]:
        """
        Convert a text string into a list of token indices.
        
        Args:
            text: The input text to encode
            
        Returns:
            A list of token indices
        """
        tokens = self.tokenize(text)
        return self.vocab.tokens_to_indices(tokens)
    
    def decode(self, token_ids: List[int]) -> str:
        """
        Convert a list of token indices back into a text string.
        
        Args:
            token_ids: The token indices to decode
            
        Returns:
            The reconstructed text
        """
        tokens = self.vocab.indices_to_tokens(token_ids)
        return ' '.join(tokens)
    
    def batch_encode(self, texts: List[str]) -> List[List[int]]:
        """
        Encode a batch of texts into token indices.
        
        Args:
            texts: List of input texts to encode
            
        Returns:
            List of token index lists
        """
        return [self.encode(text) for text in texts]
    
    @property
    def vocab_size(self) -> int:
        """
        Get the vocabulary size of this tokenizer.
        
        Returns:
            The number of tokens in the vocabulary
        """
        return len(self.vocab)
    
    @property
    def special_tokens(self) -> Dict[str, int]:
        """
        Get the special tokens used by this tokenizer.
        
        Returns:
            Dictionary mapping special token names to their indices
        """
        return self.vocab.special_token_indices
    
    def add_tokens(self, tokens: List[str]) -> int:
        """
        Add tokens to the vocabulary.
        
        Args:
            tokens: List of tokens to add
            
        Returns:
            Number of tokens added
        """
        added = 0
        for token in tokens:
            idx = self.vocab.add_token(token)
            if idx == len(self.vocab) - 1:  # Token was added at the end
                added += 1
        return added
    
    def save_pretrained(self, path: str) -> None:
        """
        Save the tokenizer vocabulary to a directory.
        
        Args:
            path: Directory path to save to
        """
        self.vocab.save(f"{path}/vocab.json")
    
    @classmethod
    def from_pretrained(cls, path: str, **kwargs) -> "WhitespaceTokenizer":
        """
        Load a tokenizer from a saved vocabulary.
        
        Args:
            path: Directory path to load from
            **kwargs: Additional arguments to pass to the constructor
            
        Returns:
            Loaded tokenizer
        """
        vocab = Vocabulary.load(f"{path}/vocab.json")
        return cls(vocab=vocab, **kwargs)