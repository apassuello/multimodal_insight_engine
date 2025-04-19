# src/data/tokenization/simple_tokenizer.py
from typing import List, Dict, Optional, Union
import os
import json

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


class SimpleTokenizer:
    """
    A HuggingFace-compatible tokenizer adapter for use with multimodal models.
    
    This tokenizer provides a simple wrapper around HuggingFace tokenizers,
    with fallback to WhitespaceTokenizer when a pre-trained model is not specified.
    """
    
    def __init__(
        self,
        pretrained_model_name: Optional[str] = None, 
        max_length: int = 77,
        add_special_tokens: bool = True
    ):
        """
        Initialize the tokenizer adapter.
        
        Args:
            pretrained_model_name: HuggingFace model name to use (optional)
            max_length: Maximum sequence length for tokenization
            add_special_tokens: Whether to add special tokens like [CLS] and [SEP]
        """
        self.max_length = max_length
        self.add_special_tokens = add_special_tokens
        self.hf_tokenizer = None
        self.pretrained_model_name = pretrained_model_name
        
        # Dictionary to store special token indices
        self._special_tokens = {
            "pad_token_idx": 0,
            "unk_token_idx": 1,
            "bos_token_idx": 2,
            "eos_token_idx": 3,
            "mask_token_idx": 4,
        }
        
        # Try to load HuggingFace tokenizer if name provided
        if pretrained_model_name:
            try:
                from transformers import AutoTokenizer
                
                # Determine the right tokenizer based on model name
                if 'mobilebert' in pretrained_model_name.lower():
                    from transformers import MobileBertTokenizer
                    self.hf_tokenizer = MobileBertTokenizer.from_pretrained(pretrained_model_name)
                elif 'albert' in pretrained_model_name.lower():
                    from transformers import AlbertTokenizer
                    self.hf_tokenizer = AlbertTokenizer.from_pretrained(pretrained_model_name)
                elif 'bert' in pretrained_model_name.lower() and 'distil' not in pretrained_model_name.lower():
                    from transformers import BertTokenizer
                    self.hf_tokenizer = BertTokenizer.from_pretrained(pretrained_model_name)
                elif 'roberta' in pretrained_model_name.lower():
                    from transformers import RobertaTokenizer
                    self.hf_tokenizer = RobertaTokenizer.from_pretrained(pretrained_model_name)
                elif 'distilbert' in pretrained_model_name.lower():
                    from transformers import DistilBertTokenizer
                    self.hf_tokenizer = DistilBertTokenizer.from_pretrained(pretrained_model_name)
                else:
                    # Default to AutoTokenizer
                    self.hf_tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name)
                
                # Update special token indices using the HF tokenizer's dictionary
                if hasattr(self.hf_tokenizer, 'pad_token_id') and self.hf_tokenizer.pad_token_id is not None:
                    self._special_tokens["pad_token_idx"] = self.hf_tokenizer.pad_token_id
                if hasattr(self.hf_tokenizer, 'unk_token_id') and self.hf_tokenizer.unk_token_id is not None:
                    self._special_tokens["unk_token_idx"] = self.hf_tokenizer.unk_token_id
                if hasattr(self.hf_tokenizer, 'bos_token_id') and self.hf_tokenizer.bos_token_id is not None:
                    self._special_tokens["bos_token_idx"] = self.hf_tokenizer.bos_token_id
                if hasattr(self.hf_tokenizer, 'eos_token_id') and self.hf_tokenizer.eos_token_id is not None:
                    self._special_tokens["eos_token_idx"] = self.hf_tokenizer.eos_token_id
                if hasattr(self.hf_tokenizer, 'mask_token_id') and self.hf_tokenizer.mask_token_id is not None:
                    self._special_tokens["mask_token_idx"] = self.hf_tokenizer.mask_token_id
                
                print(f"Loaded HuggingFace tokenizer for {pretrained_model_name}")
                
            except (ImportError, ModuleNotFoundError) as e:
                print(f"Warning: Unable to import transformers library: {e}")
                print("Falling back to WhitespaceTokenizer")
                self.hf_tokenizer = None
            except Exception as e:
                print(f"Error loading HuggingFace tokenizer: {e}")
                print("Falling back to WhitespaceTokenizer")
                self.hf_tokenizer = None
        
        # If HuggingFace tokenizer is not available, use WhitespaceTokenizer
        if self.hf_tokenizer is None:
            # Create vocabulary with special tokens
            vocab = Vocabulary(
                pad_token="<PAD>",
                unk_token="<UNK>",
                bos_token="<BOS>",
                eos_token="<EOS>",
                mask_token="<MASK>"
            )
            
            # Create tokenizer
            self.basic_tokenizer = WhitespaceTokenizer(vocab=vocab)
            print("Using WhitespaceTokenizer as fallback")
    
    def encode(self, text: str) -> List[int]:
        """
        Convert text to token indices.
        
        Args:
            text: Input text to encode
            
        Returns:
            List of token indices
        """
        if self.hf_tokenizer:
            # Use HuggingFace tokenizer if available
            encoding = self.hf_tokenizer(
                text,
                max_length=self.max_length,
                padding="max_length",
                truncation=True,
                return_tensors="pt"
            )
            
            # Return the input_ids as a list
            return encoding["input_ids"][0].tolist()
        else:
            # Use basic tokenizer
            token_ids = self.basic_tokenizer.encode(text)
            
            # Truncate or pad to max_length
            if len(token_ids) > self.max_length:
                token_ids = token_ids[:self.max_length]
            else:
                pad_token = self.special_tokens["pad_token_idx"]
                token_ids = token_ids + [pad_token] * (self.max_length - len(token_ids))
            
            return token_ids
    
    def decode(self, token_ids: List[int]) -> str:
        """
        Convert token indices back to text.
        
        Args:
            token_ids: List of token indices
            
        Returns:
            Decoded text
        """
        if self.hf_tokenizer:
            # Use HuggingFace tokenizer if available
            return self.hf_tokenizer.decode(token_ids, skip_special_tokens=True)
        else:
            # Use basic tokenizer
            return self.basic_tokenizer.decode(token_ids)
    
    def tokenize(self, text: str) -> List[str]:
        """
        Split text into tokens without converting to indices.
        
        Args:
            text: Input text to tokenize
            
        Returns:
            List of tokens
        """
        if self.hf_tokenizer:
            # Use HuggingFace tokenizer if available
            return self.hf_tokenizer.tokenize(text)
        else:
            # Use basic tokenizer
            return self.basic_tokenizer.tokenize(text)
    
    def batch_encode(self, texts: List[str]) -> List[List[int]]:
        """
        Encode a batch of texts.
        
        Args:
            texts: List of input texts
            
        Returns:
            List of token index lists
        """
        if self.hf_tokenizer:
            # Use HuggingFace tokenizer if available
            encodings = self.hf_tokenizer(
                texts,
                max_length=self.max_length,
                padding="max_length",
                truncation=True,
                return_tensors="pt"
            )
            
            # Return the input_ids as a list of lists
            return encodings["input_ids"].tolist()
        else:
            # Use basic tokenizer
            return self.basic_tokenizer.batch_encode(texts)
    
    @property
    def special_tokens(self) -> Dict[str, int]:
        """
        Get the special tokens used by this tokenizer.
        
        Returns:
            Dictionary mapping special token names to their indices
        """
        return self._special_tokens
    
    @property
    def vocab_size(self) -> int:
        """
        Get the vocabulary size.
        
        Returns:
            Size of vocabulary
        """
        if self.hf_tokenizer:
            return len(self.hf_tokenizer)
        else:
            return self.basic_tokenizer.vocab_size