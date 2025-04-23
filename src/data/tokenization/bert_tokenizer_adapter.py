"""
BERT tokenizer adapter for the multimodal dataset.

This module provides a compatibility wrapper around the HuggingFace BertTokenizer
to make it compatible with the interface expected by the multimodal dataset class.
"""

from transformers import BertTokenizer, PreTrainedTokenizer
from typing import List, Dict, Optional, Union, Any


class BertTokenizerAdapter:
    """
    Adapter for HuggingFace's BertTokenizer to make it compatible with the
    multimodal dataset interface.
    """
    
    def __init__(self, pretrained_model_name: str = "bert-base-uncased", max_length: int = 77):
        """
        Initialize the adapter.
        
        Args:
            pretrained_model_name: Name of the pretrained BERT model
            max_length: Maximum sequence length
        """
        self.max_length = max_length
        self.tokenizer = BertTokenizer.from_pretrained(pretrained_model_name)
        
        # Create special_tokens attribute for compatibility
        self._special_tokens = {
            "pad_token_idx": self.tokenizer.pad_token_id,
            "unk_token_idx": self.tokenizer.unk_token_id,
            "bos_token_idx": self.tokenizer.cls_token_id,  # Use CLS as BOS
            "eos_token_idx": self.tokenizer.sep_token_id,  # Use SEP as EOS
            "mask_token_idx": self.tokenizer.mask_token_id,
        }
        
        print(f"Created BertTokenizerAdapter with max_length={max_length}")
        print(f"Special tokens: {self._special_tokens}")
    
    def encode(self, text: str) -> List[int]:
        """
        Encode text into token IDs.
        
        Args:
            text: Input text
            
        Returns:
            List of token IDs
        """
        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        
        # Return the input_ids as a list
        return encoding["input_ids"][0].tolist()
    
    def decode(self, token_ids: List[int]) -> str:
        """
        Decode token IDs back to text.
        
        Args:
            token_ids: List of token IDs
            
        Returns:
            Decoded text
        """
        return self.tokenizer.decode(token_ids, skip_special_tokens=True)
    
    def tokenize(self, text: str) -> List[str]:
        """
        Convert text to tokens without IDs.
        
        Args:
            text: Input text
            
        Returns:
            List of tokens
        """
        return self.tokenizer.tokenize(text)
    
    def batch_encode(self, texts: List[str]) -> List[List[int]]:
        """
        Encode multiple texts.
        
        Args:
            texts: List of input texts
            
        Returns:
            List of token ID lists
        """
        encodings = self.tokenizer(
            texts,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        
        # Return the input_ids as a list of lists
        return encodings["input_ids"].tolist()
    
    @property
    def special_tokens(self) -> Dict[str, int]:
        """
        Get special token mapping.
        
        Returns:
            Dictionary of special tokens to their IDs
        """
        return self._special_tokens
    
    @property
    def vocab_size(self) -> int:
        """
        Get vocabulary size.
        
        Returns:
            Size of vocabulary
        """
        return len(self.tokenizer)