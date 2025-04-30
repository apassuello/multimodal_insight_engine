"""MODULE: bert_tokenizer_adapter.py
PURPOSE: Provides an adapter class to make HuggingFace BERT tokenizers compatible with the project's tokenizer interface.

KEY COMPONENTS:
- BertTokenizerAdapter: Adapter class wrapping HuggingFace BERT tokenizers
- Standardized interface for BERT tokenization
- Support for batch processing and special token handling
- Memory-efficient tokenization with caching

DEPENDENCIES:
- transformers
- torch
- typing
"""

import os
from transformers import BertTokenizer, PreTrainedTokenizer
from typing import List, Dict, Optional, Union, Any
from .base_tokenizer import BaseTokenizer


class BertTokenizerAdapter:
    """
    Adapter for HuggingFace's BertTokenizer to make it compatible with the
    multimodal dataset interface.
    """

    def __init__(
        self, pretrained_model_name: str = "bert-base-uncased", max_length: int = 77
    ):
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
            return_tensors="pt",
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
            return_tensors="pt",
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
        "module_purpose": "Provides an adapter class to make HuggingFace BERT tokenizers compatible with the project's tokenizer interface",
        "key_classes": [
            {
                "name": "BertTokenizerAdapter",
                "purpose": "Adapter class that wraps HuggingFace BERT tokenizers to match project interface",
                "key_methods": [
                    {
                        "name": "__init__",
                        "signature": "__init__(self, pretrained_model_name_or_path: str)",
                        "brief_description": "Initialize adapter with a pretrained BERT tokenizer",
                    },
                    {
                        "name": "tokenize",
                        "signature": "tokenize(self, text: str) -> List[str]",
                        "brief_description": "Convert text into BERT tokens",
                    },
                    {
                        "name": "encode",
                        "signature": "encode(self, text: str) -> List[int]",
                        "brief_description": "Convert text to BERT token indices",
                    },
                ],
                "inheritance": "BaseTokenizer",
                "dependencies": ["transformers.BertTokenizer"],
            }
        ],
        "external_dependencies": ["transformers", "torch", "typing"],
        "complexity_score": 4,
    }
