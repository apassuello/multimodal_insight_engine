# src/data/tokenization/preprocessing.py
"""
Text Preprocessing Utilities for Tokenization

PURPOSE:
    Provides utilities for cleaning and normalizing text before tokenization,
    ensuring consistent input for tokenizers across different data sources.

KEY COMPONENTS:
    - Unicode normalization
    - Text cleaning with configurable options
    - Punctuation segmentation for better tokenization
    - HTML handling and contraction normalization
"""

import re
import unicodedata
import html
import os
from typing import List

def normalize_unicode(text: str) -> str:
    """
    Normalize Unicode characters in text.
    
    Args:
        text: Input text
    
    Returns:
        Normalized text
    """
    # Normalize to NFKC form (compatibility decomposition followed by canonical composition)
    return unicodedata.normalize('NFKC', text)

def clean_text(
    text: str,
    lower: bool = True,
    remove_accents: bool = False,
    strip_html: bool = True,
    handle_contractions: bool = True,
) -> str:
    """
    Clean and normalize text with configurable options.
    
    Args:
        text: Input text
        lower: Whether to convert to lowercase
        remove_accents: Whether to remove accents
        strip_html: Whether to remove HTML tags and entities
        handle_contractions: Whether to standardize contractions
    
    Returns:
        Cleaned text
    """
    # Handle None or empty text
    if not text:
        return ""
    
    # Convert to unicode and normalize
    text = normalize_unicode(text)
    
    # Strip HTML
    if strip_html:
        text = html.unescape(text)  # Convert HTML entities
        text = re.sub(r'<[^>]+>', '', text)  # Remove HTML tags
    
    # Convert to lowercase
    if lower:
        text = text.lower()
    
    # Remove accents
    if remove_accents:
        text = ''.join(
            c for c in unicodedata.normalize('NFD', text)
            if not unicodedata.combining(c)
        )
    
    # Handle contractions
    if handle_contractions:
        # This is a simplified approach; a more comprehensive solution would use a dictionary
        text = re.sub(r"can't", "cannot", text)
        text = re.sub(r"n't", " not", text)
        text = re.sub(r"'ll", " will", text)
        text = re.sub(r"'re", " are", text)
        text = re.sub(r"'ve", " have", text)
        text = re.sub(r"'m", " am", text)
        text = re.sub(r"'d", " would", text)
    
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

def segment_on_punc(text: str) -> str:
    """
    Add spaces around punctuation.
    
    Args:
        text: Input text
    
    Returns:
        Text with spaces around punctuation
    """
    # Add spaces around punctuation
    # This helps in later tokenization by splitting on whitespace
    text = re.sub(r'([.,!?;:])', r' \1 ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


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
        "module_purpose": "Provides text preprocessing utilities for tokenization including Unicode normalization and text cleaning",
        "key_classes": [],
        "key_functions": [
            {
                "name": "normalize_unicode",
                "signature": "normalize_unicode(text: str) -> str",
                "brief_description": "Normalize Unicode characters in text using NFKC form"
            },
            {
                "name": "clean_text",
                "signature": "clean_text(text: str, lower: bool = True, remove_accents: bool = False, strip_html: bool = True, handle_contractions: bool = True) -> str",
                "brief_description": "Clean and normalize text with configurable options for case, accents, HTML, and contractions"
            },
            {
                "name": "segment_on_punc",
                "signature": "segment_on_punc(text: str) -> str",
                "brief_description": "Add spaces around punctuation to help with tokenization"
            }
        ],
        "external_dependencies": ["re", "unicodedata", "html"],
        "complexity_score": 3  # Moderate complexity for text processing
    }