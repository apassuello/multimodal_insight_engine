import numpy as np
from typing import List, Dict, Tuple
from collections import Counter
import re
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
import nltk
from nltk.metrics.distance import edit_distance
import os

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

def calculate_bleu(hypotheses: List[str], references: List[str], weights: Tuple[float, ...] = (0.4, 0.3, 0.2, 0.1)) -> float:
    """
    Calculate BLEU score using NLTK's implementation.
    
    Args:
        hypotheses: List of generated translations
        references: List of reference translations
        weights: Weights for n-gram precision (default: (0.4, 0.3, 0.2, 0.1))
        
    Returns:
        BLEU score
    """
    # Tokenize hypotheses and references
    hyp_tokens = [nltk.word_tokenize(hyp.lower()) for hyp in hypotheses]
    ref_tokens = [nltk.word_tokenize(ref.lower()) for ref in references]
    
    # Calculate BLEU score with smoothing
    smoothing = SmoothingFunction().method1
    scores = []
    
    for hyp, ref in zip(hyp_tokens, ref_tokens):
        score = sentence_bleu([ref], hyp, weights=weights, smoothing_function=smoothing)
        scores.append(score)
    
    return float(np.mean(scores))

def calculate_ter(hypotheses: List[str], references: List[str]) -> float:
    """
    Calculate Translation Edit Rate (TER).
    
    Args:
        hypotheses: List of generated translations
        references: List of reference translations
        
    Returns:
        Average TER score (lower is better)
    """
    scores = []
    
    for hyp, ref in zip(hypotheses, references):
        # Tokenize
        hyp_tokens = nltk.word_tokenize(hyp.lower())
        ref_tokens = nltk.word_tokenize(ref.lower())
        
        # Calculate edit distance
        distance = edit_distance(hyp_tokens, ref_tokens)
        
        # Normalize by reference length
        score = distance / len(ref_tokens)
        scores.append(score)
    
    return float(np.mean(scores))

def evaluate_translation(hypotheses: List[str], references: List[str]) -> Dict[str, float]:
    """
    Evaluate translation quality using multiple metrics.
    
    Args:
        hypotheses: List of generated translations
        references: List of reference translations
        
    Returns:
        Dictionary containing BLEU and TER scores
    """
    bleu = calculate_bleu(hypotheses, references)
    ter = calculate_ter(hypotheses, references)
    
    return {
        "bleu": bleu,
        "ter": ter
    }

def print_evaluation_results(scores: Dict[str, float]):
    """
    Print evaluation results in a formatted way.
    
    Args:
        scores: Dictionary containing metric scores
    """
    print("\n=== Translation Evaluation Results ===")
    print(f"BLEU Score: {scores['bleu']:.4f}")
    print(f"TER Score:  {scores['ter']:.4f}")
    print("=" * 35)

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
        "module_purpose": "Implements standard evaluation metrics for machine translation tasks including BLEU and TER scoring",
        "key_functions": [
            {
                "name": "calculate_bleu",
                "signature": "calculate_bleu(hypotheses: List[str], references: List[str], weights: Tuple[float, ...] = (0.4, 0.3, 0.2, 0.1)) -> float",
                "brief_description": "Calculates BLEU score for translation quality using NLTK's implementation with smoothing"
            },
            {
                "name": "calculate_ter",
                "signature": "calculate_ter(hypotheses: List[str], references: List[str]) -> float",
                "brief_description": "Calculates Translation Edit Rate (TER) to measure edit distance between translations"
            },
            {
                "name": "evaluate_translation",
                "signature": "evaluate_translation(hypotheses: List[str], references: List[str]) -> Dict[str, float]",
                "brief_description": "Evaluates translation quality using multiple metrics and returns consolidated results"
            },
            {
                "name": "print_evaluation_results",
                "signature": "print_evaluation_results(scores: Dict[str, float])",
                "brief_description": "Formats and prints evaluation results in a readable format"
            }
        ],
        "external_dependencies": ["numpy", "nltk", "re"],
        "complexity_score": 3  # Moderate-low complexity with straightforward implementation of standard metrics
    } 