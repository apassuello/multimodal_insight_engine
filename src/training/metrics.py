"""MODULE: metrics.py
PURPOSE: Implements common training metrics for model evaluation, including accuracy, perplexity, and custom metrics for specific tasks.

KEY COMPONENTS:
- Accuracy: Computes classification accuracy with support for top-k accuracy
- Perplexity: Computes perplexity for language models
- F1Score: Computes F1 score for classification tasks
- BLEUScore: Computes BLEU score for machine translation
- CustomMetric: Base class for implementing custom metrics

DEPENDENCIES:
- PyTorch (torch)
- NumPy
- NLTK (for BLEU score)

SPECIAL NOTES:
- All metrics support both CPU and GPU computation
- Includes support for batch-wise and epoch-wise metric computation
- Provides flexible reduction options for different evaluation scenarios
"""

import torch
import numpy as np
from typing import Optional, Union, List, Dict, Any
from nltk.translate.bleu_score import corpus_bleu
import nltk
import os


class Accuracy:
    """
    Computes classification accuracy with support for top-k accuracy.
    
    This metric can be used for both binary and multi-class classification tasks.
    It supports computing top-k accuracy where the prediction is considered correct
    if the true class is among the k most probable classes.
    
    Args:
        top_k (int, optional): Number of top predictions to consider. Defaults to 1.
        reduction (str, optional): Specifies the reduction to apply to the output:
            'none' | 'mean' | 'sum'. Defaults to 'mean'.
    """
    
    def __init__(self, top_k: int = 1, reduction: str = 'mean'):
        self.top_k = top_k
        self.reduction = reduction
        self.reset()
    
    def reset(self):
        """Reset the metric state."""
        self.correct = 0
        self.total = 0
    
    def update(self, pred: torch.Tensor, target: torch.Tensor):
        """
        Update the metric with new predictions and targets.
        
        Args:
            pred: Predicted logits of shape (N, C) where C is the number of classes
            target: Target indices of shape (N,) where values are 0 ≤ targets[i] ≤ C-1
        """
        if self.top_k == 1:
            pred = pred.argmax(dim=-1)
            self.correct += (pred == target).sum().item()
        else:
            _, pred = torch.topk(pred, k=self.top_k, dim=-1)
            self.correct += sum(1 for p, t in zip(pred, target) if t in p)
        self.total += target.size(0)
    
    def compute(self) -> float:
        """
        Compute the current accuracy value.
        
        Returns:
            float: The computed accuracy value
        """
        if self.total == 0:
            return 0.0
        return self.correct / self.total


class Perplexity:
    """
    Computes perplexity for language models.
    
    Perplexity is a measure of how well a probability model predicts a sample.
    A lower perplexity indicates better performance. This implementation supports
    both token-level and sequence-level perplexity computation.
    
    Args:
        reduction (str, optional): Specifies the reduction to apply to the output:
            'none' | 'mean' | 'sum'. Defaults to 'mean'.
    """
    
    def __init__(self, reduction: str = 'mean'):
        self.reduction = reduction
        self.reset()
    
    def reset(self):
        """Reset the metric state."""
        self.total_loss = 0.0
        self.total_tokens = 0
    
    def update(self, loss: torch.Tensor, num_tokens: int):
        """
        Update the metric with new loss values and token counts.
        
        Args:
            loss: Cross-entropy loss value
            num_tokens: Number of tokens in the current batch
        """
        self.total_loss += loss.item() * num_tokens
        self.total_tokens += num_tokens
    
    def compute(self) -> float:
        """
        Compute the current perplexity value.
        
        Returns:
            float: The computed perplexity value
        """
        if self.total_tokens == 0:
            return float('inf')
        return np.exp(self.total_loss / self.total_tokens)


class F1Score:
    """
    Computes F1 score for classification tasks.
    
    The F1 score is the harmonic mean of precision and recall. This implementation
    supports both binary and multi-class classification with macro and micro averaging.
    
    Args:
        num_classes (int): Number of classes
        average (str, optional): Averaging method: 'macro' | 'micro' | 'weighted'.
            Defaults to 'macro'.
    """
    
    def __init__(self, num_classes: int, average: str = 'macro'):
        self.num_classes = num_classes
        self.average = average
        self.reset()
    
    def reset(self):
        """Reset the metric state."""
        self.tp = torch.zeros(self.num_classes)
        self.fp = torch.zeros(self.num_classes)
        self.fn = torch.zeros(self.num_classes)
    
    def update(self, pred: torch.Tensor, target: torch.Tensor):
        """
        Update the metric with new predictions and targets.
        
        Args:
            pred: Predicted class indices
            target: Target class indices
        """
        for c in range(self.num_classes):
            self.tp[c] += ((pred == c) & (target == c)).sum().item()
            self.fp[c] += ((pred == c) & (target != c)).sum().item()
            self.fn[c] += ((pred != c) & (target == c)).sum().item()
    
    def compute(self) -> float:
        """
        Compute the current F1 score.
        
        Returns:
            float: The computed F1 score
        """
        if self.average == 'macro':
            precision = self.tp / (self.tp + self.fp + 1e-10)
            recall = self.tp / (self.tp + self.fn + 1e-10)
            f1 = 2 * (precision * recall) / (precision + recall + 1e-10)
            return f1.mean().item()
        elif self.average == 'micro':
            total_tp = self.tp.sum()
            total_fp = self.fp.sum()
            total_fn = self.fn.sum()
            precision = total_tp / (total_tp + total_fp + 1e-10)
            recall = total_tp / (total_tp + total_fn + 1e-10)
            return float(2 * (precision * recall) / (precision + recall + 1e-10))
        else:  # weighted
            weights = self.tp + self.fn
            precision = self.tp / (self.tp + self.fp + 1e-10)
            recall = self.tp / (self.tp + self.fn + 1e-10)
            f1 = 2 * (precision * recall) / (precision + recall + 1e-10)
            return float((f1 * weights).sum().item() / weights.sum().item())


class BLEUScore:
    """
    Computes BLEU score for machine translation.
    
    BLEU (Bilingual Evaluation Understudy) is an algorithm for evaluating the quality
    of machine-translated text. This implementation uses NLTK's BLEU implementation
    with support for different n-gram weights.
    
    Args:
        weights (tuple, optional): Weights for different n-grams. Defaults to (0.25, 0.25, 0.25, 0.25).
    """
    
    def __init__(self, weights: tuple = (0.25, 0.25, 0.25, 0.25)):
        self.weights = weights
        self.reset()
    
    def reset(self):
        """Reset the metric state."""
        self.references = []
        self.hypotheses = []
    
    def update(self, hypothesis: str, reference: str):
        """
        Update the metric with new hypothesis and reference.
        
        Args:
            hypothesis: Generated translation
            reference: Ground truth translation
        """
        self.hypotheses.append(hypothesis.split())
        self.references.append([reference.split()])
    
    def compute(self) -> float:
        """
        Compute the current BLEU score.
        
        Returns:
            float: The computed BLEU score
        """
        if not self.hypotheses or not self.references:
            return 0.0
        score: float = corpus_bleu(self.references, self.hypotheses, weights=self.weights)  # type: ignore
        return score if score is not None else 0.0


class CustomMetric:
    """
    Base class for implementing custom metrics.
    
    This class provides a template for implementing custom metrics with a consistent
    interface. It includes methods for resetting the metric state, updating with new
    values, and computing the final metric value.
    """
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        """Reset the metric state. Override this method in subclasses."""
        raise NotImplementedError
    
    def update(self, *args, **kwargs):
        """
        Update the metric with new values. Override this method in subclasses.
        
        Args:
            *args: Positional arguments for the update
            **kwargs: Keyword arguments for the update
        """
        raise NotImplementedError
    
    def compute(self) -> Any:
        """
        Compute the current metric value. Override this method in subclasses.
        
        Returns:
            Any: The computed metric value
        """
        raise NotImplementedError


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
        "module_purpose": "Implements common training metrics for model evaluation with support for various tasks",
        "key_classes": [
            {
                "name": "Accuracy",
                "purpose": "Computes classification accuracy with support for top-k accuracy",
                "key_methods": [
                    {
                        "name": "update",
                        "signature": "update(self, pred: torch.Tensor, target: torch.Tensor)",
                        "brief_description": "Updates the accuracy metric with new predictions and targets"
                    },
                    {
                        "name": "compute",
                        "signature": "compute(self) -> float",
                        "brief_description": "Computes the current accuracy value"
                    }
                ],
                "inheritance": "object",
                "dependencies": ["torch"]
            },
            {
                "name": "Perplexity",
                "purpose": "Computes perplexity for language models",
                "key_methods": [
                    {
                        "name": "update",
                        "signature": "update(self, loss: torch.Tensor, num_tokens: int)",
                        "brief_description": "Updates the perplexity metric with new loss values"
                    },
                    {
                        "name": "compute",
                        "signature": "compute(self) -> float",
                        "brief_description": "Computes the current perplexity value"
                    }
                ],
                "inheritance": "object",
                "dependencies": ["torch", "numpy"]
            },
            {
                "name": "F1Score",
                "purpose": "Computes F1 score for classification tasks",
                "key_methods": [
                    {
                        "name": "update",
                        "signature": "update(self, pred: torch.Tensor, target: torch.Tensor)",
                        "brief_description": "Updates the F1 score metric with new predictions"
                    },
                    {
                        "name": "compute",
                        "signature": "compute(self) -> float",
                        "brief_description": "Computes the current F1 score"
                    }
                ],
                "inheritance": "object",
                "dependencies": ["torch"]
            },
            {
                "name": "BLEUScore",
                "purpose": "Computes BLEU score for machine translation",
                "key_methods": [
                    {
                        "name": "update",
                        "signature": "update(self, hypothesis: str, reference: str)",
                        "brief_description": "Updates the BLEU score metric with new translations"
                    },
                    {
                        "name": "compute",
                        "signature": "compute(self) -> float",
                        "brief_description": "Computes the current BLEU score"
                    }
                ],
                "inheritance": "object",
                "dependencies": ["nltk"]
            }
        ],
        "external_dependencies": ["torch", "numpy", "nltk"],
        "complexity_score": 7,  # Medium-high complexity due to multiple metric implementations and their interactions
    }
