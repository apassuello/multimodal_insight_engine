# src/data/__init__.py
"""
Data modules for the MultiModal Insight Engine.

This package contains components for loading, preprocessing, and 
managing data for various model architectures.
"""

from .combined_dataset import CombinedDataset

# Import constitutional AI dataset utilities
from .constitutional_dataset import (
    CHAT_PROMPT_TEMPLATE,
    CONSTITUTIONAL_CRITIQUE_TEMPLATE,
    DEFAULT_PROMPT_TEMPLATE,
    INSTRUCTION_PROMPT_TEMPLATE,
    ConstitutionalTrainingDataset,
    PromptDataset,
    PromptResponseDataset,
    PromptTemplate,
    create_default_prompts,
    load_huggingface_dataset,
    save_prompts_to_file,
)
from .dataloader import create_dataloader

# Import dataset classes
from .language_modeling import LanguageModelingDataset

# Import multimodal data utilities
from .multimodal_data_utils import create_data_loaders, randomize_dataset_positions
from .sequence_data import (
    TransformerCollator,
    TransformerDataModule,
    TransformerDataset,
)

# Import tokenization utilities
from .tokenization import BaseTokenizer, BPETokenizer, SimpleTokenizer, Vocabulary

__all__ = [
    "LanguageModelingDataset",
    "TransformerDataset",
    "TransformerCollator",
    "TransformerDataModule",
    "CombinedDataset",
    "create_dataloader",
    "create_data_loaders",
    "randomize_dataset_positions",
    "SimpleTokenizer",
    "BPETokenizer",
    "BaseTokenizer",
    "Vocabulary",
    # Constitutional AI datasets
    "PromptDataset",
    "PromptResponseDataset",
    "ConstitutionalTrainingDataset",
    "PromptTemplate",
    "load_huggingface_dataset",
    "create_default_prompts",
    "save_prompts_to_file",
    "DEFAULT_PROMPT_TEMPLATE",
    "INSTRUCTION_PROMPT_TEMPLATE",
    "CHAT_PROMPT_TEMPLATE",
    "CONSTITUTIONAL_CRITIQUE_TEMPLATE"
]
