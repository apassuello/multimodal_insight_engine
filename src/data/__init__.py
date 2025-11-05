# src/data/__init__.py
"""
Data modules for the MultiModal Insight Engine.

This package contains components for loading, preprocessing, and 
managing data for various model architectures.
"""

# Import dataset classes
from .language_modeling import LanguageModelingDataset
from .sequence_data import TransformerDataset, TransformerCollator, TransformerDataModule
from .combined_dataset import CombinedDataset
from .dataloader import create_dataloader

# Import multimodal data utilities
from .multimodal_data_utils import create_data_loaders, randomize_dataset_positions

# Import tokenization utilities
from .tokenization import (
    SimpleTokenizer,
    BPETokenizer,
    BaseTokenizer,
    Vocabulary
)

# Import constitutional AI dataset utilities
from .constitutional_dataset import (
    PromptDataset,
    PromptResponseDataset,
    ConstitutionalTrainingDataset,
    PromptTemplate,
    load_huggingface_dataset,
    create_default_prompts,
    save_prompts_to_file,
    DEFAULT_PROMPT_TEMPLATE,
    INSTRUCTION_PROMPT_TEMPLATE,
    CHAT_PROMPT_TEMPLATE,
    CONSTITUTIONAL_CRITIQUE_TEMPLATE
)

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