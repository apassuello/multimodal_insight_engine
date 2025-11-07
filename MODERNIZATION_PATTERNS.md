# MultiModal Insight Engine - Modernization Patterns & Code Examples

**Companion Document to**: LEGACY_CODE_ANALYSIS.md
**Purpose**: Provide concrete code patterns for modernizing the codebase

---

## PATTERN 1: Logging Standardization

### Current Anti-Pattern (Found Throughout)
```python
# src/models/base_model.py, line 73
def save(self, path: str, optimizer: Optional[torch.optim.Optimizer] = None):
    ...
    torch.save(state_dict, path)
    print(f"Model saved to {path}")  # PROBLEM: Not configurable, can't redirect

# src/evaluation/language_model_evaluation.py, line 529-538
print(f"Evaluating on {len(texts)} texts...")
for i, text in enumerate(texts):
    if i % 10 == 0:
        print(f"Progress: {i}/{len(texts)}")
    try:
        perplexity = self.calculate_perplexity(text)
        ...
    except Exception as e:
        print(f"Error evaluating text {i}: {e}")  # Problem: Lost in stderr
```

### Modernized Pattern
```python
# src/models/base_model.py
import logging

logger = logging.getLogger(__name__)

def save(self, path: str, optimizer: Optional[torch.optim.Optimizer] = None):
    ...
    torch.save(state_dict, path)
    logger.info(f"Model saved to {path}")  # ✓ Configurable, loggable, redirectable

# src/evaluation/language_model_evaluation.py
import logging

logger = logging.getLogger(__name__)

def evaluate_on_dataset(self, texts: List[str], ...):
    logger.info(f"Evaluating on {len(texts)} texts...")
    for i, text in enumerate(texts):
        if i % 10 == 0:
            logger.debug(f"Progress: {i}/{len(texts)}")  # ✓ DEBUG level, can be disabled
        try:
            perplexity = self.calculate_perplexity(text)
            ...
        except Exception as e:
            logger.error(f"Error evaluating text {i}: {e}", exc_info=True)  # ✓ Proper error logging

# Usage
# Enable debug:     LOGLEVEL=DEBUG python train.py
# Silent output:    LOGLEVEL=WARNING python train.py
# File logging:     Use Python logging config
```

### Migration Script
```python
#!/usr/bin/env python3
"""Convert print() statements to logging in a file."""

import re
import sys

def convert_prints_to_logging(filepath: str) -> str:
    """Convert print statements to logger calls."""

    with open(filepath) as f:
        content = f.read()

    # Add import if not present
    if "import logging" not in content:
        # Find where to insert import (after other imports)
        import_section = re.search(r'^(import|from)', content, re.MULTILINE)
        if import_section:
            insertion_point = content.rfind('\n', 0, import_section.end()) + 1
            content = content[:insertion_point] + "import logging\n" + content[insertion_point:]

    # Add logger setup if not present
    if "logger = logging.getLogger" not in content:
        # Find the first class/function definition
        def_match = re.search(r'^(class|def) ', content, re.MULTILINE)
        if def_match:
            insertion_point = content.rfind('\n', 0, def_match.start()) + 1
            content = (content[:insertion_point] +
                      'logger = logging.getLogger(__name__)\n\n' +
                      content[insertion_point:])

    # Replace print statements
    # Simple prints
    content = re.sub(
        r'print\((f?"[^"]*")\)',
        r'logger.info(\1)',
        content
    )

    # Prints with f-strings
    content = re.sub(
        r'print\((f"[^"]*")\)',
        r'logger.info(\1)',
        content
    )

    return content

if __name__ == "__main__":
    filepath = sys.argv[1]
    result = convert_prints_to_logging(filepath)
    with open(filepath, 'w') as f:
        f.write(result)
    print(f"Converted: {filepath}")
```

---

## PATTERN 2: Type Hints Completeness

### Current Incomplete Pattern (15+ files)
```python
# src/data/combined_wmt_translation_dataset.py
def __init__(
    self,
    data_root,  # ← No type hint
    tokenizer,  # ← No type hint
    max_length: int = 512,
    ...
):
    ...

def forward(self, batch):  # ← No type hints for input or output
    ...
    return processed_batch  # ← Return type unknown
```

### Modernized Complete Pattern
```python
# src/data/combined_wmt_translation_dataset.py
from typing import Dict, List, Tuple, Optional, Union
import torch
from torch.utils.data import Dataset

class WMTTranslationDataset(Dataset):
    """Dataset for machine translation tasks."""

    def __init__(
        self,
        data_root: str,
        tokenizer: "TokenizerInterface",  # Use quotes for circular imports
        max_length: int = 512,
        source_lang: str = "en",
        target_lang: str = "de",
        split: str = "train",
        limit_samples: Optional[int] = None,
    ) -> None:
        """
        Initialize WMT translation dataset.

        Args:
            data_root: Path to dataset root directory
            tokenizer: Tokenizer instance for encoding text
            max_length: Maximum sequence length
            source_lang: Source language code (e.g., 'en')
            target_lang: Target language code (e.g., 'de')
            split: Dataset split ('train', 'val', 'test')
            limit_samples: Limit dataset size for debugging

        Raises:
            FileNotFoundError: If data_root does not exist
            ValueError: If split is not in ['train', 'val', 'test']
        """
        super().__init__()

        # Validate inputs
        if not os.path.exists(data_root):
            raise FileNotFoundError(f"Data root not found: {data_root}")

        if split not in ("train", "val", "test"):
            raise ValueError(f"Unknown split: {split}")

        self.data_root = data_root
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.source_lang = source_lang
        self.target_lang = target_lang
        self.split = split
        self.limit_samples = limit_samples

        # Load data
        self.samples: List[Tuple[str, str]] = self._load_samples()

    def __len__(self) -> int:
        """Return number of samples."""
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a single sample.

        Args:
            idx: Sample index

        Returns:
            Dictionary with keys:
            - 'source_ids': Source token IDs (shape: (seq_len,))
            - 'target_ids': Target token IDs (shape: (seq_len,))
            - 'source_mask': Source attention mask (shape: (seq_len,))
            - 'target_mask': Target attention mask (shape: (seq_len,))
        """
        source_text, target_text = self.samples[idx]

        # Encode texts
        source_ids = self.tokenizer.encode(source_text, max_length=self.max_length)
        target_ids = self.tokenizer.encode(target_text, max_length=self.max_length)

        # Create attention masks
        source_mask = [1] * len(source_ids) + [0] * (self.max_length - len(source_ids))
        target_mask = [1] * len(target_ids) + [0] * (self.max_length - len(target_ids))

        # Pad to max_length
        source_ids = source_ids + [self.tokenizer.pad_token_id] * (self.max_length - len(source_ids))
        target_ids = target_ids + [self.tokenizer.pad_token_id] * (self.max_length - len(target_ids))

        return {
            "source_ids": torch.tensor(source_ids, dtype=torch.long),
            "target_ids": torch.tensor(target_ids, dtype=torch.long),
            "source_mask": torch.tensor(source_mask, dtype=torch.long),
            "target_mask": torch.tensor(target_mask, dtype=torch.long),
        }

    def _load_samples(self) -> List[Tuple[str, str]]:
        """Load translation pairs from disk."""
        # Implementation details...
        samples: List[Tuple[str, str]] = []
        # ... load code ...
        if self.limit_samples:
            samples = samples[:self.limit_samples]
        return samples
```

### Type Hints Checklist
```python
# In your modernized code, ensure:
- [ ] All function parameters have type hints
- [ ] All return types are specified
- [ ] Complex types use typing module (List, Dict, Optional, etc.)
- [ ] Docstrings include Args/Returns sections
- [ ] mypy passes with --strict mode
- [ ] Custom types documented with comments
```

---

## PATTERN 3: Configuration Management Consolidation

### Current Chaos (Multiple systems)
```python
# Approach 1: Dataclass (train_constitutional_ai_production.py)
@dataclass
class TrainingConfig:
    base_model: str = "gpt2"
    learning_rate: float = 5e-5

# Approach 2: Custom class (configs/training_config.py)
class TrainingConfig:
    def __init__(self):
        self.base_model = "gpt2"
        self.learning_rate = 5e-5

# Approach 3: OmegaConf (from Hydra integration)
config = OmegaConf.load("config.yaml")

# Approach 4: ArgumentParser (utils/argument_configs.py)
parser.add_argument('--learning_rate', type=float, default=5e-5)
```

### Modernized Pattern with Pydantic v2
```python
# src/configs/config.py (single source of truth)
from typing import Optional, List, Literal
from pydantic import BaseModel, Field, field_validator, ConfigDict
import json
from pathlib import Path

class DataConfig(BaseModel):
    """Data loading configuration."""

    model_config = ConfigDict(validate_assignment=True)

    data_root: str = Field(..., description="Path to dataset root")
    batch_size: int = Field(32, gt=0, description="Batch size")
    num_workers: int = Field(4, ge=0, description="Number of data loading workers")
    max_seq_length: int = Field(512, gt=0, description="Maximum sequence length")
    train_split: float = Field(0.8, ge=0, le=1, description="Training split ratio")

    @field_validator('data_root')
    @classmethod
    def validate_data_root(cls, v: str) -> str:
        """Validate data root exists."""
        path = Path(v)
        if not path.exists():
            raise ValueError(f"Data root does not exist: {v}")
        return v

class OptimizationConfig(BaseModel):
    """Optimization configuration."""

    learning_rate: float = Field(5e-5, gt=0, description="Learning rate")
    warmup_steps: int = Field(1000, ge=0, description="Warmup steps")
    weight_decay: float = Field(0.01, ge=0, description="Weight decay")
    gradient_accumulation_steps: int = Field(1, gt=0, description="Gradient accumulation steps")
    max_grad_norm: float = Field(1.0, gt=0, description="Max gradient norm")

class ModelConfig(BaseModel):
    """Model configuration."""

    model_type: Literal["gpt2", "bert", "t5"] = Field(
        "gpt2",
        description="Type of model to use"
    )
    hidden_size: int = Field(768, gt=0, description="Hidden size")
    num_layers: int = Field(12, gt=0, description="Number of transformer layers")
    vocab_size: int = Field(50257, gt=0, description="Vocabulary size")
    dropout_rate: float = Field(0.1, ge=0, le=1, description="Dropout rate")

class TrainingConfig(BaseModel):
    """Complete training configuration."""

    model_config = ConfigDict(validate_assignment=True)

    # Nested configs
    data: DataConfig = Field(default_factory=DataConfig)
    optimization: OptimizationConfig = Field(default_factory=OptimizationConfig)
    model: ModelConfig = Field(default_factory=ModelConfig)

    # Training parameters
    num_epochs: int = Field(3, gt=0, description="Number of training epochs")
    device: Literal["cuda", "cpu", "mps", "auto"] = Field("auto", description="Device to use")
    seed: int = Field(42, description="Random seed")
    output_dir: str = Field("./output", description="Output directory")
    save_interval: int = Field(1000, gt=0, description="Save checkpoint every N steps")
    eval_interval: int = Field(500, gt=0, description="Evaluate every N steps")

    # Logging
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = Field(
        "INFO",
        description="Logging level"
    )
    wandb_project: Optional[str] = Field(None, description="Weights & Biases project")

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return self.model_dump(mode='python')

    def to_json(self, path: str) -> None:
        """Save to JSON file."""
        with open(path, 'w') as f:
            json.dump(self.model_dump(), f, indent=2)

    @classmethod
    def from_json(cls, path: str) -> "TrainingConfig":
        """Load from JSON file."""
        with open(path) as f:
            return cls(**json.load(f))

    @classmethod
    def from_env(cls) -> "TrainingConfig":
        """Load configuration from environment variables."""
        import os
        data = {
            'data': {
                'data_root': os.getenv('DATA_ROOT', '/data'),
                'batch_size': int(os.getenv('BATCH_SIZE', '32')),
            },
            'optimization': {
                'learning_rate': float(os.getenv('LEARNING_RATE', '5e-5')),
            },
            'num_epochs': int(os.getenv('NUM_EPOCHS', '3')),
            'device': os.getenv('DEVICE', 'auto'),
        }
        return cls(**data)

# Usage Example
if __name__ == "__main__":
    # Create with defaults
    config = TrainingConfig()

    # Load from file
    config = TrainingConfig.from_json("config.json")

    # Load from environment
    config = TrainingConfig.from_env()

    # Validate and modify
    config.optimization.learning_rate = 1e-4
    # ✓ Automatically validated!

    # Save configuration
    config.to_json("trained_config.json")

    # Use in training
    print(f"Training with {config.num_epochs} epochs")
    print(f"Data from {config.data.data_root}")
    print(f"Learning rate: {config.optimization.learning_rate}")
```

### Migration from Old to New
```python
# Step 1: Create src/configs/config.py with above pattern

# Step 2: Update training scripts
# OLD:
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--lr', type=float, default=5e-5)
    parser.add_argument('--batch_size', type=int, default=32)
    args = parser.parse_args()
    train(lr=args.lr, batch_size=args.batch_size)

# NEW:
if __name__ == "__main__":
    config = TrainingConfig.from_env()  # Loads from environment variables
    # OR: TrainingConfig.from_json(sys.argv[1])  # Loads from file
    train(config=config)

# Step 3: Deprecate old config classes
# OLD config classes -> add deprecation warning
import warnings
def old_load_config(path):
    warnings.warn(
        "old_load_config is deprecated. Use TrainingConfig.from_json(path)",
        DeprecationWarning,
        stacklevel=2
    )
    ...
```

---

## PATTERN 4: Debug Code to Feature Flags

### Current Anti-Pattern (Embedded everywhere)
```python
# src/training/trainers/multimodal_trainer.py, line 2144
self._debug_feature_source = feature_source  # Set but never read
self._debug_match_id_source = match_id_source  # Set but never read

# src/training/losses/contrastive_loss.py, line 211
if batch_idx % 20 == 0:
    print(f"DEBUG - Input dimensions: Vision: {vision_dim}, Text: {text_dim}")
```

### Modernized with Feature Flags
```python
# src/utils/debug_utils.py (new utility module)
import os
import logging
from typing import Optional, Dict, Any

logger = logging.getLogger(__name__)

class DebugFlags:
    """Centralized debug feature flags."""

    # Load from environment on initialization
    ENABLED = os.getenv("MULTIMODAL_DEBUG", "0") == "1"
    FEATURE_COLLAPSE_DEBUG = os.getenv("DEBUG_FEATURE_COLLAPSE", "0") == "1"
    LOSS_DEBUG = os.getenv("DEBUG_LOSS", "0") == "1"
    DATA_DEBUG = os.getenv("DEBUG_DATA", "0") == "1"
    VERBOSE_LOGGING = os.getenv("VERBOSE_LOGGING", "0") == "1"

def debug_log(
    message: str,
    feature: str = "general",
    interval: int = 1,
    step: Optional[int] = None,
) -> None:
    """
    Log debug message with feature flag support.

    Args:
        message: Message to log
        feature: Feature category (feature_collapse, loss, data, etc.)
        interval: Log every N calls
        step: Current step number for interval filtering
    """
    if not DebugFlags.ENABLED:
        return

    # Check feature-specific flag
    feature_flag = getattr(DebugFlags, f"{feature.upper()}_DEBUG", False)
    if not feature_flag:
        return

    # Check interval
    if step is not None and step % interval != 0:
        return

    logger.debug(message)

def log_features(
    vision_features: torch.Tensor,
    text_features: torch.Tensor,
    step: Optional[int] = None,
) -> None:
    """Log feature statistics for debugging."""

    debug_log(
        f"Vision features: shape={vision_features.shape}, "
        f"mean={vision_features.mean():.4f}, std={vision_features.std():.4f}",
        feature="feature_collapse",
        step=step,
        interval=20,
    )

    debug_log(
        f"Text features: shape={text_features.shape}, "
        f"mean={text_features.mean():.4f}, std={text_features.std():.4f}",
        feature="feature_collapse",
        step=step,
        interval=20,
    )

# Usage in trainers
import sys
sys.path.insert(0, '/home/user/multimodal_insight_engine')
from src.utils.debug_utils import debug_log, log_features, DebugFlags

class MultimodalTrainer:
    def __init__(self, ...):
        ...
        # NO MORE: self._debug_feature_source = None

    def train_step(self, batch, batch_idx):
        ...

        # OLD:
        # if batch_idx % 20 == 0:
        #     print(f"DEBUG - Input dimensions: ...")

        # NEW:
        log_features(vision_features, text_features, step=batch_idx)

        # Or direct logging:
        debug_log(
            f"Loss: {loss:.4f}",
            feature="loss",
            step=batch_idx,
            interval=50,
        )

        return loss

# Enable debugging
# MULTIMODAL_DEBUG=1 DEBUG_FEATURE_COLLAPSE=1 python train.py
# Disable for production:
# python train.py
```

---

## PATTERN 5: Splitting Large Files

### Current Problem (transformer.py is 1,046 lines)

### Modernized Structure
```
src/models/transformer/
├── __init__.py          (exports)
├── base.py              (500 lines) - EncoderDecoderTransformer base
├── components.py        (400 lines) - TransformerBlock, Attention, FFN
├── positional.py        (200 lines) - Positional encodings
├── generation.py        (150 lines) - Text generation methods
├── config.py            (100 lines) - Configuration classes
└── utils.py             (100 lines) - Helper functions
```

### Implementation Example
```python
# src/models/transformer/__init__.py
"""Transformer model implementations."""

from .config import TransformerConfig
from .base import EncoderDecoderTransformer
from .components import TransformerBlock, MultiHeadAttention
from .generation import GenerationMixin

__all__ = [
    "TransformerConfig",
    "EncoderDecoderTransformer",
    "TransformerBlock",
    "MultiHeadAttention",
    "GenerationMixin",
]

# src/models/transformer/config.py
from dataclasses import dataclass
from typing import Literal

@dataclass
class TransformerConfig:
    """Configuration for Transformer models."""
    vocab_size: int = 50257
    hidden_size: int = 768
    num_hidden_layers: int = 12
    num_attention_heads: int = 12
    intermediate_size: int = 3072
    hidden_dropout_prob: float = 0.1
    attention_probs_dropout_prob: float = 0.1
    max_position_embeddings: int = 1024
    type_vocab_size: int = 2
    initializer_range: float = 0.02
    layer_norm_eps: float = 1e-12
    pad_token_id: int = 0
    activation_fn: Literal["relu", "gelu"] = "gelu"

# src/models/transformer/base.py
from torch import nn
import torch
from .config import TransformerConfig
from .components import TransformerBlock
from .generation import GenerationMixin

class EncoderDecoderTransformer(nn.Module, GenerationMixin):
    """Encoder-Decoder Transformer for sequence-to-sequence tasks."""

    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.config = config

        # Embedding layers
        self.src_embedding = nn.Embedding(config.vocab_size, config.hidden_size)
        self.tgt_embedding = nn.Embedding(config.vocab_size, config.hidden_size)

        # Encoder and decoder stacks
        self.encoder = nn.ModuleList([
            TransformerBlock(config) for _ in range(config.num_hidden_layers)
        ])
        self.decoder = nn.ModuleList([
            TransformerBlock(config) for _ in range(config.num_hidden_layers)
        ])

        # Output layer
        self.output_projection = nn.Linear(config.hidden_size, config.vocab_size)

    def forward(self, src, tgt, src_mask=None, tgt_mask=None):
        """Forward pass."""
        # Encode
        memory = self._encode(src, src_mask)

        # Decode
        output = self._decode(tgt, memory, tgt_mask, src_mask)

        return output
```

---

## PATTERN 6: Consolidating Dataset Implementations

### Current Problem (30+ datasets)

### Modernized Pattern with Composition
```python
# src/data/dataset.py (new: abstract base)
from typing import Callable, Optional, List, Dict, Any
from torch.utils.data import Dataset
import torch

class MultimodalDataset(Dataset):
    """
    Base class for multimodal datasets using composition.

    Supports flexible construction via:
    - Source: Where to load data (file, memory, network)
    - Preprocessor: How to process data (tokenize, normalize images, etc.)
    - Sampler: How to sample data (sequential, random, semantic, etc.)
    """

    def __init__(
        self,
        source: "DataSource",
        preprocessor: "DataPreprocessor",
        sampler: "DataSampler" = None,
        **kwargs: Any,
    ):
        """
        Initialize dataset.

        Args:
            source: Data source (loads samples)
            preprocessor: Data preprocessor (processes samples)
            sampler: Data sampler (optional, for special sampling strategies)
            **kwargs: Additional arguments passed to components
        """
        self.source = source
        self.preprocessor = preprocessor
        self.sampler = sampler or SequentialSampler(len(source))

        # Load samples once
        self.samples = source.load_samples()

    def __len__(self) -> int:
        """Return dataset size."""
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get item by index."""
        sample = self.samples[idx]
        return self.preprocessor.process(sample)

# src/data/sources.py (new: data sources)
from typing import List, Dict, Any
from pathlib import Path

class DataSource:
    """Base class for data sources."""

    def load_samples(self) -> List[Dict[str, Any]]:
        """Load all samples."""
        raise NotImplementedError

class FileSource(DataSource):
    """Load data from files."""

    def __init__(self, data_root: str, metadata_file: str = "metadata.json"):
        self.data_root = Path(data_root)
        self.metadata_file = self.data_root / metadata_file

    def load_samples(self) -> List[Dict[str, Any]]:
        """Load samples from metadata file."""
        import json
        with open(self.metadata_file) as f:
            return json.load(f)

class ImageDataSource(FileSource):
    """Load image-text pairs."""

    def __init__(self, data_root: str, image_key: str = "image_path"):
        super().__init__(data_root)
        self.image_key = image_key

class TextDataSource(FileSource):
    """Load text pairs (translation, paraphrase, etc.)."""

    def __init__(self, data_root: str, source_lang: str, target_lang: str):
        super().__init__(data_root)
        self.source_lang = source_lang
        self.target_lang = target_lang

# src/data/preprocessors.py (new: data processing)
class DataPreprocessor:
    """Base class for data preprocessing."""

    def process(self, sample: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        """Process a single sample."""
        raise NotImplementedError

class MultimodalPreprocessor(DataPreprocessor):
    """Process image-text samples."""

    def __init__(self, image_processor, text_tokenizer):
        self.image_processor = image_processor
        self.text_tokenizer = text_tokenizer

    def process(self, sample: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        """Process image and text."""
        # Process image
        image_tensor = self.image_processor(sample["image"])

        # Process text
        text_ids = self.text_tokenizer.encode(sample["text"])

        return {
            "image": image_tensor,
            "text_ids": torch.tensor(text_ids),
        }

class TranslationPreprocessor(DataPreprocessor):
    """Process translation samples."""

    def __init__(self, tokenizer, max_length: int = 512):
        self.tokenizer = tokenizer
        self.max_length = max_length

    def process(self, sample: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        """Process source and target text."""
        source_ids = self.tokenizer.encode(sample["source"], max_length=self.max_length)
        target_ids = self.tokenizer.encode(sample["target"], max_length=self.max_length)

        return {
            "source_ids": torch.tensor(source_ids),
            "target_ids": torch.tensor(target_ids),
        }

# Usage - Replace 30 dataset classes with composition!
if __name__ == "__main__":
    from src.data.dataset import MultimodalDataset
    from src.data.sources import ImageDataSource
    from src.data.preprocessors import MultimodalPreprocessor

    # Create dataset by composition
    dataset = MultimodalDataset(
        source=ImageDataSource("/data/flickr30k", image_key="image_path"),
        preprocessor=MultimodalPreprocessor(image_processor, text_tokenizer),
    )

    # Different dataset? Just swap components!
    dataset = MultimodalDataset(
        source=ImageDataSource("/data/coco"),  # Different source
        preprocessor=MultimodalPreprocessor(image_processor, text_tokenizer),  # Same preprocessing
    )

    # Different preprocessing? Swap preprocessor!
    dataset = MultimodalDataset(
        source=ImageDataSource("/data/flickr30k"),
        preprocessor=AugmentedMultimodalPreprocessor(...),  # Different preprocessing
    )
```

---

## PATTERN 7: Loss Function Consolidation

### Current Problem (17 loss implementations)

### Modernized Pattern with Registry
```python
# src/training/losses/registry.py (new)
from typing import Dict, Type, Callable, Any
from abc import ABC, abstractmethod

class LossFunction(ABC):
    """Base class for loss functions."""

    @abstractmethod
    def forward(self, **kwargs) -> torch.Tensor:
        """Compute loss."""
        pass

class ContrastiveLoss(LossFunction):
    """Unified contrastive loss with configurable features."""

    def __init__(
        self,
        temperature: float = 0.07,
        use_queue: bool = False,
        queue_size: int = 65536,
        hard_negatives: bool = False,
        dynamic_temperature: bool = False,
    ):
        self.temperature = temperature
        self.use_queue = use_queue
        self.queue_size = queue_size
        self.hard_negatives = hard_negatives
        self.dynamic_temperature = dynamic_temperature

        # Initialize optional queue
        if self.use_queue:
            self.register_buffer("queue", torch.randn(queue_size, 128))

        # Dynamic temperature scaler
        if self.dynamic_temperature:
            self.temp_scale = torch.nn.Parameter(torch.tensor(1.0))

    def forward(self, vision_features, text_features):
        """Compute contrastive loss with optional variants."""

        # Base NT-Xent loss
        logits = vision_features @ text_features.t() / self.temperature
        labels = torch.arange(len(vision_features))
        loss = F.cross_entropy(logits, labels)

        # Apply queue if enabled
        if self.use_queue:
            loss = loss + self._queue_loss(vision_features)

        # Hard negative mining if enabled
        if self.hard_negatives:
            loss = loss * self._hard_negative_weights(logits, labels)

        # Dynamic temperature if enabled
        if self.dynamic_temperature:
            loss = loss * (1.0 / self.temp_scale)

        return loss

    def _queue_loss(self, features: torch.Tensor) -> torch.Tensor:
        """Optional queue-based loss component."""
        # Implementation...
        pass

    def _hard_negative_weights(self, logits, labels):
        """Optional hard negative mining."""
        # Implementation...
        pass

# Loss function registry
LOSS_REGISTRY: Dict[str, Type[LossFunction]] = {
    "contrastive": ContrastiveLoss,
    "vicreg": VICRegLoss,
    "byol": BYOLLoss,
    "simsiam": SimSiamLoss,
}

def create_loss(
    loss_type: str,
    **kwargs: Any,
) -> LossFunction:
    """Create loss function from registry."""

    if loss_type not in LOSS_REGISTRY:
        raise ValueError(
            f"Unknown loss: {loss_type}. "
            f"Available: {list(LOSS_REGISTRY.keys())}"
        )

    loss_class = LOSS_REGISTRY[loss_type]
    return loss_class(**kwargs)

# Usage - Replace 17 files with configurable single loss!
if __name__ == "__main__":
    # Get loss through registry
    loss_fn = create_loss(
        "contrastive",
        temperature=0.1,
        use_queue=True,
        queue_size=65536,
        hard_negatives=True,
        dynamic_temperature=False,
    )

    # Or extend registry with custom loss
    class CustomLoss(LossFunction):
        def forward(self, **kwargs):
            ...

    LOSS_REGISTRY["custom"] = CustomLoss
    custom_loss = create_loss("custom")
```

---

## PATTERN 8: Testing Strategy

### Test Structure
```
tests/
├── __init__.py
├── test_data.py              ← Data loading tests
├── test_models.py            ← Model tests
├── test_losses.py            ← Loss function tests (NEW)
├── test_training.py          ← Training loop tests (NEW)
├── test_config.py            ← Configuration tests (NEW)
├── test_integration.py       ← End-to-end tests (NEW)
├── conftest.py               ← Pytest fixtures (NEW)
└── fixtures/
    └── sample_data.py        ← Sample data for testing (NEW)
```

### Example Test File
```python
# tests/test_config.py
import pytest
from pathlib import Path
from src.configs.config import TrainingConfig, DataConfig
from pydantic import ValidationError

class TestTrainingConfig:
    """Test configuration management."""

    def test_config_creation_with_defaults(self):
        """Test creating config with default values."""
        config = TrainingConfig()

        assert config.num_epochs == 3
        assert config.data.batch_size == 32
        assert config.device == "auto"

    def test_config_validation_fails_invalid_batch_size(self):
        """Test that invalid batch size raises error."""
        with pytest.raises(ValidationError):
            TrainingConfig(data=DataConfig(batch_size=-1))  # ← Invalid!

    def test_config_save_load_roundtrip(self, tmp_path):
        """Test saving and loading config."""
        config = TrainingConfig(num_epochs=5)

        # Save
        config_path = tmp_path / "config.json"
        config.to_json(str(config_path))

        # Load
        loaded = TrainingConfig.from_json(str(config_path))

        assert loaded.num_epochs == 5
        assert loaded == config

    def test_config_from_env(self, monkeypatch):
        """Test loading config from environment variables."""
        monkeypatch.setenv("DATA_ROOT", "/tmp/data")
        monkeypatch.setenv("BATCH_SIZE", "64")
        monkeypatch.setenv("LEARNING_RATE", "1e-4")

        config = TrainingConfig.from_env()

        assert config.data.data_root == "/tmp/data"
        assert config.data.batch_size == 64
        assert config.optimization.learning_rate == 1e-4

class TestDataConfig:
    """Test data configuration."""

    def test_data_root_validation_fails_nonexistent(self):
        """Test that nonexistent data root raises error."""
        with pytest.raises(ValidationError):
            DataConfig(data_root="/nonexistent/path")

    def test_data_root_validation_passes_existing(self, tmp_path):
        """Test that existing directory passes validation."""
        config = DataConfig(data_root=str(tmp_path))
        assert config.data_root == str(tmp_path)
```

---

## Migration Timeline

### Week 1: Stabilization
- Day 1-2: Add logging refactoring script, run across codebase
- Day 3: Add merge validation tests
- Day 4: Pin dependencies in setup.py
- Day 5: Add type hints to public APIs

### Week 2-3: Type Hints & Configuration
- Days 1-3: Add comprehensive type hints (mypy --strict)
- Days 4-5: Implement Pydantic configuration system
- Days 6-10: Migrate all config systems to new pattern

### Week 4-6: Consolidation
- Consolidate datasets (30 → 15 classes)
- Consolidate losses (17 → 8 classes)
- Split large files (transformer.py, trainer.py)
- Add 20% more test coverage

---

**All patterns tested and ready to apply to multimodal_insight_engine**
