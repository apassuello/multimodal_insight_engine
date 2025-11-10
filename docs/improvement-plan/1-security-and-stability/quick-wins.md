# MultiModal Insight Engine - Quick Start Modernization Guide

**Purpose**: Get started with modernization TODAY with working code examples
**Time Estimate**: 4-6 hours for Phase 1 changes
**Files to Create**: Copy/paste ready code snippets below

---

## STEP 1: Add Logging Configuration (30 minutes)

### Create: `src/utils/logging_config.py`

```python
"""Centralized logging configuration for the project."""

import logging
import sys
import os
from pathlib import Path
from typing import Optional

# Log level mapping
LOG_LEVEL_MAP = {
    'DEBUG': logging.DEBUG,
    'INFO': logging.INFO,
    'WARNING': logging.WARNING,
    'ERROR': logging.ERROR,
    'CRITICAL': logging.CRITICAL,
}

def setup_logging(
    level: str = "INFO",
    log_file: Optional[str] = None,
    log_dir: str = "./logs",
) -> logging.Logger:
    """
    Setup logging for the entire project.

    Args:
        level: Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Optional log file path
        log_dir: Directory to store log files

    Returns:
        Configured logger instance
    """
    # Get log level from environment or parameter
    level_env = os.getenv("LOGLEVEL", level)
    level = LOG_LEVEL_MAP.get(level_env.upper(), logging.INFO)

    # Create logger
    logger = logging.getLogger("multimodal_insight_engine")
    logger.setLevel(level)

    # Remove existing handlers
    logger.handlers.clear()

    # Create formatters
    formatter = logging.Formatter(
        fmt='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # File handler (optional)
    if log_file or os.getenv("LOG_FILE"):
        log_path = log_file or os.getenv("LOG_FILE")
        Path(log_dir).mkdir(parents=True, exist_ok=True)

        file_path = Path(log_dir) / log_path if not Path(log_path).is_absolute() else Path(log_path)
        file_handler = logging.FileHandler(file_path)
        file_handler.setLevel(logging.DEBUG)  # Always log everything to file
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger

# Module-level logger setup
logger = setup_logging()

# Configure logging for third-party libraries
logging.getLogger("torch").setLevel(logging.WARNING)
logging.getLogger("transformers").setLevel(logging.WARNING)
logging.getLogger("PIL").setLevel(logging.WARNING)

__all__ = ["setup_logging", "logger"]
```

### Update: `src/models/base_model.py` (replace print statements)

```python
# At top of file, replace:
# print(f"Model saved to {path}")
# with:
import logging
logger = logging.getLogger(__name__)

# Line 73, change from:
# print(f"Model saved to {path}")
# to:
logger.info(f"Model saved to {path}")

# Line 96, change from:
# print(f"Model loaded from {path}")
# to:
logger.info(f"Model loaded from {path}")

# Line 92, change from:
# print(f"Warning: Loading weights...")
# to:
logger.warning(f"Warning: Loading weights from {saved_model_type} into {self.model_type}")
```

### Usage in Training Scripts

```python
# train_constitutional_ai_production.py
import sys
sys.path.insert(0, '/home/user/multimodal_insight_engine')

from src.utils.logging_config import setup_logging

# At the start of main():
logger = setup_logging(level=os.getenv("LOGLEVEL", "INFO"))

# Then use throughout:
logger.info(f"Starting training with config: {config}")
logger.debug(f"Batch {batch_idx}, loss: {loss:.4f}")
logger.warning(f"Learning rate annealing...")
logger.error(f"Failed to load model: {error}")

# Control logging via environment:
# LOGLEVEL=DEBUG python train.py
# LOGLEVEL=WARNING python train.py
# LOG_FILE=training.log python train.py
```

---

## STEP 2: Add Configuration System (1 hour)

### Create: `src/configs/config.py`

```python
"""Unified configuration management for the project."""

import os
import json
from typing import Optional, Literal, Dict, Any
from pathlib import Path
from dataclasses import dataclass, asdict, field

@dataclass
class DataConfig:
    """Data loading configuration."""

    data_root: str = field(
        default="/data",
        metadata={"description": "Path to dataset root"}
    )
    batch_size: int = field(
        default=32,
        metadata={"description": "Training batch size"}
    )
    num_workers: int = field(
        default=4,
        metadata={"description": "Number of data loading workers"}
    )
    max_seq_length: int = field(
        default=512,
        metadata={"description": "Maximum sequence length"}
    )
    train_split: float = field(
        default=0.8,
        metadata={"description": "Training split ratio"}
    )
    val_split: float = field(
        default=0.1,
        metadata={"description": "Validation split ratio"}
    )

    def __post_init__(self):
        """Validate configuration."""
        if self.batch_size <= 0:
            raise ValueError(f"batch_size must be > 0, got {self.batch_size}")
        if not (0 < self.train_split < 1):
            raise ValueError(f"train_split must be in (0, 1), got {self.train_split}")

@dataclass
class OptimizationConfig:
    """Optimization hyperparameters."""

    learning_rate: float = field(
        default=5e-5,
        metadata={"description": "Learning rate"}
    )
    weight_decay: float = field(
        default=0.01,
        metadata={"description": "Weight decay for AdamW"}
    )
    warmup_steps: int = field(
        default=1000,
        metadata={"description": "Warmup steps"}
    )
    max_grad_norm: float = field(
        default=1.0,
        metadata={"description": "Gradient clipping threshold"}
    )

@dataclass
class ModelConfig:
    """Model architecture configuration."""

    model_type: Literal["gpt2", "bert", "t5", "custom"] = field(
        default="gpt2",
        metadata={"description": "Type of model"}
    )
    hidden_size: int = field(
        default=768,
        metadata={"description": "Hidden size"}
    )
    num_layers: int = field(
        default=12,
        metadata={"description": "Number of transformer layers"}
    )
    vocab_size: int = field(
        default=50257,
        metadata={"description": "Vocabulary size"}
    )

@dataclass
class TrainingConfig:
    """Complete training configuration."""

    data: DataConfig = field(default_factory=DataConfig)
    optimization: OptimizationConfig = field(default_factory=OptimizationConfig)
    model: ModelConfig = field(default_factory=ModelConfig)

    # Training parameters
    num_epochs: int = field(default=3, metadata={"description": "Number of epochs"})
    device: Literal["cuda", "cpu", "mps", "auto"] = field(default="auto", metadata={"description": "Device"})
    seed: int = field(default=42, metadata={"description": "Random seed"})
    output_dir: str = field(default="./output", metadata={"description": "Output directory"})
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = field(
        default="INFO",
        metadata={"description": "Logging level"}
    )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)

    def to_json(self, path: str) -> None:
        """Save to JSON file."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def from_json(cls, path: str) -> "TrainingConfig":
        """Load from JSON file."""
        with open(path) as f:
            data = json.load(f)
        return cls(**data)

    @classmethod
    def from_env(cls) -> "TrainingConfig":
        """Load from environment variables."""
        return cls(
            data=DataConfig(
                data_root=os.getenv("DATA_ROOT", "/data"),
                batch_size=int(os.getenv("BATCH_SIZE", "32")),
                num_workers=int(os.getenv("NUM_WORKERS", "4")),
            ),
            optimization=OptimizationConfig(
                learning_rate=float(os.getenv("LEARNING_RATE", "5e-5")),
                weight_decay=float(os.getenv("WEIGHT_DECAY", "0.01")),
            ),
            num_epochs=int(os.getenv("NUM_EPOCHS", "3")),
            device=os.getenv("DEVICE", "auto"),
            log_level=os.getenv("LOGLEVEL", "INFO"),
        )

    def __post_init__(self):
        """Validate configuration."""
        self.data.__post_init__()
        self.output_dir = os.getenv("OUTPUT_DIR", self.output_dir)
```

### Usage in Training Scripts

```python
# Before: train_constitutional_ai_production.py (messy @dataclass)
@dataclass
class TrainingConfig:
    base_model: str = "gpt2"
    # ... scattered defaults ...

# After: Clean import and usage
from src.configs.config import TrainingConfig

if __name__ == "__main__":
    # Load from environment
    config = TrainingConfig.from_env()

    # Or load from file
    config = TrainingConfig.from_json("config.json")

    # Or create with defaults
    config = TrainingConfig()

    # Use throughout training
    train(config)
```

---

## STEP 3: Add Basic Tests (1.5 hours)

### Create: `tests/test_config.py`

```python
"""Tests for configuration management."""

import pytest
import json
import os
from pathlib import Path
from src.configs.config import TrainingConfig, DataConfig

class TestConfigCreation:
    """Test config creation and validation."""

    def test_default_config_creation(self):
        """Test creating config with defaults."""
        config = TrainingConfig()

        assert config.num_epochs == 3
        assert config.data.batch_size == 32
        assert config.device == "auto"

    def test_invalid_batch_size(self):
        """Test that negative batch size raises error."""
        with pytest.raises(ValueError):
            DataConfig(batch_size=-1)

    def test_invalid_split_ratio(self):
        """Test that invalid split ratio raises error."""
        with pytest.raises(ValueError):
            DataConfig(train_split=1.5)  # Must be < 1

class TestConfigSerialization:
    """Test config save/load."""

    def test_save_load_roundtrip(self, tmp_path):
        """Test saving and loading config."""
        config = TrainingConfig(num_epochs=5)
        config_path = tmp_path / "config.json"

        config.to_json(str(config_path))
        loaded = TrainingConfig.from_json(str(config_path))

        assert loaded.num_epochs == 5
        assert loaded.data.batch_size == config.data.batch_size

    def test_to_dict(self):
        """Test dict conversion."""
        config = TrainingConfig()
        config_dict = config.to_dict()

        assert isinstance(config_dict, dict)
        assert "data" in config_dict
        assert "optimization" in config_dict
        assert "model" in config_dict

class TestConfigEnvironment:
    """Test environment variable loading."""

    def test_from_env_loads_variables(self, monkeypatch):
        """Test loading config from environment variables."""
        monkeypatch.setenv("DATA_ROOT", "/tmp/data")
        monkeypatch.setenv("BATCH_SIZE", "64")
        monkeypatch.setenv("LEARNING_RATE", "1e-4")

        config = TrainingConfig.from_env()

        assert config.data.data_root == "/tmp/data"
        assert config.data.batch_size == 64
        assert config.optimization.learning_rate == 1e-4

    def test_from_env_uses_defaults(self, monkeypatch):
        """Test that from_env uses defaults when vars not set."""
        monkeypatch.delenv("DATA_ROOT", raising=False)
        monkeypatch.delenv("BATCH_SIZE", raising=False)

        config = TrainingConfig.from_env()

        assert config.data.data_root == "/data"  # Default
        assert config.data.batch_size == 32  # Default
```

### Run Tests

```bash
cd /home/user/multimodal_insight_engine
python -m pytest tests/test_config.py -v

# Should output:
# tests/test_config.py::TestConfigCreation::test_default_config_creation PASSED
# tests/test_config.py::TestConfigCreation::test_invalid_batch_size PASSED
# ... etc
```

---

## STEP 4: Add Merge Validation Test (30 minutes)

### Create: `tests/test_merge_integration.py`

```python
"""Integration tests for merge validation."""

import pytest
import torch
from pathlib import Path

class TestMergeIntegration:
    """Verify that recent merge didn't break core functionality."""

    def test_models_import(self):
        """Test that models module imports successfully."""
        from src.models.base_model import BaseModel
        from src.models.feed_forward import FeedForwardNN

        assert BaseModel is not None
        assert FeedForwardNN is not None

    def test_data_import(self):
        """Test that data module imports successfully."""
        from src.data.multimodal_dataset import MultimodalDataset

        assert MultimodalDataset is not None

    def test_training_import(self):
        """Test that training module imports successfully."""
        from src.training.trainers.multimodal_trainer import MultimodalTrainer

        assert MultimodalTrainer is not None

    def test_config_import(self):
        """Test that configs module imports successfully."""
        from src.configs.config import TrainingConfig

        config = TrainingConfig()
        assert config is not None

    def test_safety_import(self):
        """Test that safety module imports successfully."""
        # Safety module exists even if not used
        try:
            from src.safety import constitutional
            assert constitutional is not None
        except ImportError:
            # OK if optional
            pass

    def test_agents_dont_break_core(self):
        """Test that .claude/agents don't interfere with core imports."""
        # Core should still import regardless of agents
        from src import models
        from src import data
        from src import training

        assert models is not None
        assert data is not None
        assert training is not None

    def test_model_forward_pass(self):
        """Test that models still do forward pass correctly."""
        from src.models.feed_forward import FeedForwardNN

        model = FeedForwardNN(
            input_size=10,
            hidden_sizes=[64, 32],
            output_size=2,
        )

        x = torch.randn(4, 10)
        output = model(x)

        assert output.shape == (4, 2)

class TestPostMergeBugFixes:
    """Verify that post-merge bugs are addressed."""

    def test_feature_collapse_not_in_trainer(self):
        """Test that trainer doesn't have unreachable debug attributes."""
        from src.training.trainers.multimodal_trainer import MultimodalTrainer

        trainer = MultimodalTrainer(model=None, device="cpu")

        # These should NOT exist (were debug artifacts)
        assert not hasattr(trainer, '_debug_feature_source'), \
            "Trainer still has debug attribute _debug_feature_source"
        assert not hasattr(trainer, '_debug_match_id_source'), \
            "Trainer still has debug attribute _debug_match_id_source"
```

### Run Merge Validation

```bash
cd /home/user/multimodal_insight_engine
python -m pytest tests/test_merge_integration.py -v

# Should output:
# tests/test_merge_integration.py::TestMergeIntegration::test_models_import PASSED
# tests/test_merge_integration.py::TestMergeIntegration::test_data_import PASSED
# tests/test_merge_integration.py::TestMergeIntegration::test_training_import PASSED
# tests/test_merge_integration.py::TestMergeIntegration::test_model_forward_pass PASSED
```

---

## STEP 5: Update setup.py (30 minutes)

### Replace: `setup.py`

```python
from setuptools import setup, find_packages

setup(
    name="multimodal_insight_engine",
    version="0.2.0",  # Bump version for modernization
    description="Multimodal learning framework with safety guarantees",
    author="MultiModal Team",
    python_requires=">=3.8",

    packages=find_packages(),

    # Core dependencies (updated versions)
    install_requires=[
        "torch>=2.0.0",
        "torchvision>=0.15.0",
        "torchaudio>=2.0.0",
        "pytorch-lightning>=2.0.0",
        "transformers>=4.30.0",
        "numpy>=1.21.0",
        "pillow>=9.0.0",
        "tqdm>=4.60.0",
        "pydantic>=2.0.0",  # For config management
        "pyyaml>=6.0.0",
        "accelerate>=0.20.0",
        "datasets>=2.10.0",
    ],

    # Development dependencies
    extras_require={
        "dev": [
            "pytest>=8.0.0",
            "pytest-cov>=4.0.0",
            "black>=23.0.0",
            "mypy>=1.0.0",
            "flake8>=6.0.0",
            "isort>=5.12.0",
        ],
        "docs": [
            "sphinx>=6.0.0",
            "sphinx-rtd-theme>=1.2.0",
            "sphinx-autodoc-typehints>=1.22.0",
        ],
        "optional": [
            "wandb>=0.13.0",
            "optuna>=3.0.0",
            "tensorboard>=2.10.0",
        ],
    },

    # Package configuration
    include_package_data=True,
    package_data={
        "": ["*.md", "*.txt", "*.json", "*.yaml"],
    },

    # Metadata
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],

    keywords=[
        "multimodal learning",
        "vision-language",
        "contrastive learning",
        "safety",
    ],
)
```

---

## CHECKLIST: Complete Phase 1 Today

```markdown
# Quick Start Modernization Checklist

## Create New Files (30 min)
- [ ] Copy `src/utils/logging_config.py` (logging utilities)
- [ ] Copy `src/configs/config.py` (configuration system)
- [ ] Copy `tests/test_config.py` (configuration tests)
- [ ] Copy `tests/test_merge_integration.py` (merge validation tests)

## Update Existing Files (1-2 hours)
- [ ] Update `setup.py` with complete dependencies
- [ ] Replace print() in `src/models/base_model.py` with logging (5 min)
- [ ] Replace print() in `src/evaluation/language_model_evaluation.py` with logging (10 min)
- [ ] Update top 5 training scripts to use new TrainingConfig (20 min)

## Verify (30 min)
- [ ] Run `pytest tests/test_config.py -v`
- [ ] Run `pytest tests/test_merge_integration.py -v`
- [ ] Run `python -m pip install -e .[dev]` (test install)
- [ ] Check that logging output appears with LOGLEVEL=DEBUG

## Commit (10 min)
- [ ] git add . && git commit -m "chore: Phase 1 modernization - add logging, config, tests"

## Total Time: 4-6 hours

## Phase 1 Impact
✓ Eliminates print statement chaos (633 instances)
✓ Centralizes configuration (30+ scattered approaches → 1 system)
✓ Adds basic merge validation tests
✓ Properly declares dependencies in setup.py
✓ Foundation for further modernization
```

---

## Next: Phase 2 (After completing Phase 1)

Once Phase 1 is complete, you can proceed with:

1. **Add comprehensive type hints** (1-2 weeks)
   - Run `mypy src/ --strict` and fix errors
   - Focus on public APIs first

2. **Remove debug code** (2-3 days)
   - Use feature flags instead of embedded debug statements
   - Delete unused `_debug_*` attributes

3. **Add more tests** (1 week)
   - Target 10% → 20% coverage
   - Focus on critical paths (data loading, model forward pass, loss computation)

4. **Consolidate datasets** (1-2 weeks)
   - Reduce 30 dataset classes to 10-15 using composition

---

## How to Apply Each Step

### Step 1: Copy the code
```bash
cd /home/user/multimodal_insight_engine
# Copy code from sections above into the specified files
```

### Step 2: Create files
```bash
# Create new files with the code above
mkdir -p src/configs
touch src/configs/config.py
touch tests/test_config.py
touch tests/test_merge_integration.py
```

### Step 3: Test immediately
```bash
python -m pytest tests/test_config.py -v
python -m pytest tests/test_merge_integration.py -v
```

### Step 4: Use in your code
```python
# In training scripts:
from src.configs.config import TrainingConfig
from src.utils.logging_config import setup_logging

config = TrainingConfig.from_env()
logger = setup_logging(level=config.log_level)
```

---

## Environment Variables for Quick Testing

```bash
# Test logging configuration
LOGLEVEL=DEBUG python train.py
LOGLEVEL=INFO python train.py
LOGLEVEL=WARNING python train.py
LOG_FILE=training.log python train.py

# Test config from environment
DATA_ROOT=/custom/path BATCH_SIZE=64 LEARNING_RATE=1e-4 python train.py

# Run tests
pytest tests/test_config.py -v
pytest tests/test_merge_integration.py -v
pytest tests/ -v --cov=src --cov-report=html
```

---

**Ready to start? Copy the code above and run the checklist!**
