"""MODULE: multi_model_manager.py
PURPOSE: Dual model management for Constitutional AI demo
KEY COMPONENTS:
- MultiModelManager: Manages separate evaluation and generation models
- Qwen2-1.5B-Instruct for evaluation (best instruction-following)
- Phi-2 for generation/training (best fine-tuning performance)
- Optimized memory usage with model unloading
DEPENDENCIES: torch, transformers, typing
SPECIAL NOTES: Supports dual model architecture for improved performance
"""

import torch
from typing import Optional, Tuple, Dict, Any
from dataclasses import dataclass
from enum import Enum

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizer
)


class ModelRole(Enum):
    """Roles for different models in the system."""
    EVALUATION = "evaluation"  # Model used for evaluating text
    GENERATION = "generation"  # Model used for generation and training


@dataclass
class ModelConfig:
    """Configuration for a model."""
    name: str
    role: ModelRole
    hf_model_id: str  # Hugging Face model identifier
    max_memory_gb: float


# Recommended model configurations
RECOMMENDED_CONFIGS = {
    "qwen2-1.5b-instruct": ModelConfig(
        name="Qwen2-1.5B-Instruct",
        role=ModelRole.EVALUATION,
        hf_model_id="Qwen/Qwen2-1.5B-Instruct",
        max_memory_gb=3.0
    ),
    "phi-2": ModelConfig(
        name="Phi-2",
        role=ModelRole.GENERATION,
        hf_model_id="microsoft/phi-2",
        max_memory_gb=5.4
    ),
    # Fallback options
    "gpt2": ModelConfig(
        name="GPT-2",
        role=ModelRole.GENERATION,
        hf_model_id="gpt2",
        max_memory_gb=0.5
    )
}


class MultiModelManager:
    """
    Manages multiple models for Constitutional AI demo.

    Supports dual model architecture:
    - Evaluation model: Best at instruction-following and evaluation
    - Generation model: Best at fine-tuning and learning

    Memory optimization:
    - Models can be loaded/unloaded independently
    - Automatic device selection (MPS/CUDA/CPU)
    - Memory monitoring and warnings
    """

    def __init__(self):
        """Initialize multi-model manager."""
        self.eval_model: Optional[PreTrainedModel] = None
        self.eval_tokenizer: Optional[PreTrainedTokenizer] = None
        self.eval_config: Optional[ModelConfig] = None

        self.gen_model: Optional[PreTrainedModel] = None
        self.gen_tokenizer: Optional[PreTrainedTokenizer] = None
        self.gen_config: Optional[ModelConfig] = None

        self.device: Optional[torch.device] = None
        self._auto_select_device()

    def _auto_select_device(self) -> None:
        """Auto-select best available device."""
        if torch.cuda.is_available():
            self.device = torch.device('cuda')
        elif hasattr(torch, 'mps') and hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            self.device = torch.device('mps')
        else:
            self.device = torch.device('cpu')

    def load_evaluation_model(
        self,
        model_key: str = "qwen2-1.5b-instruct"
    ) -> Tuple[bool, str]:
        """
        Load model for evaluation tasks.

        Args:
            model_key: Key from RECOMMENDED_CONFIGS

        Returns:
            Tuple of (success, message)
        """
        if model_key not in RECOMMENDED_CONFIGS:
            return False, f"✗ Unknown model: {model_key}"

        config = RECOMMENDED_CONFIGS[model_key]

        try:
            print(f"Loading evaluation model: {config.name} ({config.hf_model_id})...")

            # Load tokenizer
            self.eval_tokenizer = AutoTokenizer.from_pretrained(
                config.hf_model_id,
                trust_remote_code=True
            )

            # Set padding token if not set
            if self.eval_tokenizer.pad_token is None:
                self.eval_tokenizer.pad_token = self.eval_tokenizer.eos_token

            # Load model
            self.eval_model = AutoModelForCausalLM.from_pretrained(
                config.hf_model_id,
                torch_dtype=torch.float16 if self.device.type != 'cpu' else torch.float32,
                trust_remote_code=True,
                device_map="auto" if self.device.type in ['cuda', 'mps'] else None
            )

            if self.device.type == 'cpu':
                self.eval_model = self.eval_model.to(self.device)

            self.eval_model.eval()  # Set to evaluation mode
            self.eval_config = config

            # Count parameters
            num_params = sum(p.numel() for p in self.eval_model.parameters())
            num_params_m = num_params / 1_000_000

            message = f"✓ Evaluation model loaded: {config.name}\n"
            message += f"  Parameters: {num_params_m:.1f}M\n"
            message += f"  Device: {self.device}\n"
            message += f"  Memory: ~{config.max_memory_gb:.1f}GB"

            return True, message

        except Exception as e:
            return False, f"✗ Failed to load evaluation model: {e}"

    def load_generation_model(
        self,
        model_key: str = "phi-2"
    ) -> Tuple[bool, str]:
        """
        Load model for generation and training tasks.

        Args:
            model_key: Key from RECOMMENDED_CONFIGS

        Returns:
            Tuple of (success, message)
        """
        if model_key not in RECOMMENDED_CONFIGS:
            return False, f"✗ Unknown model: {model_key}"

        config = RECOMMENDED_CONFIGS[model_key]

        try:
            print(f"Loading generation model: {config.name} ({config.hf_model_id})...")

            # Load tokenizer
            self.gen_tokenizer = AutoTokenizer.from_pretrained(
                config.hf_model_id,
                trust_remote_code=True
            )

            # Set padding token if not set
            if self.gen_tokenizer.pad_token is None:
                self.gen_tokenizer.pad_token = self.gen_tokenizer.eos_token

            # Load model
            self.gen_model = AutoModelForCausalLM.from_pretrained(
                config.hf_model_id,
                torch_dtype=torch.float16 if self.device.type != 'cpu' else torch.float32,
                trust_remote_code=True,
                device_map="auto" if self.device.type in ['cuda', 'mps'] else None
            )

            if self.device.type == 'cpu':
                self.gen_model = self.gen_model.to(self.device)

            self.gen_model.eval()  # Start in eval mode, switch to train later
            self.gen_config = config

            # Count parameters
            num_params = sum(p.numel() for p in self.gen_model.parameters())
            num_params_m = num_params / 1_000_000

            message = f"✓ Generation model loaded: {config.name}\n"
            message += f"  Parameters: {num_params_m:.1f}M\n"
            message += f"  Device: {self.device}\n"
            message += f"  Memory: ~{config.max_memory_gb:.1f}GB"

            return True, message

        except Exception as e:
            return False, f"✗ Failed to load generation model: {e}"

    def unload_evaluation_model(self) -> None:
        """Unload evaluation model to free memory."""
        if self.eval_model is not None:
            del self.eval_model
            del self.eval_tokenizer
            self.eval_model = None
            self.eval_tokenizer = None
            self.eval_config = None

            # Clear cache
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            if hasattr(torch, 'mps') and hasattr(torch.mps, 'empty_cache'):
                torch.mps.empty_cache()

    def unload_generation_model(self) -> None:
        """Unload generation model to free memory."""
        if self.gen_model is not None:
            del self.gen_model
            del self.gen_tokenizer
            self.gen_model = None
            self.gen_tokenizer = None
            self.gen_config = None

            # Clear cache
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            if hasattr(torch, 'mps') and hasattr(torch.mps, 'empty_cache'):
                torch.mps.empty_cache()

    def get_evaluation_model(self) -> Tuple[Optional[PreTrainedModel], Optional[PreTrainedTokenizer]]:
        """Get evaluation model and tokenizer."""
        return self.eval_model, self.eval_tokenizer

    def get_generation_model(self) -> Tuple[Optional[PreTrainedModel], Optional[PreTrainedTokenizer]]:
        """Get generation model and tokenizer."""
        return self.gen_model, self.gen_tokenizer

    def is_ready(self) -> bool:
        """Check if at least one model is loaded."""
        return self.eval_model is not None or self.gen_model is not None

    def get_status_info(self) -> Dict[str, Any]:
        """Get status information about loaded models."""
        info = {
            "device": str(self.device),
            "evaluation_model": None,
            "generation_model": None,
            "total_memory_gb": 0.0
        }

        if self.eval_model is not None and self.eval_config is not None:
            num_params = sum(p.numel() for p in self.eval_model.parameters())
            info["evaluation_model"] = {
                "name": self.eval_config.name,
                "parameters": num_params,
                "memory_gb": self.eval_config.max_memory_gb
            }
            info["total_memory_gb"] += self.eval_config.max_memory_gb

        if self.gen_model is not None and self.gen_config is not None:
            num_params = sum(p.numel() for p in self.gen_model.parameters())
            info["generation_model"] = {
                "name": self.gen_config.name,
                "parameters": num_params,
                "memory_gb": self.gen_config.max_memory_gb
            }
            info["total_memory_gb"] += self.gen_config.max_memory_gb

        return info
