"""MODULE: model_manager.py
PURPOSE: Model lifecycle management for Constitutional AI demo
KEY COMPONENTS:
- ModelManager: Handles model loading, caching, and state management
- Device detection with MPS → CUDA → CPU fallback
- Checkpoint save/load for before/after comparison
- Memory management and cleanup
DEPENDENCIES: torch, transformers, pathlib, typing, src.safety.constitutional.model_utils
SPECIAL NOTES: Supports M4-Pro MPS acceleration with graceful fallback
"""

import os
import gc
import torch
from pathlib import Path
from typing import Optional, Tuple, Dict, Any
from enum import Enum

from src.safety.constitutional.model_utils import load_model


class ModelStatus(Enum):
    """Model status states."""
    NOT_LOADED = "not_loaded"
    LOADING = "loading"
    READY = "ready"
    TRAINING = "training"
    ERROR = "error"


class ModelManager:
    """
    Manages model lifecycle including loading, caching, and checkpointing.

    Handles device detection (MPS → CUDA → CPU) and maintains separate
    base and trained model checkpoints for before/after comparison.
    """

    def __init__(self, checkpoint_dir: str = "demo/checkpoints"):
        """
        Initialize model manager.

        Args:
            checkpoint_dir: Directory for saving model checkpoints
        """
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Model state
        self.model = None
        self.tokenizer = None
        self.model_name: Optional[str] = None
        self.device: Optional[torch.device] = None
        self.status = ModelStatus.NOT_LOADED

        # Checkpoint paths
        self.base_checkpoint_path: Optional[Path] = None
        self.trained_checkpoint_path: Optional[Path] = None

        # Cached models for comparison
        self.base_model = None
        self.base_tokenizer = None

        # Separate evaluation model (optional - uses generation model if not set)
        self.eval_model = None
        self.eval_tokenizer = None
        self.eval_model_name: Optional[str] = None

    def load_evaluation_model(
        self,
        model_name: str,
        prefer_device: Optional[str] = None
    ) -> Tuple[bool, str]:
        """
        Load a separate model for evaluation.

        This allows using a different (potentially larger/more capable) model
        for evaluating safety violations while using a smaller model for generation.

        Args:
            model_name: HuggingFace model identifier (e.g., "microsoft/phi-2", "Qwen/Qwen2.5-1.5B")
            prefer_device: Optional device preference

        Returns:
            Tuple of (success: bool, message: str)
        """
        try:
            device = self.detect_device(prefer_device)
            print(f"Loading evaluation model: {model_name} on {device}")

            self.eval_model, self.eval_tokenizer = load_model(
                model_name=model_name,
                device=device
            )
            self.eval_model_name = model_name

            param_count = sum(p.numel() for p in self.eval_model.parameters())
            message = f"✓ Evaluation model '{model_name}' loaded on {device}\n"
            message += f"Parameters: {param_count:,}"

            return True, message

        except Exception as e:
            error_msg = f"✗ Failed to load evaluation model '{model_name}': {str(e)}"
            return False, error_msg

    def get_evaluation_model(self) -> Tuple[Any, Any, Optional[str]]:
        """
        Get the evaluation model (or generation model if no separate eval model).

        Returns:
            Tuple of (model, tokenizer, model_name)
        """
        if self.eval_model is not None:
            return self.eval_model, self.eval_tokenizer, self.eval_model_name
        return self.model, self.tokenizer, self.model_name

    def has_separate_eval_model(self) -> bool:
        """Check if a separate evaluation model is loaded."""
        return self.eval_model is not None

    def detect_device(self, prefer_device: Optional[str] = None) -> torch.device:
        """
        Detect available device with preference order: MPS → CUDA → CPU.

        Args:
            prefer_device: Optional device preference ("mps", "cuda", "cpu")

        Returns:
            torch.device object for the selected device
        """
        # If specific device requested, try it first
        if prefer_device:
            if prefer_device == "mps" and torch.backends.mps.is_available():
                return torch.device("mps")
            elif prefer_device == "cuda" and torch.cuda.is_available():
                return torch.device("cuda")
            elif prefer_device == "cpu":
                return torch.device("cpu")

        # Auto-detect with fallback order
        if torch.backends.mps.is_available():
            return torch.device("mps")
        elif torch.cuda.is_available():
            return torch.device("cuda")
        else:
            return torch.device("cpu")

    def load_model_from_pretrained(
        self,
        model_name: str = "gpt2",
        prefer_device: Optional[str] = None
    ) -> Tuple[bool, str]:
        """
        Load a pretrained model from Hugging Face.

        Args:
            model_name: Model identifier (e.g., "gpt2", "gpt2-medium", "distilgpt2")
            prefer_device: Optional device preference

        Returns:
            Tuple of (success: bool, message: str)
        """
        try:
            self.status = ModelStatus.LOADING

            # Detect device
            self.device = self.detect_device(prefer_device)
            device_name = str(self.device).upper()

            # Load model
            self.model, self.tokenizer = load_model(
                model_name=model_name,
                device=self.device
            )

            self.model_name = model_name
            self.status = ModelStatus.READY

            # Save base checkpoint immediately after loading
            base_checkpoint_name = f"base_{model_name.replace('/', '_')}"
            self.base_checkpoint_path = self.checkpoint_dir / base_checkpoint_name
            self.save_checkpoint(
                self.model,
                self.tokenizer,
                self.base_checkpoint_path,
                metadata={"type": "base", "model_name": model_name}
            )

            param_count = sum(p.numel() for p in self.model.parameters())
            message = f"✓ Model '{model_name}' loaded successfully on {device_name}\n"
            message += f"Parameters: {param_count:,}\n"
            message += f"Base checkpoint saved to: {self.base_checkpoint_path}"

            return True, message

        except Exception as e:
            self.status = ModelStatus.ERROR
            error_msg = f"✗ Failed to load model '{model_name}': {str(e)}"

            # Try CPU fallback if not already on CPU
            if self.device and self.device.type != "cpu":
                try:
                    self.device = torch.device("cpu")
                    self.model, self.tokenizer = load_model(
                        model_name=model_name,
                        device=self.device
                    )
                    self.model_name = model_name
                    self.status = ModelStatus.READY

                    # Save base checkpoint
                    base_checkpoint_name = f"base_{model_name.replace('/', '_')}"
                    self.base_checkpoint_path = self.checkpoint_dir / base_checkpoint_name
                    self.save_checkpoint(
                        self.model,
                        self.tokenizer,
                        self.base_checkpoint_path,
                        metadata={"type": "base", "model_name": model_name}
                    )

                    return True, f"✓ Loaded on CPU (fallback after error)\n{error_msg}"
                except Exception as fallback_error:
                    return False, f"{error_msg}\nCPU fallback also failed: {fallback_error}"

            return False, error_msg

    def save_checkpoint(
        self,
        model,
        tokenizer,
        checkpoint_path: Path,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Tuple[bool, str]:
        """
        Save model and tokenizer checkpoint.

        Args:
            model: Model to save
            tokenizer: Tokenizer to save
            checkpoint_path: Path to save checkpoint
            metadata: Optional metadata to save with checkpoint

        Returns:
            Tuple of (success: bool, message: str)
        """
        try:
            checkpoint_path.mkdir(parents=True, exist_ok=True)

            # Save model
            model.save_pretrained(checkpoint_path)

            # Save tokenizer
            tokenizer.save_pretrained(checkpoint_path)

            # Save metadata if provided
            if metadata:
                import json
                metadata_path = checkpoint_path / "metadata.json"
                with open(metadata_path, 'w') as f:
                    json.dump(metadata, f, indent=2)

            return True, f"✓ Checkpoint saved to {checkpoint_path}"

        except Exception as e:
            return False, f"✗ Failed to save checkpoint: {str(e)}"

    def load_checkpoint(
        self,
        checkpoint_path: Path,
        device: Optional[torch.device] = None
    ) -> Tuple[Optional[Any], Optional[Any], bool, str]:
        """
        Load model and tokenizer from checkpoint.

        Args:
            checkpoint_path: Path to checkpoint directory
            device: Optional device to load model onto (defaults to self.device or CPU)

        Returns:
            Tuple of (model, tokenizer, success: bool, message: str)
        """
        try:
            if not checkpoint_path.exists():
                return None, None, False, f"✗ Checkpoint not found: {checkpoint_path}"

            from transformers import AutoModelForCausalLM, AutoTokenizer

            # Load tokenizer
            tokenizer = AutoTokenizer.from_pretrained(checkpoint_path)

            # Load model
            model = AutoModelForCausalLM.from_pretrained(checkpoint_path)

            # Move to device (use provided device, or self.device, or default to CPU)
            target_device = device or self.device or torch.device("cpu")
            model = model.to(target_device)

            return model, tokenizer, True, f"✓ Checkpoint loaded from {checkpoint_path}"

        except Exception as e:
            return None, None, False, f"✗ Failed to load checkpoint: {str(e)}"

    def load_base_model_for_comparison(self) -> Tuple[bool, str]:
        """
        Load base model checkpoint for before/after comparison.

        Returns:
            Tuple of (success: bool, message: str)
        """
        if not self.base_checkpoint_path or not self.base_checkpoint_path.exists():
            return False, "✗ No base checkpoint available"

        try:
            self.base_model, self.base_tokenizer, success, msg = self.load_checkpoint(
                self.base_checkpoint_path
            )

            if success:
                return True, f"✓ Base model loaded for comparison\n{msg}"
            else:
                return False, msg

        except Exception as e:
            return False, f"✗ Failed to load base model: {str(e)}"

    def save_trained_checkpoint(
        self,
        epoch: Optional[int] = None,
        metrics: Optional[Dict[str, Any]] = None
    ) -> Tuple[bool, str]:
        """
        Save current model as trained checkpoint.

        Args:
            epoch: Optional epoch number for checkpoint naming
            metrics: Optional training metrics to save with checkpoint

        Returns:
            Tuple of (success: bool, message: str)
        """
        if not self.model or not self.tokenizer:
            return False, "✗ No model loaded to save"

        try:
            # Create checkpoint name
            if epoch is not None:
                checkpoint_name = f"trained_{self.model_name.replace('/', '_')}_epoch{epoch}"
            else:
                checkpoint_name = f"trained_{self.model_name.replace('/', '_')}"

            self.trained_checkpoint_path = self.checkpoint_dir / checkpoint_name

            # Prepare metadata
            metadata = {
                "type": "trained",
                "model_name": self.model_name,
                "base_checkpoint": str(self.base_checkpoint_path),
            }

            if epoch is not None:
                metadata["epoch"] = epoch

            if metrics:
                metadata["metrics"] = metrics

            # Save checkpoint
            success, msg = self.save_checkpoint(
                self.model,
                self.tokenizer,
                self.trained_checkpoint_path,
                metadata=metadata
            )

            if success:
                return True, f"✓ Trained checkpoint saved\n{msg}"
            else:
                return False, msg

        except Exception as e:
            return False, f"✗ Failed to save trained checkpoint: {str(e)}"

    def unload_model(self) -> Tuple[bool, str]:
        """
        Unload current model and free memory.

        Returns:
            Tuple of (success: bool, message: str)
        """
        try:
            if self.model:
                del self.model
                self.model = None

            if self.tokenizer:
                del self.tokenizer
                self.tokenizer = None

            if self.base_model:
                del self.base_model
                self.base_model = None

            if self.base_tokenizer:
                del self.base_tokenizer
                self.base_tokenizer = None

            # Clear GPU/MPS cache
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            elif torch.backends.mps.is_available():
                # MPS doesn't have a direct cache clearing method
                # but garbage collection helps
                pass

            # Force garbage collection
            gc.collect()

            self.status = ModelStatus.NOT_LOADED
            self.model_name = None

            return True, "✓ Model unloaded and memory freed"

        except Exception as e:
            return False, f"✗ Failed to unload model: {str(e)}"

    def get_status_info(self) -> Dict[str, Any]:
        """
        Get current model manager status information.

        Returns:
            Dictionary with status information
        """
        info = {
            "status": self.status.value,
            "model_name": self.model_name,
            "device": str(self.device) if self.device else "none",
            "model_loaded": self.model is not None,
            "base_checkpoint": str(self.base_checkpoint_path) if self.base_checkpoint_path else "none",
            "trained_checkpoint": str(self.trained_checkpoint_path) if self.trained_checkpoint_path else "none",
            "base_model_loaded": self.base_model is not None,
        }

        # Add parameter count if model is loaded
        if self.model:
            info["parameters"] = sum(p.numel() for p in self.model.parameters())

        return info

    def set_status(self, status: ModelStatus) -> None:
        """Set model status."""
        self.status = status

    def is_ready(self) -> bool:
        """Check if model is ready for use."""
        return self.status == ModelStatus.READY and self.model is not None

    def can_compare(self) -> bool:
        """Check if both base and trained models are available for comparison."""
        return (
            self.base_checkpoint_path is not None
            and self.base_checkpoint_path.exists()
            and self.trained_checkpoint_path is not None
            and self.trained_checkpoint_path.exists()
        )
