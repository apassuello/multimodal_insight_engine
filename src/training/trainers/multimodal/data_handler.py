"""MODULE: data_handler.py
PURPOSE: Handles data preprocessing, device management, and batch preparation for training.

KEY COMPONENTS:
- DataHandler: Manages data flow between dataset, model, and loss function
- Handles device placement with nested data structures
- Prepares inputs for model forward pass
- Prepares inputs for loss computation
- Manages feature extraction and pooling

DEPENDENCIES:
- PyTorch (torch, torch.nn, torch.nn.functional)
- Python standard library (logging)

SPECIAL NOTES:
- Handles complex nested dictionaries for multimodal data
- Ensures device consistency across model components (critical for MPS)
- Supports flexible feature extraction with multiple naming conventions
- Provides comprehensive feature diagnostics
"""

import logging
from typing import Dict, Optional, Any, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


class DataHandler:
    """Manages data preprocessing and device management for training."""

    def __init__(
        self,
        model: nn.Module,
        device: torch.device,
        enable_diagnostics: bool = True,
    ):
        """
        Initialize the data handler.

        Args:
            model: The model being trained
            device: Target device for training
            enable_diagnostics: Whether to enable feature diagnostics
        """
        self.model = model
        self.device = device
        self.enable_diagnostics = enable_diagnostics

        # Tracking for logging
        self._log_once_per_epoch = True
        self._last_logged_epoch = -1
        self.current_epoch = 0

    def to_device(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        """
        Move batch to device and ensure consistent naming.

        Handles complex nested data structures with device consistency checks.

        Args:
            batch: Batch of data which may contain nested dictionaries

        Returns:
            Batch on device with consistent naming
        """
        # Check model device consistency
        model_device = next(self.model.parameters()).device
        if model_device != self.device:
            self.ensure_model_on_device()

        # Process batch recursively
        processed_batch = self._process_item_recursive(batch)

        # Normalize naming conventions
        processed_batch = self._normalize_batch_keys(processed_batch)

        return processed_batch

    def _process_item_recursive(self, item: Any) -> Any:
        """
        Recursively process batch items to move tensors to device.

        Args:
            item: Item to process (tensor, dict, list, or other)

        Returns:
            Processed item
        """
        if isinstance(item, torch.Tensor):
            if item.device == self.device:
                return item
            return item.to(self.device)
        elif isinstance(item, dict):
            return {k: self._process_item_recursive(v) for k, v in item.items()}
        elif isinstance(item, list):
            return [self._process_item_recursive(i) for i in item]
        else:
            return item

    def _normalize_batch_keys(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        """
        Normalize batch keys to match model expectations.

        Args:
            batch: Batch with potentially inconsistent keys

        Returns:
            Batch with normalized keys
        """
        # Model expects 'images' but dataset might return 'image'
        if "image" in batch and "images" not in batch:
            batch["images"] = batch.pop("image")

        # Model expects 'text_data' but dataset might return 'text'
        if "text" in batch and "text_data" not in batch:
            batch["text_data"] = batch.pop("text")

        # Special handling for nested text data
        if "text_data" in batch and isinstance(batch["text_data"], dict):
            text_data = batch["text_data"]
            for k, v in text_data.items():
                if isinstance(v, torch.Tensor) and v.device != self.device:
                    text_data[k] = v.to(self.device)

        return batch

    def prepare_model_inputs(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract only the inputs needed by the model's forward method.

        Args:
            batch: Full batch of data

        Returns:
            Dictionary with only the inputs the model expects
        """
        model_inputs = {}

        # Add images if available
        if "images" in batch:
            if batch["images"].device != self.device:
                batch["images"] = batch["images"].to(self.device)
                logger.debug(f"Moved images to {self.device}")
            model_inputs["images"] = batch["images"]

        # Add text_data if available
        if "text_data" in batch:
            model_inputs["text_data"] = self._prepare_text_data(batch["text_data"])

        return model_inputs

    def _prepare_text_data(self, text_data: Any) -> Any:
        """
        Prepare text data with device consistency checks.

        Args:
            text_data: Text data (could be dict or tensor)

        Returns:
            Text data on correct device
        """
        if not isinstance(text_data, dict):
            return text_data

        # Check if any tensors need device fix
        needs_device_fix = any(
            isinstance(v, torch.Tensor) and v.device != self.device
            for v in text_data.values()
        )

        if needs_device_fix:
            text_data_fixed = {}
            for k, v in text_data.items():
                if isinstance(v, torch.Tensor) and v.device != self.device:
                    text_data_fixed[k] = v.to(self.device)
                    logger.debug(f"Moved text_data[{k}] to {self.device}")
                else:
                    text_data_fixed[k] = v
            return text_data_fixed

        return text_data

    def prepare_loss_inputs(
        self, batch: Dict[str, Any], outputs: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """
        Prepare inputs for loss computation from batch and model outputs.

        Extracts features with fallback handling and provides diagnostics.

        Args:
            batch: Original batch data
            outputs: Model outputs

        Returns:
            Dictionary with loss function inputs
        """
        # Check if we should log for this epoch
        should_log = (
            self.current_epoch != self._last_logged_epoch
            and self._log_once_per_epoch
            and self.enable_diagnostics
        )

        if should_log:
            logger.info(f"Model output keys: {sorted(outputs.keys())}")
            logger.info(f"Batch keys: {sorted(batch.keys())}")
            self._last_logged_epoch = self.current_epoch

        # Extract features
        vision_features, text_features, feature_source = self._extract_features(outputs)

        # Check if extraction succeeded
        if vision_features is None or text_features is None:
            logger.error("CRITICAL: Could not find valid feature outputs!")
            logger.error(f"Available keys: {sorted(outputs.keys())}")
            # Emergency fallback
            batch_size = batch.get("images", batch.get("text_data", torch.zeros(1))).shape[0]
            vision_features = torch.zeros((batch_size, 768), device=self.device)
            text_features = torch.zeros((batch_size, 768), device=self.device)
            feature_source = "emergency_fallback"

        # Run diagnostics
        if self.enable_diagnostics and should_log:
            self._diagnose_features(vision_features, text_features, feature_source)

        # Normalize features
        vision_features = F.normalize(vision_features, p=2, dim=1)
        text_features = F.normalize(text_features, p=2, dim=1)

        # Verify normalization
        if self.enable_diagnostics and should_log:
            self._verify_normalization(vision_features, text_features)

        # Prepare loss inputs
        loss_inputs = {
            "vision_features": vision_features,
            "text_features": text_features,
        }

        # Add any additional outputs the loss might need
        for key in ["logits", "labels", "attention_weights"]:
            if key in outputs:
                loss_inputs[key] = outputs[key]

        return loss_inputs

    def _extract_features(
        self, outputs: Dict[str, torch.Tensor]
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], str]:
        """
        Extract features from model outputs with fallback.

        Args:
            outputs: Model outputs dictionary

        Returns:
            Tuple of (vision_features, text_features, source_name)
        """
        # Priority order for feature extraction
        feature_configs = [
            ("vision_features_enhanced", "text_features_enhanced", "enhanced_features"),
            ("vision_features", "text_features", "base_features"),
            ("image_features", "text_features", "image_text_features"),
            ("embedded_images", "embedded_text", "embedded_features"),
        ]

        for vision_key, text_key, source in feature_configs:
            if vision_key in outputs and text_key in outputs:
                vision_features = self.get_pooled_features(outputs[vision_key])
                text_features = self.get_pooled_features(outputs[text_key])
                return vision_features, text_features, source

        return None, None, "not_found"

    def get_pooled_features(self, features: torch.Tensor) -> torch.Tensor:
        """
        Get pooled features from sequence features if needed.

        Args:
            features: Features tensor (2D or 3D)

        Returns:
            Pooled features tensor (2D)
        """
        # Already pooled (2D)
        if len(features.shape) == 2:
            return features

        # Sequence features (3D) - use mean pooling
        if len(features.shape) == 3:
            return features.mean(dim=1)

        raise ValueError(f"Unsupported features shape: {features.shape}")

    def _diagnose_features(
        self,
        vision_features: torch.Tensor,
        text_features: torch.Tensor,
        feature_source: str,
    ) -> None:
        """
        Run diagnostic checks on extracted features.

        Args:
            vision_features: Vision embeddings
            text_features: Text embeddings
            feature_source: Source of features for logging
        """
        logger.info(f"Using features from: {feature_source}")
        logger.info(f"Vision shape: {vision_features.shape}, Text shape: {text_features.shape}")

        # Check for NaN/Inf
        vision_has_nan = torch.isnan(vision_features).any().item()
        text_has_nan = torch.isnan(text_features).any().item()
        vision_has_inf = torch.isinf(vision_features).any().item()
        text_has_inf = torch.isinf(text_features).any().item()

        if any([vision_has_nan, text_has_nan, vision_has_inf, text_has_inf]):
            logger.error(
                f"INVALID VALUES: Vision NaN={vision_has_nan}, Inf={vision_has_inf}; "
                f"Text NaN={text_has_nan}, Inf={text_has_inf}"
            )

        # Feature statistics
        vision_stats = self._compute_feature_stats(vision_features)
        text_stats = self._compute_feature_stats(text_features)

        logger.info(
            f"Vision stats: mean={vision_stats['mean']:.6f}, std={vision_stats['std']:.6f}, "
            f"range=[{vision_stats['min']:.6f}, {vision_stats['max']:.6f}]"
        )
        logger.info(
            f"Text stats: mean={text_stats['mean']:.6f}, std={text_stats['std']:.6f}, "
            f"range=[{text_stats['min']:.6f}, {text_stats['max']:.6f}]"
        )

        # Check for feature collapse
        if vision_stats['std'] < 1e-4:
            logger.error(f"VISION FEATURE COLLAPSE! std={vision_stats['std']:.8f}")
        if text_stats['std'] < 1e-4:
            logger.error(f"TEXT FEATURE COLLAPSE! std={text_stats['std']:.8f}")

        # Check dimension collapse
        vision_dim_var = torch.var(vision_features, dim=0)
        text_dim_var = torch.var(text_features, dim=0)
        low_var_vision = (vision_dim_var < 1e-6).sum().item()
        low_var_text = (text_dim_var < 1e-6).sum().item()

        if low_var_vision > 0 or low_var_text > 0:
            logger.warning(
                f"Dimension collapse: Vision {low_var_vision}/{vision_features.shape[1]} dims, "
                f"Text {low_var_text}/{text_features.shape[1]} dims"
            )

    def _compute_feature_stats(self, features: torch.Tensor) -> Dict[str, float]:
        """Compute statistics for features."""
        return {
            "mean": features.mean().item(),
            "std": features.std().item(),
            "min": features.min().item(),
            "max": features.max().item(),
        }

    def _verify_normalization(
        self, vision_features: torch.Tensor, text_features: torch.Tensor
    ) -> None:
        """
        Verify that features are properly normalized.

        Args:
            vision_features: Normalized vision features
            text_features: Normalized text features
        """
        vision_norms = torch.norm(vision_features, dim=1)
        text_norms = torch.norm(text_features, dim=1)

        vision_norm_mean = vision_norms.mean().item()
        text_norm_mean = text_norms.mean().item()

        logger.info(f"After normalization - Vision norm: {vision_norm_mean:.6f}, Text norm: {text_norm_mean:.6f}")

        norm_threshold = 0.01
        if abs(vision_norm_mean - 1.0) > norm_threshold:
            logger.error(f"NORMALIZATION ERROR! Vision norm={vision_norm_mean:.6f}")
        if abs(text_norm_mean - 1.0) > norm_threshold:
            logger.error(f"NORMALIZATION ERROR! Text norm={text_norm_mean:.6f}")

    def ensure_model_on_device(self) -> None:
        """
        Ensure all model components are on the correct device.

        Critical for multimodal models with multiple components, especially on MPS.
        """
        if not isinstance(self.model, nn.Module):
            logger.warning("Model is not nn.Module, cannot ensure device consistency")
            return

        # Move specific multimodal components
        for attr in ["vision_model", "text_model", "fusion_module"]:
            if hasattr(self.model, attr):
                module = getattr(self.model, attr)
                self._move_module_to_device(attr, module)

        # Move all child modules
        for name, module in self.model.named_children():
            self._move_module_to_device(name, module)

        # Verify all parameters
        self._verify_parameters_on_device()

    def _move_module_to_device(self, name: str, module: nn.Module) -> None:
        """Move a module to the target device if needed."""
        if not list(module.parameters()):
            return

        param_device = next(module.parameters()).device
        if param_device != self.device:
            try:
                module.to(self.device)
            except Exception as e:
                logger.error(f"Error moving {name} to {self.device}: {str(e)}")

    def _verify_parameters_on_device(self) -> None:
        """Verify all model parameters are on the correct device."""
        for name, param in self.model.named_parameters():
            if param.device != self.device:
                logger.warning(f"Parameter {name} on {param.device}, expected {self.device}")

    def set_current_epoch(self, epoch: int) -> None:
        """Set current epoch for logging purposes."""
        self.current_epoch = epoch
