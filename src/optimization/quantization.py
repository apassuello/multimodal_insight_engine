"""MODULE: quantization.py
PURPOSE: Implements various quantization techniques for neural networks to optimize model size and performance.
KEY COMPONENTS:
- QuantizationConfig: Configuration class for model quantization settings.
- ModelOptimizer: Base class for model optimization techniques.
- DynamicQuantizer: Implements dynamic quantization for PyTorch models.
- StaticQuantizer: Implements static quantization for PyTorch models.
DEPENDENCIES: torch, typing, logging
SPECIAL NOTES: Supports dynamic, static, and quantization-aware training methods."""

import os
from typing import Any, Dict, List, Optional, Tuple, Type

import torch
import torch.nn as nn
import torch.quantization as quantization
from torch.nn.intrinsic import ConvBn2d, ConvBnReLU2d, LinearBn1d, LinearReLU
from torch.quantization import MinMaxObserver, default_observer


class QuantizationConfig:
    """
    Configuration class for model quantization settings.
    
    This class centralizes the parameters for different quantization approaches,
    making it easier to experiment with various configurations.
    """

    def __init__(
        self,
        quantization_type: str = "dynamic",  # "dynamic", "static", or "qat" (quantization-aware training)
        dtype: Optional[torch.dtype] = None,  # torch.qint8, torch.float16, etc.
        quantize_weights: bool = True,
        quantize_activations: bool = True,
        bits: int = 8,  # 8-bit or 16-bit quantization
        symmetric: bool = False,  # Symmetric or asymmetric quantization
        per_channel: bool = False,  # Per-channel or per-tensor quantization
    ):
        """
        Initialize quantization configuration.
        
        Args:
            quantization_type: Type of quantization to apply
            dtype: Target data type for quantization (if None, inferred from bits)
            quantize_weights: Whether to quantize model weights
            quantize_activations: Whether to quantize activations
            bits: Bit width for quantization (8 or 16)
            symmetric: Whether to use symmetric quantization
            per_channel: Whether to use per-channel quantization for weights
        """
        self.quantization_type = quantization_type
        self.quantize_weights = quantize_weights
        self.quantize_activations = quantize_activations
        self.bits = bits
        self.symmetric = symmetric
        self.per_channel = per_channel

        # Set dtype based on bits if not provided
        if dtype is None:
            if bits == 8:
                self.dtype = torch.qint8
            elif bits == 16:
                self.dtype = torch.float16
            else:
                raise ValueError(f"Unsupported bit width: {bits}. Use 8 or 16.")
        else:
            self.dtype = dtype

    def __str__(self) -> str:
        """String representation of the configuration."""
        return (
            f"QuantizationConfig(type={self.quantization_type}, "
            f"dtype={self.dtype}, bits={self.bits}, "
            f"weights={self.quantize_weights}, activations={self.quantize_activations}, "
            f"symmetric={self.symmetric}, per_channel={self.per_channel})"
        )


class ModelOptimizer:
    """
    Base class for model optimization techniques.
    
    This provides a common interface for all optimization methods
    (quantization, pruning, distillation, etc.).
    """

    def __init__(self, model: nn.Module):
        """
        Initialize the model optimizer.
        
        Args:
            model: The model to optimize
        """
        self.model = model
        self.original_state_dict = None
        self._save_original_state()

    def _save_original_state(self):
        """Save the original model state for potential restoration."""
        self.original_state_dict = {
            k: v.clone() for k, v in self.model.state_dict().items()
        }

    def optimize(self) -> nn.Module:
        """
        Apply optimization to the model.
        
        Returns:
            Optimized model
        """
        raise NotImplementedError("Subclasses must implement optimize method")

    def restore_original(self):
        """Restore the model to its original state."""
        if self.original_state_dict is not None:
            self.model.load_state_dict(self.original_state_dict)

    def get_size_info(self) -> Dict[str, Any]:
        """
        Get information about model size before and after optimization.
        
        Returns:
            Dictionary with size information
        """
        raise NotImplementedError("Subclasses must implement get_size_info method")


class DynamicQuantizer(ModelOptimizer):
    """
    Implements dynamic quantization for PyTorch models.
    
    Dynamic quantization quantizes weights to int8 while keeping activations in
    floating point. Weights are quantized ahead of time but activations are
    quantized during inference, based on observed activation ranges.
    """

    def __init__(
        self,
        model: nn.Module,
        config: Optional[QuantizationConfig] = None,
        dtype: torch.dtype = torch.qint8,
        qconfig_spec: Optional[Dict[Type[nn.Module], Any]] = None
    ):
        """
        Initialize the dynamic quantizer.
        
        Args:
            model: The model to quantize
            config: Quantization configuration
            dtype: Target data type for quantization (default: torch.qint8)
            qconfig_spec: Optional specification of which modules to quantize
        """
        super().__init__(model)
        self.config = config or QuantizationConfig(quantization_type="dynamic", dtype=dtype)
        self.qconfig_spec = qconfig_spec or {
            nn.Linear: torch.quantization.default_dynamic_qconfig,
            nn.LSTM: torch.quantization.default_dynamic_qconfig,
            nn.GRU: torch.quantization.default_dynamic_qconfig,
        }
        self.quantized_model = None

    def optimize(self) -> nn.Module:
        """
        Apply dynamic quantization to the model.
        
        Returns:
            Quantized model
        """
        # Create a copy of the model for quantization
        model_to_quantize = type(self.model)()
        model_to_quantize.load_state_dict(self.model.state_dict())

        # Fuse modules if applicable
        model_to_quantize = self._fuse_modules(model_to_quantize)

        # Convert to quantized model
        quantized_model = torch.quantization.quantize_dynamic(
            model_to_quantize,
            qconfig_spec=self.qconfig_spec,
            dtype=self.config.dtype
        )

        self.quantized_model = quantized_model
        return self.quantized_model

    def _fuse_modules(self, model: nn.Module) -> nn.Module:
        """
        Fuse modules for improved quantization where applicable.
        
        This method identifies and fuses common module sequences that can be
        quantized more efficiently together, such as:
        - Conv2d + BatchNorm2d + ReLU
        - Linear + ReLU
        - Conv2d + BatchNorm2d
        - Linear + BatchNorm1d
        
        Args:
            model: Model to fuse modules in
            
        Returns:
            Model with fused modules
        """
        # Define common fusion patterns
        fusion_patterns: List[Tuple[Type[nn.Module], ...]] = [
            # Conv2d + BatchNorm2d + ReLU
            (nn.Conv2d, nn.BatchNorm2d, nn.ReLU),
            # Conv2d + BatchNorm2d
            (nn.Conv2d, nn.BatchNorm2d),
            # Linear + ReLU
            (nn.Linear, nn.ReLU),
            # Linear + BatchNorm1d
            (nn.Linear, nn.BatchNorm1d),
        ]

        # Helper function to check if a sequence of modules matches a pattern
        def matches_pattern(modules: List[nn.Module], pattern: Tuple[Type[nn.Module], ...]) -> bool:
            if len(modules) < len(pattern):
                return False
            return all(isinstance(m, p) for m, p in zip(modules, pattern))

        # Helper function to fuse a sequence of modules
        def fuse_sequence(modules: List[nn.Module], pattern: Tuple[Type[nn.Module], ...]) -> Optional[nn.Module]:
            if pattern == (nn.Conv2d, nn.BatchNorm2d, nn.ReLU):
                return ConvBnReLU2d(
                    modules[0],  # Conv2d
                    modules[1],  # BatchNorm2d
                    modules[2]   # ReLU
                )
            elif pattern == (nn.Conv2d, nn.BatchNorm2d):
                return ConvBn2d(
                    modules[0],  # Conv2d
                    modules[1]   # BatchNorm2d
                )
            elif pattern == (nn.Linear, nn.ReLU):
                return LinearReLU(
                    modules[0],  # Linear
                    modules[1]   # ReLU
                )
            elif pattern == (nn.Linear, nn.BatchNorm1d):
                return LinearBn1d(
                    modules[0],  # Linear
                    modules[1]   # BatchNorm1d
                )
            return None

        # Process each module in the model
        for name, module in model.named_modules():
            if not isinstance(module, (nn.Conv2d, nn.Linear)):
                continue

            # Get the parent module
            parent_name = '.'.join(name.split('.')[:-1])
            if not parent_name:
                continue

            parent = model
            for part in parent_name.split('.'):
                parent = getattr(parent, part)

            # Get the sequence of modules starting from the current module
            current = module
            sequence: List[nn.Module] = [current]

            # Try to build sequences that match our patterns
            for pattern in fusion_patterns:
                if len(sequence) >= len(pattern):
                    continue

                next_module = None
                if hasattr(parent, name.split('.')[-1]):
                    next_module = getattr(parent, name.split('.')[-1])

                if next_module is not None:
                    sequence.append(next_module)

                    if matches_pattern(sequence, pattern):
                        # Create the fused module
                        fused_module = fuse_sequence(sequence, pattern)
                        if fused_module is not None:
                            # Replace the sequence with the fused module
                            setattr(parent, name.split('.')[-1], fused_module)
                            break

        return model

    def get_size_info(self) -> Dict[str, Any]:
        """
        Get information about model size before and after quantization.
        
        Returns:
            Dictionary with size information
        """
        if self.quantized_model is None:
            return {"error": "Model has not been quantized yet. Call optimize() first."}

        # Get size information
        original_size = sum(p.numel() * p.element_size() for p in self.model.parameters())
        quantized_size = sum(
            p.numel() * (1 if p.dtype == torch.qint8 else p.element_size())
            for p in self.quantized_model.parameters()
        )

        return {
            "original_size_mb": original_size / (1024 * 1024),
            "quantized_size_mb": quantized_size / (1024 * 1024),
            "compression_ratio": original_size / quantized_size if quantized_size > 0 else float('inf'),
            "dtype": self.config.dtype,
        }


class StaticQuantizer(ModelOptimizer):
    """
    Implements static quantization for PyTorch models.
    
    Static quantization quantizes both weights and activations to int8 based on
    calibration data. It requires a representative dataset for calibration.
    """

    def __init__(
        self,
        model: nn.Module,
        config: Optional[QuantizationConfig] = None,
        calibration_loader: Optional[torch.utils.data.DataLoader] = None,
    ):
        """
        Initialize the static quantizer.
        
        Args:
            model: The model to quantize
            config: Quantization configuration
            calibration_loader: DataLoader providing calibration data
        """
        super().__init__(model)
        self.config = config or QuantizationConfig(quantization_type="static", dtype=torch.qint8)
        self.calibration_loader = calibration_loader
        self.quantized_model = None

        # Set backend for static quantization
        self.backend = "fbgemm" if not torch.backends.mps.is_available() else "qnnpack"

    def optimize(self) -> nn.Module:
        """
        Apply static quantization to the model.
        
        Returns:
            Quantized model
        """
        if self.calibration_loader is None:
            raise ValueError("Static quantization requires calibration data")

        # Create a copy of the model for quantization
        model_to_quantize = type(self.model)()
        model_to_quantize.load_state_dict(self.model.state_dict())

        # Ensure model is in eval mode
        model_to_quantize.eval()

        # Fuse modules if applicable
        model_to_quantize = self._fuse_modules(model_to_quantize)

        # Set the qconfig for the model
        if self.config.per_channel:
            qconfig = torch.quantization.get_default_qconfig(self.backend)
        else:
            qconfig = quantization.QConfig(
                activation=default_observer,
                weight=MinMaxObserver.with_args(dtype=torch.qint8)
            )

        # Set the quantization configuration
        model_to_quantize.qconfig = qconfig  # type: ignore

        # Prepare the model for static quantization
        model_prepared = torch.quantization.prepare(model_to_quantize)

        # Calibrate the model using the calibration data
        self._calibrate_model(model_prepared)

        # Convert the model to a quantized version
        quantized_model = torch.quantization.convert(model_prepared)

        self.quantized_model = quantized_model
        return self.quantized_model

    def _calibrate_model(self, model: nn.Module):
        """
        Calibrate the model using the calibration dataset.
        
        Args:
            model: The model to calibrate
        """
        if self.calibration_loader is None:
            return

        # Run the calibration data through the model to collect activation statistics
        with torch.no_grad():
            for batch_idx, batch in enumerate(self.calibration_loader):
                # Handle different types of inputs
                if isinstance(batch, dict):
                    # Assume the batch contains input_ids at minimum
                    inputs = batch["input_ids"]
                    # Add more input handling based on your model's requirements
                elif isinstance(batch, (list, tuple)):
                    inputs = batch[0]
                else:
                    inputs = batch

                # Forward pass to record activation stats
                model(inputs)

                # Limit calibration to a reasonable number of batches
                if batch_idx >= 100:
                    break

    def _fuse_modules(self, model: nn.Module, fusion_patterns: Optional[List[Tuple[Type[nn.Module], ...]]] = None) -> nn.Module:
        """
        Fuse modules for improved quantization.
        
        Args:
            model: Model to fuse modules in
            fusion_patterns: Optional custom fusion patterns
            
        Returns:
            Model with fused modules
        """
        # Default fusion patterns for transformers
        default_fusion_patterns: List[Tuple[Type[nn.Module], ...]] = [
            (nn.Linear, nn.LayerNorm),  # Linear + LayerNorm
            (nn.Linear, nn.ReLU),       # Linear + ReLU
            (nn.Linear, nn.GELU),       # Linear + GELU
        ]

        # Use custom patterns if provided
        fusion_patterns = fusion_patterns or default_fusion_patterns

        # Helper function to check if a sequence of modules matches a pattern
        def matches_pattern(modules: List[nn.Module], pattern: Tuple[Type[nn.Module], ...]) -> bool:
            if len(modules) < len(pattern):
                return False
            return all(isinstance(m, p) for m, p in zip(modules, pattern))

        # Helper function to fuse a sequence of modules
        def fuse_sequence(modules: List[nn.Module], pattern: Tuple[Type[nn.Module], ...]) -> Optional[nn.Module]:
            if pattern == (nn.Linear, nn.LayerNorm):
                # Custom fusion logic for Linear + LayerNorm
                return nn.Sequential(modules[0], modules[1])
            elif pattern == (nn.Linear, nn.ReLU):
                return LinearReLU(modules[0], modules[1])
            elif pattern == (nn.Linear, nn.GELU):
                return nn.Sequential(modules[0], modules[1])
            return None

        # Process each module in the model
        for name, module in model.named_modules():
            if not isinstance(module, nn.Linear):
                continue

            # Get the parent module
            parent_name = '.'.join(name.split('.')[:-1])
            if not parent_name:
                continue

            parent = model
            for part in parent_name.split('.'):
                parent = getattr(parent, part)

            # Get the sequence of modules starting from the current module
            current = module
            sequence: List[nn.Module] = [current]

            # Try to build sequences that match our patterns
            for pattern in fusion_patterns:
                if len(sequence) >= len(pattern):
                    continue

                next_module = None
                if hasattr(parent, name.split('.')[-1]):
                    next_module = getattr(parent, name.split('.')[-1])

                if next_module is not None:
                    sequence.append(next_module)

                    if matches_pattern(sequence, pattern):
                        # Create the fused module
                        fused_module = fuse_sequence(sequence, pattern)
                        if fused_module is not None:
                            # Replace the sequence with the fused module
                            setattr(parent, name.split('.')[-1], fused_module)
                            break

        return model

    def get_size_info(self) -> Dict[str, Any]:
        """
        Get information about model size before and after quantization.
        
        Returns:
            Dictionary with size information
        """
        if self.quantized_model is None:
            return {"error": "Model has not been quantized yet. Call optimize() first."}

        # Get size information
        original_size = sum(p.numel() * p.element_size() for p in self.model.parameters())

        # For static quantization, we need to be more careful about size calculation
        quantized_size = 0
        for name, module in self.quantized_model.named_modules():
            if hasattr(module, "_packed_params"):
                # This is a quantized module
                quantized_size += sum(p.numel() for p in module.parameters()) // 4  # INT8 is 1 byte
            else:
                # Regular module
                quantized_size += sum(p.numel() * p.element_size() for p in module.parameters())

        return {
            "original_size_mb": original_size / (1024 * 1024),
            "quantized_size_mb": quantized_size / (1024 * 1024),
            "compression_ratio": original_size / quantized_size if quantized_size > 0 else float('inf'),
            "backend": self.backend,
            "per_channel": self.config.per_channel,
        }


def extract_file_metadata(file_path: str = __file__):
    """
    Extract structured metadata about this module.

    Args:
        file_path: Path to the source file (defaults to current file)

    Returns:
        dict: Structured metadata about the module's purpose and components
    """
    return {
        "filename": os.path.basename(file_path),
        "module_purpose": "Implements various quantization techniques for neural networks.",
        "key_classes": [
            {
                "name": "QuantizationConfig",
                "purpose": "Configuration class for model quantization settings.",
                "key_methods": [
                    {
                        "name": "__init__",
                        "signature": "(self, quantization_type: str = 'dynamic', dtype: Optional[torch.dtype] = None, quantize_weights: bool = True, quantize_activations: bool = True, bits: int = 8, symmetric: bool = False, per_channel: bool = False)",
                        "brief_description": "Initialize quantization configuration."
                    },
                    {
                        "name": "__str__",
                        "signature": "(self) -> str",
                        "brief_description": "String representation of the configuration."
                    }
                ],
                "inheritance": "",
                "dependencies": ["torch", "typing"]
            },
            {
                "name": "ModelOptimizer",
                "purpose": "Base class for model optimization techniques.",
                "key_methods": [
                    {
                        "name": "__init__",
                        "signature": "(self, model: nn.Module)",
                        "brief_description": "Initialize the model optimizer."
                    },
                    {
                        "name": "optimize",
                        "signature": "(self) -> nn.Module",
                        "brief_description": "Apply optimization to the model."
                    },
                    {
                        "name": "restore_original",
                        "signature": "(self)",
                        "brief_description": "Restore the model to its original state."
                    },
                    {
                        "name": "get_size_info",
                        "signature": "(self) -> Dict[str, Any]",
                        "brief_description": "Get information about model size before and after optimization."
                    }
                ],
                "inheritance": "",
                "dependencies": ["torch", "typing", "logging"]
            },
            {
                "name": "DynamicQuantizer",
                "purpose": "Implements dynamic quantization for PyTorch models.",
                "key_methods": [
                    {
                        "name": "__init__",
                        "signature": "(self, model: nn.Module, config: Optional[QuantizationConfig] = None, dtype: torch.dtype = torch.qint8, qconfig_spec: Optional[Dict[Type[nn.Module], Any]] = None)",
                        "brief_description": "Initialize the dynamic quantizer."
                    },
                    {
                        "name": "optimize",
                        "signature": "(self) -> nn.Module",
                        "brief_description": "Apply dynamic quantization to the model."
                    },
                    {
                        "name": "_fuse_modules",
                        "signature": "(self, model: nn.Module) -> nn.Module",
                        "brief_description": "Fuse modules for improved quantization where applicable."
                    }
                ],
                "inheritance": "ModelOptimizer",
                "dependencies": ["torch", "typing", "logging"]
            },
            {
                "name": "StaticQuantizer",
                "purpose": "Implements static quantization for PyTorch models.",
                "key_methods": [
                    {
                        "name": "__init__",
                        "signature": "(self, model: nn.Module, config: Optional[QuantizationConfig] = None, calibration_loader: Optional[torch.utils.data.DataLoader] = None)",
                        "brief_description": "Initialize the static quantizer."
                    },
                    {
                        "name": "optimize",
                        "signature": "(self) -> nn.Module",
                        "brief_description": "Apply static quantization to the model."
                    },
                    {
                        "name": "_calibrate_model",
                        "signature": "(self, model: nn.Module)",
                        "brief_description": "Calibrate the model for static quantization."
                    },
                    {
                        "name": "_fuse_modules",
                        "signature": "(self, model: nn.Module, fusion_patterns: Optional[List[Tuple[Type[nn.Module], ...]]] = None) -> nn.Module",
                        "brief_description": "Fuse modules for improved quantization where applicable."
                    }
                ],
                "inheritance": "ModelOptimizer",
                "dependencies": ["torch", "typing", "logging"]
            }
        ],
        "external_dependencies": ["torch", "typing", "logging"],
        "complexity_score": 8,
    }
