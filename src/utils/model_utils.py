"""MODULE: model_utils.py
PURPOSE: Provides utility functions for working with neural network models, including inspection, parameter counting, and device management.

KEY COMPONENTS:
- Model parameter counting and summarization
- Device detection and management
- Tensor conversion utilities
- Model device placement tools

DEPENDENCIES:
- PyTorch (torch, torch.nn)

SPECIAL NOTES:
- Includes smart device detection with MPS/Apple Silicon support
- Provides utilities for consistent model placement across heterogeneous architectures
- Handles tensor conversion for serialization
"""

# src/utils/model_utils.py
"""
Utility functions for working with neural network models.

This module provides utilities for model inspection, parameter counting,
data conversion, and other helper functions for model management.
"""

import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Optional, Union, Any
import os


def count_parameters(model: nn.Module) -> int:
    """
    Count the total number of trainable parameters in a model.

    Args:
        model: PyTorch model

    Returns:
        int: Total number of trainable parameters
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def print_model_summary(model: nn.Module, title: str = "MODEL SUMMARY") -> None:
    """
    Print a concise summary of the model architecture and parameter counts.

    Args:
        model: PyTorch model
        title: Title for the summary
    """
    total_params = count_parameters(model)

    # Try to get submodule parameter counts if available
    vision_params = text_params = fusion_params = 0

    if hasattr(model, "vision_model"):
        vision_params = count_parameters(model.vision_model)

    if hasattr(model, "text_model"):
        text_params = count_parameters(model.text_model)

    # Fusion parameters (estimate)
    fusion_params = total_params - vision_params - text_params

    print("\n" + "=" * 50)
    print(f"{title}")
    print("-" * 50)
    print(f"Total parameters:       {total_params:,}")
    if vision_params > 0:
        print(
            f"Vision model parameters: {vision_params:,} ({vision_params/total_params*100:.1f}%)"
        )
    if text_params > 0:
        print(
            f"Text model parameters:   {text_params:,} ({text_params/total_params*100:.1f}%)"
        )
    if fusion_params > 0:
        print(
            f"Fusion parameters:       {fusion_params:,} ({fusion_params/total_params*100:.1f}%)"
        )
    print("=" * 50 + "\n")


def get_device(device_name: Optional[str] = None) -> torch.device:
    """
    Get the appropriate device based on availability.

    Args:
        device_name: Optional device name to use (cuda, mps, cpu)

    Returns:
        torch.device: Appropriate device
    """
    if device_name is not None:
        return torch.device(device_name)

    if torch.cuda.is_available():
        return torch.device("cuda")
    elif (
        hasattr(torch, "backends")
        and hasattr(torch.backends, "mps")
        and torch.backends.mps.is_available()
    ):
        return torch.device("mps")
    else:
        return torch.device("cpu")


def convert_tensors_to_python_types(obj: Any) -> Any:
    """
    Convert PyTorch tensors to native Python types for JSON serialization.

    Args:
        obj: Object to convert (could be tensor, dict, list, tuple, or other)

    Returns:
        Converted object with tensors replaced by Python native types
    """
    if isinstance(obj, torch.Tensor):
        # Convert tensor to Python type
        return obj.item() if obj.numel() == 1 else obj.detach().cpu().tolist()
    elif isinstance(obj, dict):
        # Recursively convert dictionary values
        return {k: convert_tensors_to_python_types(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        # Recursively convert list elements
        return [convert_tensors_to_python_types(item) for item in obj]
    elif isinstance(obj, tuple):
        # Recursively convert tuple elements
        return tuple(convert_tensors_to_python_types(item) for item in obj)
    else:
        # Return other types as is
        return obj


def ensure_model_on_device(model: nn.Module, device: torch.device) -> nn.Module:
    """
    Ensure all model components are on the correct device.

    Args:
        model: The model to check
        device: Target device

    Returns:
        model: The model with all components on the target device
    """
    # Move the model to the specified device
    model = model.to(device)

    # Check common submodules in multimodal models
    if hasattr(model, "vision_model"):
        model.vision_model = model.vision_model.to(device)

    if hasattr(model, "text_model"):
        model.text_model = model.text_model.to(device)

    if hasattr(model, "fusion_module"):
        model.fusion_module = model.fusion_module.to(device)

    if hasattr(model, "classifier"):
        model.classifier = model.classifier.to(device)

    # Verify all parameters are on the device
    devices = {param.device for param in model.parameters()}
    if len(devices) > 1:
        # We still have parameters on different devices, force all parameters individually
        for name, param in model.named_parameters():
            if param.device != device:
                param.data = param.data.to(device)

    return model


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
        "module_purpose": "Provides utility functions for working with neural network models, including inspection, parameter counting, and device management",
        "key_functions": [
            {
                "name": "count_parameters",
                "signature": "count_parameters(model: nn.Module) -> int",
                "brief_description": "Count the total number of trainable parameters in a model",
            },
            {
                "name": "print_model_summary",
                "signature": "print_model_summary(model: nn.Module, title: str = 'MODEL SUMMARY') -> None",
                "brief_description": "Print a concise summary of the model architecture and parameter counts",
            },
            {
                "name": "get_device",
                "signature": "get_device(device_name: Optional[str] = None) -> torch.device",
                "brief_description": "Get the appropriate device based on availability (CUDA, MPS, CPU)",
            },
            {
                "name": "convert_tensors_to_python_types",
                "signature": "convert_tensors_to_python_types(obj: Any) -> Any",
                "brief_description": "Convert PyTorch tensors to native Python types for JSON serialization",
            },
            {
                "name": "ensure_model_on_device",
                "signature": "ensure_model_on_device(model: nn.Module, device: torch.device) -> nn.Module",
                "brief_description": "Ensure all model components are on the correct device, handling complex multimodal models",
            },
        ],
        "external_dependencies": ["torch", "torch.nn"],
        "complexity_score": 5,  # Medium complexity for utility functions
    }
