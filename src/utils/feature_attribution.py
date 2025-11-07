"""MODULE: feature_attribution.py
PURPOSE: Implements various feature attribution techniques for model interpretability and visualization.

KEY COMPONENTS:
- GradCAM: Implementation of Gradient-weighted Class Activation Mapping
- IntegratedGradients: Implementation of integrated gradients method
- SaliencyMap: Simple saliency map generation
- AttributionVisualizer: Tools for visualizing attribution results

DEPENDENCIES:
- PyTorch (torch, torch.nn)
- NumPy and Matplotlib for numerical computation and visualization
- PIL for image processing

SPECIAL NOTES:
- Supports both vision and text modalities with appropriate adaptations
- Includes specialized handling for multimodal models
- Visualization tools integrated for easy interpretation
"""

import logging
import os
from typing import Any, Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from PIL import Image

logger = logging.getLogger(__name__)


class GradCAM:
    """
    Implements Gradient-weighted Class Activation Mapping for vision models.

    GradCAM uses the gradients flowing into the final convolutional layer
    to produce a coarse localization map highlighting important regions
    in the image that contribute to the prediction.
    """

    def __init__(
        self,
        model: nn.Module,
        target_layer: nn.Module,
        use_cuda: bool = False,
    ):
        """
        Initialize GradCAM.

        Args:
            model: The model to analyze
            target_layer: The layer to compute GradCAM for (usually the final conv layer)
            use_cuda: Whether to use CUDA if available
        """
        self.model = model
        self.target_layer = target_layer
        self.use_cuda = use_cuda and torch.cuda.is_available()

        self.device = torch.device("cuda" if self.use_cuda else "cpu")
        self.model = self.model.to(self.device)
        self.model.eval()

        # Register hooks to save activations and gradients
        self.activations = None
        self.gradients = None

        self.target_layer.register_forward_hook(self._save_activation)
        self.target_layer.register_backward_hook(self._save_gradient)

    def _save_activation(self, module, input, output):
        """Forward hook to save activations."""
        self.activations = output.detach()

    def _save_gradient(self, module, grad_input, grad_output):
        """Backward hook to save gradients."""
        self.gradients = grad_output[0].detach()

    def __call__(
        self,
        input_image: torch.Tensor,
        target_class: Optional[int] = None,
    ) -> Tuple[np.ndarray, torch.Tensor]:
        """
        Generate a GradCAM heatmap.

        Args:
            input_image: Input image tensor of shape (1, C, H, W)
            target_class: Target class index (if None, uses the predicted class)

        Returns:
            Tuple of (heatmap as numpy array, model prediction)
        """
        # Ensure the model is in evaluation mode
        self.model.eval()

        # Forward pass
        input_image = input_image.to(self.device)
        input_image.requires_grad = True
        output = self.model(input_image)

        # Get predicted class if target_class is None
        if target_class is None:
            target_class = torch.argmax(output).item()

        # Backward pass
        self.model.zero_grad()
        one_hot = torch.zeros_like(output)
        one_hot[0, target_class] = 1
        output.backward(gradient=one_hot, retain_graph=True)

        # Compute weights (global average pooling of gradients)
        gradients = self.gradients.mean(dim=[2, 3], keepdim=True)

        # Weight the activations
        weighted_activations = self.activations * gradients

        # Generate heatmap
        heatmap = (
            torch.sum(weighted_activations, dim=1).squeeze().cpu().detach().numpy()
        )

        # Apply ReLU to focus on features that have a positive influence
        heatmap = np.maximum(heatmap, 0)

        # Normalize heatmap
        if np.max(heatmap) > 0:
            heatmap = heatmap / np.max(heatmap)

        return heatmap, output


class IntegratedGradients:
    """
    Implements Integrated Gradients attribution method.

    This method computes the path integral of gradients along
    a straight-line path from a baseline to the input. It can
    be applied to both vision and text models.
    """

    def __init__(
        self,
        model: nn.Module,
        use_cuda: bool = False,
        steps: int = 50,
    ):
        """
        Initialize Integrated Gradients.

        Args:
            model: The model to analyze
            use_cuda: Whether to use CUDA if available
            steps: Number of steps for approximating the integral
        """
        self.model = model
        self.use_cuda = use_cuda and torch.cuda.is_available()
        self.steps = steps

        self.device = torch.device("cuda" if self.use_cuda else "cpu")
        self.model = self.model.to(self.device)
        self.model.eval()

    def __call__(
        self,
        input_tensor: torch.Tensor,
        target_class: Optional[int] = None,
        baseline: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute integrated gradients for an input.

        Args:
            input_tensor: Input tensor
            target_class: Target class index (if None, uses the predicted class)
            baseline: Baseline input (if None, uses zero tensor)

        Returns:
            Tuple of (attributions, predictions)
        """
        # Ensure model is in eval mode
        self.model.eval()

        # Create baseline if not provided
        if baseline is None:
            baseline = torch.zeros_like(input_tensor)

        # Move to device
        input_tensor = input_tensor.to(self.device)
        baseline = baseline.to(self.device)

        # Get prediction
        if not input_tensor.requires_grad:
            input_tensor.requires_grad = True

        output = self.model(input_tensor)

        # Get target class if not provided
        if target_class is None:
            target_class = torch.argmax(output).item()

        # Compute integrated gradients
        attributions = torch.zeros_like(input_tensor)

        # Straightline path from baseline to input
        for step in range(self.steps):
            alpha = step / self.steps
            interpolated = baseline + alpha * (input_tensor - baseline)
            interpolated.requires_grad = True

            # Forward pass
            output = self.model(interpolated)

            # Backward pass
            self.model.zero_grad()
            one_hot = torch.zeros_like(output)
            one_hot[0, target_class] = 1
            output.backward(gradient=one_hot, retain_graph=True)

            # Accumulate gradients
            attributions += interpolated.grad

        # Scale attributions to input - baseline
        attributions *= (input_tensor - baseline) / self.steps

        return attributions, output


class SaliencyMap:
    """
    Implements a simple saliency map based on input gradients.

    This method visualizes the gradient of the output with respect
    to the input, highlighting features that have the strongest
    influence on the prediction.
    """

    def __init__(
        self,
        model: nn.Module,
        use_cuda: bool = False,
    ):
        """
        Initialize Saliency Map.

        Args:
            model: The model to analyze
            use_cuda: Whether to use CUDA if available
        """
        self.model = model
        self.use_cuda = use_cuda and torch.cuda.is_available()

        self.device = torch.device("cuda" if self.use_cuda else "cpu")
        self.model = self.model.to(self.device)
        self.model.eval()

    def __call__(
        self,
        input_tensor: torch.Tensor,
        target_class: Optional[int] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute saliency map for an input.

        Args:
            input_tensor: Input tensor
            target_class: Target class index (if None, uses the predicted class)

        Returns:
            Tuple of (saliency map, predictions)
        """
        # Ensure model is in eval mode
        self.model.eval()

        # Move to device and enable gradients
        input_tensor = input_tensor.to(self.device)
        input_tensor.requires_grad = True

        # Forward pass
        output = self.model(input_tensor)

        # Get target class if not provided
        if target_class is None:
            target_class = torch.argmax(output).item()

        # Backward pass
        self.model.zero_grad()
        one_hot = torch.zeros_like(output)
        one_hot[0, target_class] = 1
        output.backward(gradient=one_hot)

        # Compute saliency map as absolute gradient
        saliency = torch.abs(input_tensor.grad)

        return saliency, output


class AttributionVisualizer:
    """
    Utilities for visualizing feature attributions.
    """

    @staticmethod
    def overlay_heatmap(
        image: Union[np.ndarray, Image.Image, torch.Tensor],
        heatmap: np.ndarray,
        colormap: str = "jet",
        alpha: float = 0.5,
    ) -> np.ndarray:
        """
        Overlay a heatmap on an image.

        Args:
            image: Original image
            heatmap: Heatmap to overlay
            colormap: Matplotlib colormap to use
            alpha: Opacity of the heatmap overlay

        Returns:
            Overlaid image as numpy array
        """
        # Convert image to numpy if needed
        if isinstance(image, torch.Tensor):
            # Assume CxHxW format
            img_np = image.cpu().detach().numpy()
            if img_np.shape[0] in [1, 3]:  # CxHxW format
                img_np = np.transpose(img_np, (1, 2, 0))

            # Normalize to [0, 1]
            img_np = (img_np - img_np.min()) / (img_np.max() - img_np.min())
        elif isinstance(image, Image.Image):
            img_np = np.array(image) / 255.0
        else:
            img_np = image

        # Ensure heatmap has the right dimensions
        if heatmap.ndim == 2:
            heatmap = np.expand_dims(heatmap, 0)

        # Resize heatmap to match image if needed
        if heatmap.shape[:2] != img_np.shape[:2]:
            from skimage.transform import resize

            heatmap = resize(heatmap, img_np.shape[:2], order=1, mode="reflect")

        # Create colormap
        cmap = plt.get_cmap(colormap)
        heatmap_colored = cmap(heatmap)

        # Blend images
        overlaid = alpha * heatmap_colored + (1 - alpha) * img_np

        return overlaid

    @staticmethod
    def visualize_text_attribution(
        tokens: List[str],
        attributions: torch.Tensor,
        colormap: str = "RdBu_r",
        threshold: float = 0.0,
    ) -> plt.Figure:
        """
        Visualize attributions for text.

        Args:
            tokens: List of tokens
            attributions: Attributions per token
            colormap: Matplotlib colormap to use
            threshold: Threshold to filter small attributions

        Returns:
            Matplotlib figure with visualization
        """
        # Ensure attributions is a 1D tensor
        if attributions.dim() > 1:
            attributions = attributions.sum(dim=1)

        # Convert to numpy
        attr_np = attributions.cpu().detach().numpy()

        # Filter small attributions
        attr_np = np.where(np.abs(attr_np) < threshold, 0, attr_np)

        # Normalize
        max_abs = np.max(np.abs(attr_np))
        attr_np = attr_np / max_abs if max_abs > 0 else attr_np

        # Create figure
        fig, ax = plt.subplots(figsize=(12, 3))
        cmap = plt.get_cmap(colormap)

        # Plot heatmap
        im = ax.imshow(
            attr_np.reshape(1, -1), cmap=cmap, aspect="auto", vmin=-1, vmax=1
        )

        # Add tokens as x-axis labels
        ax.set_xticks(range(len(tokens)))
        ax.set_xticklabels(tokens, rotation=45, ha="right")
        ax.set_yticks([])

        # Add colorbar
        plt.colorbar(im, ax=ax)

        # Adjust layout
        plt.tight_layout()

        return fig


def attribution_for_multimodal_model(
    model: nn.Module,
    image: torch.Tensor,
    text: Union[str, torch.Tensor],
    attribution_method: str = "grad_cam",
    target_class: Optional[int] = None,
    **kwargs,
) -> Dict[str, Any]:
    """
    Compute attributions for a multimodal model.

    Args:
        model: Multimodal model
        image: Image tensor
        text: Text input (string or tensor)
        attribution_method: Method to use ('grad_cam', 'integrated_gradients', 'saliency')
        target_class: Target class index
        **kwargs: Additional arguments for the specific attribution method

    Returns:
        Dictionary with attributions for image and text, and prediction
    """
    # Handle different attribution methods
    if attribution_method == "grad_cam":
        # Find convolutional layer for GradCAM
        target_layer = None
        for name, module in model.named_modules():
            if isinstance(module, nn.Conv2d) and "vision_model" in name:
                target_layer = module

        if target_layer is None:
            raise ValueError("No suitable convolutional layer found for GradCAM")

        # Create GradCAM instance
        attributor = GradCAM(model, target_layer, **kwargs)

        # Get attributions
        image_heatmap, prediction = attributor(image, target_class)

        return {
            "image_attribution": image_heatmap,
            "text_attribution": None,  # GradCAM doesn't handle text well
            "prediction": prediction,
        }

    elif attribution_method == "integrated_gradients":
        # Create IntegratedGradients instance
        attributor = IntegratedGradients(model, **kwargs)

        # Process image
        image_attributions, prediction = attributor(image, target_class)

        # Process text if it's a tensor
        text_attributions = None
        if isinstance(text, torch.Tensor):
            text_attributions, _ = attributor(text, target_class)

        return {
            "image_attribution": image_attributions,
            "text_attribution": text_attributions,
            "prediction": prediction,
        }

    elif attribution_method == "saliency":
        # Create SaliencyMap instance
        attributor = SaliencyMap(model, **kwargs)

        # Process image
        image_saliency, prediction = attributor(image, target_class)

        # Process text if it's a tensor
        text_saliency = None
        if isinstance(text, torch.Tensor):
            text_saliency, _ = attributor(text, target_class)

        return {
            "image_attribution": image_saliency,
            "text_attribution": text_saliency,
            "prediction": prediction,
        }

    else:
        raise ValueError(f"Unknown attribution method: {attribution_method}")


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
        "module_purpose": "Implements various feature attribution techniques for model interpretability and visualization",
        "key_classes": [
            {
                "name": "GradCAM",
                "purpose": "Implements Gradient-weighted Class Activation Mapping for vision models",
                "key_methods": [
                    {
                        "name": "__init__",
                        "signature": "__init__(self, model: nn.Module, target_layer: nn.Module, use_cuda: bool = False)",
                        "brief_description": "Initialize GradCAM with model and target layer",
                    },
                    {
                        "name": "__call__",
                        "signature": "__call__(self, input_image: torch.Tensor, target_class: Optional[int] = None) -> Tuple[np.ndarray, torch.Tensor]",
                        "brief_description": "Generate a GradCAM heatmap for the input image",
                    },
                ],
                "inheritance": "object",
                "dependencies": ["torch", "torch.nn", "numpy"],
            },
            {
                "name": "IntegratedGradients",
                "purpose": "Implements the integrated gradients method for computing feature importance",
                "key_methods": [
                    {
                        "name": "__init__",
                        "signature": "__init__(self, model: nn.Module, use_cuda: bool = False, steps: int = 50)",
                        "brief_description": "Initialize integrated gradients with model and steps",
                    },
                    {
                        "name": "__call__",
                        "signature": "__call__(self, input_tensor: torch.Tensor, target_class: Optional[int] = None, baseline: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]",
                        "brief_description": "Compute integrated gradients for an input tensor",
                    },
                ],
                "inheritance": "object",
                "dependencies": ["torch", "torch.nn"],
            },
            {
                "name": "SaliencyMap",
                "purpose": "Creates simple saliency maps based on input gradients",
                "key_methods": [
                    {
                        "name": "__call__",
                        "signature": "__call__(self, input_tensor: torch.Tensor, target_class: Optional[int] = None) -> Tuple[torch.Tensor, torch.Tensor]",
                        "brief_description": "Compute saliency map highlighting important input features",
                    }
                ],
                "inheritance": "object",
                "dependencies": ["torch", "torch.nn"],
            },
            {
                "name": "AttributionVisualizer",
                "purpose": "Utilities for visualizing feature attributions for both images and text",
                "key_methods": [
                    {
                        "name": "overlay_heatmap",
                        "signature": "overlay_heatmap(image: Union[np.ndarray, Image.Image, torch.Tensor], heatmap: np.ndarray, colormap: str = 'jet', alpha: float = 0.5) -> np.ndarray",
                        "brief_description": "Overlay a heatmap on an image for visualization",
                    },
                    {
                        "name": "visualize_text_attribution",
                        "signature": "visualize_text_attribution(tokens: List[str], attributions: torch.Tensor, colormap: str = 'RdBu_r', threshold: float = 0.0) -> plt.Figure",
                        "brief_description": "Visualize attributions for text tokens",
                    },
                ],
                "inheritance": "object",
                "dependencies": ["numpy", "matplotlib", "PIL"],
            },
        ],
        "key_functions": [
            {
                "name": "attribution_for_multimodal_model",
                "signature": "attribution_for_multimodal_model(model: nn.Module, image: torch.Tensor, text: Union[str, torch.Tensor], attribution_method: str = 'grad_cam', target_class: Optional[int] = None, **kwargs) -> Dict[str, Any]",
                "brief_description": "Compute attributions for a multimodal model using the specified method",
            }
        ],
        "external_dependencies": ["torch", "numpy", "matplotlib", "PIL"],
        "complexity_score": 8,  # High complexity due to integration of multiple attribution methods and visualization techniques
    }
