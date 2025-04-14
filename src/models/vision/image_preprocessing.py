# src/models/vision/image_preprocessing.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import numpy as np
from typing import Tuple, Union, List, Optional


class ImagePreprocessor:
    """
    Preprocesses images for input to a Vision Transformer.

    This class handles:
    1. Resizing images to a consistent resolution
    2. Normalizing pixel values
    3. Converting to tensor format
    """

    def __init__(
        self,
        image_size: int = 224,
        mean: Tuple[float, float, float] = (0.485, 0.456, 0.406),
        std: Tuple[float, float, float] = (0.229, 0.224, 0.225),
    ):
        """
        Initialize the image preprocessor.

        Args:
            image_size: Target size for images (both height and width)
            mean: Mean values for RGB normalization
            std: Standard deviation values for RGB normalization
        """
        self.image_size = image_size
        self.transform = transforms.Compose(
            [
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std),
            ]
        )

    def preprocess(
        self, image: Union[str, Image.Image, np.ndarray, torch.Tensor]
    ) -> torch.Tensor:
        """
        Preprocess an image for input to a Vision Transformer.

        Args:
            image: Input image as file path, PIL Image, numpy array, or tensor

        Returns:
            Preprocessed image tensor of shape [C, H, W]
        """
        # Handle different input types
        if isinstance(image, str):
            # Load image from file path
            image = Image.open(image).convert("RGB")
        elif isinstance(image, np.ndarray):
            # Convert numpy array to PIL Image
            image = Image.fromarray(np.uint8(image))
        elif isinstance(image, torch.Tensor):
            # If already a tensor, ensure it's in the right format
            if image.dim() == 4:  # [B, C, H, W]
                image = image.squeeze(0)  # Remove batch dimension
            if image.dim() == 3:
                if image.shape[0] == 3:  # Already [C, H, W]
                    return F.interpolate(
                        image.unsqueeze(0),
                        size=(self.image_size, self.image_size),
                        mode="bilinear",
                        align_corners=False,
                    ).squeeze(0)
                else:  # [H, W, C]
                    image = image.permute(2, 0, 1)  # Convert to [C, H, W]
                    return F.interpolate(
                        image.unsqueeze(0),
                        size=(self.image_size, self.image_size),
                        mode="bilinear",
                        align_corners=False,
                    ).squeeze(0)

        # Apply the transform
        return self.transform(image)

    def batch_preprocess(
        self, images: List[Union[str, Image.Image, np.ndarray, torch.Tensor]]
    ) -> torch.Tensor:
        """
        Preprocess a batch of images.

        Args:
            images: List of images to preprocess

        Returns:
            Batch of preprocessed image tensors of shape [B, C, H, W]
        """
        return torch.stack([self.preprocess(img) for img in images])


class PatchExtractor(nn.Module):
    """
    Extracts patches from images using convolution.

    This is more efficient than manual patch extraction and reshaping.
    """

    def __init__(self, patch_size: int = 16):
        """
        Initialize the patch extractor.

        Args:
            patch_size: Size of the patches to extract (both height and width)
        """
        super().__init__()
        self.patch_size = patch_size
        # No learnable parameters - this just defines the sliding window for extraction

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Tuple[int, int]]:
        """
        Extract patches from images.

        Args:
            x: Batch of images of shape [B, C, H, W]

        Returns:
            Tuple containing:
            - Patches of shape [B, N, C*P*P] where N is the number of patches and P is patch_size
            - Tuple of (num_patches_height, num_patches_width)
        """
        B, C, H, W = x.shape

        # Ensure dimensions are divisible by patch size
        assert (
            H % self.patch_size == 0
        ), f"Image height {H} must be divisible by patch size {self.patch_size}"
        assert (
            W % self.patch_size == 0
        ), f"Image width {W} must be divisible by patch size {self.patch_size}"

        # Calculate number of patches in each dimension
        num_patches_h = H // self.patch_size
        num_patches_w = W // self.patch_size
        num_patches = num_patches_h * num_patches_w

        # Extract patches
        patches = x.unfold(2, self.patch_size, self.patch_size).unfold(
            3, self.patch_size, self.patch_size
        )
        # Shape: [B, C, num_patches_h, num_patches_w, patch_size, patch_size]

        # Reshape to get individual patches
        # [B, C*P*P, N] -> [B, N, C*P*P]
        patches = patches.permute(0, 2, 3, 1, 4, 5).contiguous()
        patches = patches.reshape(B, num_patches, -1)

        return patches, (num_patches_h, num_patches_w)
