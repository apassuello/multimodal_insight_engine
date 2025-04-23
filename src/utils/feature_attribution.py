  # src/utils/feature_attribution.py
  import torch
  import torch.nn as nn
  import torch.nn.functional as F
  import numpy as np
  import matplotlib.pyplot as plt
  from typing import Dict, List, Optional, Tuple, Union
  import cv2

  class GradCAM:
      """
      GradCAM implementation for visualizing which parts of an image contribute
      to a model's prediction. Useful for explainability in multimodal systems.
      """

      def __init__(self, model: nn.Module, target_layer: nn.Module):
          """
          Initialize GradCAM with a model and target layer for attribution.
          
          Args:
              model: The PyTorch model to analyze
              target_layer: The layer to extract gradients from (typically the last conv layer)
          """
          self.model = model
          self.target_layer = target_layer
          self.hooks = []
          self.gradients = None
          self.activations = None

          # Register hooks
          self._register_hooks()

      def _register_hooks(self):
          """Register forward and backward hooks on the target layer."""

          # Forward hook to capture activations
          def forward_hook(module, input, output):
              self.activations = output.detach()

          # Backward hook to capture gradients
          def backward_hook(module, grad_input, grad_output):
              self.gradients = grad_output[0].detach()

          # Register the hooks
          self.hooks.append(self.target_layer.register_forward_hook(forward_hook))
          self.hooks.append(self.target_layer.register_full_backward_hook(backward_hook))

      def remove_hooks(self):
          """Remove all registered hooks."""
          for hook in self.hooks:
              hook.remove()
          self.hooks = []

      def __call__(
          self, 
          images: torch.Tensor,
          text_data: Optional[Dict] = None,
          target_index: Optional[int] = None
      ) -> Tuple[np.ndarray, torch.Tensor]:
          """
          Generate CAM for the provided images.
          
          Args:
              images: Input images [B, C, H, W]
              text_data: Optional text data for multimodal models
              target_index: Optional target index to visualize (defaults to highest score)
              
          Returns:
              Tuple of (CAM visualization, model outputs)
          """
          # Put model in eval mode
          self.model.eval()

          # Forward pass
          batch_size = images.size(0)
          outputs = self.model(images=images, text_data=text_data)

          # Get the similarity or output scores
          if "similarity" in outputs:
              scores = outputs["similarity"]
          elif "logits" in outputs:
              scores = outputs["logits"]
          else:
              scores = outputs["vision_features"]

          # If target index not provided, use the highest scoring class
          if target_index is None:
              if scores.dim() > 1:
                  target_index = scores.argmax(dim=1)
              else:
                  target_index = scores.argmax(dim=0)

          # Create one-hot target for backprop
          one_hot = torch.zeros_like(scores)
          if one_hot.dim() > 1:
              for i in range(batch_size):
                  one_hot[i, target_index[i] if isinstance(target_index, torch.Tensor) else target_index] = 1
          else:
              one_hot[target_index] = 1

          # Zero gradients
          self.model.zero_grad()

          # Backward pass
          scores.backward(gradient=one_hot, retain_graph=True)

          # Get the gradient weights
          gradients = self.gradients
          activations = self.activations

          # Global average pooling of gradients
          weights = torch.mean(gradients, dim=[2, 3], keepdim=True)

          # Weight the activations by the gradients
          cam = torch.sum(weights * activations, dim=1, keepdim=True)

          # ReLU and normalize
          cam = F.relu(cam)
          cam = F.interpolate(cam, size=images.shape[2:], mode='bilinear', align_corners=False)

          # Normalize to [0, 1]
          cam_min, cam_max = cam.min(), cam.max()
          if cam_max > cam_min:
              cam = (cam - cam_min) / (cam_max - cam_min)

          # Convert to numpy for visualization
          cam_np = cam.squeeze().cpu().numpy()

          return cam_np, outputs

      def visualize(
          self, 
          image: torch.Tensor,
          cam: np.ndarray,
          alpha: float = 0.5,
          colormap: int = cv2.COLORMAP_JET
      ) -> np.ndarray:
          """
          Overlay the CAM on the original image.
          
          Args:
              image: Original image tensor [C, H, W]
              cam: Activation map [H, W]
              alpha: Transparency of the overlay
              colormap: OpenCV colormap for visualization
              
          Returns:
              Visualization as numpy array [H, W, 3]
          """
          # Convert image to numpy and denormalize if needed
          if isinstance(image, torch.Tensor):
              img = image.cpu().numpy().transpose(1, 2, 0)
              mean = np.array([0.485, 0.456, 0.406])
              std = np.array([0.229, 0.224, 0.225])
              img = img * std + mean
              img = np.clip(img, 0, 1)
          else:
              img = image

          # Convert to uint8 for OpenCV
          img_uint8 = (img * 255).astype(np.uint8)

          # Apply colormap to CAM
          cam_uint8 = (cam * 255).astype(np.uint8)
          cam_colored = cv2.applyColorMap(cam_uint8, colormap)

          # Overlay the CAM on the image
          overlay = cv2.addWeighted(img_uint8, 1-alpha, cam_colored, alpha, 0)

          return overlay
