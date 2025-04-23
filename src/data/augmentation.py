# src/data/augmentation.py
  import torch
  import torch.nn as nn
  import torchvision.transforms as T
  from typing import List, Dict, Optional, Tuple, Union, Callable
  import random

  class MultimodalAugmentationPipeline:
      """
      Advanced augmentation pipeline for multimodal data with configurable strategies
      for both image and text modalities.
      """

      def __init__(
          self,
          image_augs: Optional[List[Callable]] = None,
          text_augs: Optional[List[Callable]] = None,
          image_aug_prob: float = 0.5,
          text_aug_prob: float = 0.3,
          consistency_mode: str = "matched"
      ):
          # Default image augmentations if none provided
          self.image_augs = image_augs or [
              T.RandomHorizontalFlip(p=0.5),
              T.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
              T.RandomGrayscale(p=0.02),
              T.RandomAffine(degrees=10, translate=(0.1, 0.1), scale=(0.9, 1.1))
          ]

          # Text augmentations placeholders (implement based on your tokenizer)
          self.text_augs = text_augs or []

          self.image_aug_prob = image_aug_prob
          self.text_aug_prob = text_aug_prob
          self.consistency_mode = consistency_mode

      def augment_image(self, image: torch.Tensor) -> torch.Tensor:
          """Apply image augmentations with probability."""
          if random.random() < self.image_aug_prob:
              for aug in self.image_augs:
                  # Apply each augmentation with independent probability
                  if random.random() < 0.5:
                      image = aug(image)
          return image

      def augment_text(self, text_data: Dict) -> Dict:
          """Apply text augmentations with probability."""
          if not self.text_augs or random.random() >= self.text_aug_prob:
              return text_data

          # Apply text augmentations here
          # This depends on your text representation format
          return text_data

      def __call__(
          self, 
          batch: Dict[str, Union[torch.Tensor, Dict]]
      ) -> Dict[str, Union[torch.Tensor, Dict]]:
          """
          Apply augmentations to a batch of multimodal data.
          
          Args:
              batch: Dictionary with 'image' and 'text' keys
              
          Returns:
              Augmented batch with same structure
          """
          result = {}

          # Augment images if present
          if "image" in batch:
              # Handle both single images and batches
              if batch["image"].dim() == 3:  # Single image: [C, H, W]
                  result["image"] = self.augment_image(batch["image"])
              else:  # Batch of images: [B, C, H, W]
                  result["image"] = torch.stack([
                      self.augment_image(img) for img in batch["image"]
                  ])

          # Augment text if present
          if "text" in batch:
              result["text"] = self.augment_text(batch["text"])

          # Copy any other keys unchanged
          for k, v in batch.items():
              if k not in result:
                  result[k] = v

          return result
