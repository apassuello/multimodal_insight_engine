# src/training/losses/feature_consistency_loss.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Any, Union
import logging

logger = logging.getLogger(__name__)

"""MODULE: feature_consistency_loss.py
PURPOSE: Implements feature consistency loss to prevent catastrophic forgetting during fine-tuning.

KEY COMPONENTS:
- FeatureConsistencyLoss: Loss that maintains consistency between current and reference model features
  - Prevents catastrophic forgetting of useful features from earlier training stages
  - Supports multiple distance metrics (cosine, L2, L1, smooth_L1)
  - Works with vision features, text features, or both independently
  - Carefully handles device transfers for multi-device training
  - Provides configurable constraints through weighting

DEPENDENCIES:
- PyTorch (torch, torch.nn, torch.nn.functional)
- Python standard library (logging)

SPECIAL NOTES:
- Essential for Stage 3 of the progressive training approach
- Particularly important when unfreezing pretrained models
- Compatible with various architecture changes between stages
- Requires reference models from earlier training stages
"""


class FeatureConsistencyLoss(nn.Module):
    """
    Implements feature consistency loss to prevent catastrophic forgetting during fine-tuning.

    When fine-tuning pretrained models, especially in Stage 3 of the progressive training
    approach, there's a risk of "catastrophic forgetting" where useful features learned
    during pretraining are lost. This loss prevents this by enforcing consistency between
    the features produced by the current model and a reference model (typically a copy
    of the model from an earlier training stage).

    Key features:
    - Preserves important features from earlier training stages
    - Configurable feature distance metrics (cosine, L2, etc.)
    - Supports feature consistency for both vision and text modalities
    - Control over constraint strength via weighting
    - Can be combined with other losses like contrastive or VICReg loss
    """

    def __init__(
        self,
        reference_vision_model: Optional[nn.Module] = None,
        reference_text_model: Optional[nn.Module] = None,
        vision_weight: float = 1.0,
        text_weight: float = 1.0,
        distance_fn: str = "cosine",  # "cosine", "l2", "l1", "smooth_l1"
        normalize_features: bool = True,
        detach_reference: bool = True,
        reduction: str = "mean",
    ):
        """
        Initialize the feature consistency loss.

        Args:
            reference_vision_model: Vision model from earlier training stage (for reference features)
            reference_text_model: Text model from earlier training stage (for reference features)
            vision_weight: Weight for vision consistency constraint
            text_weight: Weight for text consistency constraint
            distance_fn: Function to measure feature distance ("cosine", "l2", "l1", "smooth_l1")
            normalize_features: Whether to normalize features before computing distance
            detach_reference: Whether to detach reference features (recommended)
            reduction: How to reduce the loss ("mean", "sum", or "none")
        """
        super().__init__()
        self.reference_vision_model = reference_vision_model
        self.reference_text_model = reference_text_model
        self.vision_weight = vision_weight
        self.text_weight = text_weight
        self.distance_fn = distance_fn
        self.normalize_features = normalize_features
        self.detach_reference = detach_reference
        self.reduction = reduction

        # Set reference models to eval mode and disable gradients if requested
        if self.reference_vision_model is not None:
            self.reference_vision_model.eval()
            if self.detach_reference:
                for param in self.reference_vision_model.parameters():
                    param.requires_grad = False

        if self.reference_text_model is not None:
            self.reference_text_model.eval()
            if self.detach_reference:
                for param in self.reference_text_model.parameters():
                    param.requires_grad = False

    def _compute_distance(
        self, current_features: torch.Tensor, reference_features: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute distance between current and reference features.

        Args:
            current_features: Features from current model
            reference_features: Features from reference model

        Returns:
            Distance tensor
        """
        # Normalize features if requested
        if self.normalize_features:
            current_features = F.normalize(current_features, p=2, dim=1)
            reference_features = F.normalize(reference_features, p=2, dim=1)

        # Compute distance based on selected function
        if self.distance_fn == "cosine":
            # Cosine distance = 1 - cosine similarity
            # Lower is better (more similar)
            similarity = torch.sum(current_features * reference_features, dim=1)
            distance = 1.0 - similarity
        elif self.distance_fn == "l2":
            # L2 distance (Euclidean)
            distance = torch.norm(current_features - reference_features, p=2, dim=1)
        elif self.distance_fn == "l1":
            # L1 distance (Manhattan)
            distance = torch.norm(current_features - reference_features, p=1, dim=1)
        elif self.distance_fn == "smooth_l1":
            # Smooth L1 (Huber loss)
            distance = F.smooth_l1_loss(
                current_features, reference_features, reduction="none"
            ).sum(dim=1)
        else:
            raise ValueError(f"Unsupported distance function: {self.distance_fn}")

        # Apply reduction
        if self.reduction == "mean":
            distance = distance.mean()
        elif self.reduction == "sum":
            distance = distance.sum()

        return distance

    def forward(
        self,
        vision_features: Optional[torch.Tensor] = None,
        text_features: Optional[torch.Tensor] = None,
        vision_inputs: Optional[torch.Tensor] = None,
        text_inputs: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Compute feature consistency loss to prevent catastrophic forgetting.

        Args:
            vision_features: Vision features from current model
            text_features: Text features from current model
            vision_inputs: Vision inputs (if reference models need to compute features)
            text_inputs: Text inputs (if reference models need to compute features)
            **kwargs: Additional arguments (ignored)

        Returns:
            Dictionary with loss values and additional metrics
        """
        # Track which modalities to compute consistency for
        compute_vision = self.reference_vision_model is not None and (
            vision_features is not None or vision_inputs is not None
        )
        compute_text = self.reference_text_model is not None and (
            text_features is not None or text_inputs is not None
        )

        device = None
        if vision_features is not None:
            device = vision_features.device
        elif text_features is not None:
            device = text_features.device
        elif vision_inputs is not None:
            device = vision_inputs.device
        elif text_inputs is not None:
            device = text_inputs.device
        else:
            return {"loss": torch.tensor(0.0)}  # Nothing to compute

        # Loss components
        vision_loss = torch.tensor(0.0, device=device)
        text_loss = torch.tensor(0.0, device=device)

        # Compute reference vision features if needed
        if compute_vision:
            # If reference features need to be computed from inputs
            if vision_features is None and vision_inputs is not None:
                with torch.set_grad_enabled(not self.detach_reference):
                    # Get model device
                    model_device = next(self.reference_vision_model.parameters()).device

                    # Ensure inputs are on the same device as the model
                    if vision_inputs.device != model_device:
                        vision_inputs_device = vision_inputs.to(model_device)
                    else:
                        vision_inputs_device = vision_inputs

                    # Compute reference features
                    reference_vision_features = self.reference_vision_model(
                        vision_inputs_device
                    )

                    # Move back to original device if needed
                    if model_device != device:
                        reference_vision_features = reference_vision_features.to(device)

                # We need current model to compute features too
                logger.warning(
                    "Current vision features not provided, cannot compute vision consistency loss"
                )
                compute_vision = False

            # If we already have current features, get reference features
            elif vision_features is not None:
                if vision_inputs is not None:
                    with torch.set_grad_enabled(not self.detach_reference):
                        # Get model device
                        model_device = next(
                            self.reference_vision_model.parameters()
                        ).device

                        # Ensure inputs are on the same device as the model
                        if vision_inputs.device != model_device:
                            vision_inputs_device = vision_inputs.to(model_device)
                        else:
                            vision_inputs_device = vision_inputs

                        # Compute reference features
                        reference_vision_features = self.reference_vision_model(
                            vision_inputs_device
                        )

                        # Move back to original device if needed
                        if model_device != device:
                            reference_vision_features = reference_vision_features.to(
                                device
                            )
                else:
                    logger.warning(
                        "Vision inputs not provided, cannot compute reference vision features"
                    )
                    compute_vision = False

                # Verify feature dimensions match
                if (
                    compute_vision
                    and vision_features.shape != reference_vision_features.shape
                ):
                    logger.warning(
                        f"Vision feature shape mismatch: current={vision_features.shape}, "
                        f"reference={reference_vision_features.shape}"
                    )
                    compute_vision = False

                # Compute vision consistency loss if everything is ready
                if compute_vision:
                    vision_loss = self._compute_distance(
                        vision_features, reference_vision_features
                    )

        # Compute reference text features if needed (similar process as vision)
        if compute_text:
            # If reference features need to be computed from inputs
            if text_features is None and text_inputs is not None:
                with torch.set_grad_enabled(not self.detach_reference):
                    # Get model device
                    model_device = next(self.reference_text_model.parameters()).device

                    # Ensure inputs are on the same device as the model
                    if isinstance(text_inputs, dict):
                        # Handle dict-style inputs (e.g. for BERT)
                        text_inputs_device = {}
                        for k, v in text_inputs.items():
                            if isinstance(v, torch.Tensor):
                                text_inputs_device[k] = v.to(model_device)
                            else:
                                text_inputs_device[k] = v
                    else:
                        # Handle tensor inputs
                        if text_inputs.device != model_device:
                            text_inputs_device = text_inputs.to(model_device)
                        else:
                            text_inputs_device = text_inputs

                    # Compute reference features
                    reference_text_features = self.reference_text_model(
                        text_inputs_device
                    )

                    # Move back to original device if needed
                    if model_device != device:
                        reference_text_features = reference_text_features.to(device)

                # We need current model to compute features too
                logger.warning(
                    "Current text features not provided, cannot compute text consistency loss"
                )
                compute_text = False

            # If we already have current features, get reference features
            elif text_features is not None:
                if text_inputs is not None:
                    with torch.set_grad_enabled(not self.detach_reference):
                        # Get model device
                        model_device = next(
                            self.reference_text_model.parameters()
                        ).device

                        # Ensure inputs are on the same device as the model
                        if isinstance(text_inputs, dict):
                            # Handle dict-style inputs (e.g. for BERT)
                            text_inputs_device = {}
                            for k, v in text_inputs.items():
                                if isinstance(v, torch.Tensor):
                                    text_inputs_device[k] = v.to(model_device)
                                else:
                                    text_inputs_device[k] = v
                        else:
                            # Handle tensor inputs
                            if text_inputs.device != model_device:
                                text_inputs_device = text_inputs.to(model_device)
                            else:
                                text_inputs_device = text_inputs

                        # Compute reference features
                        reference_text_features = self.reference_text_model(
                            text_inputs_device
                        )

                        # Move back to original device if needed
                        if model_device != device:
                            reference_text_features = reference_text_features.to(device)
                else:
                    logger.warning(
                        "Text inputs not provided, cannot compute reference text features"
                    )
                    compute_text = False

                # Verify feature dimensions match
                if (
                    compute_text
                    and text_features.shape != reference_text_features.shape
                ):
                    logger.warning(
                        f"Text feature shape mismatch: current={text_features.shape}, "
                        f"reference={reference_text_features.shape}"
                    )
                    compute_text = False

                # Compute text consistency loss if everything is ready
                if compute_text:
                    text_loss = self._compute_distance(
                        text_features, reference_text_features
                    )

        # Combine losses with weights
        total_loss = 0.0
        if compute_vision:
            total_loss = total_loss + self.vision_weight * vision_loss
        if compute_text:
            total_loss = total_loss + self.text_weight * text_loss

        # Return loss and metrics
        return {
            "loss": total_loss,
            "feature_consistency_loss": total_loss.item(),
            "vision_consistency_loss": vision_loss.item() if compute_vision else 0.0,
            "text_consistency_loss": text_loss.item() if compute_text else 0.0,
            "vision_weight": self.vision_weight,
            "text_weight": self.text_weight,
        }


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
        "module_purpose": "Implements feature consistency loss to prevent catastrophic forgetting during fine-tuning",
        "key_classes": [
            {
                "name": "FeatureConsistencyLoss",
                "purpose": "Maintains consistency between current and reference model features to prevent forgetting",
                "key_methods": [
                    {
                        "name": "forward",
                        "signature": "forward(self, vision_features: Optional[torch.Tensor] = None, text_features: Optional[torch.Tensor] = None, vision_inputs: Optional[torch.Tensor] = None, text_inputs: Optional[torch.Tensor] = None, **kwargs) -> Dict[str, Any]",
                        "brief_description": "Computes consistency loss between current and reference features",
                    },
                    {
                        "name": "_compute_distance",
                        "signature": "_compute_distance(self, current_features: torch.Tensor, reference_features: torch.Tensor) -> torch.Tensor",
                        "brief_description": "Calculates distance between feature representations using selected metric",
                    },
                ],
                "inheritance": "nn.Module",
                "dependencies": ["torch", "torch.nn", "torch.nn.functional", "logging"],
            }
        ],
        "external_dependencies": ["torch", "logging"],
        "complexity_score": 7,  # Medium-high complexity due to comprehensive device handling
    }
