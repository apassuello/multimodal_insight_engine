"""MODULE: multitask_loss.py
PURPOSE: Implements a flexible multitask loss function that combines multiple task-specific losses with configurable weights.

KEY COMPONENTS:
- MultitaskLoss: Main class that aggregates multiple loss functions with weights
- Dynamic task weighting based on task difficulty or importance
- Support for various task types (classification, regression, contrastive)

DEPENDENCIES:
- PyTorch (torch, torch.nn)
- Logging

SPECIAL NOTES:
- Supports dynamic loss weighting strategies
- Can be extended with task-specific metrics tracking
- Designed for use in multimodal and multitask learning scenarios
"""

import logging
import os
from typing import Any, Dict, Optional

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class MultitaskLoss(nn.Module):
    """
    Implements a flexible multitask loss that combines multiple losses with weights.

    This loss function is designed for multitask learning scenarios where different
    tasks might require different loss functions and importance weights.
    """

    def __init__(
        self,
        loss_functions: Dict[str, nn.Module],
        loss_weights: Optional[Dict[str, float]] = None,
        dynamic_weighting: bool = False,
        reduction: str = "mean",
    ):
        """
        Initialize the multitask loss module.

        Args:
            loss_functions: Dictionary mapping task names to loss function modules
            loss_weights: Dictionary mapping task names to their weights (defaults to equal weights)
            dynamic_weighting: Whether to use dynamic task weighting based on loss values
            reduction: How to reduce the loss ("mean", "sum", or "none")
        """
        super().__init__()
        self.loss_functions = nn.ModuleDict(loss_functions)
        self.task_names = list(loss_functions.keys())

        # Initialize weights (equal if not specified)
        if loss_weights is None:
            weight_value = 1.0 / len(self.task_names)
            self.loss_weights = {task: weight_value for task in self.task_names}
        else:
            # Normalize weights to sum to 1
            total_weight = sum(loss_weights.values())
            self.loss_weights = {
                task: weight / total_weight for task, weight in loss_weights.items()
            }

        self.dynamic_weighting = dynamic_weighting
        self.reduction = reduction

        # For dynamic weighting: store running statistics of losses
        if dynamic_weighting:
            self.running_losses = {task: 1.0 for task in self.task_names}
            self.momentum = 0.9  # For exponential moving average

    def forward(
        self, inputs: Dict[str, Any], targets: Dict[str, Any]
    ) -> Dict[str, torch.Tensor]:
        """
        Compute multitask loss by combining task-specific losses.

        Args:
            inputs: Dictionary mapping task names to model outputs
            targets: Dictionary mapping task names to target values

        Returns:
            Dictionary with total loss and individual task losses
        """
        individual_losses = {}

        # Compute individual task losses
        for task in self.task_names:
            if task in inputs and task in targets:
                loss = self.loss_functions[task](inputs[task], targets[task])
                individual_losses[f"{task}_loss"] = loss
            else:
                logger.warning(f"Task {task} missing from inputs or targets")

        # Update dynamic weights if enabled
        if self.dynamic_weighting and len(individual_losses) > 0:
            self._update_dynamic_weights(individual_losses)

        # Compute weighted sum of losses
        total_loss = 0.0
        for task in self.task_names:
            loss_key = f"{task}_loss"
            if loss_key in individual_losses:
                total_loss += self.loss_weights[task] * individual_losses[loss_key]

        # Add total loss to the results
        results = {"loss": total_loss, **individual_losses}

        # Add weights to the results for monitoring
        for task in self.task_names:
            results[f"{task}_weight"] = torch.tensor(self.loss_weights[task])

        return results

    def _update_dynamic_weights(
        self, individual_losses: Dict[str, torch.Tensor]
    ) -> None:
        """
        Update task weights dynamically based on current loss values.

        Args:
            individual_losses: Dictionary mapping task loss names to their values
        """
        # Update running averages of losses
        for task in self.task_names:
            loss_key = f"{task}_loss"
            if loss_key in individual_losses:
                current_loss = individual_losses[loss_key].detach().item()
                self.running_losses[task] = (
                    self.momentum * self.running_losses[task]
                    + (1 - self.momentum) * current_loss
                )

        # Compute inverse weights (higher loss -> higher weight)
        total_inverse_loss = sum(
            1.0 / max(loss, 1e-8) for loss in self.running_losses.values()
        )

        # Update weights based on inverse loss magnitude
        for task in self.task_names:
            inverse_loss = 1.0 / max(self.running_losses[task], 1e-8)
            self.loss_weights[task] = inverse_loss / total_inverse_loss


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
        "module_purpose": "Implements a flexible multitask loss function that combines multiple task-specific losses with configurable weights",
        "key_classes": [
            {
                "name": "MultitaskLoss",
                "purpose": "Combines multiple loss functions with configurable or dynamic weighting for multitask learning",
                "key_methods": [
                    {
                        "name": "__init__",
                        "signature": "__init__(self, loss_functions: Dict[str, nn.Module], loss_weights: Optional[Dict[str, float]] = None, dynamic_weighting: bool = False, reduction: str = 'mean')",
                        "brief_description": "Initialize with task-specific loss functions and optional weighting configuration",
                    },
                    {
                        "name": "forward",
                        "signature": "forward(self, inputs: Dict[str, Any], targets: Dict[str, Any]) -> Dict[str, torch.Tensor]",
                        "brief_description": "Compute weighted combination of task-specific losses",
                    },
                    {
                        "name": "_update_dynamic_weights",
                        "signature": "_update_dynamic_weights(self, individual_losses: Dict[str, torch.Tensor]) -> None",
                        "brief_description": "Update task weights dynamically based on current loss values",
                    },
                ],
                "inheritance": "nn.Module",
                "dependencies": ["torch", "torch.nn", "torch.nn.functional"],
            }
        ],
        "external_dependencies": ["torch", "logging"],
        "complexity_score": 7,  # High complexity due to dynamic weighting and multitask handling
    }
