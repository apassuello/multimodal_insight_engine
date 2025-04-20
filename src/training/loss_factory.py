# src/training/loss_factory.py
"""
Factory functions for creating and configuring loss functions.

This module provides factory functions to create various types of loss functions
tailored for multimodal learning, with appropriate configurations.
"""

import torch
import torch.nn as nn
import logging
from typing import Dict, Any, Optional, Union, Callable

from ..training.contrastive_learning import (
    ContrastiveLoss,
    MemoryQueueContrastiveLoss,
    DynamicTemperatureContrastiveLoss,
    HardNegativeMiningContrastiveLoss,
    MultiModalMixedContrastiveLoss,
)

# Simple and effective InfoNCE-style contrastive loss for SimpleMultimodalModel
class SimpleContrastiveLoss(nn.Module):
    """Simple and effective contrastive loss focused on clear cross-modal alignment."""
    
    def __init__(self, temperature: float = 0.1):
        """
        Initialize the contrastive loss.
        
        Args:
            temperature: Temperature parameter to scale logits
        """
        super().__init__()
        self.temperature = temperature
        self.iteration = 0
        logger.info(f"Initialized SimpleContrastiveLoss with temperature={temperature}")
    
    def forward(self, 
                vision_features: torch.Tensor, 
                text_features: torch.Tensor, 
                match_ids: Optional[torch.Tensor] = None, 
                **kwargs: Any) -> Dict[str, Any]:
        """
        Compute contrastive loss between vision and text features.
        
        Args:
            vision_features: Vision feature tensor [batch_size x dim]
            text_features: Text feature tensor [batch_size x dim]
            match_ids: Optional tensor of IDs for matching pairs
            **kwargs: Additional arguments
            
        Returns:
            Dict containing loss and metrics
        """
        self.iteration += 1
        batch_size = vision_features.shape[0]
        
        # Print raw feature statistics periodically
        if self.iteration % 5 == 0:
            with torch.no_grad():
                v_var = torch.var(vision_features).item()
                t_var = torch.var(text_features).item()
                logger.info(f"Raw feature variance - Vision: {v_var:.4f}, Text: {t_var:.4f}")
        
        # L2 normalize the features
        vision_features = torch.nn.functional.normalize(vision_features, p=2, dim=1)
        text_features = torch.nn.functional.normalize(text_features, p=2, dim=1)
        
        # Compute similarity matrix
        logits = torch.matmul(vision_features, text_features.T) / self.temperature
        
        # Create target matrix based on match_ids
        if match_ids is None:
            # If no match_ids, use identity matrix (diagonal matching)
            targets = torch.arange(batch_size, device=logits.device)
            positive_mask = torch.eye(batch_size, device=logits.device).bool()
        else:
            # Create mask where pairs with same match_id are positives
            positive_mask = torch.zeros((batch_size, batch_size), dtype=torch.bool, device=logits.device)
            for i in range(batch_size):
                for j in range(batch_size):
                    if match_ids[i] == match_ids[j]:
                        positive_mask[i, j] = True
        
        # Create matrix of positive pair masks
        # For InfoNCE, each row should sum to at least 1 (need at least one positive)
        row_sums = positive_mask.float().sum(dim=1, keepdim=True)
        if (row_sums == 0).any():
            # Add self as positive if no other positives exist
            identity_mask = torch.eye(batch_size, dtype=torch.bool, device=logits.device)
            positive_mask = positive_mask | identity_mask
        
        # Create standard targets for cross-entropy
        # For each row, randomly choose one of the positive columns as target
        targets_v2t = []
        for i in range(batch_size):
            pos_indices = torch.where(positive_mask[i])[0]
            if len(pos_indices) > 0:
                selected_idx = pos_indices[torch.randint(0, len(pos_indices), (1,))]
                targets_v2t.append(selected_idx.item())
            else:
                # Fallback to self as target
                targets_v2t.append(i)
        
        targets_v2t = torch.tensor(targets_v2t, device=logits.device)
        
        # Same for text to vision direction
        targets_t2v = []
        for j in range(batch_size):
            pos_indices = torch.where(positive_mask[:, j])[0]
            if len(pos_indices) > 0:
                selected_idx = pos_indices[torch.randint(0, len(pos_indices), (1,))]
                targets_t2v.append(selected_idx.item())
            else:
                # Fallback to self as target
                targets_t2v.append(j)
        
        targets_t2v = torch.tensor(targets_t2v, device=logits.device)
        
        # Calculate InfoNCE loss for both directions
        loss_v2t = torch.nn.functional.cross_entropy(logits, targets_v2t)
        loss_t2v = torch.nn.functional.cross_entropy(logits.T, targets_t2v)
        
        # Add direct supervised alignment loss
        mse_loss = 0.0
        mse_pairs = 0
        
        # For each vision sample, align with a random matched text sample  
        for i in range(batch_size):
            pos_indices = torch.where(positive_mask[i])[0]
            if len(pos_indices) > 0 and len(pos_indices) < batch_size:  # Make sure not all samples match
                j = pos_indices[torch.randint(0, len(pos_indices), (1,))].item()
                mse_loss += torch.nn.functional.mse_loss(vision_features[i], text_features[j])
                mse_pairs += 1
        
        if mse_pairs > 0:
            mse_loss = mse_loss / mse_pairs
        
        # Calculate metrics
        with torch.no_grad():
            # Calculate accuracy
            v2t_pred = torch.argmax(logits, dim=1)
            t2v_pred = torch.argmax(logits.T, dim=1)
            
            v2t_correct = torch.sum(positive_mask[torch.arange(batch_size), v2t_pred]).float()
            t2v_correct = torch.sum(positive_mask[t2v_pred, torch.arange(batch_size)]).float()
            
            v2t_acc = v2t_correct / batch_size
            t2v_acc = t2v_correct / batch_size
            accuracy = (v2t_acc + t2v_acc) / 2
            
            # Calculate positive and negative similarity statistics
            pos_sim = logits[positive_mask].mean().item() if positive_mask.sum() > 0 else 0.0
            neg_sim = logits[~positive_mask].mean().item() if (~positive_mask).sum() > 0 else 0.0
            separation = pos_sim - neg_sim
            
            # Log stats occasionally
            if self.iteration % 5 == 0:
                logger.info(f"Sim stats - Pos: {pos_sim:.4f}, Neg: {neg_sim:.4f}, Gap: {separation:.4f}")
        
        # Use the anti-collapse loss from the model if available
        decor_loss = kwargs.get('decor_loss', 0.0)
        if isinstance(decor_loss, torch.Tensor):
            decor_weight = 0.2  # Moderate weight
            total_loss = (loss_v2t + loss_t2v) / 2 + 0.5 * mse_loss + decor_weight * decor_loss
        else:
            total_loss = (loss_v2t + loss_t2v) / 2 + 0.5 * mse_loss
        
        return {
            "loss": total_loss,
            "infonce_loss": (loss_v2t + loss_t2v).item() / 2,
            "mse_loss": mse_loss.item() if isinstance(mse_loss, torch.Tensor) else 0.0,
            "v2t_loss": loss_v2t.item(),
            "t2v_loss": loss_t2v.item(),
            "accuracy": accuracy.item(),
            "pos_sim": pos_sim,
            "neg_sim": neg_sim,
            "separation": separation,
            "temperature": self.temperature,
            "decor_loss": decor_loss.item() if isinstance(decor_loss, torch.Tensor) else 0.0,
        }

logger = logging.getLogger(__name__)


def create_loss_function(
    args: Any, dataset_size: Optional[int] = None, train_loader: Optional[Any] = None
) -> nn.Module:
    """
    Create the appropriate loss function based on arguments.

    Args:
        args: Command line arguments
        dataset_size: Size of the dataset for auto-selecting sampling strategy
        train_loader: Training dataloader for advanced loss configuration

    Returns:
        Loss function
    """
    # Get loss type directly from args now that we've added it to the argument parser
    loss_type = args.loss_type

    # Log which loss function is being used
    logger.info(f"Creating loss function of type: {loss_type}")

    # Check if we're using SimpleMultimodalModel
    if hasattr(args, "use_simple_model") and args.use_simple_model:
        logger.info("Using SimpleContrastiveLoss for simple model architecture")
        return SimpleContrastiveLoss(temperature=args.temperature)
            
    # Use args.use_mixed_loss to override if it's explicitly set
    if args.use_mixed_loss:
        loss_type = "mixed"
        logger.info(
            "Overriding to Mixed Contrastive Loss based on --use_mixed_loss flag"
        )

    # CRITICAL: Get the actual model dimensions for proper projection setup
    # This should match what the model factory uses
    model_dim = None
    # First try getting the fusion_dim from args
    if hasattr(args, "fusion_dim") and args.fusion_dim is not None:
        # Fusion dim should already be set to match the model dim in model_factory.py
        model_dim = args.fusion_dim

        # CRITICAL CHECK: Override to 768 if we detect a ViT-base from timm
        if model_dim == 512 and getattr(args, "vision_model", "") == "vit-base":
            # If fusion_dim is 512 but we're using vit-base, there's a mismatch
            # This happens when model_factory updates local var but not args
            model_dim = 768
            logger.warning(
                f"Detected likely dimension mismatch! args.fusion_dim={args.fusion_dim} but vision_model={args.vision_model}"
            )
            logger.warning(f"Overriding dimension to {model_dim} to match ViT-base")
        else:
            logger.info(f"Using fusion_dim from args for loss function: {model_dim}")
    # Otherwise assume a modern vision transformer uses 768
    else:
        # Default to the most common ViT-base dimension (768)
        model_dim = 768
        logger.info(f"Using default ViT-base dimension for loss function: {model_dim}")

    # Make sure we're using a valid dimension
    if model_dim <= 0:
        model_dim = 768
        logger.warning(f"Invalid model dimension, using default: {model_dim}")

    # Switch based on the loss type
    if loss_type == "memory_queue":
        # Memory Queue-Based Contrastive Loss
        logger.info("Using Memory Queue-Based Contrastive Loss")

        # Use queue_size from args if provided, otherwise determine based on dataset size
        if args.queue_size:
            queue_size = args.queue_size
        elif dataset_size is not None:
            # Larger datasets benefit from larger queues
            if dataset_size > 10000:
                queue_size = 16384  # Very large queue for large datasets
            elif dataset_size > 5000:
                queue_size = 8192  # Large queue for medium-large datasets
            elif dataset_size > 1000:
                queue_size = 4096  # Medium queue for medium datasets
            else:
                queue_size = 2048  # Smaller queue for small datasets
        else:
            queue_size = 8192  # Default to a reasonably large queue size

        # Adjust temperature for memory queue approach
        # Generally needs slightly higher temperature than standard contrastive
        adjusted_temp = args.temperature * 1.1

        logger.info(
            f"Memory Queue size: {queue_size}, Temperature: {adjusted_temp:.4f}"
        )

        logger.info(f"Creating MemoryQueueContrastiveLoss with dimension {model_dim}")
        return MemoryQueueContrastiveLoss(
            dim=model_dim,  # Use the detected model dimension
            queue_size=queue_size,
            temperature=adjusted_temp,
        )

    elif loss_type == "dynamic_temp":
        # Dynamic Temperature Calibration
        logger.info("Using Dynamic Temperature Calibration Contrastive Loss")

        # Base temperature is provided by args
        base_temp = args.temperature

        # Use min/max from args if provided, otherwise calculate sensible defaults
        if args.dynamic_temp_min is not None:
            min_temp = args.dynamic_temp_min
        else:
            min_temp = max(0.01, base_temp * 0.6)  # Don't go below 0.01 or 60% of base

        if args.dynamic_temp_max is not None:
            max_temp = args.dynamic_temp_max
        else:
            max_temp = min(0.3, base_temp * 2.0)  # Don't go above 0.3 or 200% of base

        logger.info(
            f"Dynamic Temperature - Base: {base_temp:.4f}, Range: [{min_temp:.4f}, {max_temp:.4f}]"
        )

        logger.info(
            f"Creating DynamicTemperatureContrastiveLoss with dimension {model_dim}"
        )
        return DynamicTemperatureContrastiveLoss(
            base_temperature=base_temp,
            min_temp=min_temp,
            max_temp=max_temp,
            dim=model_dim,  # Use the detected model dimension
        )

    elif loss_type == "hard_negative":
        # Hard Negative Mining Contrastive Loss
        logger.info("Using Hard Negative Mining Contrastive Loss")

        # Use mining strategy from args if not 'auto', otherwise determine based on batch size
        if args.mining_strategy and args.mining_strategy != "auto":
            mining_strategy = args.mining_strategy
        elif args.batch_size < 32:
            # For small batches, semi-hard negatives work better
            mining_strategy = "semi-hard"
        else:
            # For larger batches, full hard negative mining is effective
            mining_strategy = "hard"

        # Use hard negative factor from args if provided
        if args.hard_negative_factor is not None:
            hard_negative_factor = args.hard_negative_factor
        elif mining_strategy == "semi-hard":
            # Higher weighting for semi-hard approach
            hard_negative_factor = 3.0
        else:
            # Moderate weighting for hard approach
            hard_negative_factor = 2.0

        logger.info(
            f"Hard Negative Mining - Strategy: {mining_strategy}, Weight: {hard_negative_factor:.1f}x"
        )

        logger.info(
            f"Creating HardNegativeMiningContrastiveLoss with dimension {model_dim}"
        )
        return HardNegativeMiningContrastiveLoss(
            temperature=args.temperature,
            hard_negative_factor=hard_negative_factor,
            mining_strategy=mining_strategy,
            dim=model_dim,  # Use the detected model dimension
        )

    elif loss_type == "mixed":
        # Mixed Contrastive Loss with multiple objectives
        logger.info("Using Mixed Contrastive Loss with multiple objectives")

        logger.info(
            f"Creating MultiModalMixedContrastiveLoss with dimension {model_dim}"
        )
        return MultiModalMixedContrastiveLoss(
            contrastive_weight=1.0,
            classification_weight=0.5,
            multimodal_matching_weight=0.5,
            temperature=args.temperature,
            use_hard_negatives=True,  # ENABLED hard negative mining
            hard_negative_weight=(
                0.5 if args.hard_negative_factor is None else args.hard_negative_factor
            ),
            dim=model_dim,  # Use the detected model dimension
        )

    else:
        # Standard Contrastive Loss (default)
        logger.info("Using Standard Contrastive Loss with enhanced settings")

        # Determine which sampling strategy to use
        if args.contrastive_sampling == "auto":
            if dataset_size is not None:
                if dataset_size < 1000:
                    sampling_strategy = "global"
                    logger.info(
                        f"Auto-selecting 'global' sampling strategy for small dataset size ({dataset_size})"
                    )
                elif dataset_size < 10000:
                    sampling_strategy = "memory-bank"
                    logger.info(
                        f"Auto-selecting 'memory-bank' sampling strategy for medium dataset size ({dataset_size})"
                    )
                else:
                    sampling_strategy = "in-batch"
                    logger.info(
                        f"Auto-selecting 'in-batch' sampling strategy for large dataset size ({dataset_size})"
                    )
            else:
                sampling_strategy = "memory-bank"  # CHANGED default to memory-bank
                logger.info(
                    "Dataset size unknown, defaulting to 'memory-bank' sampling strategy for better performance"
                )
        else:
            sampling_strategy = args.contrastive_sampling
            logger.info(f"Using '{sampling_strategy}' sampling strategy as specified")

        # IMPROVEMENT: Add a warning about in-batch sampling
        if sampling_strategy == "in-batch":
            logger.warning(
                "WARNING: Using 'in-batch' sampling can lead to shortcut learning for small batches. "
                "Consider using 'memory-bank' or 'global' for more robust training."
            )

        # IMPROVED: Use a more appropriate temperature based on strategy and batch size
        # Smaller batches need lower temperature for more focused learning
        if args.batch_size < 64:
            temp_scale = 0.85  # More aggressive scaling for small batches
        elif args.batch_size < 128:
            temp_scale = 0.9
        else:
            temp_scale = 0.95  # Less aggressive for large batches

        # Apply temperature adjustment based on sampling strategy
        if sampling_strategy == "in-batch":
            adjusted_temp = args.temperature * temp_scale  # Lower temp for in-batch
        elif sampling_strategy == "memory-bank":
            adjusted_temp = args.temperature * 1.0  # Standard temp for memory-bank
        else:
            adjusted_temp = (
                args.temperature * 1.05
            )  # Slightly higher for global (more diverse negatives)

        logger.info(
            f"Using Contrastive Loss with temperature {adjusted_temp:.4f} "
            f"(original: {args.temperature}, adjusted by {adjusted_temp/args.temperature:.2f}x) "
            f"and {sampling_strategy} sampling strategy"
        )

        # Calculate appropriate projection dimension based on model dimension
        # For large model dimensions, using a smaller projection can help generalization
        projection_dim = min(256, model_dim // 2)  # Don't go below model_dim/2

        logger.info(
            f"Creating ContrastiveLoss with input dimension {model_dim}, projection dimension {projection_dim}"
        )

        # CRITICAL FIX: Create a more advanced contrastive loss with better settings
        return ContrastiveLoss(
            temperature=adjusted_temp,
            loss_type="infonce",  # InfoNCE loss is standard for contrastive learning
            reduction="mean",
            add_projection=True,  # ENABLED: Use projection heads for better representation
            projection_dim=projection_dim,  # Smaller dimension for the projection space
            input_dim=model_dim,  # Use the detected model dimension
            sampling_strategy=sampling_strategy,
            memory_bank_size=args.memory_bank_size
            * 2,  # INCREASED: Use larger memory bank
            dataset_size=dataset_size,
        )


def extract_file_metadata(file_path=__file__):
    """
    Extract structured metadata about this module.

    Args:
        file_path: Path to the source file (defaults to current file)

    Returns:
        dict: Structured metadata about the module's purpose and components
    """
    import os

    return {
        "filename": os.path.basename(file_path),
        "module_purpose": "Factory functions for creating and configuring loss functions",
        "key_functions": [
            {
                "name": "create_loss_function",
                "signature": "create_loss_function(args: Any, dataset_size: Optional[int] = None, train_loader: Optional[Any] = None) -> nn.Module",
                "brief_description": "Create the appropriate loss function based on arguments",
            }
        ],
        "external_dependencies": ["torch", "logging"],
        "complexity_score": 7,  # Moderately high complexity for loss function configuration
    }
