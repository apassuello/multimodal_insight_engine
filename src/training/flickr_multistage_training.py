# src/training/flickr_multistage_training.py

"""
Multistage Training for Multimodal Systems: Flickr30k Implementation

This module implements a three-stage training approach for multimodal vision-language models:
1. Modality-Specific Learning: Strengthen individual encoders
2. Cross-Modal Fusion: Align vision and text feature spaces
3. End-to-End Fine-tuning: Carefully fine-tune all components together

Each stage uses different components, loss functions, learning rates, and batch construction
strategies to systematically develop multimodal understanding capabilities.
"""

import os
import json
import time
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR, CosineAnnealingWarmRestarts
import numpy as np
from tqdm import tqdm
from typing import Dict, List, Tuple, Optional, Callable, Union, Any
import matplotlib.pyplot as plt
from collections import defaultdict
from torch.optim.optimizer import Optimizer
from torch.optim.lr_scheduler import _LRScheduler
from torch.optim.adamw import AdamW

# Local imports
from src.models.multimodal.vicreg_multimodal_model import VICRegMultimodalModel
from src.training.trainers.multimodal_trainer import MultimodalTrainer
from src.data.multimodal_dataset import MultimodalDataset
from src.data.multimodal_data_utils import SemanticGroupBatchSampler
from src.training.losses.vicreg_loss import VICRegLoss
from src.training.losses.contrastive_loss import ContrastiveLoss
from src.training.losses.memory_queue_contrastive_loss import MemoryQueueContrastiveLoss
from src.training.losses.hard_negative_mining_contrastive_loss import (
    HardNegativeMiningContrastiveLoss,
)
from src.models.pretrained import (
    VisionTransformerWrapper,
    HuggingFaceTextModelWrapper,
)
from src.data.multimodal_dataset import EnhancedMultimodalDataset

logger = logging.getLogger(__name__)


class FlickrMultistageTrainer:
    """
    Three-stage trainer for multimodal models using the Flickr30k dataset.

    This trainer implements a systematic approach for multimodal learning:
    1. Stage 1: Modality-Specific Learning - Focus on individual encoders
    2. Stage 2: Cross-Modal Fusion - Align vision and text feature spaces
    3. Stage 3: End-to-End Fine-tuning - Carefully integrate all components

    Each stage has specific component freezing/unfreezing, loss functions,
    learning rates, and batch construction strategies.
    """

    def __init__(
        self,
        model: nn.Module,
        data_root: str,
        output_dir: str = "flickr30k",
        device: Optional[torch.device] = None,
        batch_size: int = 64,
        num_workers: int = 4,
        stage1_epochs: int = 30,
        stage2_epochs: int = 15,
        stage3_epochs: int = 15,
        image_size: int = 224,
        use_metadata: bool = True,
    ):
        """
        Initialize the multistage trainer.

        Args:
            model: The multimodal model to train
            data_root: Root directory for Flickr30k dataset
            output_dir: Directory to save checkpoints and logs
            device: Device to use for training
            batch_size: Base batch size for training
            num_workers: Number of workers for data loading
            stage1_epochs: Number of epochs for Stage 1 (modality-specific)
            stage2_epochs: Number of epochs for Stage 2 (cross-modal)
            stage3_epochs: Number of epochs for Stage 3 (end-to-end)
            image_size: Size of images for training
            use_metadata: Whether to use metadata for advanced batch sampling
        """
        self.model = model
        self.data_root = data_root
        self.output_dir = output_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.stage1_epochs = stage1_epochs
        self.stage2_epochs = stage2_epochs
        self.stage3_epochs = stage3_epochs
        self.image_size = image_size
        self.use_metadata = use_metadata

        # Set device
        self.device = (
            device
            if device is not None
            else (torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        )
        self.model = self.model.to(self.device)

        # Create directories
        self.checkpoint_dir = os.path.join(output_dir, "checkpoints")
        self.log_dir = os.path.join(output_dir, "logs")
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        os.makedirs(self.log_dir, exist_ok=True)

        # Set up logging
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s [%(levelname)s] %(message)s",
            handlers=[
                logging.FileHandler(os.path.join(self.log_dir, "training.log")),
                logging.StreamHandler(),
            ],
        )

        # Create dataloaders
        self.train_dataloader, self.val_dataloader = self._create_dataloaders()

        # Initialize training history
        self.history = {
            "stage1": {"train_loss": [], "val_loss": [], "metrics": {}},
            "stage2": {"train_loss": [], "val_loss": [], "metrics": {}},
            "stage3": {"train_loss": [], "val_loss": [], "metrics": {}},
        }

        logger.info(
            f"FlickrMultistageTrainer initialized with {len(self.train_dataloader)} batches per epoch"
        )

    def _create_dataloaders(self) -> Tuple[DataLoader, DataLoader]:
        """
        Create training and validation dataloaders.

        Returns:
            Tuple of (train_dataloader, val_dataloader)
        """
        # Create training dataset
        train_dataset = EnhancedMultimodalDataset(
            dataset_name="flickr30k",
            split="train",
            image_preprocessor=None,  # Will use default
            tokenizer=None,  # Will use default
            max_text_length=77,
            cache_dir=os.path.join(self.data_root, "flickr30k"),
            captions_per_image=5,  # Flickr30k has 5 captions per image
            min_samples_per_group=5,
            max_samples_per_group=30,
            cap_strategy="random",
        )

        # Create validation dataset
        val_dataset = EnhancedMultimodalDataset(
            dataset_name="flickr30k",
            split="val",
            image_preprocessor=None,  # Will use default
            tokenizer=None,  # Will use default
            max_text_length=77,
            cache_dir=os.path.join(self.data_root, "flickr30k"),
            captions_per_image=5,  # Flickr30k has 5 captions per image
            min_samples_per_group=5,
            max_samples_per_group=30,
            cap_strategy="random",
        )

        # Create dataloaders with custom sampler for training
        if self.use_metadata:
            # Use semantic batch sampler for training to ensure each batch has multiple samples
            # from the same image with different captions (key for contrastive learning)
            train_sampler = SemanticGroupBatchSampler(
                dataset=train_dataset,
                batch_size=self.batch_size,
                min_samples_per_group=5,  # Flickr30k has 5 captions per image
                groups_per_batch=16,  # Aim for 16 different images in a batch
            )
            train_dataloader = DataLoader(
                train_dataset,
                batch_sampler=train_sampler,
                num_workers=self.num_workers,
                pin_memory=True,
            )
        else:
            # Standard dataloader without semantic grouping
            train_dataloader = DataLoader(
                train_dataset,
                batch_size=self.batch_size,
                shuffle=True,
                num_workers=self.num_workers,
                pin_memory=True,
            )

        # Validation dataloader (no special sampling needed)
        val_dataloader = DataLoader(
            val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )

        return train_dataloader, val_dataloader

    def train_stage1(self):
        """
        Stage 1: Modality-Specific Learning.

        Focus on strengthening individual encoders before alignment:
        - Freeze cross-attention and fusion components
        - Unfreeze vision encoder, text encoder, and projection layers
        - Use unimodal contrastive loss + decorrelation loss
        - Apply different learning rates to different components
        """
        logger.info("Starting Stage 1: Modality-Specific Learning")

        # 1. Freeze/unfreeze components
        self._freeze_layers("cross_attention")
        self._freeze_layers("fusion")
        self._unfreeze_layers("vision_encoder")
        self._unfreeze_layers("text_encoder")
        self._unfreeze_layers("projection")

        # 2. Create optimizer with component-specific learning rates
        base_lr = 1e-4
        optimizer = self._create_component_optimizer(
            {
                "vision_encoder": base_lr * 0.1,  # 1e-5
                "text_encoder": base_lr * 0.1,  # 1e-5
                "projection": base_lr,  # 1e-4
                "default": base_lr * 0.01,  # 1e-6 for any other params
            }
        )

        # 3. Create learning rate scheduler with warmup
        total_steps = len(self.train_dataloader) * self.stage1_epochs
        warmup_steps = int(0.1 * total_steps)  # 10% warmup
        scheduler = CosineAnnealingLR(optimizer, T_max=total_steps - warmup_steps)

        # 4. Create combined loss function for Stage 1
        # VICReg loss for decorrelation and intra-modal similarity
        vicreg_loss = VICRegLoss(
            sim_coeff=10.0,
            var_coeff=5.0,
            cov_coeff=1.0,
            curriculum=True,
            warmup_epochs=3,
            num_epochs=self.stage1_epochs,
        )

        # Unimodal contrastive loss for stronger individual encoders
        contrastive_loss = ContrastiveLoss(
            temperature=0.07,
            loss_type="infonce",
            sampling_strategy="in-batch",
        )

        # 5. Create and run trainer for Stage 1
        trainer = MultimodalTrainer(
            model=self.model,
            train_dataloader=self.train_dataloader,
            val_dataloader=self.val_dataloader,
            optimizer=optimizer,
            scheduler=scheduler,
            loss_fn=contrastive_loss,  # Primary loss
            secondary_loss_fn=vicreg_loss,  # Secondary loss for feature regularization
            secondary_loss_weight=0.5,  # Weight for secondary loss
            num_epochs=self.stage1_epochs,
            checkpoint_dir=os.path.join(self.checkpoint_dir, "stage1"),
            log_dir=os.path.join(self.log_dir, "stage1"),
            device=self.device,
            accumulation_steps=1,
            evaluation_steps=100,
            log_steps=50,
        )

        # Run training
        trainer.train()

        # Save Stage 1 checkpoint
        stage1_checkpoint_path = os.path.join(self.checkpoint_dir, "stage1_complete.pt")
        torch.save(
            {
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "epoch": self.stage1_epochs,
            },
            stage1_checkpoint_path,
        )

        # Save metrics
        self.history["stage1"] = trainer.history
        self._save_history()

        logger.info(f"Stage 1 complete! Checkpoint saved to {stage1_checkpoint_path}")

    def train_stage2(self):
        """
        Stage 2: Cross-Modal Fusion.

        Focus on aligning vision and text feature spaces using cross-modal components:
        - Freeze vision and text encoder backbones
        - Unfreeze cross-attention layers, fusion components, joint projection layers
        - Use bidirectional InfoNCE + memory queue contrastive loss
        - Apply cosine learning rate schedule with restarts
        """
        logger.info("Starting Stage 2: Cross-Modal Fusion")

        # 1. Freeze/unfreeze components
        self._freeze_layers("vision_encoder_backbone")
        self._freeze_layers("text_encoder_backbone")
        self._unfreeze_layers("cross_attention")
        self._unfreeze_layers("fusion")
        self._unfreeze_layers("projection")

        # 2. Create optimizer with component-specific learning rates
        base_lr = 1e-4
        optimizer = self._create_component_optimizer(
            {
                "cross_attention": base_lr,  # 1e-4
                "fusion": base_lr,  # 1e-4
                "projection": base_lr,  # 1e-4
                "default": base_lr * 0.01,  # 1e-6 for any other params
            }
        )

        # 3. Create learning rate scheduler with warmup and restarts
        total_steps = len(self.train_dataloader) * self.stage2_epochs
        scheduler = CosineAnnealingWarmRestarts(
            optimizer,
            T_0=len(self.train_dataloader) * 5,  # Restart every 5 epochs
            T_mult=1,
            eta_min=base_lr * 0.01,  # Minimum learning rate
        )

        # 4. Create combined loss function for Stage 2
        # Memory queue contrastive loss for larger effective batch size
        memory_queue_loss = MemoryQueueContrastiveLoss(
            dim=self.model.projection_dim, queue_size=8192, temperature=0.07
        )

        # Hard negative mining contrastive loss for challenging examples
        hard_negative_loss = HardNegativeMiningContrastiveLoss(
            temperature=0.07, queue_size=8192, hard_negative_factor=0.5
        )

        # 5. Create and run trainer for Stage 2
        trainer = MultimodalTrainer(
            model=self.model,
            train_dataloader=self.train_dataloader,
            val_dataloader=self.val_dataloader,
            optimizer=optimizer,
            scheduler=scheduler,
            loss_fn=memory_queue_loss,  # Primary loss
            secondary_loss_fn=hard_negative_loss,  # Secondary loss
            secondary_loss_weight=0.5,  # Weight for secondary loss
            num_epochs=self.stage2_epochs,
            checkpoint_dir=os.path.join(self.checkpoint_dir, "stage2"),
            log_dir=os.path.join(self.log_dir, "stage2"),
            device=self.device,
            accumulation_steps=1,
            evaluation_steps=100,
            log_steps=50,
        )

        # Run training
        trainer.train()

        # Save Stage 2 checkpoint
        stage2_checkpoint_path = os.path.join(self.checkpoint_dir, "stage2_complete.pt")
        torch.save(
            {
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "epoch": self.stage2_epochs,
            },
            stage2_checkpoint_path,
        )

        # Save metrics
        self.history["stage2"] = trainer.history
        self._save_history()

        logger.info(f"Stage 2 complete! Checkpoint saved to {stage2_checkpoint_path}")

    def train_stage3(self):
        """
        Stage 3: End-to-End Fine-tuning.

        Carefully fine-tune all components with anti-forgetting mechanisms and layer-specific learning rates:
        - Unfreeze all components with progressive unfreezing
        - Use combined losses from Stage 2 + feature consistency distillation
        - Apply very specific learning rates for different components
        - Implement sophisticated batch construction with hard negatives
        """
        logger.info("Starting Stage 3: End-to-End Fine-tuning")

        # 1. First unfreeze all components but keep strict control on learning rates
        self._unfreeze_all_layers()

        # 2. Create optimizer with very specific learning rates for different components
        base_lr = 1e-4
        optimizer = self._create_component_optimizer(
            {
                "vision_encoder_backbone": base_lr * 0.005,  # 5e-7
                "text_encoder_backbone": base_lr * 0.01,  # 1e-6
                "projection": base_lr * 0.1,  # 1e-5
                "cross_attention": base_lr * 0.2,  # 2e-5
                "fusion": base_lr * 0.2,  # 2e-5
                "default": base_lr * 0.01,  # 1e-6 for any other params
            }
        )

        # 3. Create learning rate scheduler with warmup and 2 cycles
        total_steps = len(self.train_dataloader) * self.stage3_epochs
        warmup_steps = int(0.05 * total_steps)  # 5% warmup

        # Custom cosine scheduler with 2 cycles and restarts
        scheduler = CosineAnnealingWarmRestarts(
            optimizer,
            T_0=len(self.train_dataloader)
            * (self.stage3_epochs // 2),  # Restart halfway
            T_mult=1,
            eta_min=base_lr * 0.001,  # Minimum learning rate
        )

        # 4. Create combined loss function for Stage 3
        # Use memory queue contrastive loss with increased queue size
        memory_queue_loss = MemoryQueueContrastiveLoss(
            dim=self.model.projection_dim, queue_size=8192, temperature=0.07
        )

        # Create hard negative mining loss
        hard_negative_loss = HardNegativeMiningContrastiveLoss(
            temperature=0.07,
            hard_negative_factor=0.5,
            mining_strategy="semi-hard",
            dim=self.model.projection_dim,
        )

        # 5. Create and run trainer for Stage 3
        trainer = MultimodalTrainer(
            model=self.model,
            train_dataloader=self.train_dataloader,
            val_dataloader=self.val_dataloader,
            optimizer=optimizer,
            scheduler=scheduler,
            loss_fn=memory_queue_loss,
            num_epochs=self.stage3_epochs,
            checkpoint_dir=os.path.join(self.checkpoint_dir, "stage3"),
            log_dir=os.path.join(self.log_dir, "stage3"),
            device=self.device,
            accumulation_steps=2,
            evaluation_steps=50,
            log_steps=25,
            clip_grad_norm=1.0,
        )

        # Run training
        trainer.train()

        # Save Stage 3 checkpoint (final model)
        stage3_checkpoint_path = os.path.join(self.checkpoint_dir, "stage3_complete.pt")
        torch.save(
            {
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "epoch": self.stage3_epochs,
            },
            stage3_checkpoint_path,
        )

        # Save metrics
        self.history["stage3"] = trainer.history
        self._save_history()

        logger.info(f"Stage 3 complete! Final model saved to {stage3_checkpoint_path}")

    def train_all_stages(self):
        """
        Run all three training stages sequentially.
        """
        logger.info(
            f"Starting multistage training with {self.stage1_epochs + self.stage2_epochs + self.stage3_epochs} total epochs"
        )
        self.train_stage1()
        self.train_stage2()
        self.train_stage3()
        logger.info("Multistage training complete!")
        self.plot_training_history()

    def _freeze_layers(self, component_name: str):
        """
        Freeze layers by component name.

        Args:
            component_name: Name of component to freeze
        """
        # Map component name to model attribute or parameters containing that name
        for name, param in self.model.named_parameters():
            if component_name in name:
                param.requires_grad = False

        logger.info(f"Froze parameters containing '{component_name}'")

    def _unfreeze_layers(self, component_name: str):
        """
        Unfreeze layers by component name.

        Args:
            component_name: Name of component to unfreeze
        """
        # Map component name to model attribute or parameters containing that name
        for name, param in self.model.named_parameters():
            if component_name in name:
                param.requires_grad = True

        logger.info(f"Unfroze parameters containing '{component_name}'")

    def _unfreeze_all_layers(self):
        """
        Unfreeze all layers in the model.
        """
        for param in self.model.parameters():
            param.requires_grad = True

        logger.info("Unfroze all parameters")

    def _create_component_optimizer(self, lr_map: Dict[str, float]) -> Optimizer:
        """
        Create optimizer with different learning rates for different components.

        Args:
            lr_map: Dictionary mapping component names to learning rates

        Returns:
            Optimizer with component-specific learning rates
        """
        # Group parameters by component
        param_groups = []
        default_lr = lr_map.get("default", 1e-6)

        # Create parameter groups for each component
        grouped_params = defaultdict(list)
        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue

            # Find which component this parameter belongs to
            component_match = None
            for component_name in lr_map.keys():
                if component_name != "default" and component_name in name:
                    component_match = component_name
                    break

            # Add to appropriate group
            if component_match:
                grouped_params[component_match].append(param)
            else:
                grouped_params["default"].append(param)

        # Create parameter groups with appropriate learning rates
        for component_name, params in grouped_params.items():
            if params:  # Only create group if it has parameters
                lr = lr_map.get(component_name, default_lr)
                param_groups.append(
                    {
                        "params": params,
                        "lr": lr,
                        "name": component_name,
                    }
                )
                logger.info(
                    f"Created parameter group '{component_name}' with {len(params)} parameters at learning rate {lr}"
                )

        # Create Adam optimizer with parameter groups
        optimizer = AdamW(
            param_groups,
            weight_decay=0.01,
            betas=(0.9, 0.999),
            eps=1e-8,
        )

        return optimizer

    def _save_history(self):
        """
        Save training history to file.
        """
        with open(os.path.join(self.log_dir, "training_history.json"), "w") as f:
            # Convert any non-serializable values to strings
            serializable_history = {}
            for stage, data in self.history.items():
                serializable_history[stage] = {}
                for key, values in data.items():
                    if isinstance(values, list) and all(
                        isinstance(x, (int, float)) for x in values
                    ):
                        serializable_history[stage][key] = values
                    elif isinstance(values, dict):
                        serializable_history[stage][key] = {
                            k: v if isinstance(v, (int, float, str, bool)) else str(v)
                            for k, v in values.items()
                        }
                    else:
                        serializable_history[stage][key] = str(values)

            json.dump(serializable_history, f, indent=2)

    def plot_training_history(self):
        """
        Plot training history across all stages.
        """
        # Create figure with subplots
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        stages = ["stage1", "stage2", "stage3"]

        # Plot each stage
        for i, stage in enumerate(stages):
            ax = axes[i]
            data = self.history.get(stage, {})

            # Plot training and validation loss
            train_loss = data.get("train_loss", [])
            val_loss = data.get("val_loss", [])

            if train_loss:
                ax.plot(train_loss, label="Train Loss", color="blue")
            if val_loss:
                ax.plot(val_loss, label="Validation Loss", color="red")

            # Set title and labels
            ax.set_title(f"Stage {i+1} Training")
            ax.set_xlabel("Epoch")
            ax.set_ylabel("Loss")
            ax.legend()
            ax.grid(True, linestyle="--", alpha=0.7)

        plt.tight_layout()
        plt.savefig(os.path.join(self.log_dir, "training_history.png"))
        plt.close()


# Function to create and configure a model for Flickr30k multistage training
def create_flickr30k_multistage_model(
    vision_model: VisionTransformerWrapper,
    text_model: HuggingFaceTextModelWrapper,
    projection_dim: int = 512,
) -> VICRegMultimodalModel:
    """Create a multistage model for Flickr30k training."""
    return VICRegMultimodalModel(
        vision_model=vision_model, text_model=text_model, projection_dim=projection_dim
    )


# Main function to run Flickr30k multistage training
def train_flickr30k_multistage(
    data_root: str,
    output_dir: str = "flickr30k",
    batch_size: int = 64,
    stage1_epochs: int = 30,
    stage2_epochs: int = 15,
    stage3_epochs: int = 15,
    vision_model: str = "ViT-B/16",
    text_model: str = "bert-base-uncased",
    embedding_dim: int = 512,
    use_metadata: bool = True,
):
    """
    Run multistage training on Flickr30k dataset.

    Args:
        data_root: Root directory for Flickr30k dataset
        output_dir: Directory to save checkpoints and logs
        batch_size: Batch size for training
        stage1_epochs: Number of epochs for Stage 1
        stage2_epochs: Number of epochs for Stage 2
        stage3_epochs: Number of epochs for Stage 3
        vision_model: Vision model backbone
        text_model: Text model backbone
        embedding_dim: Dimension of joint embedding space
        use_metadata: Whether to use metadata for advanced batch sampling
    """
    # Initialize models
    vision_model = VisionTransformerWrapper(model_name=vision_model)
    text_model = HuggingFaceTextModelWrapper(model_name=text_model)

    # Create model
    model = create_flickr30k_multistage_model(
        vision_model=vision_model, text_model=text_model, projection_dim=embedding_dim
    )

    # Create trainer
    trainer = FlickrMultistageTrainer(
        model=model,
        data_root=data_root,
        output_dir=output_dir,
        batch_size=batch_size,
        stage1_epochs=stage1_epochs,
        stage2_epochs=stage2_epochs,
        stage3_epochs=stage3_epochs,
        use_metadata=use_metadata,
    )

    # Run all stages
    trainer.train_all_stages()
