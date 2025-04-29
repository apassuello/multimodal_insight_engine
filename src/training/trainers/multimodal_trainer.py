# src/training/multimodal_trainer.py

# Standard library imports
import os
import time
import json
import random
import logging
from collections import defaultdict
from typing import Dict, List, Tuple, Optional, Callable, Union, Any

# Third-party imports
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.adamw import AdamW
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

# Local imports
from src.training.loss import (
    ContrastiveLoss,
    MultiModalMixedContrastiveLoss,
    VICRegLoss,
    MemoryQueueContrastiveLoss,
    HardNegativeMiningContrastiveLoss,
)
from src.training.loss.vicreg_loss import (
    VICRegLoss,
)  # Explicit import for type checking
from src.data.tokenization.tokenizer_metrics import log_tokenizer_evaluation

logger = logging.getLogger(__name__)


class ModalityBalancingScheduler:
    def __init__(self, optimizer, target_ratio=1.0, check_interval=10):
        self.optimizer = optimizer
        self.target_ratio = target_ratio  # Desired text/vision gradient ratio
        self.check_interval = check_interval
        self.vision_grad_history = []
        self.text_grad_history = []
        self.step_count = 0

    def collect_gradient_stats(self, model):
        """Collect gradient statistics for vision and text components"""
        vision_grad_norm = 0.0
        vision_count = 0
        text_grad_norm = 0.0
        text_count = 0

        for name, param in model.named_parameters():
            if param.grad is not None:
                if "vision_model" in name:
                    vision_grad_norm += param.grad.norm().item()
                    vision_count += 1
                elif "text_model" in name:
                    text_grad_norm += param.grad.norm().item()
                    text_count += 1

        # Calculate average gradient norms
        avg_vision_grad = vision_grad_norm / max(1, vision_count)
        avg_text_grad = text_grad_norm / max(1, text_count)

        self.vision_grad_history.append(avg_vision_grad)
        self.text_grad_history.append(avg_text_grad)

        # Keep history limited to recent gradients
        if len(self.vision_grad_history) > 100:
            self.vision_grad_history = self.vision_grad_history[-100:]
            self.text_grad_history = self.text_grad_history[-100:]

        return avg_vision_grad, avg_text_grad

    def step(self, model):
        """Adjust learning rates based on gradient ratio"""
        self.step_count += 1

        # Only adjust every check_interval steps
        if self.step_count % self.check_interval != 0:
            return

        # Get gradient statistics
        avg_vision_grad, avg_text_grad = self.collect_gradient_stats(model)

        # Calculate current ratio
        if avg_vision_grad > 0:
            current_ratio = avg_text_grad / avg_vision_grad
        else:
            current_ratio = self.target_ratio  # Default if no vision gradients

        # Only adjust if ratio is far from target
        if abs(current_ratio - self.target_ratio) > 0.5:
            # Calculate adjustment factor
            adjustment = self.target_ratio / max(current_ratio, 0.1)

            # Limit adjustment to avoid extreme changes
            adjustment = max(0.5, min(2.0, adjustment))

            # Adjust learning rates
            for param_group in self.optimizer.param_groups:
                if "vision_model" in str(param_group.get("name", "")):
                    param_group["lr"] *= adjustment
                elif "text_model" in str(param_group.get("name", "")):
                    param_group["lr"] /= adjustment

            print(
                f"Adjusted learning rates - ratio: {current_ratio:.2f}, target: {self.target_ratio:.2f}"
            )


class MultimodalTrainer:
    """
    Trainer for multimodal models with support for contrastive learning.
    """

    def __init__(
        self,
        model: nn.Module,
        train_dataloader: DataLoader,
        val_dataloader: Optional[DataLoader] = None,
        test_dataloader: Optional[DataLoader] = None,
        optimizer: Optional[Any] = None,
        scheduler: Optional[Any] = None,
        loss_fn: Optional[nn.Module] = None,
        num_epochs: int = 20,
        learning_rate: float = 1e-4,
        weight_decay: float = 0.01,
        warmup_steps: int = 0,
        checkpoint_dir: str = "checkpoints",
        log_dir: str = "logs",
        device: Optional[torch.device] = None,
        mixed_precision: bool = False,
        accumulation_steps: int = 1,
        evaluation_steps: int = 0,  # 0 means evaluate only at the end of each epoch
        log_steps: int = 50,  # Increased from default 10 to reduce logging frequency
        early_stopping_patience: Optional[int] = None,
        clip_grad_norm: Optional[float] = None,
        balance_modality_gradients: bool = False,
        args: Optional[Any] = None,  # Store args for additional configuration
    ):
        """
        Initialize the multimodal trainer.

        Args:
            model: The multimodal model to train
            train_dataloader: DataLoader for training data
            val_dataloader: Optional DataLoader for validation data
            test_dataloader: Optional DataLoader for test data
            optimizer: Optional optimizer (will be created if not provided)
            scheduler: Optional learning rate scheduler
            loss_fn: Optional loss function (defaults to ContrastiveLoss)
            num_epochs: Number of epochs to train
            learning_rate: Learning rate for optimizer (if optimizer not provided)
            weight_decay: Weight decay for optimizer (if optimizer not provided)
            warmup_steps: Number of warmup steps for learning rate
            checkpoint_dir: Directory to save checkpoints
            log_dir: Directory to save logs
            device: Device to train on (auto-detected if not provided)
            mixed_precision: Whether to use mixed precision training
            accumulation_steps: Number of steps to accumulate gradients over
            evaluation_steps: Number of steps between evaluations during training (0 means only at end of epoch)
            log_steps: Number of steps between logging during training
            early_stopping_patience: Number of evaluations with no improvement to trigger early stopping
            clip_grad_norm: Maximum norm for gradient clipping
            balance_modality_gradients: Whether to balance gradients between modalities
            args: Original argument namespace containing full training configuration
        """
        self.model = model
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.test_dataloader = test_dataloader
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.loss_fn = loss_fn
        self.num_epochs = num_epochs
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.warmup_steps = warmup_steps
        self.checkpoint_dir = checkpoint_dir
        self.log_dir = log_dir
        self.mixed_precision = mixed_precision
        self.accumulation_steps = accumulation_steps
        self.evaluation_steps = evaluation_steps
        self.log_steps = log_steps
        self.early_stopping_patience = early_stopping_patience
        self.clip_grad_norm = clip_grad_norm
        self.balance_modality_gradients = balance_modality_gradients
        self.args = args  # Store args for later reference

        # Create directories
        os.makedirs(checkpoint_dir, exist_ok=True)
        os.makedirs(log_dir, exist_ok=True)

        # Set device
        if device is None:
            if torch.cuda.is_available():
                self.device = torch.device("cuda")
            elif (
                hasattr(torch, "backends")
                and hasattr(torch.backends, "mps")
                and torch.backends.mps.is_available()
            ):
                self.device = torch.device("mps")
            else:
                self.device = torch.device("cpu")
        else:
            self.device = device

        logger.info(f"Initialized MultimodalTrainer with device: {self.device}")

        # Initialize model on device with comprehensive device handling
        self.model = self.model.to(self.device)

        # Ensure all model parameters are on the same device
        # This is especially important for complex multimodal models that may have
        # submodules with different devices, especially on MPS systems
        self._ensure_model_on_device()

        # Initialize optimizer if not provided
        if self.optimizer is None:
            self.optimizer = AdamW(
                self.model.parameters(),
                lr=self.learning_rate,
                weight_decay=self.weight_decay,
            )
        self.grad_scheduler = None
        if self.balance_modality_gradients:
            self.grad_scheduler = ModalityBalancingScheduler(
                self.optimizer, target_ratio=1.0
            )
        # Initialize loss function if not provided
        if self.loss_fn is None:
            # Directly get the model fusion dimension (which should now be correctly set in model_factory.py)
            # This is the most reliable way to get the right dimension since it's been explicitly aligned
            if hasattr(self.model, "fusion_module") and hasattr(
                self.model.fusion_module, "fusion_dim"
            ):
                model_dim = self.model.fusion_module.fusion_dim
                print(f"Using fusion dimension from model: {model_dim}")
            else:
                # Get the actual model dimensions for proper projection setup as fallback
                vision_dim = None
                text_dim = None

                # For timm models
                if hasattr(self.model, "vision_model") and hasattr(
                    self.model.vision_model, "num_features"
                ):
                    vision_dim = self.model.vision_model.num_features
                    print(f"Using vision dimension from timm model: {vision_dim}")

                # For custom models with embed_dim
                elif hasattr(self.model, "vision_model") and hasattr(
                    self.model.vision_model, "embed_dim"
                ):
                    vision_dim = self.model.vision_model.embed_dim
                    print(f"Using vision dimension from embed_dim: {vision_dim}")

                # For HuggingFace text models
                if (
                    hasattr(self.model, "text_model")
                    and hasattr(self.model.text_model, "encoder")
                    and hasattr(self.model.text_model.encoder, "config")
                    and hasattr(self.model.text_model.encoder.config, "hidden_size")
                ):
                    text_dim = self.model.text_model.encoder.config.hidden_size
                    print(f"Using text dimension from HuggingFace config: {text_dim}")

                # For custom text models with d_model
                elif hasattr(self.model, "text_model") and hasattr(
                    self.model.text_model, "d_model"
                ):
                    text_dim = self.model.text_model.d_model
                    print(f"Using text dimension from d_model: {text_dim}")

                # Choose appropriate dimension
                if vision_dim is not None and text_dim is not None:
                    # If both are available, need to match what the projection layer does
                    model_dim = vision_dim  # Use vision dimension as that's used by the integration layer
                    logger.info(
                        f"Using vision dimension {model_dim} for contrastive loss"
                    )
                elif vision_dim is not None:
                    model_dim = vision_dim
                    logger.info(
                        f"Using vision dimension {model_dim} for contrastive loss"
                    )
                elif text_dim is not None:
                    model_dim = text_dim
                    logger.info(
                        f"Using text dimension {model_dim} for contrastive loss"
                    )
                else:
                    # Default fallback
                    model_dim = 768  # Modern vision transformers use 768
                    logger.info(
                        f"Using default dimension {model_dim} for contrastive loss"
                    )

            # Create a proper mixed contrastive loss with correct dimensions
            logger.info(
                f"Creating MultiModalMixedContrastiveLoss with dimension {model_dim}"
            )
            self.loss_fn = MultiModalMixedContrastiveLoss(
                temperature=0.2,
                dim=model_dim,  # Use detected model dimension
                # Add projections with dimensions aligned to model
                contrastive_weight=1.0,
                classification_weight=0.5,
                multimodal_matching_weight=0.2,
                use_hard_negatives=True,
                hard_negative_weight=0.3,
            )

        # Initialize mixed precision scaler if needed
        self.scaler = None
        if self.mixed_precision and torch.cuda.is_available():
            self.scaler = torch.cuda.amp.GradScaler()
        logger.info(f"Mixed precision: {self.mixed_precision}")

        # Initialize counters
        self.global_step = 0
        self.current_epoch = 0
        self.best_val_metric = 0.0
        self.patience_counter = 0

        # Initialize training start time
        self.start_time = time.time()

        # Log accumulation steps
        if self.accumulation_steps > 1:
            logger.info(f"Accumulation steps: {self.accumulation_steps}")

        # Initialize metric storage
        self.history = defaultdict(list)

        # Debug tracking variables
        self._debug_feature_source = None
        self._debug_match_id_source = None

    def _ensure_model_on_device(self):
        """
        Ensure all model components are on the correct device.
        This is critically important for models with multiple components
        like multimodal models, especially when using MPS device.
        """
        # First check if model is an instance of nn.Module
        if not isinstance(self.model, nn.Module):
            logger.warning(
                f"Model is not an instance of nn.Module, cannot ensure device consistency"
            )
            return

        # Helper for explicitly moving submodules when needed
        def move_submodules(module_name, module):
            # Check if this module has parameters
            if list(module.parameters()):
                # Check if any parameter is on a different device
                param_device = next(module.parameters()).device
                if param_device != self.device:
                    # logger.warning(
                    #     f"Moving {module_name} from {param_device} to {self.device}"
                    # )
                    try:
                        module.to(self.device)
                    except Exception as e:
                        logger.error(
                            f"Error moving {module_name} to {self.device}: {str(e)}"
                        )

        # Start with the main model modules
        # First, check if the model has vision/text/fusion components (common in multimodal models)
        if hasattr(self.model, "vision_model"):
            move_submodules("vision_model", self.model.vision_model)

        if hasattr(self.model, "text_model"):
            move_submodules("text_model", self.model.text_model)

        if hasattr(self.model, "fusion_module"):
            move_submodules("fusion_module", self.model.fusion_module)

        # Handle other common components
        for name, module in self.model.named_children():
            move_submodules(name, module)

        # Verify all parameters are on the correct device
        for name, param in self.model.named_parameters():
            if param.device != self.device:
                # logger.warning(
                #     f"Parameter {name} is on {param.device}, expected {self.device}"
                # )
                try:
                    # Try to move this specific parameter
                    param.data = param.data.to(self.device)
                except Exception as e:
                    logger.error(
                        f"Error moving parameter {name} to {self.device}: {str(e)}"
                    )

        # Final verification
        devices = {param.device for param in self.model.parameters()}
        if len(devices) > 1:
            logger.warning(f"Model parameters are on multiple devices: {devices}")
        # else:
        #     logger.info(f"All model parameters are on device: {self.device}")

    def diagnose_training_issues(self) -> str:
        """
        Perform diagnostics to identify potential training issues based on collected metrics.

        Returns:
            String with diagnostic results and recommendations
        """
        issues = []
        recommendations = []

        # Check if we have training history
        if (
            not self.history
            or "train_loss" not in self.history
            or not self.history["train_loss"]
        ):
            return "Insufficient training history for diagnosis. Run at least one epoch first."

        # Check for persistently high loss values
        losses = self.history["train_loss"]
        if len(losses) >= 3 and all(l > 4.0 for l in losses[-3:]):
            issues.append("Persistently high loss values (> 4.0)")
            recommendations.append(
                "- Try reducing learning rate by 10x\n"
                "- Verify that match_ids are correctly grouped in your dataset\n"
                "- Check feature dimensions match between model and loss function"
            )

        # Check for NaN losses
        if any(np.isnan(l) for l in losses):
            issues.append("NaN losses detected")
            recommendations.append(
                "- Add gradient clipping (--clip_grad_norm 1.0)\n"
                "- Check for NaN in input features or labels\n"
                "- Reduce learning rate significantly"
            )

        # Check for zero/near-zero accuracy
        if "train_accuracy" in self.history and self.history["train_accuracy"]:
            accuracies = self.history["train_accuracy"]
            if all(acc < 0.01 for acc in accuracies[-3:]):
                issues.append("Near-zero accuracy values")
                recommendations.append(
                    "- Verify the semantic batch sampler is working (check logs for 'Match IDs')\n"
                    "- Check loss function temperature parameter (try increasing to 0.1-0.2)\n"
                    "- Ensure initialization provides sufficient feature diversity"
                )

        # Check for dimension mismatch
        if hasattr(self.loss_fn, "dim") and hasattr(self.model, "vision_model"):
            loss_dim = self.loss_fn.dim
            vision_dim = None

            # Try to extract vision dimension from model
            if hasattr(self.model.vision_model, "num_features"):
                vision_dim = self.model.vision_model.num_features
            elif hasattr(self.model.vision_model, "embed_dim"):
                vision_dim = self.model.vision_model.embed_dim

            if vision_dim is not None and vision_dim != loss_dim:
                issues.append(
                    f"Dimension mismatch between model ({vision_dim}) and loss function ({loss_dim})"
                )
                recommendations.append(
                    f"- Update loss function dimension to match model: {vision_dim}\n"
                    f"- Alternatively, add projection layers to match dimensions"
                )

        # Compile diagnostic report
        if not issues:
            return "No obvious training issues detected from metrics."

        report = "Training Diagnostic Report:\n\n"
        report += "Issues Detected:\n"
        for i, issue in enumerate(issues, 1):
            report += f"{i}. {issue}\n"

        report += "\nRecommendations:\n"
        for i, rec in enumerate(recommendations, 1):
            report += f"{i}. {rec}\n"

        return report

    def train(self) -> Dict[str, List[float]]:
        """
        Train the model for specified number of epochs.

        Returns:
            Dictionary with training history
        """
        logger.info(f"Starting training for {self.num_epochs} epochs")

        # Initialize history
        history = defaultdict(list)

        # Early stopping variables
        best_val_metric = float("inf")
        early_stopping_counter = 0

        # Training loop
        for epoch in range(self.num_epochs):
            self.current_epoch = epoch

            # Train for one epoch
            train_metrics = self.train_epoch()

            # Log training metrics
            self._log_metrics(train_metrics, "train")

            # Evaluate on validation set if available
            if self.val_dataloader:
                val_metrics = self.evaluate(self.val_dataloader)
                self._log_metrics(val_metrics, "val")

                # Update history
                for k, v in val_metrics.items():
                    history[f"val_{k}"].append(v)

                # Check for early stopping
                if self.early_stopping_patience is not None:
                    validation_metric = val_metrics.get("loss", float("inf"))
                    if validation_metric < best_val_metric:
                        best_val_metric = validation_metric
                        early_stopping_counter = 0
                        # Save best model
                        self.save_checkpoint(
                            os.path.join(self.checkpoint_dir, "best_model.pt")
                        )
                    else:
                        early_stopping_counter += 1
                        logger.info(
                            f"Validation metric did not improve, counter: {early_stopping_counter}/{self.early_stopping_patience}"
                        )
                        if early_stopping_counter >= self.early_stopping_patience:
                            logger.info(
                                f"Early stopping triggered after {epoch + 1} epochs"
                            )
                            break

            # Update history with training metrics
            for k, v in train_metrics.items():
                history[f"train_{k}"].append(v)

            # Save checkpoint for this epoch
            self.save_checkpoint(
                os.path.join(self.checkpoint_dir, f"checkpoint_epoch_{epoch + 1}.pt")
            )

            # After each epoch, run diagnostics if we've completed at least 1 epoch
            if epoch > 0:
                logger.info("Running training diagnostics...")
                diagnostic_report = self.diagnose_training_issues()
                logger.info(f"\n{diagnostic_report}")

                # Evaluate tokenizer quality (once per epoch)
                self._evaluate_tokenizer_quality(epoch)

                # Plot alignment progress after each epoch
                if (
                    hasattr(self, "_alignment_history")
                    and self._alignment_history["step"]
                ):
                    # Create epochwise alignment plots directory
                    alignment_plot_dir = os.path.join(self.log_dir, "alignment_plots")
                    os.makedirs(alignment_plot_dir, exist_ok=True)

                    # Generate alignment plot
                    plt.figure(figsize=(12, 8))

                    # Plot diagonal similarity vs. mean similarity
                    plt.subplot(2, 1, 1)
                    steps = self._alignment_history["step"]
                    plt.plot(
                        steps,
                        self._alignment_history["diag_mean"],
                        label="Diagonal Similarity",
                        color="blue",
                    )
                    plt.plot(
                        steps,
                        self._alignment_history["sim_mean"],
                        label="Mean Similarity",
                        color="red",
                        linestyle="--",
                    )
                    plt.title(f"Semantic Alignment Progress (Epoch {epoch+1})")
                    plt.xlabel("Training Steps")
                    plt.ylabel("Cosine Similarity")
                    plt.legend()
                    plt.grid(True)

                    # Plot alignment gap and SNR
                    plt.subplot(2, 1, 2)
                    plt.plot(
                        steps,
                        self._alignment_history["alignment_gap"],
                        label="Alignment Gap",
                        color="green",
                    )
                    plt.plot(
                        steps,
                        self._alignment_history["alignment_snr"],
                        label="Signal-to-Noise Ratio",
                        color="purple",
                        linestyle="--",
                    )
                    plt.title("Alignment Quality Metrics")
                    plt.xlabel("Training Steps")
                    plt.ylabel("Metric Value")
                    plt.legend()
                    plt.grid(True)

                    plt.tight_layout()
                    plt.savefig(
                        os.path.join(
                            alignment_plot_dir, f"alignment_epoch_{epoch+1}.png"
                        )
                    )
                    plt.close()

                    logger.info(f"Saved alignment progress plot for epoch {epoch+1}")

        # Training completed
        logger.info(
            f"Training completed in {time.time() - self.start_time:.2f} seconds"
        )

        return dict(history)

    def train_multistage(self):
        """
        Implement multi-stage training for multimodal models.

        Stage 1: Train modality-specific components
        Stage 2: Train cross-modal fusion components
        Stage 3: Fine-tune the full model with hard negative mining
        """
        logger.info("Starting multi-stage training...")

        # Save initial model state for reference - use the full path
        initial_checkpoint_path = os.path.join(
            self.checkpoint_dir, "initial_checkpoint.pt"
        )
        self.save_checkpoint(initial_checkpoint_path)

        # Store original configuration
        original_optimizer = self.optimizer
        original_loss_fn = self.loss_fn
        original_num_epochs = self.num_epochs

        # Get total epochs and divide among stages
        total_epochs = self.num_epochs
        stage1_epochs = max(2, total_epochs // 3)
        stage2_epochs = max(2, total_epochs // 3)
        stage3_epochs = total_epochs - stage1_epochs - stage2_epochs

        # === Stage 1: Train modality-specific projections ===
        logger.info(
            f"=== Stage 1: Training modality-specific projections ({stage1_epochs} epochs) ==="
        )

        # Stage 1: Freeze base models entirely, only train projection layers
        # This is the typical first stage in progressive unfreezing
        for name, param in self.model.named_parameters():
            # Freeze base vision and text models
            if any(x in name for x in ["vision_model", "text_model"]):
                param.requires_grad = False
            # Freeze cross-attention and fusion layers
            elif any(x in name for x in ["cross", "fusion", "gate"]): 
                param.requires_grad = False
            # Only train projection layers and other non-base components
            else:
                param.requires_grad = True
                
        logger.info("Stage 1: Base models frozen, training only projection layers")

        # Create optimizer for Stage 1
        stage1_optimizer = torch.optim.AdamW(
            [
                {
                    "params": [
                        p
                        for n, p in self.model.named_parameters()
                        if "vision_model" in n and p.requires_grad
                    ],
                    "lr": self.learning_rate * 0.2,  # Increase from 0.1 to 0.2
                },
                {
                    "params": [
                        p
                        for n, p in self.model.named_parameters()
                        if "text_model" in n and p.requires_grad
                    ],
                    "lr": self.learning_rate * 0.1,  # Keep as is
                },
                {
                    "params": [
                        p
                        for n, p in self.model.named_parameters()
                        if not any(
                            x in n
                            for x in [
                                "vision_model",
                                "text_model",
                                "cross",
                                "fusion",
                                "gate",
                            ]
                        )
                        and p.requires_grad
                    ],
                    "lr": self.learning_rate,
                },
            ],
            weight_decay=self.weight_decay,
        )

        # Use standard contrastive loss for Stage 1
        # Get the actual model dimensions for proper projection setup
        # First try to get vision dimension directly
        vision_dim = None
        text_dim = None

        # For timm models
        if hasattr(self.model, "vision_model") and hasattr(
            self.model.vision_model, "num_features"
        ):
            vision_dim = self.model.vision_model.num_features
            print(f"Using vision dimension from timm model: {vision_dim}")

        # For custom models with embed_dim
        elif hasattr(self.model, "vision_model") and hasattr(
            self.model.vision_model, "embed_dim"
        ):
            vision_dim = self.model.vision_model.embed_dim
            print(f"Using vision dimension from embed_dim: {vision_dim}")

        # For HuggingFace text models
        if (
            hasattr(self.model, "text_model")
            and hasattr(self.model.text_model, "encoder")
            and hasattr(self.model.text_model.encoder, "config")
            and hasattr(self.model.text_model.encoder.config, "hidden_size")
        ):
            text_dim = self.model.text_model.encoder.config.hidden_size
            print(f"Using text dimension from HuggingFace config: {text_dim}")

        # For custom text models with d_model
        elif hasattr(self.model, "text_model") and hasattr(
            self.model.text_model, "d_model"
        ):
            text_dim = self.model.text_model.d_model
            print(f"Using text dimension from d_model: {text_dim}")

        # Choose appropriate dimension
        if vision_dim is not None and text_dim is not None:
            # If both are available, need to match what the projection layer does
            model_dim = vision_dim  # Use vision dimension as that's used by the integration layer
            logger.info(f"Using vision dimension {model_dim} for contrastive loss")
        elif vision_dim is not None:
            model_dim = vision_dim
            logger.info(f"Using vision dimension {model_dim} for contrastive loss")
        elif text_dim is not None:
            model_dim = text_dim
            logger.info(f"Using text dimension {model_dim} for contrastive loss")
        else:
            # Default fallback
            model_dim = 768  # Modern vision transformers use 768
            logger.info(f"Using default dimension {model_dim} for contrastive loss")

        # Create loss with proper dimension settings
        # Check if we should use VICReg loss based on original loss type
        if isinstance(self.loss_fn, VICRegLoss):
            # Use VICReg loss with higher variance coefficient for stage 1
            logger.info(
                "Using VICRegLoss for Stage 1 with higher variance regularization"
            )
            stage1_loss = VICRegLoss(
                sim_coeff=5.0,
                var_coeff=5.0,  # Higher variance coefficient for Stage 1
                cov_coeff=1.0,
                epsilon=1e-3,
            )
        else:
            # Default to standard contrastive loss
            # For ViT-base/BERT-base compatibility, don't reduce dimension
            if model_dim == 768:  # ViT-base/BERT-base case
                print(
                    f"Found ViT-base/BERT-base model (dim={model_dim}) - using matching projection dimension"
                )
                stage1_loss = ContrastiveLoss(
                    temperature=0.5,
                    input_dim=model_dim,  # Use actual model dimension
                    add_projection=False,  # IMPORTANT: Skip projection for 768 dimension models
                )
            else:
                # For other models, use reasonable projection dimension
                stage1_loss = ContrastiveLoss(
                    temperature=0.5,
                    input_dim=model_dim,  # Use actual model dimension
                    add_projection=True,
                    projection_dim=model_dim
                    // 2,  # Half of the model dimension (not quarter)
                )

        # Set up for Stage 1
        self.optimizer = stage1_optimizer  # Replace optimizer
        self.loss_fn = stage1_loss  # Replace loss function
        self.num_epochs = stage1_epochs  # Set epochs for this stage

        # Train Stage 1
        self.train()  # Use the existing train method

        # Save stage 1 model
        stage1_checkpoint_path = os.path.join(
            self.checkpoint_dir, "stage1_checkpoint.pt"
        )
        self.save_checkpoint(stage1_checkpoint_path)

        # === Stage 2: Unfreeze top layers of base models + fusion ===
        logger.info(
            f"=== Stage 2: Training top layers of base models + fusion ({stage2_epochs} epochs) ==="
        )

        # Stage 2: Unfreeze last few layers of base models + fusion components
        # This is the typical second stage in progressive unfreezing
        for name, param in self.model.named_parameters():
            # Selectively unfreeze top layers of vision model
            if "vision_model" in name:
                if any(f"layer.{i}" in name for i in range(9, 12)) or "pooler" in name:
                    # Unfreeze only layers 9-11 and pooler (last 3 layers)
                    param.requires_grad = True
                else:
                    # Keep earlier layers frozen
                    param.requires_grad = False
            # Selectively unfreeze top layers of text model
            elif "text_model" in name:
                if any(f"layer.{i}" in name for i in range(9, 12)) or "pooler" in name:
                    # Unfreeze only layers 9-11 and pooler (last 3 layers)
                    param.requires_grad = True
                else:
                    # Keep earlier layers frozen
                    param.requires_grad = False
            # Unfreeze all fusion components
            else:
                param.requires_grad = True
                
        logger.info("Stage 2: Unfrozen top 3 layers of base models + all fusion components")

        # Create optimizer for Stage 2 with separate learning rates
        stage2_optimizer = torch.optim.AdamW(
            [
                {
                    "params": [
                        p
                        for n, p in self.model.named_parameters()
                        if "vision_model" in n and p.requires_grad
                    ],
                    "lr": self.learning_rate * 0.01,  # Very small learning rate for vision model layers
                    "name": "vision_model"  # Add name for gradient scheduler
                },
                {
                    "params": [
                        p
                        for n, p in self.model.named_parameters()
                        if "text_model" in n and p.requires_grad
                    ],
                    "lr": self.learning_rate * 0.01,  # Very small learning rate for text model layers
                    "name": "text_model"  # Add name for gradient scheduler
                },
                {
                    "params": [
                        p
                        for n, p in self.model.named_parameters()
                        if any(x in n for x in ["cross", "fusion", "gate"])
                        and p.requires_grad
                    ],
                    "lr": self.learning_rate * 0.1,  # 10% of base learning rate for fusion
                    "name": "fusion_components"  # Add name for gradient scheduler
                },
                {
                    "params": [
                        p
                        for n, p in self.model.named_parameters()
                        if not any(x in n for x in ["vision_model", "text_model", "cross", "fusion", "gate"])
                        and p.requires_grad
                    ],
                    "lr": self.learning_rate,  # Full learning rate for projection layers
                    "name": "projection_layers"  # Add name for gradient scheduler
                }
            ],
            weight_decay=self.weight_decay,
        )

        # Use memory queue contrastive loss for Stage 2
        # Re-detect model dimension for Stage 2 (in case it changed)
        if vision_dim is not None:
            model_dim = vision_dim

        stage2_loss = MemoryQueueContrastiveLoss(
            dim=model_dim,  # Use the model dimension detected earlier
            queue_size=8192,
            temperature=0.07,
        )

        # Set up for Stage 2
        self.optimizer = stage2_optimizer  # Replace optimizer
        self.loss_fn = stage2_loss  # Replace loss function
        self.num_epochs = stage2_epochs  # Set epochs for this stage

        # Train Stage 2
        self.train()  # Use the existing train method

        # Save stage 2 model
        stage2_checkpoint_path = os.path.join(
            self.checkpoint_dir, "stage2_checkpoint.pt"
        )
        self.save_checkpoint(stage2_checkpoint_path)

        # === Stage 3: Full fine-tuning with all layers unfrozen ===
        logger.info(
            f"=== Stage 3: Full fine-tuning with all layers unfrozen ({stage3_epochs} epochs) ==="
        )

        # Unfreeze everything for full fine-tuning
        # This is the typical final stage in progressive unfreezing
        for param in self.model.parameters():
            param.requires_grad = True
            
        logger.info("Stage 3: All layers unfrozen for final fine-tuning with differential learning rates")

        # Create optimizer for Stage 3 with layer-wise learning rates
        stage3_optimizer = torch.optim.AdamW(
            [
                {
                    "params": [
                        p
                        for n, p in self.model.named_parameters()
                        if "vision_model" in n and p.requires_grad
                    ],
                    "lr": self.learning_rate * 0.02,  # Increase from 0.01 to 0.02
                },
                {
                    "params": [
                        p
                        for n, p in self.model.named_parameters()
                        if "text_model" in n and p.requires_grad
                    ],
                    "lr": self.learning_rate * 0.01,  # Keep as is
                },
                {
                    "params": [
                        p
                        for n, p in self.model.named_parameters()
                        if any(x in n for x in ["cross", "fusion", "gate"])
                        and p.requires_grad
                    ],
                    "lr": self.learning_rate * 0.1,
                },
                {
                    "params": [
                        p
                        for n, p in self.model.named_parameters()
                        if not any(
                            x in n
                            for x in [
                                "vision_model",
                                "text_model",
                                "cross",
                                "fusion",
                                "gate",
                            ]
                        )
                        and p.requires_grad
                    ],
                    "lr": self.learning_rate,
                },
            ],
            weight_decay=self.weight_decay,
        )

        # Use hard negative mining loss for Stage 3
        stage3_loss = HardNegativeMiningContrastiveLoss(
            temperature=0.07,
            hard_negative_factor=2.0,
            mining_strategy="semi-hard",
            dim=model_dim,  # Use the model dimension detected earlier
        )

        # Set up for Stage 3
        self.optimizer = stage3_optimizer  # Replace optimizer
        self.loss_fn = stage3_loss  # Replace loss function
        self.num_epochs = stage3_epochs  # Set epochs for this stage

        # Train Stage 3
        self.train()  # Use the existing train method

        # Save final model
        final_checkpoint_path = os.path.join(self.checkpoint_dir, "final_checkpoint.pt")
        self.save_checkpoint(final_checkpoint_path)

        logger.info("Multi-stage training completed successfully!")

        # Restore original configuration in case needed later
        self.optimizer = original_optimizer
        self.loss_fn = original_loss_fn
        self.num_epochs = original_num_epochs

    def train_epoch(self) -> Dict[str, float]:
        """
        Train for one epoch.

        Returns:
            Dictionary with epoch metrics
        """
        self.model.train()

        epoch_metrics = defaultdict(float)
        nested_metrics = {}  # For storing nested dictionary metrics
        num_batches = 0
        total_loss = 0.0

        # Update VICReg or HybridPretrainVICReg loss epoch counter if applicable
        if hasattr(self.loss_fn, "update_epoch"):
            self.loss_fn.update_epoch(self.current_epoch)
            # Also set total steps for better warm-up calculation
            total_steps_estimate = len(self.train_dataloader) * self.num_epochs

            if hasattr(self.loss_fn, "update_step"):
                self.loss_fn.update_step(self.global_step, total_steps_estimate)

                # Log phase information
                if hasattr(self.loss_fn, "current_phase"):
                    phase = getattr(self.loss_fn, "current_phase", "unknown")
                    logger.info(
                        f"Training curriculum: phase={phase}, epoch={self.current_epoch}, global_step={self.global_step}"
                    )
                else:
                    logger.info(
                        f"VICReg curriculum: epoch={self.current_epoch}, global_step={self.global_step}"
                    )

        # Use tqdm for progress bar
        pbar = tqdm(
            self.train_dataloader,
            desc=f"Epoch {self.current_epoch + 1}/{self.num_epochs}",
        )

        # Reset gradients
        self.optimizer.zero_grad()

        for batch_idx, batch in enumerate(pbar):
            # Move batch to device
            batch = self._to_device(batch)

            # Extract only the inputs expected by the model
            model_inputs = self._prepare_model_inputs(batch)

            # Forward pass with mixed precision if enabled
            if self.mixed_precision and self.scaler is not None:
                with torch.cuda.amp.autocast():
                    outputs = self.model(**model_inputs)
                    loss_inputs = self._prepare_loss_inputs(batch, outputs)
                    loss_dict = self.loss_fn(**loss_inputs)
                    loss = loss_dict["loss"]
            else:
                outputs = self.model(**model_inputs)
                loss_inputs = self._prepare_loss_inputs(batch, outputs)
                loss_dict = self.loss_fn(**loss_inputs)
                loss = loss_dict["loss"]

            # Scale loss by accumulation steps
            loss = loss / self.accumulation_steps

            # Backward pass with mixed precision if enabled
            if self.mixed_precision and self.scaler is not None:
                self.scaler.scale(loss).backward()
            else:
                loss.backward()

            if (
                hasattr(self, "grad_scheduler")
                and self.grad_scheduler is not None
                and batch_idx % 10 == 0
            ):
                self.grad_scheduler.step(self.model)
            # Update metrics and analyze loss values for anomalies
            loss_value = loss.item() * self.accumulation_steps
            total_loss += loss_value

            # Check for problematic loss values
            if torch.isnan(loss).any().item():
                logger.error(f"NaN LOSS DETECTED at batch {batch_idx}!")
                logger.error(
                    f"Loss breakdown: {', '.join([f'{k}={v.item() if isinstance(v, torch.Tensor) else v:.4f}' for k, v in loss_dict.items() if k != 'loss'])}"
                )
            elif torch.isinf(loss).any().item():
                logger.error(f"Infinite LOSS DETECTED at batch {batch_idx}!")
                logger.error(
                    f"Loss breakdown: {', '.join([f'{k}={v.item() if isinstance(v, torch.Tensor) else v:.4f}' for k, v in loss_dict.items() if k != 'loss'])}"
                )
            elif (
                loss_value > 8.0
            ):  # High loss value threshold based on logs (~4.85 seen in training)
                logger.warning(
                    f"Unusually HIGH LOSS: {loss_value:.4f} at batch {batch_idx}"
                )
                # Log each loss component separately for analysis
                for k, v in loss_dict.items():
                    if k != "loss" and not isinstance(v, dict):
                        logger.info(
                            f"  {k}: {v.item() if isinstance(v, torch.Tensor) else v:.4f}"
                        )

                # If this is a contrastive loss, check the feature similarity distribution
                if "v2t_loss" in loss_dict and "t2v_loss" in loss_dict:
                    logger.info(
                        f"  v2t_loss: {loss_dict['v2t_loss']:.4f}, t2v_loss: {loss_dict['t2v_loss']:.4f}"
                    )

                    # Check if accuracy is very low, suggesting random performance
                    if "accuracy" in loss_dict and loss_dict["accuracy"] < 0.01:
                        logger.error(
                            "NEAR-ZERO ACCURACY DETECTED - model is likely not learning!"
                        )

            elif loss_value < 1e-6 and loss_value > 0:
                logger.warning(
                    f"Unusually LOW LOSS: {loss_value:.8f} at batch {batch_idx} - model may have converged too quickly or loss computation issue"
                )
            num_batches += 1

            # DEBUG: Check for feature collapse (every 5 batches to avoid excessive logging)
            if batch_idx % 5 == 0:
                # Get features
                vision_features = loss_inputs.get("vision_features")
                text_features = loss_inputs.get("text_features")

                if vision_features is not None and text_features is not None:
                    # Check feature variance
                    vision_var = torch.var(vision_features, dim=0).mean().item()
                    text_var = torch.var(text_features, dim=0).mean().item()
                    logger.info(
                        f"Feature variance - Vision: {vision_var:.6f}, Text: {text_var:.6f}"
                    )

                    # Check similarity distribution
                    similarity = torch.matmul(vision_features, text_features.T)
                    sim_mean = similarity.mean().item()
                    sim_std = similarity.std().item()
                    logger.info(
                        f"Similarity stats - Mean: {sim_mean:.4f}, Std: {sim_std:.4f}"
                    )

                    # Feature collapse warning
                    if vision_var < 1e-4 or text_var < 1e-4:
                        logger.warning(
                            f"FEATURE COLLAPSE DETECTED! Vision var: {vision_var:.6f}, Text var: {text_var:.6f}"
                        )

            # ADVANCED GRADIENT DIAGNOSTICS (at accumulation step boundaries)
            if (batch_idx + 1) % self.accumulation_steps == 0:
                total_grad_norm = 0.0
                param_count = 0
                model_component_grads = {}

                # Track key model components for focused gradient analysis
                component_prefixes = [
                    "vision_model",
                    "text_model",
                    "fusion_module",
                    "vision_projection",
                    "text_projection",
                    "cross_attention",
                ]

                for prefix in component_prefixes:
                    model_component_grads[prefix] = {
                        "count": 0,
                        "total_norm": 0.0,
                        "max_norm": 0.0,
                        "min_norm": float("inf"),
                        "has_zero": False,
                        "has_nan": False,
                    }

                # Analyze gradients by model component
                for name, param in self.model.named_parameters():
                    if param.grad is not None:
                        # Calculate gradient statistics
                        param_norm = param.grad.norm().item()
                        total_grad_norm += param_norm
                        param_count += 1

                        # Check for problematic gradients
                        has_nan = torch.isnan(param.grad).any().item()
                        has_inf = torch.isinf(param.grad).any().item()
                        is_zero = (param.grad == 0).all().item()

                        # Categorize by model component
                        component_match = False
                        for prefix in component_prefixes:
                            if prefix in name:
                                component_match = True
                                model_component_grads[prefix]["count"] += 1
                                model_component_grads[prefix][
                                    "total_norm"
                                ] += param_norm
                                model_component_grads[prefix]["max_norm"] = max(
                                    model_component_grads[prefix]["max_norm"],
                                    param_norm,
                                )

                                if param_norm > 0:
                                    model_component_grads[prefix]["min_norm"] = min(
                                        model_component_grads[prefix]["min_norm"],
                                        param_norm,
                                    )

                                if is_zero:
                                    model_component_grads[prefix]["has_zero"] = True

                                if has_nan or has_inf:
                                    model_component_grads[prefix]["has_nan"] = True

                        # Check if parameter is from a frozen model component
                        is_frozen_model_param = False
                        if hasattr(self, "args") and getattr(
                            self.args, "freeze_base_models", False
                        ):
                            if any(
                                frozen_prefix in name
                                for frozen_prefix in ["vision_model", "text_model"]
                            ):
                                is_frozen_model_param = True

                        # Log critical gradient issues immediately
                        if has_nan or has_inf:
                            logger.error(
                                f"CRITICAL: NaN/Inf detected in gradients for {name}"
                            )
                        elif param_norm > 10.0:
                            logger.warning(
                                f"Unusually LARGE gradient norm for {name}: {param_norm:.4f}"
                            )
                        elif (
                            param_norm < 1e-6
                            and param_norm > 0
                            and not is_frozen_model_param
                        ):
                            # Only warn about small gradients for non-frozen parameters
                            if (
                                self.current_epoch > 1
                            ):  # Skip warnings in early training
                                logger.warning(
                                    f"Unusually SMALL gradient norm for {name}: {param_norm:.8f}"
                                )
                            else:
                                logger.debug(
                                    f"Small gradient norm for {name}: {param_norm:.8f} (expected in early training)"
                                )
                        elif (
                            is_zero
                            and not name.endswith("bias")
                            and not is_frozen_model_param
                        ):
                            # Only warn about zero gradients for non-frozen parameters
                            if (
                                self.current_epoch > 1
                            ):  # Skip warnings in early training
                                logger.warning(
                                    f"ZERO gradient detected for {name} - check if properly connected in backward graph"
                                )
                            else:
                                logger.debug(
                                    f"Zero gradient for {name} (expected in early training)"
                                )

                # Calculate and log overall gradient statistics
                if param_count > 0:
                    avg_grad_norm = total_grad_norm / param_count

                    # Prepare components summary sorted by gradient magnitude
                    component_summary = []
                    for prefix, stats in model_component_grads.items():
                        if stats["count"] > 0:
                            avg_norm = stats["total_norm"] / stats["count"]
                            status = "✓"

                            # Mark components with potential issues
                            if stats["has_nan"]:
                                status = "❌ NaN"
                            elif stats["has_zero"]:
                                status = "⚠️ ZeroGrad"
                            elif avg_norm < 1e-5:
                                status = "⚠️ VerySmall"
                            elif avg_norm > 5.0:
                                status = "⚠️ VeryLarge"

                            component_summary.append(
                                (prefix, avg_norm, stats["count"], status)
                            )

                    # Sort by gradient norm (descending)
                    component_summary.sort(key=lambda x: x[1], reverse=True)

                    # Log summary header and overall stats
                    logger.info(
                        f"Gradient Analysis - Overall avg: {avg_grad_norm:.4f}, {param_count} parameters with grad"
                    )

                    # Log per-component summary
                    for prefix, avg_norm, count, status in component_summary:
                        logger.info(
                            f"  {prefix}: avg={avg_norm:.4f}, params={count} {status}"
                        )

                    # Calculate ratio between highest and lowest component gradients
                    # High ratios suggest imbalanced learning or vanishing gradients
                    if len(component_summary) > 1:
                        highest_avg = component_summary[0][1]
                        lowest_avg = component_summary[-1][1]
                        if lowest_avg > 0:
                            gradient_ratio = highest_avg / lowest_avg
                            logger.info(
                                f"  Gradient ratio (highest/lowest): {gradient_ratio:.1f}x"
                            )

                            if gradient_ratio > 1000:
                                logger.warning(
                                    f"SEVERE GRADIENT IMBALANCE DETECTED: {gradient_ratio:.1f}x - learning will be dominated by {component_summary[0][0]}"
                                )
                            elif gradient_ratio > 100:
                                logger.warning(
                                    f"GRADIENT IMBALANCE DETECTED: {gradient_ratio:.1f}x - consider gradient clipping or separate optimizers"
                                )

                    # Suggest actions based on gradient analysis
                    if avg_grad_norm < 1e-4:
                        logger.warning(
                            "POTENTIAL VANISHING GRADIENT - consider adjusting learning rate or checking loss calculation"
                        )
                    elif avg_grad_norm > 10.0:
                        logger.warning(
                            "POTENTIAL EXPLODING GRADIENT - consider gradient clipping or reducing learning rate"
                        )

            # Update metrics, handling nested dictionaries properly
            for k, v in loss_dict.items():
                if k != "loss":
                    if isinstance(v, dict):
                        # Handle nested dictionary (like 'recalls')
                        if k not in nested_metrics:
                            nested_metrics[k] = defaultdict(float)
                        for sub_k, sub_v in v.items():
                            nested_metrics[k][sub_k] += sub_v / len(pbar)
                    else:
                        # Handle simple metrics
                        epoch_metrics[k] += v / len(pbar)

            # Update progress bar with appropriate metrics based on the loss type
            if isinstance(self.loss_fn, VICRegLoss) and "warmup_factor" in loss_dict:
                # For VICReg, show invariance loss and warmup factor
                pbar.set_postfix(
                    {
                        "loss": loss_dict["loss"].item(),
                        "invariance": loss_dict.get("invariance_loss", 0.0),
                        "warmup": loss_dict.get("warmup_factor", 0.0),
                    }
                )
            elif (
                hasattr(self.loss_fn, "current_phase") and "current_phase" in loss_dict
            ):
                # For hybrid loss, show phase and appropriate metrics
                phase = loss_dict["current_phase"]
                if phase == "contrastive_pretrain":
                    pbar.set_postfix(
                        {
                            "loss": loss_dict["loss"].item(),
                            "acc": loss_dict.get("accuracy", 0.0),
                            "phase": "contrastive",
                            "progress": f"{loss_dict.get('pretrain_progress', 0.0):.2f}",
                        }
                    )
                else:  # vicreg phase
                    pbar.set_postfix(
                        {
                            "loss": loss_dict["loss"].item(),
                            "invariance": loss_dict.get("invariance_loss", 0.0),
                            "var_weight": loss_dict.get("var_weight", 0.0),
                            "phase": "vicreg",
                        }
                    )
            else:
                # For other losses, show accuracy
                pbar.set_postfix(
                    {
                        "loss": loss_dict["loss"].item(),
                        "acc": loss_dict.get("accuracy", 0.0),
                    }
                )

            # Step optimizer after accumulation or at the end of epoch
            if (batch_idx + 1) % self.accumulation_steps == 0 or batch_idx == len(
                self.train_dataloader
            ) - 1:
                # Clip gradients if specified
                if self.clip_grad_norm is not None:
                    if self.mixed_precision and self.scaler is not None:
                        self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.clip_grad_norm
                    )

                # Step optimizer and scheduler
                if self.mixed_precision and self.scaler is not None:
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    self.optimizer.step()

                # Step scheduler if available
                if self.scheduler is not None:
                    self.scheduler.step()

                # Update VICReg or HybridPretrainVICReg loss step counter if applicable
                if hasattr(self.loss_fn, "update_step"):
                    self.loss_fn.update_step(
                        self.global_step, len(self.train_dataloader) * self.num_epochs
                    )

                # Reset gradients
                self.optimizer.zero_grad()

            # Log metrics less frequently (only at specified intervals)
            if self.global_step % self.log_steps == 0:
                lr = self.optimizer.param_groups[0]["lr"]
                if (
                    self.log_steps >= 50
                ):  # Only log to file, not to console for frequent updates
                    logger.debug(
                        f"Step {self.global_step}: loss={loss_dict['loss'].item():.4f}, lr={lr:.6f}"
                    )

            # Evaluate during training if evaluation_steps > 0
            if (
                self.val_dataloader
                and self.evaluation_steps > 0
                and self.global_step % self.evaluation_steps == 0
            ):
                val_metrics = self.evaluate(self.val_dataloader)
                self._log_metrics(val_metrics, "val_step")
                self.model.train()  # Return to training mode

            self.global_step += 1

        # Compute average loss
        epoch_metrics["loss"] = total_loss / max(1, num_batches)

        # Merge regular and nested metrics
        result = dict(epoch_metrics)
        for k, v in nested_metrics.items():
            result[k] = dict(v)

        return result

    def evaluate(self, dataloader: DataLoader) -> Dict[str, float]:
        """
        Evaluate the model using global comparison across all samples.

        This implementation addresses the critical issue where in-batch metrics can
        give artificially high performance. Instead, it compares each image against
        all possible captions in the dataset.

        Args:
            dataloader: DataLoader for evaluation

        Returns:
            Dictionary with evaluation metrics
        """
        self.model.eval()

        # Initialize metrics
        all_metrics = defaultdict(float)

        # CRITICAL FIX: We'll collect ALL embeddings first, then do global comparison
        all_image_embeddings = []
        all_text_embeddings = []
        all_original_indices = []
        all_raw_texts = []

        print(f"Collecting embeddings from all {len(dataloader)} batches...")

        with torch.no_grad():
            # Step 1: Extract all embeddings
            for batch in tqdm(dataloader, desc="Extracting features"):
                # Move batch to device
                batch = self._to_device(batch)

                # Extract only the inputs expected by the model
                model_inputs = self._prepare_model_inputs(batch)

                # Skip invalid batches
                if not model_inputs:
                    print("WARNING: No valid images or text data found in batch!")
                    continue

                # Forward pass to get embeddings
                outputs = self.model(**model_inputs)

                # Get image features - prefer enhanced features if available
                if "vision_features_enhanced" in outputs:
                    image_features = outputs["vision_features_enhanced"]
                elif "image_features" in outputs:
                    image_features = outputs["image_features"]
                else:
                    image_features = outputs.get("vision_features", None)

                # Get text features - prefer enhanced features if available
                if "text_features_enhanced" in outputs:
                    text_features = outputs["text_features_enhanced"]
                elif "text_features" in outputs:
                    text_features = outputs["text_features"]
                else:
                    text_features = None

                # Process image features if available
                if image_features is not None:
                    # Apply mean pooling if features are sequences
                    if len(image_features.shape) == 3:
                        image_features = image_features.mean(dim=1)

                    # Normalize features
                    image_features = F.normalize(image_features, p=2, dim=1)
                    all_image_embeddings.append(image_features.cpu())

                # Process text features if available
                if text_features is not None:
                    # Apply mean pooling if features are sequences
                    if len(text_features.shape) == 3:
                        text_features = text_features.mean(dim=1)

                    # Normalize features
                    text_features = F.normalize(text_features, p=2, dim=1)
                    all_text_embeddings.append(text_features.cpu())

                # Track original indices for ground truth matching
                if "original_idx" in batch:
                    all_original_indices.extend(batch["original_idx"].cpu().tolist())
                elif "idx" in batch:
                    all_original_indices.extend(batch["idx"].cpu().tolist())

                # Store raw texts for analysis
                if "raw_text" in batch:
                    all_raw_texts.extend(batch["raw_text"])

            # Ensure we have embeddings to process
            if not all_image_embeddings or not all_text_embeddings:
                print("ERROR: No embeddings collected during evaluation")
                return {"error": 1.0}

            # Step 2: Concatenate all embeddings
            all_image_embeddings = torch.cat(all_image_embeddings, dim=0)
            all_text_embeddings = torch.cat(all_text_embeddings, dim=0)

            # Ensure we have original indices for all samples
            if len(all_original_indices) != len(all_image_embeddings):
                print(
                    f"WARNING: Mismatch between indices ({len(all_original_indices)}) and embeddings ({len(all_image_embeddings)})"
                )
                # Fall back to position-based matching if indices are incomplete
                all_original_indices = list(range(len(all_image_embeddings)))

            # Step 3: Compute GLOBAL similarity matrix - CRITICALLY IMPORTANT!
            # This compares each image against ALL texts, not just those in its batch
            print(
                f"Computing global similarity matrix of shape {all_image_embeddings.shape[0]}×{all_text_embeddings.shape[0]}..."
            )
            similarity = torch.matmul(all_image_embeddings, all_text_embeddings.T)

            # Step 4: Compute global retrieval metrics
            # First build a mapping from original index to position
            idx_to_position = {idx: i for i, idx in enumerate(all_original_indices)}

            # Now calculate recall@K metrics
            recall_K = [1, 5, 10]
            recalls = {}

            # Also calculate accuracy (recall@1)
            correct_image_to_text = 0
            correct_text_to_image = 0

            # For each position, the correct match has the same original index
            for i, orig_idx in enumerate(all_original_indices):
                # Get the positions of all samples with the same original index
                matching_positions = [
                    j for j, idx in enumerate(all_original_indices) if idx == orig_idx
                ]

                # Skip if we have no matches (shouldn't happen)
                if not matching_positions:
                    continue

                # For each k, check if any matching position is in the top-k
                for k in recall_K:
                    k_adjusted = min(k, len(all_text_embeddings))

                    # Image-to-text: Get top-k text matches for this image
                    i2t_topk = (
                        torch.topk(similarity[i], k_adjusted, dim=0)[1].cpu().tolist()
                    )
                    i2t_hit = any(pos in i2t_topk for pos in matching_positions)

                    # If k=1, count toward accuracy
                    if k == 1 and i2t_hit:
                        correct_image_to_text += 1

                    # Text-to-image: Get top-k image matches for this text
                    t2i_topk = (
                        torch.topk(similarity[:, i], k_adjusted, dim=0)[1]
                        .cpu()
                        .tolist()
                    )
                    t2i_hit = any(pos in t2i_topk for pos in matching_positions)

                    # If k=1, count toward accuracy
                    if k == 1 and t2i_hit:
                        correct_text_to_image += 1

                    # Add to appropriate recalls bucket
                    recall_key = f"recall@{k}"
                    if recall_key not in recalls:
                        recalls[recall_key] = {
                            "i2t_hits": 0,
                            "t2i_hits": 0,
                            "total": 0,
                        }

                    recalls[recall_key]["i2t_hits"] += int(i2t_hit)
                    recalls[recall_key]["t2i_hits"] += int(t2i_hit)
                    recalls[recall_key]["total"] += 1

            # Calculate final recall metrics
            for k in recall_K:
                recall_key = f"recall@{k}"
                bucket = recalls[recall_key]

                # Avoid division by zero
                if bucket["total"] > 0:
                    i2t_recall = bucket["i2t_hits"] / bucket["total"]
                    t2i_recall = bucket["t2i_hits"] / bucket["total"]
                    avg_recall = (i2t_recall + t2i_recall) / 2
                else:
                    i2t_recall = t2i_recall = avg_recall = 0.0

                # Store in metrics dictionary - these are GLOBAL metrics
                all_metrics[f"global_i2t_recall@{k}"] = i2t_recall
                all_metrics[f"global_t2i_recall@{k}"] = t2i_recall
                all_metrics[f"global_avg_recall@{k}"] = avg_recall

            # Calculate final accuracy metrics
            total_samples = len(all_image_embeddings)
            if total_samples > 0:
                i2t_accuracy = correct_image_to_text / total_samples
                t2i_accuracy = correct_text_to_image / total_samples
                avg_accuracy = (i2t_accuracy + t2i_accuracy) / 2
            else:
                i2t_accuracy = t2i_accuracy = avg_accuracy = 0.0

            all_metrics["global_i2t_accuracy"] = i2t_accuracy
            all_metrics["global_t2i_accuracy"] = t2i_accuracy
            all_metrics["global_accuracy"] = avg_accuracy

            # Print global metrics for clarity - with emphasis that these are the CORRECT ones to use
            print(
                "\n*** GLOBAL EVALUATION METRICS (THESE ARE THE REAL METRICS - USE THESE) ***"
            )
            print(
                f"  Accuracy: {avg_accuracy:.4f} (I2T: {i2t_accuracy:.4f}, T2I: {t2i_accuracy:.4f})"
            )
            for k in recall_K:
                print(
                    f"  Recall@{k}: {all_metrics[f'global_avg_recall@{k}']:.4f} "
                    + f"(I2T: {all_metrics[f'global_i2t_recall@{k}']:.4f}, "
                    + f"T2I: {all_metrics[f'global_t2i_recall@{k}']:.4f})"
                )

            # Step 5: Also compute in-batch metrics for comparison
            # This helps illustrate the difference between global and in-batch evaluation
            in_batch_metrics = self._compute_traditional_metrics(similarity)
            for k, v in in_batch_metrics.items():
                all_metrics[f"in_batch_{k}"] = v

            # Print comparison with warning about in-batch metrics being misleading
            print("\n⚠️ IN-BATCH VS GLOBAL METRICS COMPARISON ⚠️")
            print(
                "WARNING: In-batch metrics are often misleadingly high due to batch-level shortcuts!"
            )
            print(
                f"  Accuracy: In-Batch={in_batch_metrics['accuracy']:.4f}, Global={avg_accuracy:.4f}, "
                + f"Ratio={in_batch_metrics['accuracy']/max(1e-5, avg_accuracy):.1f}x higher (artificial)"
            )
            for k in recall_K:
                in_batch = in_batch_metrics[f"avg_recall@{k}"]
                global_val = all_metrics[f"global_avg_recall@{k}"]
                print(
                    f"  Recall@{k}: In-Batch={in_batch:.4f}, Global={global_val:.4f}, "
                    + f"Ratio={in_batch/max(1e-5, global_val):.1f}x higher (artificial)"
                )

        return dict(all_metrics)

    def _compute_traditional_metrics(self, similarity):
        """
        Compute traditional in-batch metrics assuming diagonal matches.
        This is provided for comparison with global metrics.

        Args:
            similarity: Similarity matrix

        Returns:
            Dictionary with metrics
        """
        batch_size = min(similarity.shape[0], similarity.shape[1])

        # Create targets assuming diagonal matches
        targets = torch.arange(batch_size, device=similarity.device)

        # Image-to-text metrics
        i2t_sim = similarity[:batch_size, :batch_size]
        i2t_pred = torch.argmax(i2t_sim, dim=1)
        i2t_accuracy = (i2t_pred == targets).float().mean().item()

        # Text-to-image metrics
        t2i_sim = similarity[:batch_size, :batch_size].T
        t2i_pred = torch.argmax(t2i_sim, dim=1)
        t2i_accuracy = (t2i_pred == targets).float().mean().item()

        # Average accuracy
        accuracy = (i2t_accuracy + t2i_accuracy) / 2

        # Recall@K
        metrics = {
            "i2t_accuracy": i2t_accuracy,
            "t2i_accuracy": t2i_accuracy,
            "accuracy": accuracy,
        }

        # Compute recall@K
        for k in [1, 5, 10]:
            k_adjusted = min(k, batch_size)

            # Image-to-text
            i2t_topk = torch.topk(i2t_sim, k_adjusted, dim=1)[1]
            i2t_hits = torch.zeros(
                batch_size, dtype=torch.bool, device=similarity.device
            )
            for i in range(batch_size):
                i2t_hits[i] = (i2t_topk[i] == i).any()
            i2t_recall = i2t_hits.float().mean().item()

            # Text-to-image
            t2i_topk = torch.topk(t2i_sim, k_adjusted, dim=1)[1]
            t2i_hits = torch.zeros(
                batch_size, dtype=torch.bool, device=similarity.device
            )
            for i in range(batch_size):
                t2i_hits[i] = (t2i_topk[i] == i).any()
            t2i_recall = t2i_hits.float().mean().item()

            # Average recall
            avg_recall = (i2t_recall + t2i_recall) / 2

            metrics[f"i2t_recall@{k}"] = i2t_recall
            metrics[f"t2i_recall@{k}"] = t2i_recall
            metrics[f"avg_recall@{k}"] = avg_recall

        return metrics

    def _compute_global_metrics(self, image_embeddings, text_embeddings, indices):
        # Normalize embeddings for cosine similarity
        image_embeddings = F.normalize(image_embeddings, p=2, dim=1)
        text_embeddings = F.normalize(text_embeddings, p=2, dim=1)

        # Compute similarity matrix
        similarity = torch.matmul(image_embeddings, text_embeddings.T)
        batch_size = similarity.shape[0]

        # Compute recall@K
        recall_K = [1, 5, 10]
        recalls = {}

        for k in recall_K:
            k_adjusted = min(k, batch_size)

            # Image-to-text retrieval
            v2t_topk = torch.topk(similarity, k_adjusted, dim=1)[1]
            v2t_matches = torch.zeros(batch_size, dtype=torch.bool)
            for i in range(batch_size):
                v2t_matches[i] = (v2t_topk[i] == i).any()
            v2t_recall = v2t_matches.float().mean().item()

            # Text-to-image retrieval - fixed for MPS
            t2v_matches = torch.zeros(batch_size, dtype=torch.bool)
            for j in range(batch_size):
                # For each text, get its similarities with all images
                text_sims = similarity[:, j]
                topk_images = torch.topk(text_sims, k_adjusted, dim=0)[1]
                t2v_matches[j] = (topk_images == j).any()
            t2v_recall = t2v_matches.float().mean().item()

            # Average recall
            avg_recall = (v2t_recall + t2v_recall) / 2

            recalls[f"v2t_recall@{k}"] = v2t_recall
            recalls[f"t2v_recall@{k}"] = t2v_recall
            recalls[f"avg_recall@{k}"] = avg_recall

        return recalls

    def _prepare_loss_inputs(self, batch, outputs):
        loss_inputs = {}

        # Initialize a flag for controlling logging frequency
        if not hasattr(self, "_log_once_per_epoch"):
            self._log_once_per_epoch = True
            self._last_logged_epoch = -1

        # Check if we need to log for this epoch
        should_log = (
            self.current_epoch != self._last_logged_epoch
        ) and self._log_once_per_epoch

        # Log only once per epoch or if requested
        if should_log:
            # COMPREHENSIVE DEBUG: Log detailed information about model outputs and batch
            logger.info(f"Model output keys: {sorted(outputs.keys())}")
            logger.info(f"Batch keys: {sorted(batch.keys())}")

            # Check model architecture and loss function configuration
            if hasattr(self.loss_fn, "dim"):
                loss_dim = self.loss_fn.dim
                logger.info(f"Loss function dimension: {loss_dim}")

            # Mark that we've logged for this epoch
            self._last_logged_epoch = self.current_epoch

        # CRITICAL FIX: Make sure features are properly extracted and normalized
        # No matter which feature type is used, always normalize them for cosine similarity
        vision_features = None
        text_features = None

        # Step 1: Extract the right features with priority order
        # Try multiple naming conventions to handle different model architectures
        feature_extraction_success = False

        # Use enhanced features if available (first priority)
        if (
            "vision_features_enhanced" in outputs
            and "text_features_enhanced" in outputs
        ):
            # Get pooled features and explicitly normalize
            vision_features = self._get_pooled_features(
                outputs["vision_features_enhanced"]
            )
            text_features = self._get_pooled_features(outputs["text_features_enhanced"])

            # DEBUG: Log feature dimensions and detailed stats
            logger.info(
                f"Enhanced vision features shape: {vision_features.shape}, text features shape: {text_features.shape}"
            )
            feature_extraction_success = True
            feature_source = "enhanced_features"

        # Otherwise use base features (second priority)
        elif "vision_features" in outputs and "text_features" in outputs:
            # Get pooled features and explicitly normalize
            vision_features = self._get_pooled_features(outputs["vision_features"])
            text_features = self._get_pooled_features(outputs["text_features"])

            # DEBUG: Log feature dimensions and detailed stats
            logger.info(
                f"Base vision features shape: {vision_features.shape}, text features shape: {text_features.shape}"
            )
            feature_extraction_success = True
            feature_source = "base_features"

        # Check if image_features/text_features naming is used instead (third priority)
        elif "image_features" in outputs and "text_features" in outputs:
            # Get pooled features and explicitly normalize
            vision_features = self._get_pooled_features(outputs["image_features"])
            text_features = self._get_pooled_features(outputs["text_features"])

            # DEBUG: Log feature dimensions
            logger.info(
                f"Image features shape: {vision_features.shape}, text features shape: {text_features.shape}"
            )
            feature_extraction_success = True
            feature_source = "image_text_features"

        # If still no features found, try to extract from embedded outputs (fourth priority)
        elif "embedded_images" in outputs and "embedded_text" in outputs:
            # Get pooled features and explicitly normalize
            vision_features = self._get_pooled_features(outputs["embedded_images"])
            text_features = self._get_pooled_features(outputs["embedded_text"])

            # DEBUG: Log feature dimensions
            logger.info(
                f"Embedded vision features shape: {vision_features.shape}, text features shape: {text_features.shape}"
            )
            feature_extraction_success = True
            feature_source = "embedded_features"

        # If no features found, log a critical error
        if not feature_extraction_success:
            logger.error(
                "CRITICAL ERROR: Could not find valid feature outputs from model!"
            )
            logger.error(f"Available keys: {sorted(outputs.keys())}")
            # Try to proceed with empty tensors as last resort
            vision_features = torch.zeros(
                (batch["images"].shape[0], 768), device=self.device
            )
            text_features = torch.zeros(
                (batch["images"].shape[0], 768), device=self.device
            )
            feature_source = "emergency_fallback"

        # Step 2: Comprehensive feature diagnostics
        if feature_extraction_success:
            # Check for NaN or Inf values
            vision_has_nan = torch.isnan(vision_features).any().item()
            text_has_nan = torch.isnan(text_features).any().item()
            vision_has_inf = torch.isinf(vision_features).any().item()
            text_has_inf = torch.isinf(text_features).any().item()

            if vision_has_nan or text_has_nan or vision_has_inf or text_has_inf:
                logger.error(
                    f"INVALID VALUES DETECTED: nan in vision={vision_has_nan}, nan in text={text_has_nan}, "
                    f"inf in vision={vision_has_inf}, inf in text={text_has_inf}"
                )

            # Calculate feature statistics for diagnosis
            vision_mean = vision_features.mean().item()
            vision_std = vision_features.std().item()
            vision_min = vision_features.min().item()
            vision_max = vision_features.max().item()

            text_mean = text_features.mean().item()
            text_std = text_features.std().item()
            text_min = text_features.min().item()
            text_max = text_features.max().item()

            # Only log feature stats once per epoch
            if should_log:
                logger.info(
                    f"Vision features stats: mean={vision_mean:.6f}, std={vision_std:.6f}, "
                    f"min={vision_min:.6f}, max={vision_max:.6f}, range={vision_max-vision_min:.6f}"
                )
                logger.info(
                    f"Text features stats: mean={text_mean:.6f}, std={text_std:.6f}, "
                    f"min={text_min:.6f}, max={text_max:.6f}, range={text_max-text_min:.6f}"
                )

            # Check for feature collapse (extremely low variance)
            if vision_std < 1e-4:
                logger.error(
                    f"VISION FEATURE COLLAPSE DETECTED! Standard deviation: {vision_std:.8f}"
                )

            if text_std < 1e-4:
                logger.error(
                    f"TEXT FEATURE COLLAPSE DETECTED! Standard deviation: {text_std:.8f}"
                )

            # Calculate per-dimension variance to check for dimension collapse
            vision_dim_var = torch.var(vision_features, dim=0)
            text_dim_var = torch.var(text_features, dim=0)

            # Check for dimensions with very low variance
            low_var_dims_vision = (vision_dim_var < 1e-6).sum().item()
            low_var_dims_text = (text_dim_var < 1e-6).sum().item()

            if low_var_dims_vision > 0 or low_var_dims_text > 0:
                logger.warning(
                    f"Dimension collapse: Vision has {low_var_dims_vision}/{vision_features.shape[1]} low-variance dims, "
                    f"Text has {low_var_dims_text}/{text_features.shape[1]} low-variance dims"
                )

            # Step 3: Normalize features for cosine similarity
            vision_features = F.normalize(vision_features, p=2, dim=1)
            text_features = F.normalize(text_features, p=2, dim=1)

            # Verify normalization worked correctly by checking norms
            vision_norms = torch.norm(vision_features, dim=1)
            text_norms = torch.norm(text_features, dim=1)

            vision_norm_mean = vision_norms.mean().item()
            text_norm_mean = text_norms.mean().item()

            # Only log normalization stats once per epoch
            if should_log:
                logger.info(
                    f"After normalization - Vision norm: {vision_norm_mean:.6f}, Text norm: {text_norm_mean:.6f}"
                )

            # Check if normalization worked well
            norm_threshold = 0.01
            if (
                abs(vision_norm_mean - 1.0) > norm_threshold
                or abs(text_norm_mean - 1.0) > norm_threshold
            ):
                logger.error(
                    f"NORMALIZATION ERROR! Vision norm={vision_norm_mean:.6f}, Text norm={text_norm_mean:.6f}"
                )

            # Step 4: Calculate feature similarity diagnostics
            similarity = torch.matmul(vision_features, text_features.T)
            sim_diag = torch.diagonal(similarity)
            sim_mean = similarity.mean().item()
            sim_std = similarity.std().item()
            sim_min = similarity.min().item()
            sim_max = similarity.max().item()
            diag_mean = sim_diag.mean().item()
            diag_std = sim_diag.std().item()

            # Calculate alignment metrics
            alignment_gap = diag_mean - sim_mean
            alignment_snr = abs(alignment_gap) / (
                sim_std + 1e-6
            )  # Signal-to-noise ratio

            # Track alignment metrics history
            if not hasattr(self, "_alignment_history"):
                self._alignment_history = {
                    "epoch": [],
                    "step": [],
                    "diag_mean": [],
                    "sim_mean": [],
                    "alignment_gap": [],
                    "alignment_snr": [],
                }

            # Store current values
            self._alignment_history["epoch"].append(self.current_epoch)
            self._alignment_history["step"].append(self.global_step)
            self._alignment_history["diag_mean"].append(diag_mean)
            self._alignment_history["sim_mean"].append(sim_mean)
            self._alignment_history["alignment_gap"].append(alignment_gap)
            self._alignment_history["alignment_snr"].append(alignment_snr)

            # Keep history manageable
            max_history = 100
            if len(self._alignment_history["step"]) > max_history:
                for key in self._alignment_history:
                    self._alignment_history[key] = self._alignment_history[key][
                        -max_history:
                    ]

            # Only log similarity stats once per epoch
            if should_log:
                logger.info(
                    f"Similarity stats: mean={sim_mean:.4f}, std={sim_std:.4f}, min={sim_min:.4f}, max={sim_max:.4f}"
                )
                logger.info(
                    f"Diagonal similarity (should be high for matched pairs): mean={diag_mean:.4f}, gap={alignment_gap:.4f}, SNR={alignment_snr:.2f}"
                )

            # Adaptive warning threshold based on training stage
            # Early in training, diag_mean can be close to or even below sim_mean
            base_threshold = 0.1
            early_training_factor = max(
                0.05, min(1.0, (self.current_epoch + 1) / 5.0)
            )  # Adjust threshold for first 5 epochs

            # For VICReg specifically, be more lenient in early training
            if isinstance(self.loss_fn, VICRegLoss) and self.current_epoch < 3:
                threshold = base_threshold * 0.5 * early_training_factor
            else:
                threshold = base_threshold * early_training_factor

            if diag_mean < sim_mean + threshold:
                # In early training, use INFO level instead of WARNING for expected behavior
                if self.current_epoch < 2 and isinstance(self.loss_fn, VICRegLoss):
                    logger.info(
                        f"Early training alignment: Diagonal similarity ({diag_mean:.4f}) gap from mean ({sim_mean:.4f}) is {alignment_gap:.4f}"
                    )
                else:
                    logger.warning(
                        f"POTENTIAL ISSUE: Diagonal similarity ({diag_mean:.4f}) is not significantly higher than mean ({sim_mean:.4f})"
                    )

            # Check similarity histogram to detect distribution issues (only once per epoch)
            if should_log:
                sim_hist = torch.histc(similarity.flatten(), bins=10, min=-1.0, max=1.0)
                logger.info(f"Similarity histogram: {sim_hist.tolist()}")

            # Store feature_source in a class variable for debugging but don't pass to loss function
            self._debug_feature_source = feature_source

            # HANDLE DIFFERENT LOSS TYPES APPROPRIATELY
            if isinstance(self.loss_fn, VICRegLoss):
                # For VICReg, completely reset loss_inputs to avoid any unexpected arguments
                # Create a new dictionary with only the needed inputs
                loss_inputs = {"z_a": vision_features, "z_b": text_features}
                logger.info("Prepared inputs for VICRegLoss (z_a and z_b only)")

                # Log VICReg batch statistics if match_id is available, but don't use them in loss
                if "match_id" in batch:
                    match_ids = batch["match_id"]
                    unique_ids = len(set(match_ids))
                    batch_size = len(match_ids)
                    logger.info(
                        f"VICReg batch: {unique_ids} unique IDs in batch of {batch_size} (not used in loss)"
                    )

                # Set match_id_source for debugging but don't include in loss inputs
                self._debug_match_id_source = "not_used_in_vicreg"

            else:
                # OTHER LOSS TYPES (ContrastiveLoss, MultiModalMixedContrastiveLoss, etc.)
                # Standard inputs for other loss types
                loss_inputs["vision_features"] = vision_features
                loss_inputs["text_features"] = text_features

                # ENHANCEMENT: Add noise to features during training to discourage shortcut learning
                # This makes the task harder and forces the model to learn more robust representations
                if self.model.training:
                    # Small amount of noise to prevent trivial solutions and shortcut learning
                    # Scale noise based on global_step - start higher, then gradually reduce
                    noise_scale = max(0.05 * (1.0 - self.global_step / 10000), 0.01)

                    # Only apply noise with 70% probability for variability
                    if random.random() < 0.7:
                        # Apply the noise to both vision and text features
                        vision_noise = torch.randn_like(vision_features) * noise_scale
                        text_noise = torch.randn_like(text_features) * noise_scale

                        # Apply noise and renormalize
                        loss_inputs["vision_features"] = F.normalize(
                            vision_features + vision_noise, p=2, dim=1
                        )
                        loss_inputs["text_features"] = F.normalize(
                            text_features + text_noise, p=2, dim=1
                        )

                        # Log noise level occasionally
                        if self.global_step % 500 == 0:
                            logger.info(
                                f"Applied feature noise with scale {noise_scale:.4f} at step {self.global_step}"
                            )

                # Add match_ids for contrastive losses
                match_id_source = None
                if "match_id" in batch:
                    # This is critical - the match_id determines the semantic relationships
                    loss_inputs["match_ids"] = batch["match_id"]
                    match_id_source = "match_id"

                    # Detailed diagnostics on match_ids (only once per epoch)
                    match_ids = batch["match_id"]
                    unique_ids = len(set(match_ids))
                    batch_size = len(match_ids)

                    # Only log detailed match ID info once per epoch
                    if should_log:
                        # Count frequency of each match_id
                        match_id_counts = {}
                        for mid in match_ids:
                            match_id_counts[mid] = match_id_counts.get(mid, 0) + 1

                        # Find the most common match_id and its frequency
                        most_common_id = (
                            max(match_id_counts.items(), key=lambda x: x[1])
                            if match_id_counts
                            else None
                        )

                        logger.info(
                            f"Match IDs: {unique_ids} unique IDs in batch of {batch_size}"
                        )
                        if most_common_id:
                            logger.info(
                                f"Most common match_id appears {most_common_id[1]} times ({most_common_id[1]/batch_size:.1%} of batch)"
                            )

                    # Check if we have proper semantic grouping
                    if unique_ids == batch_size:
                        logger.error(
                            "CRITICAL ERROR: All match_ids are unique - no semantic grouping possible!"
                        )
                    elif unique_ids == 1:
                        logger.error(
                            "CRITICAL ERROR: All match_ids are identical - treating all pairs as matches!"
                        )

                elif "idx" in batch:
                    # Fallback to indices - not ideal but better than nothing
                    loss_inputs["match_ids"] = batch["idx"]
                    match_id_source = "idx_fallback"
                    logger.warning(
                        "Using idx as fallback for match_ids - this may lead to poor performance!"
                    )
                else:
                    # Last resort - use position-based matching
                    # This is highly problematic and will cause shortcut learning!
                    match_id_source = "position_fallback"
                    logger.error(
                        "No match_id or idx found in batch - using position-based matching!"
                    )
                    logger.error(
                        "THIS WILL LIKELY CAUSE SHORTCUT LEARNING AND POOR EVALUATION RESULTS!"
                    )

                # Store match_id_source in a class variable for debugging but don't pass to loss function
                self._debug_match_id_source = match_id_source

                # Add classification logits if available
                if "classification" in outputs and "label" in batch:
                    loss_inputs["class_logits"] = outputs["classification"]
                    loss_inputs["class_labels"] = batch["label"]

                # Add multimodal matching logits if available
                if "matching_logits" in outputs:
                    loss_inputs["matching_logits"] = outputs["matching_logits"]
                    # For matching, diagonal elements are positives (matching pairs)
                    batch_size = outputs["matching_logits"].shape[0]
                    matching_labels = torch.eye(batch_size, device=self.device)
                    loss_inputs["matching_labels"] = matching_labels

        return loss_inputs

    def _get_pooled_features(self, features: torch.Tensor) -> torch.Tensor:
        """
        Get pooled features from sequence features if needed.

        Args:
            features: Features tensor

        Returns:
            Pooled features tensor
        """
        # If features are already pooled, return as is
        if len(features.shape) == 2:
            return features

        # If features are sequences, use mean pooling
        if len(features.shape) == 3:
            return features.mean(dim=1)

        # Unsupported shape
        raise ValueError(f"Unsupported features shape: {features.shape}")

    def _to_device(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Move batch to device and ensure consistent naming for the model.
        Handles complex nested data structures with device consistency checks.

        Args:
            batch: Batch of data which may contain nested dictionaries

        Returns:
            Batch on device with consistent naming
        """
        # Check model device for consistency - ensures the model is on the expected device
        model_device = next(self.model.parameters()).device
        if model_device != self.device:
            # logger.warning(
            #     f"Model device ({model_device}) differs from trainer device ({self.device})"
            # )
            # Re-ensure model is on correct device
            self._ensure_model_on_device()

        # Process batch items recursively
        def process_item(item):
            if isinstance(item, torch.Tensor):
                # If tensor already on device, skip moving
                if item.device == self.device:
                    return item
                # Otherwise move to device
                return item.to(self.device)
            elif isinstance(item, dict):
                # For dictionaries, process each item recursively
                return {k: process_item(v) for k, v in item.items()}
            elif isinstance(item, list):
                # For lists, process each item recursively
                return [process_item(i) for i in item]
            else:
                # Return other types unchanged
                return item

        # Process the full batch
        processed_batch = process_item(batch)

        # Make sure naming is consistent with model's expected arguments
        # Model expects 'images' but dataset returns 'image'
        if "image" in processed_batch and "images" not in processed_batch:
            processed_batch["images"] = processed_batch.pop("image")

        # Model expects 'text_data' but dataset returns 'text'
        if "text" in processed_batch and "text_data" not in processed_batch:
            processed_batch["text_data"] = processed_batch.pop("text")

        # Special handling for text data if it's a nested dictionary
        if "text_data" in processed_batch and isinstance(
            processed_batch["text_data"], dict
        ):
            # Ensure all text data tensors are on the device
            text_data = processed_batch["text_data"]
            for k, v in text_data.items():
                if isinstance(v, torch.Tensor) and v.device != self.device:
                    text_data[k] = v.to(self.device)

        return processed_batch

    def _prepare_model_inputs(
        self, batch: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """
        Extract only the inputs that are needed by the model's forward method.
        Includes additional device consistency checks.

        Args:
            batch: Full batch of data

        Returns:
            Dictionary with only the inputs the model expects
        """
        # Create a new dictionary with only the expected inputs
        # EnhancedMultiModalTransformer.forward expects:
        # - images: Optional image tensor
        # - text_data: Optional text dictionary
        # - return_attention: Whether to return attention maps

        model_inputs = {}

        # Final device check - critical for MPS compatibility
        target_device = self.device

        # Add images if available
        if "images" in batch:
            # Double check device
            if batch["images"].device != target_device:
                batch["images"] = batch["images"].to(target_device)
                logger.debug(
                    f"Had to move images to {target_device} in _prepare_model_inputs"
                )

            model_inputs["images"] = batch["images"]

        # Add text_data if available with device consistency check
        if "text_data" in batch:
            # For text_data, we need to check nested dictionaries
            if isinstance(batch["text_data"], dict):
                # First check if any tensors are on wrong device
                text_data = batch["text_data"]
                needs_device_fix = False

                for k, v in text_data.items():
                    if isinstance(v, torch.Tensor) and v.device != target_device:
                        needs_device_fix = True
                        break

                if needs_device_fix:
                    # Copy dictionary and move all tensors to correct device
                    text_data_fixed = {}
                    for k, v in text_data.items():
                        if isinstance(v, torch.Tensor) and v.device != target_device:
                            text_data_fixed[k] = v.to(target_device)
                            logger.debug(
                                f"Had to move text_data[{k}] to {target_device}"
                            )
                        else:
                            text_data_fixed[k] = v
                    model_inputs["text_data"] = text_data_fixed
                else:
                    # Dictionary already has all tensors on correct device
                    model_inputs["text_data"] = text_data
            else:
                # Not a dictionary, just use directly
                model_inputs["text_data"] = batch["text_data"]

        # Other arguments like raw_text, idx, etc. should not be passed to the model

        return model_inputs

    def _log_metrics(self, metrics: Dict[str, float], prefix: str) -> None:
        """
        Log metrics and add to history.

        Args:
            metrics: Dictionary of metrics (can contain nested dictionaries)
            prefix: Prefix for metric names in history
        """
        # Format metrics for logging
        metrics_parts = []

        for k, v in metrics.items():
            if isinstance(v, dict):
                # Handle nested metrics (like 'recalls')
                sub_metrics = [f"{k}.{sub_k}={sub_v:.4f}" for sub_k, sub_v in v.items()]
                metrics_parts.extend(sub_metrics)
            else:
                # Handle simple metrics
                metrics_parts.append(f"{k}={v:.4f}")

        metrics_str = ", ".join(metrics_parts)
        # Use print instead of logger.info for cleaner output
        print(f"{prefix.capitalize()}: {metrics_str}")

        # Add to history, handling nested dictionaries
        for k, v in metrics.items():
            if isinstance(v, dict):
                # Handle nested metrics
                for sub_k, sub_v in v.items():
                    self.history[f"{prefix}_{k}.{sub_k}"].append(sub_v)
            else:
                # Handle simple metrics
                self.history[f"{prefix}_{k}"].append(v)

    def save_checkpoint(self, path: str) -> None:
        """
        Save a checkpoint.

        Args:
            path: Path to save checkpoint
        """
        # Create directory if needed
        os.makedirs(os.path.dirname(path), exist_ok=True)

        checkpoint = {
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "current_epoch": self.current_epoch,
            "global_step": self.global_step,
            "best_val_metric": self.best_val_metric,
            "patience_counter": self.patience_counter,
            "history": dict(self.history),
        }

        # Add scheduler state if available
        if self.scheduler is not None:
            checkpoint["scheduler_state_dict"] = self.scheduler.state_dict()

        # Save checkpoint
        torch.save(checkpoint, path)
        logger.info(f"Checkpoint saved to {path}")

    def _evaluate_tokenizer_quality(self, epoch: int) -> Dict[str, float]:
        """
        Evaluate tokenizer quality metrics and log results once per epoch.

        Args:
            epoch: Current epoch number

        Returns:
            Dictionary with tokenizer quality metrics
        """
        # Find the tokenizer from the dataset
        tokenizer = None
        text_samples = []
        match_ids = []

        # Try to extract tokenizer and text samples from dataloader's dataset
        if hasattr(self.train_dataloader, "dataset"):
            dataset = self.train_dataloader.dataset

            # Common multimodal dataset attribute names
            if hasattr(dataset, "tokenizer"):
                tokenizer = dataset.tokenizer
            elif hasattr(dataset, "text_tokenizer"):
                tokenizer = dataset.text_tokenizer

            # Try to extract text samples from a recent batch
            # The goal is to get a sample of real training data
            sample_size = min(100, len(dataset))
            indices = np.random.randint(0, len(dataset), size=sample_size)

            # Collect text samples and match IDs (if available)
            for idx in indices:
                try:
                    sample = dataset[idx]

                    # Handle different dataset formats
                    if isinstance(sample, dict):
                        if "text" in sample:
                            if isinstance(sample["text"], str):
                                text_samples.append(sample["text"])
                            elif (
                                isinstance(sample["text"], dict)
                                and "raw_text" in sample["text"]
                            ):
                                text_samples.append(sample["text"]["raw_text"])
                        elif "raw_text" in sample:
                            text_samples.append(sample["raw_text"])
                        elif "caption" in sample:
                            text_samples.append(sample["caption"])

                        # Collect match_ids if available
                        if "match_id" in sample:
                            match_ids.append(sample["match_id"])
                except Exception as e:
                    logger.warning(f"Error sampling dataset item: {str(e)}")
                    continue

        # If we couldn't find the tokenizer or text samples, return empty metrics
        if tokenizer is None:
            logger.warning("Could not find tokenizer for quality evaluation")
            return {}

        if not text_samples:
            logger.warning("Could not find text samples for tokenizer evaluation")
            return {}

        # Check if we have match_ids
        if not match_ids or len(match_ids) != len(text_samples):
            match_ids = None
            logger.info("Match IDs not available for semantic tokenizer evaluation")

        # Evaluate tokenizer quality and log results
        try:
            metrics = log_tokenizer_evaluation(
                tokenizer=tokenizer,
                text_data=text_samples,
                match_ids=match_ids,
                epoch=epoch,
            )

            # Save metrics to history
            for k, v in metrics.items():
                if not isinstance(v, dict) and not isinstance(v, list):
                    self.history[f"tokenizer_{k}"].append(v)

            return metrics
        except Exception as e:
            logger.error(f"Error evaluating tokenizer quality: {str(e)}")
            return {}

    def load_checkpoint(self, path: str) -> None:
        """
        Load a checkpoint.

        Args:
            path: Path to checkpoint
        """
        if not os.path.exists(path):
            logger.warning(f"Checkpoint not found: {path}")
            return

        checkpoint = torch.load(path, map_location=self.device)

        # Load model weights
        self.model.load_state_dict(checkpoint["model_state_dict"])

        # Load optimizer state
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        # Load scheduler state if available
        if self.scheduler is not None and "scheduler_state_dict" in checkpoint:
            self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

        # Load training state
        self.current_epoch = checkpoint["current_epoch"] + 1  # Resume from next epoch
        self.global_step = checkpoint["global_step"]
        self.best_val_metric = checkpoint["best_val_metric"]
        self.patience_counter = checkpoint["patience_counter"]

        # Load history
        if "history" in checkpoint:
            self.history = defaultdict(list, checkpoint["history"])

        logger.info(f"Checkpoint loaded from {path}")
        logger.info(
            f"Resuming from epoch {self.current_epoch}, step {self.global_step}"
        )
        logger.info(f"Best validation metric: {self.best_val_metric:.4f}")

    def plot_history(self, save_dir: Optional[str] = None) -> None:
        """
        Plot training history metrics.

        Args:
            save_dir: Directory to save plots
        """
        if not self.history:
            return

        # Create directory if not exists
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)

        # Group metrics by type
        metric_groups = {}
        for key in self.history.keys():
            # Skip if there are no values
            if not self.history[key]:
                continue

            # Skip dictionary values
            if isinstance(self.history[key][0], dict):
                continue

            # Split the key into prefix and metric name
            if "_" in key:
                prefix, metric = key.split("_", 1)
                if metric not in metric_groups:
                    metric_groups[metric] = []
                metric_groups[metric].append((prefix, key))

        # Plot each metric group
        for metric, prefixes in metric_groups.items():
            plt.figure(figsize=(10, 6))

            for prefix, key in prefixes:
                values = self.history[key]
                # Only plot if values are not dictionaries
                if values and not isinstance(values[0], dict):
                    # Convert tensors to CPU numpy arrays
                    cpu_values = []
                    for v in values:
                        if isinstance(v, torch.Tensor):
                            cpu_values.append(v.detach().cpu().numpy())
                        else:
                            cpu_values.append(v)

                    epochs = range(1, len(cpu_values) + 1)
                    plt.plot(epochs, cpu_values, label=f"{prefix}")

            plt.title(f"{metric} over epochs")
            plt.xlabel("Epoch")
            plt.ylabel(metric)
            plt.legend()
            plt.grid(True)

            if save_dir:
                plt.savefig(os.path.join(save_dir, f"{metric}.png"))
            plt.close()

        # Plot alignment metrics if available
        if hasattr(self, "_alignment_history") and self._alignment_history["step"]:
            plt.figure(figsize=(12, 8))

            # Plot diagonal similarity vs. mean similarity
            plt.subplot(2, 1, 1)
            steps = self._alignment_history["step"]
            plt.plot(
                steps,
                self._alignment_history["diag_mean"],
                label="Diagonal Similarity",
                color="blue",
            )
            plt.plot(
                steps,
                self._alignment_history["sim_mean"],
                label="Mean Similarity",
                color="red",
                linestyle="--",
            )
            plt.title("Semantic Alignment Progress")
            plt.xlabel("Training Steps")
            plt.ylabel("Cosine Similarity")
            plt.legend()
            plt.grid(True)

            # Plot alignment gap and SNR
            plt.subplot(2, 1, 2)
            plt.plot(
                steps,
                self._alignment_history["alignment_gap"],
                label="Alignment Gap",
                color="green",
            )
            plt.plot(
                steps,
                self._alignment_history["alignment_snr"],
                label="Signal-to-Noise Ratio",
                color="purple",
                linestyle="--",
            )
            plt.title("Alignment Quality Metrics")
            plt.xlabel("Training Steps")
            plt.ylabel("Metric Value")
            plt.legend()
            plt.grid(True)

            plt.tight_layout()

            if save_dir:
                plt.savefig(os.path.join(save_dir, "alignment_progress.png"))
            plt.close()

        # Plot tokenizer quality metrics if available
        tokenizer_metrics = [
            k for k in self.history.keys() if k.startswith("tokenizer_")
        ]
        if tokenizer_metrics:
            plt.figure(figsize=(12, 8))

            # Plot overall quality score
            if "tokenizer_overall_quality_score" in self.history:
                epochs = range(
                    1, len(self.history["tokenizer_overall_quality_score"]) + 1
                )
                plt.subplot(2, 2, 1)
                plt.plot(
                    epochs,
                    self.history["tokenizer_overall_quality_score"],
                    color="blue",
                    marker="o",
                )
                plt.title("Tokenizer Quality Score")
                plt.xlabel("Epoch")
                plt.ylabel("Score (0-100)")
                plt.grid(True)

            # Plot unknown token percentage
            if "tokenizer_unknown_token_percentage" in self.history:
                epochs = range(
                    1, len(self.history["tokenizer_unknown_token_percentage"]) + 1
                )
                plt.subplot(2, 2, 2)
                plt.plot(
                    epochs,
                    self.history["tokenizer_unknown_token_percentage"],
                    color="red",
                    marker="o",
                )
                plt.title("Unknown Token Percentage")
                plt.xlabel("Epoch")
                plt.ylabel("Percentage")
                plt.grid(True)

            # Plot semantic metrics if available
            semantic_plots = 0
            if "tokenizer_semantic_token_overlap" in self.history:
                epochs = range(
                    1, len(self.history["tokenizer_semantic_token_overlap"]) + 1
                )
                plt.subplot(2, 2, 3)
                plt.plot(
                    epochs,
                    self.history["tokenizer_semantic_token_overlap"],
                    color="green",
                    marker="o",
                )
                plt.title("Semantic Token Overlap")
                plt.xlabel("Epoch")
                plt.ylabel("Overlap Score")
                plt.grid(True)
                semantic_plots += 1

            if "tokenizer_semantic_length_consistency" in self.history:
                epochs = range(
                    1, len(self.history["tokenizer_semantic_length_consistency"]) + 1
                )
                plt.subplot(2, 2, 4)
                plt.plot(
                    epochs,
                    self.history["tokenizer_semantic_length_consistency"],
                    color="purple",
                    marker="o",
                )
                plt.title("Length Consistency")
                plt.xlabel("Epoch")
                plt.ylabel("Consistency Score")
                plt.grid(True)
                semantic_plots += 1

            # If we don't have semantic metrics, plot other available metrics
            if semantic_plots == 0:
                # Plot token distribution entropy
                if "tokenizer_token_distribution_entropy" in self.history:
                    epochs = range(
                        1, len(self.history["tokenizer_token_distribution_entropy"]) + 1
                    )
                    plt.subplot(2, 2, 3)
                    plt.plot(
                        epochs,
                        self.history["tokenizer_token_distribution_entropy"],
                        color="green",
                        marker="o",
                    )
                    plt.title("Token Distribution Entropy")
                    plt.xlabel("Epoch")
                    plt.ylabel("Entropy")
                    plt.grid(True)

                # Plot tokens per word ratio
                if "tokenizer_tokens_per_word_ratio" in self.history:
                    epochs = range(
                        1, len(self.history["tokenizer_tokens_per_word_ratio"]) + 1
                    )
                    plt.subplot(2, 2, 4)
                    plt.plot(
                        epochs,
                        self.history["tokenizer_tokens_per_word_ratio"],
                        color="orange",
                        marker="o",
                    )
                    plt.title("Tokens per Word Ratio")
                    plt.xlabel("Epoch")
                    plt.ylabel("Ratio")
                    plt.grid(True)

            plt.tight_layout()

            if save_dir:
                plt.savefig(os.path.join(save_dir, "tokenizer_quality.png"))
            plt.close()
