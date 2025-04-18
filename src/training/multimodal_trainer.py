# src/training/multimodal_trainer.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.adamw import AdamW
from torch.utils.data import DataLoader
import numpy as np
from typing import Dict, List, Tuple, Optional, Callable, Union, Any
import time
import os
import json
import random
import matplotlib.pyplot as plt
from tqdm import tqdm
from collections import defaultdict
import logging

from .contrastive_learning import ContrastiveLoss, MultiModalMixedContrastiveLoss

logger = logging.getLogger(__name__)


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

        # Initialize model on device
        self.model = self.model.to(self.device)

        # Initialize optimizer if not provided
        if self.optimizer is None:
            self.optimizer = AdamW(
                self.model.parameters(),
                lr=self.learning_rate,
                weight_decay=self.weight_decay,
            )

        # Initialize loss function if not provided
        if self.loss_fn is None:
            self.loss_fn = MultiModalMixedContrastiveLoss(temperature=0.07)

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

        # Freeze cross-attention and fusion layers
        for name, param in self.model.named_parameters():
            if any(x in name for x in ["cross", "fusion", "gate"]):
                param.requires_grad = False
            else:
                param.requires_grad = True

        # Create optimizer for Stage 1
        stage1_optimizer = torch.optim.AdamW(
            [
                {
                    "params": [
                        p
                        for n, p in self.model.named_parameters()
                        if "vision_model" in n and p.requires_grad
                    ],
                    "lr": self.learning_rate * 0.1,
                },
                {
                    "params": [
                        p
                        for n, p in self.model.named_parameters()
                        if "text_model" in n and p.requires_grad
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

        # Use standard contrastive loss for Stage 1
        from src.training.contrastive_learning import ContrastiveLoss

        stage1_loss = ContrastiveLoss(temperature=0.07)

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

        # === Stage 2: Train cross-modal fusion ===
        logger.info(
            f"=== Stage 2: Training cross-modal fusion ({stage2_epochs} epochs) ==="
        )

        # Freeze base models, unfreeze fusion components
        for name, param in self.model.named_parameters():
            if any(x in name for x in ["vision_model", "text_model"]):
                param.requires_grad = False
            else:
                param.requires_grad = True

        # Create optimizer for Stage 2
        stage2_optimizer = torch.optim.AdamW(
            [
                {
                    "params": [
                        p
                        for n, p in self.model.named_parameters()
                        if any(x in n for x in ["cross", "fusion", "gate"])
                        and p.requires_grad
                    ],
                    "lr": self.learning_rate,
                }
            ],
            weight_decay=self.weight_decay,
        )

        # Use memory queue contrastive loss for Stage 2
        from src.training.contrastive_learning import MemoryQueueContrastiveLoss

        stage2_loss = MemoryQueueContrastiveLoss(
            dim=512, queue_size=8192, temperature=0.07  # Use fixed dimension for safety
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

        # === Stage 3: Fine-tune everything with hard negative mining ===
        logger.info(
            f"=== Stage 3: Fine-tuning with hard negative mining ({stage3_epochs} epochs) ==="
        )

        # Unfreeze everything for full fine-tuning
        for param in self.model.parameters():
            param.requires_grad = True

        # Create optimizer for Stage 3 with layer-wise learning rates
        stage3_optimizer = torch.optim.AdamW(
            [
                {
                    "params": [
                        p
                        for n, p in self.model.named_parameters()
                        if "vision_model" in n and p.requires_grad
                    ],
                    "lr": self.learning_rate * 0.01,
                },
                {
                    "params": [
                        p
                        for n, p in self.model.named_parameters()
                        if "text_model" in n and p.requires_grad
                    ],
                    "lr": self.learning_rate * 0.01,
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
        from src.training.contrastive_learning import HardNegativeMiningContrastiveLoss

        stage3_loss = HardNegativeMiningContrastiveLoss(
            temperature=0.07, hard_negative_factor=2.0, mining_strategy="semi-hard"
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

            # Update metrics
            total_loss += loss.item() * self.accumulation_steps
            num_batches += 1

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

            # Update progress bar
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
        epoch_metrics["loss"] = total_loss / num_batches

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

        # CRITICAL FIX: Make sure features are properly extracted and normalized
        # No matter which feature type is used, always normalize them for cosine similarity

        # Use enhanced features if available (properly overriding originals)
        if (
            "vision_features_enhanced" in outputs
            and "text_features_enhanced" in outputs
        ):
            # Get pooled features and explicitly normalize
            vision_features = self._get_pooled_features(
                outputs["vision_features_enhanced"]
            )
            text_features = self._get_pooled_features(outputs["text_features_enhanced"])

            # Normalize for cosine similarity (CRUCIAL for contrastive learning)
            vision_features = F.normalize(vision_features, p=2, dim=1)
            text_features = F.normalize(text_features, p=2, dim=1)

            loss_inputs["vision_features"] = vision_features
            loss_inputs["text_features"] = text_features

        # Otherwise use base features
        elif "vision_features" in outputs and "text_features" in outputs:
            # Get pooled features and explicitly normalize
            vision_features = self._get_pooled_features(outputs["vision_features"])
            text_features = self._get_pooled_features(outputs["text_features"])

            # Normalize for cosine similarity (CRUCIAL for contrastive learning)
            vision_features = F.normalize(vision_features, p=2, dim=1)
            text_features = F.normalize(text_features, p=2, dim=1)

            loss_inputs["vision_features"] = vision_features
            loss_inputs["text_features"] = text_features

        # ENHANCEMENT: Add noise to features during training to discourage shortcut learning
        # This makes the task harder and forces the model to learn more robust representations
        if (
            self.model.training
        ):  # Check if model is in training mode instead of self.training
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

        # Store original features for other loss components that might need them
        # loss_inputs["original_vision_features"] = vision_features
        # loss_inputs["original_text_features"] = text_features

        # FIX: Ensure match IDs are properly passed for training and evaluation
        if "match_id" in batch:
            # This is critical - the match_id determines the semantic relationships
            loss_inputs["match_ids"] = batch["match_id"]
        elif "idx" in batch:
            # Fallback to indices - not ideal but better than nothing
            loss_inputs["indices"] = batch["idx"]

            # Print warning if we have to use indices instead of match_ids
            if self.global_step == 0:
                logger.warning(
                    "Using idx as fallback for match_ids - this may lead to poor performance!"
                )
        else:
            # Last resort - use position-based matching
            # This is highly problematic and will cause shortcut learning!
            if self.global_step == 0:
                logger.warning(
                    "No match_id or idx found in batch - using position-based matching!"
                )
                logger.warning(
                    "THIS WILL LIKELY CAUSE SHORTCUT LEARNING AND POOR EVALUATION RESULTS!"
                )

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

        # Debugging: Check feature norms to ensure proper normalization
        if self.global_step % 100 == 0:
            if "vision_features" in loss_inputs and "text_features" in loss_inputs:
                vision_norm = (
                    torch.norm(loss_inputs["vision_features"], dim=1).mean().item()
                )
                text_norm = (
                    torch.norm(loss_inputs["text_features"], dim=1).mean().item()
                )
                logger.debug(
                    f"Feature norms - Vision: {vision_norm:.6f}, Text: {text_norm:.6f}"
                )

                # Check if match_ids are properly distributed
                if "match_ids" in loss_inputs:
                    unique_ids = len(set(loss_inputs["match_ids"]))
                    batch_size = len(loss_inputs["match_ids"])
                    logger.debug(
                        f"Match IDs - Unique: {unique_ids}, Total: {batch_size}, Ratio: {unique_ids/max(1, batch_size):.2f}"
                    )

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

        Args:
            batch: Batch of data

        Returns:
            Batch on device with consistent naming
        """
        # Move tensors to device
        processed_batch = {
            k: v.to(self.device) if isinstance(v, torch.Tensor) else v
            for k, v in batch.items()
        }

        # Make sure naming is consistent with model's expected arguments
        # Model expects 'images' but dataset returns 'image'
        if "image" in processed_batch and "images" not in processed_batch:
            processed_batch["images"] = processed_batch.pop("image")

        # Model expects 'text_data' but dataset returns 'text'
        if "text" in processed_batch and "text_data" not in processed_batch:
            processed_batch["text_data"] = processed_batch.pop("text")

        return processed_batch

    def _prepare_model_inputs(
        self, batch: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """
        Extract only the inputs that are needed by the model's forward method.

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

        # Add images if available
        if "images" in batch:
            model_inputs["images"] = batch["images"]

        # Add text_data if available
        if "text_data" in batch:
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
