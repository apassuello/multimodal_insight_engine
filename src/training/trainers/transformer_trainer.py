"""MODULE: transformer_trainer.py
PURPOSE: Implements a specialized trainer for transformer models with support for encoder-decoder architectures, including features like learning rate scheduling, label smoothing, and efficient masking.

KEY COMPONENTS:
- TransformerTrainer: Main trainer class for transformer model training with support for:
  - Learning rate scheduling with warmup and inverse square root decay
  - Label smoothing for improved generalization
  - Efficient padding and causal masking
  - Early stopping with patience
  - Comprehensive training metrics tracking
  - Training visualization capabilities

DEPENDENCIES:
- PyTorch (torch, torch.nn)
- NumPy
- Matplotlib
- tqdm
- transformer_utils (internal module)

SPECIAL NOTES:
- Implements efficient masking for transformer attention
- Supports both CPU and GPU training with automatic device selection
- Includes early stopping mechanism to prevent overfitting
- Provides comprehensive training visualization tools
"""

import math
import os
import time
from typing import Any, Callable, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np

# src/training/trainers/transformer_trainer.py
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.training.transformer_utils import (
    LabelSmoothing,
    create_causal_mask,
    create_padding_mask,
)


class TransformerTrainer:
    """
    Specialized trainer for transformer models.

    This class provides utilities for training encoder-decoder transformer models
    with features like learning rate scheduling, label smoothing, and efficient masking.
    """

    def __init__(
        self,
        model: nn.Module,
        train_dataloader: DataLoader,
        val_dataloader: Optional[DataLoader] = None,
        pad_idx: int = 0,
        lr: float = 0.0001,
        betas: Tuple[float, float] = (0.9, 0.98),
        eps: float = 1e-9,
        warmup_steps: int = 4000,
        label_smoothing: float = 0.1,
        clip_grad: float = 1.0,
        early_stopping_patience: Optional[int] = None,
        device: Optional[torch.device] = None,
        track_perplexity: bool = False,
        scheduler: str = "inverse_sqrt",
        use_gradient_scaling: bool = False,
        gradient_accumulation_steps: int = 1,
        weight_decay: float = 0.01,
    ):
        """
        Initialize the transformer trainer.

        Args:
            model: Transformer model to train
            train_dataloader: DataLoader for training data
            val_dataloader: Optional DataLoader for validation data
            pad_idx: Padding token index
            lr: Base learning rate
            betas: Adam optimizer betas
            eps: Adam optimizer epsilon
            warmup_steps: Number of warmup steps for learning rate
            label_smoothing: Label smoothing factor
            clip_grad: Gradient clipping value
            early_stopping_patience: Number of epochs to wait for improvement
            device: Device to train on
            track_perplexity: Whether to track perplexity during training
            scheduler: Learning rate scheduler type ("inverse_sqrt", "cosine", "linear", "constant")
            use_gradient_scaling: Whether to use gradient scaling for mixed precision training
            gradient_accumulation_steps: Number of steps to accumulate gradients before updating weights
        """
        self.model = model
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.pad_idx = pad_idx
        self.warmup_steps = warmup_steps
        self.clip_grad = clip_grad
        self.early_stopping_patience = early_stopping_patience
        self.scheduler_type = scheduler
        self.use_gradient_scaling = use_gradient_scaling
        self.weight_decay = weight_decay
        # Set device
        if device is None:
            if torch.backends.mps.is_available():
                self.device = torch.device("mps")
            else:
                self.device = torch.device(
                    "cuda" if torch.cuda.is_available() else "cpu"
                )
        else:
            self.device = device

        # Move model to device
        self.model.to(self.device)

        # Configure optimizer
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=self.weight_decay,
        )

        # Configure learning rate scheduler
        self.scheduler = self.get_lr_scheduler(self.optimizer)

        # Configure loss function
        self.criterion = LabelSmoothing(smoothing=label_smoothing, pad_idx=pad_idx)

        # Initialize gradient scaler for mixed precision
        self.scaler = torch.cuda.amp.GradScaler() if self.use_gradient_scaling else None

        # Initialize training state
        self.current_epoch = 0
        self.global_step = 0
        self.best_val_loss = float("inf")
        self.patience_counter = 0

        # Enhanced history tracking
        self.history = {
            "train_loss": [],
            "train_ppl": [],
            "val_loss": [],
            "val_ppl": [],
            "learning_rates": [],
            # Add detailed statistics tracking
            "train_loss_stats": [],  # Will store [min, max, mean, var] for each epoch
            "train_ppl_stats": [],
            "val_loss_stats": [],
            "val_ppl_stats": [],
            # Per-epoch metrics
            "epoch_losses": [],  # Will store all losses for current epoch
            "epoch_ppls": [],  # Will store all perplexities for current epoch
            "epoch_lrs": [],  # Will store all learning rates for current epoch
        }

        # Initialize epoch end callback with proper type annotation
        self.epoch_end_callback: Optional[
            Callable[[int, torch.nn.Module, Any], None]
        ] = None

        # Gradient accumulation settings
        self.gradient_accumulation_steps = gradient_accumulation_steps

        # Calculate effective batch size, handling None case
        dataloader_batch_size = getattr(self.train_dataloader, "batch_size", None)
        base_batch_size = (
            dataloader_batch_size if dataloader_batch_size is not None else 32
        )
        self.effective_batch_size = base_batch_size * gradient_accumulation_steps

        # For logging purposes
        if self.gradient_accumulation_steps > 1:
            print(
                f"Using gradient accumulation with {self.gradient_accumulation_steps} steps"
            )
            print(f"Effective batch size: {self.effective_batch_size}")

    def get_lr_scheduler(self, optimizer):
        """
        Create a learning rate scheduler based on the specified type.

        Args:
            optimizer: Optimizer to schedule

        Returns:
            Learning rate scheduler
        """
        # Define total steps for all schedulers that need it
        total_steps = (
            len(self.train_dataloader) * 100
        )  # Assume max 100 epochs as safety

        if self.scheduler_type == "inverse_sqrt":
            # Define inverse square root learning rate function with warmup
            def lr_lambda(step):
                # Linear warmup followed by inverse square root decay
                if step == 0:
                    step = 1
                return (
                    min(step ** (-0.5), step * self.warmup_steps ** (-1.5))
                    * self.warmup_steps**0.5
                )

            return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

        elif self.scheduler_type == "cosine":
            # Create cosine annealing scheduler with warm starts
            # First do linear warmup then cosine annealing
            warmup_steps = self.warmup_steps  # Create local reference for closure

            def cosine_warmup(step):
                if step < warmup_steps:
                    # Linear warmup
                    return float(step) / float(max(1, warmup_steps))
                else:
                    # Cosine annealing decay
                    step_adjusted = step - warmup_steps
                    total_adjusted = total_steps - warmup_steps
                    return 0.5 * (
                        1 + math.cos(math.pi * step_adjusted / total_adjusted)
                    )

            return torch.optim.lr_scheduler.LambdaLR(optimizer, cosine_warmup)

        elif self.scheduler_type == "linear":
            # Linear decay after warmup
            warmup_steps = self.warmup_steps  # Create local reference for closure

            def linear_warmup_decay(step):
                if step < warmup_steps:
                    # Linear warmup
                    return float(step) / float(max(1, warmup_steps))
                else:
                    # Linear decay
                    step_adjusted = step - warmup_steps
                    total_adjusted = total_steps - warmup_steps
                    return max(0.0, 1.0 - step_adjusted / total_adjusted)

            return torch.optim.lr_scheduler.LambdaLR(optimizer, linear_warmup_decay)

        elif self.scheduler_type == "constant":
            # Constant learning rate after warmup
            warmup_steps = self.warmup_steps  # Create local reference for closure

            def constant_warmup(step):
                if step < warmup_steps:
                    # Linear warmup
                    return float(step) / float(max(1, warmup_steps))
                else:
                    # Constant learning rate
                    return 1.0

            return torch.optim.lr_scheduler.LambdaLR(optimizer, constant_warmup)

        else:
            raise ValueError(
                f"Unknown scheduler type: {self.scheduler_type}. "
                "Supported types: 'inverse_sqrt', 'cosine', 'linear', 'constant'"
            )

    def train_epoch(self):
        """
        Train the model for one epoch.

        Returns:
            Average loss for the epoch
        """
        self.model.train()
        total_loss = 0
        total_tokens = 0
        start_time = time.time()

        # Reset epoch-specific tracking
        self.history["epoch_losses"] = []
        self.history["epoch_ppls"] = []
        self.history["epoch_lrs"] = []

        # Use tqdm for progress bar
        progress_bar = tqdm(
            enumerate(self.train_dataloader),
            total=len(self.train_dataloader),
            desc=f"Epoch {self.current_epoch+1}",
        )

        for i, batch in progress_bar:
            # Move batch to device
            src, tgt = batch["src"].to(self.device), batch["tgt"].to(self.device)

            # Prepare target for loss calculation (shift right)
            tgt_input = tgt[:, :-1]  # Remove last token
            tgt_output = tgt[:, 1:]  # Remove first token (usually BOS)

            # Create masks
            src_mask = create_padding_mask(src, self.pad_idx)
            tgt_mask = create_causal_mask(tgt_input.size(1), self.device)

            # Forward pass
            if self.use_gradient_scaling:
                with torch.cuda.amp.autocast():
                    logits = self.model(
                        src, tgt_input, src_mask=src_mask, tgt_mask=tgt_mask
                    )
            else:
                logits = self.model(
                    src, tgt_input, src_mask=src_mask, tgt_mask=tgt_mask
                )

            # Calculate loss
            loss = self.criterion(logits, tgt_output)

            # Backward pass
            if i % self.gradient_accumulation_steps == 0:
                self.optimizer.zero_grad()

            if self.use_gradient_scaling and self.scaler is not None:
                self.scaler.scale(loss / self.gradient_accumulation_steps).backward()
            else:
                (loss / self.gradient_accumulation_steps).backward()

            # Only update weights after accumulating gradients
            if (i + 1) % self.gradient_accumulation_steps == 0:
                if self.use_gradient_scaling and self.scaler is not None:
                    # Gradient clipping
                    if self.clip_grad > 0:
                        self.scaler.unscale_(self.optimizer)
                        torch.nn.utils.clip_grad_norm_(
                            self.model.parameters(), self.clip_grad
                        )
                    # Update weights
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    # Gradient clipping
                    if self.clip_grad > 0:
                        torch.nn.utils.clip_grad_norm_(
                            self.model.parameters(), self.clip_grad
                        )
                    # Update weights
                    self.optimizer.step()

                # Update learning rate
                self.scheduler.step()

            # Update statistics
            current_loss = loss.item()
            current_tokens = (tgt_output != self.pad_idx).sum().item()
            total_loss += current_loss * current_tokens
            total_tokens += current_tokens
            current_ppl = math.exp(min(current_loss, 100))

            # Track per-step metrics for this epoch
            self.history["epoch_losses"].append(current_loss)
            self.history["epoch_ppls"].append(current_ppl)
            self.history["epoch_lrs"].append(self.scheduler.get_last_lr()[0])

            # Update progress bar
            progress_bar.set_postfix(
                {
                    "loss": f"{current_loss:.4f}",
                    "ppl": f"{current_ppl:.2f}",
                    "lr": f"{self.scheduler.get_last_lr()[0]:.7f}",
                }
            )

            # Record learning rate
            self.history["learning_rates"].append(self.scheduler.get_last_lr()[0])

            # Update global step
            self.global_step += 1

        # Calculate epoch statistics
        avg_loss = total_loss / total_tokens
        avg_ppl = math.exp(min(avg_loss, 100))

        # Calculate detailed statistics
        epoch_losses = np.array(self.history["epoch_losses"])
        epoch_ppls = np.array(self.history["epoch_ppls"])

        loss_stats = [
            float(np.min(epoch_losses)),
            float(np.max(epoch_losses)),
            float(np.mean(epoch_losses)),
            float(np.var(epoch_losses)),
            float(np.median(epoch_losses)),
            float(np.quantile(epoch_losses, 0.25)),
            float(np.quantile(epoch_losses, 0.75)),
        ]

        ppl_stats = [
            float(np.min(epoch_ppls)),
            float(np.max(epoch_ppls)),
            float(np.mean(epoch_ppls)),
            float(np.var(epoch_ppls)),
            float(np.median(epoch_ppls)),
            float(np.quantile(epoch_ppls, 0.25)),
            float(np.quantile(epoch_ppls, 0.75)),
        ]

        # Record metrics
        self.history["train_loss"].append(avg_loss)
        self.history["train_ppl"].append(avg_ppl)
        self.history["train_loss_stats"].append(loss_stats)
        self.history["train_ppl_stats"].append(ppl_stats)

        # Print epoch summary with detailed statistics
        elapsed = time.time() - start_time
        print(
            f"\nEpoch {self.current_epoch+1} Training Statistics:"
            f"\n  Time: {elapsed:.2f}s"
            f"\n  Loss - Mean: {avg_loss:.4f}, Min: {loss_stats[0]:.4f}, Max: {loss_stats[1]:.4f}, Var: {loss_stats[3]:.4f}, Median: {loss_stats[4]:.4f}, Q1: {loss_stats[5]:.4f}, Q3: {loss_stats[6]:.4f}"
            f"\n  Perplexity - Mean: {avg_ppl:.2f}, Min: {ppl_stats[0]:.2f}, Max: {ppl_stats[1]:.2f}, Var: {ppl_stats[3]:.2f}, Median: {ppl_stats[4]:.2f}, Q1: {ppl_stats[5]:.2f}, Q3: {ppl_stats[6]:.2f}"
        )

        return avg_loss

    def validate(self):
        """
        Validate the model.

        Returns:
            Average validation loss
        """
        if self.val_dataloader is None:
            return None

        self.model.eval()
        total_loss = 0
        total_tokens = 0
        val_losses = []
        val_ppls = []

        with torch.no_grad():
            for batch in tqdm(self.val_dataloader, desc="Validation"):
                # Move batch to device
                src, tgt = batch["src"].to(self.device), batch["tgt"].to(self.device)

                # Prepare target for loss calculation (shift right)
                tgt_input = tgt[:, :-1]  # Remove last token
                tgt_output = tgt[:, 1:]  # Remove first token (usually BOS)

                # Create masks
                src_mask = create_padding_mask(src, self.pad_idx)
                tgt_mask = create_causal_mask(tgt_input.size(1), self.device)

                # Forward pass
                if self.use_gradient_scaling:
                    with torch.cuda.amp.autocast():
                        logits = self.model(
                            src, tgt_input, src_mask=src_mask, tgt_mask=tgt_mask
                        )
                else:
                    logits = self.model(
                        src, tgt_input, src_mask=src_mask, tgt_mask=tgt_mask
                    )

                # Calculate loss
                loss = self.criterion(logits, tgt_output)

                # Update statistics
                current_loss = loss.item()
                current_tokens = (tgt_output != self.pad_idx).sum().item()
                total_loss += current_loss * current_tokens
                total_tokens += current_tokens
                current_ppl = math.exp(min(current_loss, 100))

                val_losses.append(current_loss)
                val_ppls.append(current_ppl)

        # Calculate average metrics
        avg_loss = total_loss / total_tokens if total_tokens > 0 else float("inf")
        avg_ppl = math.exp(min(avg_loss, 100))

        # Calculate detailed statistics
        val_losses = np.array(val_losses)
        val_ppls = np.array(val_ppls)

        loss_stats = [
            float(np.min(val_losses)),
            float(np.max(val_losses)),
            float(np.mean(val_losses)),
            float(np.var(val_losses)),
            float(np.median(val_losses)),
            float(np.quantile(val_losses, 0.25)),
            float(np.quantile(val_losses, 0.75)),
        ]

        ppl_stats = [
            float(np.min(val_ppls)),
            float(np.max(val_ppls)),
            float(np.mean(val_ppls)),
            float(np.var(val_ppls)),
            float(np.median(val_ppls)),
            float(np.quantile(val_ppls, 0.25)),
            float(np.quantile(val_ppls, 0.75)),
        ]

        # Record metrics
        self.history["val_loss"].append(avg_loss)
        self.history["val_ppl"].append(avg_ppl)
        self.history["val_loss_stats"].append(loss_stats)
        self.history["val_ppl_stats"].append(ppl_stats)

        # Print validation summary with detailed statistics
        print(
            f"\nValidation Statistics:"
            f"\n  Loss - Mean: {avg_loss:.4f}, Min: {loss_stats[0]:.4f}, Max: {loss_stats[1]:.4f}, Var: {loss_stats[3]:.4f}, Median: {loss_stats[4]:.4f}, Q1: {loss_stats[5]:.4f}, Q3: {loss_stats[6]:.4f}"
            f"\n  Perplexity - Mean: {avg_ppl:.2f}, Min: {ppl_stats[0]:.2f}, Max: {ppl_stats[1]:.2f}, Var: {ppl_stats[3]:.2f}, Median: {ppl_stats[4]:.2f}, Q1: {ppl_stats[5]:.2f}, Q3: {ppl_stats[6]:.2f}   "
        )

        return avg_loss

    def train(self, epochs: int, save_path: Optional[str] = None):
        """
        Train the model for the specified number of epochs.

        Args:
            epochs: Number of epochs to train for
            save_path: Optional path to save model checkpoints

        Returns:
            Dictionary containing training history
        """
        print(f"Starting training on {self.device}...")
        start_time = time.time()

        # Create directories for plots if save_path is provided
        if save_path:
            plot_dir = os.path.join(os.path.dirname(save_path), "plots")
            os.makedirs(plot_dir, exist_ok=True)

        for epoch in range(epochs):
            self.current_epoch = epoch

            # Train one epoch
            train_loss = self.train_epoch()

            # Save epoch metrics plot if save_path is provided
            if save_path:
                epoch_plot_path = os.path.join(plot_dir, f"epoch_{epoch+1}_metrics.png")
                self.plot_epoch_metrics(epoch, save_path=epoch_plot_path)

            # Validate
            val_loss = self.validate()

            if val_loss is not None:
                # Record validation metrics
                self.history["val_loss"].append(val_loss)
                self.history["val_ppl"].append(math.exp(min(val_loss, 100)))

                # Early stopping check
                if self.early_stopping_patience is not None:
                    if val_loss < self.best_val_loss:
                        self.best_val_loss = val_loss
                        self.patience_counter = 0
                        if save_path:
                            self.save_checkpoint(save_path)
                    else:
                        self.patience_counter += 1
                        if self.patience_counter >= self.early_stopping_patience:
                            print(f"Early stopping triggered after {epoch + 1} epochs")
                            break

            # Call epoch end callback if defined
            if self.epoch_end_callback is not None:
                try:
                    self.epoch_end_callback(epoch, self.model, self)
                except Exception as e:
                    print(f"Error in epoch end callback: {e}")

        # Print total training time
        total_time = time.time() - start_time
        print(f"Training completed in {total_time:.2f}s")

        # Save final plots if save_path is provided
        if save_path:
            # Save overall training history plot
            history_plot_path = os.path.join(plot_dir, "training_history.png")
            self.plot_training_history(save_path=history_plot_path)

        # Always save the final model if a save path is provided
        if save_path and (
            self.early_stopping_patience is None
            or self.patience_counter < self.early_stopping_patience
        ):
            print("Saving final model checkpoint...")
            self.save_checkpoint(save_path)

        return self.history

    def save_checkpoint(self, path: str):
        """
        Save a training checkpoint to disk.

        Args:
            path: Path where the checkpoint should be saved
        """
        try:
            # Extract model configuration information
            model_config = {}
            if hasattr(self.model, "encoder") and hasattr(self.model, "decoder"):
                # Get vocabulary sizes
                model_config["src_vocab_size"] = (
                    self.model.encoder.token_embedding.embedding.weight.shape[0]
                )
                model_config["tgt_vocab_size"] = (
                    self.model.decoder.token_embedding.embedding.weight.shape[0]
                )

                # Get embedding dimensions
                model_config["d_model"] = (
                    self.model.encoder.token_embedding.embedding.weight.shape[1]
                )

                # Get number of layers
                if hasattr(self.model.encoder, "layers"):
                    model_config["num_encoder_layers"] = len(self.model.encoder.layers)
                if hasattr(self.model.decoder, "layers"):
                    model_config["num_decoder_layers"] = len(self.model.decoder.layers)

                # Get number of attention heads if available
                if hasattr(self.model.encoder.layers[0].self_attn, "num_heads"):
                    model_config["num_heads"] = self.model.encoder.layers[
                        0
                    ].self_attn.num_heads

                # Get feed-forward dimension if available
                if hasattr(self.model.encoder.layers[0].feed_forward, "linear1"):
                    if hasattr(
                        self.model.encoder.layers[0].feed_forward.linear1, "linear"
                    ):
                        model_config["d_ff"] = self.model.encoder.layers[
                            0
                        ].feed_forward.linear1.linear.weight.shape[0]

            checkpoint = {
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "scheduler_state_dict": self.scheduler.state_dict(),
                "current_epoch": self.current_epoch,
                "global_step": self.global_step,
                "best_val_loss": self.best_val_loss,
                "patience_counter": self.patience_counter,
                "history": self.history,
                "model_config": model_config,
            }

            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(path), exist_ok=True)

            # Save the checkpoint
            torch.save(checkpoint, path)
            print(f"\n===== Model saved to: {os.path.abspath(path)} =====")
            print(f"Model configuration: {model_config}")

            # Verify the file was created
            if os.path.exists(path):
                file_size = os.path.getsize(path) / (1024 * 1024)  # Size in MB
                print(f"File size: {file_size:.2f} MB")
            else:
                print(f"WARNING: File {path} was not created despite no errors!")

        except Exception as e:
            print(f"\n===== ERROR SAVING MODEL: {e} =====")
            print(f"Attempted to save to: {os.path.abspath(path)}")
            import traceback

            traceback.print_exc()

    def load_checkpoint(self, path: str, strict: bool = True):
        """
        Load a training checkpoint from disk.

        Args:
            path: Path to the checkpoint file
            strict: Whether to strictly enforce that the keys in state_dict match the keys
                   returned by the module's state_dict function
        """
        print(f"Loading checkpoint from {path}...")
        try:
            checkpoint = torch.load(path, map_location=self.device)

            # Check what's in the checkpoint
            checkpoint_contains = [k for k in checkpoint.keys()]
            print(f"Checkpoint contains: {checkpoint_contains}")

            # Load model state
            if "model_state_dict" in checkpoint:
                try:
                    self.model.load_state_dict(
                        checkpoint["model_state_dict"], strict=strict
                    )
                    print("Successfully loaded model state")
                except Exception as e:
                    print(f"Error loading model state: {e}")
                    if not strict:
                        print("Continuing with non-strict loading")
                        # Try again with strict=False if not already
                        try:
                            self.model.load_state_dict(
                                checkpoint["model_state_dict"], strict=False
                            )
                            print("Successfully loaded model state with strict=False")
                        except Exception as e2:
                            print(f"Error even with non-strict loading: {e2}")
            else:
                print("No model state found in checkpoint")

            # Load optimizer state
            if "optimizer_state_dict" in checkpoint:
                try:
                    self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
                    # Move optimizer state to the current device if needed
                    for state in self.optimizer.state.values():
                        for k, v in state.items():
                            if isinstance(v, torch.Tensor):
                                state[k] = v.to(self.device)
                    print("Successfully loaded optimizer state")
                except Exception as e:
                    print(f"Error loading optimizer state: {e}")
            else:
                print("No optimizer state found in checkpoint")

            # Load scheduler state
            if "scheduler_state_dict" in checkpoint:
                try:
                    self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
                    print("Successfully loaded scheduler state")
                except Exception as e:
                    print(f"Error loading scheduler state: {e}")
            else:
                print("No scheduler state found in checkpoint")

            # Restore training state
            if "current_epoch" in checkpoint:
                self.current_epoch = checkpoint["current_epoch"]
                print(f"Restored training at epoch {self.current_epoch}")

            if "global_step" in checkpoint:
                self.global_step = checkpoint["global_step"]
                print(f"Restored global step {self.global_step}")

            if "best_val_loss" in checkpoint:
                self.best_val_loss = checkpoint["best_val_loss"]
                print(f"Restored best validation loss: {self.best_val_loss:.4f}")

            if "patience_counter" in checkpoint:
                self.patience_counter = checkpoint["patience_counter"]
                print(f"Restored patience counter: {self.patience_counter}")

            if "history" in checkpoint:
                self.history = checkpoint["history"]
                print("Restored training history")
                if self.history.get("train_loss"):
                    print(f"History contains {len(self.history['train_loss'])} epochs")

            print("Checkpoint loading complete!")

        except Exception as e:
            print(f"Error loading checkpoint: {e}")
            import traceback

            traceback.print_exc()
            raise

    def restore_from_checkpoint(
        self,
        path: str,
        strict: bool = True,
        reset_optimizer: bool = False,
        reset_scheduler: bool = False,
    ):
        """
        Restore the full training state from a checkpoint.

        This is a more comprehensive version of load_checkpoint that handles all aspects of
        training state restoration and provides options for partial resets.

        Args:
            path: Path to the checkpoint file
            strict: Whether to strictly enforce model state dict key matching
            reset_optimizer: If True, don't restore optimizer state (useful when changing batch size)
            reset_scheduler: If True, don't restore scheduler state (useful when changing learning rate)

        Returns:
            bool: Whether restoration was successful
        """
        print(f"Restoring training state from: {path}")

        try:
            checkpoint = torch.load(path, map_location=self.device)

            # Validate checkpoint contents
            if "model_state_dict" not in checkpoint:
                print("ERROR: Checkpoint does not contain model state dict")
                return False

            # Load model state dict (with proper error handling)
            try:
                self.model.load_state_dict(
                    checkpoint["model_state_dict"], strict=strict
                )
                print("✓ Model weights restored successfully")
            except Exception as e:
                print(f"! Error restoring model weights: {e}")
                if not strict:
                    print("  Attempting non-strict loading...")
                    try:
                        self.model.load_state_dict(
                            checkpoint["model_state_dict"], strict=False
                        )
                        print("✓ Model weights restored with strict=False")
                    except Exception as e2:
                        print(f"! Failed even with non-strict loading: {e2}")
                        return False

            # Restore optimizer state if requested
            if not reset_optimizer and "optimizer_state_dict" in checkpoint:
                try:
                    self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
                    # Handle device placement for optimizer state
                    for state in self.optimizer.state.values():
                        for k, v in state.items():
                            if isinstance(v, torch.Tensor):
                                state[k] = v.to(self.device)
                    print("✓ Optimizer state restored")
                except Exception as e:
                    print(f"! Error restoring optimizer state: {e}")
                    print("  Creating fresh optimizer state")
            else:
                # Log that we're intentionally skipping optimizer restoration
                if reset_optimizer:
                    print("✓ Optimizer state reset as requested")
                else:
                    print("! No optimizer state found in checkpoint")

            # Restore scheduler state if requested
            if not reset_scheduler and "scheduler_state_dict" in checkpoint:
                try:
                    self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
                    print("✓ Scheduler state restored")
                except Exception as e:
                    print(f"! Error restoring scheduler state: {e}")
                    print("  Creating fresh scheduler state")
            else:
                # Log that we're intentionally skipping scheduler restoration
                if reset_scheduler:
                    print("✓ Scheduler state reset as requested")
                else:
                    print("! No scheduler state found in checkpoint")

            # Restore training progress metrics
            training_metrics = {
                "current_epoch": "Current epoch",
                "global_step": "Global step",
                "best_val_loss": "Best validation loss",
                "patience_counter": "Early stopping patience counter",
            }

            # Restore each training metric if available
            for attr, desc in training_metrics.items():
                if attr in checkpoint:
                    setattr(self, attr, checkpoint[attr])
                    if attr == "best_val_loss":
                        print(f"✓ {desc} restored: {getattr(self, attr):.4f}")
                    else:
                        print(f"✓ {desc} restored: {getattr(self, attr)}")
                else:
                    print(f"! {desc} not found in checkpoint")

            # Restore training history
            if "history" in checkpoint:
                # Check for non-empty history to avoid initializing with empty data
                if (
                    checkpoint["history"].get("train_loss")
                    and len(checkpoint["history"]["train_loss"]) > 0
                ):
                    self.history = checkpoint["history"]
                    print(
                        f"✓ Training history restored ({len(self.history['train_loss'])} epochs)"
                    )
                else:
                    print("! History found but contains no training data")
            else:
                print("! No training history found in checkpoint")

            print("Training state restoration complete!")
            return True

        except FileNotFoundError:
            print(f"! Checkpoint file not found: {path}")
            return False
        except Exception as e:
            print(f"! Error during checkpoint restoration: {e}")
            import traceback

            traceback.print_exc()
            return False

    def plot_learning_rate(self):
        """
        Plot the learning rate schedule over training steps.
        """
        plt.figure(figsize=(10, 5))
        plt.plot(self.history["learning_rates"])
        plt.title("Learning Rate Schedule")
        plt.xlabel("Steps")
        plt.ylabel("Learning Rate")
        plt.grid(True)
        plt.show()

    def plot_training_history(self, save_path: Optional[str] = None):
        """
        Plot training and validation metrics over epochs.

        Args:
            save_path: Path to save the plot. If None, displays the plot.
        """
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))

        # Plot losses
        ax1.plot(self.history["train_loss"], label="Train Loss")
        if self.history["val_loss"]:
            ax1.plot(self.history["val_loss"], label="Val Loss")
        ax1.set_title("Training and Validation Loss")
        ax1.set_xlabel("Epochs")
        ax1.set_ylabel("Loss")
        ax1.legend()
        ax1.grid(True)

        # Plot perplexities
        ax2.plot(self.history["train_ppl"], label="Train Perplexity")
        if self.history["val_ppl"]:
            ax2.plot(self.history["val_ppl"], label="Val Perplexity")
        ax2.set_title("Training and Validation Perplexity")
        ax2.set_xlabel("Epochs")
        ax2.set_ylabel("Perplexity")
        ax2.legend()
        ax2.grid(True)

        # Plot learning rate
        ax3.plot(self.history["learning_rates"])
        ax3.set_title("Learning Rate Schedule")
        ax3.set_xlabel("Steps")
        ax3.set_ylabel("Learning Rate")
        ax3.grid(True)

        # Plot min/max ranges
        epochs = range(1, len(self.history["train_loss"]) + 1)

        # Plot loss ranges
        train_loss_min = [stats[0] for stats in self.history["train_loss_stats"]]
        train_loss_max = [stats[1] for stats in self.history["train_loss_stats"]]
        ax4.fill_between(
            epochs, train_loss_min, train_loss_max, alpha=0.3, label="Train Loss Range"
        )

        if self.history["val_loss"]:
            val_loss_min = [stats[0] for stats in self.history["val_loss_stats"]]
            val_loss_max = [stats[1] for stats in self.history["val_loss_stats"]]
            ax4.fill_between(
                epochs, val_loss_min, val_loss_max, alpha=0.3, label="Val Loss Range"
            )

        ax4.set_title("Loss Ranges per Epoch")
        ax4.set_xlabel("Epochs")
        ax4.set_ylabel("Loss")
        ax4.legend()
        ax4.grid(True)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path)
            plt.close()
        else:
            plt.show()

    def plot_epoch_metrics(self, epoch: int, save_path: Optional[str] = None):
        """
        Plot detailed metrics for a specific epoch.

        Args:
            epoch: The epoch number to plot metrics for
            save_path: Path to save the plot. If None, displays the plot.
        """
        if not self.history["epoch_losses"]:
            print("No epoch data available to plot")
            return

        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 12))

        steps = range(len(self.history["epoch_losses"]))

        # Plot loss evolution
        ax1.plot(steps, self.history["epoch_losses"], label="Loss")
        ax1.set_title(f"Loss Evolution - Epoch {epoch+1}")
        ax1.set_xlabel("Steps")
        ax1.set_ylabel("Loss")
        ax1.grid(True)

        # Plot perplexity evolution
        ax2.plot(steps, self.history["epoch_ppls"], label="Perplexity")
        ax2.set_title(f"Perplexity Evolution - Epoch {epoch+1}")
        ax2.set_xlabel("Steps")
        ax2.set_ylabel("Perplexity")
        ax2.grid(True)

        # Plot learning rate evolution
        ax3.plot(steps, self.history["epoch_lrs"], label="Learning Rate")
        ax3.set_title(f"Learning Rate Evolution - Epoch {epoch+1}")
        ax3.set_xlabel("Steps")
        ax3.set_ylabel("Learning Rate")
        ax3.grid(True)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path)
            plt.close()
        else:
            plt.show()


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
        "module_purpose": "Implements a specialized trainer for transformer models with support for encoder-decoder architectures and advanced training features",
        "key_classes": [
            {
                "name": "TransformerTrainer",
                "purpose": "Main trainer class for transformer model training with comprehensive training and evaluation capabilities",
                "key_methods": [
                    {
                        "name": "train",
                        "signature": "train(self, epochs: int, save_path: Optional[str] = None)",
                        "brief_description": "Main training loop with support for validation and early stopping",
                    },
                    {
                        "name": "train_epoch",
                        "signature": "train_epoch(self)",
                        "brief_description": "Trains the model for a single epoch with progress tracking",
                    },
                    {
                        "name": "validate",
                        "signature": "validate(self)",
                        "brief_description": "Evaluates the model on validation data and returns loss metrics",
                    },
                    {
                        "name": "get_lr_scheduler",
                        "signature": "get_lr_scheduler(self, optimizer)",
                        "brief_description": "Creates learning rate scheduler with warmup and decay strategies",
                    },
                    {
                        "name": "save_checkpoint",
                        "signature": "save_checkpoint(self, path: str)",
                        "brief_description": "Saves model checkpoint with training state",
                    },
                    {
                        "name": "load_checkpoint",
                        "signature": "load_checkpoint(self, path: str)",
                        "brief_description": "Loads model checkpoint and training state",
                    },
                    {
                        "name": "plot_learning_rate",
                        "signature": "plot_learning_rate(self)",
                        "brief_description": "Visualizes the learning rate schedule",
                    },
                    {
                        "name": "plot_training_history",
                        "signature": "plot_training_history(self)",
                        "brief_description": "Visualizes training and validation metrics over time",
                    },
                    {
                        "name": "plot_epoch_metrics",
                        "signature": "plot_epoch_metrics(self, epoch: int)",
                        "brief_description": "Visualizes detailed metrics for a specific epoch",
                    },
                ],
                "inheritance": "object",
                "dependencies": [
                    "torch",
                    "torch.nn",
                    "numpy",
                    "matplotlib",
                    "tqdm",
                    "transformer_utils",
                ],
            }
        ],
        "external_dependencies": ["torch", "numpy", "matplotlib", "tqdm"],
        "complexity_score": 8,  # High complexity due to comprehensive training loop, masking, and visualization features
    }
