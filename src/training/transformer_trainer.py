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

# src/training/transformer_trainer.py
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Dict, List, Tuple, Optional, Any
import time
import math
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import os

from .transformer_utils import create_padding_mask, create_causal_mask, LabelSmoothing

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
        
        # Set device
        if device is None:
            if torch.backends.mps.is_available():
                self.device = torch.device("mps")
            else:
                self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device
        
        # Move model to device
        self.model.to(self.device)
        
        # Configure optimizer
        self.optimizer = torch.optim.Adam(
            model.parameters(), lr=lr, betas=betas, eps=eps
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
        
        # Initialize history
        self.history = {
            "train_loss": [],
            "train_ppl": [],
            "val_loss": [],
            "val_ppl": [],
            "learning_rates": [],
        }
    
    def get_lr_scheduler(self, optimizer):
        """
        Create a learning rate scheduler based on the specified type.
        
        Args:
            optimizer: Optimizer to schedule
            
        Returns:
            Learning rate scheduler
        """
        # Define total steps for all schedulers that need it
        total_steps = len(self.train_dataloader) * 100  # Assume max 100 epochs as safety
        
        if self.scheduler_type == "inverse_sqrt":
            # Define inverse square root learning rate function with warmup
            def lr_lambda(step):
                # Linear warmup followed by inverse square root decay
                if step == 0:
                    step = 1
                return min(
                    step ** (-0.5),
                    step * self.warmup_steps ** (-1.5)
                ) * self.warmup_steps ** 0.5
                
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
                    return 0.5 * (1 + math.cos(math.pi * step_adjusted / total_adjusted))
                    
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
            raise ValueError(f"Unknown scheduler type: {self.scheduler_type}. " 
                            "Supported types: 'inverse_sqrt', 'cosine', 'linear', 'constant'")
    
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
                    logits = self.model(src, tgt_input, src_mask=src_mask, tgt_mask=tgt_mask)
            else:
                logits = self.model(src, tgt_input, src_mask=src_mask, tgt_mask=tgt_mask)
            
            # Calculate loss
            loss = self.criterion(logits, tgt_output)
            
            # Backward pass
            self.optimizer.zero_grad()
            
            if self.use_gradient_scaling and self.scaler is not None:
                self.scaler.scale(loss).backward()
                
                # Gradient clipping
                if self.clip_grad > 0:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip_grad)
                
                # Update weights
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                
                # Gradient clipping
                if self.clip_grad > 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip_grad)
                
                # Update weights
                self.optimizer.step()
            
            # Update learning rate
            self.scheduler.step()
            
            # Update statistics
            current_loss = loss.item()
            current_tokens = (tgt_output != self.pad_idx).sum().item()
            total_loss += current_loss * current_tokens
            total_tokens += current_tokens
            
            # Update progress bar
            progress_bar.set_postfix({
                "loss": f"{current_loss:.4f}",
                "ppl": f"{math.exp(current_loss):.2f}",
                "lr": f"{self.scheduler.get_last_lr()[0]:.7f}",
            })
            
            # Record learning rate
            self.history["learning_rates"].append(self.scheduler.get_last_lr()[0])
            
            # Update global step
            self.global_step += 1
        
        # Calculate average loss and perplexity
        avg_loss = total_loss / total_tokens
        avg_ppl = math.exp(min(avg_loss, 100))  # Cap perplexity at exp(100)
        
        # Record metrics
        self.history["train_loss"].append(avg_loss)
        self.history["train_ppl"].append(avg_ppl)
        
        # Print epoch summary
        elapsed = time.time() - start_time
        print(f"Epoch {self.current_epoch+1} completed in {elapsed:.2f}s - "
              f"loss: {avg_loss:.4f}, ppl: {avg_ppl:.2f}")
        
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
                        logits = self.model(src, tgt_input, src_mask=src_mask, tgt_mask=tgt_mask)
                else:
                    logits = self.model(src, tgt_input, src_mask=src_mask, tgt_mask=tgt_mask)
                
                # Calculate loss
                loss = self.criterion(logits, tgt_output)
                
                # Update statistics
                current_loss = loss.item()
                current_tokens = (tgt_output != self.pad_idx).sum().item()
                total_loss += current_loss * current_tokens
                total_tokens += current_tokens
        
        # Calculate average loss and perplexity
        avg_loss = total_loss / total_tokens if total_tokens > 0 else float("inf")
        avg_ppl = math.exp(min(avg_loss, 100))  # Cap perplexity at exp(100)
        
        # Record metrics
        self.history["val_loss"].append(avg_loss)
        self.history["val_ppl"].append(avg_ppl)
        
        print(f"Validation - loss: {avg_loss:.4f}, ppl: {avg_ppl:.2f}")
        
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
        
        for epoch in range(epochs):
            self.current_epoch = epoch
            
            # Train one epoch
            train_loss = self.train_epoch()
            
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
        
        # Print total training time
        total_time = time.time() - start_time
        print(f"Training completed in {total_time:.2f}s")
        
        return self.history
    
    def save_checkpoint(self, path: str):
        """
        Save a training checkpoint to disk.
        
        Args:
            path: Path where the checkpoint should be saved
        """
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'current_epoch': self.current_epoch,
            'global_step': self.global_step,
            'best_val_loss': self.best_val_loss,
            'patience_counter': self.patience_counter,
            'history': self.history
        }
        torch.save(checkpoint, path)
    
    def load_checkpoint(self, path: str):
        """
        Load a training checkpoint from disk.
        
        Args:
            path: Path to the checkpoint file
        """
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.current_epoch = checkpoint['current_epoch']
        self.global_step = checkpoint['global_step']
        self.best_val_loss = checkpoint['best_val_loss']
        self.patience_counter = checkpoint['patience_counter']
        self.history = checkpoint['history']
    
    def plot_learning_rate(self):
        """
        Plot the learning rate schedule over training steps.
        """
        plt.figure(figsize=(10, 5))
        plt.plot(self.history['learning_rates'])
        plt.title('Learning Rate Schedule')
        plt.xlabel('Steps')
        plt.ylabel('Learning Rate')
        plt.grid(True)
        plt.show()
    
    def plot_training_history(self):
        """
        Plot training and validation metrics over epochs.
        """
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10))
        
        # Plot losses
        ax1.plot(self.history['train_loss'], label='Train Loss')
        if self.history['val_loss']:
            ax1.plot(self.history['val_loss'], label='Val Loss')
        ax1.set_title('Training and Validation Loss')
        ax1.set_xlabel('Epochs')
        ax1.set_ylabel('Loss')
        ax1.legend()
        ax1.grid(True)
        
        # Plot perplexities
        ax2.plot(self.history['train_ppl'], label='Train Perplexity')
        if self.history['val_ppl']:
            ax2.plot(self.history['val_ppl'], label='Val Perplexity')
        ax2.set_title('Training and Validation Perplexity')
        ax2.set_xlabel('Epochs')
        ax2.set_ylabel('Perplexity')
        ax2.legend()
        ax2.grid(True)
        
        plt.tight_layout()
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
                        "brief_description": "Main training loop with support for validation and early stopping"
                    },
                    {
                        "name": "train_epoch",
                        "signature": "train_epoch(self)",
                        "brief_description": "Trains the model for a single epoch with progress tracking"
                    },
                    {
                        "name": "validate",
                        "signature": "validate(self)",
                        "brief_description": "Evaluates the model on validation data and returns loss metrics"
                    },
                    {
                        "name": "get_lr_scheduler",
                        "signature": "get_lr_scheduler(self, optimizer)",
                        "brief_description": "Creates learning rate scheduler with warmup and decay strategies"
                    },
                    {
                        "name": "save_checkpoint",
                        "signature": "save_checkpoint(self, path: str)",
                        "brief_description": "Saves model checkpoint with training state"
                    },
                    {
                        "name": "load_checkpoint",
                        "signature": "load_checkpoint(self, path: str)",
                        "brief_description": "Loads model checkpoint and training state"
                    },
                    {
                        "name": "plot_learning_rate",
                        "signature": "plot_learning_rate(self)",
                        "brief_description": "Visualizes the learning rate schedule"
                    },
                    {
                        "name": "plot_training_history",
                        "signature": "plot_training_history(self)",
                        "brief_description": "Visualizes training and validation metrics over time"
                    }
                ],
                "inheritance": "object",
                "dependencies": ["torch", "torch.nn", "numpy", "matplotlib", "tqdm", "transformer_utils"]
            }
        ],
        "external_dependencies": ["torch", "numpy", "matplotlib", "tqdm"],
        "complexity_score": 8  # High complexity due to comprehensive training loop, masking, and visualization features
    }