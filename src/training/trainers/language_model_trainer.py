"""MODULE: language_model_trainer.py
PURPOSE: Implements a specialized trainer for language modeling tasks with support for causal language modeling, evaluation, and generation.

KEY COMPONENTS:
- LanguageModelTrainer: Main trainer class for language model training with support for:
  - Learning rate scheduling with warmup and decay
  - Gradient clipping and optimization
  - Training and validation loops
  - Model checkpointing and loading
  - Training visualization and metrics tracking

DEPENDENCIES:
- PyTorch (torch, torch.nn, torch.nn.functional)
- NumPy
- Matplotlib
- tqdm

SPECIAL NOTES:
- Supports both CPU and GPU training with automatic device selection
- Implements perplexity-based evaluation metrics
- Includes comprehensive training visualization capabilities
"""

# src/training/language_model_trainer.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Union, Any
import time
import math
import numpy as np
import matplotlib.pyplot as plt
import os
from tqdm import tqdm

class LanguageModelTrainer:
    """
    Trainer specialized for language modeling tasks.
    
    This trainer handles the causal language modeling objective
    and includes utilities for evaluation and generation.
    """
    
    def __init__(
        self,
        model: nn.Module,
        train_dataloader: torch.utils.data.DataLoader,
        val_dataloader: Optional[torch.utils.data.DataLoader] = None,
        learning_rate: float = 5e-5,
        weight_decay: float = 0.01,
        warmup_steps: int = 1000,
        max_grad_norm: float = 1.0,
        device: Optional[torch.device] = None,
        log_dir: str = "logs",
        **kwargs
    ):
        """
        Initialize the language model trainer.
        
        Args:
            model: The language model to train
            train_dataloader: DataLoader for training data
            val_dataloader: Optional DataLoader for validation data
            learning_rate: Peak learning rate after warmup
            weight_decay: Weight decay for regularization
            warmup_steps: Number of warmup steps
            max_grad_norm: Maximum gradient norm for clipping
            device: Device to train on
            log_dir: Directory for logging
            **kwargs: Additional arguments
        """
        self.model = model
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.warmup_steps = warmup_steps
        self.max_grad_norm = max_grad_norm
        self.log_dir = log_dir
        
        # Create log directory
        os.makedirs(log_dir, exist_ok=True)
        
        # Set device
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else 
                                     "mps" if torch.backends.mps.is_available() else 
                                     "cpu")
        else:
            self.device = device
        
        # Move model to device
        self.model.to(self.device)
        
        # Create optimizer
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
        )
        
        # Create learning rate scheduler
        self.scheduler = self._create_lr_scheduler()
        
        # Initialize tracking variables
        self.global_step = 0
        self.best_val_loss = float('inf')
        self.train_losses = []
        self.val_losses = []
        self.learning_rates = []
        self.train_perplexities = []
        self.val_perplexities = []
        
    def _create_lr_scheduler(self):
        """
        Create a learning rate scheduler with linear warmup and decay.
        
        Returns:
            Learning rate scheduler
        """
        def lr_lambda(current_step):
            # Linear warmup
            if current_step < self.warmup_steps:
                return float(current_step) / float(max(1, self.warmup_steps))
            # Linear decay
            return max(0.0, 
                     float(1.0 - current_step / (len(self.train_dataloader) * self.num_epochs)))
        
        return torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda)
    
    def _log_training_step(self, loss, lr, step):
        """
        Log training metrics for a single step.
        
        Args:
            loss: Current training loss value
            lr: Current learning rate
            step: Current training step number
        """
        self.train_losses.append(loss)
        self.learning_rates.append(lr)
        
        # Calculate perplexity
        perplexity = math.exp(loss)
        self.train_perplexities.append(perplexity)
        
        # Don't print every few seconds, let the tqdm progress bar handle displaying metrics
        # Only print at milestone steps for record-keeping
        if step % 10000 == 0:
            print(f"Step {step}: Loss = {loss:.4f}, Perplexity = {perplexity:.2f}, LR = {lr:.7f}")
    
    def train(self, num_epochs, save_dir="models/language", model_name="language_model"):
        """
        Train the language model.
        
        Args:
            num_epochs: Number of epochs to train for
            save_dir: Directory to save model checkpoints
            model_name: Base name for saved models
            
        Returns:
            Dictionary with training statistics
        """
        # Create save directory
        os.makedirs(save_dir, exist_ok=True)
        
        # Store number of epochs for scheduler
        self.num_epochs = num_epochs
        
        print(f"Starting training on {self.device}...")
        start_time = time.time()
        
        for epoch in range(num_epochs):
            epoch_start_time = time.time()
            
            # Training loop
            self.model.train()
            train_loss = 0.0
            num_batches = len(self.train_dataloader)
            
            # Create tqdm progress bar with additional metrics
            pbar = tqdm(self.train_dataloader, desc=f"Epoch {epoch+1}/{num_epochs}", 
                         dynamic_ncols=True, leave=True, position=0)
            
            for i, batch in enumerate(pbar):
                # Move batch to device
                batch = {k: v.to(self.device) for k, v in batch.items()}
                
                # Forward pass
                outputs = self.model(
                    src=batch["input_ids"],
                    tgt=batch["labels"],
                    src_mask=batch.get("src_mask"),
                    tgt_mask=batch.get("tgt_mask"),
                    memory_mask=batch.get("memory_mask")
                )
                
                # Calculate loss
                logits = outputs.logits if hasattr(outputs, "logits") else outputs
                
                # Reshape for cross-entropy
                shift_logits = logits[..., :-1, :].contiguous()
                shift_labels = batch["labels"][..., 1:].contiguous()
                
                # Calculate loss (ignore padding)
                loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
                loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), 
                              shift_labels.view(-1))
                
                # Backward pass
                self.optimizer.zero_grad()
                loss.backward()
                
                # Gradient clipping
                if self.max_grad_norm > 0:
                    nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                
                # Update weights
                self.optimizer.step()
                
                # Update learning rate
                self.scheduler.step()
                
                # Log training information
                self._log_training_step(
                    loss.item(),
                    self.scheduler.get_last_lr()[0],
                    self.global_step
                )
                
                # Update training loss
                curr_loss = loss.item()
                train_loss += curr_loss
                
                # Update progress bar with current metrics - force update
                curr_avg_loss = train_loss / (i + 1)
                curr_ppl = math.exp(curr_avg_loss)
                pbar.set_postfix({
                    'loss': f'{curr_loss:.4f}',
                    'avg_loss': f'{curr_avg_loss:.4f}',
                    'ppl': f'{curr_ppl:.2f}',
                    'lr': f'{self.scheduler.get_last_lr()[0]:.7f}'
                }, refresh=True)  # Force refresh
                
                # Update global step
                self.global_step += 1
            
            # Calculate average training loss
            avg_train_loss = train_loss / num_batches
            train_perplexity = math.exp(avg_train_loss)
            print(f"Epoch {epoch+1} - Train Loss: {avg_train_loss:.4f}, Train Perplexity: {train_perplexity:.2f}")
            
            # Validation
            if self.val_dataloader is not None:
                val_loss, val_perplexity = self.evaluate()
                print(f"Epoch {epoch+1} - Val Loss: {val_loss:.4f}, Val Perplexity: {val_perplexity:.2f}")
                
                # Save best model
                if val_loss < self.best_val_loss:
                    self.best_val_loss = val_loss
                    self.save_model(f"{save_dir}/{model_name}_best.pt")
                    print(f"New best model saved with validation loss: {val_loss:.4f}")
            
            # Save checkpoint
            if (epoch + 1) % 5 == 0 or epoch == num_epochs - 1:
                self.save_model(f"{save_dir}/{model_name}_epoch{epoch+1}.pt")
            
            # Print epoch time
            epoch_time = time.time() - epoch_start_time
            print(f"Epoch {epoch+1} completed in {epoch_time:.2f}s")
        
        # Print total training time
        total_time = time.time() - start_time
        print(f"Training completed in {total_time:.2f}s")
        
        # Return training statistics
        return {
            "train_losses": self.train_losses,
            "val_losses": self.val_losses,
            "learning_rates": self.learning_rates,
            "train_perplexities": self.train_perplexities,
            "val_perplexities": self.val_perplexities,
            "best_val_loss": self.best_val_loss,
        }
    
    def evaluate(self):
        """
        Evaluate the model on the validation set.
        
        Returns:
            Tuple of (average loss, perplexity)
        """
        if self.val_dataloader is None:
            return 0.0, 0.0
        
        self.model.eval()
        val_loss = 0.0
        num_batches = len(self.val_dataloader)
        
        with torch.no_grad():
            # Create tqdm progress bar with metrics
            pbar = tqdm(self.val_dataloader, desc="Validation", 
                        dynamic_ncols=True, leave=True, position=0)
            
            for i, batch in enumerate(pbar):
                # Move batch to device
                batch = {k: v.to(self.device) for k, v in batch.items()}
                
                # Forward pass
                outputs = self.model(
                    src=batch["input_ids"],
                    tgt=batch["labels"],
                    src_mask=batch.get("src_mask"),
                    tgt_mask=batch.get("tgt_mask"),
                    memory_mask=batch.get("memory_mask")
                )
                
                # Calculate loss
                logits = outputs.logits if hasattr(outputs, "logits") else outputs
                
                # Reshape for cross-entropy
                shift_logits = logits[..., :-1, :].contiguous()
                shift_labels = batch["labels"][..., 1:].contiguous()
                
                # Calculate loss (ignore padding)
                loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
                loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), 
                              shift_labels.view(-1))
                
                # Update validation loss
                curr_loss = loss.item()
                val_loss += curr_loss
                
                # Update progress bar
                curr_avg_loss = val_loss / (i + 1)
                curr_ppl = math.exp(curr_avg_loss)
                pbar.set_postfix({
                    'loss': f'{curr_loss:.4f}',
                    'avg_loss': f'{curr_avg_loss:.4f}',
                    'ppl': f'{curr_ppl:.2f}'
                }, refresh=True)
        
        # Calculate average validation loss
        avg_val_loss = val_loss / num_batches
        
        # Calculate perplexity
        perplexity = math.exp(avg_val_loss)
        
        # Store metrics
        self.val_losses.append(avg_val_loss)
        self.val_perplexities.append(perplexity)
        
        return avg_val_loss, perplexity
    
    def save_model(self, path):
        """
        Save the model checkpoint to disk.
        
        Args:
            path: Path where the model should be saved
        """
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'global_step': self.global_step,
            'best_val_loss': self.best_val_loss,
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'learning_rates': self.learning_rates,
            'train_perplexities': self.train_perplexities,
            'val_perplexities': self.val_perplexities,
        }
        torch.save(checkpoint, path)
    
    def load_model(self, path):
        """
        Load a model checkpoint from disk.
        
        Args:
            path: Path to the model checkpoint
        """
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.global_step = checkpoint['global_step']
        self.best_val_loss = checkpoint['best_val_loss']
        self.train_losses = checkpoint['train_losses']
        self.val_losses = checkpoint['val_losses']
        self.learning_rates = checkpoint['learning_rates']
        self.train_perplexities = checkpoint['train_perplexities']
        self.val_perplexities = checkpoint['val_perplexities']
    
    def plot_training_curves(self, save_path=None):
        """
        Plot training curves for loss, perplexity, and learning rate.
        
        Args:
            save_path: Optional path to save the plot. If None, displays the plot.
        """
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 12))
        
        # Plot losses
        ax1.plot(self.train_losses, label='Train Loss')
        if self.val_losses:
            ax1.plot(self.val_losses, label='Val Loss')
        ax1.set_title('Training and Validation Loss')
        ax1.set_xlabel('Steps')
        ax1.set_ylabel('Loss')
        ax1.legend()
        
        # Plot perplexities
        ax2.plot(self.train_perplexities, label='Train Perplexity')
        if self.val_perplexities:
            ax2.plot(self.val_perplexities, label='Val Perplexity')
        ax2.set_title('Training and Validation Perplexity')
        ax2.set_xlabel('Steps')
        ax2.set_ylabel('Perplexity')
        ax2.legend()
        
        # Plot learning rates
        ax3.plot(self.learning_rates, label='Learning Rate')
        ax3.set_title('Learning Rate Schedule')
        ax3.set_xlabel('Steps')
        ax3.set_ylabel('Learning Rate')
        ax3.legend()
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
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
        "module_purpose": "Implements a specialized trainer for language modeling tasks with support for causal language modeling, evaluation, and generation",
        "key_classes": [
            {
                "name": "LanguageModelTrainer",
                "purpose": "Main trainer class for language model training with comprehensive training and evaluation capabilities",
                "key_methods": [
                    {
                        "name": "train",
                        "signature": "train(self, num_epochs, save_dir='models/language', model_name='language_model')",
                        "brief_description": "Main training loop with support for validation and checkpointing"
                    },
                    {
                        "name": "evaluate",
                        "signature": "evaluate(self)",
                        "brief_description": "Evaluates the model on validation data and returns loss and perplexity"
                    },
                    {
                        "name": "save_model",
                        "signature": "save_model(self, path)",
                        "brief_description": "Saves the model and training state to disk"
                    },
                    {
                        "name": "load_model",
                        "signature": "load_model(self, path)",
                        "brief_description": "Loads a saved model and training state from disk"
                    },
                    {
                        "name": "plot_training_curves",
                        "signature": "plot_training_curves(self, save_path=None)",
                        "brief_description": "Visualizes training metrics including loss, perplexity, and learning rate"
                    }
                ],
                "inheritance": "object",
                "dependencies": ["torch", "torch.nn", "torch.nn.functional", "numpy", "matplotlib", "tqdm"]
            }
        ],
        "external_dependencies": ["torch", "numpy", "matplotlib", "tqdm"],
        "complexity_score": 8,  # High complexity due to comprehensive training loop, evaluation, and visualization features
    }