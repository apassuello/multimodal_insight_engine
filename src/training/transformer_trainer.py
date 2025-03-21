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
        """
        self.model = model
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.pad_idx = pad_idx
        self.warmup_steps = warmup_steps
        self.clip_grad = clip_grad
        self.early_stopping_patience = early_stopping_patience
        
        # Set device
        if device is None:
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
        Create a learning rate scheduler with warmup and decay.
        
        Args:
            optimizer: Optimizer to schedule
            
        Returns:
            Learning rate scheduler
        """
        # Define learning rate function
        def lr_lambda(step):
            # Linear warmup followed by inverse square root decay
            if step == 0:
                step = 1
            return min(
                step ** (-0.5),
                step * self.warmup_steps ** (-1.5)
            ) * self.warmup_steps ** 0.5
        
        return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    
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
            logits = self.model(src, tgt_input, src_mask=src_mask, tgt_mask=tgt_mask)
            
            # Calculate loss
            loss = self.criterion(logits, tgt_output)
            
            # Backward pass
            self.optimizer.zero_grad()
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
        Train the model for multiple epochs.
        
        Args:
            epochs: Number of epochs to train
            save_path: Path to save model checkpoints
            
        Returns:
            Training history
        """
        print(f"Starting training for {epochs} epochs on {self.device}")
        print(f"Model has {sum(p.numel() for p in self.model.parameters() if p.requires_grad):,} trainable parameters")
        save_idx = 0
        for epoch in range(epochs):
            self.current_epoch = epoch
            
            # Train for one epoch
            train_loss = self.train_epoch()
            
            # Validate
            val_loss = self.validate()
            
            # Early stopping check
            if val_loss is not None and self.early_stopping_patience is not None:
                if val_loss < self.best_val_loss:
                    self.best_val_loss = val_loss
                    self.patience_counter = 0
                    
                    # Save best model
                    # if save_path is not None:
                    #     self.save_checkpoint(f"{save_path}_best.pt")
                else:
                    self.patience_counter += 1
                    if self.patience_counter >= self.early_stopping_patience:
                        print(f"Early stopping triggered after {epoch+1} epochs")
                        break
            
            # Save checkpoint
            if save_path is not None and save_idx % 10 == 0:
                self.save_checkpoint(f"{save_path}_epoch{epoch+1}.pt")
                save_idx += 1
        print("Training completed")
        return self.history
    
    def save_checkpoint(self, path: str):
        """
        Save a model checkpoint.
        
        Args:
            path: Path to save the checkpoint
        """
        checkpoint = {
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "epoch": self.current_epoch,
            "global_step": self.global_step,
            "best_val_loss": self.best_val_loss,
            "history": self.history,
        }
        
        torch.save(checkpoint, path)
        print(f"Checkpoint saved to {path}")
    
    def load_checkpoint(self, path: str):
        """
        Load a model checkpoint.
        
        Args:
            path: Path to the checkpoint
        """
        checkpoint = torch.load(path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        self.current_epoch = checkpoint["epoch"]
        self.global_step = checkpoint["global_step"]
        self.best_val_loss = checkpoint["best_val_loss"]
        self.history = checkpoint["history"]
        
        print(f"Checkpoint loaded from {path}")
    
    def plot_learning_rate(self):
        """
        Plot the learning rate schedule.
        
        Returns:
            Matplotlib figure
        """
        plt.figure(figsize=(10, 5))
        plt.plot(self.history["learning_rates"])
        plt.xlabel("Step")
        plt.ylabel("Learning Rate")
        plt.title("Learning Rate Schedule")
        plt.grid(True)
        return plt.gcf()
    
    def plot_training_history(self):
        """
        Plot the training history.
        
        Returns:
            Matplotlib figure
        """
        fig, axs = plt.subplots(1, 2, figsize=(16, 6))
        
        # Plot loss
        axs[0].plot(self.history["train_loss"], label="Train")
        if self.val_dataloader is not None:
            axs[0].plot(self.history["val_loss"], label="Validation")
        axs[0].set_xlabel("Epoch")
        axs[0].set_ylabel("Loss")
        axs[0].set_title("Training and Validation Loss")
        axs[0].legend()
        axs[0].grid(True)
        
        # Plot perplexity
        axs[1].plot(self.history["train_ppl"], label="Train")
        if self.val_dataloader is not None:
            axs[1].plot(self.history["val_ppl"], label="Validation")
        axs[1].set_xlabel("Epoch")
        axs[1].set_ylabel("Perplexity")
        axs[1].set_title("Training and Validation Perplexity")
        axs[1].legend()
        axs[1].grid(True)
        
        plt.tight_layout()
        return fig