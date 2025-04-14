# src/training/vision_transformer_trainer.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from typing import Dict, List, Optional, Tuple, Any, Callable
import time
import math
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import os

from ..models.vision.vision_transformer import VisionTransformer


class VisionTransformerTrainer:
    """
    Specialized trainer for Vision Transformer models.
    """

    def __init__(
        self,
        model: VisionTransformer,
        train_dataloader: DataLoader,
        val_dataloader: Optional[DataLoader] = None,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[Any] = None,
        criterion: Optional[nn.Module] = None,
        num_epochs: int = 100,
        early_stopping_patience: Optional[int] = 10,
        device: Optional[torch.device] = None,
        save_dir: str = "checkpoints",
        experiment_name: str = "vit_experiment",
        mixup_alpha: float = 0.0,
        cutmix_alpha: float = 0.0,
        label_smoothing: float = 0.1,
    ):
        """
        Initialize the Vision Transformer trainer.

        Args:
            model: Vision Transformer model
            train_dataloader: DataLoader for training data
            val_dataloader: Optional DataLoader for validation data
            optimizer: Optimizer (if None, will be created using model.configure_optimizers())
            scheduler: Learning rate scheduler
            criterion: Loss function (if None, CrossEntropyLoss with label smoothing will be used)
            num_epochs: Number of training epochs
            early_stopping_patience: Patience for early stopping
            device: Device to use for training
            save_dir: Directory to save checkpoints
            experiment_name: Name of the experiment
            mixup_alpha: Alpha parameter for mixup data augmentation (0 to disable)
            cutmix_alpha: Alpha parameter for cutmix data augmentation (0 to disable)
            label_smoothing: Label smoothing factor
        """
        self.model = model
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.num_epochs = num_epochs
        self.early_stopping_patience = early_stopping_patience
        self.save_dir = save_dir
        self.experiment_name = experiment_name
        self.mixup_alpha = mixup_alpha
        self.cutmix_alpha = cutmix_alpha

        # Set device (prefer Apple Silicon MPS if available)
        if device is None:
            if torch.backends.mps.is_available():
                self.device = torch.device("mps")
            else:
                self.device = torch.device(
                    "cuda" if torch.cuda.is_available() else "cpu"
                )
        else:
            self.device = device

        self.model = self.model.to(self.device)

        # Set optimizer
        self.optimizer = optimizer or model.configure_optimizers()

        # Set criterion (loss function)
        self.criterion = criterion or nn.CrossEntropyLoss(
            label_smoothing=label_smoothing
        )

        # Set scheduler
        self.scheduler = scheduler

        # Training state
        self.current_epoch = 0
        self.best_val_acc = 0.0
        self.patience_counter = 0

        # History for tracking metrics
        self.history = {
            "train_loss": [],
            "train_acc": [],
            "val_loss": [],
            "val_acc": [],
            "learning_rates": [],
        }

        # Create save directory
        os.makedirs(self.save_dir, exist_ok=True)

    def train_epoch(self):
        """
        Train the model for one epoch.

        Returns:
            Tuple of (average loss, accuracy)
        """
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0

        # Progress bar for training
        progress_bar = tqdm(
            enumerate(self.train_dataloader),
            total=len(self.train_dataloader),
            desc=f"Epoch {self.current_epoch+1}/{self.num_epochs}",
        )

        for batch_idx, batch in progress_bar:
            # Handle batch format - could be tuple (images, labels) or dict {"image": images, "label": labels}
            if isinstance(batch, dict):
                images = batch["image"].to(self.device)
                targets = batch["label"].to(self.device)
            else:
                # Assume batch is a tuple of (images, labels)
                images, targets = batch
                images = images.to(self.device)
                targets = targets.to(self.device)

            # Apply mixup or cutmix if enabled
            if self.mixup_alpha > 0 and torch.rand(1) < 0.5:
                images, targets_a, targets_b, lam = self._mixup_data(
                    images, targets, self.mixup_alpha
                )
                outputs = self.model(images)
                loss = self._mixup_criterion(outputs, targets_a, targets_b, lam)
            elif self.cutmix_alpha > 0 and torch.rand(1) < 0.5:
                images, targets_a, targets_b, lam = self._cutmix_data(
                    images, targets, self.cutmix_alpha
                )
                outputs = self.model(images)
                loss = self._mixup_criterion(outputs, targets_a, targets_b, lam)
            else:
                # Standard forward pass
                outputs = self.model(images)
                loss = self.criterion(outputs, targets)

            # Backward pass and optimize
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # Update metrics
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            # Update progress bar
            progress_bar.set_postfix(
                {
                    "loss": f"{loss.item():.4f}",
                    "acc": f"{100.*correct/total:.2f}%",
                    "lr": f"{self.optimizer.param_groups[0]['lr']:.6f}",
                }
            )

            # Update scheduler if batch-based
            if self.scheduler is not None and hasattr(self.scheduler, "step_batch"):
                self.scheduler.step_batch()

        # Calculate average metrics
        avg_loss = total_loss / len(self.train_dataloader)
        accuracy = 100.0 * correct / total

        # Update history
        self.history["train_loss"].append(avg_loss)
        self.history["train_acc"].append(accuracy)
        self.history["learning_rates"].append(self.optimizer.param_groups[0]["lr"])

        return avg_loss, accuracy

    def validate(self):
        """
        Validate the model.

        Returns:
            Tuple of (average loss, accuracy)
        """
        if self.val_dataloader is None:
            return 0.0, 0.0

        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for batch in tqdm(self.val_dataloader, desc="Validation"):
                # Handle batch format - could be tuple (images, labels) or dict {"image": images, "label": labels}
                if isinstance(batch, dict):
                    images = batch["image"].to(self.device)
                    targets = batch["label"].to(self.device)
                else:
                    # Assume batch is a tuple of (images, labels)
                    images, targets = batch
                    images = images.to(self.device)
                    targets = targets.to(self.device)

                # Forward pass
                outputs = self.model(images)
                loss = self.criterion(outputs, targets)

                # Update metrics
                total_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()

        # Calculate average metrics
        avg_loss = total_loss / len(self.val_dataloader)
        accuracy = 100.0 * correct / total

        # Update history
        self.history["val_loss"].append(avg_loss)
        self.history["val_acc"].append(accuracy)

        print(f"Validation - Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%")

        return avg_loss, accuracy

    def train(self) -> Dict[str, List[float]]:
        """
        Train the model for the specified number of epochs.

        Returns:
            Training history
        """
        print(f"Starting training on {self.device}...")

        for epoch in range(self.num_epochs):
            self.current_epoch = epoch

            # Train one epoch
            train_loss, train_acc = self.train_epoch()
            print(
                f"Epoch {epoch+1}/{self.num_epochs} - Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%"
            )

            # Validate
            val_loss, val_acc = self.validate()

            # Update scheduler if epoch-based
            if self.scheduler is not None and not hasattr(self.scheduler, "step_batch"):
                self.scheduler.step()

            # Early stopping and model saving
            if self.val_dataloader is not None:
                if val_acc > self.best_val_acc:
                    print(
                        f"Validation accuracy improved from {self.best_val_acc:.2f}% to {val_acc:.2f}%"
                    )
                    self.best_val_acc = val_acc
                    self.patience_counter = 0
                    self.save_checkpoint(f"{self.experiment_name}_best.pth")
                else:
                    self.patience_counter += 1
                    print(
                        f"Validation accuracy did not improve. Patience: {self.patience_counter}/{self.early_stopping_patience}"
                    )

                    if (
                        self.early_stopping_patience is not None
                        and self.patience_counter >= self.early_stopping_patience
                    ):
                        print(f"Early stopping triggered after {epoch+1} epochs")
                        break

            # Save checkpoint
            if (epoch + 1) % 10 == 0:
                self.save_checkpoint(f"{self.experiment_name}_epoch{epoch+1}.pth")

        # Save final model
        self.save_checkpoint(f"{self.experiment_name}_final.pth")

        print("Training completed!")
        return self.history

    def save_checkpoint(self, filename: str) -> None:
        """
        Save a checkpoint of the model and training state.

        Args:
            filename: Name of the checkpoint file
        """
        checkpoint_path = os.path.join(self.save_dir, filename)

        checkpoint = {
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "epoch": self.current_epoch,
            "best_val_acc": self.best_val_acc,
            "history": self.history,
        }

        if self.scheduler is not None:
            checkpoint["scheduler_state_dict"] = self.scheduler.state_dict()

        torch.save(checkpoint, checkpoint_path)
        print(f"Checkpoint saved to {checkpoint_path}")

    def load_checkpoint(self, filename: str) -> None:
        """
        Load a checkpoint of the model and training state.

        Args:
            filename: Name of the checkpoint file
        """
        checkpoint_path = os.path.join(self.save_dir, filename)

        if not os.path.exists(checkpoint_path):
            print(f"Checkpoint {checkpoint_path} does not exist")
            return

        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.current_epoch = checkpoint["epoch"]
        self.best_val_acc = checkpoint["best_val_acc"]
        self.history = checkpoint["history"]

        if "scheduler_state_dict" in checkpoint and self.scheduler is not None:
            self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

        print(f"Checkpoint loaded from {checkpoint_path}")

    def plot_training_history(self) -> None:
        """
        Plot training and validation metrics.
        """
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10))

        # Plot loss
        ax1.plot(self.history["train_loss"], label="Train Loss")
        if self.val_dataloader is not None:
            ax1.plot(self.history["val_loss"], label="Val Loss")
        ax1.set_xlabel("Epoch")
        ax1.set_ylabel("Loss")
        ax1.set_title("Training and Validation Loss")
        ax1.legend()
        ax1.grid(True)

        # Plot accuracy
        ax2.plot(self.history["train_acc"], label="Train Acc")
        if self.val_dataloader is not None:
            ax2.plot(self.history["val_acc"], label="Val Acc")
        ax2.set_xlabel("Epoch")
        ax2.set_ylabel("Accuracy (%)")
        ax2.set_title("Training and Validation Accuracy")
        ax2.legend()
        ax2.grid(True)

        plt.tight_layout()
        plt.show()

    def _mixup_data(
        self, x: torch.Tensor, y: torch.Tensor, alpha: float
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
        """
        Mixup data augmentation.

        Args:
            x: Input data
            y: Targets
            alpha: Mixup alpha value

        Returns:
            Tuple of (mixed inputs, targets_a, targets_b, lambda)
        """
        if alpha > 0:
            lam = np.random.beta(alpha, alpha)
        else:
            lam = 1

        batch_size = x.size(0)
        index = torch.randperm(batch_size).to(self.device)

        mixed_x = lam * x + (1 - lam) * x[index, :]
        y_a, y_b = y, y[index]

        return mixed_x, y_a, y_b, lam

    def _cutmix_data(
        self, x: torch.Tensor, y: torch.Tensor, alpha: float
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
        """
        CutMix data augmentation.

        Args:
            x: Input data
            y: Targets
            alpha: CutMix alpha value

        Returns:
            Tuple of (mixed inputs, targets_a, targets_b, lambda)
        """
        if alpha > 0:
            lam = np.random.beta(alpha, alpha)
        else:
            lam = 1

        batch_size = x.size(0)
        index = torch.randperm(batch_size).to(self.device)

        y_a, y_b = y, y[index]

        # Get dimensions
        _, c, h, w = x.shape

        # Get random coordinates
        cut_rat = np.sqrt(1.0 - lam)
        cut_w = int(w * cut_rat)
        cut_h = int(h * cut_rat)

        cx = np.random.randint(w)
        cy = np.random.randint(h)

        # Bounds calculation with proper clipping
        bbx1 = np.clip(cx - cut_w // 2, 0, w)
        bby1 = np.clip(cy - cut_h // 2, 0, h)
        bbx2 = np.clip(cx + cut_w // 2, 0, w)
        bby2 = np.clip(cy + cut_h // 2, 0, h)

        # Create mixed images
        x_mixed = x.clone()
        x_mixed[:, :, bby1:bby2, bbx1:bbx2] = x[index, :, bby1:bby2, bbx1:bbx2]

        # Adjust lambda
        lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (w * h))

        return x_mixed, y_a, y_b, lam

    def _mixup_criterion(
        self, pred: torch.Tensor, y_a: torch.Tensor, y_b: torch.Tensor, lam: float
    ) -> torch.Tensor:
        """
        Mixup criterion.

        Args:
            pred: Model predictions
            y_a: First targets
            y_b: Second targets
            lam: Mixup lambda

        Returns:
            Mixed loss
        """
        return lam * self.criterion(pred, y_a) + (1 - lam) * self.criterion(pred, y_b)
