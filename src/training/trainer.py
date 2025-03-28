import torch
import torch.nn as nn
from typing import Dict, List, Callable, Optional, Union
from tqdm import tqdm
import time


def train_model(
    model: nn.Module,
    train_dataloader: torch.utils.data.DataLoader,
    val_dataloader: Optional[torch.utils.data.DataLoader] = None,
    epochs: int = 10,
    learning_rate: float = 0.001,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
    early_stopping_patience: Optional[int] = None,
    device: Optional[Union[str, torch.device]] = None,
    callbacks: List[Callable] = None,
) -> Dict[str, List[float]]:
    """
    A generic training loop for PyTorch models.

    Args:
        model: The model to train
        train_dataloader: DataLoader for training data
        val_dataloader: Optional DataLoader for validation data
        epochs: Number of epochs to train for
        learning_rate: Learning rate for the optimizer
        optimizer: Optional pre-configured optimizer (will create one if not provided)
        scheduler: Optional learning rate scheduler
        early_stopping_patience: Number of epochs to wait for improvement before stopping
        device: Device to train on (will use CUDA if available if not specified)
        callbacks: List of callback functions to call after each epoch

    Returns:    
        Dictionary containing training history (losses, metrics, etc.)
    """
    # Set device
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Configure optimizer if not provided
    if optimizer is None:
        if hasattr(model, "configure_optimizers"):
            optimizer = model.configure_optimizers(lr=learning_rate)
        else:
            optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # Initialize history
    history = {
        "train_loss": [],
        "train_accuracy": [],
        "val_loss": [],
        "val_accuracy": [],
    }

    # Initialize early stopping variables
    best_val_loss = float("inf")
    patience_counter = 0

    # Training loop
    for epoch in range(epochs):
        print(f"Epoch {epoch+1}/{epochs}")
        start_time = time.time()

        # Training phase
        model.train()
        train_losses = []
        train_accuracies = []

        for batch in tqdm(train_dataloader, desc="Training"):
            # Move batch to device
            batch = {
                k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()
            }

            # Forward pass
            if hasattr(model, "training_step"):
                step_output = model.training_step(batch)
                loss = step_output["loss"]
                if "accuracy" in step_output:
                    train_accuracies.append(step_output["accuracy"].item())
            else:
                outputs = model(batch["inputs"])
                targets = batch["targets"]
                loss = torch.nn.functional.cross_entropy(outputs, targets)
                predictions = torch.argmax(outputs, dim=1)
                accuracy = (predictions == targets).float().mean().item()
                train_accuracies.append(accuracy)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_losses.append(loss.item())

        # Update scheduler if provided
        if scheduler is not None:
            scheduler.step()

        # Calculate average training metrics
        avg_train_loss = sum(train_losses) / len(train_losses)
        avg_train_accuracy = (
            sum(train_accuracies) / len(train_accuracies) if train_accuracies else 0
        )
        history["train_loss"].append(avg_train_loss)
        history["train_accuracy"].append(avg_train_accuracy)

        # Validation phase
        if val_dataloader is not None:
            model.eval()
            val_losses = []
            val_accuracies = []

            with torch.no_grad():
                for batch in tqdm(val_dataloader, desc="Validation"):
                    # Move batch to device
                    batch = {
                        k: v.to(device) if torch.is_tensor(v) else v
                        for k, v in batch.items()
                    }

                    # Forward pass
                    if hasattr(model, "validation_step"):
                        step_output = model.validation_step(batch)
                        loss = step_output["loss"]
                        if "accuracy" in step_output:
                            val_accuracies.append(step_output["accuracy"].item())
                    else:
                        outputs = model(batch["inputs"])
                        targets = batch["targets"]
                        loss = torch.nn.functional.cross_entropy(outputs, targets)
                        predictions = torch.argmax(outputs, dim=1)
                        accuracy = (predictions == targets).float().mean().item()
                        val_accuracies.append(accuracy)

                    val_losses.append(loss.item())

            # Calculate average validation metrics
            avg_val_loss = sum(val_losses) / len(val_losses)
            avg_val_accuracy = (
                sum(val_accuracies) / len(val_accuracies) if val_accuracies else 0
            )
            history["val_loss"].append(avg_val_loss)
            history["val_accuracy"].append(avg_val_accuracy)

            # Early stopping check
            if early_stopping_patience is not None:
                if avg_val_loss < best_val_loss:
                    best_val_loss = avg_val_loss
                    patience_counter = 0
                else:
                    patience_counter += 1
                    if patience_counter >= early_stopping_patience:
                        print(f"Early stopping triggered after {epoch+1} epochs")
                        break

        # Print epoch summary
        epoch_time = time.time() - start_time
        print(
            f"Epoch {epoch+1}/{epochs} - {epoch_time:.2f}s - loss: {avg_train_loss:.4f}",
            end="",
        )
        if train_accuracies:
            print(f" - accuracy: {avg_train_accuracy:.4f}", end="")
        if val_dataloader is not None:
            print(f" - val_loss: {avg_val_loss:.4f}", end="")
            if val_accuracies:
                print(f" - val_accuracy: {avg_val_accuracy:.4f}", end="")
        print()

        # Call callbacks if provided
        if callbacks:
            for callback in callbacks:
                callback(model, epoch, history)

    return history
