# demos/vision_transformer_demo.py
"""
Vision Transformer (ViT) Image Classification Demo

This script demonstrates how to train and evaluate a Vision Transformer
on the CIFAR-10 dataset for image classification.
"""

import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.adamw import AdamW  # Fixed import path for AdamW
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import argparse

# Import our Vision Transformer implementation
from src.models.vision.vision_transformer import VisionTransformer
from src.training.vision_transformer_trainer import VisionTransformerTrainer
from src.data.dataset_wrapper import load_cifar10_dict


class EnhancedVisionTransformerTrainer(VisionTransformerTrainer):
    """
    Enhanced trainer for Vision Transformer models that adds support for:
    - Mixed precision training
    - Gradient scaling
    - Gradient clipping
    - Learning rate warmup
    """

    def __init__(
        self,
        model,
        train_dataloader,
        val_dataloader=None,
        optimizer=None,
        scheduler=None,
        criterion=None,
        num_epochs=100,
        early_stopping_patience=10,
        device=None,
        save_dir="checkpoints",
        experiment_name="vit_experiment",
        mixup_alpha=0.0,
        cutmix_alpha=0.0,
        label_smoothing=0.1,
        use_gradient_scaling=False,
        use_mixed_precision=False,
        clip_grad=None,
        warmup_steps=0,
    ):
        """
        Initialize the enhanced Vision Transformer trainer.

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
            use_gradient_scaling: Whether to use gradient scaling for mixed precision
            use_mixed_precision: Whether to use mixed precision training
            clip_grad: Gradient clipping threshold (None to disable)
            warmup_steps: Number of warmup steps for learning rate
        """
        # Call parent constructor
        super().__init__(
            model=model,
            train_dataloader=train_dataloader,
            val_dataloader=val_dataloader,
            optimizer=optimizer,
            scheduler=scheduler,
            criterion=criterion,
            num_epochs=num_epochs,
            early_stopping_patience=early_stopping_patience,
            device=device,
            save_dir=save_dir,
            experiment_name=experiment_name,
            mixup_alpha=mixup_alpha,
            cutmix_alpha=cutmix_alpha,
            label_smoothing=label_smoothing,
        )

        # Store additional parameters
        self.use_gradient_scaling = use_gradient_scaling
        self.use_mixed_precision = use_mixed_precision
        self.clip_grad = clip_grad
        self.warmup_steps = warmup_steps
        self.current_step = 0

        # Set up gradient scaler for mixed precision training
        self.scaler = None
        # Only create scaler if gradient scaling is enabled and using CUDA device
        if (
            self.use_gradient_scaling
            and hasattr(torch.cuda, "amp")
            and self.device.type == "cuda"
        ):
            self.scaler = torch.cuda.amp.GradScaler()

        # Base learning rate for warmup
        self.base_lr = self.optimizer.param_groups[0]["lr"]

        # Override scheduler if warmup is used
        if self.warmup_steps > 0 and self.scheduler is None:
            print(f"Using linear warmup for {warmup_steps} steps")

    def get_warmup_lr(self):
        """Calculate learning rate during warmup period"""
        if self.current_step >= self.warmup_steps or self.warmup_steps <= 0:
            return self.base_lr

        # Linear warmup
        return self.base_lr * (self.current_step / self.warmup_steps)

    def train_epoch(self):
        """
        Train the model for one epoch with support for mixed precision,
        gradient clipping and learning rate warmup.

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

            # Apply warmup learning rate if in warmup phase
            if self.warmup_steps > 0 and self.current_step < self.warmup_steps:
                for param_group in self.optimizer.param_groups:
                    param_group["lr"] = self.get_warmup_lr()

            # Apply mixup or cutmix if enabled
            apply_mixup = self.mixup_alpha > 0 and torch.rand(1) < 0.5
            apply_cutmix = self.cutmix_alpha > 0 and torch.rand(1) < 0.5

            # Zero gradients
            self.optimizer.zero_grad()

            # Mixed precision training path (only supported with CUDA)
            if (
                self.use_mixed_precision
                and hasattr(torch.cuda, "amp")
                and self.device.type == "cuda"
            ):
                with torch.cuda.amp.autocast():
                    if apply_mixup:
                        images, targets_a, targets_b, lam = self._mixup_data(
                            images, targets, self.mixup_alpha
                        )
                        outputs = self.model(images)
                        loss = self._mixup_criterion(outputs, targets_a, targets_b, lam)
                    elif apply_cutmix:
                        images, targets_a, targets_b, lam = self._cutmix_data(
                            images, targets, self.cutmix_alpha
                        )
                        outputs = self.model(images)
                        loss = self._mixup_criterion(outputs, targets_a, targets_b, lam)
                    else:
                        # Standard forward pass
                        outputs = self.model(images)
                        loss = self.criterion(outputs, targets)

                # Backward with scaling
                if self.use_gradient_scaling and self.scaler is not None:
                    self.scaler.scale(loss).backward()

                    # Clip gradients if specified
                    if self.clip_grad is not None:
                        self.scaler.unscale_(self.optimizer)
                        torch.nn.utils.clip_grad_norm_(
                            self.model.parameters(), self.clip_grad
                        )

                    # Step with scaler
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    loss.backward()

                    # Clip gradients if specified
                    if self.clip_grad is not None:
                        torch.nn.utils.clip_grad_norm_(
                            self.model.parameters(), self.clip_grad
                        )

                    self.optimizer.step()
            else:
                # Standard precision training path
                if apply_mixup:
                    images, targets_a, targets_b, lam = self._mixup_data(
                        images, targets, self.mixup_alpha
                    )
                    outputs = self.model(images)
                    loss = self._mixup_criterion(outputs, targets_a, targets_b, lam)
                elif apply_cutmix:
                    images, targets_a, targets_b, lam = self._cutmix_data(
                        images, targets, self.cutmix_alpha
                    )
                    outputs = self.model(images)
                    loss = self._mixup_criterion(outputs, targets_a, targets_b, lam)
                else:
                    # Standard forward pass
                    outputs = self.model(images)
                    loss = self.criterion(outputs, targets)

                # Backward and optimize
                loss.backward()

                # Clip gradients if specified
                if self.clip_grad is not None:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.clip_grad
                    )

                self.optimizer.step()

            # Update metrics
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            # Update step counter for warmup
            self.current_step += 1

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


def get_best_device(force_cpu=False):
    """
    Determine the best available device for training.
    For Mac, will try to use MPS first, but if issues occur, will fall back to CPU.

    Args:
        force_cpu: If True, always use CPU regardless of GPU availability

    Returns:
        torch.device: Best available device
    """
    if force_cpu:
        print("Forcing CPU usage as requested.")
        return torch.device("cpu")

    if torch.cuda.is_available():
        return torch.device("cuda")

    if torch.backends.mps.is_available():
        try:
            # Try a small tensor operation on MPS to see if it works
            test_tensor = torch.randn(10, 10).to("mps")
            test_tensor @ test_tensor.T
            return torch.device("mps")
        except:
            print(
                "WARNING: MPS (Apple Silicon GPU) is available but encountered an error during test."
            )
            print("Falling back to CPU. This will be much slower.")
            return torch.device("cpu")

    return torch.device("cpu")


def load_cifar10(batch_size=128, num_workers=4, use_dict_format=False):
    """
    Load CIFAR-10 dataset.

    Args:
        batch_size: Batch size for dataloaders
        num_workers: Number of worker processes for data loading
        use_dict_format: Whether to wrap the datasets to return dictionary format

    Returns:
        train_loader: Training data loader
        val_loader: Validation data loader
        classes: List of class names
    """
    # Use dictionary format loader if requested
    if use_dict_format:
        from src.data.dataset_wrapper import load_cifar10_dict

        return load_cifar10_dict(batch_size, num_workers)

    # Define transformations for training data (with augmentation)
    transform_train = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
        ]
    )

    # Define transformations for validation data (no augmentation)
    transform_val = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
        ]
    )

    # Load training dataset
    train_dataset = torchvision.datasets.CIFAR10(
        root="./data", train=True, download=True, transform=transform_train
    )

    # Load validation dataset
    val_dataset = torchvision.datasets.CIFAR10(
        root="./data", train=False, download=True, transform=transform_val
    )

    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    # CIFAR-10 classes
    classes = (
        "plane",
        "car",
        "bird",
        "cat",
        "deer",
        "dog",
        "frog",
        "horse",
        "ship",
        "truck",
    )

    return train_loader, val_loader, classes


def visualize_batch(dataloader, classes, num_images=16):
    """
    Visualize a batch of images from the dataset.

    Args:
        dataloader: DataLoader containing the images
        classes: List of class names
        num_images: Number of images to display
    """
    # Get a batch of training images
    batch = next(iter(dataloader))

    # Handle both tuple and dictionary formats
    if isinstance(batch, dict):
        images, labels = batch["image"], batch["label"]
    else:
        images, labels = batch

    # Convert images from tensor to numpy for display
    images = images.numpy()

    # Normalize images for display
    images = np.transpose(images, (0, 2, 3, 1))
    images = np.clip(
        (
            images * np.array([0.2470, 0.2435, 0.2616])
            + np.array([0.4914, 0.4822, 0.4465])
        )
        * 255.0,
        0,
        255,
    ).astype(np.uint8)

    # Plot the images
    fig = plt.figure(figsize=(10, 10))
    for idx in range(min(num_images, len(images))):
        ax = fig.add_subplot(4, 4, idx + 1, xticks=[], yticks=[])
        ax.imshow(images[idx])
        ax.set_title(f"{classes[labels[idx]]}")
    plt.tight_layout()
    plt.show()


def create_model(args):
    """
    Create a Vision Transformer model based on command-line arguments.

    Args:
        args: Command-line arguments

    Returns:
        Configured Vision Transformer model
    """
    # Create Vision Transformer model
    model = VisionTransformer(
        image_size=32,  # CIFAR-10 images are 32x32
        patch_size=args.patch_size,
        in_channels=3,  # RGB images
        num_classes=10,  # CIFAR-10 has 10 classes
        embed_dim=args.embed_dim,
        depth=args.depth,
        num_heads=args.num_heads,
        mlp_ratio=args.mlp_ratio,
        dropout=args.dropout,
        attention_dropout=args.attention_dropout,
        pool=args.pool,
        positional_encoding=args.positional_encoding,
    )

    return model


def main(args):
    """
    Main function for the demo.

    Args:
        args: Command-line arguments
    """
    print("Loading CIFAR-10 dataset...")
    train_loader, val_loader, classes = load_cifar10(
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        use_dict_format=args.dict_format,
    )

    # Visualize a batch of training images if requested
    if args.visualize:
        visualize_batch(train_loader, classes)

    # Create model
    print("Creating Vision Transformer model...")
    model = create_model(args)
    print(
        f"Model parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}"
    )

    # Configure learning rate scheduler
    optimizer = AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
        betas=(0.9, 0.999),
    )

    if args.scheduler == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=args.epochs, eta_min=0
        )
    elif args.scheduler == "reduce":
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="max", factor=0.5, patience=5
        )
    else:
        scheduler = None

    # Create trainer
    print("Creating trainer...")
    trainer = EnhancedVisionTransformerTrainer(
        model=model,
        train_dataloader=train_loader,
        val_dataloader=val_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        criterion=nn.CrossEntropyLoss(label_smoothing=args.label_smoothing),
        num_epochs=args.epochs,
        early_stopping_patience=args.patience,
        device=get_best_device(
            force_cpu=args.force_cpu
        ),  # Use the best available device
        save_dir=args.save_dir,
        experiment_name=args.name,
        mixup_alpha=args.mixup_alpha,
        cutmix_alpha=args.cutmix_alpha,
        label_smoothing=args.label_smoothing,
        use_gradient_scaling=args.use_gradient_scaling,
        use_mixed_precision=args.use_mixed_precision,
        clip_grad=args.clip_grad,
        warmup_steps=args.warmup_steps,
    )

    # Train the model
    print("Starting training...")
    start_time = time.time()
    history = trainer.train()
    training_time = time.time() - start_time
    print(f"Training completed in {training_time:.2f} seconds")

    # Plot training history
    trainer.plot_training_history()

    # Save the model
    final_checkpoint = os.path.join(args.save_dir, f"{args.name}_final.pth")
    trainer.save_checkpoint(final_checkpoint)
    print(f"Model saved to {final_checkpoint}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Vision Transformer Demo for CIFAR-10")

    # Model configuration
    parser.add_argument(
        "--patch-size", type=int, default=4, help="Patch size (default: 4)"
    )
    parser.add_argument(
        "--embed-dim", type=int, default=384, help="Embedding dimension (default: 384)"
    )
    parser.add_argument(
        "--depth",
        type=int,
        default=12,
        help="Number of transformer layers (default: 12)",
    )
    parser.add_argument(
        "--num-heads",
        type=int,
        default=6,
        help="Number of attention heads (default: 6)",
    )
    parser.add_argument(
        "--mlp-ratio", type=float, default=4.0, help="MLP ratio (default: 4.0)"
    )
    parser.add_argument(
        "--dropout", type=float, default=0.1, help="Dropout rate (default: 0.1)"
    )
    parser.add_argument(
        "--attention-dropout",
        type=float,
        default=0.0,
        help="Attention dropout rate (default: 0.0)",
    )
    parser.add_argument(
        "--pool",
        type=str,
        default="cls",
        choices=["cls", "mean"],
        help="Pooling type (default: cls)",
    )
    parser.add_argument(
        "--positional-encoding",
        type=str,
        default="learned",
        choices=["learned", "sinusoidal"],
        help="Positional encoding type (default: learned)",
    )

    # Training configuration
    parser.add_argument(
        "--batch-size", type=int, default=64, help="Batch size (default: 64)"
    )
    parser.add_argument(
        "--epochs", type=int, default=10, help="Number of epochs (default: 10)"
    )
    parser.add_argument(
        "--lr", type=float, default=1e-3, help="Learning rate (default: 1e-3)"
    )
    parser.add_argument(
        "--weight-decay", type=float, default=0.05, help="Weight decay (default: 0.05)"
    )
    parser.add_argument(
        "--patience", type=int, default=10, help="Early stopping patience (default: 10)"
    )
    parser.add_argument(
        "--warmup-steps",
        type=int,
        default=0,
        help="Number of warmup steps for learning rate scheduler (default: 0)",
    )
    parser.add_argument(
        "--clip-grad",
        type=float,
        default=None,
        help="Gradient clipping value (default: None to disable)",
    )
    parser.add_argument(
        "--use-gradient-scaling",
        action="store_true",
        help="Whether to use gradient scaling for mixed precision training",
    )
    parser.add_argument(
        "--use-mixed-precision",
        action="store_true",
        help="Whether to use mixed precision training",
    )
    parser.add_argument(
        "--mixup-alpha",
        type=float,
        default=0.2,
        help="Mixup alpha (default: 0.2, 0 to disable)",
    )
    parser.add_argument(
        "--cutmix-alpha",
        type=float,
        default=0.0,
        help="CutMix alpha (default: 0.0, 0 to disable)",
    )
    parser.add_argument(
        "--label-smoothing",
        type=float,
        default=0.1,
        help="Label smoothing (default: 0.1)",
    )
    parser.add_argument(
        "--scheduler",
        type=str,
        default="cosine",
        choices=["cosine", "reduce", "none"],
        help="Learning rate scheduler (default: cosine)",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=4,
        help="Number of data loading workers (default: 4)",
    )

    # Other configuration
    parser.add_argument(
        "--name",
        type=str,
        default="vit_cifar10",
        help="Experiment name (default: vit_cifar10)",
    )
    parser.add_argument(
        "--save-dir",
        type=str,
        default="checkpoints",
        help="Directory to save checkpoints (default: checkpoints)",
    )
    parser.add_argument(
        "--visualize", action="store_true", help="Visualize a batch of training images"
    )
    parser.add_argument(
        "--dict-format",
        action="store_true",
        help="Use dictionary format for datasets (default: False)",
    )
    parser.add_argument(
        "--force-cpu",
        action="store_true",
        help="Force CPU usage even if GPU is available (default: False)",
    )

    args = parser.parse_args()

    main(args)
