#!/usr/bin/env python
"""
VICReg-based multimodal training configuration for the Multimodal Insight Engine.

This script sets up and runs a training session using the VICReg model with
appropriate configuration for variance-invariance-covariance regularization.
"""

import argparse
import torch
import os
import sys
import logging
from pathlib import Path
import torch.nn as nn

# Add the repository root to the system path
sys.path.append(str(Path(__file__).parent.parent))

from src.utils.argument_configs import get_multimodal_training_args
from src.training.multimodal_trainer import MultimodalTrainer
from src.models.multimodal.vicreg_multimodal_model import VICRegMultimodalModel
from src.training.loss.vicreg_loss import VICRegLoss
from src.training.loss.loss_factory import create_loss_function
from src.data.tokenization.bert_tokenizer_adapter import (
    BertTokenizerAdapter,
)  # Use exactly matching tokenizer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def main():
    # Get multimodal training arguments
    parser = get_multimodal_training_args()

    # No need to re-define VICReg arguments here since they are already in argument_configs.py
    # We'll just use the ones from the base argument parser
    parser.add_argument(
        "--use_orthogonal_init",
        action="store_true",
        default=True,
        help="Use orthogonal initialization with custom gains for vision (1.6) and text (1.4)",
    )
    # Note: Batch size is already defined in get_multimodal_training_args()
    # We'll rely on the command line to override it instead of defining it twice
    # Override default loss_type to vicreg
    # The loss_type argument is already defined in get_multimodal_training_args()
    # We'll set the default in args after parsing

    # Parse args
    args = parser.parse_args()

    # Make sure we're using VICReg loss type
    # args.loss_type = "vicreg"
    # logger.info("Setting loss_type to 'vicreg'")

    # Map vicreg_* parameters to sim_weight, var_weight, cov_weight for loss_factory
    # For backwards compatibility, handle both naming conventions
    if hasattr(args, "vicreg_sim_weight"):
        args.sim_weight = args.vicreg_sim_weight
    if hasattr(args, "vicreg_var_weight"):
        args.var_weight = args.vicreg_var_weight
    if hasattr(args, "vicreg_cov_weight"):
        args.cov_weight = args.vicreg_cov_weight

    # Log the VICReg configuration that will be used
    logger.info(
        f"Using VICReg with: sim_weight={args.sim_weight}, var_weight={args.var_weight}, cov_weight={args.cov_weight}"
    )
    logger.info(
        f"VICReg curriculum: enabled={args.use_curriculum}, warmup_epochs={args.vicreg_warmup_epochs}"
    )

    # Set random seed for reproducibility
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)

    # Select device
    if args.device == "cuda" and torch.cuda.is_available():
        device = torch.device("cuda")
    elif (
        args.device == "mps"
        and hasattr(torch.backends, "mps")
        and torch.backends.mps.is_available()
    ):
        device = torch.device("mps")
        # Set environment variable for MPS fallback to CPU for unsupported operations
        os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
        logger.info("Set MPS fallback to CPU for unsupported operations")
    else:
        device = torch.device("cpu")

    logger.info(f"Using device: {device}")

    # Load dataset
    # Use the existing data loading utilities from multimodal_data_utils
    from src.data.multimodal_data_utils import create_data_loaders
    from src.models.vision.image_preprocessing import ImagePreprocessor
    from src.data.tokenization.simple_tokenizer import SimpleTokenizer

    # Setup image preprocessor and tokenizer
    image_preprocessor = ImagePreprocessor(image_size=224)
    tokenizer = BertTokenizerAdapter(
        pretrained_model_name="bert-base-uncased", max_length=77
    )

    # Create dataloaders
    logger.info(f"Loading dataset: {args.dataset}")

    # Use the create_data_loaders utility to create all the datasets and dataloaders
    train_dataloader, val_dataloader, test_dataloader = create_data_loaders(
        args, image_preprocessor=image_preprocessor, tokenizer=tokenizer
    )

    # Create model
    logger.info("Creating VICReg multimodal model")

    # Initialize base vision and text models
    # We'll create the models directly since the registry imported functions don't seem to exist
    from src.models.pretrained.vision_transformer import VisionTransformerWrapper
    from src.models.pretrained.huggingface_wrapper import HuggingFaceTextModelWrapper

    # Create vision model
    vision_model = VisionTransformerWrapper(model_name=args.vision_model)
    # Note: The VisionTransformerWrapper will load a pretrained model by default

    # Create text model - directly use the full HuggingFace model name without validation
    text_model = HuggingFaceTextModelWrapper(model_name=args.text_model)

    # Create VICReg model with CPU initialization and MPS target
    model = VICRegMultimodalModel(
        vision_model=vision_model,
        text_model=text_model,
        projection_dim=args.fusion_dim,
        device=device,  # Pass target device for final model placement
    )

    # Create optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )

    # Set up learning rate scheduler with warmup
    total_steps = len(train_dataloader) * args.num_epochs
    warmup_steps = args.warmup_steps

    from torch.optim.lr_scheduler import LambdaLR

    def lr_lambda(current_step):
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        return max(
            0.0,
            float(total_steps - current_step)
            / float(max(1, total_steps - warmup_steps)),
        )

    scheduler = LambdaLR(optimizer, lr_lambda)

    # Create loss function
    # Use the size of the dataloader as an approximation of dataset size
    dataset_size = len(train_dataloader) * args.batch_size
    loss_fn = create_loss_function(args, dataset_size=dataset_size)
    logger.info(f"Using loss function: {loss_fn.__class__.__name__}")

    # Print detailed loss function info for debugging
    print(f"LOSS FUNCTION DEBUG: type={type(loss_fn).__name__}")

    # For HybridPretrainVICRegLoss, check its contrastive loss
    if hasattr(loss_fn, "contrastive_loss"):
        contrastive_loss = loss_fn.contrastive_loss
        print(
            f"HYBRID LOSS DEBUG: contrastive_loss.add_projection={contrastive_loss.add_projection}"
        )

        # If projection is enabled, check projection dimensions
        if contrastive_loss.add_projection:
            vision_proj = contrastive_loss.vision_projection
            if hasattr(vision_proj, "0") and isinstance(vision_proj[0], nn.Linear):
                in_dim = vision_proj[0].in_features

                # Get output dimension from the last linear layer
                out_dim = None
                for m in vision_proj:
                    if isinstance(m, nn.Linear):
                        out_dim = m.out_features

                print(
                    f"PROJECTION DIM DEBUG: vision projection dimensions {in_dim} -> {out_dim}"
                )

    # Set up trainer
    trainer = MultimodalTrainer(
        model=model,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        optimizer=optimizer,
        scheduler=scheduler,
        loss_fn=loss_fn,
        num_epochs=args.num_epochs,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        warmup_steps=args.warmup_steps,
        checkpoint_dir=os.path.join(args.output_dir, args.checkpoint_dir),
        log_dir=os.path.join(args.output_dir, args.log_dir),
        device=device,
        mixed_precision=args.use_mixed_precision,
        accumulation_steps=1,
        evaluation_steps=100,
        log_steps=10,
        early_stopping_patience=5,
        clip_grad_norm=args.clip_grad_norm if args.clip_grad_norm is not None else 1.0,
        balance_modality_gradients=True,
        args=args,  # Pass the full args for advanced configuration
    )

    # Run training
    if args.use_multistage_training:
        logger.info("Using multi-stage training")
        history = trainer.train_multistage()
    else:
        logger.info("Using standard training")
        history = trainer.train()

    logger.info("Training completed!")


if __name__ == "__main__":
    main()
