#!/usr/bin/env python
# demos/flickr30k_multistage_training_demo.py

"""
Demo script for multistage training on Flickr30k dataset.

This script demonstrates how to use the FlickrMultistageTrainer for
training multimodal vision-language models using the three-stage approach.
"""

import os
import sys
import argparse
import logging
import torch
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Import the training module
from src.training.flickr_multistage_training import (
    train_flickr30k_multistage,
    create_flickr30k_multistage_model,
    FlickrMultistageTrainer,
)
from src.models.multimodal.vicreg_multimodal_model import VICRegMultimodalModel


# Set up argument parser
def parse_args():
    parser = argparse.ArgumentParser(description="Flickr30k Multistage Training Demo")
    
    # Path arguments
    parser.add_argument(
        "--data-root", type=str, required=True,
        help="Path to Flickr30k dataset root directory"
    )
    parser.add_argument(
        "--output-dir", type=str, default="flickr30k_output",
        help="Directory to save checkpoints and logs"
    )
    
    # Model arguments
    parser.add_argument(
        "--vision-model", type=str, default="ViT-B/16",
        help="Vision model backbone"
    )
    parser.add_argument(
        "--text-model", type=str, default="bert-base-uncased",
        help="Text model backbone"
    )
    parser.add_argument(
        "--embedding-dim", type=int, default=512,
        help="Dimension of joint embedding space"
    )
    
    # Training arguments
    parser.add_argument(
        "--batch-size", type=int, default=64,
        help="Batch size for training"
    )
    parser.add_argument(
        "--stage1-epochs", type=int, default=30,
        help="Number of epochs for Stage 1 (modality-specific)"
    )
    parser.add_argument(
        "--stage2-epochs", type=int, default=15,
        help="Number of epochs for Stage 2 (cross-modal)"
    )
    parser.add_argument(
        "--stage3-epochs", type=int, default=15,
        help="Number of epochs for Stage 3 (end-to-end)"
    )
    parser.add_argument(
        "--num-workers", type=int, default=4,
        help="Number of workers for data loading"
    )
    parser.add_argument(
        "--use-metadata", action="store_true",
        help="Whether to use metadata for advanced batch sampling"
    )
    
    # Device arguments
    parser.add_argument(
        "--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to use for training"
    )
    
    # Run specific stage
    parser.add_argument(
        "--stage", type=int, default=0,
        help="Run specific stage (1, 2, or 3) instead of all stages (0)"
    )
    
    return parser.parse_args()


def main():
    # Parse arguments
    args = parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(os.path.join(args.output_dir, "demo.log")),
            logging.StreamHandler()
        ]
    )
    
    logging.info("Starting Flickr30k Multistage Training Demo")
    logging.info(f"Data root: {args.data_root}")
    logging.info(f"Output directory: {args.output_dir}")
    logging.info(f"Vision model: {args.vision_model}")
    logging.info(f"Text model: {args.text_model}")
    logging.info(f"Device: {args.device}")
    
    # Create device
    device = torch.device(args.device)
    
    # Create model
    logging.info("Creating model...")
    model = create_flickr30k_multistage_model(
        vision_model=args.vision_model,
        text_model=args.text_model,
        embedding_dim=args.embedding_dim,
        freeze_backbone=False,  # We'll handle freezing in the trainer
    )
    
    # If running all stages
    if args.stage == 0:
        logging.info("Running all training stages...")
        train_flickr30k_multistage(
            data_root=args.data_root,
            output_dir=args.output_dir,
            batch_size=args.batch_size,
            stage1_epochs=args.stage1_epochs,
            stage2_epochs=args.stage2_epochs,
            stage3_epochs=args.stage3_epochs,
            vision_model=args.vision_model,
            text_model=args.text_model,
            embedding_dim=args.embedding_dim,
            use_metadata=args.use_metadata,
        )
    else:
        # Create trainer for specific stage
        trainer = FlickrMultistageTrainer(
            model=model,
            data_root=args.data_root,
            output_dir=args.output_dir,
            device=device,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            stage1_epochs=args.stage1_epochs,
            stage2_epochs=args.stage2_epochs,
            stage3_epochs=args.stage3_epochs,
            use_metadata=args.use_metadata,
        )
        
        # Run specific stage
        if args.stage == 1:
            logging.info("Running Stage 1: Modality-Specific Learning...")
            trainer.train_stage1()
        elif args.stage == 2:
            logging.info("Running Stage 2: Cross-Modal Fusion...")
            trainer.train_stage2()
        elif args.stage == 3:
            logging.info("Running Stage 3: End-to-End Fine-tuning...")
            trainer.train_stage3()
        else:
            logging.error(f"Invalid stage: {args.stage}. Must be 0, 1, 2, or 3.")
            return
    
    logging.info("Training complete!")


if __name__ == "__main__":
    main()