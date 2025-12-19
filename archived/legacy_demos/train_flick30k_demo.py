# demos/train_flickr30k_multistage.py

import os
import sys
import argparse
import logging
import torch
from torch.utils.data import DataLoader
import time
from pathlib import Path

# Add root directory to path to ensure imports work correctly
project_root = str(Path(__file__).parent.parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.configs.training_config import (
    TrainingConfig,
    StageConfig,
    LossConfig,
    OptimizerConfig,
    ComponentConfig,
    SchedulerConfig,
)
from src.training.trainers.multistage_trainer import MultistageTrainer
from src.models.model_factory import create_multimodal_model
from src.data.multimodal_dataset import EnhancedMultimodalDataset
from src.data.multimodal_data_utils import create_data_loaders
from src.models.vision.image_preprocessing import ImagePreprocessor
from src.data.tokenization.simple_tokenizer import SimpleTokenizer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(), logging.FileHandler("training.log")],
)

logger = logging.getLogger(__name__)


def create_flickr30k_training_config(
    output_dir="outputs/flickr30k_experiment",
    vision_model="google/vit-base-patch16-224",
    text_model="bert-base-uncased",
    batch_sizes=(64, 64, 32),
    learning_rates=(5e-5, 1e-4, 5e-6),
    projection_dim=512,
    seed=42,
    data_dir="data/flickr30k",
    cache_dir=None,
) -> TrainingConfig:
    """
    Create a complete training configuration for Flickr30k dataset with our multistage approach.

    Args:
        output_dir: Directory to save outputs
        vision_model: Vision model name
        text_model: Text model name
        batch_sizes: Tuple of batch sizes for each stage
        learning_rates: Tuple of learning rates for each stage
        projection_dim: Projection dimension for multimodal fusion
        seed: Random seed
        data_dir: Directory containing Flickr30k dataset
        cache_dir: Directory to cache processed data

    Returns:
        Complete training configuration
    """
    config = TrainingConfig(output_dir=output_dir, seed=seed)

    # Stage 1: Modality-specific learning with contrastive loss
    stage1 = StageConfig(
        name="modality_specific_learning",
        epochs=5,
        batch_size=batch_sizes[0],
        optimizer=OptimizerConfig(lr=learning_rates[0]),
        scheduler=SchedulerConfig(
            name="warmup_cosine", warmup_steps=100, warmup_ratio=0.1
        ),
        losses=[
            LossConfig(
                name="clip_style_loss", weight=1.0, params={"temperature": 0.07}
            ),
            LossConfig(name="decorrelation_loss", weight=0.5, params={}),
        ],
        components=[
            ComponentConfig(name="vision_model", freeze=True, lr_multiplier=0.1),
            ComponentConfig(name="text_model", freeze=True, lr_multiplier=0.1),
            ComponentConfig(name="vision_projection", freeze=False, lr_multiplier=1.0),
            ComponentConfig(name="text_projection", freeze=False, lr_multiplier=1.0),
            ComponentConfig(name="cross_attention", freeze=True, lr_multiplier=0.0),
        ],
        monitor_metric="val_r@5",
        monitor_mode="max",
        mixed_precision=True,
    )

    # Stage 2: Cross-modal fusion with memory queue
    stage2 = StageConfig(
        name="cross_modal_fusion",
        epochs=8,
        batch_size=batch_sizes[1],
        optimizer=OptimizerConfig(lr=learning_rates[1]),
        scheduler=SchedulerConfig(
            name="warmup_cosine", warmup_steps=100, warmup_ratio=0.1
        ),
        losses=[
            LossConfig(
                name="ema_moco_loss",
                weight=1.0,
                params={"queue_size": 8192, "momentum": 0.99, "temperature": 0.07},
            ),
            LossConfig(name="decorrelation_loss", weight=0.25, params={}),
        ],
        components=[
            ComponentConfig(name="vision_model", freeze=True, lr_multiplier=0.0),
            ComponentConfig(name="text_model", freeze=True, lr_multiplier=0.0),
            ComponentConfig(name="vision_projection", freeze=False, lr_multiplier=0.5),
            ComponentConfig(name="text_projection", freeze=False, lr_multiplier=0.5),
            ComponentConfig(name="cross_attention", freeze=False, lr_multiplier=1.0),
        ],
        monitor_metric="val_r@5",
        monitor_mode="max",
        mixed_precision=True,
    )

    # Stage 3: End-to-end fine-tuning with hard negative mining
    stage3 = StageConfig(
        name="end_to_end_fine_tuning",
        epochs=10,
        batch_size=batch_sizes[2],
        optimizer=OptimizerConfig(lr=learning_rates[2]),
        scheduler=SchedulerConfig(
            name="warmup_cosine", warmup_steps=50, warmup_ratio=0.05
        ),
        losses=[
            LossConfig(
                name="hard_negative_mining_contrastive_loss",
                weight=1.0,
                params={"mining_ratio": 0.5, "temperature": 0.05},
            ),
            LossConfig(
                name="feature_consistency_loss",
                weight=0.5,
                params={"distance": "cosine"},
            ),
            LossConfig(name="decorrelation_loss", weight=0.1, params={}),
        ],
        components=[
            ComponentConfig(name="vision_model", freeze=False, lr_multiplier=0.01),
            ComponentConfig(name="text_model", freeze=False, lr_multiplier=0.01),
            ComponentConfig(name="vision_projection", freeze=False, lr_multiplier=0.1),
            ComponentConfig(name="text_projection", freeze=False, lr_multiplier=0.1),
            ComponentConfig(name="cross_attention", freeze=False, lr_multiplier=0.5),
        ],
        monitor_metric="val_r@5",
        monitor_mode="max",
        gradient_accumulation_steps=2,
        mixed_precision=True,
    )

    config.stages = [stage1, stage2, stage3]

    # Data configuration
    config.data_config = {
        "dataset": "flickr30k",
        "data_dir": data_dir,
        "cache_dir": cache_dir,
        "train_split_ratio": 0.8,
        "val_split_ratio": 0.1,
        "test_split_ratio": 0.1,
        "image_size": 224,
        "max_text_length": 77,
        "captions_per_image": 5,
        "min_samples_per_group": 2,
        "aug_image": True,
        "aug_text": False,
        "use_synthetic": False,
        "synthetic_samples": 0,
        "batch_size": config.stages[0].batch_size,
        "max_train_examples": None,
        "max_val_examples": None,
        "max_test_examples": None,
    }

    # Model configuration
    config.model_config = {
        "vision_model": vision_model,
        "text_model": text_model,
        "projection_dim": projection_dim,
        "num_cross_attention_heads": 8,
        "cross_attention_dropout": 0.1,
        "use_gated_fusion": True,
        "use_pretrained": True,
        "use_pretrained_text": True,
        "fusion_dim": projection_dim,
        "fusion_type": "bidirectional",
        "model_size": None,
        "freeze_base_models": False,
    }

    return config


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train a multimodal model on Flickr30k with multistage approach"
    )

    # Model configuration
    parser.add_argument(
        "--vision_model",
        type=str,
        default="google/vit-base-patch16-224",
        help="Vision model name",
    )
    parser.add_argument(
        "--text_model", type=str, default="bert-base-uncased", help="Text model name"
    )
    parser.add_argument(
        "--projection_dim", type=int, default=512, help="Projection dimension"
    )

    # Training configuration
    parser.add_argument(
        "--batch_size",
        type=int,
        default=None,
        help="Base batch size (will be adjusted per stage)",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=None,
        help="Base learning rate (will be adjusted per stage)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="outputs/flickr30k_multistage",
        help="Output directory",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--config_path",
        type=str,
        default=None,
        help="Path to existing config file (optional)",
    )

    # Device configuration
    parser.add_argument(
        "--device", type=str, default=None, help="Device to use (cuda, mps, cpu)"
    )

    # Stage selection
    parser.add_argument(
        "--start_stage",
        type=int,
        default=0,
        help="Stage to start training from (0-indexed)",
    )
    parser.add_argument(
        "--checkpoint", type=str, default=None, help="Path to checkpoint to resume from"
    )

    # Data configuration
    parser.add_argument(
        "--data_dir",
        type=str,
        default="data/flickr30k",
        help="Data directory containing Flickr30k dataset",
    )
    parser.add_argument(
        "--cache_dir", type=str, default=None, help="Directory to cache processed data"
    )

    return parser.parse_args()


def main():
    args = parse_args()

    # Create timestamp for unique run identification
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    output_dir = os.path.join(args.output_dir, timestamp)

    # Set device
    if args.device:
        device = torch.device(args.device)
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    elif hasattr(torch, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    logger.info(f"Using device: {device}")

    # Load or create config
    if args.config_path and os.path.exists(args.config_path):
        logger.info(f"Loading config from {args.config_path}")
        config = TrainingConfig.load(args.config_path)
    else:
        # Adjust batch sizes based on device
        if args.batch_size:
            base_batch_size = args.batch_size
        else:
            base_batch_size = 64 if device.type in ["cuda", "mps"] else 32

        batch_sizes = (base_batch_size, base_batch_size, base_batch_size // 2)

        # Adjust learning rates if specified
        if args.learning_rate:
            base_lr = args.learning_rate
            learning_rates = (base_lr, base_lr * 2, base_lr / 10)
        else:
            learning_rates = (5e-5, 1e-4, 5e-6)

        logger.info("Creating new training config")
        config = create_flickr30k_training_config(
            output_dir=output_dir,
            vision_model=args.vision_model,
            text_model=args.text_model,
            batch_sizes=batch_sizes,
            learning_rates=learning_rates,
            projection_dim=args.projection_dim,
            seed=args.seed,
            data_dir=args.data_dir,
            cache_dir=args.cache_dir,
        )

        # Save config
        os.makedirs(output_dir, exist_ok=True)
        config_path = os.path.join(output_dir, "config.json")
        config.save(config_path)
        logger.info(f"Config saved to {config_path}")

    # Update data directory in config
    if args.data_dir:
        config.data_config["data_dir"] = args.data_dir
    if args.cache_dir:
        config.data_config["cache_dir"] = args.cache_dir

    # Create model
    logger.info("Creating model")

    # Convert model_config dict to an object with attributes that create_multimodal_model can access
    class ModelConfig:
        def __init__(self, config_dict):
            for key, value in config_dict.items():
                setattr(self, key, value)

    model_config_obj = ModelConfig(config.model_config)
    model = create_multimodal_model(model_config_obj, device)

    logger.info(
        f"Model created: Vision={config.model_config['vision_model']}, Text={config.model_config['text_model']}"
    )

    # Initialize image preprocessor and tokenizer (simplified versions for demo)
    logger.info("Creating image preprocessor and tokenizer")
    image_preprocessor = ImagePreprocessor(
        image_size=224,
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225),
    )
    tokenizer = SimpleTokenizer()

    # Create dataloaders
    logger.info("Creating data loaders")

    # Convert data_config dict to an object with attributes
    data_config_obj = ModelConfig(config.data_config)

    train_loader, val_loader, test_loader = create_data_loaders(
        args=data_config_obj,
        image_preprocessor=image_preprocessor,
        tokenizer=tokenizer,
    )

    # Create trainer
    logger.info("Creating multistage trainer")
    trainer = MultistageTrainer(
        model=model,
        config=config,
        train_dataloader=train_loader,
        val_dataloader=val_loader,
        test_dataloader=test_loader,
        device=device,
    )

    # Load checkpoint if specified
    start_stage = args.start_stage
    if args.checkpoint:
        logger.info(f"Loading checkpoint from {args.checkpoint}")
        trainer.load_checkpoint(args.checkpoint)
        # Start from the next stage
        start_stage = trainer.current_stage_idx + 1

    # Train stages
    if start_stage >= len(config.stages):
        logger.warning(
            f"Start stage {start_stage} is beyond the number of stages ({len(config.stages)})"
        )
        logger.info("Skipping training and proceeding to evaluation")
    else:
        logger.info(f"Starting training from stage {start_stage+1}")
        for stage_idx in range(start_stage, len(config.stages)):
            trainer.train_stage(stage_idx)

    # Evaluate
    logger.info("Evaluating final model")
    eval_metrics = trainer.evaluate()

    # Save final model
    final_model_path = os.path.join(output_dir, "final_model.pt")
    trainer.save_model(final_model_path)

    # Print training summary with nice formatting
    print("\n" + "=" * 50)
    print("          FLICKR30K TRAINING SUMMARY          ")
    print("=" * 50)
    print(
        f"Model: {config.model_config['vision_model']} + {config.model_config['text_model']}"
    )
    print(f"Projection dimension: {config.model_config['projection_dim']}")
    print(f"Device: {device}")
    print("-" * 50)
    print("Results by stage:")
    for stage_idx, stage_results in trainer.stage_results.items():
        stage_name = config.stages[stage_idx].name
        best_metric = stage_results.get("best_metric", "N/A")
        if isinstance(best_metric, float):
            best_metric = f"{best_metric:.4f}"
        print(f"  Stage {stage_idx+1} ({stage_name}): Best metric = {best_metric}")

    print("-" * 50)
    print("Final Evaluation Metrics:")
    for metric, value in eval_metrics.items():
        if isinstance(value, float):
            value = f"{value:.4f}"
        print(f"  {metric}: {value}")

    print("-" * 50)
    print(f"Output directory: {output_dir}")
    print("=" * 50 + "\n")

    logger.info("Training complete!")

    # Return metrics for potential scripting use
    return {
        "stages": trainer.stage_results,
        "evaluation": eval_metrics,
        "model_path": final_model_path,
        "config_path": config_path,
    }


if __name__ == "__main__":
    results = main()
