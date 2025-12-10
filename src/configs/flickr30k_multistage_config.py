# src/configs/flickr30k_multistage_config.py

from src.configs.training_config import (
    ComponentConfig,
    LossConfig,
    OptimizerConfig,
    StageConfig,
    TrainingConfig,
)


def create_flickr30k_training_config(
    output_dir="outputs/flickr30k_experiment",
    vision_model="google/vit-base-patch16-224",
    text_model="bert-base-uncased",
    batch_sizes=(64, 64, 32),
    learning_rates=(5e-5, 1e-4, 5e-6),
    projection_dim=512,
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

    Returns:
        Complete training configuration
    """
    config = TrainingConfig(output_dir=output_dir)

    # Stage 1: Modality-specific learning with contrastive loss
    stage1 = StageConfig(
        name="modality_specific_learning",
        epochs=5,
        batch_size=batch_sizes[0],
        optimizer=OptimizerConfig(lr=learning_rates[0]),
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
        "train_split_ratio": 0.8,
        "val_split_ratio": 0.1,
        "test_split_ratio": 0.1,
        "image_size": 224,
        "max_text_length": 77,
        "captions_per_image": 5,
        "min_samples_per_group": 2,
        "aug_image": True,
        "aug_text": False,
    }

    # Model configuration
    config.model_config = {
        "vision_model": vision_model,
        "text_model": text_model,
        "projection_dim": projection_dim,
        "num_cross_attention_heads": 8,
        "cross_attention_dropout": 0.1,
        "use_gated_fusion": True,
    }

    return config
