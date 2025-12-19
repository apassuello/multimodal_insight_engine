#!/bin/bash

# Fix the VICReg training issues with more aggressive changes

# Stop the existing process (if running)
echo "Stopping existing VICReg training process..."
pkill -f "vicreg_training_config.py"
sleep 2

# Create output directories (ensure they exist)
mkdir -p multimodal_outputs/checkpoints/enhanced_multimodal
mkdir -p multimodal_outputs/logs/enhanced_multimodal/alignment_plots

# Remove existing checkpoints to start fresh
echo "Removing existing checkpoints..."
rm -f multimodal_outputs/checkpoints/enhanced_multimodal/checkpoint_epoch_*.pt
rm -f multimodal_outputs/checkpoints/enhanced_multimodal/initial_checkpoint.pt
rm -f multimodal_outputs/checkpoints/enhanced_multimodal/stage1_checkpoint.pt
rm -f multimodal_outputs/logs/history.json

# Run the fixed VICReg training command with aggressive settings
echo "Starting fixed VICReg training with significantly adjusted parameters..."
python demos/vicreg_training_config.py \
    --dataset flickr30k \
    --vision_model vit-base \
    --text_model bert-base \
    --sim_weight 100.0 \
    --var_weight 0.1 \
    --cov_weight 0.1 \
    --vicreg_warmup_epochs 15 \
    --use_curriculum \
    --use_contrastive_pretrain \
    --contrastive_pretrain_steps 2000 \
    --temperature 0.5 \
    --batch_size 64 \
    --num_epochs 30 \
    --learning_rate 0.000005 \
    --weight_decay 0.001 \
    --warmup_steps 2000 \
    --use_multistage_training \
    --freeze_base_models \
    --use_pretrained \
    --use_semantic_batching \
    --min_samples_per_group 8 \
    --fusion_dim 768 \
    --device mps \
    --output_dir multimodal_outputs \
    --checkpoint_dir checkpoints/enhanced_multimodal \
    --log_dir logs/enhanced_multimodal \
    --verbose_logging \
    --captions_per_image 5