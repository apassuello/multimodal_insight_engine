#!/bin/bash

# Fix the VICReg training issues

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

# Run the fixed VICReg training command
echo "Starting fixed VICReg training with adjusted parameters..."
python demos/vicreg_training_config.py \
    --dataset flickr30k \
    --vision_model vit-base \
    --text_model bert-base \
    --sim_weight 75.0 \
    --var_weight 1.0 \
    --cov_weight 0.5 \
    --vicreg_warmup_epochs 10 \
    --use_curriculum \
    --use_contrastive_pretrain \
    --contrastive_pretrain_steps 1000 \
    --temperature 0.2 \
    --batch_size 96 \
    --num_epochs 30 \
    --learning_rate 0.00003 \
    --weight_decay 0.01 \
    --warmup_steps 1000 \
    --use_multistage_training \
    --freeze_base_models \
    --use_pretrained \
    --use_semantic_batching \
    --min_samples_per_group 5 \
    --fusion_dim 768 \
    --device mps \
    --output_dir multimodal_outputs \
    --checkpoint_dir checkpoints/enhanced_multimodal \
    --log_dir logs/enhanced_multimodal \
    --verbose_logging