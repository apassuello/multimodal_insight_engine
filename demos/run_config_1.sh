#!/bin/bash

# Stop any existing VICReg training
echo "Stopping existing VICReg processes..."
pkill -f "vicreg_training_config.py"
sleep 2

# Remove previous outputs for clean start
echo "Removing previous outputs..."
rm -f multimodal_outputs/checkpoints/enhanced_multimodal/checkpoint_epoch_*.pt
rm -f multimodal_outputs/checkpoints/enhanced_multimodal/stage1_checkpoint.pt
rm -f multimodal_outputs/checkpoints/enhanced_multimodal/initial_checkpoint.pt
rm -f multimodal_outputs/logs/history.json

# Create output directories if needed
mkdir -p multimodal_outputs/checkpoints/enhanced_multimodal
mkdir -p multimodal_outputs/logs/enhanced_multimodal/alignment_plots

# Run one-time script to update the modified script
cat > /tmp/update_tokenizer.py << 'EOF'
#!/usr/bin/env python3
import sys
import os
from pathlib import Path

# Add the repository root to the system path
sys.path.append(str(Path(__file__).parent.parent))

# Update vicreg_training_config.py to use the adapter
config_file = "demos/vicreg_training_config.py"

with open(config_file, "r") as f:
    content = f.read()

# Replace BertTokenizer with our adapter
if "from transformers import BertTokenizer" in content:
    content = content.replace(
        "from transformers import BertTokenizer",
        "from src.data.tokenization.bert_tokenizer_adapter import BertTokenizerAdapter"
    )
    content = content.replace(
        "tokenizer = BertTokenizer.from_pretrained(\"bert-base-uncased\")",
        "tokenizer = BertTokenizerAdapter(pretrained_model_name=\"bert-base-uncased\", max_length=77)"
    )
    
    with open(config_file, "w") as f:
        f.write(content)
    print("Updated vicreg_training_config.py to use BertTokenizerAdapter")
else:
    print("No BertTokenizer import found in vicreg_training_config.py")
EOF

# Execute the update script
python /tmp/update_tokenizer.py

# Run Config 1: Pure Contrastive Learning with Higher Learning Rate
echo "Starting training with Config 1: Pure Contrastive Learning..."
python demos/vicreg_training_config.py \
    --dataset flickr30k \
    --vision_model vit-base \
    --text_model bert-base \
    --loss_type vicreg \
    --use_curriculum \
  --vicreg_warmup_epochs 2 \
    --temperature 0.07 \
    --batch_size 64 \
    --num_epochs 20 \
    --learning_rate 0.001 \
    --weight_decay 0.001 \
    --sim_weight 5.0 \
    --var_weight 5.0 \
    --cov_weight 1.0 \
    --warmup_steps 300 \
    --clip_grad_norm 1.0 \
    --use_pretrained \
    --use_semantic_batching \
    --min_samples_per_group 5 \
    --max_samples_per_group 30 \
    --cap_strategy random \
    --fusion_dim 512 \
    --device mps \
    --output_dir multimodal_outputs \
    --checkpoint_dir checkpoints/enhanced_multimodal \
    --log_dir logs/enhanced_multimodal \
    --captions_per_image 5 \
    --verbose_logging \
    --use_multistage_training \
    --use_contrastive_pretrain \
    --contrastive_pretrain_steps 300 \
