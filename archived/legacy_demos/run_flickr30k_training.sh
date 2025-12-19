#!/bin/bash
# demos/run_flickr30k_training.sh

# Set the path to your Flickr30k dataset
DATA_ROOT="/path/to/flickr30k"

# Set the output directory
OUTPUT_DIR="flickr30k_output"

# Create output directory if it doesn't exist
mkdir -p $OUTPUT_DIR

# Run the multistage training demo with default parameters
python demos/flickr30k_multistage_training_demo.py \
  --data-root $DATA_ROOT \
  --output-dir $OUTPUT_DIR \
  --vision-model vit-base \
  --text-model bert-base-uncased \
  --embedding-dim 512 \
  --batch-size 64 \
  --stage1-epochs 30 \
  --stage2-epochs 15 \
  --stage3-epochs 15 \
  --num-workers 4 \
  --use-metadata \
  --stage 0

# You can also run individual stages with:
# --stage 1  # Run only Stage 1: Modality-Specific Learning
# --stage 2  # Run only Stage 2: Cross-Modal Fusion 
# --stage 3  # Run only Stage 3: End-to-End Fine-tuning