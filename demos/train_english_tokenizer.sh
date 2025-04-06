#!/bin/sh
# train_english_tokenizer.sh
# Script to train an English BPE tokenizer with vocabulary size 8000

# Ensure the output directory exists
mkdir -p models/tokenizers

# Run the tokenizer training script with the English side of the de-en pair
python train_opensubtitles_tokenizer.py \
  --src_lang de \
  --tgt_lang en \
  --train_lang en \
  --vocab_size 8000 \
  --data_dir data/os \
  --min_frequency 2 \
  --max_examples 5000000 \
  --output_dir models/tokenizers

echo "Tokenizer training completed. The tokenizer is saved in models/tokenizers/en/" 