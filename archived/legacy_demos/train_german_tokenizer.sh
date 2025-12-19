#!/bin/sh
# train_german_tokenizer.sh
# Script to train a German BPE tokenizer with vocabulary size 8000

# Ensure the output directory exists
mkdir -p models/tokenizers

# Run the tokenizer training script with the German side of the de-en pair
python demos/train_opensubtitles_tokenizer.py \
  --src_lang de \
  --tgt_lang en \
  --train_lang de \
  --vocab_size 16000 \
  --data_dir data/os \
  --min_frequency 2 \
  --max_examples 5000000 \
  --preserve_case \
  --preserve_punctuation \
  --output_dir models/tokenizers

echo "Tokenizer training completed. The tokenizer is saved in models/tokenizers/de/" 