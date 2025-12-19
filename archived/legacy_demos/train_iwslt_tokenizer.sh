#!/bin/sh
# train_iwslt_tokenizer.sh
# Script to train German and English BPE tokenizers with vocabulary size 16000 using IWSLT dataset

# Ensure the output directory exists
mkdir -p models/tokenizers/iwslt/joint

# Train German tokenizer
echo "Training German tokenizer..."
python demos/train_iwslt_tokenizer.py \
  --src_lang de \
  --tgt_lang en \
  --train_lang de \
  --vocab_size 16000 \
  --data_dir data/iwslt \
  --min_frequency 2 \
  --max_examples 5000000 \
  --preserve_case \
  --preserve_punctuation \
  --output_dir models/tokenizers/iwslt \
  --year 2017 \
  --split train

# Train English tokenizer
echo "Training English tokenizer..."
python demos/train_iwslt_tokenizer.py \
  --src_lang de \
  --tgt_lang en \
  --train_lang en \
  --vocab_size 16000 \
  --data_dir data/iwslt \
  --min_frequency 2 \
  --max_examples 5000000 \
  --preserve_case \
  --preserve_punctuation \
  --output_dir models/tokenizers/iwslt/joint \
  --year 2017 \
  --split train

# Train joint tokenizer
echo "Training joint tokenizer..."
python demos/train_iwslt_tokenizer.py \
  --src_lang de \
  --tgt_lang en \
  --train_lang joint \
  --vocab_size 32000 \
  --data_dir data/iwslt \
  --min_frequency 2 \
  --max_examples 5000000 \
  --preserve_case \
  --preserve_punctuation \
  --output_dir models/tokenizers/iwslt \
  --year 2017 \
  --split train

echo "Tokenizer training completed. The tokenizers are saved in models/tokenizers/"
echo "- German tokenizer: models/tokenizers/de/"
echo "- English tokenizer: models/tokenizers/en/"
echo "- Joint tokenizer: models/tokenizers/joint/" 