#!/bin/bash
# Script to train BPE tokenizers for WMT datasets

# Set default values
VOCAB_SIZE=32000
SRC_LANG=en
TGT_LANG=de
YEAR=14
MAX_EXAMPLES=1000000
NUM_WORKERS=4
BATCH_SIZE=1000

# Parse command line arguments
while [[ $# -gt 0 ]]; do
  case $1 in
    --vocab_size)
      VOCAB_SIZE="$2"
      shift 2
      ;;
    --src_lang)
      SRC_LANG="$2"
      shift 2
      ;;
    --tgt_lang)
      TGT_LANG="$2"
      shift 2
      ;;
    --year)
      YEAR="$2"
      shift 2
      ;;
    --max_examples)
      MAX_EXAMPLES="$2"
      shift 2
      ;;
    --num_workers)
      NUM_WORKERS="$2"
      shift 2
      ;;
    --batch_size)
      BATCH_SIZE="$2"
      shift 2
      ;;
    *)
      echo "Unknown option: $1"
      exit 1
      ;;
  esac
done

echo "Training WMT$YEAR tokenizers for $SRC_LANG-$TGT_LANG with vocab size $VOCAB_SIZE"
echo "Using $NUM_WORKERS workers and batch size $BATCH_SIZE"

# Train a joint tokenizer for both languages
python demos/train_wmt_tokenizer.py \
  --vocab_size $VOCAB_SIZE \
  --src_lang $SRC_LANG \
  --tgt_lang $TGT_LANG \
  --year $YEAR \
  --max_examples $MAX_EXAMPLES \
  --num_workers $NUM_WORKERS \
  --batch_size $BATCH_SIZE \
  --joint

# Train separate tokenizers for source and target languages
python demos/train_wmt_tokenizer.py \
  --vocab_size $VOCAB_SIZE \
  --src_lang $SRC_LANG \
  --tgt_lang $TGT_LANG \
  --year $YEAR \
  --max_examples $MAX_EXAMPLES \
  --num_workers $NUM_WORKERS \
  --batch_size $BATCH_SIZE

echo "Tokenizer training complete!" 