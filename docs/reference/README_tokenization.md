# Tokenizer Training and Evaluation Guide

This guide explains how to train, evaluate, and use BPE tokenizers with the OpenSubtitles dataset.

## 1. Training BPE Tokenizers

### German Tokenizer

To train a German BPE tokenizer with a vocabulary size of 8000 using the OpenSubtitles dataset:

```bash
./train_german_tokenizer.sh
```

This will train a tokenizer on 500,000 examples from the German side of the de-en OpenSubtitles corpus and save it to `models/tokenizers/de/`.

### English Tokenizer

To train an English BPE tokenizer with a vocabulary size of 8000 using the OpenSubtitles dataset:

```bash
./train_english_tokenizer.sh
```

This will train a tokenizer on 500,000 examples from the English side of the de-en OpenSubtitles corpus and save it to `models/tokenizers/en/`.

### Training with Custom Parameters

If you want to customize the training parameters:

```bash
python train_opensubtitles_tokenizer.py \
  --src_lang de \
  --tgt_lang en \
  --train_lang de \
  --vocab_size 10000 \
  --max_examples 300000 \
  --data_dir data/os \
  --min_frequency 3 \
  --output_dir models/tokenizers
```

## 2. Evaluating Tokenizers

After training your tokenizers, you can evaluate their quality:

```bash
python evaluate_tokenizer.py --tokenizer_path models/tokenizers --lang en
```

This will evaluate the English tokenizer. To evaluate the German tokenizer:

```bash
python evaluate_tokenizer.py --tokenizer_path models/tokenizers --lang de
```

The evaluation will measure:
- Token-to-word ratio
- Out-of-vocabulary (OOV) rate 
- Roundtrip accuracy
- Most common tokens
- Tokenization examples

## 3. Using Tokenizers for Translation

Now you can use your trained tokenizers for machine translation with the OpenSubtitles dataset:

```bash
python demos/translation_example.py \
  --dataset opensubtitles \
  --src_lang de \
  --tgt_lang en \
  --max_train_examples 100000 \
  --max_val_examples 20000 \
  --learning_rate 0.0005 \
  --warmup_steps 400
```

This will:
1. Load the OpenSubtitles dataset
2. Use your trained BPE tokenizers
3. Train a transformer model for translation
4. Save model checkpoints and training graphs

### Training Options

You can customize your training with these options:

- `--dataset`: Choose between `europarl` or `opensubtitles`
- `--src_lang`: Source language code (default: de)
- `--tgt_lang`: Target language code (default: en)
- `--max_train_examples`: Max number of training examples (default: 100000)
- `--max_val_examples`: Max number of validation examples (default: 20000)
- `--learning_rate`: Learning rate (default: 0.0001)
- `--warmup_steps`: Warmup steps for scheduler (default: 2000)
- `--use_mixed_precision`: Enable mixed precision training
- `--use_gradient_scaling`: Use gradient scaling with mixed precision

## 4. Tips for Better Results

1. **Tokenizer Quality**: A good BPE tokenizer should have:
   - Tokens-per-word ratio of 1.2-1.5
   - OOV rate < 1%
   - High roundtrip accuracy (>95%)

2. **Training Data Size**: 
   - For tokenizers: 300k-500k examples is usually sufficient
   - For translation: Start with 100k examples and gradually increase

3. **Hyperparameters**:
   - Learning rate: Start with 0.0005
   - Batch size: 32-128 depending on your hardware
   - Warmup steps: 400-2000 depending on dataset size
   - Encoder/decoder layers: 4-6

4. **Mixed Precision**:
   - Use `--use_mixed_precision --use_gradient_scaling` for faster training if your GPU supports it
   - Note that mixed precision can sometimes cause instability

## 5. Troubleshooting

If you encounter issues:

1. **High Loss/Perplexity**: Try reducing learning rate or increasing warmup steps
2. **OOM Errors**: Reduce batch size or max sequence length
3. **Poor Tokenization**: Train on more diverse data or increase vocab size
4. **NaN Loss**: Disable mixed precision or use gradient scaling

## 6. Example Results

Good tokenization examples:

```
Original: This is a translation example.
Tokens: ['This', ' is', ' a', ' trans', 'lation', ' example', '.']
```

Poor tokenization examples:

```
Original: This is a translation example.
Tokens: ['T', 'h', 'i', 's', ' ', 'i', 's', ' ', 'a', ' ', 't', 'r', 'a', 'n', 's', 'l', 'a', 't', 'i', 'o', 'n', ' ', 'e', 'x', 'a', 'm', 'p', 'l', 'e', '.']
``` 