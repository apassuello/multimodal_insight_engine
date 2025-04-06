# Translation Model Architecture and Training Pipeline

## Overview

This document provides a comprehensive description of the neural machine translation system implementation, based on the command:

```bash
python demos/translation_example.py --dataset opensubtitles --src_lang de --tgt_lang en --learning_rate 0.0005
```

The system implements a Transformer-based neural machine translation model for German to English translation, using the OpenSubtitles dataset.

## Model Architecture

### Encoder-Decoder Transformer

The model is an implementation of the standard Transformer architecture as described in "Attention is All You Need" (Vaswani et al., 2017).

**Key Specifications:**
- **Model Dimension (d_model)**: 512
- **Number of Attention Heads**: 8
- **Number of Encoder Layers**: 4
- **Number of Decoder Layers**: 4
- **Feed-Forward Network Size (d_ff)**: 2048
- **Dropout Rate**: 0.1
- **Maximum Sequence Length**: 5000
- **Positional Encoding**: Sinusoidal
- **Parameter Count**: 41,731,397 trainable parameters
- **Embedding Sharing**: No shared embeddings between encoder and decoder

### Encoder Architecture

Each encoder layer consists of:
1. **Multi-Head Self-Attention**: 8 attention heads with scaled dot-product attention
2. **Position-wise Feed-Forward Network**: Two linear transformations with a ReLU activation
3. **Layer Normalization**: Applied after each sub-layer with residual connections
4. **Dropout**: Applied after each sub-layer (p=0.1)

### Decoder Architecture

Each decoder layer consists of:
1. **Masked Multi-Head Self-Attention**: 8 attention heads with causal masking
2. **Multi-Head Cross-Attention**: 8 attention heads attending to encoder outputs
3. **Position-wise Feed-Forward Network**: Two linear transformations with a ReLU activation
4. **Layer Normalization**: Applied after each sub-layer with residual connections
5. **Dropout**: Applied after each sub-layer (p=0.1)

### Embedding Layers

- **Token Embeddings**: Separate embeddings for source and target vocabulary
- **Positional Encodings**: Fixed sinusoidal position embeddings
- **Input Dimension**: 512 (same as model dimension)
- **Output Projection**: Linear projection to target vocabulary size

## Training Parameters

### Optimization

- **Optimizer**: Adam
- **Learning Rate**: 0.0005 (specified in command line)
- **Beta Parameters**: (0.9, 0.98)
- **Epsilon**: 1e-9
- **Gradient Clipping**: 1.0
- **Weight Initialization**: Xavier Uniform for most layers, 1.0 for LayerNorm

### Learning Rate Schedule

- **Scheduler Type**: Inverse square root decay
- **Warmup Steps**: 4000
- **Schedule Formula**: min(step^(-0.5), step * warmup_steps^(-1.5)) * warmup_steps^0.5

### Regularization

- **Dropout Rate**: 0.1 throughout the network
- **Label Smoothing**: 0.1
- **Early Stopping**: Based on validation loss with configurable patience

### Training Configuration

- **Batch Size**: 128
- **Training Examples**: 100,000 parallel sentences (default)
- **Validation Examples**: 20,000 parallel sentences (default)
- **Device**: MPS (Apple Silicon GPU) auto-detected
- **Epochs**: Default is not specified in the command; uses a reasonable default

## Tokenization Pipeline

### Tokenizer Type

- **Tokenizer**: OptimizedBPETokenizer (Byte Pair Encoding)
- **Vocabulary Size**: 8,005 tokens for both source (German) and target (English)
- **Special Tokens**: BOS (Beginning of Sequence), EOS (End of Sequence), PAD (Padding)

### Pre-processing Steps

1. **Text Cleaning**:
   - Normalization of Unicode characters
   - Lowercasing (optional, default=True)
   - Whitespace normalization
   - Punctuation segmentation

2. **BPE Tokenization**:
   - Splitting text into subword units
   - Adding special tokens:
     - BOS token added at the beginning
     - EOS token added at the end
     - PAD token used for batch padding

3. **Sequence Preparation**:
   - Converting tokens to indices
   - Source sequence: [BOS] + subwords + [EOS]
   - Target sequence: [BOS] + subwords + [EOS]
   - Padding sequences to the same length in each batch

## Data Processing

### Dataset Processing

1. **Loading Data**:
   - Loading parallel sentences from OpenSubtitles dataset (German-English pairs)
   - Limiting to specified maximum examples (100,000 by default)

2. **Preprocessing Data**:
   - Tokenizing source and target sentences using BPE
   - Creating numerical token sequences with special tokens
   - Creating attention masks for padding

3. **Batch Creation**:
   - Dynamic batching based on sequence length
   - Padding to maximum length within each batch
   - Creating source and target masks

### Training Loop

1. **Forward Pass**:
   - Shifting target sequences for teacher forcing:
     - Input: [BOS] + subwords
     - Output: subwords + [EOS]
   - Creating source padding mask and target causal mask
   - Running model forward pass to get logits

2. **Loss Calculation**:
   - Using label smoothing loss function
   - Masking out padding tokens in loss calculation
   - Computing per-token loss and averaging

3. **Backward Pass**:
   - Computing gradients using backpropagation
   - Optional gradient scaling for mixed precision (disabled by default)
   - Clipping gradients to prevent explosion
   - Updating model parameters with Adam optimizer
   - Updating learning rate with scheduler

### Validation Process

1. **Forward Pass**:
   - Similar to training but with no gradient calculation
   - Using teacher forcing with ground truth targets

2. **Metrics Calculation**:
   - Computing validation loss
   - Computing perplexity (exp(loss))
   - Tracking best model based on validation loss

## Full System Diagram

```
┌─────────────────────────────────────────────────────────────────────────┐
│                          PREPROCESSING PIPELINE                          │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────┐   ┌────────────────┐   ┌────────────────┐
│ OpenSubtitles Dataset   │──▶│  Text Cleaning  │──▶│   BPE Encoding  │
│ (German-English pairs)  │   │  & Normalization│   │  (Vocab: 8005)  │
└─────────────────────────┘   └────────────────┘   └────────────────┘
                                                            │
                                                            ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                           TRANSFORMER MODEL                              │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  ENCODER (4 layers)                 DECODER (4 layers)                  │
│  ┌───────────────────┐              ┌───────────────────┐              │
│  │ Token Embedding   │              │ Token Embedding   │              │
│  │ (d_model=512)     │              │ (d_model=512)     │              │
│  └───────────────────┘              └───────────────────┘              │
│           │                                  │                          │
│           ▼                                  ▼                          │
│  ┌───────────────────┐              ┌───────────────────┐              │
│  │ Positional Encoding              │ Positional Encoding              │
│  │ (Sinusoidal)      │              │ (Sinusoidal)      │              │
│  └───────────────────┘              └───────────────────┘              │
│           │                                  │                          │
│           ▼                                  ▼                          │
│  ┌───────────────────┐              ┌───────────────────┐              │
│  │ N=4 Encoder Layers│              │ Masked Self-Attn  │              │
│  │ ┌─────────────┐   │              │ (8 heads)         │              │
│  │ │ Self-Attn   │   │              └───────────────────┘              │
│  │ │ (8 heads)   │   │                       │                          │
│  │ └─────────────┘   │                       ▼                          │
│  │        │          │              ┌───────────────────┐              │
│  │        ▼          │              │ Cross-Attention   │◀─────────────┘
│  │ ┌─────────────┐   │              │ (8 heads)         │
│  │ │ Feed-Forward │   │              └───────────────────┘
│  │ │ (2048 units) │   │                       │
│  │ └─────────────┘   │                       ▼
│  └───────────────────┘              ┌───────────────────┐
│           │                         │ Feed-Forward      │
│           └─────────────┐           │ (2048 units)      │
│                         │           └───────────────────┘
│                         │                    │
│                         └──────────────┐    │
│                                        ▼    ▼
│                               ┌───────────────────┐
│                               │ Linear Projection │
│                               │ to Vocab Size     │
│                               └───────────────────┘
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                       TRAINING & OPTIMIZATION                            │
├─────────────────────────────────────────────────────────────────────────┤
│  ┌─────────────────┐   ┌─────────────────┐   ┌─────────────────┐       │
│  │ Label Smoothing │   │ Adam Optimizer  │   │ Learning Rate   │       │
│  │ Loss (ε=0.1)    │   │ (lr=0.0005)     │   │ Scheduler       │       │
│  └─────────────────┘   └─────────────────┘   └─────────────────┘       │
└─────────────────────────────────────────────────────────────────────────┘
``` 