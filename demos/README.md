# MultiModal Insight Engine - Demo Scripts

This directory contains demonstration scripts showcasing different features of the MultiModal Insight Engine.

**New to the project?** Start with [Getting Started Guide](../GETTING_STARTED.md)

---

## Quick Navigation

### Start Here (5-30 minutes)
- **[language_model_demo.py](#language-modeling)** - Train a simple language model
- **[translation_example.py](#translation)** - Neural machine translation
- **[demo_safety.py](#safety-evaluation)** - Safety filtering and evaluation

### Advanced (30+ minutes)
- **[constitutional_ai_demo.py](#constitutional-ai)** - Full Constitutional AI pipeline
- **[red_teaming_demo.py](#red-teaming)** - Adversarial testing
- **[model_optimization_demo.py](#model-optimization)** - Pruning and quantization

### Vision & Multimodal (30+ minutes)
- **[multimodal_training_demo.py](#multimodal-training)** - Combined text and vision
- **[vision_transformer_demo.py](#vision-transformer)** - Vision transformer models
- **[hardware_profiling_demo.py](#hardware-profiling)** - Performance analysis

---

## Demo Details

### Language Modeling
**File**: `language_model_demo.py`
**Duration**: 10-30 minutes
**Difficulty**: Beginner

Train a transformer-based language model from scratch.

```bash
# Basic usage
python demos/language_model_demo.py

# With custom parameters
python demos/language_model_demo.py \
  --dataset wikitext \
  --model_config small \
  --num_epochs 5 \
  --batch_size 32

# With different models
python demos/language_model_demo.py --model_config medium
python demos/language_model_demo.py --model_config large
```

**What you'll learn:**
- How to build a tokenizer
- How to set up training loops
- How to evaluate a language model
- How to generate text

**Output**: Trained model, training metrics, generated text samples

---

### Translation
**File**: `translation_example.py`
**Duration**: 15-45 minutes
**Difficulty**: Beginner

Implement neural machine translation between languages.

```bash
# German to English translation
python demos/translation_example.py \
  --src_lang de \
  --tgt_lang en \
  --dataset europarl

# Other language pairs
python demos/translation_example.py --src_lang fr --tgt_lang en
python demos/translation_example.py --src_lang es --tgt_lang en

# With custom data
python demos/translation_example.py \
  --src_lang de \
  --tgt_lang en \
  --data_file path/to/corpus.txt
```

**What you'll learn:**
- Parallel corpus training
- Sequence-to-sequence models
- Translation evaluation metrics (BLEU)
- Tokenizer training for multiple languages

**Output**: Trained translation model, translation samples, BLEU scores

---

### Safety Evaluation
**File**: `demo_safety.py`
**Duration**: 5-10 minutes
**Difficulty**: Beginner

Evaluate model outputs for safety and filter harmful content.

```bash
# Basic safety demo
python demos/demo_safety.py

# With specific model
python demos/demo_safety.py --model_name phi-2

# With custom inputs
python demos/demo_safety.py \
  --model_name gpt2 \
  --input_file path/to/prompts.txt

# Verbose output
python demos/demo_safety.py --verbose
```

**What you'll learn:**
- Content filtering techniques
- Safety evaluation metrics
- Red teaming basics
- Harmful pattern detection

**Output**: Safety scores, filtered responses, violation reports

---

### Constitutional AI
**File**: `constitutional_ai_demo.py`
**Duration**: 10 minutes - 2+ hours (depending on scale)
**Difficulty**: Intermediate

Full Constitutional AI training pipeline: critique-revision + preference learning + PPO.

```bash
# Quick test (5 min, educational only)
python demos/constitutional_ai_demo.py --quick-test

# Small scale (30 min)
python demos/constitutional_ai_demo.py \
  --phase both \
  --num-prompts 50 \
  --num-ppo-steps 10

# Medium scale (2 hours)
python demos/constitutional_ai_demo.py \
  --phase both \
  --num-prompts 200 \
  --num-ppo-steps 50

# Production scale (8-12 hours)
python demos/constitutional_ai_demo.py \
  --phase both \
  --num-prompts 1000 \
  --num-ppo-steps 100

# Just Phase 1 (critique-revision)
python demos/constitutional_ai_demo.py --phase 1

# Just Phase 2 (RLAIF)
python demos/constitutional_ai_demo.py --phase 2
```

**Important**: The quick-test and small examples are for learning. See `CRITICAL_README.md` for production training guidance.

**What you'll learn:**
- Critique and revision generation
- Preference comparison
- Reward model training (Bradley-Terry loss)
- PPO training for RLHF
- Constitutional AI principles

**Output**: Fine-tuned model, reward models, training curves

---

### Red Teaming
**File**: `red_teaming_demo.py`
**Duration**: 5-15 minutes
**Difficulty**: Intermediate

Adversarial testing to identify model vulnerabilities.

```bash
# Basic red teaming
python demos/red_teaming_demo.py

# With specific model
python demos/red_teaming_demo.py --model phi-2

# Verbose output shows attack strategies
python demos/red_teaming_demo.py --verbose

# With custom prompts
python demos/red_teaming_demo.py \
  --prompt_file path/to/attacks.txt \
  --num_tests 20

# Specific attack types
python demos/red_teaming_demo.py \
  --attack_types prompt_injection,directive_smuggling
```

**What you'll learn:**
- Prompt injection attacks
- Jailbreak techniques
- Adversarial input generation
- Model robustness evaluation

**Output**: Attack results, vulnerability reports, model responses

---

### Model Optimization
**File**: `model_optimization_demo.py`
**Duration**: 20-45 minutes
**Difficulty**: Intermediate

Optimize models for speed and size using pruning and quantization.

```bash
# Magnitude pruning
python demos/model_optimization_demo.py \
  --technique pruning \
  --compression_ratio 0.5

# Quantization (INT8)
python demos/model_optimization_demo.py \
  --technique quantization \
  --quantization_type int8

# Quantization (INT4)
python demos/model_optimization_demo.py \
  --technique quantization \
  --quantization_type int4

# Mixed precision training
python demos/model_optimization_demo.py \
  --technique mixed_precision

# Compare multiple techniques
python demos/model_optimization_demo.py --compare_all
```

**What you'll learn:**
- Model pruning strategies
- Quantization techniques
- Mixed precision training
- Speed/accuracy trade-offs
- Memory efficiency

**Output**: Optimized models, performance benchmarks, compression ratios

---

### Multimodal Training
**File**: `multimodal_training_demo.py`
**Duration**: 30+ minutes
**Difficulty**: Advanced

Train on combined text and image data (CLIP-style alignment).

```bash
# Basic multimodal training
python demos/multimodal_training_demo.py

# With Flickr30K dataset
python demos/multimodal_training_demo.py --dataset flickr30k

# With custom data
python demos/multimodal_training_demo.py \
  --image_dir path/to/images \
  --caption_file path/to/captions.txt

# Different training strategies
python demos/multimodal_training_demo.py --strategy end_to_end
python demos/multimodal_training_demo.py --strategy cross_modal
```

**What you'll learn:**
- Vision and language alignment
- Contrastive learning
- Multimodal embeddings
- Cross-modal retrieval

**Output**: Multimodal model, alignment metrics, retrieval results

---

### Vision Transformer
**File**: `vision_transformer_demo.py`
**Duration**: 20-45 minutes
**Difficulty**: Advanced

Build and train vision transformers for image classification.

```bash
# Basic vision transformer
python demos/vision_transformer_demo.py

# With different input sizes
python demos/vision_transformer_demo.py --image_size 224
python demos/vision_transformer_demo.py --image_size 384

# Different patch sizes
python demos/vision_transformer_demo.py --patch_size 16
python demos/vision_transformer_demo.py --patch_size 8

# With pretrained backbone
python demos/vision_transformer_demo.py --pretrained
```

**What you'll learn:**
- Vision transformer architecture
- Patch embeddings
- Image classification
- Transfer learning
- Fine-tuning strategies

**Output**: Trained ViT model, classification metrics, attention visualizations

---

### Hardware Profiling
**File**: `hardware_profiling_demo.py`
**Duration**: 10-20 minutes
**Difficulty**: Intermediate

Analyze model performance on your hardware.

```bash
# Profile default model
python demos/hardware_profiling_demo.py

# Small model
python demos/hardware_profiling_demo.py --model_size small

# Large model
python demos/hardware_profiling_demo.py --model_size large

# Generate comparison report
python demos/hardware_profiling_demo.py --compare
```

**What you'll learn:**
- Memory profiling
- Throughput measurement
- Latency analysis
- Hardware utilization
- Optimization opportunities

**Output**: Performance metrics, memory usage, bottleneck analysis

---

## Running a Demo

### General Pattern
```bash
# 1. Activate virtual environment
source venv/bin/activate

# 2. Run the demo
python demos/DEMO_NAME.py [OPTIONS]

# 3. View results
# Check console output or generated files
```

### Common Options (for most demos)
```bash
--help              Show available options
--verbose           Verbose output
--seed 42          Set random seed for reproducibility
--device cpu       Run on CPU (default: cuda if available)
--output_dir ./    Save results to directory
```

### Typical Workflow
```bash
# 1. Check available options
python demos/language_model_demo.py --help

# 2. Run with small settings first
python demos/language_model_demo.py --num_epochs 1

# 3. Increase if happy with results
python demos/language_model_demo.py --num_epochs 5
```

---

## Demo Organization Plan

The demos are currently being reorganized. Future structure:

```
demos/
├── README.md                        (this file)
├── 01_quickstart/
│   ├── language_model_demo.py
│   └── translation_example.py
├── 02_safety/
│   ├── demo_safety.py
│   └── red_teaming_demo.py
├── 03_advanced/
│   ├── constitutional_ai_demo.py
│   └── model_optimization_demo.py
├── 04_multimodal/
│   ├── multimodal_training_demo.py
│   └── vision_transformer_demo.py
├── 05_profiling/
│   └── hardware_profiling_demo.py
└── archive/                        (deprecated demos)
```

---

## Troubleshooting

### "ModuleNotFoundError: No module named 'src'"
Make sure you're running from the project root:
```bash
cd /path/to/multimodal_insight_engine
python demos/language_model_demo.py
```

### "CUDA out of memory"
Use CPU instead:
```bash
python demos/language_model_demo.py --device cpu
```

Or reduce model/batch size:
```bash
python demos/language_model_demo.py --batch_size 8
```

### "Demo is too slow"
Check if it's downloading data or models first. Some demos download datasets on first run.

For reproducibility, use a seed:
```bash
python demos/language_model_demo.py --seed 42
```

### "No data files found"
Some demos download data automatically. Ensure you have internet connection and disk space.

---

## Learning Path

### Recommended Progression:
1. **Start**: GETTING_STARTED.md
2. **Learn Models**: `language_model_demo.py`
3. **Learn Training**: `translation_example.py`
4. **Learn Safety**: `demo_safety.py`
5. **Learn Advanced**: `constitutional_ai_demo.py`
6. **Explore Special Topics**: Multimodal, Vision, Optimization

### By Topic:

**Language Models**:
1. language_model_demo.py
2. red_teaming_demo.py
3. constitutional_ai_demo.py

**Translation**:
1. translation_example.py
2. multimodal_training_demo.py

**Safety**:
1. demo_safety.py
2. red_teaming_demo.py
3. constitutional_ai_demo.py

**Vision**:
1. vision_transformer_demo.py
2. multimodal_training_demo.py

**Optimization**:
1. model_optimization_demo.py
2. hardware_profiling_demo.py

---

## Contributing New Demos

Want to add a demo? Follow these guidelines:

1. **File naming**: `descriptive_name_demo.py` or `feature_name_example.py`
2. **Structure**:
   - Module docstring explaining what it does
   - argparse for CLI options
   - Main execution block
   - Clear comments
3. **Documentation**: Add entry to this README
4. **Testing**: Ensure it runs without errors
5. **Output**: Save results to `--output_dir` if applicable

---

## Additional Resources

- **docs/DEMO_GUIDE.md** - Detailed Constitutional AI demo guide
- **docs/INDEX.md** - Complete documentation navigation
- **GETTING_STARTED.md** - Setup and first steps
- **CRITICAL_README.md** - Important project notes
- **CLAUDE.md** - Development guidelines

---

## Quick Reference

| Demo | Time | Difficulty | Topic |
|------|------|-----------|-------|
| language_model_demo.py | 15 min | Beginner | Language Models |
| translation_example.py | 20 min | Beginner | NMT |
| demo_safety.py | 5 min | Beginner | Safety |
| red_teaming_demo.py | 10 min | Intermediate | Adversarial |
| constitutional_ai_demo.py | Variable | Intermediate | RLAIF |
| model_optimization_demo.py | 30 min | Intermediate | Optimization |
| multimodal_training_demo.py | 40 min | Advanced | Multimodal |
| vision_transformer_demo.py | 35 min | Advanced | Vision |
| hardware_profiling_demo.py | 15 min | Intermediate | Profiling |

---

**Need help?** Check the main GETTING_STARTED.md guide or see docs/INDEX.md for more resources.
