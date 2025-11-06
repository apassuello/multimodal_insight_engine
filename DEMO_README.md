# Constitutional AI Demo Scripts

This directory contains demo scripts to help you get started with Constitutional AI training.

---

## Quick Start

### 1. Quick Demo (5 minutes)

See Constitutional AI in action with minimal setup:

```bash
python examples/quick_start_demo.py
```

This interactive demo shows:
- ✅ Critique-Revision cycle
- ✅ Preference comparison
- ✅ Reward model scoring

**No long training required!** Perfect for understanding the concepts.

---

### 2. Full Pipeline Demo (30 minutes - 2 hours)

Run the complete Constitutional AI pipeline:

```bash
# Quick test (5 prompts, ~10 minutes)
python demo_constitutional_ai.py --quick-test

# Small scale (50 prompts, ~30 minutes)
python demo_constitutional_ai.py --phase both --num-prompts 50 --num-ppo-steps 10

# Medium scale (200 prompts, ~2 hours)
python demo_constitutional_ai.py --phase both --num-prompts 200 --num-ppo-steps 50

# Production scale (1000 prompts, ~8-12 hours)
python demo_constitutional_ai.py --phase both --num-prompts 1000 --num-ppo-steps 100
```

---

## What Do These Demos Do?

### Quick Start Demo (`examples/quick_start_demo.py`)

**Purpose**: Educational - shows how each component works

**What it does**:
1. Loads a small model (GPT-2)
2. Demonstrates critique-revision on one example
3. Shows preference comparison between two responses
4. Demonstrates reward model scoring

**Time**: ~5 minutes
**Output**: Console output showing each step
**Training**: None (uses pre-trained models for demonstration)

---

### Full Pipeline Demo (`demo_constitutional_ai.py`)

**Purpose**: End-to-end training pipeline

**What it does**:

#### Phase 1: Supervised Learning
1. Load prompts (from file or use samples)
2. Generate initial responses
3. Critique each response
4. Revise based on critiques
5. Create training dataset
6. (In production) Fine-tune model on revisions

#### Phase 2: Reinforcement Learning
1. Generate multiple responses per prompt
2. Compare responses to create preference pairs
3. Train reward model on preferences
4. Run PPO training with reward model feedback
5. Save aligned model

**Time**: Varies (10 minutes to 12 hours depending on scale)
**Output**:
- `outputs/phase1/phase1_data.json` - Critique-revision data
- `outputs/phase2/phase2_data.json` - RLAIF training data
**Training**: Real training (minimal in quick mode)

---

## Command Line Options

### Full Demo (`demo_constitutional_ai.py`)

```bash
python demo_constitutional_ai.py [OPTIONS]

Options:
  --phase {1,2,both}        Which phase to run (default: both)
  --model MODEL             Base model name (default: gpt2)
  --num-prompts N           Number of training prompts (default: 50)
  --num-epochs N            SFT epochs for Phase 1 (default: 3)
  --num-ppo-steps N         PPO training steps (default: 10)
  --quick-test              Quick test mode (5 prompts, 5 steps)
  --output-dir DIR          Output directory (default: outputs)
```

### Examples

```bash
# Run only Phase 1
python demo_constitutional_ai.py --phase 1 --num-prompts 100

# Run only Phase 2 (assumes Phase 1 is done)
python demo_constitutional_ai.py --phase 2 --num-ppo-steps 50

# Quick test on GPU
python demo_constitutional_ai.py --quick-test

# Full production run
python demo_constitutional_ai.py \
    --phase both \
    --model gpt2-medium \
    --num-prompts 1000 \
    --num-epochs 5 \
    --num-ppo-steps 100 \
    --output-dir results/run1
```

---

## Output Files

### Phase 1 Output

**File**: `outputs/phase1/phase1_data.json`

```json
{
  "prompts": ["What is AI?", ...],
  "original_responses": ["AI is...", ...],
  "critiques": ["The response could be...", ...],
  "revised_responses": ["AI is the simulation...", ...],
  "model_name": "gpt2"
}
```

### Phase 2 Output

**File**: `outputs/phase2/phase2_data.json`

```json
{
  "num_preference_pairs": 100,
  "reward_model_accuracy": 0.78,
  "ppo_steps": 50,
  "final_reward": 1.234,
  "model_name": "gpt2"
}
```

---

## System Requirements

### Minimum (Quick Demo)

- **RAM**: 4 GB
- **GPU**: Not required (can run on CPU)
- **Time**: 5-10 minutes
- **Disk**: 1 GB

### Recommended (Full Training)

- **RAM**: 16 GB
- **GPU**: NVIDIA GPU with 8+ GB VRAM
- **Time**: 2-12 hours depending on scale
- **Disk**: 10 GB (for models and outputs)

### Optimal (Production)

- **RAM**: 32 GB
- **GPU**: NVIDIA A100 or similar (40+ GB VRAM)
- **Time**: 8-24 hours for full dataset
- **Disk**: 50 GB

---

## Installation

```bash
# Install required packages
pip install torch transformers tqdm

# Verify installation
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
python -c "import transformers; print(f'Transformers: {transformers.__version__}')"
```

---

## Understanding the Flow

### Visual Pipeline

```
1. Quick Start Demo:
   Input Prompt → Generate → Critique → Revise → Display
   (Educational, no training)

2. Full Pipeline Demo:

   Phase 1:
   Prompts → Generate Responses → Critique → Revise → SFT → Phase 1 Model

   Phase 2:
   Phase 1 Model → Generate Pairs → Compare → Train Reward Model → PPO → Aligned Model
```

### Data Flow

```
┌──────────┐
│ Prompts  │
└────┬─────┘
     │
     ├─────────────────────┐
     │                     │
     ▼                     ▼
┌──────────┐         ┌──────────┐
│ Phase 1  │         │ Phase 2  │
│ (SFT)    │────────▶│ (RLAIF)  │
└──────────┘         └──────────┘
     │                     │
     ▼                     ▼
┌──────────┐         ┌──────────┐
│ Revised  │         │ Aligned  │
│ Model    │         │  Model   │
└──────────┘         └──────────┘
```

---

## Tips for Best Results

### 1. Prompt Quality

**Good prompts**:
- Clear and specific
- Diverse topics
- Varied complexity
- Represent your use case

**Bad prompts**:
- Too vague ("Tell me about things")
- All identical topics
- Too simple or too complex only

### 2. Training Scale

| Prompts | Quality | Time | Use Case |
|---------|---------|------|----------|
| 5-10 | Demo | 10 min | Learning/testing |
| 50-100 | Basic | 1 hour | Proof of concept |
| 200-500 | Good | 4 hours | Development |
| 500-1000 | Better | 8 hours | Pre-production |
| 1000+ | Best | 12+ hours | Production |

### 3. Hardware Considerations

**CPU only**:
- Use `--quick-test` mode
- Expect slow training
- Works for learning

**Single GPU (8GB)**:
- Can train up to 500 prompts
- Use gpt2 or gpt2-medium
- Set batch_size=2

**Multiple GPUs or Large GPU**:
- Scale up to 1000+ prompts
- Use larger models
- Increase batch sizes

### 4. Monitoring Training

Watch for:
- **Reward model accuracy**: Target >75%
- **KL divergence**: Should stay <1.0
- **PPO rewards**: Should increase over time
- **Policy loss**: Should decrease initially

---

## Troubleshooting

### Out of Memory

```bash
# Reduce batch size
# Reduce max sequence length
# Use smaller model
# Use fewer prompts
```

### Training Too Slow

```bash
# Use GPU instead of CPU
# Reduce number of prompts
# Reduce PPO steps
# Use --quick-test mode
```

### Poor Results

```bash
# Increase number of prompts
# Train for more epochs
# Use larger base model
# Check prompt quality
```

### Import Errors

```bash
# Make sure you're in the project root
cd /path/to/multimodal_insight_engine

# Install dependencies
pip install torch transformers tqdm

# Verify imports
python -c "from src.safety.constitutional import setup_default_framework"
```

---

## Next Steps

After running the demos:

1. **Analyze Results**
   ```bash
   cat outputs/phase1/phase1_data.json | jq
   cat outputs/phase2/phase2_data.json | jq
   ```

2. **Customize for Your Use Case**
   - Add your own prompts
   - Modify constitutional principles
   - Adjust training hyperparameters

3. **Scale Up**
   - Generate 1000+ prompts
   - Train for more epochs
   - Use larger models

4. **Evaluate**
   - Test on held-out data
   - Measure safety improvements
   - Compare against baseline

---

## Example Session

```bash
$ python examples/quick_start_demo.py

╔══════════════════════════════════════════════════════════════════════════════╗
║                                                                              ║
║            CONSTITUTIONAL AI - QUICK START DEMO                              ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝

This demo will show the key components of Constitutional AI:
  1. Phase 1: Critique-Revision Cycle
  2. Phase 2: Preference Comparison
  3. Reward Model

Press Enter to start the demo...

================================================================================
PHASE 1 DEMO: CRITIQUE-REVISION CYCLE
================================================================================

[1] Loading model and constitutional framework...
Using device: cuda
✓ Loaded model: GPT-2
✓ Loaded 4 constitutional principles

[2] Generating initial response...
Prompt: What is artificial intelligence?
Original response: AI is when computers do smart things...

[3] Generating constitutional critique...
Critique: The response is overly simplistic and lacks technical accuracy...

[4] Generating improved revision...
Revised response: Artificial intelligence is a field of computer science...

✅ Phase 1 complete! The response has been improved via critique-revision.
...

$ python demo_constitutional_ai.py --quick-test

================================================================================
CONSTITUTIONAL AI TRAINING DEMO
================================================================================
Using device: cuda
GPU: NVIDIA GeForce RTX 3090

================================================================================
STEP 0: Loading/Generating Prompts
================================================================================
Quick test mode: Using 5 sample prompts
...

[Training proceeds through Phase 1 and Phase 2]

================================================================================
TRAINING COMPLETE!
================================================================================

Summary:
  Base model: gpt2
  Training prompts: 4
  Test prompts: 1
  Phase 1: ✅ Critique-revision complete
  Phase 2: ✅ RLAIF complete

Outputs saved to: outputs/
...
```

---

## Resources

- **Architecture Diagrams**: `docs/CONSTITUTIONAL_AI_ARCHITECTURE.md`
- **Implementation Details**: `docs/CONSTITUTIONAL_AI_IMPLEMENTATION_SPEC.md`
- **Verification Report**: `VERIFICATION_REPORT.md`
- **Prompt Generation Guide**: `docs/PROMPT_GENERATION_GUIDE.md`

---

## FAQ

**Q: Do I need a GPU?**
A: No, but it's much faster. CPU works for `--quick-test` mode.

**Q: How long does training take?**
A: 10 minutes (quick test) to 12 hours (production), depending on scale.

**Q: What model should I use?**
A: Start with `gpt2` for testing. Use `gpt2-medium` or `gpt2-large` for production.

**Q: How many prompts do I need?**
A: 500-1000 prompts recommended for good results.

**Q: Can I use my own prompts?**
A: Yes! Put them in `data/constitutional_prompts.json` as `{"prompts": [...]}`.

**Q: Can I customize the principles?**
A: Yes! Modify `src/safety/constitutional/principles.py`.

**Q: Where are the trained models saved?**
A: In `outputs/` directory by default. Change with `--output-dir`.

---

## Support

For issues or questions:
1. Check `VERIFICATION_REPORT.md` for implementation details
2. Review `docs/CONSTITUTIONAL_AI_ARCHITECTURE.md` for architecture
3. Open an issue on GitHub

---

## Citation

If you use Constitutional AI in your research:

```bibtex
@article{bai2022constitutional,
  title={Constitutional AI: Harmlessness from AI Feedback},
  author={Bai, Yuntao and others},
  journal={arXiv preprint arXiv:2212.08073},
  year={2022}
}
```

---

**Happy Training! 🚀**
