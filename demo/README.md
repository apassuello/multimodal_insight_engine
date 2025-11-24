# Constitutional AI Interactive Demo

Complete implementation of the Constitutional AI Interactive Demo with both Phase 1 (SFT) and Phase 2 (RLAIF) training.

## Overview

This demo showcases the complete Constitutional AI pipeline across 6 interactive tabs:

| Tab | Description |
|-----|-------------|
| **Evaluation** | Test text against constitutional principles using AI, Regex, or HuggingFace API |
| **Phase 1 SFT** | Supervised fine-tuning with critique-revision methodology |
| **Phase 2 RLAIF** | Reinforcement Learning from AI Feedback (preference pairs, reward model, PPO) |
| **Generation** | Compare base vs trained model outputs side-by-side |
| **Impact** | Analyze training effects with comprehensive metrics |
| **Architecture** | Visualize the Constitutional AI pipeline |

## Requirements

```bash
pip install torch transformers gradio
# Optional for HuggingFace API evaluation:
pip install huggingface_hub
```

For Apple Silicon (M1/M2/M4):
- PyTorch with MPS support
- 8GB+ RAM recommended

## Quick Start

### 1. Launch the Demo

```bash
cd /home/user/multimodal_insight_engine
python -m demo.main
```

The interface will launch at `http://localhost:7860`

### 2. Load a Model

1. Select a model (default: `gpt2`)
2. Choose device preference (default: `auto` - will detect MPS/CUDA/CPU)
3. Click "Load Model"
4. Wait ~30 seconds for first-time download

### 3. Explore the 6 Tabs

#### Evaluation Tab
Test text against constitutional principles:
- Load example text or enter your own
- Choose evaluation mode:
  - **AI**: Uses the loaded model for evaluation
  - **Regex**: Fast pattern-based detection
  - **HF API**: Uses HuggingFace's toxicity classifier (requires API token)
  - **Both**: AI + Regex comparison
- See which constitutional principles are violated
- View detailed violation reports

#### Phase 1 SFT Tab
Supervised Fine-Tuning using critique-revision:
- Select training mode:
  - **Quick Demo**: 2 epochs, 20 examples (~10-15 minutes)
  - **Standard**: 5 epochs, 50 examples (~25-35 minutes)
- Click "Start Training"
- Monitor real-time progress and metrics
- View training logs and loss curves

#### Phase 2 RLAIF Tab
Reinforcement Learning from AI Feedback:
- **Step 1: Collect Preferences** - Generate response pairs and compare them
- **Step 2: Train Reward Model** - Train on preference data with Bradley-Terry loss
- **Step 3: PPO Training** - Optimize policy using the trained reward model
- Monitor each step's progress independently
- View reward model accuracy and PPO metrics

#### Generation Tab
Compare model outputs:
- Enter a prompt (or load adversarial prompt)
- Adjust temperature and max length
- Generate from both base and trained models
- See side-by-side comparison
- View evaluation scores for both outputs

#### Impact Tab
Analyze training effects:
- View before/after metrics
- Compare evaluation scores across training
- Visualize improvement trajectories
- Export analysis results

#### Architecture Tab
Visualize the pipeline:
- Interactive CAI architecture diagrams
- Phase 1 and Phase 2 flow visualization
- Component relationships

## Architecture

```
demo/
├── main.py                    # Gradio application (6-tab interface)
├── managers/
│   ├── __init__.py            # Manager exports
│   ├── model_manager.py       # Model loading and checkpointing
│   ├── multi_model_manager.py # Multi-model management for comparisons
│   ├── evaluation_manager.py  # Constitutional evaluation
│   ├── training_manager.py    # Phase 1 SFT training orchestration
│   └── comparison_engine.py   # Model comparison utilities
├── data/
│   └── test_examples.py       # Test cases and prompts
├── checkpoints/               # Saved model checkpoints
└── README.md                  # This file
```

## Key Features

### Device Detection
Automatically detects and uses the best available device:
- **MPS** (Apple Silicon): Metal Performance Shaders acceleration
- **CUDA** (NVIDIA): GPU acceleration
- **CPU**: Fallback for compatibility

### Checkpoint Management
- **Base checkpoint**: Saved immediately after model loading
- **SFT checkpoint**: Saved after Phase 1 training
- **RLAIF checkpoint**: Saved after Phase 2 training
- Enables before/after comparison in Generation tab

### Constitutional Principles
1. **Harm Prevention**: Detects harmful, dangerous, or violent content
2. **Truthfulness**: Identifies misleading or deceptive information
3. **Fairness**: Flags stereotyping and biased language
4. **Autonomy Respect**: Detects coercive or manipulative language

### Evaluation Modes
- **Regex**: Fast pattern matching (~0.1s)
- **AI**: Model-based evaluation (~2-3s)
- **HF API**: Production-grade toxicity classifier
- **Both**: Side-by-side comparison

## Performance Expectations

### Model Loading
- First load: ~30 seconds (downloads from Hugging Face)
- Cached load: <5 seconds

### Evaluation
- AI evaluation: ~2-3 seconds per text
- Regex evaluation: <0.1 seconds per text
- HF API evaluation: ~1-2 seconds per text

### Phase 1 Training (SFT)
- **Quick Demo** (2 epochs, 20 examples):
  - Data generation: ~3 minutes (3 generations per example)
  - Fine-tuning: ~5-10 minutes
  - Total: ~10-15 minutes

- **Standard** (5 epochs, 50 examples):
  - Data generation: ~7-8 minutes
  - Fine-tuning: ~15-20 minutes
  - Total: ~25-35 minutes

### Phase 2 Training (RLAIF)
- **Preference Collection**: ~5-10 minutes (50 pairs)
- **Reward Model Training**: ~3-5 minutes (3 epochs)
- **PPO Training**: ~10-15 minutes (100 steps)

### Generation
- ~3-5 seconds per generation (50-150 tokens)

## Troubleshooting

### "No model loaded" error
- Make sure to load a model first using the configuration panel
- Check that the model loaded successfully (status should show "ready")

### MPS/CUDA not detected
- Verify PyTorch is installed with MPS/CUDA support
- Check device availability: `python -c "import torch; print(torch.backends.mps.is_available())"`
- Demo will automatically fall back to CPU

### Training is slow
- Expected for GPT-2 on CPU (~2-3x slower than MPS)
- Consider using smaller model (distilgpt2)
- Use Quick Demo mode for faster iteration

### Out of memory
- Reduce batch size in training config
- Use smaller model (distilgpt2 instead of gpt2-medium)
- Close other applications to free memory

### HuggingFace API errors
- Ensure `HF_API_TOKEN` environment variable is set
- Check API quota and rate limits
- Falls back gracefully to regex evaluation

## File Locations

### Checkpoints
Saved in: `demo/checkpoints/`
- `base_gpt2/` - Base model before training
- `sft_gpt2_epochN/` - After Phase 1 SFT training
- `rlaif_gpt2_stepN/` - After Phase 2 RLAIF training

### Logs
Training logs saved to: `demo/logs/`

### Cache
Models cached by Hugging Face in: `~/.cache/huggingface/`

## Integration with Existing Code

The demo integrates with the Constitutional AI implementation:

```python
# Phase 1 (SFT)
from src.safety.constitutional.framework import ConstitutionalFramework
from src.safety.constitutional.principles import setup_default_framework
from src.safety.constitutional.model_utils import load_model, generate_text
from src.safety.constitutional.critique_revision import (
    critique_revision_pipeline,
    supervised_finetune
)

# Phase 2 (RLAIF)
from src.safety.constitutional.preference_comparison import PreferenceCollector
from src.safety.constitutional.reward_model import RewardModel, train_reward_model
from src.safety.constitutional.ppo_trainer import PPOTrainer

# Evaluation
from src.safety.constitutional.hf_api_evaluator import HFAPIEvaluator
```

## Support

For issues or questions:
1. Check this README
2. Review architecture document: `docs/demo/DEMO_ARCHITECTURE.md`
3. See implementation docs: `docs/CONSTITUTIONAL_AI_IMPLEMENTATION.md`

## Success Criteria

The demo is complete when:
- [x] Loads GPT-2 on MPS device in <30 seconds
- [x] Evaluates text with AI in <3 seconds
- [x] Compares AI vs Regex side-by-side
- [x] HuggingFace API evaluation integration
- [x] Completes Phase 1 SFT training
- [x] Collects preference pairs for RLAIF
- [x] Trains reward model on preferences
- [x] Runs PPO training with reward model
- [x] Saves checkpoints at each stage
- [x] Compares base vs trained model outputs
- [x] Shows impact analysis metrics
- [x] Displays architecture visualization
