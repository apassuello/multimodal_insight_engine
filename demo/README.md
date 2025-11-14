# Constitutional AI Interactive Demo

Phase 1 (MVP) implementation of the Constitutional AI Interactive Demo.

## Overview

This demo showcases the complete Constitutional AI pipeline:
- **Evaluation**: Test text against constitutional principles using AI or regex
- **Training**: Train models using critique-revision methodology
- **Generation**: Compare base vs trained model outputs

## Requirements

```bash
pip install torch transformers gradio
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

### 3. Try the Tabs

#### Evaluation Tab
- Load example text or enter your own
- Choose evaluation mode (AI, Regex, or Both)
- See which constitutional principles are violated

#### Training Tab
- Select training mode:
  - **Quick Demo**: 2 epochs, 20 examples (~10-15 minutes)
  - **Standard**: 5 epochs, 50 examples (~25-35 minutes)
- Click "Start Training"
- Monitor real-time progress and metrics

#### Generation Tab
- Enter a prompt (or load adversarial prompt)
- Adjust temperature and max length
- Compare base vs trained model outputs
- See evaluation of both generations

## Architecture

```
demo/
├── main.py                    # Gradio application
├── managers/
│   ├── model_manager.py       # Model loading and checkpointing
│   ├── evaluation_manager.py  # Constitutional evaluation
│   └── training_manager.py    # Training orchestration
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
- **Trained checkpoint**: Saved after training completion
- Enables before/after comparison in Generation tab

### Constitutional Principles
1. **Harm Prevention**: Detects harmful, dangerous, or violent content
2. **Truthfulness**: Identifies misleading or deceptive information
3. **Fairness**: Flags stereotyping and biased language
4. **Autonomy Respect**: Detects coercive or manipulative language

## Performance Expectations

### Model Loading
- First load: ~30 seconds (downloads from Hugging Face)
- Cached load: <5 seconds

### Evaluation
- AI evaluation: ~2-3 seconds per text
- Regex evaluation: <0.1 seconds per text

### Training
- **Quick Demo** (2 epochs, 20 examples):
  - Data generation: ~3 minutes (3 generations per example)
  - Fine-tuning: ~5-10 minutes
  - Total: ~10-15 minutes

- **Standard** (5 epochs, 50 examples):
  - Data generation: ~7-8 minutes
  - Fine-tuning: ~15-20 minutes
  - Total: ~25-35 minutes

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

## File Locations

### Checkpoints
Saved in: `demo/checkpoints/`
- `base_gpt2/` - Base model before training
- `trained_gpt2_epochN/` - Trained model at epoch N

### Cache
Models cached by Hugging Face in: `~/.cache/huggingface/`

## Integration with Existing Code

The demo integrates with the existing Constitutional AI implementation:

```python
from src.safety.constitutional.framework import ConstitutionalFramework
from src.safety.constitutional.principles import setup_default_framework
from src.safety.constitutional.model_utils import load_model, generate_text
from src.safety.constitutional.critique_revision import (
    critique_revision_pipeline,
    supervised_finetune
)
```

## Next Steps (Phase 2+)

Future enhancements (not in MVP):
- Impact analysis tab with comprehensive metrics
- Architecture visualization tab
- Batch evaluation with test suites
- Comparison engine for quantitative analysis
- Export results (JSON, CSV, PDF)
- Custom training configurations

## Support

For issues or questions:
1. Check this README
2. Review architecture document: `DEMO_ARCHITECTURE.md`
3. Check implementation: Phase 1 files in `demo/`

## Success Criteria

Phase 1 MVP is successful if:
- ✅ Loads GPT-2 on MPS device in <30 seconds
- ✅ Evaluates text with AI in <3 seconds
- ✅ Compares AI vs Regex side-by-side
- ✅ Completes Quick Demo training (2 epochs, 20 examples) in <15 minutes
- ✅ Saves base and trained checkpoints
- ✅ Generates from both models and compares outputs
