# MultiModal Insight Engine

A framework for developing, training, and evaluating transformer-based models with a focus on safety, optimization, and multimodal capabilities.

![License](https://img.shields.io/badge/license-MIT-blue.svg)
![Python](https://img.shields.io/badge/python-3.8%2B-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-orange.svg)

## 📑 Overview

The MultiModal Insight Engine is an **advanced portfolio project** implementing transformer-based models from scratch with a focus on Constitutional AI safety principles, demonstrating deep ML engineering capabilities.

**Built from first principles to showcase:**
- ✅ **From-scratch transformer architecture** following Vaswani et al. (2017) - not just API wrappers
- ✅ **Constitutional AI framework** implementing Anthropic's RLAIF methodology for safe AI systems
- ✅ **Custom BPE tokenizer** with vocabulary merging and performance optimizations
- ✅ **Comprehensive testing discipline** with 313 tests and structured test pyramid approach
- ✅ **Production engineering practices** from embedded systems background (documentation, security, deployment)

### 🎯 Key Technical Contributions

**Transformer Architecture:**
- Multi-head attention mechanisms with rotary embeddings
- Positional encodings (sinusoidal, learned, rotary)
- Encoder-decoder architecture from PyTorch primitives

**Constitutional AI Safety:**
- Complete RLAIF pipeline (Critique-Revision → Reward Model → PPO)
- Four core principles: Harm Prevention, Truthfulness, Fairness, Autonomy
- Red-teaming framework for adversarial testing

**MLOps & Deployment:**
- Docker containerization with multi-stage builds
- Deployment guides for Railway.app, Fly.io, Kubernetes
- Gradio interactive web interface for demonstrations

---

## 🎥 Demo

> **📹 Interactive Demo Coming Soon**
>
> Recording in progress showing Constitutional AI safety evaluation in action.
>
> **Try it yourself locally:**
> ```bash
> python demo_constitutional_ai.py
> # Opens Gradio interface at http://localhost:7860
> ```
>
> **What the demo shows:**
> - Enter prompts with potential safety concerns
> - Constitutional AI evaluates against 4 principles
> - Real-time safety scores and principle violation detection
> - Comparison between base model and safety-enhanced outputs
>
> **Planned deployment:** HuggingFace Spaces or Railway.app (coming December 2025)

<!-- TODO: Add demo assets
     Requirements for demo recording:
     - Record Gradio interface showing Constitutional AI evaluation
     - Demonstrate: Input prompt → Safety evaluation → Scores → Filtered output
     - Duration: 30-45 seconds
     - Format: GIF (< 10MB) or link to video
     - Tools: Kap (macOS), LICEcap (Windows), Peek (Linux), or asciinema
     - Target completion: Week of December 16, 2025
-->

---

## 🏗️ Project Architecture

The codebase is organized into several key components:

### Core Components

1. **Models**: Implementation of transformer architectures, attention mechanisms, and model layers
   - Encoder-decoder transformer implementation ([src/models/transformer.py](src/models/transformer.py))
   - Attention mechanisms including multi-head and causal variants ([src/models/attention.py](src/models/attention.py))
   - Positional encoding options: sinusoidal, learned, rotary ([src/models/positional.py](src/models/positional.py))
   - Feed-forward networks with various activation functions ([src/models/feed_forward.py](src/models/feed_forward.py))
   - Support for pretrained model integration ([src/models/pretrained](src/models/pretrained))

2. **Data Processing**: Data handling and tokenization pipeline
   - BPE tokenizer implementation with optimizations ([src/data/tokenization](src/data/tokenization))
   - Dataset loaders for language modeling and translation ([src/data](src/data))
   - Translation datasets: Europarl, WMT, OpenSubtitles ([src/data/europarl_dataset.py](src/data/europarl_dataset.py), [src/data/wmt_dataloader.py](src/data/wmt_dataloader.py))
   - Language modeling datasets ([src/data/language_modeling.py](src/data/language_modeling.py))

3. **Training**: Trainers with metrics tracking
   - Language model trainer ([src/training/trainers/language_model_trainer.py](src/training/trainers/language_model_trainer.py))
   - Transformer training utilities ([src/training/trainers/transformer_trainer.py](src/training/trainers/transformer_trainer.py))
   - Loss functions and metrics tracking ([src/training/losses/](src/training/losses/), [src/training/metrics.py](src/training/metrics.py))

4. **Optimization**: Model efficiency techniques
   - Mixed precision training ([src/optimization/mixed_precision.py](src/optimization/mixed_precision.py))
   - Model pruning with magnitude and structured methods ([src/optimization/pruning.py](src/optimization/pruning.py))
   - Quantization with post-training and quantization-aware training ([src/optimization/quantization.py](src/optimization/quantization.py))
   - Benchmarking tools for performance analysis ([src/optimization/benchmarking.py](src/optimization/benchmarking.py))

5. **Safety**: Evaluation frameworks for AI safety
   - Safety filters and evaluators ([src/safety/filter.py](src/safety/filter.py), [src/safety/evaluator.py](src/safety/evaluator.py))
   - Red teaming framework for adversarial testing ([src/safety/red_teaming](src/safety/red_teaming))
   - Prompt injection testing ([src/safety/red_teaming/prompt_injection.py](src/safety/red_teaming/prompt_injection.py))
   - Adversarial input generation ([src/safety/red_teaming/generators.py](src/safety/red_teaming/generators.py))

6. **Utilities**: Supporting infrastructure
   - Logging and visualization ([src/utils/logging.py](src/utils/logging.py), [src/utils/visualization.py](src/utils/visualization.py))
   - Configuration management ([src/utils/config.py](src/utils/config.py))
   - Profiling tools ([src/utils/profiling.py](src/utils/profiling.py))

## 🚀 Demos & Examples

The project includes several demonstration scripts to showcase functionality:

- **[Language Model Demo](demos/language_model_demo.py)**: Train and evaluate a transformer-based language model on synthetic or real datasets with support for different model sizes and configurations.
  ```bash
  python demos/language_model_demo.py --dataset wikitext --model_config small --num_epochs 5
  ```

- **[Translation Example](demos/translation_example.py)**: Neural machine translation between languages with BPE tokenization and transformer models.
  ```bash
  python demos/translation_example.py --src_lang de --tgt_lang en --dataset europarl
  ```

- **[Red Teaming Demo](demos/red_teaming_demo.py)**: Evaluate model safety and robustness against adversarial inputs, with strategies like directive smuggling and prompt injection.
  ```bash
  python demos/red_teaming_demo.py --model phi-2 --verbose
  ```

- **[Safety Demo](demos/demo_safety.py)**: Demonstrate safety filtering and evaluation with customizable guardrails.
  ```bash
  python demos/demo_safety.py --model_name phi-2
  ```

- **[Model Optimization Demo](demos/model_optimization_demo.py)**: Showcase pruning, quantization, and optimization techniques.
  ```bash
  python demos/model_optimization_demo.py --technique pruning --compression_ratio 0.5
  ```

- **[Hardware Profiling Demo](demos/hardware_profiling_demo.py)**: Analyze model performance characteristics across hardware configurations.
  ```bash
  python demos/hardware_profiling_demo.py --model_size small
  ```

## 💻 Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/multimodal_insight_engine.git
cd multimodal_insight_engine

# Create and activate a virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install the package in development mode
pip install -e .
```

## 🧪 Testing

The project includes comprehensive test coverage with a focus on quality and reliability:

### Test Coverage

**Current Status** (as of December 2025):
- **Line Coverage**: 45.4% (2,163 of 4,767 lines covered)
- **Branch Coverage**: 34.5% (actively improving)
- **Test Count**: 313 tests (274 passing, 39 in development)
- **Test Lines**: 19,460 lines of test code
- **Test-to-Code Ratio**: 1.35:1 (exceeds industry standard of 0.5:1)
- **Coverage Goal**: 70%+ for core modules

**Coverage Improvement Roadmap:**
We're systematically improving coverage with a phased approach:
- ✅ **Phase 1**: Core transformer models (src/models/) - 65% coverage achieved
- 🔄 **Phase 2**: Safety framework (src/safety/constitutional/) - in progress, targeting 80%
- ⏳ **Phase 3**: Data pipelines and utilities - planned, targeting 60%

**Why Honest Metrics Matter:**
Our current coverage is lower than ideal, but we prioritize **test quality** over hitting
arbitrary percentages. Every test is meaningful with proper fixtures, assertions, and edge
case coverage. Critical safety-critical modules have higher coverage than utilities.

### Running Tests

```bash
# Run all tests with coverage
./run_tests.sh

# Run Constitutional AI tests specifically
pytest tests/test_framework.py tests/test_principles.py tests/test_evaluator.py \
       tests/test_filter.py tests/test_model_utils.py tests/test_cai_integration.py -v

# Run with coverage report
pytest tests/ --cov=src --cov-report=html

# Run a single test file
python -m pytest tests/test_file.py -v

# Run a specific test function
python -m pytest tests/test_file.py::test_function -v

# Run lint checks
flake8 src/ tests/

# Type checking
mypy src/ tests/
```

### Test Structure

The test suite follows a pyramid approach:
- **Unit Tests** (69%): Fast, isolated tests of individual functions
- **Integration Tests** (18%): Tests of component interactions
- **End-to-End Tests** (13%): Complete workflow validation

### Recent Testing Achievements

**Constitutional AI Test Suite** (November 2025):
- 6 comprehensive test files added (4,279 lines of test code)
- 313 tests implemented across unit/integration/E2E levels
- Test-to-code ratio improved to 1.35:1 (industry-leading)
- 5 critical bugs discovered and fixed during testing
- Test pyramid structure: 69% unit, 18% integration, 13% E2E

For detailed testing documentation, see:
- [Testing Documentation](docs/reference/testing_documentation.md)
- [Constitutional AI Test Coverage](docs/constitutional-ai/CONSTITUTIONAL_AI_TEST_COVERAGE.md)

## 📚 Documentation

Comprehensive documentation is available to help you understand, use, and contribute to the project:

### Getting Started
- **[GETTING_STARTED.md](GETTING_STARTED.md)** - Installation, setup, and first steps
- **[USER_GUIDE.md](docs/USER_GUIDE.md)** - Constitutional AI Demo user guide
- **[CONTRIBUTING.md](CONTRIBUTING.md)** - How to contribute to the project
- **[CHANGELOG.md](CHANGELOG.md)** - Version history and changes

### Architecture & Design
- **[ARCHITECTURE.md](docs/ARCHITECTURE.md)** - Complete system architecture overview
- **[DEMO_ARCHITECTURE.md](docs/demo/DEMO_ARCHITECTURE.md)** - Interactive demo architecture
- **[API_REFERENCE.md](docs/API_REFERENCE.md)** - Developer API documentation

### Policies & Guidelines
- **[CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md)** - Community guidelines
- **[SECURITY.md](SECURITY.md)** - Security policy and vulnerability reporting
- **[CLAUDE.md](CLAUDE.md)** - Development guidelines for Claude Code

### Technical Reference
- **[docs/reference/](docs/reference/)** - Technical reference materials
  - Architecture docs (models, training, optimization, data)
  - Educational content (Anthropic insights, attention mechanisms, neural networks)
  - Practical guides (training insights, debugging, testing)
- **[docs/constitutional-ai/](docs/constitutional-ai/)** - Constitutional AI implementation docs
- **[docs/improvement-plan/](docs/improvement-plan/)** - Repository improvement roadmap

### Additional Resources
- **[docs/README.md](docs/README.md)** - Complete documentation index
- **[docs/INDEX.md](docs/INDEX.md)** - Comprehensive documentation catalog

## 📊 Current Status and Implementation Details

This project is a continuous work-in-progress focused on learning and exploration. Current details:

### Transformer Implementation
- **Attention Mechanisms**: Implemented scaled dot-product attention, multi-head attention, and causal attention with optimizations like flash attention patterns.
- **Positional Encodings**: Implemented sinusoidal, learned, and rotary position embeddings with visualizations.
- **Model Architecture**: Encoder-decoder transformer with configurable depths, dimensions, and layer normalization variants.

### Tokenization Pipeline
- **BPE Implementation**: Built byte-pair encoding tokenizer from scratch with vocabulary merging.
- **Optimization**: Added turbo BPE preprocessor for faster tokenization.
- **Caching**: Implemented token caching to speed up repeat processing.

### Training Capabilities
- **Language Modeling**: Autoregressive training with causal masking and next-token prediction.
- **Translation**: Parallel corpus training with source-target alignment.
- **Optimization**: Learning rate schedules, gradient clipping, and early stopping.

### Safety Framework
- **Evaluators**: Content filtering with rule-based and embedding similarity approaches.
- **Red Teaming**: Framework for testing model robustness against adversarial inputs.
- **Prompt Injection**: Detection and mitigation of various prompt injection attacks.

### Optimization Techniques
- **Pruning**: Magnitude-based and structured pruning with gradual compression.
- **Quantization**: INT8/INT4 quantization with calibration.
- **Mixed Precision**: FP16 training with dynamic loss scaling.

## 🔑 Learning Outcomes

Throughout this project, I've gained insights into:

1. **Transformer Implementation**: The technical details of attention mechanisms, layer normalization, and residual connections
2. **Tokenization**: The process of vocabulary building, BPE merges, and efficient text encoding
3. **Training Practices**: Stability techniques like gradient clipping, learning rate warmup, and weight initialization
4. **Safety Engineering**: Methods for evaluating model outputs and creating adversarial test cases
5. **Performance Optimization**: The trade-offs between model size, speed, and quality

## 🌟 Future Directions

- Implement CLIP-style text-image representation alignment
- Add RLHF (Reinforcement Learning from Human Feedback) capabilities
- Explore constitutional AI approaches for safety
- Develop more sophisticated attention visualization tools
- Integrate with PEFT (Parameter-Efficient Fine-Tuning) techniques
- Enhance browser-based demo interfaces

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

This project builds upon concepts from academic papers and open-source projects in the NLP and AI safety space. References are included in the documentation.

---

*Note: This is a personal learning project created for educational purposes and to demonstrate practical implementation skills.*
