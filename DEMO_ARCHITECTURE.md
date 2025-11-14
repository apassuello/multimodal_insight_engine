# Constitutional AI Interactive Demo
## Architecture & Specification Document

---

## Document Revision History

**Version 2.0** - Critical Issues Addressed (Post-Architectural Audit)

This revision addresses critical issues identified in the architectural audit:

**✅ Fixed: Critical Issue #1 - Metrics Mismatch**
- **Problem**: Spec described separate "Critique Loss" and "Revision Loss" tracking, but `supervised_finetune()` only tracks single combined loss
- **Solution**: Updated FR4.3 and VR5.3 to reflect single "Training Loss" metric matching actual implementation
- **Impact**: Metrics tracking now accurately reflects the training pipeline

**✅ Fixed: Critical Issue #2 - Undefined Alignment Score**
- **Problem**: "Alignment score" referenced throughout spec but never defined or calculated
- **Solution**: Added complete alignment score definition in FR5.2 with calculation formula, interpretation guide, and usage examples
- **Impact**: Provides quantitative metric for measuring training effectiveness (0.0-1.0 scale)

**✅ Fixed: Critical Issue #3 - Model Checkpoint Strategy**
- **Problem**: Unclear how to maintain separate base/trained models when `supervised_finetune()` modifies in-place
- **Solution**: Added detailed checkpoint strategy in FR4.4 with pseudo-code showing save/load workflow
- **Impact**: Enables before/after comparison by checkpointing base model before training

**✅ Fixed: Critical Issue #4 - Comparison Engine Unspecified**
- **Problem**: `managers/comparison_engine.py` listed in structure but had no implementation specification
- **Solution**: Added complete TA6 section with API design, data structures, algorithm, and usage example
- **Impact**: Full specification for quantifying base vs. trained model improvements

**✅ Updated: Performance Estimates (Realistic)**
- **Problem**: Training time estimates were overly optimistic (5-10 min for 2 epochs)
- **Solution**: Updated all time estimates to reflect 3 generations per example (initial/critique/revision)
  - Quick Demo: 2 epochs, 20 examples → 10-15 minutes (was 5-10 min)
  - Standard: 5 epochs, 50 examples → 25-35 minutes (was 15-20 min)
- **Impact**: Sets realistic expectations for demo timing

**✅ Adjusted: MPS-Specific Requirements**
- **Problem**: Spec required real-time GPU memory/utilization graphs, but MPS doesn't expose these APIs
- **Solution**: Updated VR2.1, VR2.3, and success criteria to use system RAM monitoring via `psutil`
- **Impact**: Requirements now match macOS/MPS technical capabilities

---

## Executive Summary

This document specifies the requirements, architecture, and design guidelines for an interactive web-based demonstration of the Constitutional AI (CAI) evaluation and training system. The demo will showcase the complete pipeline from principle-based evaluation to model training and behavioral improvement, utilizing real language models on Apple Silicon (M4-Pro) hardware.

**Primary Objectives:**
1. Demonstrate AI-based evaluation superiority over regex-based approaches
2. Prove the seamless AI-first architecture implementation
3. Show quantifiable behavioral improvements through Constitutional AI training
4. Provide interactive exploration of the complete CAI pipeline

**Target Audience:** Technical stakeholders, ML researchers, product managers, potential users of the CAI system

**Technology Constraints:** Must leverage M4-Pro (48GB RAM, MPS acceleration), support models <1B parameters, provide both real and mock modes for different use cases.

---

## Goals & Objectives

### Primary Goals

**G1: Validation of Implementation**
- Prove the AI-first architecture works as specified
- Demonstrate backward compatibility is maintained
- Show all four constitutional principles function correctly
- Validate the complete training pipeline integration

**G2: Demonstration of Value**
- Showcase AI evaluation detecting nuanced violations that regex misses
- Quantify improvement in model behavior post-training
- Illustrate real-world applicability of Constitutional AI methodology
- Provide concrete before/after comparisons

**G3: Educational Impact**
- Make Constitutional AI concepts accessible and understandable
- Show the complete pipeline from evaluation to improved model
- Enable hands-on exploration of the system
- Build confidence in the implementation quality

**G4: Practical Utility**
- Support multiple usage modes (quick demo vs. in-depth exploration)
- Enable configuration for different hardware capabilities
- Allow export of trained models and results
- Provide reusable test suites

### Success Metrics

- Demo runs successfully on M4-Pro with real models
- Training shows measurable improvement (>40% increase in alignment scores)
- Side-by-side comparisons clearly illustrate AI vs. regex differences
- UI is intuitive and requires minimal explanation
- Complete training cycle completes in reasonable time (<15 minutes for Quick Demo, <35 minutes for Standard)

---

## Functional Requirements

### FR1: Model Management

**FR1.1: Model Selection**
- Support multiple pre-trained models: GPT-2 Small (124M), GPT-2 Medium (355M), DistilGPT-2 (82M)
- Enable dynamic model loading without application restart
- Display model status (not loaded, loading, ready, training)
- Cache loaded models to avoid reloading

**FR1.2: Device Management**
- Auto-detect available devices (MPS, CUDA, CPU)
- Allow manual device selection
- Display device utilization metrics (memory, compute)
- Gracefully handle device failures with fallback to CPU

**FR1.3: Model State Tracking**
- Distinguish between base (untrained) and trained model states
- Enable checkpoint saving and loading
- Support multiple trained model versions
- Allow comparison across different training runs

### FR2: Principle Evaluation System

**FR2.1: Single Text Evaluation**
- Accept arbitrary text input (up to 2000 characters)
- Evaluate against all four constitutional principles:
  - Harm Prevention
  - Truthfulness
  - Fairness (Stereotyping)
  - Autonomy Respect
- Support three evaluation modes:
  - AI-only evaluation
  - Regex-only evaluation (fallback)
  - Side-by-side comparison
- Display detailed results per principle with reasoning

**FR2.2: Batch Evaluation**
- Evaluate multiple texts from predefined test suites
- Calculate aggregate statistics (pass rate, flagged rate)
- Support custom test suite creation
- Export batch results in structured format

**FR2.3: Comparative Analysis**
- Run identical text through AI and regex methods
- Highlight discrepancies (AI caught, regex missed)
- Quantify superiority metrics
- Provide specific examples of nuanced detection

### FR3: Text Generation & Comparison

**FR3.1: Prompted Generation**
- Accept prompt input for text generation
- Generate from base (untrained) model
- Generate from trained model (after training)
- Display generation parameters (temperature, max length, etc.)

**FR3.2: Before/After Analysis**
- Generate identical prompts from both model versions
- Automatically evaluate both outputs
- Calculate improvement delta
- Highlight specific behavioral changes

**FR3.3: Adversarial Prompts**
- Include predefined adversarial prompts designed to elicit violations
- Test suite targeting each constitutional principle
- Demonstrate training effectiveness on challenging cases

### FR4: Constitutional Training Pipeline

**FR4.1: Training Configuration**
- Select training mode:
  - Quick Demo: 2 epochs, 20 examples (~10-15 minutes)
  - Standard: 5 epochs, 50 examples (~25-35 minutes)
  - Full: Custom epochs (1-20), custom dataset size (50-500)
- **Time Estimation Note**: Each training example requires 3 generation passes (initial response → critique → revision), plus forward/backward passes during fine-tuning. Actual time depends on model size and hardware.
- Configure hyperparameters:
  - Learning rate
  - Batch size
  - Gradient accumulation steps
- Select which principles to enforce during training

**FR4.2: Training Execution**
- Start/pause/stop training
- Display real-time progress with:
  - Current epoch/total epochs
  - Steps completed/total steps
  - Estimated time remaining
  - Current batch being processed
- Handle training errors gracefully with clear messaging

**FR4.3: Metrics Tracking**
- Display live training metrics:
  - Training loss (average per epoch, descending trend expected)
  - Samples processed (running count)
  - Learning rate (if using scheduler)
  - Alignment score (evaluated on validation set per epoch, ascending trend expected)
- Update visualization at epoch boundaries (not per-step to avoid overhead)
- Log all metrics for post-training analysis

**Note**: The current `supervised_finetune()` implementation tracks a single combined loss value per batch. The training process uses critique-revision data generation (pre-training) followed by standard supervised fine-tuning on the improved responses.

**FR4.4: Checkpoint Management**
- Automatically save checkpoints at epoch boundaries
- Allow manual checkpoint saves
- Enable loading from previous checkpoints
- Display checkpoint metadata (epoch, timestamp, metrics)

**Checkpoint Strategy for Before/After Comparison**:
The demo requires separate base and trained models for comparison. Since `supervised_finetune()` modifies the model in-place, implement this strategy:

1. **Initial Load**: Load base model (e.g., GPT-2) → Save as `checkpoints/base_model/`
2. **Training**: Clone or keep base checkpoint, train model → Save as `checkpoints/trained_epoch_N/`
3. **Comparison Mode**:
   - Load `base_model` checkpoint for "before" generation
   - Load `trained_epoch_N` checkpoint for "after" generation
   - Both models can be kept in memory if RAM permits (requires ~1-2GB for GPT-2 Small × 2)
4. **Memory Optimization**: If memory is constrained, swap between checkpoints on-demand (slower but feasible)

**Implementation Detail**:
```python
# Pseudo-code for checkpoint strategy
base_model_path = "demo/checkpoints/base_gpt2"
trained_model_path = "demo/checkpoints/trained_gpt2_epoch5"

# Save base before training
save_checkpoint(model, tokenizer, base_model_path)

# Train (modifies model in-place)
trained_result = supervised_finetune(model, ...)

# Save trained
save_checkpoint(model, tokenizer, trained_model_path)

# For comparison: reload both
base_model = load_checkpoint(base_model_path)
trained_model = load_checkpoint(trained_model_path)
```

### FR5: Impact Analysis & Visualization

**FR5.1: Principle-Specific Impact**
- Run comprehensive test suite on both base and trained models
- Calculate per-principle improvement:
  - Harm Prevention: % reduction in harmful content
  - Fairness: % reduction in stereotyping
  - Truthfulness: % improvement in verifiable claims
  - Autonomy: % reduction in manipulative language
- Display improvement as delta and percentage

**FR5.2: Aggregate Statistics**
- Overall alignment improvement (see Alignment Score definition below)
- Total violations before/after
- Weighted scoring changes
- Confidence intervals on improvements

**Alignment Score Definition**:
The alignment score is a quantitative metric measuring how well model outputs adhere to constitutional principles. It ranges from 0.0 (completely misaligned) to 1.0 (perfectly aligned).

**Calculation Method**:
```python
def calculate_alignment_score(
    outputs: List[str],
    framework: ConstitutionalFramework
) -> float:
    """
    Calculate alignment score for a set of model outputs.

    Args:
        outputs: List of generated texts to evaluate
        framework: Constitutional framework with principles

    Returns:
        Alignment score between 0.0 and 1.0
    """
    total_weighted_violations = 0.0
    max_possible_score = 0.0

    for output in outputs:
        result = framework.evaluate_text(output)

        # Sum weighted scores for all flagged principles
        # Higher weighted_score = worse alignment
        total_weighted_violations += result['weighted_score']

        # Maximum possible violation score
        max_possible_score += sum(p.weight for p in framework.principles.values())

    # Invert: 0 violations = score 1.0, max violations = score 0.0
    if max_possible_score > 0:
        violation_ratio = total_weighted_violations / max_possible_score
        alignment_score = 1.0 - violation_ratio
    else:
        alignment_score = 1.0  # No principles to violate

    return max(0.0, min(1.0, alignment_score))  # Clamp to [0, 1]
```

**Interpretation**:
- **1.0**: Perfect alignment - no principle violations detected
- **0.8-1.0**: Strong alignment - minor or infrequent violations
- **0.6-0.8**: Moderate alignment - some violations present
- **0.4-0.6**: Weak alignment - frequent violations
- **<0.4**: Poor alignment - severe or widespread violations

**Usage in Demo**:
- Evaluate alignment on a validation set (10-20 prompts) after each training epoch
- Display alignment trend: "Epoch 1: 0.45 → Epoch 5: 0.82 (+82% improvement)"
- Success criterion: >40% relative improvement (e.g., 0.50 → 0.70 or better)

**FR5.3: Visual Analytics**
- Loss curves over training (interactive charts)
- Before/after comparison bar charts
- Per-principle radar charts showing coverage
- Example-level drill-down (click to see specific cases)

### FR6: Architecture Demonstration

**FR6.1: Code Examples**
- Show actual usage patterns of the AI-first API
- Demonstrate backward compatibility with code snippets
- Highlight key architectural decisions
- Provide copy-paste ready examples

**FR6.2: System Overview**
- Visual diagram of the complete pipeline
- Data flow illustration
- Component interaction explanation
- Link to source code and documentation

### FR7: Operating Modes

**FR7.1: Real Model Mode**
- Uses actual pre-trained models
- Performs real training with gradient updates
- Generates authentic text completions
- Provides ground-truth results

**FR7.2: Mock Mode (Fast Development)**
- Uses simulated model responses
- Instant "training" with predetermined improvements
- Predictable, deterministic outputs
- Enables rapid UI/UX testing without GPU

**FR7.3: Hybrid Mode**
- Real evaluation, mocked training (for quick demos)
- Configurable per-component
- Clear indicators of what's real vs. mocked

---

## Visual & UX Requirements

### VR1: Layout & Navigation

**VR1.1: Tab-Based Organization**
- Five primary tabs with clear iconography:
  1. 🎯 **Evaluation** - Single text principle evaluation
  2. 📝 **Generation** - Before/after text comparison
  3. 🔧 **Training** - Run Constitutional AI training
  4. 📊 **Impact** - Quantitative analysis and metrics
  5. 🏗️ **Architecture** - System overview and examples
- Persistent configuration panel accessible from all tabs
- Status bar showing current model, device, mode

**VR1.2: Visual Hierarchy**
- Primary actions prominently displayed (large buttons)
- Secondary actions contextually available
- Tertiary options in expandable sections
- Clear visual separation between input, process, output

### VR2: Configuration Panel

**VR2.1: Model Selection**
- Dropdown with available models (icon + name + size)
- "Load Model" button with loading indicator
- Model status badge (🔴 Not Loaded, 🟡 Loading, 🟢 Ready, 🔵 Training)
- Memory usage indicator (system RAM via `psutil`, GPU-specific unavailable on MPS)

**VR2.2: Mode Selection**
- Radio buttons for Real/Mock/Hybrid modes
- Tooltip explanations for each mode
- Visual indicator showing current mode in status bar
- Warning when switching modes with loaded model

**VR2.3: Device Selection**
- Auto-detected device with manual override
- Device capabilities display (memory, compute type)
- System memory utilization (via `psutil`)
- **Note**: MPS (Metal Performance Shaders) on macOS does not expose GPU-specific memory or utilization metrics like CUDA. Display system-wide RAM usage instead.

### VR3: Evaluation Tab

**VR3.1: Input Area**
- Large text area (4-6 lines) with character count
- Predefined example buttons (quick load test cases)
- Clear button
- Evaluation mode selector (AI / Regex / Both)

**VR3.2: Results Display**
- Card-based layout for each principle:
  ```
  ┌─────────────────────────────────────────┐
  │ 🛡️ Harm Prevention         ✅ CLEAN     │
  ├─────────────────────────────────────────┤
  │ Flagged: No                             │
  │ Method: ai_evaluation                   │
  │ Score: 0.12                             │
  │ Reasoning: "No harmful content..."      │
  └─────────────────────────────────────────┘
  ```
- Color coding: Green (clean), Red (flagged), Gray (disabled)
- Expandable reasoning sections
- Aggregated summary at bottom (weighted score, overall pass/fail)

**VR3.3: Comparison Mode (AI vs Regex)**
- Split-screen layout showing both results side-by-side
- Highlight discrepancies in yellow
- Summary box: "AI detected X additional violations"
- Example cases where AI outperformed regex

### VR4: Generation Tab

**VR4.1: Prompt Input**
- Text area for custom prompts
- Predefined adversarial prompts (dropdown)
- Generation parameters (collapsible advanced section)
- "Generate from Base" and "Generate from Trained" buttons

**VR4.2: Output Comparison**
- Side-by-side panels:
  ```
  ┌────────────────────┬────────────────────┐
  │ BEFORE TRAINING    │ AFTER TRAINING     │
  ├────────────────────┼────────────────────┤
  │ Generated text...  │ Generated text...  │
  │                    │                    │
  │ ❌ Evaluation:     │ ✅ Evaluation:     │
  │ Harm: FLAGGED      │ Harm: CLEAN        │
  │ Fairness: FLAGGED  │ Fairness: CLEAN    │
  └────────────────────┴────────────────────┘
  ```
- Diff highlighting (strikethrough for removed, underline for added conceptually)
- Evaluation badges on each output
- Improvement indicator at bottom

**VR4.3: Batch Generation**
- Run multiple adversarial prompts sequentially
- Progress indicator
- Aggregate improvement statistics
- Gallery view of all comparisons

### VR5: Training Tab

**VR5.1: Configuration Section**
- Training mode cards (visual selection):
  - Quick Demo (⚡ 10-15 min, 2 epochs, 20 examples)
  - Standard (⚙️ 25-35 min, 5 epochs, 50 examples)
  - Custom (🔧 configurable)
- Slider for custom epochs (1-20)
- Slider for dataset size (50-500)
- Advanced options (collapsible): learning rate, batch size, etc.
- Principle toggles (which principles to enforce)

**VR5.2: Training Execution**
- Large "Start Training" button (converts to "Pause"/"Stop" when active)
- Progress bar with percentage and ETA
- Current status text: "Epoch 3/5 - Step 87/150 - Processing critique..."
- Warning modal before starting (will take time, GPU usage)

**VR5.3: Live Metrics**
- Real-time updating metric cards:
  ```
  ┌─────────────────┐ ┌─────────────────┐
  │ Critique Loss   │ │ Revision Loss   │
  │ 0.234 ↓ -15%    │ │ 0.189 ↓ -22%    │
  └─────────────────┘ └─────────────────┘
  ```
- Trend indicators (↑↓ and percentage change)
- Color coding (green for improving, red for degrading)

**VR5.4: Visualization**
- Interactive line chart (loss curves over time)
- Dual y-axis (critique and revision loss)
- Epoch markers (vertical lines)
- Hover tooltips with exact values
- Zoom/pan capabilities
- Export chart as image

### VR6: Impact Tab

**VR6.1: Test Suite Selection**
- Dropdown to select predefined test suites:
  - Harmful Content (20 prompts)
  - Stereotyping & Bias (20 prompts)
  - Truthfulness (15 prompts)
  - Manipulation & Coercion (15 prompts)
  - Comprehensive (all 70 prompts)
- "Run Comparison" button
- Progress bar during batch evaluation

**VR6.2: Results Summary**
- Table view:
  ```
  ┌─────────────────┬────────┬───────┬──────────────┐
  │ Principle       │ Before │ After │ Improvement  │
  ├─────────────────┼────────┼───────┼──────────────┤
  │ Harm Prevention │  30%   │  95%  │ +65% ✅      │
  │ Fairness        │  45%   │  92%  │ +47% ✅      │
  │ Truthfulness    │  60%   │  88%  │ +28% ✅      │
  │ Autonomy        │  55%   │  90%  │ +35% ✅      │
  └─────────────────┴────────┴───────┴──────────────┘
  ```
- Visual indicators for significance (✅ >20%, ⚠️ 10-20%, ❌ <10%)
- Overall alignment score (aggregate metric)

**VR6.3: Detailed Examples**
- Expandable accordion for each test case
- Shows: prompt → base output → trained output → evaluations
- Filter options (show only improved, show only degraded, show all)
- Search/filter by keywords

**VR6.4: Export Options**
- Export results as JSON, CSV, or Markdown
- Generate PDF report with charts
- Copy to clipboard (formatted)

### VR7: Architecture Tab

**VR7.1: Overview Section**
- Visual system diagram (pipeline flow)
- Component descriptions with tooltips
- Link to full documentation

**VR7.2: API Examples**
- Code snippets with syntax highlighting
- Copy button for each snippet
- Runnable examples (execute in demo)
- Comments explaining key aspects

**VR7.3: Performance Characteristics**
- Comparison table (AI vs Regex):
  - Accuracy
  - Speed
  - Resource usage
  - Use case recommendations

### VR8: Global UX Patterns

**VR8.1: Loading States**
- Skeleton screens during initial load
- Spinners for quick operations (<3s)
- Progress bars for long operations (>3s)
- Disable controls during processing with visual feedback

**VR8.2: Error Handling**
- Non-intrusive error messages (toast notifications)
- Actionable error messages with suggestions
- Graceful degradation (fallback to CPU if MPS fails)
- Error log accessible in advanced settings

**VR8.3: Responsive Feedback**
- Immediate visual feedback on all interactions
- Hover states on clickable elements
- Active states on buttons
- Success confirmations (checkmarks, green highlights)

**VR8.4: Accessibility**
- Keyboard navigation support
- Screen reader friendly labels
- High contrast mode option
- Font size adjustment

---

## Technical Architecture

### TA1: System Components

```
┌─────────────────────────────────────────────────────────┐
│                    Gradio Web Interface                 │
├─────────────────────────────────────────────────────────┤
│  Tab Controllers (5 tabs)  │  Configuration Manager     │
│  Event Handlers            │  State Manager             │
└────────────────┬────────────────────────────────────────┘
                 │
┌────────────────┴────────────────────────────────────────┐
│                  Application Layer                      │
├─────────────────────────────────────────────────────────┤
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐ │
│  │   Model      │  │  Evaluation  │  │   Training   │ │
│  │   Manager    │  │   Manager    │  │   Manager    │ │
│  └──────────────┘  └──────────────┘  └──────────────┘ │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐ │
│  │  Generation  │  │  Comparison  │  │     Mock     │ │
│  │   Manager    │  │   Engine     │  │    Mode      │ │
│  └──────────────┘  └──────────────┘  └──────────────┘ │
└────────────────┬────────────────────────────────────────┘
                 │
┌────────────────┴────────────────────────────────────────┐
│                Core Implementation Layer                │
├─────────────────────────────────────────────────────────┤
│  src/safety/constitutional/                             │
│  ├── framework.py          (ConstitutionalFramework)    │
│  ├── principles.py         (Evaluation functions)       │
│  ├── critique_revision.py  (Training pipeline)          │
│  └── model_utils.py        (Model loading/inference)    │
└────────────────┬────────────────────────────────────────┘
                 │
┌────────────────┴────────────────────────────────────────┐
│                Infrastructure Layer                     │
├─────────────────────────────────────────────────────────┤
│  PyTorch / Transformers / MPS Backend                   │
│  Model Checkpoints / Dataset Cache / Metrics Storage    │
└─────────────────────────────────────────────────────────┘
```

### TA2: Data Flow

**Evaluation Flow:**
```
User Input → Tokenize → Load Model → Evaluate with AI/Regex
→ Generate Results → Format for Display → Render in UI
```

**Training Flow:**
```
Start Training → Load Base Model → Initialize Optimizer
→ For each epoch:
    → Load Dataset Batch
    → Critique Phase (identify violations)
    → Revision Phase (generate improvements)
    → Calculate Loss → Backward Pass → Update Weights
    → Log Metrics → Update UI
→ Save Checkpoint → Display Completion
```

**Comparison Flow:**
```
Prompt Input → Generate from Base Model → Evaluate Output
             → Generate from Trained Model → Evaluate Output
             → Calculate Delta → Format Comparison → Display
```

### TA3: State Management

**Application State (Global):**
- `current_model`: Loaded model reference (or None)
- `model_name`: String identifier of loaded model
- `device`: Current compute device (mps/cuda/cpu)
- `mode`: Operating mode (real/mock/hybrid)
- `base_checkpoint_path`: Path to base model checkpoint
- `trained_checkpoint_path`: Path to trained model checkpoint (or None)
- `training_active`: Boolean flag for training state
- `training_config`: Dictionary of training hyperparameters

**Component State (Scoped):**
- `evaluation_results`: Last evaluation results (per tab)
- `generation_outputs`: Last generated texts (before/after)
- `training_metrics`: List of metric dictionaries per step
- `comparison_results`: Batch comparison statistics

**State Persistence:**
- Save checkpoints to `demo/checkpoints/`
- Cache models in `demo/cache/`
- Log metrics to `demo/logs/`
- Export results to `demo/exports/`

### TA4: Model Management Strategy

**Caching:**
- First model load downloads and caches (Hugging Face cache)
- Subsequent loads read from cache (fast)
- Track cache size and allow clearing

**Memory Management:**
- Unload model when switching (free memory)
- Option to keep both base and trained in memory (if sufficient RAM)
- Monitor memory usage and warn before OOM

**Checkpoint Strategy:**
- Base checkpoint: saved before any training (read-only)
- Training checkpoints: saved every epoch in `demo/checkpoints/epoch_N/`
- Best checkpoint: saved when validation metrics improve (if implemented)
- Allow loading any historical checkpoint

### TA5: Performance Considerations

**Optimization Targets:**
- Model loading: <30 seconds (first load), <5 seconds (cached)
- Single evaluation: <2 seconds (AI), <0.1 seconds (regex)
- Text generation: <5 seconds for 50 tokens
- Training (realistic): 2 epochs in ~10-15 minutes, 5 epochs in ~25-35 minutes
  - Note: Each example requires 3 generations (initial, critique, revision) @ ~3s each = ~9s + fine-tuning overhead
  - Quick Demo (20 examples): 20 × 9s = 180s generation + ~300s fine-tuning ≈ 8-15 min
- UI responsiveness: <100ms for all interactions (excluding compute)

**Acceleration:**
- Use MPS backend for M4-Pro (Metal Performance Shaders)
- Batch evaluations where possible
- Cache tokenized inputs for repeated use
- Use mixed precision if supported (float16)

**Scalability:**
- Support models up to 1B parameters
- Handle datasets up to 500 examples
- Track up to 1000 metric points
- Store up to 10 checkpoints before cleanup

### TA6: Comparison Engine Specification

The Comparison Engine (`managers/comparison_engine.py`) is responsible for quantifying improvements between base and trained models.

**Core Responsibilities:**
1. Generate outputs from both models on identical test suite
2. Evaluate all outputs using Constitutional Framework
3. Calculate per-principle and aggregate improvements
4. Provide detailed drill-down data for UI display

**API Design:**
```python
@dataclass
class ComparisonResult:
    """Results from comparing base vs. trained model."""
    test_suite_name: str
    num_prompts: int

    # Per-principle metrics
    principle_results: Dict[str, PrincipleComparison]

    # Aggregate metrics
    overall_alignment_before: float
    overall_alignment_after: float
    alignment_improvement: float  # Percentage improvement

    # Example outputs for drill-down
    examples: List[ExampleComparison]

@dataclass
class PrincipleComparison:
    """Comparison for a single principle."""
    principle_name: str
    violations_before: int
    violations_after: int
    improvement_pct: float

@dataclass
class ExampleComparison:
    """Single prompt comparison."""
    prompt: str
    base_output: str
    trained_output: str
    base_evaluation: Dict[str, Any]
    trained_evaluation: Dict[str, Any]
    improved: bool  # True if trained is better

class ComparisonEngine:
    def __init__(self, framework: ConstitutionalFramework):
        self.framework = framework

    def compare_models(
        self,
        base_model,
        base_tokenizer,
        trained_model,
        trained_tokenizer,
        test_suite: List[str],
        device: torch.device,
        generation_config: GenerationConfig
    ) -> ComparisonResult:
        """
        Compare base and trained models on test suite.

        Algorithm:
        1. For each prompt in test_suite:
           a. Generate from base_model
           b. Generate from trained_model
           c. Evaluate both outputs with framework
           d. Store results

        2. Aggregate results:
           a. Count violations per principle (before/after)
           b. Calculate alignment scores (before/after)
           c. Compute improvement percentages

        3. Return structured ComparisonResult

        Performance: ~2-3 seconds per prompt (2 generations + 2 evaluations)
        Memory: Minimal (sequential processing, no batching needed)
        """
        pass
```

**Implementation Considerations:**
- **Sequential vs. Parallel**: Process prompts sequentially to avoid memory spikes
- **Error Handling**: Skip failed generations, log errors, continue comparison
- **Progress Tracking**: Yield progress updates for UI (via callbacks or generators)
- **Caching**: Cache evaluation results keyed by (model_checkpoint, prompt, text_hash)

**Usage Example:**
```python
# In Impact tab
engine = ComparisonEngine(framework)
result = engine.compare_models(
    base_model, base_tokenizer,
    trained_model, trained_tokenizer,
    test_suite=HARMFUL_CONTENT_SUITE,
    device=device,
    generation_config=GenerationConfig(...)
)

print(f"Alignment improvement: {result.alignment_improvement:.1f}%")
print(f"Harm Prevention: {result.principle_results['harm_prevention'].improvement_pct:.1f}%")
```

---

## Key Considerations

### KC1: Hardware Constraints

**M4-Pro Specifications:**
- 48GB unified memory (shared between CPU and GPU)
- MPS acceleration (Metal)
- No CUDA support

**Implications:**
- Use `device='mps'` for GPU acceleration
- Monitor unified memory usage (model + data + activations)
- Test with largest target model (1B params) to ensure fit
- Provide CPU fallback if MPS initialization fails

### KC2: Model Selection Criteria

**Recommended Models:**
1. **GPT-2 Small (124M)** - Fast, fits easily, good for demos
2. **GPT-2 Medium (355M)** - Balanced performance/quality
3. **DistilGPT-2 (82M)** - Fastest, educational purposes
4. **GPT-2 Large (774M)** - High quality, slower (optional)

**Selection Factors:**
- Parameter count (smaller = faster)
- Pre-training data quality
- Tokenizer compatibility
- Community usage (debugging support)

**Out of Scope:**
- GPT-2 XL (1.5B) - too large, diminishing returns
- Domain-specific models - focus on general language models
- Encoder-only models (BERT) - CAI requires generation capability

### KC3: Training Data & Examples

**Dataset Requirements:**
- Diverse examples covering all four principles
- Mix of explicit and nuanced violations
- Balanced distribution across principles
- Real-world relevance (not artificial/toy examples)

**Test Suites:**
- **Harmful Content**: Physical harm, psychological harm, dangerous advice
- **Stereotyping**: Gender, race, nationality, age, occupation
- **Truthfulness**: False claims, unverifiable statements, misleading framing
- **Autonomy**: Commands, manipulation, pressure tactics, false dichotomies

**Quality Criteria:**
- Clear ground truth (obvious what constitutes violation)
- Varied difficulty (easy/medium/hard for AI to detect)
- Realistic (plausible user-generated content)
- Non-controversial labeling (avoid edge cases in demo)

### KC4: User Experience Priorities

**Primary UX Goals:**
1. **Clarity**: Users understand what's happening at each step
2. **Confidence**: Results are trustworthy and reproducible
3. **Efficiency**: Common tasks require minimal clicks/time
4. **Exploration**: Users can easily experiment and learn

**Design Principles:**
- Show don't tell (visualizations > text explanations)
- Progressive disclosure (simple by default, advanced available)
- Immediate feedback (no silent operations)
- Forgiving (undo, reset, clear options)

**Anti-Patterns to Avoid:**
- Hidden operations (always show what's executing)
- Ambiguous states (clear loading/ready/error indicators)
- Jargon without explanation (tooltip for technical terms)
- Dead ends (always provide next action suggestions)

### KC5: Error Scenarios & Handling

**Expected Errors:**
- Model download failure (network issues)
- Out of memory (model too large for device)
- MPS initialization failure (Metal not available)
- Training divergence (loss explodes)
- Invalid input (empty text, non-UTF8 characters)

**Handling Strategy:**
- Graceful degradation (try MPS → CUDA → CPU)
- Clear error messages with suggested fixes
- Automatic retry for transient failures
- Preserve user data on error (don't clear inputs)
- Log errors for debugging (accessible in UI)

### KC6: Testing & Validation

**Pre-Launch Testing:**
- Smoke test: Load model, evaluate, train 1 epoch, compare
- Performance test: Full 5-epoch training, measure time/memory
- Error testing: Trigger each error scenario, verify handling
- UI testing: Navigate all tabs, test all interactions
- Model comparison: Verify base vs. trained shows improvement

**Acceptance Criteria:**
- All tabs functional with real model
- Training completes without errors
- Before/after comparison shows improvement
- No crashes or freezes during normal operation
- UI remains responsive during training

### KC7: Documentation & Onboarding

**In-App Guidance:**
- Tooltip on every control explaining purpose
- "First Time?" tutorial mode (optional walkthrough)
- Example scenarios with expected results
- Link to comprehensive documentation

**External Documentation:**
- README with setup instructions
- Architecture document (this document)
- API reference for code examples
- Troubleshooting guide

### KC8: Extensibility

**Future Enhancements (Out of Initial Scope):**
- Additional constitutional principles (user-defined)
- Multi-model comparison (run multiple models simultaneously)
- A/B testing framework (compare training approaches)
- Integration with external APIs (OpenAI, Anthropic for comparison)
- Fine-tuning on custom datasets (upload your own)

**Design Considerations:**
- Modular architecture (easy to add new tabs/features)
- Plugin system for new principles
- Config-driven test suites (JSON format)
- Extensible metrics tracking

---

## Success Criteria

### Functional Success

**Critical (Must Have):**
- ✅ Load and run GPT-2 model on M4-Pro with MPS acceleration
- ✅ Perform AI-based evaluation on all four constitutional principles
- ✅ Complete full training cycle (5 epochs) in <30 minutes
- ✅ Generate before/after comparison showing measurable improvement
- ✅ Display live training metrics with visualizations
- ✅ Switch between real and mock modes without errors

**Important (Should Have):**
- ✅ Batch evaluation on test suites with aggregate statistics
- ✅ Export trained models and results
- ✅ Comparison view showing AI vs. regex side-by-side
- ✅ Multiple model options (GPT-2 Small/Medium/Distil)
- ✅ Checkpoint management (save/load/resume)

**Nice to Have (Could Have):**
- ⭕ Real-time system memory graphs (GPU-specific metrics not available on MPS)
- ⭕ Advanced hyperparameter tuning interface
- ⭕ Custom test suite upload
- ⭕ Multi-language support (for constitutional principles)

### Visual Success

- ✅ Clean, modern UI that doesn't require documentation
- ✅ Intuitive navigation (users find features without guidance)
- ✅ Responsive during long operations (progress indicators)
- ✅ Accessible color schemes (sufficient contrast)
- ✅ Mobile-friendly layout (bonus, not required)

### Performance Success

- ✅ UI interactions respond in <100ms
- ✅ Single evaluation completes in <3 seconds
- ✅ Quick training mode (2 epochs, 20 examples) completes in <15 minutes
- ✅ Standard training mode (5 epochs, 50 examples) completes in <35 minutes
- ✅ Memory usage stays within 40GB (with 48GB available)
- ✅ No memory leaks during extended sessions

### Educational Success

- ✅ Non-experts can understand Constitutional AI from demo
- ✅ Technical stakeholders can validate implementation quality
- ✅ Provides concrete examples for documentation/papers
- ✅ Enables reproducible results for further research

---

## Out of Scope

**Explicitly Excluded:**
- Production deployment infrastructure (this is a demo, not a product)
- User authentication or multi-user support
- Cloud/remote execution (local only)
- Integration with external LLM APIs (OpenAI, etc.) except for optional comparison
- Support for non-English languages
- Video/audio modality support
- Real-time streaming generation (batch only)
- Distributed training across multiple GPUs
- Model compression or quantization (use full precision models)
- Commercial deployment features (rate limiting, billing, etc.)

---

## Implementation Guidelines

### IG1: Code Organization

```
demo/
├── README.md                          # Setup and usage instructions
├── requirements.txt                   # Python dependencies
├── config.yaml                        # Configuration file
├── main.py                            # Gradio app entry point
├── managers/                          # Business logic layer
│   ├── __init__.py
│   ├── model_manager.py               # Model loading/caching
│   ├── evaluation_manager.py          # Evaluation orchestration
│   ├── training_manager.py            # Training orchestration
│   ├── generation_manager.py          # Text generation
│   └── comparison_engine.py           # Before/after analysis
├── ui/                                # UI components
│   ├── __init__.py
│   ├── tabs/
│   │   ├── evaluation_tab.py
│   │   ├── generation_tab.py
│   │   ├── training_tab.py
│   │   ├── impact_tab.py
│   │   └── architecture_tab.py
│   ├── components/
│   │   ├── config_panel.py
│   │   ├── metric_card.py
│   │   └── result_display.py
│   └── theme.py                       # Gradio theme customization
├── mock/                              # Mock mode implementation
│   ├── __init__.py
│   ├── mock_model.py
│   └── mock_responses.py
├── data/                              # Test suites and examples
│   ├── test_suites/
│   │   ├── harmful_content.json
│   │   ├── stereotyping.json
│   │   ├── truthfulness.json
│   │   └── autonomy.json
│   └── examples/
│       └── adversarial_prompts.json
├── assets/                            # Static assets
│   ├── diagrams/
│   └── icons/
├── checkpoints/                       # Saved model checkpoints
├── logs/                              # Metric logs and debug info
└── exports/                           # User-exported results
```

### IG2: Key Design Patterns

**Manager Pattern:**
- Each major function has a dedicated manager class
- Managers handle business logic, not UI concerns
- Managers are stateless (state passed as parameters)
- Easy to test independently of UI

**Event-Driven UI:**
- Gradio event handlers delegate to managers
- Managers return structured results
- UI components format results for display
- Clear separation of concerns

**Mock/Real Strategy Pattern:**
- Common interface for real and mock implementations
- Mode selection at runtime
- No conditional logic in business code (use polymorphism)

### IG3: Error Handling Philosophy

**Fail-Fast with Recovery:**
- Validate inputs early
- Catch errors close to source
- Provide recovery options (retry, fallback)
- Never silent failure

**User-Friendly Errors:**
- Technical error → user-friendly message
- Include suggested fix ("Try: ...")
- Log full traceback for debugging
- Non-blocking notifications (toasts)

### IG4: Performance Best Practices

**Lazy Loading:**
- Don't load model until needed
- Load UI first, models on-demand
- Cache expensive computations

**Async Where Possible:**
- Long operations in background threads
- Update UI progressively
- Cancel-able operations

**Memory Management:**
- Explicitly delete models when switching
- Clear GPU cache after training
- Monitor and display memory usage

### IG5: Configuration Management

**YAML Configuration:**
```yaml
models:
  default: "gpt2"
  options:
    - name: "gpt2"
      size: "124M"
      path: "gpt2"
    - name: "gpt2-medium"
      size: "355M"
      path: "gpt2-medium"

training:
  quick_demo:
    epochs: 2
    dataset_size: 50
  standard:
    epochs: 5
    dataset_size: 100

devices:
  prefer: "mps"  # mps, cuda, cpu
  fallback: true
```

**Environment Variables:**
- `DEMO_CACHE_DIR`: Override cache location
- `DEMO_DEVICE`: Force specific device
- `DEMO_MOCK_MODE`: Start in mock mode

---

## Summary

This demo represents the culmination of the Constitutional AI implementation, providing:

1. **Validation**: Proves the entire pipeline works end-to-end
2. **Education**: Makes Constitutional AI accessible and understandable
3. **Confidence**: Shows quantifiable improvements in model behavior
4. **Usability**: Enables exploration and experimentation

**Core Innovation**: This isn't just a UI for evaluation—it demonstrates the complete Constitutional AI training methodology with real, measurable behavioral improvements in language models.

**Key Differentiator**: The ability to show before/after training comparisons with real models, not just mock demonstrations, provides unambiguous proof of implementation quality.

**Success Metric**: A technical stakeholder can run this demo and conclude, "Yes, Constitutional AI is fully implemented and effective."
