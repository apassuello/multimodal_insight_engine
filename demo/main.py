"""MODULE: main.py
PURPOSE: Constitutional AI Interactive Demo - Gradio Application
KEY COMPONENTS:
- Gradio web interface with 4 tabs (Evaluation, Training, Generation, Impact)
- Integration with ModelManager, EvaluationManager, TrainingManager, ComparisonEngine
- Real-time progress tracking and status updates
- Before/after model comparison with quantitative impact analysis
DEPENDENCIES: gradio, torch, typing, demo.managers, demo.data
SPECIAL NOTES: Phase 2 implementation with Impact Analysis tab (VR6)
"""

import gradio as gr
import torch
from typing import Dict, List, Any, Tuple, Optional

from demo.managers.model_manager import ModelManager, ModelStatus
from demo.managers.evaluation_manager import EvaluationManager
from demo.managers.training_manager import TrainingManager, TrainingConfig
from demo.managers.comparison_engine import ComparisonEngine, ComparisonResult
from demo.data.test_examples import (
    EVALUATION_EXAMPLES,
    get_training_prompts,
    get_adversarial_prompts,
    TRAINING_CONFIGS,
    TEST_SUITES
)

from src.safety.constitutional.principles import setup_default_framework
from src.safety.constitutional.model_utils import generate_text, GenerationConfig


# Global managers
model_manager = ModelManager()
evaluation_manager = EvaluationManager()
training_manager = TrainingManager()


# ============================================================================
# Model Management Functions
# ============================================================================

def load_model_handler(model_name: str, device_preference: str) -> Tuple[str, str]:
    """
    Handle model loading request.

    Args:
        model_name: Model identifier
        device_preference: Preferred device

    Returns:
        Tuple of (status_message, model_info)
    """
    success, message = model_manager.load_model_from_pretrained(
        model_name=model_name,
        prefer_device=device_preference if device_preference != "auto" else None
    )

    if success:
        # Initialize evaluation frameworks
        eval_success, eval_msg = evaluation_manager.initialize_frameworks(
            model=model_manager.model,
            tokenizer=model_manager.tokenizer,
            device=model_manager.device
        )

        if not eval_success:
            message += f"\n\nWarning: {eval_msg}"

        # Get model info
        info = model_manager.get_status_info()
        model_info = f"Model: {info['model_name']}\n"
        model_info += f"Device: {info['device']}\n"
        model_info += f"Parameters: {info.get('parameters', 0):,}\n"
        model_info += f"Status: {info['status']}"

        return message, model_info
    else:
        return message, "No model loaded"


# ============================================================================
# Evaluation Tab Functions
# ============================================================================

def evaluate_text_handler(
    text: str,
    mode: str
) -> Tuple[str, str]:
    """
    Handle text evaluation request.

    Args:
        text: Text to evaluate
        mode: Evaluation mode

    Returns:
        Tuple of (status_message, results_display)
    """
    if not model_manager.is_ready() and mode == "AI Evaluation":
        return "✗ Please load a model first", ""

    # Map display names to internal modes
    mode_map = {
        "AI Evaluation": "ai",
        "Regex Evaluation": "regex",
        "Both (Comparison)": "both"
    }
    internal_mode = mode_map.get(mode, "regex")

    result, success, message = evaluation_manager.evaluate_text(text, internal_mode)

    if not success:
        return message, ""

    # Format results for display
    if internal_mode == "both":
        display = format_comparison_results(result)
    else:
        display = format_evaluation_results(result)

    return message, display


def format_evaluation_results(result: Dict[str, Any]) -> str:
    """Format evaluation results for display."""
    if not result:
        return "No results"

    summary = result.get("summary", {})
    principles = result.get("principles", {})

    # Build display text
    output = "# Evaluation Results\n\n"
    output += f"**Method:** {summary.get('evaluation_method', 'unknown')}\n"
    output += f"**Evaluation Time:** {summary.get('evaluation_time', 0):.2f}s\n\n"

    if summary.get("any_flagged", False):
        output += f"**⚠️ VIOLATIONS DETECTED**\n"
        output += f"Flagged Principles: {', '.join(summary.get('flagged_principles', []))}\n"
        output += f"Weighted Score: {summary.get('weighted_score', 0):.2f}\n\n"
    else:
        output += f"**✓ NO VIOLATIONS DETECTED**\n\n"

    # Per-principle results
    output += "## Principle Details\n\n"

    for principle_name, principle_data in principles.items():
        flagged = principle_data.get("flagged", False)
        status_icon = "❌" if flagged else "✅"

        output += f"### {status_icon} {principle_name.replace('_', ' ').title()}\n"
        output += f"- **Flagged:** {flagged}\n"
        output += f"- **Weight:** {principle_data.get('weight', 1.0)}\n"

        details = principle_data.get("details", {})
        if details:
            for key, value in details.items():
                if isinstance(value, list) and len(value) > 0:
                    output += f"- **{key}:** {len(value)} found\n"
                    for item in value[:3]:  # Show first 3
                        output += f"  - {item}\n"
                elif isinstance(value, (int, float, bool)):
                    output += f"- **{key}:** {value}\n"
                elif isinstance(value, str) and value:
                    output += f"- **{key}:** {value[:200]}\n"

        output += "\n"

    return output


def format_comparison_results(result: Dict[str, Any]) -> str:
    """Format AI vs Regex comparison results."""
    if not result:
        return "No results"

    comparison = result.get("comparison", {})

    output = "# AI vs Regex Comparison\n\n"
    output += f"**Evaluation Time:** {result.get('evaluation_time', 0):.2f}s\n\n"

    output += "## Summary\n"
    output += f"- **AI Detected:** {comparison.get('ai_flagged_count', 0)} violations\n"
    output += f"- **Regex Detected:** {comparison.get('regex_flagged_count', 0)} violations\n"
    output += f"- **Agreement:** {comparison.get('agreement', 0):.1%}\n\n"

    if comparison.get('only_ai_detected'):
        output += f"**AI Advantage:** Detected {len(comparison['only_ai_detected'])} violations missed by regex\n"
        output += f"Principles: {', '.join(comparison['only_ai_detected'])}\n\n"

    if comparison.get('only_regex_detected'):
        output += f"**Regex Advantage:** Detected {len(comparison['only_regex_detected'])} violations missed by AI\n"
        output += f"Principles: {', '.join(comparison['only_regex_detected'])}\n\n"

    if comparison.get('both_flagged'):
        output += f"**Agreement:** Both detected violations in: {', '.join(comparison['both_flagged'])}\n\n"

    return output


def load_example_handler(example_name: str) -> str:
    """Load example text based on selection."""
    for example in EVALUATION_EXAMPLES:
        if example["name"] == example_name:
            return example["text"]
    return ""


# ============================================================================
# Training Tab Functions
# ============================================================================

def start_training_handler(
    training_mode: str,
    progress=gr.Progress()
) -> Tuple[str, str, str]:
    """
    Handle training start request.

    Args:
        training_mode: Training mode selection
        progress: Gradio progress tracker

    Returns:
        Tuple of (status_message, metrics_display, checkpoint_info)
    """
    if not model_manager.is_ready():
        return "✗ Please load a model first", "", ""

    if training_manager.is_training:
        return "✗ Training already in progress", "", ""

    # Get training configuration
    mode_map = {
        "Quick Demo (2 epochs, 20 examples, ~10-15 min)": "quick_demo",
        "Standard (5 epochs, 50 examples, ~25-35 min)": "standard"
    }
    mode_key = mode_map.get(training_mode, "quick_demo")
    config_dict = TRAINING_CONFIGS[mode_key]

    config = TrainingConfig(
        num_epochs=config_dict["num_epochs"],
        num_examples=config_dict["num_examples"],
        batch_size=config_dict["batch_size"],
        learning_rate=config_dict["learning_rate"],
        mode=mode_key
    )

    # Get training prompts
    training_prompts = config_dict["prompts"]

    # Setup constitutional framework
    framework = setup_default_framework(
        model=model_manager.model,
        tokenizer=model_manager.tokenizer,
        device=model_manager.device
    )

    # Progress callback
    def progress_callback(status: str, progress_pct: float):
        progress(progress_pct, desc=status)

    # Checkpoint callback
    def checkpoint_callback(epoch: int, metrics: Dict[str, Any]):
        model_manager.save_trained_checkpoint(epoch=epoch, metrics=metrics)

    # Set model to training status
    model_manager.set_status(ModelStatus.TRAINING)

    # Execute training
    result, success, message = training_manager.train_model(
        model=model_manager.model,
        tokenizer=model_manager.tokenizer,
        framework=framework,
        device=model_manager.device,
        training_prompts=training_prompts,
        config=config,
        progress_callback=progress_callback,
        checkpoint_callback=checkpoint_callback
    )

    # Reset model status
    model_manager.set_status(ModelStatus.READY)

    if success:
        # Save final trained checkpoint
        model_manager.save_trained_checkpoint(
            epoch=config.num_epochs,
            metrics=result.get("metrics", {})
        )

        # Format metrics
        metrics_display = format_training_metrics(result)

        # Checkpoint info
        checkpoint_info = f"Base checkpoint: {model_manager.base_checkpoint_path}\n"
        checkpoint_info += f"Trained checkpoint: {model_manager.trained_checkpoint_path}"

        return message, metrics_display, checkpoint_info
    else:
        return message, "", ""


def format_training_metrics(result: Dict[str, Any]) -> str:
    """Format training metrics for display."""
    output = "# Training Metrics\n\n"

    config = result.get("config", {})
    output += f"**Configuration:**\n"
    output += f"- Epochs: {config.get('num_epochs', 0)}\n"
    output += f"- Examples: {config.get('num_examples', 0)}\n"
    output += f"- Batch Size: {config.get('batch_size', 0)}\n"
    output += f"- Learning Rate: {config.get('learning_rate', 0)}\n\n"

    output += f"**Timing:**\n"
    output += f"- Data Generation: {result.get('data_generation_time', 0):.1f}s\n"
    output += f"- Fine-tuning: {result.get('training_time', 0):.1f}s\n"
    output += f"- Total: {result.get('total_time', 0):.1f}s\n\n"

    metrics = result.get("metrics", {})
    losses = metrics.get("losses", [])

    if losses:
        output += f"**Loss Progress:**\n"
        for i, (epoch, loss) in enumerate(zip(metrics.get("epochs", []), losses)):
            output += f"- Epoch {epoch}: {loss:.4f}\n"

        output += f"\n**Improvement:** {losses[0]:.4f} → {losses[-1]:.4f} "
        improvement_pct = ((losses[0] - losses[-1]) / losses[0] * 100) if losses[0] > 0 else 0
        output += f"({improvement_pct:+.1f}%)\n"

    return output


# ============================================================================
# Generation Tab Functions
# ============================================================================

def generate_comparison_handler(
    prompt: str,
    temperature: float,
    max_length: int
) -> Tuple[str, str, str, str]:
    """
    Generate text from both base and trained models for comparison.

    Args:
        prompt: Input prompt
        temperature: Generation temperature
        max_length: Maximum generation length

    Returns:
        Tuple of (base_output, trained_output, base_eval, trained_eval)
    """
    if not model_manager.can_compare():
        error_msg = "✗ Need both base and trained checkpoints for comparison"
        return error_msg, error_msg, "", ""

    base_model = None
    base_tokenizer = None

    try:
        # Load base model
        base_model, base_tokenizer, success, msg = model_manager.load_checkpoint(
            model_manager.base_checkpoint_path
        )
        if not success:
            return msg, msg, "", ""

        # Load trained model (current model should be trained)
        trained_model = model_manager.model
        trained_tokenizer = model_manager.tokenizer

        # Generation config
        gen_config = GenerationConfig(
            max_length=max_length,
            temperature=temperature,
            do_sample=True
        )

        # Generate from base
        base_output = generate_text(
            base_model,
            base_tokenizer,
            prompt,
            gen_config,
            model_manager.device
        )

        # Generate from trained
        trained_output = generate_text(
            trained_model,
            trained_tokenizer,
            prompt,
            gen_config,
            model_manager.device
        )

        # Evaluate both outputs
        framework = setup_default_framework(
            model=model_manager.model,
            tokenizer=model_manager.tokenizer,
            device=model_manager.device
        )

        base_eval_result = framework.evaluate_text(base_output)
        trained_eval_result = framework.evaluate_text(trained_output)

        # Format evaluations
        base_eval = format_generation_evaluation(base_eval_result, "Base Model")
        trained_eval = format_generation_evaluation(trained_eval_result, "Trained Model")

        return base_output, trained_output, base_eval, trained_eval

    except Exception as e:
        error_msg = f"✗ Generation failed: {str(e)}"
        return error_msg, error_msg, "", ""

    finally:
        # Cleanup base model to free memory
        if base_model is not None:
            del base_model
        if base_tokenizer is not None:
            del base_tokenizer

        # Clear GPU cache if available
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            # Also try MPS cache clear (if available in PyTorch version)
            if hasattr(torch, 'mps') and hasattr(torch.mps, 'empty_cache'):
                torch.mps.empty_cache()
        except:
            pass  # Ignore cache clear errors

        import gc
        gc.collect()


def format_generation_evaluation(eval_result: Dict[str, Any], model_type: str) -> str:
    """Format evaluation results for generated text."""
    output = f"## {model_type} Evaluation\n\n"

    if eval_result.get("any_flagged", False):
        output += "**⚠️ VIOLATIONS DETECTED**\n"
        output += f"Flagged: {', '.join(eval_result.get('flagged_principles', []))}\n"
        output += f"Weighted Score: {eval_result.get('weighted_score', 0):.2f}\n"
    else:
        output += "**✓ NO VIOLATIONS**\n"

    return output


def load_adversarial_prompt_handler() -> str:
    """Load a random adversarial prompt."""
    import random
    prompts = get_adversarial_prompts("all")
    if prompts:
        return random.choice(prompts)
    return "How should I approach learning a new skill?"


# ============================================================================
# Impact Tab Functions
# ============================================================================

def run_comparison_handler(
    test_suite_name: str,
    temperature: float,
    max_length: int,
    progress=gr.Progress()
) -> Tuple[str, str, str]:
    """
    Run comparison between base and trained models on selected test suite.

    Args:
        test_suite_name: Name of test suite to run
        temperature: Generation temperature
        max_length: Maximum generation length
        progress: Gradio progress tracker

    Returns:
        Tuple of (results_summary, detailed_examples, export_data)
    """
    if not model_manager.can_compare():
        error_msg = "✗ Cannot run comparison: Need both base and trained model checkpoints.\n"
        error_msg += "Please train a model first in the Training tab."
        return error_msg, "", ""

    try:
        progress(0, desc="Loading models...")

        # Load base model
        base_model, base_tokenizer, success, msg = model_manager.load_checkpoint(
            model_manager.base_checkpoint_path
        )
        if not success:
            return f"✗ Failed to load base model: {msg}", "", ""

        # Trained model is current model
        trained_model = model_manager.model
        trained_tokenizer = model_manager.tokenizer

        # Get test suite prompts
        if test_suite_name == "Comprehensive (All)":
            # Combine all test suites
            test_prompts = []
            for prompts in TEST_SUITES.values():
                test_prompts.extend(prompts)
        else:
            # Map display name to key
            suite_key_map = {
                "Harmful Content": "harmful_content",
                "Stereotyping & Bias": "stereotyping",
                "Truthfulness": "truthfulness",
                "Autonomy & Manipulation": "autonomy_manipulation"
            }
            suite_key = suite_key_map.get(test_suite_name)
            if not suite_key:
                return f"✗ Unknown test suite: {test_suite_name}", "", ""

            test_prompts = TEST_SUITES[suite_key]

        progress(0.1, desc=f"Running comparison on {len(test_prompts)} prompts...")

        # Setup framework
        framework = setup_default_framework(
            model=model_manager.model,
            tokenizer=model_manager.tokenizer,
            device=model_manager.device
        )

        # Create comparison engine
        engine = ComparisonEngine(framework)

        # Progress callback
        def progress_callback(current, total, message):
            progress((0.1 + (current / total) * 0.8), desc=message)

        # Run comparison
        gen_config = GenerationConfig(
            max_length=max_length,
            temperature=temperature,
            do_sample=True
        )

        result = engine.compare_models(
            base_model=base_model,
            base_tokenizer=base_tokenizer,
            trained_model=trained_model,
            trained_tokenizer=trained_tokenizer,
            test_suite=test_prompts,
            device=model_manager.device,
            generation_config=gen_config,
            test_suite_name=test_suite_name,
            progress_callback=progress_callback
        )

        progress(0.95, desc="Formatting results...")

        # Format results
        summary = format_comparison_summary(result)
        detailed = format_detailed_examples(result)
        export_data = format_export_data(result)

        progress(1.0, desc="Complete!")

        # Cleanup base model
        del base_model
        del base_tokenizer

        # Clear cache
        try:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            if hasattr(torch, 'mps') and hasattr(torch.mps, 'empty_cache'):
                torch.mps.empty_cache()
        except:
            pass

        import gc
        gc.collect()

        return summary, detailed, export_data

    except Exception as e:
        error_msg = f"✗ Comparison failed: {str(e)}"
        import traceback
        error_msg += f"\n\nTraceback:\n{traceback.format_exc()}"
        return error_msg, "", ""


def format_comparison_summary(result: ComparisonResult) -> str:
    """Format comparison results as summary markdown."""
    output = f"# 🎯 {result.test_suite_name} - Impact Analysis\n\n"

    # Overall metrics
    output += "## 📊 Overall Performance\n\n"
    output += f"**Prompts Tested:** {result.num_prompts}\n"
    output += f"**Prompts Successful:** {result.num_prompts - result.skipped_prompts}\n"
    output += f"**Prompts Skipped:** {result.skipped_prompts}\n\n"

    output += f"**Alignment Score (Before):** `{result.overall_alignment_before:.3f}`\n"
    output += f"**Alignment Score (After):** `{result.overall_alignment_after:.3f}`\n"

    # Color code improvement
    improvement = result.alignment_improvement
    if improvement > 20:
        indicator = "✅"
    elif improvement > 10:
        indicator = "⚠️"
    else:
        indicator = "❌"

    output += f"**Alignment Improvement:** `{improvement:+.1f}%` {indicator}\n\n"

    # Per-principle results
    if result.principle_results:
        output += "## 📈 Per-Principle Results\n\n"
        output += "| Principle | Violations Before | Violations After | Improvement | Status |\n"
        output += "|-----------|-------------------|------------------|-------------|--------|\n"

        for principle_name, comparison in sorted(result.principle_results.items()):
            # Determine indicator
            improvement_pct = comparison.improvement_pct
            if improvement_pct > 20:
                status = "✅"
            elif improvement_pct > 10:
                status = "⚠️"
            elif improvement_pct >= 0:
                status = "➖"
            else:
                status = "❌"

            output += f"| {principle_name} | {comparison.violations_before} | "
            output += f"{comparison.violations_after} | {improvement_pct:+.1f}% | {status} |\n"

    # Errors if any
    if result.errors:
        output += f"\n## ⚠️ Errors ({len(result.errors)})\n\n"
        for i, error in enumerate(result.errors[:3], 1):
            output += f"{i}. {error}\n"
        if len(result.errors) > 3:
            output += f"\n... and {len(result.errors) - 3} more errors\n"

    return output


def format_detailed_examples(result: ComparisonResult) -> str:
    """Format detailed examples as expandable markdown."""
    if not result.examples:
        return "No examples available."

    output = f"# 📝 Detailed Examples ({len(result.examples)} total)\n\n"

    # Show first 10 examples
    for idx, example in enumerate(result.examples[:10], 1):
        output += f"## Example {idx}\n\n"

        # Improvement indicator
        if example.improved:
            output += "**Status:** ✅ Improved\n\n"
        else:
            base_score = example.base_evaluation.get('weighted_score', 0)
            trained_score = example.trained_evaluation.get('weighted_score', 0)
            if trained_score > base_score:
                output += "**Status:** ❌ Degraded\n\n"
            else:
                output += "**Status:** ➖ No change\n\n"

        output += f"**Prompt:** {example.prompt}\n\n"

        # Base output
        output += "**Base Model Output:**\n"
        output += f"> {example.base_output}\n\n"
        base_flagged = example.base_evaluation.get('flagged_principles', [])
        if base_flagged:
            output += f"⚠️ Violations: {', '.join(base_flagged)}\n\n"
        else:
            output += "✓ No violations\n\n"

        # Trained output
        output += "**Trained Model Output:**\n"
        output += f"> {example.trained_output}\n\n"
        trained_flagged = example.trained_evaluation.get('flagged_principles', [])
        if trained_flagged:
            output += f"⚠️ Violations: {', '.join(trained_flagged)}\n\n"
        else:
            output += "✓ No violations\n\n"

        output += "---\n\n"

    if len(result.examples) > 10:
        output += f"\n*Showing 10 of {len(result.examples)} examples. Use export to see all.*\n"

    return output


def format_export_data(result: ComparisonResult) -> str:
    """Format results as JSON for export."""
    import json

    export_dict = {
        "test_suite": result.test_suite_name,
        "num_prompts": result.num_prompts,
        "skipped_prompts": result.skipped_prompts,
        "overall_metrics": {
            "alignment_before": result.overall_alignment_before,
            "alignment_after": result.overall_alignment_after,
            "improvement_pct": result.alignment_improvement
        },
        "principle_results": {
            name: {
                "violations_before": comp.violations_before,
                "violations_after": comp.violations_after,
                "improvement_pct": comp.improvement_pct
            }
            for name, comp in result.principle_results.items()
        },
        "examples": [
            {
                "prompt": ex.prompt,
                "base_output": ex.base_output,
                "trained_output": ex.trained_output,
                "improved": ex.improved,
                "base_violations": ex.base_evaluation.get('flagged_principles', []),
                "trained_violations": ex.trained_evaluation.get('flagged_principles', [])
            }
            for ex in result.examples
        ],
        "errors": result.errors
    }

    return json.dumps(export_dict, indent=2)


# ============================================================================
# Gradio Interface
# ============================================================================

def create_demo() -> gr.Blocks:
    """Create the Gradio demo interface."""

    with gr.Blocks(title="Constitutional AI Interactive Demo", theme=gr.themes.Soft()) as demo:
        gr.Markdown("# Constitutional AI Interactive Demo")
        gr.Markdown("Demonstration of AI-based constitutional principle evaluation and training")

        # Global configuration section
        with gr.Row():
            with gr.Column(scale=2):
                model_dropdown = gr.Dropdown(
                    choices=["gpt2", "gpt2-medium", "distilgpt2"],
                    value="gpt2",
                    label="Model Selection"
                )
                device_dropdown = gr.Dropdown(
                    choices=["auto", "mps", "cuda", "cpu"],
                    value="auto",
                    label="Device Preference"
                )
                load_model_btn = gr.Button("Load Model", variant="primary")

            with gr.Column(scale=1):
                model_status = gr.Textbox(
                    label="Model Status",
                    value="No model loaded",
                    interactive=False
                )

        load_status = gr.Textbox(label="Status Messages", interactive=False)

        # Load model handler
        load_model_btn.click(
            fn=load_model_handler,
            inputs=[model_dropdown, device_dropdown],
            outputs=[load_status, model_status]
        )

        # Tabs
        with gr.Tabs():
            # ================================================================
            # Tab 1: Evaluation
            # ================================================================
            with gr.Tab("🎯 Evaluation"):
                gr.Markdown("## Evaluate Text Against Constitutional Principles")

                with gr.Row():
                    with gr.Column():
                        eval_text = gr.Textbox(
                            label="Text to Evaluate",
                            placeholder="Enter text to evaluate...",
                            lines=6
                        )

                        with gr.Row():
                            eval_mode = gr.Radio(
                                choices=["AI Evaluation", "Regex Evaluation", "Both (Comparison)"],
                                value="AI Evaluation",
                                label="Evaluation Mode"
                            )

                        with gr.Row():
                            example_dropdown = gr.Dropdown(
                                choices=[ex["name"] for ex in EVALUATION_EXAMPLES],
                                label="Load Example"
                            )
                            load_example_btn = gr.Button("Load")

                        evaluate_btn = gr.Button("Evaluate", variant="primary")

                    with gr.Column():
                        eval_status = gr.Textbox(label="Status", interactive=False)
                        eval_results = gr.Markdown(label="Results")

                # Event handlers
                load_example_btn.click(
                    fn=load_example_handler,
                    inputs=[example_dropdown],
                    outputs=[eval_text]
                )

                evaluate_btn.click(
                    fn=evaluate_text_handler,
                    inputs=[eval_text, eval_mode],
                    outputs=[eval_status, eval_results]
                )

            # ================================================================
            # Tab 2: Training
            # ================================================================
            with gr.Tab("🔧 Training"):
                gr.Markdown("## Train Model with Constitutional AI")

                with gr.Row():
                    with gr.Column():
                        training_mode_radio = gr.Radio(
                            choices=[
                                "Quick Demo (2 epochs, 20 examples, ~10-15 min)",
                                "Standard (5 epochs, 50 examples, ~25-35 min)"
                            ],
                            value="Quick Demo (2 epochs, 20 examples, ~10-15 min)",
                            label="Training Mode"
                        )

                        start_training_btn = gr.Button("Start Training", variant="primary")

                    with gr.Column():
                        training_status = gr.Textbox(label="Status", interactive=False)
                        training_metrics = gr.Markdown(label="Metrics")
                        checkpoint_info = gr.Textbox(label="Checkpoints", interactive=False)

                # Event handler
                start_training_btn.click(
                    fn=start_training_handler,
                    inputs=[training_mode_radio],
                    outputs=[training_status, training_metrics, checkpoint_info]
                )

            # ================================================================
            # Tab 3: Generation (Before/After Comparison)
            # ================================================================
            with gr.Tab("📝 Generation"):
                gr.Markdown("## Compare Base vs Trained Model Generation")

                with gr.Row():
                    with gr.Column():
                        gen_prompt = gr.Textbox(
                            label="Prompt",
                            placeholder="Enter a prompt...",
                            lines=3
                        )

                        load_adversarial_btn = gr.Button("Load Adversarial Prompt")

                        with gr.Row():
                            temperature_slider = gr.Slider(
                                minimum=0.1,
                                maximum=2.0,
                                value=0.7,
                                step=0.1,
                                label="Temperature"
                            )
                            max_length_slider = gr.Slider(
                                minimum=50,
                                maximum=500,
                                value=150,
                                step=50,
                                label="Max Length"
                            )

                        generate_btn = gr.Button("Generate from Both Models", variant="primary")

                with gr.Row():
                    with gr.Column():
                        gr.Markdown("### Base Model Output")
                        base_output = gr.Textbox(label="Generated Text", lines=6, interactive=False)
                        base_eval = gr.Markdown(label="Evaluation")

                    with gr.Column():
                        gr.Markdown("### Trained Model Output")
                        trained_output = gr.Textbox(label="Generated Text", lines=6, interactive=False)
                        trained_eval = gr.Markdown(label="Evaluation")

                # Event handlers
                load_adversarial_btn.click(
                    fn=load_adversarial_prompt_handler,
                    inputs=[],
                    outputs=[gen_prompt]
                )

                generate_btn.click(
                    fn=generate_comparison_handler,
                    inputs=[gen_prompt, temperature_slider, max_length_slider],
                    outputs=[base_output, trained_output, base_eval, trained_eval]
                )

            # ================================================================
            # Tab 4: Impact Analysis
            # ================================================================
            with gr.Tab("📊 Impact"):
                gr.Markdown("## Model Training Impact Analysis")
                gr.Markdown("Compare base and trained models on standardized test suites to measure improvement.")

                with gr.Row():
                    with gr.Column(scale=2):
                        # Test suite selection
                        test_suite_dropdown = gr.Dropdown(
                            choices=[
                                "Harmful Content",
                                "Stereotyping & Bias",
                                "Truthfulness",
                                "Autonomy & Manipulation",
                                "Comprehensive (All)"
                            ],
                            value="Harmful Content",
                            label="Test Suite Selection"
                        )

                        # Generation parameters
                        with gr.Row():
                            impact_temp_slider = gr.Slider(
                                minimum=0.1,
                                maximum=2.0,
                                value=0.7,
                                step=0.1,
                                label="Temperature"
                            )
                            impact_len_slider = gr.Slider(
                                minimum=50,
                                maximum=300,
                                value=100,
                                step=50,
                                label="Max Length"
                            )

                        run_comparison_btn = gr.Button("Run Comparison", variant="primary", size="lg")

                    with gr.Column(scale=1):
                        gr.Markdown("### Test Suite Details")
                        gr.Markdown("""
                        - **Harmful Content**: 20 prompts testing harm prevention
                        - **Stereotyping & Bias**: 20 prompts testing fairness
                        - **Truthfulness**: 15 prompts testing accuracy
                        - **Autonomy & Manipulation**: 15 prompts testing respect for autonomy
                        - **Comprehensive**: All 70 prompts combined

                        *Note: Comparison requires both base and trained models. Train a model first in the Training tab.*
                        """)

                # Results section
                gr.Markdown("---")
                gr.Markdown("## Results")

                with gr.Tabs():
                    with gr.Tab("Summary"):
                        results_summary = gr.Markdown(
                            value="*Run a comparison to see results*"
                        )

                    with gr.Tab("Detailed Examples"):
                        results_detailed = gr.Markdown(
                            value="*Run a comparison to see detailed examples*"
                        )

                    with gr.Tab("Export Data"):
                        gr.Markdown("### Export Results")
                        gr.Markdown("Copy the JSON below to save results for further analysis.")
                        export_data_textbox = gr.Textbox(
                            label="Export Data (JSON)",
                            lines=20,
                            max_lines=30,
                            interactive=False
                        )

                # Event handler
                run_comparison_btn.click(
                    fn=run_comparison_handler,
                    inputs=[test_suite_dropdown, impact_temp_slider, impact_len_slider],
                    outputs=[results_summary, results_detailed, export_data_textbox]
                )

    return demo


# ============================================================================
# Main Entry Point
# ============================================================================

if __name__ == "__main__":
    demo = create_demo()
    demo.launch(
        share=False,
        server_name="0.0.0.0",
        server_port=7860
    )
