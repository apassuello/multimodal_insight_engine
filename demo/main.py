"""MODULE: main.py
PURPOSE: Constitutional AI Interactive Demo - Gradio Application
KEY COMPONENTS:
- Gradio web interface with 3 tabs (Evaluation, Training, Generation)
- Integration with ModelManager, EvaluationManager, TrainingManager
- Real-time progress tracking and status updates
- Before/after model comparison
DEPENDENCIES: gradio, torch, typing, demo.managers, demo.data
SPECIAL NOTES: Phase 1 MVP implementation with essential features
"""

import gradio as gr
import torch
from typing import Dict, List, Any, Tuple, Optional

from demo.managers.model_manager import ModelManager, ModelStatus
from demo.managers.evaluation_manager import EvaluationManager
from demo.managers.training_manager import TrainingManager, TrainingConfig
from demo.data.test_examples import (
    EVALUATION_EXAMPLES,
    get_training_prompts,
    get_adversarial_prompts,
    TRAINING_CONFIGS
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
