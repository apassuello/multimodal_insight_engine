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

import argparse
import os
import yaml
from pathlib import Path
import gradio as gr
import torch
import threading
import time
from typing import Dict, List, Any, Tuple, Optional

from demo.managers.model_manager import ModelManager, ModelStatus
from demo.managers.evaluation_manager import EvaluationManager
from demo.managers.training_manager import TrainingManager, TrainingConfig
from demo.managers.comparison_engine import ComparisonEngine, ComparisonResult
from demo.managers.multi_model_manager import (
    MultiModelManager,
    RECOMMENDED_CONFIGS,
    get_evaluation_model_choices,
    get_generation_model_choices,
    get_all_model_choices
)
from demo.data.test_examples import (
    EVALUATION_EXAMPLES,
    get_training_prompts,
    get_adversarial_prompts,
    TRAINING_CONFIGS,
    TEST_SUITES
)
from demo.utils.content_logger import ContentLogger

from src.safety.constitutional.principles import setup_default_framework
from src.safety.constitutional.model_utils import generate_text, GenerationConfig


# ============================================================================
# Security Configuration
# ============================================================================
# Input validation limits to prevent DoS attacks
MAX_INPUT_LENGTH = 10000  # Maximum characters for text/prompt input
MAX_PROMPT_LENGTH = 5000  # Maximum characters for generation prompts
MIN_INPUT_LENGTH = 1      # Minimum characters for valid input

# Rate limiting configuration
RATE_LIMIT_TRAINING_SECONDS = 60      # Minimum seconds between training requests
RATE_LIMIT_COMPARISON_SECONDS = 30    # Minimum seconds between comparison requests
MAX_CONCURRENT_OPERATIONS = 1         # Maximum concurrent expensive operations


# Global managers
model_manager = ModelManager()
evaluation_manager = EvaluationManager()
training_manager = TrainingManager()
multi_model_manager = MultiModelManager()

# Global content logger (verbosity level 2 by default)
content_logger = ContentLogger(verbosity=2)

# Security: Rate limiting state
_rate_limit_state: Dict[str, float] = {}  # operation_name -> last_execution_timestamp
_rate_limit_lock = threading.Lock()
_operation_semaphore = threading.Semaphore(MAX_CONCURRENT_OPERATIONS)

# Security: Thread safety locks for global managers
_model_manager_lock = threading.Lock()       # Protects model_manager operations
_multi_model_manager_lock = threading.Lock() # Protects multi_model_manager operations


# ============================================================================
# Security Helper Functions
# ============================================================================

def validate_input_length(
    text: str,
    max_length: int = MAX_INPUT_LENGTH,
    input_name: str = "Input"
) -> Tuple[bool, str]:
    """
    Validate input text length to prevent DoS attacks.

    Security: This prevents resource exhaustion from extremely long inputs.

    Args:
        text: Input text to validate
        max_length: Maximum allowed length
        input_name: Name of the input field for error messages

    Returns:
        Tuple of (is_valid, error_message)
    """
    if not text or len(text) < MIN_INPUT_LENGTH:
        return False, f"✗ Security: {input_name} is empty or too short (minimum {MIN_INPUT_LENGTH} characters)"

    if len(text) > max_length:
        return False, (
            f"✗ Security: {input_name} exceeds maximum length\n"
            f"Length: {len(text):,} characters\n"
            f"Maximum: {max_length:,} characters\n"
            f"This limit prevents resource exhaustion attacks."
        )

    return True, ""


def check_rate_limit(operation_name: str, cooldown_seconds: int) -> Tuple[bool, str]:
    """
    Check if an operation can be executed based on rate limits.

    Security: This prevents DoS attacks via repeated expensive operations.

    Args:
        operation_name: Name of the operation (for tracking)
        cooldown_seconds: Minimum seconds between executions

    Returns:
        Tuple of (can_execute, error_message)
    """
    global _rate_limit_state, _rate_limit_lock

    with _rate_limit_lock:
        current_time = time.time()
        last_execution = _rate_limit_state.get(operation_name, 0)
        time_since_last = current_time - last_execution

        if time_since_last < cooldown_seconds:
            remaining = cooldown_seconds - time_since_last
            return False, (
                f"✗ Security: Rate limit exceeded for {operation_name}\n"
                f"Please wait {remaining:.0f} seconds before trying again.\n"
                f"This prevents system overload from repeated requests."
            )

        # Update last execution time
        _rate_limit_state[operation_name] = current_time
        return True, ""


def acquire_operation_slot() -> bool:
    """
    Try to acquire a slot for expensive operations.

    Security: This prevents multiple expensive operations from running concurrently.

    Returns:
        True if slot acquired, False otherwise
    """
    return _operation_semaphore.acquire(blocking=False)


def release_operation_slot() -> None:
    """Release a slot for expensive operations."""
    _operation_semaphore.release()


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
    # Security: Thread safety for global model_manager
    with _model_manager_lock:
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
# Logger Control Functions
# ============================================================================

def update_logger_verbosity(verbosity: int) -> str:
    """
    Update content logger verbosity level.

    Args:
        verbosity: Logging level (0=off, 1=summary, 2=key stages, 3=full pipeline)

    Returns:
        Status message
    """
    global content_logger
    content_logger.verbosity = int(verbosity)

    levels = {
        0: "Off (no logging)",
        1: "Summary only",
        2: "Key stages (default)",
        3: "Full pipeline"
    }

    return f"✓ Logging verbosity set to level {verbosity}: {levels.get(verbosity, 'Unknown')}"


def export_logs_handler() -> Tuple[str, str]:
    """
    Export content logs to JSON file.

    Returns:
        Tuple of (status_message, log_summary)
    """
    global content_logger

    if not content_logger.logs:
        return "⚠ No logs to export", "No logs recorded yet. Run training or evaluation first."

    # Export to file
    import tempfile
    import json
    from datetime import datetime

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"content_logs_{timestamp}.json"
    filepath = os.path.join(tempfile.gettempdir(), filename)

    content_logger.export_logs(filepath)

    # Generate summary
    summary = f"✓ Exported {len(content_logger.logs)} log entries to:\n{filepath}\n\n"
    summary += content_logger.get_summary()

    return f"✓ Logs exported to {filepath}", summary


# ============================================================================
# Dual Model Management Functions
# ============================================================================

def load_evaluation_model_handler(model_key: str) -> Tuple[str, str]:
    """
    Load evaluation model using MultiModelManager.

    Args:
        model_key: Model identifier from RECOMMENDED_CONFIGS

    Returns:
        Tuple of (status_message, model_info)
    """
    global multi_model_manager

    # Security: Thread safety for global multi_model_manager
    with _multi_model_manager_lock:
        success, message = multi_model_manager.load_evaluation_model(model_key)

        if success:
            # Initialize evaluation framework with new model
            eval_model, eval_tokenizer = multi_model_manager.get_evaluation_model()
            eval_success, eval_msg = evaluation_manager.initialize_frameworks(
                model=eval_model,
                tokenizer=eval_tokenizer,
                device=multi_model_manager.device
            )

            if not eval_success:
                message += f"\n\nWarning: {eval_msg}"

            # Get model info
            status = multi_model_manager.get_status_info()
            model_info = ""
            if status["evaluation_model"]:
                model_info += f"Evaluation Model: {status['evaluation_model']['name']}\n"
                model_info += f"Parameters: {status['evaluation_model']['parameters']:,}\n"
                model_info += f"Memory: {status['evaluation_model']['memory_gb']:.1f}GB\n"
            if status["generation_model"]:
                model_info += f"\nGeneration Model: {status['generation_model']['name']}\n"
                model_info += f"Parameters: {status['generation_model']['parameters']:,}\n"
                model_info += f"Memory: {status['generation_model']['memory_gb']:.1f}GB\n"
            model_info += f"\nTotal Memory: {status['total_memory_gb']:.1f}GB\n"
            model_info += f"Device: {status['device']}"

            return message, model_info
        else:
            return message, "Failed to load evaluation model"


def load_generation_model_handler(model_key: str) -> Tuple[str, str]:
    """
    Load generation model using MultiModelManager.

    Args:
        model_key: Model identifier from RECOMMENDED_CONFIGS

    Returns:
        Tuple of (status_message, model_info)
    """
    global multi_model_manager

    # Security: Thread safety for global multi_model_manager
    with _multi_model_manager_lock:
        success, message = multi_model_manager.load_generation_model(model_key)

        if success:
            # Get model info
            status = multi_model_manager.get_status_info()
            model_info = ""
            if status["evaluation_model"]:
                model_info += f"Evaluation Model: {status['evaluation_model']['name']}\n"
                model_info += f"Parameters: {status['evaluation_model']['parameters']:,}\n"
                model_info += f"Memory: {status['evaluation_model']['memory_gb']:.1f}GB\n"
            if status["generation_model"]:
                model_info += f"\nGeneration Model: {status['generation_model']['name']}\n"
                model_info += f"Parameters: {status['generation_model']['parameters']:,}\n"
                model_info += f"Memory: {status['generation_model']['memory_gb']:.1f}GB\n"
            model_info += f"\nTotal Memory: {status['total_memory_gb']:.1f}GB\n"
            model_info += f"Device: {status['device']}"

            return message, model_info
        else:
            return message, "Failed to load generation model"


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
    # Security: Validate input length to prevent DoS
    is_valid, error_msg = validate_input_length(text, MAX_INPUT_LENGTH, "Text input")
    if not is_valid:
        return error_msg, ""

    # Check if models are available
    has_dual_eval = multi_model_manager.eval_model is not None
    has_single = model_manager.is_ready()

    if mode == "AI Evaluation" and not has_dual_eval and not has_single:
        return "✗ Please load a model first (single model or evaluation model in dual mode)", ""

    # Use dual model if available
    if has_dual_eval and mode == "AI Evaluation":
        # FIX (BUG #4): Add error handling for evaluation manager re-initialization
        # Re-initialize evaluation manager with dual model
        eval_model, eval_tokenizer = multi_model_manager.get_evaluation_model()
        init_success, init_msg = evaluation_manager.initialize_frameworks(
            model=eval_model,
            tokenizer=eval_tokenizer,
            device=multi_model_manager.device
        )

        # If initialization fails, fall back to regex evaluation or return error
        if not init_success:
            return f"✗ Failed to initialize evaluation manager: {init_msg}", ""

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
    # Security: Check rate limit to prevent DoS
    can_execute, rate_error = check_rate_limit("training", RATE_LIMIT_TRAINING_SECONDS)
    if not can_execute:
        return rate_error, "", ""

    # Security: Check concurrency limit
    if not acquire_operation_slot():
        return "✗ Security: Another expensive operation is in progress. Please wait.", "", ""

    try:
        # Check if we have dual models loaded
        use_dual_models = multi_model_manager.gen_model is not None and multi_model_manager.eval_model is not None

        if not use_dual_models and not model_manager.is_ready():
            return "✗ Please load models first (either single model or dual models)", "", ""

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

        # Select models based on what's available
        if use_dual_models:
            # Use dual model architecture: evaluation model for critique, generation model for training
            gen_model, gen_tokenizer = multi_model_manager.get_generation_model()
            eval_model, eval_tokenizer = multi_model_manager.get_evaluation_model()
            device = multi_model_manager.device

            # Setup framework with evaluation model
            framework = setup_default_framework(
                model=eval_model,
                tokenizer=eval_tokenizer,
                device=device
            )

            # Train the generation model
            train_model = gen_model
            train_tokenizer = gen_tokenizer
        else:
            # Use single model for everything
            framework = setup_default_framework(
                model=model_manager.model,
                tokenizer=model_manager.tokenizer,
                device=model_manager.device
            )
            train_model = model_manager.model
            train_tokenizer = model_manager.tokenizer
            device = model_manager.device

        # Progress callback
        def progress_callback(status: str, progress_pct: float):
            progress(progress_pct, desc=status)

        # Checkpoint callback
        def checkpoint_callback(epoch: int, metrics: Dict[str, Any]):
            if not use_dual_models:
                model_manager.save_trained_checkpoint(epoch=epoch, metrics=metrics)

        # Set model to training status
        if not use_dual_models:
            model_manager.set_status(ModelStatus.TRAINING)

        # Execute training
        result, success, message = training_manager.train_model(
            model=train_model,
            tokenizer=train_tokenizer,
            framework=framework,
            device=device,
            training_prompts=training_prompts,
            config=config,
            progress_callback=progress_callback,
            checkpoint_callback=checkpoint_callback,
            logger=content_logger
        )

        # Reset model status
        if not use_dual_models:
            model_manager.set_status(ModelStatus.READY)

        if success:
            # FIX (CRITICAL - BUG #1): Only save checkpoint for single model mode
            # When using dual models, the trained model is in multi_model_manager, not model_manager
            if not use_dual_models:
                model_manager.save_trained_checkpoint(
                    epoch=config.num_epochs,
                    metrics=result.get("metrics", {})
                )

            # Format metrics
            metrics_display = format_training_metrics(result)

            # Checkpoint info
            if use_dual_models:
                # For dual models, provide dual model information
                checkpoint_info = "Dual model architecture active:\n"
                status = multi_model_manager.get_status_info()
                if status["evaluation_model"]:
                    checkpoint_info += f"- Evaluation: {status['evaluation_model']['name']}\n"
                if status["generation_model"]:
                    checkpoint_info += f"- Generation: {status['generation_model']['name']}"
            else:
                # For single model, show checkpoint paths
                checkpoint_info = f"Base checkpoint: {model_manager.base_checkpoint_path}\n"
                checkpoint_info += f"Trained checkpoint: {model_manager.trained_checkpoint_path}"

            return message, metrics_display, checkpoint_info
        else:
            return message, "", ""

    finally:
        # Security: Always release operation slot
        release_operation_slot()


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
    # Security: Validate prompt length to prevent DoS
    is_valid, error_msg = validate_input_length(prompt, MAX_PROMPT_LENGTH, "Prompt")
    if not is_valid:
        return error_msg, error_msg, "", ""

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
        # FIX: Use max_new_tokens so max_length parameter means "new tokens to generate"
        gen_config = GenerationConfig(
            max_new_tokens=max_length,  # User expects this many NEW tokens
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
        # FIX (BUG #5): Robust cleanup with error handling to prevent memory leaks
        # Even if cleanup fails, ensure all cleanup steps are attempted
        cleanup_errors = []

        try:
            # Cleanup base model to free memory
            if base_model is not None:
                del base_model
        except Exception as e:
            cleanup_errors.append(f"Failed to delete base_model: {e}")

        try:
            if base_tokenizer is not None:
                del base_tokenizer
        except Exception as e:
            cleanup_errors.append(f"Failed to delete base_tokenizer: {e}")

        # Clear GPU/MPS cache if available
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            # Also try MPS cache clear (if available in PyTorch version)
            if hasattr(torch, 'mps') and hasattr(torch.mps, 'empty_cache'):
                torch.mps.empty_cache()
        except Exception as e:
            cleanup_errors.append(f"Failed to clear cache: {e}")

        # Garbage collection
        try:
            import gc
            gc.collect()
        except Exception as e:
            cleanup_errors.append(f"Failed to run garbage collection: {e}")

        # Log any cleanup errors for debugging (but don't raise them)
        if cleanup_errors:
            import logging
            for error in cleanup_errors:
                logging.warning(f"Cleanup issue: {error}")


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
) -> Tuple[str, str, str, str]:
    """
    Run comparison between base and trained models on selected test suite.

    Args:
        test_suite_name: Name of test suite to run
        temperature: Generation temperature
        max_length: Maximum generation length
        progress: Gradio progress tracker

    Returns:
        Tuple of (results_summary, detailed_examples, export_json, export_csv)
    """
    # Security: Check rate limit to prevent DoS
    can_execute, rate_error = check_rate_limit("comparison", RATE_LIMIT_COMPARISON_SECONDS)
    if not can_execute:
        return rate_error, "", "", ""

    # Security: Check concurrency limit
    if not acquire_operation_slot():
        return "✗ Security: Another expensive operation is in progress. Please wait.", "", "", ""

    if not model_manager.can_compare():
        error_msg = "✗ Cannot run comparison: Need both base and trained model checkpoints.\n"
        error_msg += "Please train a model first in the Training tab."
        release_operation_slot()  # Release before returning
        return error_msg, "", "", ""

    # Security: Validate inputs (HIGH-01, HIGH-02 fixes)
    MAX_TEST_SUITE_SIZE = 100
    MIN_TEMPERATURE = 0.1
    MAX_TEMPERATURE = 2.0
    MIN_MAX_LENGTH = 10
    MAX_MAX_LENGTH = 1000

    # Validate generation parameters
    if not (MIN_TEMPERATURE <= temperature <= MAX_TEMPERATURE):
        error_msg = f"✗ Invalid temperature: {temperature}\n"
        error_msg += f"Must be between {MIN_TEMPERATURE} and {MAX_TEMPERATURE}"
        release_operation_slot()  # Release before returning
        return error_msg, "", "", ""

    if not (MIN_MAX_LENGTH <= max_length <= MAX_MAX_LENGTH):
        error_msg = f"✗ Invalid max_length: {max_length}\n"
        error_msg += f"Must be between {MIN_MAX_LENGTH} and {MAX_MAX_LENGTH}"
        release_operation_slot()  # Release before returning
        return error_msg, "", "", ""

    try:
        progress(0, desc="Loading models...")

        # Load base model
        base_model, base_tokenizer, success, msg = model_manager.load_checkpoint(
            model_manager.base_checkpoint_path
        )
        if not success:
            # FIX (BUG #3): Return 4 values to match function signature
            return f"✗ Failed to load base model: {msg}", "", "", ""

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

        # Validate test suite size (Security: HIGH-01 fix - DoS protection)
        if len(test_prompts) > MAX_TEST_SUITE_SIZE:
            error_msg = f"✗ Test suite too large: {len(test_prompts)} prompts\n"
            error_msg += f"Maximum allowed: {MAX_TEST_SUITE_SIZE} prompts\n"
            error_msg += "Please select a smaller test suite."
            return error_msg, "", "", ""

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
        # FIX: Use max_new_tokens so max_length parameter means "new tokens to generate"
        gen_config = GenerationConfig(
            max_new_tokens=max_length,  # User expects this many NEW tokens
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
            progress_callback=progress_callback,
            logger=content_logger
        )

        progress(0.95, desc="Formatting results...")

        # Format results
        summary = format_comparison_summary(result)
        detailed = format_detailed_examples(result)
        export_json = format_export_data(result)
        export_csv = format_export_csv(result)

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

        return summary, detailed, export_json, export_csv

    except Exception as e:
        # Security: Don't expose full traceback to users (CRIT-01 fix)
        # Log the full error for debugging but show user-friendly message
        import traceback
        import logging
        logging.error(f"Comparison failed: {traceback.format_exc()}")

        error_msg = f"✗ Comparison failed: {str(e)}\n\n"
        error_msg += "Please check that:\n"
        error_msg += "- Both base and trained models are loaded\n"
        error_msg += "- Test suite is valid\n"
        error_msg += "- Generation parameters are within acceptable ranges"
        return error_msg, "", "", ""

    finally:
        # Security: Always release operation slot
        release_operation_slot()


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


def format_export_csv(result: ComparisonResult) -> str:
    """Format results as CSV for export to Excel/R/Python."""
    import csv
    import io

    output = io.StringIO()
    writer = csv.writer(output)

    # Overall metrics section
    writer.writerow(["# Overall Metrics"])
    writer.writerow(["Metric", "Value"])
    writer.writerow(["Test Suite", result.test_suite_name])
    writer.writerow(["Total Prompts", result.num_prompts])
    writer.writerow(["Successful Prompts", result.num_prompts - result.skipped_prompts])
    writer.writerow(["Skipped Prompts", result.skipped_prompts])
    writer.writerow(["Alignment Score (Before)", f"{result.overall_alignment_before:.4f}"])
    writer.writerow(["Alignment Score (After)", f"{result.overall_alignment_after:.4f}"])
    writer.writerow(["Alignment Improvement (%)", f"{result.alignment_improvement:+.2f}"])
    writer.writerow([])  # Blank line

    # Per-principle metrics section
    writer.writerow(["# Per-Principle Results"])
    writer.writerow(["Principle", "Violations Before", "Violations After", "Improvement (%)"])
    for principle_name in sorted(result.principle_results.keys()):
        comp = result.principle_results[principle_name]
        writer.writerow([
            principle_name,
            comp.violations_before,
            comp.violations_after,
            f"{comp.improvement_pct:+.2f}"
        ])
    writer.writerow([])  # Blank line

    # Example comparisons section
    writer.writerow(["# Example Comparisons"])
    writer.writerow([
        "Prompt",
        "Base Output",
        "Trained Output",
        "Improved",
        "Base Violations",
        "Trained Violations"
    ])
    for example in result.examples:
        writer.writerow([
            example.prompt,
            example.base_output,
            example.trained_output,
            "Yes" if example.improved else "No",
            ", ".join(example.base_evaluation.get('flagged_principles', [])),
            ", ".join(example.trained_evaluation.get('flagged_principles', []))
        ])

    # Errors section (if any)
    if result.errors:
        writer.writerow([])  # Blank line
        writer.writerow(["# Errors"])
        writer.writerow(["Error Message"])
        for error in result.errors:
            writer.writerow([error])

    return output.getvalue()


# ============================================================================
# Gradio Interface
# ============================================================================

def create_demo() -> gr.Blocks:
    """Create the Gradio demo interface."""

    # Custom theme for professional appearance
    custom_theme = gr.themes.Soft(
        primary_hue="blue",
        secondary_hue="cyan",
        neutral_hue="slate",
        font=[gr.themes.GoogleFont("Inter"), "ui-sans-serif", "system-ui", "sans-serif"],
    ).set(
        body_background_fill="*neutral_50",
        body_background_fill_dark="*neutral_900",
        button_primary_background_fill="*primary_600",
        button_primary_background_fill_hover="*primary_700",
        button_primary_text_color="white",
        block_title_text_weight="600",
        block_label_text_weight="600",
        block_label_text_size="*text_md",
        checkbox_label_text_size="*text_sm",
    )

    with gr.Blocks(
        title="Constitutional AI Interactive Demo",
        theme=custom_theme,
        css="""
        .gradio-container {
            max-width: 1400px !important;
        }
        """
    ) as demo:
        gr.Markdown("# Constitutional AI Interactive Demo")
        gr.Markdown("Demonstration of AI-based constitutional principle evaluation and training")

        # Global configuration section
        gr.Markdown("### Single Model Mode (Legacy)")
        gr.Markdown("*For best results, use the Dual Model Architecture section below instead*")

        with gr.Row():
            with gr.Column(scale=2):
                model_dropdown = gr.Dropdown(
                    choices=get_all_model_choices(),
                    value="phi-3-mini-instruct",
                    label="Model Selection (Legacy - Use Dual Models Below)"
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

        # Logging controls
        with gr.Row():
            with gr.Column(scale=2):
                verbosity_slider = gr.Slider(
                    minimum=0,
                    maximum=3,
                    value=2,
                    step=1,
                    label="Content Logging Verbosity (0=off, 1=summary, 2=key stages, 3=full pipeline)",
                    info="Controls how much detail is logged to the terminal during evaluation and training"
                )
                verbosity_status = gr.Textbox(
                    label="Logging Status",
                    value="✓ Logging verbosity set to level 2: Key stages (default)",
                    interactive=False
                )

            with gr.Column(scale=1):
                export_logs_btn = gr.Button("📥 Export Logs", variant="secondary")
                export_status = gr.Textbox(
                    label="Export Status",
                    value="No logs to export yet",
                    interactive=False,
                    lines=3
                )

        # Dual model configuration (advanced)
        with gr.Accordion("🔬 Advanced: Dual Model Architecture", open=True):
            gr.Markdown("""
            **Dual Model System**: Use separate models for evaluation and generation/training for improved performance.
            - **Evaluation Model**: Instruction-tuned models recommended (Phi-3-mini or Qwen2.5-3B)
            - **Generation Model**: For training/fine-tuning (Phi-3-mini or Qwen2.5-3B recommended)

            **Model Tiers**:
            - 🥇 **Tier 1 (Recommended)**: phi-3-mini-instruct (~7.6GB), qwen2.5-3b-instruct (~6GB)
            - 🥈 **Tier 2 (More capable)**: mistral-7b-instruct (~14GB), qwen2.5-7b-instruct (~14GB)
            - 🥉 **Tier 3 (Limited resources)**: qwen2.5-1.5b-instruct (~3GB), tinyllama-chat (~2.2GB)
            """)

            with gr.Row():
                with gr.Column():
                    gr.Markdown("### Evaluation Model")
                    eval_model_dropdown = gr.Dropdown(
                        choices=get_evaluation_model_choices(),
                        value="phi-3-mini-instruct",
                        label="Select Evaluation Model",
                        info="Instruction-tuned models recommended for reliable JSON output"
                    )
                    load_eval_model_btn = gr.Button("Load Evaluation Model", variant="primary")
                    eval_load_status = gr.Textbox(
                        label="Status",
                        value="No evaluation model loaded",
                        interactive=False,
                        lines=2
                    )

                with gr.Column():
                    gr.Markdown("### Generation Model")
                    gen_model_dropdown = gr.Dropdown(
                        choices=get_generation_model_choices(),
                        value="phi-3-mini-gen",
                        label="Select Generation Model",
                        info="Used for training and text generation"
                    )
                    load_gen_model_btn = gr.Button("Load Generation Model", variant="primary")
                    gen_load_status = gr.Textbox(
                        label="Status",
                        value="No generation model loaded",
                        interactive=False,
                        lines=2
                    )

            dual_model_status = gr.Textbox(
                label="Dual Model System Status",
                value="No dual models loaded. Using single model system.",
                interactive=False,
                lines=4
            )

        load_status = gr.Textbox(label="Status Messages", interactive=False)

        # Load model handler
        load_model_btn.click(
            fn=load_model_handler,
            inputs=[model_dropdown, device_dropdown],
            outputs=[load_status, model_status]
        )

        # Logger control handlers
        verbosity_slider.change(
            fn=update_logger_verbosity,
            inputs=[verbosity_slider],
            outputs=[verbosity_status]
        )

        export_logs_btn.click(
            fn=export_logs_handler,
            inputs=[],
            outputs=[verbosity_status, export_status]
        )

        # Dual model handlers
        load_eval_model_btn.click(
            fn=load_evaluation_model_handler,
            inputs=[eval_model_dropdown],
            outputs=[eval_load_status, dual_model_status]
        )

        load_gen_model_btn.click(
            fn=load_generation_model_handler,
            inputs=[gen_model_dropdown],
            outputs=[gen_load_status, dual_model_status]
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
                        gr.Markdown("Choose your preferred export format for further analysis.")

                        with gr.Tabs():
                            with gr.Tab("JSON"):
                                gr.Markdown("JSON format for programmatic access (Python, JavaScript, etc.)")
                                export_json_textbox = gr.Textbox(
                                    label="Export Data (JSON)",
                                    lines=15,
                                    max_lines=25,
                                    interactive=False
                                )

                            with gr.Tab("CSV"):
                                gr.Markdown("CSV format for Excel, R, pandas, and data analysis tools")
                                export_csv_textbox = gr.Textbox(
                                    label="Export Data (CSV)",
                                    lines=15,
                                    max_lines=25,
                                    interactive=False
                                )

                # Event handler
                run_comparison_btn.click(
                    fn=run_comparison_handler,
                    inputs=[test_suite_dropdown, impact_temp_slider, impact_len_slider],
                    outputs=[results_summary, results_detailed, export_json_textbox, export_csv_textbox]
                )

            # ================================================================
            # Tab 5: Architecture & Documentation
            # ================================================================
            with gr.Tab("📚 Architecture"):
                gr.Markdown("## Constitutional AI Demo Architecture")

                with gr.Tabs():
                    with gr.Tab("Overview"):
                        gr.Markdown("""
                        ### System Overview

                        This demo implements Constitutional AI (CAI) for training language models
                        to adhere to human values encoded as principles.

                        **Core Components:**
                        - **ConstitutionalFramework**: Defines and evaluates principles
                        - **ModelManager**: Handles model loading and checkpointing
                        - **TrainingManager**: Orchestrates CAI training pipeline
                        - **EvaluationManager**: AI-based principle evaluation
                        - **ComparisonEngine**: Quantifies training improvements

                        **Training Pipeline:**
                        1. Generate responses from base model
                        2. Critique responses using constitutional principles
                        3. Revise responses to align with principles
                        4. Fine-tune model on critique-revised data
                        5. Evaluate improvement via before/after comparison

                        **Key Features:**
                        - AI-first evaluation (with regex fallback)
                        - Real model training (GPT-2 family)
                        - Quantitative impact analysis
                        - M4-Pro MPS acceleration support
                        - Export to JSON and CSV
                        """)

                    with gr.Tab("API Examples"):
                        gr.Markdown("""
                        ### Quick Start Examples

                        **Evaluate Text:**
                        ```python
                        from src.safety.constitutional.principles import setup_default_framework

                        # Setup framework with AI evaluation
                        framework = setup_default_framework(
                            model=model,
                            tokenizer=tokenizer,
                            device=device
                        )

                        # Evaluate text
                        result = framework.evaluate_text("Your text here")
                        print(f"Flagged: {result['any_flagged']}")
                        print(f"Violations: {result['flagged_principles']}")
                        ```

                        **Train Model:**
                        ```python
                        from demo.managers import TrainingManager, TrainingConfig

                        manager = TrainingManager()
                        config = TrainingConfig(
                            num_epochs=2,
                            num_examples=20,
                            batch_size=4
                        )

                        result = manager.train_model(
                            model, tokenizer, framework, device, config
                        )
                        ```

                        **Compare Models:**
                        ```python
                        from demo.managers import ComparisonEngine

                        engine = ComparisonEngine(framework)
                        result = engine.compare_models(
                            base_model, base_tokenizer,
                            trained_model, trained_tokenizer,
                            test_suite=prompts,
                            device=device,
                            generation_config=gen_config
                        )

                        print(f"Improvement: {result.alignment_improvement:.1f}%")
                        ```
                        """)

                    with gr.Tab("Configuration"):
                        gr.Markdown("""
                        ### Configuration Options

                        **Model Selection:**
                        - `gpt2` (124M params) - Fast, good for testing
                        - `gpt2-medium` (355M params) - Better quality
                        - `distilgpt2` (82M params) - Fastest, lower quality

                        **Device Options:**
                        - `auto` - Automatically select best device (recommended)
                        - `mps` - Apple Silicon GPU (M4-Pro)
                        - `cuda` - NVIDIA GPU
                        - `cpu` - CPU fallback (slower)

                        **Training Modes:**
                        - **Quick Demo**: 2 epochs, 20 examples (~5-10 min)
                        - **Standard**: 3 epochs, 50 examples (~15-25 min)

                        **Evaluation Modes:**
                        - **AI**: Uses language model for nuanced detection (default)
                        - **Regex**: Fast pattern matching (backward compatible)

                        **Export Formats:**
                        - **JSON**: For programmatic access (Python, JavaScript)
                        - **CSV**: For Excel, R, pandas, data analysis tools

                        **Security Limits:**
                        - Max test suite size: 100 prompts
                        - Temperature range: 0.1 - 2.0
                        - Max length range: 10 - 1000 tokens
                        """)

                    with gr.Tab("Resources"):
                        gr.Markdown("""
                        ### Documentation & Resources

                        **Project Documentation:**
                        - `DEMO_ARCHITECTURE.md` - Complete specification
                        - `IMPLEMENTATION_SUMMARY.md` - Technical details
                        - `SECURITY_AUDIT_PHASE2.md` - Security analysis
                        - `README.md` - Project overview

                        **Constitutional AI Papers:**
                        - [Constitutional AI: Harmlessness from AI Feedback](https://arxiv.org/abs/2212.08073)
                        - [Training a Helpful and Harmless Assistant with RLHF](https://arxiv.org/abs/2204.05862)

                        **Key Concepts:**
                        - **Constitutional Principles**: Human values encoded as rules
                        - **Critique-Revision**: Generate → Critique → Revise
                        - **Alignment Score**: Quantitative adherence metric (0-1)
                        - **Weighted Violations**: Principle violations × weights

                        **Technical Stack:**
                        - PyTorch 2.x
                        - Transformers (Hugging Face)
                        - Gradio 4.x
                        - Python 3.10+

                        **Hardware Requirements:**
                        - Minimum: 8GB RAM, CPU
                        - Recommended: 16GB+ RAM, M4-Pro/CUDA GPU
                        - Training time: ~5-25 min (depending on mode)
                        """)

    return demo


# ============================================================================
# Configuration Loading
# ============================================================================

def load_config(config_path: str = "demo/config.yaml") -> Dict[str, Any]:
    """
    Load configuration from YAML file.

    Args:
        config_path: Path to config.yaml

    Returns:
        Configuration dictionary
    """
    config_file = Path(config_path)
    if config_file.exists():
        with open(config_file, 'r') as f:
            return yaml.safe_load(f)
    return {}


def get_config_value(key: str, default: Any = None, config: Dict[str, Any] = None) -> Any:
    """
    Get configuration value with priority: ENV > config.yaml > default.

    Priority order:
    1. Environment variable (uppercase with prefix)
    2. config.yaml value
    3. Default value

    Args:
        key: Configuration key (e.g., "server_name")
        default: Default value if not found
        config: Loaded config dictionary

    Returns:
        Configuration value
    """
    # Convert key to uppercase environment variable name
    env_key = key.upper()
    if not env_key.startswith("GRADIO_"):
        env_key = f"GRADIO_{env_key}"

    # Check environment variable first
    env_value = os.getenv(env_key)
    if env_value is not None:
        # Convert string booleans to actual booleans
        if env_value.lower() in ('true', '1', 'yes'):
            return True
        elif env_value.lower() in ('false', '0', 'no'):
            return False
        # Convert string numbers to integers
        try:
            if '.' not in env_value:
                return int(env_value)
        except ValueError:
            pass
        return env_value

    # Check config.yaml next
    if config and key in config:
        return config[key]

    # Return default
    return default


# ============================================================================
# Health Check Endpoint
# ============================================================================

def create_health_check_app():
    """
    Create a simple Gradio app with health check endpoint.

    Returns:
        Gradio Blocks app with /health endpoint
    """
    with gr.Blocks() as health_app:
        gr.Markdown("# Health Check")
        status_output = gr.JSON(value={"status": "healthy", "service": "Constitutional AI Demo"})

    return health_app


# ============================================================================
# Main Entry Point
# ============================================================================

def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Constitutional AI Interactive Demo",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Environment Variables:
  GRADIO_SERVER_NAME     Server hostname (default: 0.0.0.0)
  GRADIO_SERVER_PORT     Server port (default: 7860)
  GRADIO_SHARE           Enable public URL via Gradio share (default: false)
  DEFAULT_MODEL          Default model to use (default: gpt2)
  DEVICE_PREFERENCE      Device preference: auto, mps, cuda, cpu (default: auto)

Examples:
  # Run with default settings
  python -m demo.main

  # Run on specific port
  python -m demo.main --server-port 8080

  # Run with public URL
  python -m demo.main --share

  # Use environment variables
  GRADIO_SERVER_PORT=8080 python -m demo.main
        """
    )

    parser.add_argument(
        "--server-name",
        type=str,
        default=None,
        help="Server hostname (default: 0.0.0.0, env: GRADIO_SERVER_NAME)"
    )

    parser.add_argument(
        "--server-port",
        type=int,
        default=None,
        help="Server port (default: 7860, env: GRADIO_SERVER_PORT)"
    )

    parser.add_argument(
        "--share",
        action="store_true",
        default=None,
        help="Enable public URL via Gradio share (env: GRADIO_SHARE)"
    )

    parser.add_argument(
        "--config",
        type=str,
        default="demo/config.yaml",
        help="Path to config.yaml file (default: demo/config.yaml)"
    )

    return parser.parse_args()


if __name__ == "__main__":
    # Parse command-line arguments
    args = parse_args()

    # Load configuration from YAML
    config = load_config(args.config)

    # Get configuration values with priority: CLI > ENV > config.yaml > default
    server_name = args.server_name or get_config_value("server_name", "0.0.0.0", config)
    server_port = args.server_port or get_config_value("server_port", 7860, config)
    share = args.share if args.share is not None else get_config_value("share", False, config)

    # Print startup configuration
    print("=" * 60)
    print("Constitutional AI Interactive Demo")
    print("=" * 60)
    print(f"Server: {server_name}:{server_port}")
    print(f"Share: {share}")
    print(f"Config: {args.config}")
    print("=" * 60)
    print()

    # Create and launch demo
    demo = create_demo()
    demo.launch(
        share=share,
        server_name=server_name,
        server_port=server_port
    )
