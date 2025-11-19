"""MODULE: critique_revision.py
PURPOSE: Critique-Revision Cycle for Constitutional AI (Phase 1: Supervised Learning)
KEY COMPONENTS:
- generate_critique: Generate constitutional critique of responses
- generate_revision: Generate improved responses addressing critiques
- critique_revision_pipeline: Complete pipeline for dataset generation
- ConstitutionalDataset: PyTorch dataset for supervised fine-tuning
- supervised_finetune: Train model on critique-revised responses
DEPENDENCIES: torch, transformers, tqdm, typing, framework, model_utils
SPECIAL NOTES: Implements Phase 1 of Constitutional AI methodology from Anthropic (2022)
"""

import torch
from torch.utils.data import DataLoader, Dataset
from typing import List, Dict, Any, Optional
from tqdm import tqdm
from transformers import PreTrainedModel, PreTrainedTokenizer

from .framework import ConstitutionalFramework
from .model_utils import generate_text, GenerationConfig

# Prompt templates from Anthropic's Constitutional AI paper
CRITIQUE_TEMPLATE = """Consider the following conversation between a human and an assistant:
Human: {prompt}

Assistant: {response}

Identify any ways in which the assistant's response is harmful, unethical, racist, sexist, toxic, dangerous, or illegal.

Specifically, evaluate the response against these principles:
{principles_text}

Critique:
"""

REVISION_TEMPLATE = """Consider the following conversation:

Human: {prompt}

Assistant: {response}

A critique of this response identified the following issues:
{critique}

Please rewrite the assistant's response to address these issues while still being helpful to the human.

Revised Response:
"""



def generate_critique(
    prompt: str,
    response: str,
    principles: List[str],
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    device: torch.device,
    logger=None  # type: ignore
) -> str:
    """
    Generate constitutional critique of a response.

    Args:
        prompt: Original user prompt
        response: Model's response to critique
        principles: List of constitutional principles to evaluate
        model: Language model for generation
        tokenizer: Model tokenizer
        device: Computation device
        logger: Optional ContentLogger for pipeline visibility

    Returns:
        Critique text identifying violations
    """
    # Format principles
    principles_text = '\n'.join([
        f'{i+1}. {p}' for i, p in enumerate(principles)
    ])

    # Build critique prompt
    critique_prompt = CRITIQUE_TEMPLATE.format(
        prompt=prompt,
        response=response,
        principles_text=principles_text
    )

    if logger:
        logger.log_stage("CRITIQUE-PROMPT", critique_prompt, truncate=400)

    # Generate critique using model
    # FIX: Use max_new_tokens to avoid probability tensor errors
    config = GenerationConfig(
        max_new_tokens=256,
        temperature=0.7,
        do_sample=True
    )

    try:
        # PERFORMANCE: Use torch.no_grad() for inference-only operations
        # Expected speedup: 10-30%, memory reduction: ~50%
        with torch.no_grad():
            critique = generate_text(model, tokenizer, critique_prompt, config, device)
        # Handle empty responses
        if not critique or critique.strip() == '':
            critique = "No specific issues identified."

        if logger:
            logger.log_stage("CRITIQUE-GENERATION", critique)

        return critique
    except (RuntimeError, ValueError, TypeError) as e:
        if logger:
            logger.log_stage("CRITIQUE-ERROR", f"Critique generation failed: {e}")
        print(f"Warning: Critique generation failed: {e}")
        return "Error generating critique."


def generate_revision(
    prompt: str,
    response: str,
    critique: str,
    principles: List[str],
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    device: torch.device,
    logger=None  # type: ignore
) -> str:
    """
    Generate revised response based on critique.

    Args:
        prompt: Original user prompt
        response: Original response that was critiqued
        critique: Critique identifying issues
        principles: Constitutional principles
        model: Language model for generation
        tokenizer: Model tokenizer
        device: Computation device
        logger: Optional ContentLogger for pipeline visibility

    Returns:
        Revised response addressing critique
    """
    revision_prompt = REVISION_TEMPLATE.format(
        prompt=prompt,
        response=response,
        critique=critique
    )

    if logger:
        logger.log_stage("REVISION-PROMPT", revision_prompt, truncate=400)

    # FIX: Use max_new_tokens to avoid probability tensor errors
    config = GenerationConfig(
        max_new_tokens=256,
        temperature=0.7,
        do_sample=True
    )

    try:
        # PERFORMANCE: Use torch.no_grad() for inference-only operations
        # Expected speedup: 10-30%, memory reduction: ~50%
        with torch.no_grad():
            revision = generate_text(model, tokenizer, revision_prompt, config, device)
        # Handle empty responses - fall back to original
        if not revision or revision.strip() == '':
            revision = response

        if logger:
            logger.log_stage("REVISION-GENERATION", revision)

        return revision
    except (RuntimeError, ValueError, TypeError) as e:
        if logger:
            logger.log_stage("REVISION-ERROR", f"Revision generation failed: {e}, using original")
        print(f"Warning: Revision generation failed: {e}")
        return response  # Fall back to original


def critique_revision_pipeline(
    prompts: List[str],
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    framework: ConstitutionalFramework,
    device: torch.device,
    num_revisions: int = 1,
    logger=None  # type: ignore
) -> List[Dict[str, Any]]:
    """
    Complete critique-revision pipeline for dataset generation.

    Args:
        prompts: List of prompts to generate data for
        model: Language model
        tokenizer: Tokenizer
        framework: Constitutional framework with principles
        device: Computation device
        num_revisions: Number of critique-revision iterations
        logger: Optional ContentLogger for pipeline visibility

    Returns:
        List of training examples with revised responses
    """
    training_data = []
    principles = [p.description for p in framework.principles.values()]

    for idx, prompt in enumerate(tqdm(prompts, desc='Generating revised responses')):
        if logger:
            logger.log_stage(
                f"TRAINING-EXAMPLE {idx + 1}/{len(prompts)}",
                f"Prompt: {prompt}"
            )

        # Generate initial response
        # FIX: Use max_new_tokens to avoid probability tensor errors
        config = GenerationConfig(max_new_tokens=150, temperature=1.0, do_sample=True)

        try:
            # PERFORMANCE: Use torch.no_grad() for inference-only operations
            # Expected speedup: 10-30%, memory reduction: ~50%
            with torch.no_grad():
                response = generate_text(model, tokenizer, prompt, config, device)

            if logger:
                logger.log_stage("INITIAL-GENERATION", response)

            # Evaluate initial response
            initial_score = framework.evaluate_text(response)

            if logger:
                violations = [
                    p for p, v in initial_score.items()
                    if isinstance(v, dict) and v.get('flagged', False)
                ]
                weighted_score = initial_score.get('weighted_score', 0.0)
                logger.log_stage(
                    "INITIAL-EVALUATION",
                    f"Violations: {violations}\nWeighted score: {weighted_score:.2f}",
                    metadata={"violations": violations, "score": weighted_score}
                )

            # Iterative critique and revision
            for iteration in range(num_revisions):
                critique = generate_critique(
                    prompt, response, principles, model, tokenizer, device, logger=logger
                )
                response = generate_revision(
                    prompt, response, critique, principles, model, tokenizer, device, logger=logger
                )

            # Evaluate revised response
            revised_score = framework.evaluate_text(response)

            # Calculate improvement (lower score is better, so positive improvement = better)
            initial_weighted_score = initial_score.get('weighted_score', 0.0)
            revised_weighted_score = revised_score.get('weighted_score', 0.0)
            improvement = initial_weighted_score - revised_weighted_score

            if logger:
                revised_violations = [
                    p for p, v in revised_score.items()
                    if isinstance(v, dict) and v.get('flagged', False)
                ]
                logger.log_stage(
                    "REVISION-EVALUATION",
                    f"Violations: {revised_violations if revised_violations else 'NONE'}\n"
                    f"Weighted score: {revised_weighted_score:.2f}",
                    metadata={
                        "violations": revised_violations,
                        "score": revised_weighted_score,
                        "improvement": improvement
                    }
                )

            # CRITICAL FIX: Only train on examples that actually improved
            # Positive improvement means revision made response better (lower violation score)
            if improvement > 0:
                # Store training example
                training_data.append({
                    'prompt': prompt,
                    'response': response,  # This is the revised, improved response
                    'num_revisions': num_revisions,
                    'improvement': improvement
                })

                if logger:
                    logger.log_stage(
                        "TRAINING-PAIR-CREATED",
                        f"✓ Training example added\n"
                        f"  Improvement: {initial_weighted_score:.2f} → "
                        f"{revised_weighted_score:.2f} ({improvement:.2f} reduction)"
                    )
            else:
                # Skip examples that didn't improve or got worse
                if logger:
                    logger.log_stage(
                        "TRAINING-PAIR-SKIPPED",
                        f"✗ Training example skipped (improvement: {improvement:+.2f})\n"
                        f"  Score: {initial_weighted_score:.2f} → {revised_weighted_score:.2f}"
                    )
        except (RuntimeError, ValueError, TypeError) as e:
            if logger:
                logger.log_stage("TRAINING-EXAMPLE-ERROR", f"Failed: {e}")
            print(f"Warning: Failed to process prompt '{prompt[:50]}...': {e}")
            continue

    return training_data


class ConstitutionalDataset(Dataset):
    """
    PyTorch dataset for Constitutional AI supervised fine-tuning.

    Wraps critique-revised training data for use with PyTorch DataLoader.
    """

    def __init__(self, data: List[Dict[str, Any]], tokenizer, max_length: int = 512):
        """
        Initialize dataset.

        Args:
            data: List of training examples from critique_revision_pipeline
            tokenizer: Tokenizer for encoding text
            max_length: Maximum sequence length
        """
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        """Return dataset size."""
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a single training example.

        Args:
            idx: Index of example

        Returns:
            Dictionary with input_ids and attention_mask tensors
        """
        item = self.data[idx]
        text = item['prompt'] + item['response']

        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        return {
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze()
        }


def supervised_finetune(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    training_data: List[Dict[str, Any]],
    num_epochs: int = 3,
    batch_size: int = 8,
    learning_rate: float = 5e-5,
    device: torch.device = None,
    use_amp: bool = True
) -> Dict[str, Any]:
    """
    Fine-tune model on critique-revised responses.

    Args:
        model: Base model to fine-tune
        tokenizer: Tokenizer
        training_data: Data from critique-revision pipeline
        num_epochs: Training epochs
        batch_size: Batch size
        learning_rate: Learning rate
        device: Computation device
        use_amp: Use automatic mixed precision for faster training (default: True)

    Returns:
        Training metrics and fine-tuned model
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Validate training data
    if not training_data or len(training_data) == 0:
        raise ValueError("Training data is empty. Cannot train model.")

    # Filter out invalid training examples
    valid_data = []
    for idx, item in enumerate(training_data):
        # Check if required fields exist and are non-empty
        if 'prompt' not in item or 'response' not in item:
            print(f"Warning: Skipping training example {idx}: missing prompt or response")
            continue

        prompt = item.get('prompt', '').strip()
        response = item.get('response', '').strip()

        if not prompt or not response:
            print(f"Warning: Skipping training example {idx}: empty prompt or response")
            continue

        # Check for NaN or None values
        if prompt == 'nan' or response == 'nan' or prompt == 'None' or response == 'None':
            print(f"Warning: Skipping training example {idx}: NaN or None value detected")
            continue

        valid_data.append(item)

    if not valid_data:
        raise ValueError(f"All {len(training_data)} training examples are invalid. Cannot train.")

    print(f"Using {len(valid_data)}/{len(training_data)} valid training examples")

    model = model.to(device)
    model.train()

    # Create dataset and dataloader
    dataset = ConstitutionalDataset(valid_data, tokenizer)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    # PERFORMANCE: Initialize GradScaler for Automatic Mixed Precision (AMP)
    # Expected speedup: 2-3x on compatible GPUs, reduced memory usage
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp and device.type == 'cuda')

    # Training loop
    metrics = {'losses': [], 'epochs': []}

    for epoch in range(num_epochs):
        epoch_loss = 0
        batch_count = 0
        nan_batches = 0

        for batch_idx, batch in enumerate(tqdm(dataloader, desc=f'Epoch {epoch+1}/{num_epochs}')):
            try:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)

                # Check for NaN in input tensors
                if torch.isnan(input_ids.float()).any() or torch.isnan(attention_mask.float()).any():
                    print(f"Warning: NaN detected in batch {batch_idx} input tensors, skipping")
                    nan_batches += 1
                    continue

                # PERFORMANCE: Use automatic mixed precision for forward pass
                # autocast() automatically handles float16/float32 conversions
                with torch.cuda.amp.autocast(enabled=use_amp and device.type == 'cuda'):
                    # Forward pass
                    outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=input_ids)
                    loss = outputs.loss

                # Check for NaN loss
                if torch.isnan(loss) or torch.isinf(loss):
                    print(f"Warning: NaN/Inf loss detected in batch {batch_idx}, skipping")
                    nan_batches += 1
                    continue

                # PERFORMANCE: Use GradScaler for backward pass with AMP
                # Scales loss to prevent gradient underflow in float16
                optimizer.zero_grad()
                scaler.scale(loss).backward()

                # Check for NaN gradients before clipping
                has_nan_grad = False
                for param in model.parameters():
                    if param.grad is not None and (torch.isnan(param.grad).any() or torch.isinf(param.grad).any()):
                        has_nan_grad = True
                        break

                if has_nan_grad:
                    print(f"Warning: NaN/Inf gradient detected in batch {batch_idx}, skipping")
                    nan_batches += 1
                    optimizer.zero_grad()
                    continue

                # Gradient clipping to prevent explosion (unscales gradients first)
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

                # Update weights with scaler
                scaler.step(optimizer)
                scaler.update()

                epoch_loss += loss.item()
                batch_count += 1

            except (RuntimeError, ValueError, TypeError) as e:
                print(f"Warning: Error processing batch {batch_idx}: {e}")
                nan_batches += 1
                continue

        if batch_count == 0:
            print(f"ERROR: Epoch {epoch+1} - All batches were invalid or produced NaN")
            # Still record the epoch with 0 loss
            metrics['losses'].append(0.0)
            metrics['epochs'].append(epoch + 1)
        else:
            avg_loss = epoch_loss / batch_count
            metrics['losses'].append(avg_loss)
            metrics['epochs'].append(epoch + 1)
            print(f'Epoch {epoch+1} - Avg Loss: {avg_loss:.4f} ({batch_count} batches, {nan_batches} skipped)')

    return {
        'model': model,
        'metrics': metrics
    }
