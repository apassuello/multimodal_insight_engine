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
    model,
    tokenizer,
    device: torch.device
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

    # Generate critique using model
    config = GenerationConfig(
        max_length=256,
        temperature=0.7,
        do_sample=True
    )

    try:
        critique = generate_text(model, tokenizer, critique_prompt, config, device)
        # Handle empty responses
        if not critique or critique.strip() == '':
            critique = "No specific issues identified."
        return critique
    except Exception as e:
        print(f"Warning: Critique generation failed: {e}")
        return "Error generating critique."


def generate_revision(
    prompt: str,
    response: str,
    critique: str,
    principles: List[str],
    model,
    tokenizer,
    device: torch.device
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

    Returns:
        Revised response addressing critique
    """
    revision_prompt = REVISION_TEMPLATE.format(
        prompt=prompt,
        response=response,
        critique=critique
    )

    config = GenerationConfig(
        max_length=256,
        temperature=0.7,
        do_sample=True
    )

    try:
        revision = generate_text(model, tokenizer, revision_prompt, config, device)
        # Handle empty responses - fall back to original
        if not revision or revision.strip() == '':
            revision = response
        return revision
    except Exception as e:
        print(f"Warning: Revision generation failed: {e}")
        return response  # Fall back to original


def critique_revision_pipeline(
    prompts: List[str],
    model,
    tokenizer,
    framework: ConstitutionalFramework,
    device: torch.device,
    num_revisions: int = 1
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

    Returns:
        List of training examples with revised responses
    """
    training_data = []
    principles = [p.description for p in framework.principles.values()]

    for prompt in tqdm(prompts, desc='Generating revised responses'):
        # Generate initial response
        config = GenerationConfig(max_length=150, temperature=1.0, do_sample=True)

        try:
            response = generate_text(model, tokenizer, prompt, config, device)

            # Iterative critique and revision
            for iteration in range(num_revisions):
                critique = generate_critique(prompt, response, principles, model, tokenizer, device)
                response = generate_revision(prompt, response, critique, principles, model, tokenizer, device)

            # Store training example
            training_data.append({
                'prompt': prompt,
                'response': response,  # This is the revised, improved response
                'num_revisions': num_revisions
            })
        except Exception as e:
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
    model,
    tokenizer,
    training_data: List[Dict[str, Any]],
    num_epochs: int = 3,
    batch_size: int = 8,
    learning_rate: float = 5e-5,
    device: torch.device = None
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

    Returns:
        Training metrics and fine-tuned model
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = model.to(device)
    model.train()

    # Create dataset and dataloader
    dataset = ConstitutionalDataset(training_data, tokenizer)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    # Training loop
    metrics = {'losses': [], 'epochs': []}

    for epoch in range(num_epochs):
        epoch_loss = 0
        batch_count = 0

        for batch in tqdm(dataloader, desc=f'Epoch {epoch+1}/{num_epochs}'):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)

            # Forward pass
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=input_ids)
            loss = outputs.loss

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            epoch_loss += loss.item()
            batch_count += 1

        avg_loss = epoch_loss / batch_count if batch_count > 0 else 0.0
        metrics['losses'].append(avg_loss)
        metrics['epochs'].append(epoch + 1)
        print(f'Epoch {epoch+1} - Avg Loss: {avg_loss:.4f}')

    return {
        'model': model,
        'metrics': metrics
    }
