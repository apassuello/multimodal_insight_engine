"""MODULE: constitutional_trainer.py
PURPOSE: Trainer for fine-tuning models with Constitutional AI principles
KEY COMPONENTS:
- ConstitutionalTrainer: Extends LanguageModelTrainer with CAI feedback
- Iterative improvement loop with constitutional evaluation
- Integration with RLAIF for scalable training
DEPENDENCIES: torch, language_model_trainer, constitutional AI modules
SPECIAL NOTES: Implements Constitutional AI training approach for safer model outputs
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Union, Any, Tuple
import time
import math
import numpy as np
import matplotlib.pyplot as plt
import os
from tqdm import tqdm

from .language_model_trainer import LanguageModelTrainer

# Import constitutional AI components
try:
    from src.safety.constitutional import (
        ConstitutionalFramework,
        ConstitutionalSafetyEvaluator,
        RLAIFTrainer,
        setup_default_framework,
    )
    CONSTITUTIONAL_AI_AVAILABLE = True
except ImportError:
    CONSTITUTIONAL_AI_AVAILABLE = False


class ConstitutionalTrainer(LanguageModelTrainer):
    """
    Trainer that extends LanguageModelTrainer with Constitutional AI principles.

    This trainer adds:
    1. Constitutional evaluation of model outputs
    2. Iterative improvement based on constitutional feedback
    3. RLAIF (Reinforcement Learning from AI Feedback) integration
    4. Constitutional compliance metrics tracking
    """

    def __init__(
        self,
        model: nn.Module,
        train_dataloader: torch.utils.data.DataLoader,
        val_dataloader: Optional[torch.utils.data.DataLoader] = None,
        learning_rate: float = 5e-5,
        weight_decay: float = 0.01,
        warmup_steps: int = 1000,
        max_grad_norm: float = 1.0,
        device: Optional[torch.device] = None,
        log_dir: str = "logs/constitutional",
        constitutional_framework: Optional[Any] = None,
        use_rlaif: bool = False,
        critique_model: Optional[nn.Module] = None,
        constitutional_weight: float = 0.5,
        **kwargs
    ):
        """
        Initialize the constitutional trainer.

        Args:
            model: The language model to train
            train_dataloader: DataLoader for training data
            val_dataloader: Optional DataLoader for validation data
            learning_rate: Peak learning rate after warmup
            weight_decay: Weight decay for regularization
            warmup_steps: Number of warmup steps
            max_grad_norm: Maximum gradient norm for clipping
            device: Device to train on
            log_dir: Directory for logging
            constitutional_framework: Constitutional framework (uses default if None)
            use_rlaif: Whether to use RLAIF training
            critique_model: Optional critique model (uses main model if None)
            constitutional_weight: Weight for constitutional loss (0-1)
            **kwargs: Additional arguments for base trainer
        """
        # Initialize base trainer
        super().__init__(
            model=model,
            train_dataloader=train_dataloader,
            val_dataloader=val_dataloader,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            warmup_steps=warmup_steps,
            max_grad_norm=max_grad_norm,
            device=device,
            log_dir=log_dir,
            **kwargs
        )

        # Check if constitutional AI is available
        if not CONSTITUTIONAL_AI_AVAILABLE:
            raise ImportError(
                "Constitutional AI components not available. "
                "Please ensure the constitutional module is properly installed."
            )

        # Setup constitutional framework
        if constitutional_framework is None:
            self.constitutional_framework = setup_default_framework()
        else:
            self.constitutional_framework = constitutional_framework

        # Setup constitutional evaluator
        self.constitutional_evaluator = ConstitutionalSafetyEvaluator(
            framework=self.constitutional_framework,
            critique_model=critique_model,
            use_self_critique=(critique_model is not None)
        )

        # Setup RLAIF trainer if requested
        self.use_rlaif = use_rlaif
        self.rlaif_trainer = None
        if use_rlaif:
            self.rlaif_trainer = RLAIFTrainer(
                policy_model=model,
                constitutional_framework=self.constitutional_framework,
                critique_model=critique_model,
                learning_rate=learning_rate,
                device=self.device
            )

        # Constitutional training parameters
        self.constitutional_weight = constitutional_weight
        self.critique_model = critique_model

        # Additional tracking metrics
        self.constitutional_scores = []
        self.principle_violation_history = []

    def train_step(
        self,
        batch: Dict[str, torch.Tensor],
        step: int
    ) -> Dict[str, float]:
        """
        Perform a single training step with constitutional feedback.

        Args:
            batch: Training batch
            step: Current training step

        Returns:
            Dictionary with training metrics
        """
        self.model.train()
        self.optimizer.zero_grad()

        # Standard language modeling loss
        lm_loss = self._compute_language_modeling_loss(batch)

        # Constitutional loss (if using RLAIF)
        # Compute constitutional loss periodically to balance computation cost
        constitutional_loss = torch.tensor(0.0, device=self.device)
        if self.use_rlaif and self.rlaif_trainer is not None and step % 10 == 0:
            # Extract prompts from batch and generate responses for evaluation
            constitutional_loss = self._compute_constitutional_loss(batch)

        # Combined loss
        total_loss = lm_loss + (self.constitutional_weight * constitutional_loss)

        # Backward pass
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
        self.optimizer.step()
        self.scheduler.step()

        # Track metrics
        metrics = {
            "total_loss": total_loss.item(),
            "lm_loss": lm_loss.item(),
            "constitutional_loss": constitutional_loss.item(),
            "learning_rate": self.optimizer.param_groups[0]["lr"]
        }

        return metrics

    def _compute_language_modeling_loss(
        self,
        batch: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """
        Compute standard language modeling loss.

        Args:
            batch: Training batch

        Returns:
            Loss tensor
        """
        # Extract inputs
        if isinstance(batch, dict):
            input_ids = batch.get("input_ids", batch.get("inputs", None))
            labels = batch.get("labels", batch.get("targets", input_ids))
        else:
            input_ids = batch
            labels = batch

        # Move to device
        input_ids = input_ids.to(self.device)
        labels = labels.to(self.device)

        # Forward pass
        outputs = self.model(input_ids)

        # Compute loss
        if hasattr(outputs, "loss") and outputs.loss is not None:
            loss = outputs.loss
        else:
            # Compute cross-entropy loss
            logits = outputs.logits if hasattr(outputs, "logits") else outputs
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                labels.view(-1),
                ignore_index=-100
            )

        return loss

    def _compute_constitutional_loss(
        self,
        batch: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """
        Compute constitutional loss by evaluating generated responses.

        This method:
        1. Extracts prompts from the batch
        2. Generates responses using the current model
        3. Evaluates responses with constitutional evaluator
        4. Returns loss based on constitutional violations

        Args:
            batch: Training batch

        Returns:
            Constitutional loss tensor
        """
        try:
            # Extract prompts from batch
            prompts = self._extract_prompts_from_batch(batch, max_samples=4)

            if not prompts:
                return torch.tensor(0.0, device=self.device)

            # Generate responses for evaluation
            responses = []
            with torch.no_grad():
                for prompt in prompts:
                    response = self._generate_response_for_evaluation(prompt)
                    responses.append(response)

            # Evaluate responses
            violation_scores = []
            for prompt, response in zip(prompts, responses):
                eval_result = self.constitutional_evaluator.evaluate(response)

                # Extract violation score (weighted_score from direct evaluation)
                direct_eval = eval_result.get("direct_evaluation", {})
                score = direct_eval.get("weighted_score", 0.0)
                violation_scores.append(score)

            # Convert to loss (higher violations = higher loss)
            if violation_scores:
                avg_violation = float(np.mean(violation_scores))
                constitutional_loss = torch.tensor(avg_violation, device=self.device)
            else:
                constitutional_loss = torch.tensor(0.0, device=self.device)

            return constitutional_loss

        except Exception as e:
            # If constitutional loss computation fails, return zero loss
            # and continue training
            print(f"Warning: Constitutional loss computation failed: {e}")
            return torch.tensor(0.0, device=self.device)

    def _extract_prompts_from_batch(
        self,
        batch: Dict[str, torch.Tensor],
        max_samples: int = 4
    ) -> List[str]:
        """
        Extract prompts from training batch for constitutional evaluation.

        Args:
            batch: Training batch
            max_samples: Maximum number of prompts to extract

        Returns:
            List of prompt strings
        """
        try:
            # Check if model has tokenizer
            if not hasattr(self.model, 'tokenizer'):
                return []

            tokenizer = self.model.tokenizer

            # Extract input_ids
            if isinstance(batch, dict):
                input_ids = batch.get("input_ids", batch.get("inputs", None))
            else:
                input_ids = batch

            if input_ids is None:
                return []

            # Move to CPU for decoding
            input_ids = input_ids.cpu()

            # Decode prompts (take first max_samples)
            prompts = []
            num_samples = min(max_samples, input_ids.size(0))

            for i in range(num_samples):
                # Take first half of sequence as prompt
                seq_len = input_ids.size(1)
                prompt_len = seq_len // 2
                prompt_ids = input_ids[i, :prompt_len]

                # Decode to text
                prompt_text = tokenizer.decode(prompt_ids, skip_special_tokens=True)
                prompts.append(prompt_text)

            return prompts

        except Exception as e:
            print(f"Warning: Failed to extract prompts from batch: {e}")
            return []

    def _generate_response_for_evaluation(
        self,
        prompt: str,
        max_length: int = 100
    ) -> str:
        """
        Generate a response for constitutional evaluation.

        Args:
            prompt: Input prompt
            max_length: Maximum generation length

        Returns:
            Generated response text
        """
        try:
            # Check if model has tokenizer and generation capability
            if not hasattr(self.model, 'tokenizer'):
                return ""

            tokenizer = self.model.tokenizer

            # Tokenize prompt
            inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            # Generate response
            self.model.eval()
            with torch.no_grad():
                if hasattr(self.model, 'generate'):
                    # Use model's generate method if available
                    output_ids = self.model.generate(
                        **inputs,
                        max_length=max_length,
                        do_sample=True,
                        temperature=0.9,
                        top_p=0.9,
                        pad_token_id=tokenizer.pad_token_id if hasattr(tokenizer, 'pad_token_id') else 0
                    )
                else:
                    # Fallback: use forward pass for one token
                    outputs = self.model(**inputs)
                    logits = outputs.logits if hasattr(outputs, 'logits') else outputs
                    output_ids = inputs['input_ids']

            self.model.train()

            # Decode response
            response = tokenizer.decode(output_ids[0], skip_special_tokens=True)

            # Remove prompt from response
            if response.startswith(prompt):
                response = response[len(prompt):].strip()

            return response

        except Exception as e:
            print(f"Warning: Failed to generate response for evaluation: {e}")
            return ""

    def evaluate_constitutional_compliance(
        self,
        texts: List[str]
    ) -> Dict[str, Any]:
        """
        Evaluate texts for constitutional compliance.

        Args:
            texts: List of texts to evaluate

        Returns:
            Dictionary with compliance metrics
        """
        evaluations = []
        for text in texts:
            eval_result = self.constitutional_evaluator.evaluate(text)
            evaluations.append(eval_result)

        # Aggregate metrics
        num_flagged = sum(1 for ev in evaluations if ev["flagged"])
        avg_weighted_score = np.mean([
            ev.get("direct_evaluation", {}).get("weighted_score", 0.0)
            for ev in evaluations
        ])

        # Count violations by principle
        principle_violations = {}
        for ev in evaluations:
            flagged_principles = ev.get("direct_evaluation", {}).get("flagged_principles", [])
            for principle in flagged_principles:
                principle_violations[principle] = principle_violations.get(principle, 0) + 1

        return {
            "total_evaluated": len(texts),
            "num_flagged": num_flagged,
            "compliance_rate": 1.0 - (num_flagged / len(texts)) if texts else 1.0,
            "avg_weighted_score": avg_weighted_score,
            "principle_violations": principle_violations
        }

    def train(
        self,
        num_epochs: int,
        eval_interval: int = 1000,
        save_interval: int = 5000,
        eval_samples: int = 100
    ) -> Dict[str, Any]:
        """
        Train the model with constitutional AI supervision.

        Args:
            num_epochs: Number of training epochs
            eval_interval: Steps between evaluations
            save_interval: Steps between checkpoints
            eval_samples: Number of samples for constitutional evaluation

        Returns:
            Training history
        """
        print(f"Starting Constitutional AI training for {num_epochs} epochs...")
        print(f"Constitutional principles: {self.constitutional_framework.get_active_principles()}")

        self.num_epochs = num_epochs
        training_history = {
            "train_losses": [],
            "val_losses": [],
            "constitutional_scores": [],
            "compliance_rates": []
        }

        for epoch in range(num_epochs):
            print(f"\n{'='*60}")
            print(f"Epoch {epoch + 1}/{num_epochs}")
            print(f"{'='*60}")

            epoch_losses = []
            epoch_metrics = []

            # Training loop
            progress_bar = tqdm(self.train_dataloader, desc=f"Epoch {epoch + 1}")
            for step, batch in enumerate(progress_bar):
                # Training step
                metrics = self.train_step(batch, self.global_step)
                epoch_losses.append(metrics["total_loss"])
                epoch_metrics.append(metrics)

                # Update progress bar
                progress_bar.set_postfix({
                    "loss": f"{metrics['total_loss']:.4f}",
                    "lr": f"{metrics['learning_rate']:.7f}"
                })

                # Log training step
                self._log_training_step(metrics["total_loss"], metrics["learning_rate"], self.global_step)

                self.global_step += 1

                # Periodic evaluation
                if self.global_step % eval_interval == 0:
                    self._evaluate_and_log(eval_samples)

                # Save checkpoint
                if self.global_step % save_interval == 0:
                    self._save_checkpoint(epoch, metrics["total_loss"])

            # Epoch summary
            avg_loss = np.mean(epoch_losses)
            print(f"\nEpoch {epoch + 1} Summary:")
            print(f"  Average Loss: {avg_loss:.4f}")

            training_history["train_losses"].append(avg_loss)

            # Validation if available
            if self.val_dataloader is not None:
                val_loss = self._validate()
                training_history["val_losses"].append(val_loss)
                print(f"  Validation Loss: {val_loss:.4f}")

            # Constitutional evaluation
            if epoch % max(1, num_epochs // 5) == 0:  # Evaluate every 20% of training
                print(f"\n  Evaluating constitutional compliance...")
                # This would need actual text generation capability
                # For now, we track the framework statistics
                framework_stats = self.constitutional_framework.get_statistics()
                print(f"  Framework Statistics: {framework_stats}")

        return training_history

    def _evaluate_and_log(self, num_samples: int = 100):
        """Evaluate constitutional compliance and log results."""
        # This would need actual text generation
        # For now, just log framework statistics
        stats = self.constitutional_evaluator.get_statistics()
        print(f"\n  Constitutional Evaluator Stats: {stats}")

    def _save_checkpoint(self, epoch: int, loss: float):
        """Save training checkpoint."""
        checkpoint_path = os.path.join(
            self.log_dir,
            f"checkpoint_epoch_{epoch}_step_{self.global_step}.pt"
        )
        self.model.save(
            checkpoint_path,
            optimizer=self.optimizer,
            epoch=epoch,
            loss=loss,
            additional_info={
                "global_step": self.global_step,
                "constitutional_framework": self.constitutional_framework.name,
                "constitutional_scores": self.constitutional_scores[-100:] if self.constitutional_scores else []
            }
        )

    def _validate(self) -> float:
        """Run validation and return average loss."""
        self.model.eval()
        val_losses = []

        with torch.no_grad():
            for batch in self.val_dataloader:
                loss = self._compute_language_modeling_loss(batch)
                val_losses.append(loss.item())

        return float(np.mean(val_losses))

    def get_training_summary(self) -> Dict[str, Any]:
        """Get comprehensive training summary including constitutional metrics."""
        base_summary = {
            "global_step": self.global_step,
            "train_losses": self.train_losses[-100:] if self.train_losses else [],
            "val_losses": self.val_losses[-100:] if self.val_losses else [],
        }

        constitutional_summary = {
            "constitutional_scores": self.constitutional_scores[-100:] if self.constitutional_scores else [],
            "evaluator_stats": self.constitutional_evaluator.get_statistics(),
            "framework_stats": self.constitutional_framework.get_statistics(),
        }

        return {**base_summary, **constitutional_summary}
