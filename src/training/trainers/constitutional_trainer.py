"""MODULE: constitutional_trainer.py
PURPOSE: Trainer for fine-tuning models with Constitutional AI principles
KEY COMPONENTS:
- ConstitutionalTrainer: Extends LanguageModelTrainer with CAI feedback
- Iterative improvement loop with constitutional evaluation
- Integration with RLAIF for scalable training
DEPENDENCIES: torch, language_model_trainer, constitutional AI modules
SPECIAL NOTES: Implements Constitutional AI training approach for safer model outputs
"""

import os
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from .language_model_trainer import LanguageModelTrainer

# Import constitutional AI components
try:
    from src.safety.constitutional import (
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
        constitutional_loss = torch.tensor(0.0, device=self.device)
        if self.use_rlaif and self.rlaif_trainer is not None:
            # Placeholder for RLAIF loss computation
            # Real implementation would compute rewards based on constitutional compliance
            pass

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
