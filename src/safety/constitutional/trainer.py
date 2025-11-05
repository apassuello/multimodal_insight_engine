"""MODULE: trainer.py
PURPOSE: RLAIF (Reinforcement Learning from AI Feedback) trainer for Constitutional AI
KEY COMPONENTS:
- RLAIFTrainer: Trainer implementing RL from AI Feedback with constitutional principles
- Constitutional feedback generation and scoring
- Actual training implementation with policy gradient
DEPENDENCIES: torch, typing, framework, evaluator, model_utils
SPECIAL NOTES: Implements scalable AI feedback for model fine-tuning
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Any, Optional, Tuple
import numpy as np
from tqdm import tqdm

from .framework import ConstitutionalFramework
from .evaluator import ConstitutionalSafetyEvaluator
from .principles import setup_default_framework


class RLAIFTrainer:
    """
    Reinforcement Learning from AI Feedback (RLAIF) trainer.

    Extends traditional RLHF by using AI-generated feedback based on
    constitutional principles, making training more scalable and consistent.
    """

    def __init__(
        self,
        policy_model: nn.Module,
        constitutional_framework: Optional[ConstitutionalFramework] = None,
        critique_model: Optional[nn.Module] = None,
        learning_rate: float = 1e-5,
        temperature: float = 1.0,
        device: Optional[torch.device] = None
    ):
        """
        Initialize the RLAIF trainer.

        Args:
            policy_model: The model being trained
            constitutional_framework: Framework for evaluation (uses default if None)
            critique_model: Optional separate model for critique (uses policy_model if None)
            learning_rate: Learning rate for policy updates
            temperature: Sampling temperature for response generation
            device: Device for training
        """
        self.policy_model = policy_model
        self.constitutional_framework = (
            constitutional_framework if constitutional_framework is not None
            else setup_default_framework()
        )
        self.critique_model = critique_model if critique_model is not None else policy_model
        self.learning_rate = learning_rate
        self.temperature = temperature

        # Setup device
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device

        # Create evaluator
        self.evaluator = ConstitutionalSafetyEvaluator(
            framework=self.constitutional_framework,
            critique_model=self.critique_model,
            use_self_critique=True
        )

        # Create optimizer
        self.optimizer = torch.optim.AdamW(
            self.policy_model.parameters(),
            lr=learning_rate
        )

        # Training statistics
        self.stats = {
            "training_iterations": 0,
            "total_prompts_processed": 0,
            "total_responses_generated": 0,
            "avg_constitutional_score": 0.0,
            "improvement_rate": 0.0
        }

    def generate_training_data(
        self,
        prompts: List[str],
        num_responses_per_prompt: int = 5,
        use_tokenizer: Optional[Any] = None
    ) -> List[Dict[str, Any]]:
        """
        Generate training data with constitutional feedback.

        Args:
            prompts: List of prompts to generate responses for
            num_responses_per_prompt: Number of response candidates per prompt
            use_tokenizer: Optional tokenizer for text processing

        Returns:
            List of training examples with responses and evaluations
        """
        training_data = []

        for prompt in tqdm(prompts, desc="Generating training data"):
            responses = []
            evaluations = []

            # Generate multiple responses per prompt
            for _ in range(num_responses_per_prompt):
                # Generate response
                response = self._generate_response(prompt, use_tokenizer)

                # Evaluate with constitutional framework
                evaluation = self.evaluator.evaluate(response)

                # Generate detailed critique
                critique = self._generate_critique(prompt, response)

                # Compute combined score
                combined_score = self._compute_combined_score(evaluation, critique)

                responses.append(response)
                evaluations.append({
                    "constitutional_eval": evaluation,
                    "critique": critique,
                    "combined_score": combined_score,
                    "flagged": evaluation["flagged"]
                })

                self.stats["total_responses_generated"] += 1

            # Add to training data
            training_data.append({
                "prompt": prompt,
                "responses": responses,
                "evaluations": evaluations,
                "best_response_idx": self._select_best_response(evaluations)
            })

            self.stats["total_prompts_processed"] += 1

        return training_data

    def _generate_response(self, prompt: str, tokenizer: Optional[Any] = None) -> str:
        """
        Generate a response from the policy model.

        Args:
            prompt: Input prompt
            tokenizer: Optional tokenizer

        Returns:
            Generated response text
        """
        try:
            from .model_utils import generate_text, GenerationConfig

            # Get tokenizer
            if tokenizer is None:
                if hasattr(self.policy_model, 'tokenizer'):
                    tokenizer = self.policy_model.tokenizer
                else:
                    return f"[No tokenizer available for {prompt[:30]}...]"

            # Generate response
            config = GenerationConfig(
                max_length=150,
                temperature=self.temperature,
                do_sample=True
            )

            response = generate_text(
                self.policy_model,
                tokenizer,
                prompt,
                config,
                device=self.device
            )

            return response

        except Exception as e:
            return f"[Generation error: {str(e)}]"

    def _generate_critique(self, prompt: str, response: str) -> str:
        """
        Generate AI critique of response.

        Args:
            prompt: Original prompt
            response: Generated response

        Returns:
            Critique text
        """
        critique_prompt = f"""Critique the following AI response based on safety, helpfulness, and accuracy:

User prompt: {prompt}

AI response: {response}

Provide a detailed analysis of any issues with respect to:
1. Harm prevention
2. Truthfulness and accuracy
3. Fairness and bias
4. Respect for autonomy

Analysis:"""

        try:
            from .model_utils import generate_text, GenerationConfig

            # Use critique model or policy model
            model = self.critique_model if self.critique_model is not None else self.policy_model

            # Get tokenizer
            if hasattr(model, 'tokenizer'):
                tokenizer = model.tokenizer
            else:
                return "[No tokenizer available for critique generation]"

            # Generate critique
            config = GenerationConfig(
                max_length=256,
                temperature=0.7,
                do_sample=True
            )

            critique = generate_text(model, tokenizer, critique_prompt, config, device=self.device)
            return critique

        except Exception as e:
            return f"[Critique generation error: {str(e)}]"

    def _compute_combined_score(
        self,
        evaluation: Dict[str, Any],
        critique: str
    ) -> float:
        """
        Compute combined score from constitutional evaluation and critique.

        Lower scores are better (0 = perfect, higher = more violations).

        Args:
            evaluation: Constitutional evaluation results
            critique: Critique text

        Returns:
            Combined score (lower is better)
        """
        # Constitutional score (weighted sum of violations)
        constitutional_score = evaluation.get("direct_evaluation", {}).get("weighted_score", 0.0)

        # Critique score (based on negative terms)
        critique_score = self._extract_score_from_critique(critique)

        # Combine scores (could use more sophisticated weighting)
        combined = constitutional_score + (critique_score * 0.5)

        return combined

    def _extract_score_from_critique(self, critique: str) -> float:
        """
        Extract numerical score from critique text.

        Args:
            critique: Critique text

        Returns:
            Score representing severity of issues (0-10 scale)
        """
        negative_terms = [
            "unsafe", "harmful", "biased", "incorrect", "misleading",
            "deceptive", "inappropriate", "problematic", "concerning",
            "violation", "issue", "unfair", "coercive", "manipulative"
        ]

        critique_lower = critique.lower()
        severity_score = sum(1 for term in negative_terms if term in critique_lower)

        # Cap at 10
        return min(severity_score, 10.0)

    def _select_best_response(self, evaluations: List[Dict[str, Any]]) -> int:
        """
        Select the best response based on evaluations.

        Args:
            evaluations: List of evaluation dictionaries

        Returns:
            Index of best response
        """
        # Select response with lowest combined score (fewer violations)
        scores = [eval_dict["combined_score"] for eval_dict in evaluations]
        return int(np.argmin(scores))

    def train_step(
        self,
        prompt: str,
        responses: List[str],
        rewards: List[float],
        tokenizer: Any
    ) -> Dict[str, float]:
        """
        Perform a single training step with constitutional feedback using policy gradient.

        Args:
            prompt: Input prompt
            responses: List of generated responses
            rewards: Reward for each response (based on constitutional compliance)
            tokenizer: Tokenizer

        Returns:
            Dictionary with training metrics
        """
        self.policy_model.train()
        self.optimizer.zero_grad()

        # Tokenize prompt
        prompt_ids = tokenizer(prompt, return_tensors="pt", padding=True)
        prompt_ids = {k: v.to(self.device) for k, v in prompt_ids.items()}

        total_loss = 0.0
        valid_samples = 0

        # Compute policy gradient loss
        for response, reward in zip(responses, rewards):
            # Tokenize response
            full_text = prompt + response
            full_ids = tokenizer(full_text, return_tensors="pt", padding=True, truncation=True, max_length=512)
            full_ids = {k: v.to(self.device) for k, v in full_ids.items()}

            # Get model outputs
            outputs = self.policy_model(**full_ids, labels=full_ids["input_ids"])

            # Extract loss for this response
            response_loss = outputs.loss

            # Weight by reward (inverse - lower score is better)
            # Convert score to reward: reward = -score (higher reward for lower violations)
            advantage = -reward

            # Policy gradient: maximize reward = minimize -reward * log_prob
            weighted_loss = response_loss * advantage

            total_loss += weighted_loss
            valid_samples += 1

        if valid_samples > 0:
            # Average loss
            loss = total_loss / valid_samples

            # Backward pass
            loss.backward()

            # Clip gradients
            torch.nn.utils.clip_grad_norm_(self.policy_model.parameters(), 1.0)

            # Update parameters
            self.optimizer.step()

            self.stats["training_iterations"] += 1

            return {
                "loss": loss.item(),
                "avg_reward": float(np.mean([-r for r in rewards])),  # Flip back for reporting
                "learning_rate": self.optimizer.param_groups[0]["lr"]
            }
        else:
            return {
                "loss": 0.0,
                "avg_reward": 0.0,
                "learning_rate": self.optimizer.param_groups[0]["lr"]
            }

    def train(
        self,
        prompts: List[str],
        num_epochs: int = 3,
        num_responses_per_prompt: int = 5,
        batch_size: int = 8,
        tokenizer: Optional[Any] = None,
        validation_prompts: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Train the policy model using constitutional AI feedback.

        Args:
            prompts: Training prompts
            num_epochs: Number of training epochs
            num_responses_per_prompt: Response candidates per prompt
            batch_size: Training batch size (for prompt batching)
            tokenizer: Tokenizer for the model
            validation_prompts: Optional validation prompts

        Returns:
            Training results and metrics
        """
        # Get tokenizer
        if tokenizer is None:
            if hasattr(self.policy_model, 'tokenizer'):
                tokenizer = self.policy_model.tokenizer
            else:
                raise ValueError("Tokenizer required for training")

        print(f"Starting Constitutional AI training for {num_epochs} epochs...")
        print(f"Training on {len(prompts)} prompts")
        print(f"Generating {num_responses_per_prompt} responses per prompt")

        training_history = {
            "epoch_losses": [],
            "epoch_rewards": [],
            "validation_scores": []
        }

        for epoch in range(num_epochs):
            print(f"\nEpoch {epoch + 1}/{num_epochs}")

            # Generate training data
            print("Generating responses and evaluating...")
            training_data = self.generate_training_data(
                prompts,
                num_responses_per_prompt,
                tokenizer
            )

            # Training loop
            epoch_losses = []
            epoch_rewards = []

            print("Training on generated data...")
            for batch_data in tqdm(training_data, desc="Training"):
                prompt = batch_data["prompt"]
                responses = batch_data["responses"]
                evaluations = batch_data["evaluations"]

                # Extract rewards (use combined scores)
                rewards = [eval_dict["combined_score"] for eval_dict in evaluations]

                # Training step
                metrics = self.train_step(prompt, responses, rewards, tokenizer)

                epoch_losses.append(metrics["loss"])
                epoch_rewards.append(metrics["avg_reward"])

            # Compute epoch metrics
            avg_loss = np.mean(epoch_losses)
            avg_reward = np.mean(epoch_rewards)

            training_history["epoch_losses"].append(avg_loss)
            training_history["epoch_rewards"].append(avg_reward)

            print(f"Epoch {epoch + 1} - Avg Loss: {avg_loss:.4f}, Avg Reward: {avg_reward:.4f}")

            # Validation if prompts provided
            if validation_prompts:
                val_score = self.validate(validation_prompts, tokenizer)
                training_history["validation_scores"].append(val_score)
                print(f"Validation Constitutional Score: {val_score:.4f}")

        # Update final statistics
        if training_history["epoch_rewards"]:
            self.stats["avg_constitutional_score"] = float(np.mean(training_history["epoch_rewards"]))

            if len(training_history["epoch_rewards"]) > 1:
                improvement = training_history["epoch_rewards"][0] - training_history["epoch_rewards"][-1]
                self.stats["improvement_rate"] = float(improvement)

        return {
            "training_history": training_history,
            "final_stats": self.stats
        }

    def validate(
        self,
        validation_prompts: List[str],
        tokenizer: Optional[Any] = None
    ) -> float:
        """
        Validate model on validation prompts.

        Args:
            validation_prompts: Prompts for validation
            tokenizer: Optional tokenizer

        Returns:
            Average constitutional score
        """
        self.policy_model.eval()
        scores = []

        with torch.no_grad():
            for prompt in validation_prompts:
                response = self._generate_response(prompt, tokenizer)
                evaluation = self.evaluator.evaluate(response)
                critique = self._generate_critique(prompt, response)
                score = self._compute_combined_score(evaluation, critique)
                scores.append(score)

        return float(np.mean(scores))

    def get_statistics(self) -> Dict[str, Any]:
        """Get training statistics."""
        return {
            **self.stats,
            "evaluator_stats": self.evaluator.get_statistics(),
            "framework_stats": self.constitutional_framework.get_statistics()
        }
