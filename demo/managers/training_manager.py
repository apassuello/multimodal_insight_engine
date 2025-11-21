"""MODULE: training_manager.py
PURPOSE: Training orchestration for Constitutional AI demo
KEY COMPONENTS:
- TrainingManager: Handles Constitutional AI training pipeline
- Critique-revision data generation
- Supervised fine-tuning with metrics tracking
- Checkpoint management at epoch boundaries
- Progress tracking and status updates
DEPENDENCIES: typing, time, src.safety.constitutional
SPECIAL NOTES: Integrates critique_revision_pipeline and supervised_finetune
"""

import time
from typing import Dict, List, Any, Optional, Tuple, Callable
from dataclasses import dataclass

from src.safety.constitutional.framework import ConstitutionalFramework
from src.safety.constitutional.critique_revision import (
    critique_revision_pipeline,
    supervised_finetune
)


@dataclass
class TrainingConfig:
    """Configuration for Constitutional AI training."""
    num_epochs: int = 2
    num_examples: int = 20
    batch_size: int = 4
    learning_rate: float = 5e-5
    num_revisions: int = 1
    mode: str = "quick_demo"


class TrainingManager:
    """
    Manages Constitutional AI training pipeline.

    Handles data generation via critique-revision,
    supervised fine-tuning, and checkpoint management.
    """

    def __init__(self):
        """Initialize training manager."""
        self.is_training = False
        self.training_data: Optional[List[Dict[str, Any]]] = None
        self.preference_pairs: Optional[List[Dict[str, Any]]] = None  # For RLAIF Phase 2
        self.pipeline_stats: Dict[str, Any] = {}
        self.metrics: Dict[str, List[float]] = {
            "losses": [],
            "epochs": []
        }
        self.current_status = "Ready"
        self.progress = 0.0

    def train_model(
        self,
        model,
        tokenizer,
        framework: ConstitutionalFramework,
        device,
        training_prompts: List[str],
        config: TrainingConfig,
        progress_callback: Optional[Callable[[str, float], None]] = None,
        checkpoint_callback: Optional[Callable[[int, Dict[str, Any]], None]] = None,
        logger=None  # type: ignore
    ) -> Tuple[Dict[str, Any], bool, str]:
        """
        Execute complete Constitutional AI training pipeline.

        Args:
            model: Model to train (will be modified in-place)
            tokenizer: Tokenizer
            framework: Constitutional framework with principles
            device: Computation device
            training_prompts: List of prompts for training data generation
            config: Training configuration
            progress_callback: Optional callback for progress updates (status, progress)
            checkpoint_callback: Optional callback for checkpointing (epoch, metrics)
            logger: Optional ContentLogger for pipeline visibility

        Returns:
            Tuple of (metrics: dict, success: bool, message: str)
        """
        if self.is_training:
            return {}, False, "✗ Training already in progress"

        try:
            self.is_training = True
            self.metrics = {"losses": [], "epochs": []}
            overall_start_time = time.time()

            # Phase 1: Generate critique-revision training data
            if progress_callback:
                progress_callback("Generating training data via critique-revision...", 0.05)

            self.current_status = "Generating training data..."
            data_start_time = time.time()

            # Limit prompts to num_examples
            prompts = training_prompts[:config.num_examples]

            pipeline_result = critique_revision_pipeline(
                prompts=prompts,
                model=model,
                tokenizer=tokenizer,
                framework=framework,
                device=device,
                num_revisions=config.num_revisions,
                logger=logger,
                collect_preference_pairs=True  # Collect pairs for future RLAIF
            )

            data_gen_time = time.time() - data_start_time

            # Handle new return format (dict with training_data and preference_pairs)
            self.training_data = pipeline_result.get('training_data', [])
            self.preference_pairs = pipeline_result.get('preference_pairs', [])
            self.pipeline_stats = pipeline_result.get('stats', {})

            if not self.training_data or len(self.training_data) == 0:
                self.is_training = False
                return {}, False, "✗ Failed to generate training data"

            if progress_callback:
                msg = f"Generated {len(self.training_data)} training examples"
                if self.preference_pairs:
                    msg += f" + {len(self.preference_pairs)} preference pairs"
                msg += f" in {data_gen_time:.1f}s"
                progress_callback(msg, 0.3)

            # Phase 2: Supervised fine-tuning
            if progress_callback:
                progress_callback("Starting supervised fine-tuning...", 0.35)

            self.current_status = "Fine-tuning model..."
            training_start_time = time.time()

            # Custom training loop with progress updates
            training_result = self._train_with_progress(
                model=model,
                tokenizer=tokenizer,
                training_data=self.training_data,
                config=config,
                device=device,
                progress_callback=progress_callback,
                checkpoint_callback=checkpoint_callback
            )

            training_time = time.time() - training_start_time
            total_time = time.time() - overall_start_time

            if not training_result:
                self.is_training = False
                return {}, False, "✗ Training failed"

            # Store metrics
            self.metrics = training_result.get("metrics", {})

            if progress_callback:
                progress_callback("Training completed!", 1.0)

            self.is_training = False
            self.current_status = "Training completed"

            # Prepare result summary
            result = {
                "training_data_size": len(self.training_data),
                "preference_pairs_size": len(self.preference_pairs) if self.preference_pairs else 0,
                "data_generation_time": data_gen_time,
                "training_time": training_time,
                "total_time": total_time,
                "metrics": self.metrics,
                "pipeline_stats": self.pipeline_stats,
                "config": {
                    "num_epochs": config.num_epochs,
                    "num_examples": config.num_examples,
                    "batch_size": config.batch_size,
                    "learning_rate": config.learning_rate
                }
            }

            message = f"✓ Phase 1 (SFT) Training completed!\n"
            message += f"Total time: {total_time:.1f}s\n"
            message += f"Training examples: {len(self.training_data)}\n"
            if self.preference_pairs:
                message += f"Preference pairs for RLAIF: {len(self.preference_pairs)}\n"
            message += f"Epochs: {config.num_epochs}\n"

            if self.metrics.get("losses"):
                final_loss = self.metrics["losses"][-1]
                initial_loss = self.metrics["losses"][0]
                message += f"Loss: {initial_loss:.4f} → {final_loss:.4f}"

            return result, True, message

        except Exception as e:
            self.is_training = False
            self.current_status = "Training failed"
            return {}, False, f"✗ Training failed: {str(e)}"

    def _train_with_progress(
        self,
        model,
        tokenizer,
        training_data: List[Dict[str, Any]],
        config: TrainingConfig,
        device,
        progress_callback: Optional[Callable[[str, float], None]] = None,
        checkpoint_callback: Optional[Callable[[int, Dict[str, Any]], None]] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Execute supervised fine-tuning with progress tracking.

        Args:
            model: Model to train
            tokenizer: Tokenizer
            training_data: Training data from critique-revision pipeline
            config: Training configuration
            device: Computation device
            progress_callback: Optional callback for progress updates
            checkpoint_callback: Optional callback for checkpointing

        Returns:
            Training result with metrics, or None if failed
        """
        try:
            import torch
            from torch.utils.data import DataLoader
            from src.safety.constitutional.critique_revision import ConstitutionalDataset

            model = model.to(device)
            model.train()

            # Create dataset and dataloader
            dataset = ConstitutionalDataset(training_data, tokenizer)
            dataloader = DataLoader(
                dataset,
                batch_size=config.batch_size,
                shuffle=True
            )

            # Optimizer
            optimizer = torch.optim.AdamW(
                model.parameters(),
                lr=config.learning_rate
            )

            # Training loop with progress tracking
            metrics = {'losses': [], 'epochs': []}
            total_batches = len(dataloader) * config.num_epochs
            completed_batches = 0

            for epoch in range(config.num_epochs):
                epoch_loss = 0.0
                batch_count = 0

                for batch_idx, batch in enumerate(dataloader):
                    input_ids = batch['input_ids'].to(device)
                    attention_mask = batch['attention_mask'].to(device)

                    # Forward pass
                    outputs = model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        labels=input_ids
                    )
                    loss = outputs.loss

                    # Backward pass
                    optimizer.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()

                    epoch_loss += loss.item()
                    batch_count += 1
                    completed_batches += 1

                    # Update progress
                    if progress_callback and batch_idx % 5 == 0:
                        progress_pct = 0.35 + (0.65 * completed_batches / total_batches)
                        status = f"Epoch {epoch+1}/{config.num_epochs} - Batch {batch_idx+1}/{len(dataloader)} - Loss: {loss.item():.4f}"
                        progress_callback(status, progress_pct)

                # Epoch completed
                avg_loss = epoch_loss / batch_count if batch_count > 0 else 0.0
                metrics['losses'].append(avg_loss)
                metrics['epochs'].append(epoch + 1)

                # Checkpoint callback
                if checkpoint_callback:
                    checkpoint_callback(epoch + 1, {"loss": avg_loss})

                # Update progress
                if progress_callback:
                    progress_pct = 0.35 + (0.65 * completed_batches / total_batches)
                    status = f"Completed Epoch {epoch+1}/{config.num_epochs} - Avg Loss: {avg_loss:.4f}"
                    progress_callback(status, progress_pct)

            return {
                'model': model,
                'metrics': metrics
            }

        except Exception as e:
            print(f"Training error: {str(e)}")
            return None

    def get_training_status(self) -> Dict[str, Any]:
        """
        Get current training status.

        Returns:
            Status information dictionary
        """
        return {
            "is_training": self.is_training,
            "current_status": self.current_status,
            "progress": self.progress,
            "metrics": self.metrics,
            "has_training_data": self.training_data is not None,
            "training_data_size": len(self.training_data) if self.training_data else 0
        }

    def get_metrics(self) -> Dict[str, List[float]]:
        """Get training metrics."""
        return self.metrics

    def clear_metrics(self) -> None:
        """Clear training metrics."""
        self.metrics = {"losses": [], "epochs": []}
        self.training_data = None
        self.preference_pairs = None
        self.pipeline_stats = {}

    def get_preference_pairs(self) -> Optional[List[Dict[str, Any]]]:
        """Get preference pairs collected during Phase 1 for RLAIF training."""
        return self.preference_pairs

    def has_preference_pairs(self) -> bool:
        """Check if preference pairs are available for RLAIF training."""
        return self.preference_pairs is not None and len(self.preference_pairs) > 0

    def train_rlaif(
        self,
        model,
        tokenizer,
        framework,
        device,
        num_ppo_steps: int = 50,
        batch_size: int = 4,
        learning_rate: float = 1e-6,
        progress_callback: Optional[Callable[[str, float], None]] = None
    ) -> Tuple[Dict[str, Any], bool, str]:
        """
        Execute RLAIF (Phase 2) training using collected preference pairs.

        This uses PPO to train the model on preference pairs collected during
        Phase 1 (critique-revision).

        Args:
            model: Model to train
            tokenizer: Tokenizer
            framework: Constitutional framework
            device: Computation device
            num_ppo_steps: Number of PPO training steps
            batch_size: Batch size for PPO
            learning_rate: Learning rate for PPO
            progress_callback: Optional progress callback

        Returns:
            Tuple of (result_dict, success, message)
        """
        if not self.has_preference_pairs():
            return {}, False, "✗ No preference pairs available. Run Phase 1 training first."

        if self.is_training:
            return {}, False, "✗ Training already in progress"

        try:
            self.is_training = True
            self.current_status = "RLAIF Training (Phase 2)..."
            start_time = time.time()

            if progress_callback:
                progress_callback("Initializing RLAIF trainer...", 0.05)

            # Import RLAIF components
            from src.safety.constitutional.trainer import RLAIFTrainer
            from src.safety.constitutional.reward_model import RewardModel, train_reward_model

            # Step 1: Train reward model on preference pairs
            if progress_callback:
                progress_callback("Training reward model on preference pairs...", 0.1)

            reward_model = RewardModel(model)
            reward_model = reward_model.to(device)

            # Train reward model
            reward_metrics = train_reward_model(
                reward_model=reward_model,
                preference_pairs=self.preference_pairs,
                tokenizer=tokenizer,
                device=device,
                num_epochs=2,
                batch_size=batch_size,
                learning_rate=learning_rate * 10  # Higher LR for reward model
            )

            if progress_callback:
                progress_callback("Reward model trained. Starting PPO...", 0.3)

            # Step 2: Initialize RLAIF trainer with reward model
            rlaif_trainer = RLAIFTrainer(
                policy_model=model,
                constitutional_framework=framework,
                reward_model=reward_model,
                learning_rate=learning_rate,
                device=device
            )

            # Step 3: Run PPO training
            # Get prompts from preference pairs
            prompts = [p['prompt'] for p in self.preference_pairs]

            ppo_results = rlaif_trainer.train(
                prompts=prompts,
                num_steps=num_ppo_steps,
                batch_size=batch_size,
                tokenizer=tokenizer
            )

            training_time = time.time() - start_time

            if progress_callback:
                progress_callback("RLAIF training completed!", 1.0)

            self.is_training = False
            self.current_status = "RLAIF Training completed"

            # Prepare results
            result = {
                "preference_pairs_used": len(self.preference_pairs),
                "ppo_steps": num_ppo_steps,
                "training_time": training_time,
                "reward_model_metrics": reward_metrics,
                "ppo_results": ppo_results
            }

            message = f"✓ Phase 2 (RLAIF) Training completed!\n"
            message += f"Preference pairs used: {len(self.preference_pairs)}\n"
            message += f"PPO steps: {num_ppo_steps}\n"
            message += f"Training time: {training_time:.1f}s\n"

            if ppo_results.get("final_avg_reward"):
                message += f"Final avg reward: {ppo_results['final_avg_reward']:.4f}"

            return result, True, message

        except ImportError as e:
            self.is_training = False
            return {}, False, f"✗ RLAIF components not available: {e}"
        except Exception as e:
            self.is_training = False
            self.current_status = "RLAIF Training failed"
            return {}, False, f"✗ RLAIF training failed: {str(e)}"

    def estimate_training_time(
        self,
        config: TrainingConfig
    ) -> Tuple[int, int]:
        """
        Estimate training time in seconds.

        Args:
            config: Training configuration

        Returns:
            Tuple of (min_seconds, max_seconds)
        """
        # Time estimates based on architecture spec
        # Each example requires ~9s (3 generations @ 3s each)
        # Plus fine-tuning overhead

        data_gen_time = config.num_examples * 9  # seconds

        # Fine-tuning time estimate
        batches_per_epoch = max(1, config.num_examples // config.batch_size)
        seconds_per_batch = 2  # Approximate
        finetuning_time = batches_per_epoch * config.num_epochs * seconds_per_batch

        total_min = int(data_gen_time + finetuning_time)
        total_max = int(total_min * 1.5)  # Add 50% buffer

        return total_min, total_max

    def format_time_estimate(
        self,
        config: TrainingConfig
    ) -> str:
        """
        Format training time estimate as human-readable string.

        Args:
            config: Training configuration

        Returns:
            Formatted time estimate string
        """
        min_sec, max_sec = self.estimate_training_time(config)

        min_min = min_sec // 60
        max_min = max_sec // 60

        if min_min == max_min:
            return f"~{min_min} minutes"
        else:
            return f"{min_min}-{max_min} minutes"
