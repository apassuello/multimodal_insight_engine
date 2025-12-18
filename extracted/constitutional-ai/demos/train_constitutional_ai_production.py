#!/usr/bin/env python3
"""
Production Constitutional AI Training Script

This is the REAL training script for Constitutional AI, NOT the educational demo.

Key Differences from demo_constitutional_ai.py:
- ✅ Actually performs Phase 1 fine-tuning (not skipped)
- ✅ Uses full dataset for Phase 2 (not limited to 20 examples)
- ✅ Trains for proper number of epochs (3-5, not 1)
- ✅ Includes validation and checkpointing
- ✅ Production-scale hyperparameters

This script will take HOURS to complete (4-14 hours typical).
If you want a quick demo, use demo_constitutional_ai.py instead.

Usage:
    python train_constitutional_ai_production.py --config config.json

Requirements:
    pip install torch transformers tqdm accelerate

WARNING: This script performs REAL training. It will:
- Use significant GPU memory (~8GB+)
- Take many hours to complete
- Require hundreds/thousands of prompts
- Actually update model parameters

Author: Constitutional AI Implementation Team
Date: 2025-11-06
"""

import argparse
import json
import torch
import logging
from pathlib import Path
from typing import List, Dict, Optional
from dataclasses import dataclass, asdict
import sys

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('training.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


@dataclass
class TrainingConfig:
    """Production training configuration."""

    # Model settings
    base_model: str = "gpt2"
    device: str = "auto"  # auto, cuda, cpu

    # Data settings
    prompts_file: str = "data/constitutional_prompts.json"
    min_prompts: int = 500  # Minimum for meaningful training
    train_split: float = 0.8
    val_split: float = 0.1
    test_split: float = 0.1

    # Phase 1: Supervised Learning
    phase1_enabled: bool = True
    phase1_num_epochs: int = 3
    phase1_batch_size: int = 8
    phase1_learning_rate: float = 5e-5
    phase1_max_length: int = 512
    phase1_save_dir: str = "models/phase1"

    # Phase 2a: Preference Generation
    phase2_enabled: bool = True
    num_responses_per_prompt: int = 3  # Generate N responses to compare
    min_preference_pairs: int = 500  # Minimum for good reward model

    # Phase 2b: Reward Model Training
    reward_model_epochs: int = 3
    reward_model_batch_size: int = 4
    reward_model_learning_rate: float = 1e-5
    reward_model_save_dir: str = "models/reward_model"
    target_reward_accuracy: float = 0.75  # Stop if this accuracy reached

    # Phase 2c: PPO Training
    ppo_num_steps: int = 100
    ppo_batch_size: int = 4
    ppo_epochs_per_batch: int = 4
    ppo_learning_rate: float = 1e-5
    ppo_clip_epsilon: float = 0.2
    ppo_kl_penalty: float = 0.1
    ppo_max_kl: float = 0.5  # Stop if KL divergence exceeds this
    ppo_save_dir: str = "models/ppo"

    # Checkpointing
    checkpoint_every: int = 10
    save_best_only: bool = True

    # Logging
    log_every: int = 10

    def validate(self):
        """Validate configuration."""
        errors = []

        if self.min_prompts < 100:
            errors.append(f"min_prompts ({self.min_prompts}) should be >= 100 for meaningful training")

        if self.min_preference_pairs < 200:
            errors.append(f"min_preference_pairs ({self.min_preference_pairs}) should be >= 200 for good reward model")

        if self.train_split + self.val_split + self.test_split != 1.0:
            errors.append(f"Data splits must sum to 1.0, got {self.train_split + self.val_split + self.test_split}")

        if self.target_reward_accuracy < 0.7:
            logger.warning(f"target_reward_accuracy ({self.target_reward_accuracy}) is low. Consider >= 0.75")

        if errors:
            raise ValueError("Configuration validation failed:\n" + "\n".join(f"  - {e}" for e in errors))

        logger.info("✅ Configuration validated")


def load_config(config_path: Optional[str] = None) -> TrainingConfig:
    """Load configuration from file or use defaults."""
    if config_path and Path(config_path).exists():
        logger.info(f"Loading configuration from {config_path}")
        with open(config_path) as f:
            config_dict = json.load(f)
        config = TrainingConfig(**config_dict)
    else:
        logger.info("Using default configuration")
        config = TrainingConfig()

    config.validate()
    return config


def setup_device(device_str: str) -> torch.device:
    """Setup computation device."""
    if device_str == "auto":
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(device_str)

    logger.info(f"Using device: {device}")
    if device.type == 'cuda':
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
        logger.info(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")

    return device


def load_prompts(config: TrainingConfig) -> Dict[str, List[str]]:
    """Load and split prompts into train/val/test."""
    prompts_file = Path(config.prompts_file)

    if not prompts_file.exists():
        raise FileNotFoundError(
            f"Prompts file not found: {prompts_file}\n"
            f"Generate prompts with: python scripts/generate_constitutional_prompts.py"
        )

    logger.info(f"Loading prompts from {prompts_file}")
    with open(prompts_file) as f:
        data = json.load(f)
        prompts = data.get('prompts', [])

    if len(prompts) < config.min_prompts:
        raise ValueError(
            f"Only {len(prompts)} prompts found, but min_prompts={config.min_prompts}.\n"
            f"Generate more prompts with: python scripts/generate_constitutional_prompts.py --num-prompts {config.min_prompts}"
        )

    # Split data
    total = len(prompts)
    train_end = int(total * config.train_split)
    val_end = train_end + int(total * config.val_split)

    splits = {
        'train': prompts[:train_end],
        'val': prompts[train_end:val_end],
        'test': prompts[val_end:]
    }

    logger.info(f"Dataset split:")
    logger.info(f"  Train: {len(splits['train'])} prompts")
    logger.info(f"  Val:   {len(splits['val'])} prompts")
    logger.info(f"  Test:  {len(splits['test'])} prompts")
    logger.info(f"  Total: {total} prompts")

    return splits


def phase_1_supervised_learning(
    prompts: Dict[str, List[str]],
    config: TrainingConfig,
    device: torch.device
) -> str:
    """
    Phase 1: REAL Supervised Fine-Tuning with Critique-Revision.

    This actually performs fine-tuning, unlike the demo script.
    """
    logger.info("=" * 80)
    logger.info("PHASE 1: SUPERVISED LEARNING (REAL TRAINING)")
    logger.info("=" * 80)
    logger.info("")
    logger.info("⚠️  This will perform ACTUAL fine-tuning and take hours.")
    logger.info("⚠️  NOT a quick demo - this is production training.")
    logger.info("")

    from src.safety.constitutional import (
        setup_default_framework,
        generate_critique,
        generate_revision,
        supervised_finetune
    )
    from src.safety.constitutional.model_utils import load_model, generate_text, GenerationConfig
    from src.data.constitutional_dataset import ConstitutionalTrainingDataset
    from torch.utils.data import DataLoader

    # Load model
    logger.info("[1/6] Loading base model...")
    model, tokenizer = load_model(config.base_model, device=device)
    framework = setup_default_framework()
    principles = [p.description for p in framework.principles]
    logger.info(f"✓ Model: {config.base_model}")
    logger.info(f"✓ Principles: {len(principles)}")

    # Generate training data
    logger.info("\n[2/6] Generating critique-revision data...")
    logger.info(f"Processing {len(prompts['train'])} training prompts...")
    logger.info("This will take 1-2 hours...")

    critique_revision_data = []
    generation_config = GenerationConfig(max_length=150, temperature=0.8, do_sample=True)

    for i, prompt in enumerate(prompts['train'], 1):
        if i % config.log_every == 0:
            logger.info(f"  Progress: {i}/{len(prompts['train'])}")

        # Generate initial response
        response = generate_text(model, tokenizer, prompt, generation_config, device)

        # Generate critique
        critique = generate_critique(
            prompt=prompt,
            response=response,
            principles=principles,
            model=model,
            tokenizer=tokenizer,
            device=device
        )

        # Generate revision
        revision = generate_revision(
            prompt=prompt,
            response=response,
            critique=critique,
            principles=principles,
            model=model,
            tokenizer=tokenizer,
            device=device
        )

        critique_revision_data.append({
            'prompt': prompt,
            'original_response': response,
            'critique': critique,
            'revised_response': revision
        })

    logger.info(f"✓ Generated {len(critique_revision_data)} critique-revision pairs")

    # Save critique-revision data
    logger.info("\n[3/6] Saving critique-revision data...")
    save_dir = Path(config.phase1_save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    with open(save_dir / 'critique_revision_data.json', 'w') as f:
        json.dump(critique_revision_data, f, indent=2)
    logger.info(f"✓ Saved to {save_dir / 'critique_revision_data.json'}")

    # Create dataset
    logger.info("\n[4/6] Creating training dataset...")
    train_dataset = ConstitutionalTrainingDataset(
        prompts=[d['prompt'] for d in critique_revision_data],
        responses=[d['revised_response'] for d in critique_revision_data],
        tokenizer=tokenizer,
        max_length=config.phase1_max_length
    )
    logger.info(f"✓ Dataset size: {len(train_dataset)} examples")

    # ⚠️ IMPORTANT: This is the REAL fine-tuning step (not skipped like in demo)
    logger.info("\n[5/6] Fine-tuning model on revised responses...")
    logger.info("⚠️  REAL TRAINING - This will take 2-4 hours")
    logger.info(f"  Epochs: {config.phase1_num_epochs}")
    logger.info(f"  Batch size: {config.phase1_batch_size}")
    logger.info(f"  Learning rate: {config.phase1_learning_rate}")

    finetuned_model = supervised_finetune(
        model=model,
        dataset=train_dataset,
        tokenizer=tokenizer,
        num_epochs=config.phase1_num_epochs,
        batch_size=config.phase1_batch_size,
        learning_rate=config.phase1_learning_rate,
        device=device,
        save_dir=str(save_dir),
        checkpoint_every=config.checkpoint_every
    )

    logger.info("✓ Fine-tuning complete!")

    # Save final model
    logger.info("\n[6/6] Saving fine-tuned model...")
    model_path = save_dir / "final_model"
    finetuned_model.save_pretrained(model_path)
    tokenizer.save_pretrained(model_path)
    logger.info(f"✓ Model saved to {model_path}")

    logger.info("\n✅ Phase 1 complete!")
    logger.info(f"   Fine-tuned model: {model_path}")

    return str(model_path)


def phase_2_reinforcement_learning(
    prompts: Dict[str, List[str]],
    phase1_model_path: str,
    config: TrainingConfig,
    device: torch.device
) -> str:
    """
    Phase 2: REAL RLAIF Training.

    Uses FULL dataset, not limited to 20 examples like demo.
    """
    logger.info("\n" + "=" * 80)
    logger.info("PHASE 2: REINFORCEMENT LEARNING (REAL TRAINING)")
    logger.info("=" * 80)
    logger.info("")
    logger.info("⚠️  This will train with FULL dataset and take hours.")
    logger.info("")

    from src.safety.constitutional import (
        setup_default_framework,
        RewardModel,
        PPOTrainer,
        generate_preference_pairs,
        train_reward_model
    )
    from src.safety.constitutional.model_utils import load_model
    import torch.nn as nn

    # Load Phase 1 model
    logger.info("[1/4] Loading Phase 1 model...")
    policy_model, tokenizer = load_model(phase1_model_path, device=device)
    framework = setup_default_framework()
    logger.info(f"✓ Loaded from {phase1_model_path}")

    # Generate preference pairs
    logger.info("\n[2/4] Generating preference pairs...")
    logger.info(f"Processing {len(prompts['train'])} prompts...")
    logger.info(f"Generating {config.num_responses_per_prompt} responses per prompt...")
    logger.info("This will take 2-3 hours...")

    preference_data = generate_preference_pairs(
        prompts=prompts['train'],
        model=policy_model,
        tokenizer=tokenizer,
        framework=framework,
        device=device,
        num_responses_per_prompt=config.num_responses_per_prompt
    )

    if len(preference_data) < config.min_preference_pairs:
        logger.warning(f"Only {len(preference_data)} preference pairs generated")
        logger.warning(f"Minimum recommended: {config.min_preference_pairs}")
        logger.warning("Consider using more prompts or increasing num_responses_per_prompt")

    logger.info(f"✓ Generated {len(preference_data)} preference pairs")

    # Save preference data
    pref_save_dir = Path(config.reward_model_save_dir)
    pref_save_dir.mkdir(parents=True, exist_ok=True)
    with open(pref_save_dir / 'preference_data.json', 'w') as f:
        json.dump(preference_data, f, indent=2)
    logger.info(f"✓ Saved to {pref_save_dir / 'preference_data.json'}")

    # Train reward model
    logger.info("\n[3/4] Training reward model...")
    logger.info("⚠️  REAL TRAINING with FULL dataset")
    logger.info(f"  Training examples: {len(preference_data)}")  # NOT limited to 20!
    logger.info(f"  Epochs: {config.reward_model_epochs}")  # NOT limited to 1!
    logger.info(f"  Target accuracy: {config.target_reward_accuracy:.1%}")
    logger.info("This will take 2-4 hours...")

    base_model_for_reward, _ = load_model(phase1_model_path, device=device)
    reward_model = RewardModel(base_model_for_reward, hidden_size=768)
    reward_model = reward_model.to(device)

    # Split preference data
    train_size = int(len(preference_data) * 0.9)
    train_prefs = preference_data[:train_size]
    val_prefs = preference_data[train_size:]

    metrics = train_reward_model(
        reward_model=reward_model,
        training_data=train_prefs,  # ✅ Use ALL training data
        tokenizer=tokenizer,
        num_epochs=config.reward_model_epochs,  # ✅ Full epochs
        batch_size=config.reward_model_batch_size,
        learning_rate=config.reward_model_learning_rate,
        device=device,
        validation_data=val_prefs
    )

    final_accuracy = metrics['accuracy'][-1]
    logger.info(f"✓ Training complete!")
    logger.info(f"  Final accuracy: {final_accuracy:.2%}")

    if final_accuracy < config.target_reward_accuracy:
        logger.warning(f"Accuracy {final_accuracy:.2%} < target {config.target_reward_accuracy:.1%}")
        logger.warning("Consider training longer or using more data")

    # Save reward model
    reward_model_path = pref_save_dir / "reward_model.pt"
    torch.save(reward_model.state_dict(), reward_model_path)
    logger.info(f"✓ Reward model saved to {reward_model_path}")

    # PPO Training
    logger.info("\n[4/4] PPO training...")
    logger.info("⚠️  REAL PPO TRAINING")
    logger.info(f"  Prompts: {len(prompts['train'])}")  # NOT limited to 10!
    logger.info(f"  Steps: {config.ppo_num_steps}")
    logger.info(f"  Epochs per batch: {config.ppo_epochs_per_batch}")
    logger.info("This will take 4-8 hours...")

    # Create value model
    class ValueModel(nn.Module):
        def __init__(self, base_model, hidden_size=768):
            super().__init__()
            self.base_model = base_model
            self.value_head = nn.Sequential(
                nn.Linear(hidden_size, 256),
                nn.ReLU(),
                nn.Linear(256, 1)
            )

        def forward(self, input_ids, attention_mask):
            outputs = self.base_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True
            )
            hidden_states = outputs.hidden_states[-1]
            sequence_lengths = attention_mask.sum(dim=1) - 1
            last_hidden = hidden_states[torch.arange(hidden_states.size(0)), sequence_lengths]
            value = self.value_head(last_hidden).squeeze(-1)
            return value

    value_model_base, _ = load_model(phase1_model_path, device=device)
    value_model = ValueModel(value_model_base, hidden_size=768)
    value_model = value_model.to(device)

    # Initialize PPO trainer
    ppo_trainer = PPOTrainer(
        policy_model=policy_model,
        value_model=value_model,
        reward_model=reward_model,
        tokenizer=tokenizer,
        device=device,
        learning_rate=config.ppo_learning_rate,
        clip_epsilon=config.ppo_clip_epsilon,
        kl_penalty=config.ppo_kl_penalty
    )

    # Train with checkpointing
    ppo_save_dir = Path(config.ppo_save_dir)
    ppo_save_dir.mkdir(parents=True, exist_ok=True)

    results = ppo_trainer.train(
        prompts=prompts['train'],  # ✅ Use ALL prompts
        num_steps=config.ppo_num_steps,
        batch_size=config.ppo_batch_size,
        num_epochs_per_batch=config.ppo_epochs_per_batch,
        checkpoint_dir=str(ppo_save_dir) if config.checkpoint_every else None,
        checkpoint_freq=config.checkpoint_every
    )

    logger.info("✓ PPO training complete!")
    logger.info(f"  Final reward: {results['final_stats']['mean_rewards'][-1]:.4f}")
    logger.info(f"  Final KL divergence: {results['final_stats']['kl_divergences'][-1]:.4f}")

    # Check KL divergence
    final_kl = results['final_stats']['kl_divergences'][-1]
    if final_kl > config.ppo_max_kl:
        logger.warning(f"KL divergence {final_kl:.4f} > max {config.ppo_max_kl}")
        logger.warning("Model may have drifted too far from reference")

    # Save final model
    final_model_path = ppo_save_dir / "final_model"
    policy_model.save_pretrained(final_model_path)
    tokenizer.save_pretrained(final_model_path)
    logger.info(f"✓ Final model saved to {final_model_path}")

    logger.info("\n✅ Phase 2 complete!")
    return str(final_model_path)


def main():
    parser = argparse.ArgumentParser(
        description="Production Constitutional AI Training",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
⚠️  WARNING: This is PRODUCTION training, not a quick demo.

Expected time: 4-14 hours depending on dataset size
Expected GPU memory: 8-16 GB
Expected disk space: 10-50 GB for models and checkpoints

For a quick demo, use: python demo_constitutional_ai.py --quick-test

Example usage:
  python train_constitutional_ai_production.py --config config.json
  python train_constitutional_ai_production.py --phase 1
  python train_constitutional_ai_production.py --phase 2 --phase1-model models/phase1/final_model
        """
    )

    parser.add_argument('--config', help='Path to configuration JSON file')
    parser.add_argument('--phase', choices=['1', '2', 'both'], default='both', help='Which phase to run')
    parser.add_argument('--phase1-model', help='Path to Phase 1 model (for Phase 2 only)')

    args = parser.parse_args()

    # Load configuration
    config = load_config(args.config)

    # Print warning
    logger.info("")
    logger.info("╔" + "=" * 78 + "╗")
    logger.info("║" + " " * 78 + "║")
    logger.info("║" + "    ⚠️  PRODUCTION TRAINING - THIS WILL TAKE HOURS ⚠️".center(78) + "║")
    logger.info("║" + " " * 78 + "║")
    logger.info("╚" + "=" * 78 + "╝")
    logger.info("")
    logger.info("This is NOT the quick demo. This script will:")
    logger.info("  • Actually fine-tune models (Phase 1)")
    logger.info("  • Use full dataset (not limited to 20 examples)")
    logger.info("  • Train for multiple epochs (3-5, not 1)")
    logger.info("  • Take 4-14 hours to complete")
    logger.info("")

    response = input("Are you sure you want to continue? (yes/no): ")
    if response.lower() != 'yes':
        logger.info("Training cancelled.")
        return

    # Setup
    device = setup_device(config.device)
    prompts = load_prompts(config)

    # Save configuration
    config_save_path = Path("training_config_used.json")
    with open(config_save_path, 'w') as f:
        json.dump(asdict(config), f, indent=2)
    logger.info(f"Configuration saved to {config_save_path}")

    # Run training
    phase1_model = args.phase1_model

    if args.phase in ['1', 'both'] and config.phase1_enabled:
        phase1_model = phase_1_supervised_learning(prompts, config, device)

    if args.phase in ['2', 'both'] and config.phase2_enabled:
        if not phase1_model:
            raise ValueError("Phase 2 requires --phase1-model or running Phase 1 first")

        final_model = phase_2_reinforcement_learning(prompts, phase1_model, config, device)

    # Summary
    logger.info("\n" + "=" * 80)
    logger.info("TRAINING COMPLETE!")
    logger.info("=" * 80)
    logger.info("")
    logger.info("Models saved:")
    if args.phase in ['1', 'both']:
        logger.info(f"  Phase 1: {config.phase1_save_dir}/final_model")
    if args.phase in ['2', 'both']:
        logger.info(f"  Phase 2: {config.ppo_save_dir}/final_model")
    logger.info("")
    logger.info("Logs saved to: training.log")
    logger.info("")


if __name__ == '__main__':
    main()
