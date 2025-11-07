"""
Constitutional AI Module

This module implements Constitutional AI principles for safer, more aligned
model training and evaluation. Based on Anthropic's Constitutional AI research.

Components:
- ConstitutionalPrinciple: Individual principle with evaluation logic
- ConstitutionalFramework: Collection of principles for comprehensive evaluation
- ConstitutionalSafetyEvaluator: Two-stage evaluator with self-critique
- ConstitutionalSafetyFilter: Input/output filtering with constitutional principles
- RLAIFTrainer: Reinforcement Learning from AI Feedback trainer (simple policy gradient)
- PPOTrainer: Proximal Policy Optimization trainer (advanced RL with clipping and GAE)
- RewardModel: Neural network for scoring responses based on constitutional compliance
- RewardModelTrainer: Complete pipeline for training reward models with validation

Key Features:
- Four core principles: harm prevention, truthfulness, fairness, autonomy respect
- Two-stage evaluation: direct checks + model self-critique
- RLAIF: Scalable training with AI-generated feedback
- Reward modeling: Learn to score responses from preference pairs (Component 2)
- PPO: Stable RL training with clipped objectives and advantage estimation
- Flexible: Enable/disable principles, adjust weights, extend with custom principles
"""

from .evaluator import ConstitutionalSafetyEvaluator, critique_indicates_issues
from .filter import ConstitutionalSafetyFilter
from .framework import ConstitutionalFramework, ConstitutionalPrinciple
from .model_utils import GenerationConfig, batch_generate, generate_text, load_model
from .ppo_trainer import PPOTrainer
from .principles import (
    evaluate_autonomy_respect,
    evaluate_fairness,
    evaluate_harm_potential,
    evaluate_truthfulness,
    setup_default_framework,
)
from .reward_model import (
    RewardModel,
    RewardModelTrainer,
    compute_reward_loss,
    evaluate_reward_model,
    train_reward_model,
)
from .trainer import RLAIFTrainer

__all__ = [
    # Core framework
    "ConstitutionalPrinciple",
    "ConstitutionalFramework",

    # Principle evaluators
    "evaluate_harm_potential",
    "evaluate_truthfulness",
    "evaluate_fairness",
    "evaluate_autonomy_respect",
    "setup_default_framework",

    # Evaluator and filter
    "ConstitutionalSafetyEvaluator",
    "ConstitutionalSafetyFilter",
    "critique_indicates_issues",

    # Training
    "RLAIFTrainer",
    "PPOTrainer",

    # Reward Model (Component 2)
    "RewardModel",
    "RewardModelTrainer",
    "compute_reward_loss",
    "train_reward_model",
    "evaluate_reward_model",

    # Model utilities
    "load_model",
    "generate_text",
    "batch_generate",
    "GenerationConfig",
]

__version__ = "0.1.0"
