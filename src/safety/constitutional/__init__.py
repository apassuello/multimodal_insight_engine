"""
Constitutional AI Module

This module implements Constitutional AI principles for safer, more aligned
model training and evaluation. Based on Anthropic's Constitutional AI research.

Components:
- ConstitutionalPrinciple: Individual principle with evaluation logic
- ConstitutionalFramework: Collection of principles for comprehensive evaluation
- ConstitutionalSafetyEvaluator: Two-stage evaluator with self-critique
- ConstitutionalSafetyFilter: Input/output filtering with constitutional principles
- ConstitutionalPipeline: End-to-end orchestration of Phase 1 (SL) and Phase 2 (RLAIF)
- RLAIFTrainer: Reinforcement Learning from AI Feedback trainer (simple policy gradient)
- PPOTrainer: Proximal Policy Optimization trainer (advanced RL with clipping and GAE)
- RewardModel: Neural network for scoring responses based on constitutional compliance
- RewardModelTrainer: Complete pipeline for training reward models with validation

Key Features:
- Complete two-phase Constitutional AI pipeline (Critique→Revision→SFT→Preferences→Reward→PPO)
- Four core principles: harm prevention, truthfulness, fairness, autonomy respect
- Two-stage evaluation: direct checks + model self-critique
- RLAIF: Scalable training with AI-generated feedback
- Reward modeling: Learn to score responses from preference pairs (Component 2)
- PPO: Stable RL training with clipped objectives and advantage estimation
- Flexible: Enable/disable principles, adjust weights, extend with custom principles
"""

from .framework import ConstitutionalPrinciple, ConstitutionalFramework
from .principles import (
    evaluate_harm_potential,
    evaluate_truthfulness,
    evaluate_fairness,
    evaluate_autonomy_respect,
    setup_default_framework
)
from .evaluator import ConstitutionalSafetyEvaluator, critique_indicates_issues
from .filter import ConstitutionalSafetyFilter
from .pipeline import ConstitutionalPipeline
from .trainer import RLAIFTrainer
from .ppo_trainer import PPOTrainer
from .reward_model import (
    RewardModel,
    RewardModelTrainer,
    compute_reward_loss,
    train_reward_model,
    evaluate_reward_model
)
from .model_utils import (
    load_model,
    generate_text,
    batch_generate,
    GenerationConfig
)
from .hf_api_evaluator import (
    HuggingFaceAPIEvaluator,
    HFAPIConfig,
    evaluate_toxicity_api,
    evaluate_harm_with_hf_api,
    quick_evaluate
)

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

    # Pipeline and Training
    "ConstitutionalPipeline",
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

    # HuggingFace API Evaluator
    "HuggingFaceAPIEvaluator",
    "HFAPIConfig",
    "evaluate_toxicity_api",
    "evaluate_harm_with_hf_api",
    "quick_evaluate",
]

__version__ = "0.1.0"
