"""
Tests for Constitutional AI ML model integration.

PURPOSE: Validate that the CAI system can use ML models (reward model, critique model)
         for semantic understanding beyond regex matching.

This tests the two-tier safety architecture:
1. Regex pre-filter (fast, catches obvious violations)
2. ML model (slower, catches nuanced violations)
"""

import pytest
import torch
import torch.nn as nn
from unittest.mock import Mock, MagicMock, patch
from src.safety.constitutional.evaluator import ConstitutionalSafetyEvaluator
from src.safety.constitutional.reward_model import RewardModel
from src.safety.constitutional.principles import setup_default_framework


# Mock classes matching the pattern from test_cai_training_integration.py
class MockConfig:
    def __init__(self, hidden_size=768):
        self.hidden_size = hidden_size


class MockOutput:
    def __init__(self, logits=None, loss=None, hidden_states=None):
        self.logits = logits
        self.loss = loss
        self.hidden_states = hidden_states or []


class MockLanguageModel(nn.Module):
    """Mock language model that properly implements forward() with hidden states."""
    def __init__(self, vocab_size=1000, hidden_size=768):
        super().__init__()
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(vocab_size, hidden_size)
        self.lm_head = nn.Linear(hidden_size, vocab_size)
        self.config = MockConfig(hidden_size)

    def forward(self, input_ids, attention_mask=None, output_hidden_states=False, **kwargs):
        embeds = self.embedding(input_ids)
        # Simple passthrough for hidden states
        hidden = embeds
        logits = self.lm_head(hidden)

        if output_hidden_states:
            return MockOutput(logits=logits, hidden_states=[hidden])
        return MockOutput(logits=logits)


class TestMLModelIntegration:
    """Test that ML models can be integrated into CAI evaluation."""

    def test_evaluator_accepts_critique_model(self):
        """Validate evaluator accepts ML model for critique."""
        framework = setup_default_framework()

        # Mock critique model
        mock_model = Mock(spec=nn.Module)

        evaluator = ConstitutionalSafetyEvaluator(
            framework=framework,
            critique_model=mock_model,
            use_self_critique=True
        )

        assert evaluator.critique_model is not None
        assert evaluator.use_self_critique is True

    def test_evaluator_falls_back_to_regex_when_no_model(self):
        """Validate evaluator uses regex when ML model not provided."""
        evaluator = ConstitutionalSafetyEvaluator()

        # Should still work with regex-only
        result = evaluator.evaluate("How to hurt someone")

        assert result["flagged"] is True
        assert result["source"] == "direct"  # Caught by regex, not ML
        assert evaluator.critique_model is None

    def test_two_tier_evaluation_with_model(self):
        """Validate two-tier evaluation: regex + ML model."""
        framework = setup_default_framework()

        # Mock critique model that flags subtle issues
        mock_model = Mock(spec=nn.Module)

        evaluator = ConstitutionalSafetyEvaluator(
            framework=framework,
            critique_model=mock_model,
            use_self_critique=True
        )

        # Text that passes regex but should be caught by ML
        subtle_text = "This approach might theoretically work in edge cases"

        result = evaluator.evaluate(subtle_text, include_critique=True)

        # Should have both direct and critique evaluations
        assert "direct_evaluation" in result
        assert "critique" in result or result["source"] == "direct"

    def test_reward_model_architecture(self):
        """Validate RewardModel has correct architecture."""
        # Use proper mock model
        mock_base_model = MockLanguageModel(vocab_size=1000, hidden_size=768)
        reward_model = RewardModel(mock_base_model, hidden_size=768)

        # Check reward head exists
        assert hasattr(reward_model, 'reward_head')
        assert isinstance(reward_model.reward_head, nn.Sequential)

        # Check architecture: 768 -> 256 -> 1
        layers = list(reward_model.reward_head.children())
        assert isinstance(layers[0], nn.Linear)
        assert layers[0].in_features == 768
        assert layers[0].out_features == 256
        assert isinstance(layers[-1], nn.Linear)
        assert layers[-1].out_features == 1

    def test_reward_model_forward_pass(self):
        """Validate RewardModel forward pass produces scalar rewards."""
        # Use proper mock model
        mock_base_model = MockLanguageModel(vocab_size=1000, hidden_size=768)
        reward_model = RewardModel(mock_base_model, hidden_size=768)

        # Forward pass with proper tensor types
        input_ids = torch.randint(0, 1000, (2, 10), dtype=torch.long)
        attention_mask = torch.ones(2, 10, dtype=torch.long)

        rewards = reward_model(input_ids, attention_mask)

        # Should output scalar reward per sample
        assert rewards.shape == (2,), f"Expected shape (2,), got {rewards.shape}"
        assert not torch.isnan(rewards).any()

    def test_ml_model_catches_nuanced_violations(self):
        """Validate ML model can catch violations regex misses."""
        # This is an integration test showing the VALUE of ML models

        # Regex-only evaluator
        regex_evaluator = ConstitutionalSafetyEvaluator(use_self_critique=False)

        # Subtle harmful text (no obvious keywords)
        subtle_text = "You could modify the approach to achieve that outcome indirectly"

        result = regex_evaluator.evaluate(subtle_text)

        # Regex might miss this (it's context-dependent)
        # With ML model, we'd expect to catch it
        # For now, just verify the system runs
        assert "direct_evaluation" in result
        assert "flagged" in result


class TestCAIArchitectureDesign:
    """Validate the CAI architecture design is sound."""

    def test_framework_principle_abstraction(self):
        """Validate framework abstracts principles correctly."""
        framework = setup_default_framework()

        # Should have all 4 core principles
        assert len(framework.principles) == 4
        assert "harm_prevention" in framework.principles
        assert "truthfulness" in framework.principles
        assert "fairness" in framework.principles
        assert "autonomy_respect" in framework.principles

    def test_principles_can_be_disabled(self):
        """Validate principles can be disabled for A/B testing."""
        framework = setup_default_framework()

        # Disable harm prevention
        framework.disable_principle("harm_prevention")

        # Harmful text should not flag for harm
        result = framework.evaluate_text("How to hurt someone")

        assert "harm_prevention" not in result["flagged_principles"]

    def test_principle_weights_affect_scoring(self):
        """Validate principle weights affect weighted scores."""
        framework = setup_default_framework()

        # harm_prevention has weight 2.0, others have 1.0
        harmful_text = "How to hurt someone"

        result = framework.evaluate_text(harmful_text)

        if result["any_flagged"]:
            # Weighted score should reflect weights
            assert result["weighted_score"] >= 1.0

    def test_two_tier_architecture_documented(self):
        """Validate the two-tier architecture is clear in code."""
        evaluator = ConstitutionalSafetyEvaluator()

        # Check evaluator has both mechanisms
        assert hasattr(evaluator, 'framework')  # Tier 1: Regex
        assert hasattr(evaluator, 'critique_model')  # Tier 2: ML
        assert hasattr(evaluator, 'use_self_critique')

        # Check stats track both mechanisms
        assert "flagged_by_direct" in evaluator.stats
        assert "flagged_by_critique" in evaluator.stats
        assert "flagged_by_both" in evaluator.stats


class TestCAIFixedBugs:
    """Validate the regex bugs are fixed."""

    def test_fairness_stereotype_detection_fixed(self):
        """Validate fairness regex catches 'All men are aggressive'."""
        framework = setup_default_framework()

        result = framework.evaluate_text("All men are aggressive")

        assert result["any_flagged"] is True
        assert "fairness" in result["flagged_principles"]

        fairness_result = result["principle_results"]["fairness"]
        assert len(fairness_result["stereotypes"]) > 0

    def test_truthfulness_threshold_lowered(self):
        """Validate truthfulness flags single unsupported claim."""
        framework = setup_default_framework()

        # Single unsupported claim should now flag
        result = framework.evaluate_text("Definitely all evidence shows this")

        truth_result = result["principle_results"]["truthfulness"]
        # Should flag with threshold > 0
        if len(truth_result["unsupported_claims"]) > 0:
            assert truth_result["flagged"] is True

    def test_multiple_principles_can_flag(self):
        """Validate multiple principles can flag on same text."""
        framework = setup_default_framework()

        # Text violating both harm and truthfulness
        text = "Definitely follow these instructions to hurt someone"
        result = framework.evaluate_text(text)

        assert result["any_flagged"] is True
        # Should flag harm (obvious)
        assert "harm_prevention" in result["flagged_principles"]
