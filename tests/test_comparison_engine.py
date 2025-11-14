"""
Tests for ComparisonEngine module.

Tests comparison functionality, alignment calculations, and error handling.
"""

import pytest
import torch
from unittest.mock import Mock, MagicMock, patch
from typing import List, Dict, Any

from demo.managers.comparison_engine import (
    ComparisonEngine,
    ComparisonResult,
    PrincipleComparison,
    ExampleComparison
)
from src.safety.constitutional.framework import ConstitutionalFramework
from src.safety.constitutional.model_utils import GenerationConfig


@pytest.fixture
def mock_framework():
    """Create a mock Constitutional Framework."""
    framework = Mock(spec=ConstitutionalFramework)
    framework.principles = {
        'harm_prevention': Mock(weight=1.0),
        'fairness': Mock(weight=1.0),
        'truthfulness': Mock(weight=1.0),
        'autonomy': Mock(weight=1.0)
    }
    return framework


@pytest.fixture
def mock_models():
    """Create mock models and tokenizers."""
    base_model = Mock()
    base_tokenizer = Mock()
    trained_model = Mock()
    trained_tokenizer = Mock()
    return base_model, base_tokenizer, trained_model, trained_tokenizer


@pytest.fixture
def comparison_engine(mock_framework):
    """Create a ComparisonEngine instance with mock framework."""
    return ComparisonEngine(mock_framework)


class TestComparisonResult:
    """Tests for ComparisonResult dataclass."""

    def test_comparison_result_initialization(self):
        """Test that ComparisonResult initializes with correct defaults."""
        result = ComparisonResult(
            test_suite_name="Test Suite",
            num_prompts=10
        )

        assert result.test_suite_name == "Test Suite"
        assert result.num_prompts == 10
        assert result.principle_results == {}
        assert result.overall_alignment_before == 0.0
        assert result.overall_alignment_after == 0.0
        assert result.alignment_improvement == 0.0
        assert result.examples == []
        assert result.errors == []
        assert result.skipped_prompts == 0


class TestPrincipleComparison:
    """Tests for PrincipleComparison dataclass."""

    def test_principle_comparison_initialization(self):
        """Test PrincipleComparison initialization."""
        comparison = PrincipleComparison(
            principle_name="harm_prevention",
            violations_before=10,
            violations_after=2,
            improvement_pct=80.0
        )

        assert comparison.principle_name == "harm_prevention"
        assert comparison.violations_before == 10
        assert comparison.violations_after == 2
        assert comparison.improvement_pct == 80.0


class TestExampleComparison:
    """Tests for ExampleComparison dataclass."""

    def test_example_comparison_initialization(self):
        """Test ExampleComparison initialization."""
        example = ExampleComparison(
            prompt="Test prompt",
            base_output="Base output",
            trained_output="Trained output",
            base_evaluation={'weighted_score': 0.5, 'flagged_principles': []},
            trained_evaluation={'weighted_score': 0.1, 'flagged_principles': []},
            improved=True
        )

        assert example.prompt == "Test prompt"
        assert example.base_output == "Base output"
        assert example.trained_output == "Trained output"
        assert example.improved is True


class TestComparisonEngine:
    """Tests for ComparisonEngine class."""

    def test_initialization(self, mock_framework):
        """Test that ComparisonEngine initializes correctly."""
        engine = ComparisonEngine(mock_framework)
        assert engine.framework == mock_framework

    @patch('demo.managers.comparison_engine.generate_text')
    def test_compare_models_basic(self, mock_generate, comparison_engine, mock_models):
        """Test basic model comparison with successful generations."""
        base_model, base_tokenizer, trained_model, trained_tokenizer = mock_models

        # Mock text generation
        mock_generate.side_effect = ["Base output 1", "Trained output 1"]

        # Mock framework evaluation
        comparison_engine.framework.evaluate_text.side_effect = [
            {'weighted_score': 0.5, 'flagged_principles': ['harm_prevention'], 'any_flagged': True},
            {'weighted_score': 0.1, 'flagged_principles': [], 'any_flagged': False}
        ]

        test_suite = ["Test prompt 1"]
        device = torch.device("cpu")
        gen_config = GenerationConfig(max_length=100, temperature=0.7)

        result = comparison_engine.compare_models(
            base_model, base_tokenizer,
            trained_model, trained_tokenizer,
            test_suite, device, gen_config,
            test_suite_name="Test Suite"
        )

        # Verify result structure
        assert result.test_suite_name == "Test Suite"
        assert result.num_prompts == 1
        assert result.skipped_prompts == 0
        assert len(result.examples) == 1
        assert len(result.errors) == 0

        # Verify example
        example = result.examples[0]
        assert example.prompt == "Test prompt 1"
        assert example.base_output == "Base output 1"
        assert example.trained_output == "Trained output 1"
        assert example.improved is True  # Lower score = better

    @patch('demo.managers.comparison_engine.generate_text')
    def test_compare_models_with_errors(self, mock_generate, comparison_engine, mock_models):
        """Test that compare_models handles generation errors gracefully."""
        base_model, base_tokenizer, trained_model, trained_tokenizer = mock_models

        # First prompt succeeds, second fails
        mock_generate.side_effect = [
            "Base output 1",
            "Trained output 1",
            RuntimeError("Generation failed"),
            "Trained output 2"
        ]

        comparison_engine.framework.evaluate_text.side_effect = [
            {'weighted_score': 0.5, 'flagged_principles': [], 'any_flagged': False},
            {'weighted_score': 0.1, 'flagged_principles': [], 'any_flagged': False}
        ]

        test_suite = ["Prompt 1", "Prompt 2"]
        device = torch.device("cpu")
        gen_config = GenerationConfig()

        result = comparison_engine.compare_models(
            base_model, base_tokenizer,
            trained_model, trained_tokenizer,
            test_suite, device, gen_config
        )

        # Should have processed 1 example, skipped 1
        assert result.num_prompts == 2
        assert result.skipped_prompts == 1
        assert len(result.examples) == 1
        assert len(result.errors) == 1
        assert "Generation failed" in result.errors[0]

    def test_alignment_score_calculation_perfect(self, comparison_engine, mock_models):
        """Test alignment score calculation with perfect alignment."""
        base_model, base_tokenizer, trained_model, trained_tokenizer = mock_models

        with patch('demo.managers.comparison_engine.generate_text') as mock_generate:
            mock_generate.side_effect = ["Base output", "Trained output"]

            # Both have no violations (weighted_score = 0.0)
            comparison_engine.framework.evaluate_text.side_effect = [
                {'weighted_score': 0.0, 'flagged_principles': [], 'any_flagged': False},
                {'weighted_score': 0.0, 'flagged_principles': [], 'any_flagged': False}
            ]

            test_suite = ["Test prompt"]
            device = torch.device("cpu")
            gen_config = GenerationConfig()

            result = comparison_engine.compare_models(
                base_model, base_tokenizer,
                trained_model, trained_tokenizer,
                test_suite, device, gen_config
            )

            # Perfect alignment = 1.0
            assert result.overall_alignment_before == 1.0
            assert result.overall_alignment_after == 1.0
            assert result.alignment_improvement == 0.0

    def test_alignment_score_calculation_improvement(self, comparison_engine, mock_models):
        """Test alignment score calculation with improvement."""
        base_model, base_tokenizer, trained_model, trained_tokenizer = mock_models

        with patch('demo.managers.comparison_engine.generate_text') as mock_generate:
            # Generate 2 prompts
            mock_generate.side_effect = [
                "Base 1", "Trained 1",
                "Base 2", "Trained 2"
            ]

            # Base has violations, trained has fewer
            # Total max possible = 2 prompts * 4 principles * 1.0 weight = 8.0
            # Base: 2.0 weighted violations -> ratio 0.25 -> alignment 0.75
            # Trained: 0.5 weighted violations -> ratio 0.0625 -> alignment 0.9375
            comparison_engine.framework.evaluate_text.side_effect = [
                {'weighted_score': 1.0, 'flagged_principles': ['harm_prevention'], 'any_flagged': True},
                {'weighted_score': 0.25, 'flagged_principles': [], 'any_flagged': False},
                {'weighted_score': 1.0, 'flagged_principles': ['fairness'], 'any_flagged': True},
                {'weighted_score': 0.25, 'flagged_principles': [], 'any_flagged': False}
            ]

            test_suite = ["Prompt 1", "Prompt 2"]
            device = torch.device("cpu")
            gen_config = GenerationConfig()

            result = comparison_engine.compare_models(
                base_model, base_tokenizer,
                trained_model, trained_tokenizer,
                test_suite, device, gen_config
            )

            # Check that alignment scores are calculated
            assert 0.0 <= result.overall_alignment_before <= 1.0
            assert 0.0 <= result.overall_alignment_after <= 1.0
            assert result.overall_alignment_after > result.overall_alignment_before

    def test_principle_comparison_calculation(self, comparison_engine, mock_models):
        """Test per-principle violation tracking and improvement calculation."""
        base_model, base_tokenizer, trained_model, trained_tokenizer = mock_models

        with patch('demo.managers.comparison_engine.generate_text') as mock_generate:
            mock_generate.side_effect = [
                "Base 1", "Trained 1",
                "Base 2", "Trained 2",
                "Base 3", "Trained 3"
            ]

            # Pattern: harm_prevention violations decrease from 3 -> 1
            comparison_engine.framework.evaluate_text.side_effect = [
                # Prompt 1: Base has harm violation, trained doesn't
                {'weighted_score': 1.0, 'flagged_principles': ['harm_prevention'], 'any_flagged': True},
                {'weighted_score': 0.0, 'flagged_principles': [], 'any_flagged': False},
                # Prompt 2: Both have harm violation
                {'weighted_score': 1.0, 'flagged_principles': ['harm_prevention'], 'any_flagged': True},
                {'weighted_score': 1.0, 'flagged_principles': ['harm_prevention'], 'any_flagged': True},
                # Prompt 3: Base has harm violation, trained doesn't
                {'weighted_score': 1.0, 'flagged_principles': ['harm_prevention'], 'any_flagged': True},
                {'weighted_score': 0.0, 'flagged_principles': [], 'any_flagged': False}
            ]

            test_suite = ["Prompt 1", "Prompt 2", "Prompt 3"]
            device = torch.device("cpu")
            gen_config = GenerationConfig()

            result = comparison_engine.compare_models(
                base_model, base_tokenizer,
                trained_model, trained_tokenizer,
                test_suite, device, gen_config
            )

            # Verify principle comparison
            assert 'harm_prevention' in result.principle_results
            harm_comparison = result.principle_results['harm_prevention']
            assert harm_comparison.violations_before == 3
            assert harm_comparison.violations_after == 1
            # Improvement: (3-1)/3 * 100 = 66.67%
            assert harm_comparison.improvement_pct == pytest.approx(66.67, rel=0.1)

    def test_progress_callback(self, comparison_engine, mock_models):
        """Test that progress callback is invoked correctly."""
        base_model, base_tokenizer, trained_model, trained_tokenizer = mock_models

        callback_calls = []

        def progress_callback(current, total, message):
            callback_calls.append((current, total, message))

        with patch('demo.managers.comparison_engine.generate_text') as mock_generate:
            mock_generate.side_effect = ["Base 1", "Trained 1", "Base 2", "Trained 2"]
            comparison_engine.framework.evaluate_text.side_effect = [
                {'weighted_score': 0.0, 'flagged_principles': [], 'any_flagged': False}
            ] * 4

            test_suite = ["Prompt 1", "Prompt 2"]
            device = torch.device("cpu")
            gen_config = GenerationConfig()

            result = comparison_engine.compare_models(
                base_model, base_tokenizer,
                trained_model, trained_tokenizer,
                test_suite, device, gen_config,
                progress_callback=progress_callback
            )

            # Should have 2 callback invocations (one per prompt)
            assert len(callback_calls) == 2
            assert callback_calls[0] == (1, 2, "Processing prompt 1/2")
            assert callback_calls[1] == (2, 2, "Processing prompt 2/2")

    def test_empty_test_suite(self, comparison_engine, mock_models):
        """Test behavior with empty test suite."""
        base_model, base_tokenizer, trained_model, trained_tokenizer = mock_models

        test_suite = []
        device = torch.device("cpu")
        gen_config = GenerationConfig()

        result = comparison_engine.compare_models(
            base_model, base_tokenizer,
            trained_model, trained_tokenizer,
            test_suite, device, gen_config
        )

        assert result.num_prompts == 0
        assert result.skipped_prompts == 0
        assert len(result.examples) == 0
        assert len(result.errors) == 0
        # With no prompts, alignment scores should be 1.0 (perfect by default)
        assert result.overall_alignment_before == 1.0
        assert result.overall_alignment_after == 1.0

    def test_format_comparison_summary(self, comparison_engine):
        """Test summary formatting."""
        result = ComparisonResult(
            test_suite_name="Test Suite",
            num_prompts=10
        )
        result.overall_alignment_before = 0.75
        result.overall_alignment_after = 0.90
        result.alignment_improvement = 20.0
        result.principle_results = {
            'harm_prevention': PrincipleComparison(
                principle_name='harm_prevention',
                violations_before=5,
                violations_after=1,
                improvement_pct=80.0
            )
        }

        summary = comparison_engine.format_comparison_summary(result)

        assert "Test Suite" in summary
        assert "10" in summary  # num_prompts
        assert "0.750" in summary  # alignment before
        assert "0.900" in summary  # alignment after
        assert "harm_prevention" in summary
        assert "80.0%" in summary

    def test_regression_detection(self, comparison_engine, mock_models):
        """Test that regressions are properly detected (trained worse than base)."""
        base_model, base_tokenizer, trained_model, trained_tokenizer = mock_models

        with patch('demo.managers.comparison_engine.generate_text') as mock_generate:
            mock_generate.side_effect = ["Base output", "Trained output"]

            # Trained model is WORSE (higher weighted_score)
            comparison_engine.framework.evaluate_text.side_effect = [
                {'weighted_score': 0.1, 'flagged_principles': [], 'any_flagged': False},
                {'weighted_score': 0.5, 'flagged_principles': ['harm_prevention'], 'any_flagged': True}
            ]

            test_suite = ["Test prompt"]
            device = torch.device("cpu")
            gen_config = GenerationConfig()

            result = comparison_engine.compare_models(
                base_model, base_tokenizer,
                trained_model, trained_tokenizer,
                test_suite, device, gen_config
            )

            # Should detect regression
            example = result.examples[0]
            assert example.improved is False  # Trained is worse
            assert result.overall_alignment_after < result.overall_alignment_before
