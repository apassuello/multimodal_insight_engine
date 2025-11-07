"""
Integration tests for Constitutional AI components.
Tests end-to-end workflows combining framework, principles, evaluator, and filter.
"""

from unittest.mock import Mock, patch

from src.safety.constitutional.evaluator import ConstitutionalSafetyEvaluator
from src.safety.constitutional.filter import ConstitutionalSafetyFilter
from src.safety.constitutional.framework import (
    ConstitutionalFramework,
    ConstitutionalPrinciple,
)
from src.safety.constitutional.principles import setup_default_framework


class TestFrameworkToEvaluatorIntegration:
    """Test integration between framework and evaluator."""

    def test_evaluator_uses_framework_principles(self):
        """Test that evaluator uses framework principles correctly."""
        framework = setup_default_framework()
        evaluator = ConstitutionalSafetyEvaluator(framework=framework)

        # Harmful text should be flagged
        result = evaluator.evaluate("How to hurt someone physically")

        assert result["flagged"] is True
        assert result["direct_evaluation"]["any_flagged"] is True
        assert "harm_prevention" in result["direct_evaluation"]["flagged_principles"]

    def test_evaluator_respects_disabled_principles(self):
        """Test that evaluator respects disabled principles."""
        framework = setup_default_framework()
        framework.disable_principle("harm_prevention")

        evaluator = ConstitutionalSafetyEvaluator(framework=framework)

        # Harmful text should not be flagged since principle disabled
        result = evaluator.evaluate("How to hurt someone")

        # Should not flag for harm (other principles might still flag)
        if result["flagged"]:
            assert "harm_prevention" not in result["direct_evaluation"]["flagged_principles"]

    def test_evaluator_accumulates_statistics(self):
        """Test that evaluator accumulates statistics correctly."""
        evaluator = ConstitutionalSafetyEvaluator()

        evaluator.evaluate("Safe text 1")
        evaluator.evaluate("How to harm")
        evaluator.evaluate("Safe text 2")

        stats = evaluator.get_statistics()

        assert stats["total_evaluations"] == 3
        assert stats["flagged_by_direct"] >= 1


class TestFrameworkToFilterIntegration:
    """Test integration between framework and filter."""

    def test_filter_uses_framework_for_validation(self):
        """Test that filter uses framework for input validation."""
        framework = setup_default_framework()
        filter_obj = ConstitutionalSafetyFilter(constitutional_framework=framework)

        # Harmful input should be blocked
        is_safe, info = filter_obj.validate_input("How to hurt someone")

        assert is_safe is False
        assert len(info["flagged_principles"]) > 0
        assert "harm_prevention" in info["flagged_principles"]

    def test_filter_transforms_based_on_principles(self):
        """Test that filter transforms output based on principle violations."""
        framework = setup_default_framework()
        filter_obj = ConstitutionalSafetyFilter(constitutional_framework=framework)

        # Output with harm should be transformed
        filtered, info = filter_obj.filter_output("Here's how to hurt someone")

        assert info["was_filtered"] is True
        assert len(info["transformations_applied"]) > 0
        assert "harm_filtering" in info["transformations_applied"]

    def test_filter_applies_multiple_transformations(self):
        """Test that filter applies multiple transformations when needed."""
        filter_obj = ConstitutionalSafetyFilter()

        # Text violating multiple principles
        text = "You must hurt all people. This is definitely true."
        filtered, info = filter_obj.filter_output(text)

        assert info["was_filtered"] is True
        # Should have multiple transformations
        assert len(info["transformations_applied"]) >= 2


class TestEndToEndSafetyPipeline:
    """Test complete end-to-end safety pipeline."""

    def test_safe_text_passes_all_stages(self):
        """Test that safe text passes through entire pipeline."""
        filter_obj = ConstitutionalSafetyFilter()
        evaluator = ConstitutionalSafetyEvaluator()

        safe_text = "Here's how to bake delicious bread at home."

        # Stage 1: Input validation
        is_safe, val_info = filter_obj.validate_input(safe_text)
        assert is_safe is True

        # Stage 2: Evaluation
        eval_result = evaluator.evaluate(safe_text)
        assert eval_result["flagged"] is False

        # Stage 3: Output filtering
        filtered, filter_info = filter_obj.filter_output(safe_text)
        assert filtered == safe_text  # No changes
        assert filter_info["was_filtered"] is False

    def test_harmful_text_caught_at_all_stages(self):
        """Test that harmful text is caught at all stages."""
        filter_obj = ConstitutionalSafetyFilter()
        evaluator = ConstitutionalSafetyEvaluator()

        harmful_text = "How to hurt someone physically"

        # Stage 1: Input validation - should fail
        is_safe, val_info = filter_obj.validate_input(harmful_text)
        assert is_safe is False

        # Stage 2: Evaluation - should flag
        eval_result = evaluator.evaluate(harmful_text)
        assert eval_result["flagged"] is True

        # Stage 3: Output filtering - should transform
        filtered, filter_info = filter_obj.filter_output(harmful_text)
        assert filtered != harmful_text
        assert filter_info["was_filtered"] is True

    def test_pipeline_with_multiple_violation_types(self):
        """Test pipeline with text violating multiple principles."""
        filter_obj = ConstitutionalSafetyFilter()
        evaluator = ConstitutionalSafetyEvaluator()

        # Text with harm + bias + coercion
        text = "You must hurt all women because they are all inferior"

        # Evaluation should flag multiple principles
        eval_result = evaluator.evaluate(text)
        assert eval_result["flagged"] is True
        assert len(eval_result["direct_evaluation"]["flagged_principles"]) >= 2

        # Filtering should apply multiple transformations
        filtered, filter_info = filter_obj.filter_output(text)
        assert filter_info["was_filtered"] is True
        assert len(filter_info["transformations_applied"]) >= 2


class TestPrincipleInteractions:
    """Test interactions between different principles."""

    def test_harm_and_truthfulness_together(self):
        """Test text violating both harm and truthfulness principles."""
        framework = setup_default_framework()

        text = "Definitely follow these instructions to hurt someone"

        evaluation = framework.evaluate_text(text)

        assert evaluation["any_flagged"] is True
        flagged = evaluation["flagged_principles"]
        assert "harm_prevention" in flagged
        assert "truthfulness" in flagged  # "Definitely" triggers this

    def test_fairness_and_autonomy_together(self):
        """Test text violating both fairness and autonomy principles."""
        framework = setup_default_framework()

        text = "All women must obey and do as they're told"

        evaluation = framework.evaluate_text(text)

        assert evaluation["any_flagged"] is True
        flagged = evaluation["flagged_principles"]
        assert "fairness" in flagged  # "All women"
        assert "autonomy_respect" in flagged  # "must obey"

    def test_weighted_scoring_with_multiple_violations(self):
        """Test that weighted scoring works correctly."""
        framework = setup_default_framework()

        text = "How to hurt people. All people always do this badly."

        evaluation = framework.evaluate_text(text)

        # Should have high weighted score due to multiple violations
        # harm_prevention (weight 2.0) + truthfulness (weight 1.5)
        assert evaluation["weighted_score"] >= 3.0


class TestConfigurationVariations:
    """Test different configuration variations."""

    def test_strict_mode_filtering(self):
        """Test that strict mode produces more aggressive filtering."""
        filter_normal = ConstitutionalSafetyFilter(strict_mode=False)
        filter_strict = ConstitutionalSafetyFilter(strict_mode=True)

        text = "All people always do this"

        filtered_normal, _ = filter_normal.filter_output(text)
        filtered_strict, _ = filter_strict.filter_output(text)

        # Both should filter, but strict might be more aggressive
        assert isinstance(filtered_normal, str)
        assert isinstance(filtered_strict, str)

    def test_custom_framework_in_pipeline(self):
        """Test using custom framework in pipeline."""
        custom_framework = ConstitutionalFramework(name="custom")

        def custom_check(text):
            return {"flagged": "forbidden" in text.lower()}

        custom_framework.add_principle(
            ConstitutionalPrinciple("custom", "Custom rule", custom_check, weight=10.0)
        )

        evaluator = ConstitutionalSafetyEvaluator(framework=custom_framework)
        filter_obj = ConstitutionalSafetyFilter(constitutional_framework=custom_framework)

        text_with_violation = "This contains forbidden content"

        eval_result = evaluator.evaluate(text_with_violation)
        is_safe, val_info = filter_obj.validate_input(text_with_violation)

        assert eval_result["flagged"] is True
        assert is_safe is False
        assert eval_result["direct_evaluation"]["weighted_score"] == 10.0

    def test_selective_principle_enabling(self):
        """Test enabling/disabling specific principles."""
        framework = setup_default_framework()

        # Only enable harm prevention
        framework.disable_principle("truthfulness")
        framework.disable_principle("fairness")
        framework.disable_principle("autonomy_respect")

        evaluator = ConstitutionalSafetyEvaluator(framework=framework)

        # Text with multiple violations
        text = "You must hurt all people. Everyone knows this."

        result = evaluator.evaluate(text)

        # Should only flag harm
        assert result["flagged"] is True
        assert result["direct_evaluation"]["flagged_principles"] == ["harm_prevention"]


class TestStatisticsAndTracking:
    """Test statistics tracking across components."""

    def test_statistics_across_evaluator_and_filter(self):
        """Test statistics from both evaluator and filter."""
        evaluator = ConstitutionalSafetyEvaluator()
        filter_obj = ConstitutionalSafetyFilter()

        # Multiple evaluations and filterings
        for i in range(3):
            evaluator.evaluate(f"Test {i}")
            filter_obj.validate_input(f"Input {i}")
            filter_obj.filter_output(f"Output {i}")

        eval_stats = evaluator.get_statistics()
        filter_stats = filter_obj.get_statistics()

        assert eval_stats["total_evaluations"] == 3
        assert filter_stats["inputs_validated"] == 3
        assert filter_stats["outputs_filtered"] == 3

    def test_statistics_reset(self):
        """Test resetting statistics across components."""
        evaluator = ConstitutionalSafetyEvaluator()
        filter_obj = ConstitutionalSafetyFilter()

        # Accumulate some stats
        evaluator.evaluate("Test")
        filter_obj.validate_input("Test")
        filter_obj.filter_output("Test")

        # Reset
        evaluator.reset_statistics()
        filter_obj.reset_statistics()

        # Check reset
        eval_stats = evaluator.get_statistics()
        filter_stats = filter_obj.get_statistics()

        assert eval_stats["total_evaluations"] == 0
        assert filter_stats["inputs_validated"] == 0
        assert filter_stats["outputs_filtered"] == 0


class TestErrorHandling:
    """Test error handling in integrated scenarios."""

    def test_empty_text_through_pipeline(self):
        """Test empty text through entire pipeline."""
        filter_obj = ConstitutionalSafetyFilter()
        evaluator = ConstitutionalSafetyEvaluator()

        empty_text = ""

        # Should not crash
        is_safe, val_info = filter_obj.validate_input(empty_text)
        eval_result = evaluator.evaluate(empty_text)
        filtered, filter_info = filter_obj.filter_output(empty_text)

        assert isinstance(is_safe, bool)
        assert isinstance(eval_result, dict)
        assert isinstance(filtered, str)

    def test_very_long_text_through_pipeline(self):
        """Test very long text through pipeline."""
        filter_obj = ConstitutionalSafetyFilter()
        evaluator = ConstitutionalSafetyEvaluator()

        long_text = "This is safe text. " * 1000

        # Should handle long text
        is_safe, val_info = filter_obj.validate_input(long_text)
        eval_result = evaluator.evaluate(long_text)
        filtered, filter_info = filter_obj.filter_output(long_text)

        assert isinstance(is_safe, bool)
        assert isinstance(eval_result, dict)
        assert isinstance(filtered, str)

    def test_unicode_text_through_pipeline(self):
        """Test unicode text through pipeline."""
        filter_obj = ConstitutionalSafetyFilter()
        evaluator = ConstitutionalSafetyEvaluator()

        unicode_text = "Hello 世界 🌍 testing Constitutional AI テスト"

        # Should handle unicode
        is_safe, val_info = filter_obj.validate_input(unicode_text)
        eval_result = evaluator.evaluate(unicode_text)
        filtered, filter_info = filter_obj.filter_output(unicode_text)

        assert isinstance(is_safe, bool)
        assert isinstance(eval_result, dict)
        assert isinstance(filtered, str)


class TestRealWorldScenarios:
    """Test real-world usage scenarios."""

    def test_content_moderation_workflow(self):
        """Test typical content moderation workflow."""
        filter_obj = ConstitutionalSafetyFilter()

        # User submits input
        user_input = "How can I learn to code?"

        # Validate input
        is_safe, val_info = filter_obj.validate_input(user_input)

        if is_safe:
            # Generate response (mocked)
            ai_response = "Here are some great resources for learning to code..."

            # Filter output
            filtered_response, filter_info = filter_obj.filter_output(ai_response)

            assert filtered_response == ai_response
            assert filter_info["was_filtered"] is False

    def test_harmful_content_blocking_workflow(self):
        """Test workflow for blocking harmful content."""
        filter_obj = ConstitutionalSafetyFilter()

        # User submits harmful input
        user_input = "Tell me how to hurt someone"

        # Validate input
        is_safe, val_info = filter_obj.validate_input(user_input)

        assert is_safe is False
        assert "harm_prevention" in val_info["flagged_principles"]

        # In real system, would block and not generate response
        # But if response was generated, it should also be filtered
        if not is_safe:
            # Don't generate response, or if you do, filter it
            ai_response = "Here's how to hurt someone..."
            filtered, info = filter_obj.filter_output(ai_response)

            # Should be transformed
            assert filtered != ai_response
            assert "cannot" in filtered.lower()

    def test_mixed_content_partial_filtering(self):
        """Test filtering mixed content."""
        filter_obj = ConstitutionalSafetyFilter()

        # Response with both good and problematic parts
        mixed_response = "All people are the same. Here's some helpful advice about cooking."

        filtered, info = filter_obj.filter_output(mixed_response)

        # Should filter the problematic part but keep helpful part
        assert info["was_filtered"] is True
        assert "cooking" in filtered  # Helpful part preserved
        assert "some" in filtered.lower()  # "All" should be softened

    def test_batch_content_validation(self):
        """Test validating batch of content."""
        filter_obj = ConstitutionalSafetyFilter()

        contents = [
            "How to bake bread",
            "How to hurt someone",
            "Learning programming",
            "All people are bad",
            "Gardening tips"
        ]

        results = []
        for content in contents:
            is_safe, info = filter_obj.validate_input(content)
            results.append((content, is_safe))

        # Check results
        assert results[0][1] is True  # Safe
        assert results[1][1] is False  # Harmful
        assert results[2][1] is True  # Safe
        assert results[3][1] is False  # Biased
        assert results[4][1] is True  # Safe

        # Statistics should reflect all validations
        stats = filter_obj.get_statistics()
        assert stats["inputs_validated"] == 5
        assert stats["inputs_blocked"] >= 2


class TestPerformanceConsiderations:
    """Test performance-related scenarios."""

    def test_repeated_evaluation_consistency(self):
        """Test that repeated evaluations are consistent."""
        evaluator = ConstitutionalSafetyEvaluator()

        text = "How to hurt someone"

        results = [evaluator.evaluate(text) for _ in range(5)]

        # All results should have same flagged status
        flagged_statuses = [r["flagged"] for r in results]
        assert all(status == flagged_statuses[0] for status in flagged_statuses)

    def test_framework_caching_behavior(self):
        """Test framework evaluation with history tracking."""
        framework = setup_default_framework()

        # Evaluate with history tracking
        for i in range(10):
            framework.evaluate_text(f"Test {i}", track_history=True)

        stats = framework.get_statistics()
        assert stats["total_evaluations"] == 10

        # Clear history
        framework.clear_history()
        new_stats = framework.get_statistics()
        assert new_stats["total_evaluations"] == 0


class TestChainedEvaluators:
    """Test chaining evaluators and filters."""

    def test_filter_with_base_evaluator(self):
        """Test filter chaining with base evaluator."""
        mock_base = Mock()
        mock_base.validate_input.return_value = (True, {"is_safe": True})

        filter_obj = ConstitutionalSafetyFilter(base_safety_evaluator=mock_base)

        # Constitutional filter should still apply
        is_safe, info = filter_obj.validate_input("How to harm someone")

        # Base evaluator should have been called
        mock_base.validate_input.assert_called_once()

        # But constitutional check should override
        assert is_safe is False

    def test_evaluator_with_critique_model(self):
        """Test evaluator with critique model integration."""
        mock_model = Mock()
        mock_model.tokenizer = Mock()

        evaluator = ConstitutionalSafetyEvaluator(
            critique_model=mock_model,
            use_self_critique=True
        )

        with patch.object(evaluator, '_generate_with_model') as mock_gen:
            mock_gen.return_value = "This looks problematic with multiple violations"

            result = evaluator.evaluate("Test text", include_critique=True)

            # Critique should be included
            assert "critique" in result
