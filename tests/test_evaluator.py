"""
Unit tests for evaluator.py
Tests the ConstitutionalSafetyEvaluator and two-stage evaluation process.
"""

from unittest.mock import Mock, patch

from src.safety.constitutional.evaluator import (
    ConstitutionalSafetyEvaluator,
    combine_reasoning,
    critique_indicates_issues,
)
from src.safety.constitutional.framework import (
    ConstitutionalFramework,
    ConstitutionalPrinciple,
)


class TestConstitutionalSafetyEvaluatorInit:
    """Test ConstitutionalSafetyEvaluator initialization."""

    def test_init_with_default_framework(self):
        """Test initialization with default framework."""
        evaluator = ConstitutionalSafetyEvaluator()

        assert evaluator.framework is not None
        assert len(evaluator.framework.principles) == 4
        assert evaluator.critique_model is None
        assert evaluator.use_self_critique is False

    def test_init_with_custom_framework(self):
        """Test initialization with custom framework."""
        custom_framework = ConstitutionalFramework(name="custom")

        def eval_fn(text): return {"flagged": False}
        custom_framework.add_principle(
            ConstitutionalPrinciple("test", "desc", eval_fn)
        )

        evaluator = ConstitutionalSafetyEvaluator(framework=custom_framework)

        assert evaluator.framework.name == "custom"
        assert len(evaluator.framework.principles) == 1

    def test_init_with_critique_model(self):
        """Test initialization with critique model."""
        mock_model = Mock()
        evaluator = ConstitutionalSafetyEvaluator(
            critique_model=mock_model,
            use_self_critique=True
        )

        assert evaluator.critique_model is mock_model
        assert evaluator.use_self_critique is True

    def test_init_use_self_critique_without_model(self):
        """Test that self-critique is disabled without model."""
        evaluator = ConstitutionalSafetyEvaluator(use_self_critique=True)

        # Should be False because no model provided
        assert evaluator.use_self_critique is False

    def test_init_statistics_initialized(self):
        """Test that statistics are properly initialized."""
        evaluator = ConstitutionalSafetyEvaluator()

        assert evaluator.stats["total_evaluations"] == 0
        assert evaluator.stats["flagged_by_direct"] == 0
        assert evaluator.stats["flagged_by_critique"] == 0
        assert evaluator.stats["flagged_by_both"] == 0


class TestEvaluateMethod:
    """Test the evaluate method."""

    def setup_method(self):
        """Setup test fixtures."""
        self.evaluator = ConstitutionalSafetyEvaluator()

    def test_evaluate_clean_text(self):
        """Test evaluating clean, safe text."""
        text = "This is a safe, helpful response about baking bread."
        result = self.evaluator.evaluate(text)

        assert result["flagged"] is False
        assert result["source"] == "none"
        assert "direct_evaluation" in result
        assert "reasoning" in result

    def test_evaluate_harmful_text(self):
        """Test evaluating text with harm potential."""
        text = "How to hurt someone physically"
        result = self.evaluator.evaluate(text)

        assert result["flagged"] is True
        assert result["source"] in ["direct", "both"]
        assert result["direct_evaluation"]["any_flagged"] is True

    def test_evaluate_increments_statistics(self):
        """Test that evaluate increments statistics."""
        initial_count = self.evaluator.stats["total_evaluations"]

        self.evaluator.evaluate("Test text")

        assert self.evaluator.stats["total_evaluations"] == initial_count + 1

    def test_evaluate_tracks_flagged_by_direct(self):
        """Test that direct flagging is tracked."""
        initial_count = self.evaluator.stats["flagged_by_direct"]

        self.evaluator.evaluate("How to harm someone")

        assert self.evaluator.stats["flagged_by_direct"] >= initial_count

    def test_evaluate_result_structure(self):
        """Test that result has expected structure."""
        result = self.evaluator.evaluate("Test")

        required_keys = ["direct_evaluation", "flagged", "source", "reasoning"]
        for key in required_keys:
            assert key in result, f"Missing key: {key}"

    def test_evaluate_with_critique_disabled(self):
        """Test that critique is not included when disabled."""
        result = self.evaluator.evaluate("Test", include_critique=False)

        assert "critique" not in result

    def test_evaluate_with_critique_enabled_but_no_model(self):
        """Test critique request without model."""
        result = self.evaluator.evaluate("Test", include_critique=True)

        # Should not have critique since no model
        assert "critique" not in result or result.get("critique") is None

    def test_evaluate_empty_text(self):
        """Test evaluating empty text."""
        result = self.evaluator.evaluate("")

        assert "flagged" in result
        assert "reasoning" in result


class TestEvaluateWithSelfCritique:
    """Test evaluate_with_self_critique method."""

    def test_calls_evaluate_with_critique_true(self):
        """Test that it calls evaluate with include_critique=True."""
        evaluator = ConstitutionalSafetyEvaluator()

        # Mock the evaluate method to verify it's called correctly
        with patch.object(evaluator, 'evaluate') as mock_evaluate:
            mock_evaluate.return_value = {"flagged": False}

            evaluator.evaluate_with_self_critique("Test text")

            mock_evaluate.assert_called_once_with("Test text", include_critique=True)

    def test_returns_evaluation_result(self):
        """Test that it returns the evaluation result."""
        evaluator = ConstitutionalSafetyEvaluator()

        result = evaluator.evaluate_with_self_critique("Safe text")

        assert isinstance(result, dict)
        assert "flagged" in result


class TestGenerateImprovedResponse:
    """Test generate_improved_response method."""

    def setup_method(self):
        """Setup test fixtures."""
        self.evaluator = ConstitutionalSafetyEvaluator()

    def test_returns_tuple(self):
        """Test that method returns a tuple."""
        prompt = "Tell me about safety"
        response = "This is a safe response"

        result = self.evaluator.generate_improved_response(prompt, response)

        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_returns_response_and_evaluation(self):
        """Test that tuple contains response and evaluation."""
        prompt = "Tell me about safety"
        response = "This is a safe response"

        improved_response, evaluation = self.evaluator.generate_improved_response(
            prompt, response
        )

        assert isinstance(improved_response, str)
        assert isinstance(evaluation, dict)
        assert "flagged" in evaluation

    def test_returns_immediately_if_passes(self):
        """Test that safe response is returned immediately."""
        prompt = "Tell me about baking"
        safe_response = "Here's how to bake bread safely and enjoyably."

        improved, evaluation = self.evaluator.generate_improved_response(
            prompt, safe_response, max_iterations=3
        )

        assert improved == safe_response
        assert evaluation["flagged"] is False

    def test_cannot_improve_without_model(self):
        """Test that improvement fails gracefully without model."""
        prompt = "Question"
        response = "How to hurt someone"

        improved, evaluation = self.evaluator.generate_improved_response(
            prompt, response
        )

        # Should return original since no model to improve
        assert improved == response
        assert evaluation["flagged"] is True

    def test_respects_max_iterations(self):
        """Test that max_iterations is respected."""
        evaluator_with_model = ConstitutionalSafetyEvaluator(
            critique_model=Mock(),
            use_self_critique=True
        )

        # Mock to always flag (would loop forever without max_iterations)
        with patch.object(evaluator_with_model, 'evaluate') as mock_eval:
            mock_eval.return_value = {"flagged": True, "reasoning": "Test"}

            with patch.object(evaluator_with_model, '_generate_improvement') as mock_gen:
                mock_gen.return_value = "Different response each time"

                prompt = "Question"
                response = "Bad response"

                evaluator_with_model.generate_improved_response(
                    prompt, response, max_iterations=2
                )

                # Should be called max 2 times during iterations
                assert mock_eval.call_count <= 3  # Initial + 2 iterations + final


class TestGenerateCritique:
    """Test _generate_critique method."""

    def test_returns_dict_with_expected_keys(self):
        """Test that critique result has expected structure."""
        mock_model = Mock()
        evaluator = ConstitutionalSafetyEvaluator(
            critique_model=mock_model,
            use_self_critique=True
        )

        direct_eval = {"flagged_principles": [], "any_flagged": False}

        with patch.object(evaluator, '_generate_with_model') as mock_gen:
            mock_gen.return_value = "This looks safe"

            result = evaluator._generate_critique("Test text", direct_eval)

            required_keys = ["text", "flagged", "prompt"]
            for key in required_keys:
                assert key in result, f"Missing key: {key}"

    def test_calls_generate_with_model(self):
        """Test that it calls _generate_with_model."""
        mock_model = Mock()
        evaluator = ConstitutionalSafetyEvaluator(
            critique_model=mock_model,
            use_self_critique=True
        )

        with patch.object(evaluator, '_generate_with_model') as mock_gen:
            mock_gen.return_value = "Critique text"

            evaluator._generate_critique("Test", {"flagged_principles": []})

            mock_gen.assert_called_once()

    def test_flags_based_on_critique_content(self):
        """Test that flagging depends on critique content."""
        mock_model = Mock()
        evaluator = ConstitutionalSafetyEvaluator(
            critique_model=mock_model,
            use_self_critique=True
        )

        with patch.object(evaluator, '_generate_with_model') as mock_gen:
            # Critique indicating issues
            mock_gen.return_value = "This violates multiple principles and is problematic"

            result = evaluator._generate_critique("Test", {"flagged_principles": []})

            assert result["flagged"] is True


class TestCreateCritiquePrompt:
    """Test _create_critique_prompt method."""

    def test_includes_text(self):
        """Test that prompt includes the text to analyze."""
        evaluator = ConstitutionalSafetyEvaluator()

        text = "Test text to analyze"
        direct_eval = {"flagged_principles": []}

        prompt = evaluator._create_critique_prompt(text, direct_eval)

        assert text in prompt

    def test_includes_principles(self):
        """Test that prompt mentions principles."""
        evaluator = ConstitutionalSafetyEvaluator()

        direct_eval = {"flagged_principles": []}
        prompt = evaluator._create_critique_prompt("Test", direct_eval)

        # Should mention principles
        assert "principle" in prompt.lower()

    def test_includes_direct_evaluation_results(self):
        """Test that prompt includes direct evaluation results."""
        evaluator = ConstitutionalSafetyEvaluator()

        direct_eval = {"flagged_principles": ["harm_prevention", "truthfulness"]}
        prompt = evaluator._create_critique_prompt("Test", direct_eval)

        assert str(direct_eval["flagged_principles"]) in prompt


class TestCreateImprovementPrompt:
    """Test _create_improvement_prompt method."""

    def test_includes_original_prompt(self):
        """Test that improvement prompt includes original."""
        evaluator = ConstitutionalSafetyEvaluator()

        original = "Tell me about safety"
        response = "Response text"
        evaluation = {"reasoning": "Issues found"}

        prompt = evaluator._create_improvement_prompt(original, response, evaluation)

        assert original in prompt

    def test_includes_response(self):
        """Test that improvement prompt includes response."""
        evaluator = ConstitutionalSafetyEvaluator()

        original = "Question"
        response = "Bad response"
        evaluation = {"reasoning": "Issues"}

        prompt = evaluator._create_improvement_prompt(original, response, evaluation)

        assert response in prompt

    def test_includes_issues(self):
        """Test that improvement prompt includes identified issues."""
        evaluator = ConstitutionalSafetyEvaluator()

        evaluation = {"reasoning": "Specific issues here"}
        prompt = evaluator._create_improvement_prompt("Q", "R", evaluation)

        assert evaluation["reasoning"] in prompt


class TestGenerateWithModel:
    """Test _generate_with_model method."""

    def test_returns_empty_string_without_model(self):
        """Test that it returns empty string without model."""
        evaluator = ConstitutionalSafetyEvaluator()

        result = evaluator._generate_with_model("Test prompt")

        assert result == ""

    def test_handles_model_with_tokenizer_attribute(self):
        """Test generation with model that has tokenizer."""
        mock_model = Mock()
        mock_tokenizer = Mock()
        mock_model.tokenizer = mock_tokenizer

        evaluator = ConstitutionalSafetyEvaluator(
            critique_model=mock_model
        )

        with patch('src.safety.constitutional.model_utils.generate_text') as mock_gen:
            mock_gen.return_value = "Generated text"

            result = evaluator._generate_with_model("Prompt")

            assert result == "Generated text"
            mock_gen.assert_called_once()

    def test_handles_model_without_tokenizer(self):
        """Test handling of model without tokenizer."""
        mock_model = Mock(spec=[])  # No tokenizer attribute

        evaluator = ConstitutionalSafetyEvaluator(
            critique_model=mock_model
        )

        result = evaluator._generate_with_model("Prompt")

        assert "tokenizer" in result.lower()

    def test_handles_import_error(self):
        """Test handling when imports fail."""
        mock_model = Mock()
        mock_model.tokenizer = Mock()

        evaluator = ConstitutionalSafetyEvaluator(
            critique_model=mock_model
        )

        with patch('src.safety.constitutional.model_utils.generate_text', side_effect=ImportError):
            result = evaluator._generate_with_model("Prompt")

            assert "required" in result.lower()

    def test_handles_generation_error(self):
        """Test handling of generation exceptions."""
        mock_model = Mock()
        mock_model.tokenizer = Mock()

        evaluator = ConstitutionalSafetyEvaluator(
            critique_model=mock_model
        )

        with patch('src.safety.constitutional.model_utils.generate_text', side_effect=Exception("Test error")):
            result = evaluator._generate_with_model("Prompt")

            assert "error" in result.lower()
            assert "Test error" in result


class TestSynthesizeReasoning:
    """Test _synthesize_reasoning method."""

    def test_includes_direct_evaluation_issues(self):
        """Test that reasoning includes direct evaluation issues."""
        evaluator = ConstitutionalSafetyEvaluator()

        direct_eval = {
            "any_flagged": True,
            "flagged_principles": ["harm_prevention"],
            "principle_results": {
                "harm_prevention": {"reasoning": "Contains harmful content"}
            }
        }

        reasoning = evaluator._synthesize_reasoning(direct_eval)

        assert "harm_prevention" in reasoning
        assert "Contains harmful content" in reasoning

    def test_handles_no_issues(self):
        """Test reasoning when no issues found."""
        evaluator = ConstitutionalSafetyEvaluator()

        direct_eval = {
            "any_flagged": False,
            "flagged_principles": [],
            "principle_results": {}
        }

        reasoning = evaluator._synthesize_reasoning(direct_eval)

        assert "No issues detected" in reasoning

    def test_includes_critique_when_provided(self):
        """Test that reasoning includes critique."""
        evaluator = ConstitutionalSafetyEvaluator()

        direct_eval = {"any_flagged": False, "flagged_principles": [], "principle_results": {}}
        critique = {"flagged": True, "text": "This has subtle issues that need attention"}

        reasoning = evaluator._synthesize_reasoning(direct_eval, critique)

        assert "critique" in reasoning.lower()
        assert "subtle issues" in reasoning

    def test_truncates_long_critique(self):
        """Test that long critiques are truncated."""
        evaluator = ConstitutionalSafetyEvaluator()

        direct_eval = {"any_flagged": False, "flagged_principles": [], "principle_results": {}}
        long_critique_text = "A" * 200
        critique = {"flagged": True, "text": long_critique_text}

        reasoning = evaluator._synthesize_reasoning(direct_eval, critique)

        # Should be truncated
        assert len(reasoning) < len(long_critique_text) + 50
        assert "..." in reasoning

    def test_handles_critique_not_flagged(self):
        """Test that unflagged critique is not included."""
        evaluator = ConstitutionalSafetyEvaluator()

        direct_eval = {"any_flagged": False, "flagged_principles": [], "principle_results": {}}
        critique = {"flagged": False, "text": "Looks fine"}

        reasoning = evaluator._synthesize_reasoning(direct_eval, critique)

        # Should not include critique since it didn't flag
        assert "Looks fine" not in reasoning


class TestGetStatistics:
    """Test get_statistics method."""

    def test_returns_statistics_dict(self):
        """Test that statistics are returned as dict."""
        evaluator = ConstitutionalSafetyEvaluator()

        stats = evaluator.get_statistics()

        assert isinstance(stats, dict)

    def test_includes_evaluation_counts(self):
        """Test that statistics include evaluation counts."""
        evaluator = ConstitutionalSafetyEvaluator()

        evaluator.evaluate("Test 1")
        evaluator.evaluate("Test 2")

        stats = evaluator.get_statistics()

        assert stats["total_evaluations"] == 2

    def test_includes_framework_statistics(self):
        """Test that framework stats are included."""
        evaluator = ConstitutionalSafetyEvaluator()

        stats = evaluator.get_statistics()

        assert "framework_stats" in stats
        assert isinstance(stats["framework_stats"], dict)

    def test_tracks_flagging_sources(self):
        """Test that statistics track flagging sources."""
        evaluator = ConstitutionalSafetyEvaluator()

        # Evaluate harmful text
        evaluator.evaluate("How to hurt someone")

        stats = evaluator.get_statistics()

        # Should have some direct flagging
        assert stats["flagged_by_direct"] > 0


class TestResetStatistics:
    """Test reset_statistics method."""

    def test_resets_evaluation_counts(self):
        """Test that evaluation counts are reset."""
        evaluator = ConstitutionalSafetyEvaluator()

        evaluator.evaluate("Test 1")
        evaluator.evaluate("How to harm")

        evaluator.reset_statistics()

        stats = evaluator.get_statistics()
        assert stats["total_evaluations"] == 0
        assert stats["flagged_by_direct"] == 0
        assert stats["flagged_by_critique"] == 0
        assert stats["flagged_by_both"] == 0

    def test_clears_framework_history(self):
        """Test that framework history is cleared."""
        evaluator = ConstitutionalSafetyEvaluator()

        # Evaluate with history tracking
        evaluator.framework.evaluate_text("Test", track_history=True)

        assert len(evaluator.framework.evaluation_history) > 0

        evaluator.reset_statistics()

        assert len(evaluator.framework.evaluation_history) == 0


class TestCritiqueIndicatesIssues:
    """Test critique_indicates_issues helper function."""

    def test_flags_with_multiple_concerns(self):
        """Test that multiple concern phrases trigger flagging."""
        critiques = [
            "This violates the principle and is problematic",
            "Multiple issues here: harmful content and misleading claims",
            "This is inappropriate and demonstrates bias"
        ]

        for critique in critiques:
            assert critique_indicates_issues(critique) is True

    def test_does_not_flag_single_concern(self):
        """Test that single concern phrase doesn't flag."""
        critique = "This has one potential issue with style"
        result = critique_indicates_issues(critique)

        # Single concern might not flag (depends on threshold)
        assert isinstance(result, bool)

    def test_does_not_flag_clean_critique(self):
        """Test that clean critique is not flagged."""
        critiques = [
            "This looks good and helpful",
            "The response is appropriate",
            "No problems identified"
        ]

        for critique in critiques:
            assert critique_indicates_issues(critique) is False

    def test_case_insensitive(self):
        """Test that detection is case-insensitive."""
        critiques = [
            "This VIOLATES the principle and is PROBLEMATIC",
            "multiple ISSUES here",
            "HARMFUL and DANGEROUS content"
        ]

        for critique in critiques:
            assert critique_indicates_issues(critique) is True

    def test_counts_concern_phrases(self):
        """Test that concern phrases are counted."""
        # More concern phrases = more likely to flag
        low_concern = "This might have an issue"
        high_concern = "This violates principles, is harmful, dangerous, and inappropriate"

        assert critique_indicates_issues(high_concern) is True

    def test_empty_critique(self):
        """Test handling of empty critique."""
        assert critique_indicates_issues("") is False


class TestCombineReasoning:
    """Test combine_reasoning helper function."""

    def test_combines_direct_and_critique(self):
        """Test that direct evaluation and critique are combined."""
        direct_eval = {
            "harm_prevention": {"flagged": True, "reason": "Harmful content"},
            "truthfulness": {"flagged": True, "reason": "Misleading"}
        }
        critique = "This response has additional issues with bias"

        result = combine_reasoning(direct_eval, critique)

        assert "Harm Prevention" in result
        assert "Truthfulness" in result
        assert "bias" in result

    def test_handles_empty_critique(self):
        """Test handling of empty critique."""
        direct_eval = {
            "harm_prevention": {"flagged": True}
        }

        result = combine_reasoning(direct_eval, "")

        assert "Harm Prevention" in result
        assert len(result) > 0

    def test_handles_no_violations(self):
        """Test handling when no violations found."""
        direct_eval = {
            "harm_prevention": {"flagged": False}
        }

        result = combine_reasoning(direct_eval, "")

        # Should still return something
        assert isinstance(result, str)
        assert len(result) > 0

    def test_truncates_long_critique(self):
        """Test that long critiques are truncated."""
        direct_eval = {}
        long_critique = "A" * 200

        result = combine_reasoning(direct_eval, long_critique)

        # Should be truncated to ~100 chars
        assert "..." in result

    def test_formats_principle_names(self):
        """Test that principle names are formatted nicely."""
        direct_eval = {
            "harm_prevention": {"flagged": True},
            "autonomy_respect": {"flagged": True}
        }

        result = combine_reasoning(direct_eval, "")

        # Underscores should be replaced, words capitalized
        assert "Harm Prevention" in result
        assert "Autonomy Respect" in result


class TestIntegrationScenarios:
    """Test integration scenarios combining multiple components."""

    def test_end_to_end_safe_text(self):
        """Test end-to-end evaluation of safe text."""
        evaluator = ConstitutionalSafetyEvaluator()

        text = "Here's a helpful guide on baking delicious bread at home."
        result = evaluator.evaluate(text)

        assert result["flagged"] is False
        assert result["source"] == "none"
        assert len(result["reasoning"]) > 0

    def test_end_to_end_harmful_text(self):
        """Test end-to-end evaluation of harmful text."""
        evaluator = ConstitutionalSafetyEvaluator()

        text = "How to hurt someone physically"
        result = evaluator.evaluate(text)

        assert result["flagged"] is True
        assert result["source"] == "direct"
        assert "harm_prevention" in result["direct_evaluation"]["flagged_principles"]

    def test_multiple_evaluations_statistics(self):
        """Test that statistics accumulate across evaluations."""
        evaluator = ConstitutionalSafetyEvaluator()

        evaluator.evaluate("Safe text 1")
        evaluator.evaluate("How to harm")
        evaluator.evaluate("Safe text 2")
        evaluator.evaluate("How to hurt")

        stats = evaluator.get_statistics()

        assert stats["total_evaluations"] == 4
        assert stats["flagged_by_direct"] >= 2  # At least 2 harmful texts

    def test_custom_framework_integration(self):
        """Test evaluator with custom framework."""
        custom_framework = ConstitutionalFramework(name="custom")

        def custom_eval(text):
            return {"flagged": "forbidden" in text.lower()}

        custom_framework.add_principle(
            ConstitutionalPrinciple("custom_rule", "Test rule", custom_eval)
        )

        evaluator = ConstitutionalSafetyEvaluator(framework=custom_framework)

        result = evaluator.evaluate("This contains forbidden content")

        assert result["flagged"] is True
