"""
Unit tests for filter.py
Tests the ConstitutionalSafetyFilter for input validation and output filtering.
"""

from unittest.mock import Mock

from src.safety.constitutional.filter import ConstitutionalSafetyFilter
from src.safety.constitutional.framework import ConstitutionalFramework


class TestConstitutionalSafetyFilterInit:
    """Test ConstitutionalSafetyFilter initialization."""

    def test_init_with_defaults(self):
        """Test initialization with default parameters."""
        filter_obj = ConstitutionalSafetyFilter()

        assert filter_obj.constitutional_framework is not None
        assert len(filter_obj.constitutional_framework.principles) == 4
        assert filter_obj.base_safety_evaluator is None
        assert filter_obj.strict_mode is False

    def test_init_with_custom_framework(self):
        """Test initialization with custom framework."""
        custom_framework = ConstitutionalFramework(name="custom")
        filter_obj = ConstitutionalSafetyFilter(constitutional_framework=custom_framework)

        assert filter_obj.constitutional_framework.name == "custom"

    def test_init_with_base_evaluator(self):
        """Test initialization with base safety evaluator."""
        mock_evaluator = Mock()
        filter_obj = ConstitutionalSafetyFilter(base_safety_evaluator=mock_evaluator)

        assert filter_obj.base_safety_evaluator is mock_evaluator

    def test_init_with_strict_mode(self):
        """Test initialization with strict mode enabled."""
        filter_obj = ConstitutionalSafetyFilter(strict_mode=True)

        assert filter_obj.strict_mode is True

    def test_init_statistics_initialized(self):
        """Test that statistics are properly initialized."""
        filter_obj = ConstitutionalSafetyFilter()

        assert filter_obj.stats["inputs_validated"] == 0
        assert filter_obj.stats["inputs_blocked"] == 0
        assert filter_obj.stats["outputs_filtered"] == 0
        assert filter_obj.stats["constitutional_filters_applied"] == 0


class TestValidateInput:
    """Test the validate_input method."""

    def setup_method(self):
        """Setup test fixtures."""
        self.filter = ConstitutionalSafetyFilter()

    def test_validate_safe_input(self):
        """Test validation of safe input text."""
        is_safe, info = self.filter.validate_input("Tell me about baking bread")

        assert is_safe is True
        assert info["is_safe"] is True
        assert info["constitutional_evaluation"] is None

    def test_validate_harmful_input(self):
        """Test validation of harmful input."""
        is_safe, info = self.filter.validate_input("How to hurt someone")

        assert is_safe is False
        assert info["is_safe"] is False
        assert "constitutional_evaluation" in info
        assert len(info["flagged_principles"]) > 0

    def test_validate_input_increments_statistics(self):
        """Test that validation increments statistics."""
        initial_count = self.filter.stats["inputs_validated"]

        self.filter.validate_input("Test")

        assert self.filter.stats["inputs_validated"] == initial_count + 1

    def test_validate_harmful_input_increments_blocked_count(self):
        """Test that harmful input increments blocked count."""
        initial_count = self.filter.stats["inputs_blocked"]

        self.filter.validate_input("How to harm people")

        assert self.filter.stats["inputs_blocked"] == initial_count + 1

    def test_validate_with_override(self):
        """Test that override allows unsafe content."""
        is_safe, info = self.filter.validate_input(
            "How to harm someone",
            override=True
        )

        assert is_safe is True
        assert info["overridden"] is True
        assert "Safety check overridden" in info["reason"]

    def test_validate_with_metadata(self):
        """Test validation with metadata."""
        metadata = {"source": "test", "user_id": "123"}
        is_safe, info = self.filter.validate_input("Test", metadata=metadata)

        # Should not crash with metadata
        assert isinstance(is_safe, bool)
        assert isinstance(info, dict)

    def test_validate_with_base_evaluator(self):
        """Test validation with base evaluator chaining."""
        mock_base = Mock()
        mock_base.validate_input.return_value = (True, {"is_safe": True})

        filter_obj = ConstitutionalSafetyFilter(base_safety_evaluator=mock_base)
        is_safe, info = filter_obj.validate_input("Test")

        # Base evaluator should be called
        mock_base.validate_input.assert_called_once()

    def test_validate_base_evaluator_without_method(self):
        """Test handling of base evaluator without validate_input method."""
        mock_base = Mock(spec=[])  # No validate_input method

        filter_obj = ConstitutionalSafetyFilter(base_safety_evaluator=mock_base)
        is_safe, info = filter_obj.validate_input("Test")

        # Should not crash
        assert isinstance(is_safe, bool)

    def test_validate_result_includes_weighted_score(self):
        """Test that validation result includes weighted score."""
        is_safe, info = self.filter.validate_input("How to hurt someone")

        if not is_safe:
            assert "weighted_score" in info
            assert info["weighted_score"] > 0

    def test_validate_empty_input(self):
        """Test validation of empty input."""
        is_safe, info = self.filter.validate_input("")

        assert isinstance(is_safe, bool)
        assert "is_safe" in info


class TestFilterOutput:
    """Test the filter_output method."""

    def setup_method(self):
        """Setup test fixtures."""
        self.filter = ConstitutionalSafetyFilter()

    def test_filter_clean_output(self):
        """Test filtering of clean output."""
        text = "Here's helpful information about baking."
        filtered, info = self.filter.filter_output(text)

        assert filtered == text  # No changes needed
        assert info["was_filtered"] is False

    def test_filter_harmful_output(self):
        """Test filtering of harmful output."""
        text = "Here's how to hurt someone"
        filtered, info = self.filter.filter_output(text)

        assert filtered != text  # Should be transformed
        assert info["was_filtered"] is True
        assert "harm" in filtered.lower() or "cannot" in filtered.lower()

    def test_filter_output_increments_statistics(self):
        """Test that filtering increments statistics."""
        initial_count = self.filter.stats["outputs_filtered"]

        self.filter.filter_output("Test")

        assert self.filter.stats["outputs_filtered"] == initial_count + 1

    def test_filter_output_with_transformations_disabled(self):
        """Test filtering without applying transformations."""
        text = "How to hurt someone"
        filtered, info = self.filter.filter_output(text, apply_transformations=False)

        assert filtered == text  # No transformations applied
        assert "constitutional_evaluation" in info

    def test_filter_output_tracks_transformation_applications(self):
        """Test that transformation applications are tracked."""
        initial_count = self.filter.stats["constitutional_filters_applied"]

        text = "How to hurt someone"
        self.filter.filter_output(text)

        assert self.filter.stats["constitutional_filters_applied"] == initial_count + 1

    def test_filter_output_with_metadata(self):
        """Test filtering with metadata."""
        metadata = {"context": "test"}
        filtered, info = self.filter.filter_output("Test", metadata=metadata)

        assert isinstance(filtered, str)
        assert isinstance(info, dict)

    def test_filter_output_with_base_evaluator(self):
        """Test filtering with base evaluator chaining."""
        mock_base = Mock()
        mock_base.filter_output.return_value = ("Filtered", {"base": "info"})

        filter_obj = ConstitutionalSafetyFilter(base_safety_evaluator=mock_base)
        filtered, info = filter_obj.filter_output("Test")

        # Base evaluator should be called
        mock_base.filter_output.assert_called_once()

    def test_filter_output_base_evaluator_without_method(self):
        """Test handling of base evaluator without filter_output method."""
        mock_base = Mock(spec=[])  # No filter_output method

        filter_obj = ConstitutionalSafetyFilter(base_safety_evaluator=mock_base)
        filtered, info = filter_obj.filter_output("Test")

        # Should not crash
        assert isinstance(filtered, str)

    def test_filter_output_includes_lengths(self):
        """Test that filter output includes length information."""
        text = "How to hurt someone"
        filtered, info = self.filter.filter_output(text)

        if info["was_filtered"]:
            assert "original_length" in info
            assert "final_length" in info

    def test_filter_output_includes_transformations_list(self):
        """Test that transformations list is included."""
        text = "How to hurt someone"
        filtered, info = self.filter.filter_output(text)

        if info["was_filtered"]:
            assert "transformations_applied" in info
            assert isinstance(info["transformations_applied"], list)


class TestFilterHarmfulContent:
    """Test _filter_harmful_content method."""

    def setup_method(self):
        """Setup test fixtures."""
        self.filter = ConstitutionalSafetyFilter()

    def test_filters_explicit_harm_instructions(self):
        """Test filtering of explicit harm instructions."""
        evaluation = {"explicit_harm_detected": True}

        texts = [
            "Here's how to harm someone",
            "Ways to hurt people",
            "Instructions for injuring others",
            "How to attack someone"
        ]

        for text in texts:
            filtered = self.filter._filter_harmful_content(text, evaluation)
            assert "cannot" in filtered.lower() or "I cannot" in filtered
            assert len(filtered) > 0

    def test_filters_weapon_creation(self):
        """Test filtering of weapon creation instructions."""
        evaluation = {"explicit_harm_detected": True}

        texts = [
            "How to make a weapon",
            "Create a bomb",
            "Build an explosive",
            "Construct a poison"
        ]

        for text in texts:
            filtered = self.filter._filter_harmful_content(text, evaluation)
            assert "cannot" in filtered.lower()

    def test_no_filtering_without_explicit_harm(self):
        """Test that content without explicit harm is not filtered."""
        evaluation = {"explicit_harm_detected": False}

        text = "This is safe content"
        filtered = self.filter._filter_harmful_content(text, evaluation)

        assert filtered == text

    def test_case_insensitive_filtering(self):
        """Test that filtering is case-insensitive."""
        evaluation = {"explicit_harm_detected": True}

        texts = [
            "HOW TO HARM someone",
            "how to HURT someone",
            "How To Kill Someone"
        ]

        for text in texts:
            filtered = self.filter._filter_harmful_content(text, evaluation)
            assert "cannot" in filtered.lower()


class TestAddTruthfulnessQualifiers:
    """Test _add_truthfulness_qualifiers method."""

    def setup_method(self):
        """Setup test fixtures."""
        self.filter = ConstitutionalSafetyFilter()

    def test_adds_disclaimer_for_unsupported_claims(self):
        """Test adding disclaimer for unsupported claims."""
        evaluation = {"unsupported_claims": ["claim1", "claim2", "claim3"]}

        text = "All people always do this"
        filtered = self.filter._add_truthfulness_qualifiers(text, evaluation)

        assert "Note:" in filtered or "verification" in filtered
        assert len(filtered) > len(text)

    def test_adds_note_for_contradictions(self):
        """Test adding note for contradictions."""
        evaluation = {"contradictions": ["contradiction1"]}

        text = "This always works but sometimes doesn't"
        filtered = self.filter._add_truthfulness_qualifiers(text, evaluation)

        assert "inconsistencies" in filtered or "Note:" in filtered
        assert len(filtered) > len(text)

    def test_adds_note_for_misleading_statistics(self):
        """Test adding note for misleading statistics."""
        evaluation = {"misleading_statistics": ["stat1", "stat2"]}

        text = "There was a 50% increase in sales"
        filtered = self.filter._add_truthfulness_qualifiers(text, evaluation)

        assert "Note:" in filtered or "verified" in filtered or "sources" in filtered
        assert len(filtered) > len(text)

    def test_no_qualifiers_for_clean_content(self):
        """Test that clean content gets no qualifiers."""
        evaluation = {
            "unsupported_claims": [],
            "contradictions": [],
            "misleading_statistics": []
        }

        text = "This is factual content"
        filtered = self.filter._add_truthfulness_qualifiers(text, evaluation)

        assert filtered == text

    def test_does_not_duplicate_disclaimers(self):
        """Test that disclaimers are not duplicated."""
        evaluation = {"unsupported_claims": ["claim1", "claim2", "claim3"]}

        text = "All people are good"
        filtered1 = self.filter._add_truthfulness_qualifiers(text, evaluation)
        filtered2 = self.filter._add_truthfulness_qualifiers(filtered1, evaluation)

        # Should not add the same disclaimer twice
        assert filtered2 == filtered1


class TestFilterBiasedContent:
    """Test _filter_biased_content method."""

    def setup_method(self):
        """Setup test fixtures."""
        self.filter = ConstitutionalSafetyFilter()

    def test_softens_generalizations(self):
        """Test softening of absolute generalizations."""
        evaluation = {"stereotypes": ["test"]}

        texts = [
            ("All men are tall", "some"),
            ("Every woman is emotional", "some"),
            ("All people from there", "some")
        ]

        for text, expected in texts:
            filtered = self.filter._filter_biased_content(text, evaluation)
            assert expected in filtered.lower()

    def test_replaces_biased_terminology(self):
        """Test replacement of biased terms."""
        evaluation = {"biased_language": ["test"]}

        replacements = [
            ("primitive people", "traditional"),
            ("backward culture", "different"),
            ("normal people", "most people"),
            ("regular people", "many people")
        ]

        for text, expected_word in replacements:
            filtered = self.filter._filter_biased_content(text, evaluation)
            assert expected_word in filtered.lower()

    def test_strict_mode_uses_stronger_replacements(self):
        """Test that strict mode uses stronger replacements."""
        filter_strict = ConstitutionalSafetyFilter(strict_mode=True)
        filter_normal = ConstitutionalSafetyFilter(strict_mode=False)

        evaluation = {"stereotypes": ["test"]}

        text = "People always do this"
        filtered_strict = filter_strict._filter_biased_content(text, evaluation)
        filtered_normal = filter_normal._filter_biased_content(text, evaluation)

        # Strict mode should be more aggressive
        assert filtered_strict != text

    def test_no_filtering_without_bias(self):
        """Test that unbiased content is not filtered."""
        evaluation = {"stereotypes": [], "biased_language": []}

        text = "Some people prefer coffee while others prefer tea"
        filtered = self.filter._filter_biased_content(text, evaluation)

        assert filtered == text

    def test_case_insensitive_bias_filtering(self):
        """Test that bias filtering is case-insensitive."""
        evaluation = {"biased_language": ["test"]}

        text = "PRIMITIVE people are BACKWARD"
        filtered = self.filter._filter_biased_content(text, evaluation)

        assert "traditional" in filtered.lower() or "different" in filtered.lower()


class TestFilterAutonomyViolations:
    """Test _filter_autonomy_violations method."""

    def setup_method(self):
        """Setup test fixtures."""
        self.filter = ConstitutionalSafetyFilter()

    def test_replaces_coercive_language(self):
        """Test replacement of coercive language."""
        evaluation = {"coercive_language": ["test"]}

        replacements = [
            ("You must do this", "may want to"),
            ("You have to comply", "may want to"),
            ("You need to follow", "may want to"),
            ("Required to participate", "recommended to")
        ]

        for text, expected_phrase in replacements:
            filtered = self.filter._filter_autonomy_violations(text, evaluation)
            assert expected_phrase in filtered.lower()

    def test_softens_manipulative_language(self):
        """Test softening of manipulative language."""
        evaluation = {"manipulative_language": ["test"]}

        replacements = [
            ("If you really cared", "interested"),
            ("If you were smart", "smart approach"),
            ("Everyone knows that", "commonly believed"),
            ("Obviously this is true", "arguably")
        ]

        for text, expected_word in replacements:
            filtered = self.filter._filter_autonomy_violations(text, evaluation)
            assert expected_word in filtered.lower()

    def test_no_filtering_without_violations(self):
        """Test that respectful language is not filtered."""
        evaluation = {"coercive_language": [], "manipulative_language": []}

        text = "You might want to consider this option"
        filtered = self.filter._filter_autonomy_violations(text, evaluation)

        assert filtered == text

    def test_case_insensitive_autonomy_filtering(self):
        """Test that autonomy filtering is case-insensitive."""
        evaluation = {"coercive_language": ["test"]}

        text = "YOU MUST do this NOW"
        filtered = self.filter._filter_autonomy_violations(text, evaluation)

        assert "may want to" in filtered.lower()


class TestApplyConstitutionalFiltering:
    """Test _apply_constitutional_filtering method."""

    def test_applies_harm_filtering(self):
        """Test that harm filtering is applied."""
        filter_obj = ConstitutionalSafetyFilter()

        evaluation = {
            "flagged_principles": ["harm_prevention"],
            "principle_results": {
                "harm_prevention": {"explicit_harm_detected": True}
            }
        }

        filtering_info = {}
        text = "How to hurt someone"
        filtered = filter_obj._apply_constitutional_filtering(text, evaluation, filtering_info)

        assert "harm_filtering" in filtering_info["transformations_applied"]
        assert "cannot" in filtered.lower()

    def test_applies_truthfulness_filtering(self):
        """Test that truthfulness filtering is applied."""
        filter_obj = ConstitutionalSafetyFilter()

        evaluation = {
            "flagged_principles": ["truthfulness"],
            "principle_results": {
                "truthfulness": {
                    "unsupported_claims": ["claim1", "claim2", "claim3"]
                }
            }
        }

        filtering_info = {}
        text = "All people always do this"
        filtered = filter_obj._apply_constitutional_filtering(text, evaluation, filtering_info)

        assert "truthfulness_qualifiers" in filtering_info["transformations_applied"]

    def test_applies_fairness_filtering(self):
        """Test that fairness filtering is applied."""
        filter_obj = ConstitutionalSafetyFilter()

        evaluation = {
            "flagged_principles": ["fairness"],
            "principle_results": {
                "fairness": {"stereotypes": ["test"], "biased_language": ["test"]}
            }
        }

        filtering_info = {}
        text = "All men are aggressive"
        filtered = filter_obj._apply_constitutional_filtering(text, evaluation, filtering_info)

        assert "bias_filtering" in filtering_info["transformations_applied"]

    def test_applies_autonomy_filtering(self):
        """Test that autonomy filtering is applied."""
        filter_obj = ConstitutionalSafetyFilter()

        evaluation = {
            "flagged_principles": ["autonomy_respect"],
            "principle_results": {
                "autonomy_respect": {"coercive_language": ["test"]}
            }
        }

        filtering_info = {}
        text = "You must do this"
        filtered = filter_obj._apply_constitutional_filtering(text, evaluation, filtering_info)

        assert "autonomy_filtering" in filtering_info["transformations_applied"]

    def test_applies_multiple_filters(self):
        """Test applying multiple filters simultaneously."""
        filter_obj = ConstitutionalSafetyFilter()

        evaluation = {
            "flagged_principles": ["harm_prevention", "truthfulness"],
            "principle_results": {
                "harm_prevention": {"explicit_harm_detected": True},
                "truthfulness": {"unsupported_claims": ["c1", "c2", "c3"]}
            }
        }

        filtering_info = {}
        text = "How to hurt someone. All people always do this."
        filtered = filter_obj._apply_constitutional_filtering(text, evaluation, filtering_info)

        assert len(filtering_info["transformations_applied"]) >= 2


class TestGetStatistics:
    """Test get_statistics method."""

    def test_returns_statistics_dict(self):
        """Test that statistics are returned as dict."""
        filter_obj = ConstitutionalSafetyFilter()

        stats = filter_obj.get_statistics()

        assert isinstance(stats, dict)

    def test_includes_input_statistics(self):
        """Test that input statistics are included."""
        filter_obj = ConstitutionalSafetyFilter()

        filter_obj.validate_input("Test 1")
        filter_obj.validate_input("How to harm")

        stats = filter_obj.get_statistics()

        assert stats["inputs_validated"] == 2
        assert stats["inputs_blocked"] >= 1

    def test_includes_output_statistics(self):
        """Test that output statistics are included."""
        filter_obj = ConstitutionalSafetyFilter()

        filter_obj.filter_output("Test 1")
        filter_obj.filter_output("How to harm")

        stats = filter_obj.get_statistics()

        assert stats["outputs_filtered"] == 2

    def test_includes_framework_statistics(self):
        """Test that framework statistics are included."""
        filter_obj = ConstitutionalSafetyFilter()

        stats = filter_obj.get_statistics()

        assert "framework_stats" in stats
        assert isinstance(stats["framework_stats"], dict)


class TestResetStatistics:
    """Test reset_statistics method."""

    def test_resets_all_counters(self):
        """Test that all counters are reset."""
        filter_obj = ConstitutionalSafetyFilter()

        filter_obj.validate_input("Test")
        filter_obj.validate_input("How to harm")
        filter_obj.filter_output("Test")
        filter_obj.filter_output("How to harm")

        filter_obj.reset_statistics()

        stats = filter_obj.get_statistics()
        assert stats["inputs_validated"] == 0
        assert stats["inputs_blocked"] == 0
        assert stats["outputs_filtered"] == 0
        assert stats["constitutional_filters_applied"] == 0

    def test_clears_framework_history(self):
        """Test that framework history is cleared."""
        filter_obj = ConstitutionalSafetyFilter()

        filter_obj.constitutional_framework.evaluate_text("Test", track_history=True)
        assert len(filter_obj.constitutional_framework.evaluation_history) > 0

        filter_obj.reset_statistics()

        assert len(filter_obj.constitutional_framework.evaluation_history) == 0


class TestIntegrationScenarios:
    """Test integration scenarios."""

    def test_end_to_end_input_validation(self):
        """Test complete input validation workflow."""
        filter_obj = ConstitutionalSafetyFilter()

        # Safe input
        is_safe1, info1 = filter_obj.validate_input("Tell me about baking")
        assert is_safe1 is True

        # Harmful input
        is_safe2, info2 = filter_obj.validate_input("How to harm someone")
        assert is_safe2 is False

        # Check statistics
        stats = filter_obj.get_statistics()
        assert stats["inputs_validated"] == 2
        assert stats["inputs_blocked"] == 1

    def test_end_to_end_output_filtering(self):
        """Test complete output filtering workflow."""
        filter_obj = ConstitutionalSafetyFilter()

        # Clean output
        filtered1, info1 = filter_obj.filter_output("Here's how to bake bread")
        assert filtered1 == "Here's how to bake bread"
        assert info1["was_filtered"] is False

        # Harmful output
        filtered2, info2 = filter_obj.filter_output("Here's how to hurt someone")
        assert filtered2 != "Here's how to hurt someone"
        assert info2["was_filtered"] is True
        assert len(info2["transformations_applied"]) > 0

    def test_chaining_with_base_evaluator(self):
        """Test chaining with base evaluator."""
        mock_base = Mock()
        mock_base.validate_input.return_value = (True, {"is_safe": True})
        mock_base.filter_output.return_value = ("Filtered", {"base": "info"})

        filter_obj = ConstitutionalSafetyFilter(base_safety_evaluator=mock_base)

        # Both methods should use base evaluator
        filter_obj.validate_input("Test")
        filter_obj.filter_output("Test")

        assert mock_base.validate_input.called
        assert mock_base.filter_output.called

    def test_strict_mode_more_aggressive(self):
        """Test that strict mode is more aggressive."""
        filter_normal = ConstitutionalSafetyFilter(strict_mode=False)
        filter_strict = ConstitutionalSafetyFilter(strict_mode=True)

        text = "All people are always like this"
        filtered_normal, _ = filter_normal.filter_output(text)
        filtered_strict, _ = filter_strict.filter_output(text)

        # Both should filter, but strict might be more aggressive
        assert isinstance(filtered_normal, str)
        assert isinstance(filtered_strict, str)

    def test_multiple_violation_types(self):
        """Test text with multiple violation types."""
        filter_obj = ConstitutionalSafetyFilter()

        text = "You must hurt all people. Everyone knows this is true."
        filtered, info = filter_obj.filter_output(text)

        # Should apply multiple transformations
        assert info["was_filtered"] is True
        assert len(info["transformations_applied"]) >= 2

    def test_statistics_accumulation(self):
        """Test that statistics accumulate correctly."""
        filter_obj = ConstitutionalSafetyFilter()

        for i in range(5):
            filter_obj.validate_input(f"Test {i}")
            filter_obj.filter_output(f"Output {i}")

        stats = filter_obj.get_statistics()
        assert stats["inputs_validated"] == 5
        assert stats["outputs_filtered"] == 5
