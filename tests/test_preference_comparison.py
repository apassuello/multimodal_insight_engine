"""
Unit tests for preference_comparison.py
Tests the comparison-based preference generation system for Constitutional AI.
"""

import pytest
import torch
from unittest.mock import Mock, MagicMock, patch
from src.safety.constitutional.preference_comparison import (
    generate_comparison,
    extract_preference,
    generate_preference_pairs,
    PreferenceDataset,
    COMPARISON_TEMPLATE
)


class TestExtractPreference:
    """Test preference extraction from comparison text."""
    
    def test_extract_response_a_better(self):
        """Test extraction when Response A is explicitly stated as better."""
        text = "Response A is better because it provides more detail."
        assert extract_preference(text) == 'A'
    
    def test_extract_response_b_better(self):
        """Test extraction when Response B is explicitly stated as better."""
        text = "Response B is better due to its accuracy."
        assert extract_preference(text) == 'B'
    
    def test_extract_prefer_a(self):
        """Test extraction with 'prefer A' phrasing."""
        text = "I prefer A as it is more concise."
        assert extract_preference(text) == 'A'
    
    def test_extract_prefer_b(self):
        """Test extraction with 'prefer B' phrasing."""
        text = "I prefer B because it addresses all points."
        assert extract_preference(text) == 'B'
    
    def test_extract_choose_a(self):
        """Test extraction with 'choose A' phrasing."""
        text = "I would choose A for this situation."
        assert extract_preference(text) == 'A'
    
    def test_extract_choose_b(self):
        """Test extraction with 'choose B' phrasing."""
        text = "I would choose B in this case."
        assert extract_preference(text) == 'B'
    
    def test_extract_response_b_superior(self):
        """Test extraction with 'superior' terminology."""
        text = "Response B is superior in terms of clarity."
        assert extract_preference(text) == 'B'
    
    def test_extract_response_a_preferred(self):
        """Test extraction with 'preferred' terminology."""
        text = "Response A is preferred for its brevity."
        assert extract_preference(text) == 'A'
    
    def test_extract_better_response_b(self):
        """Test extraction with 'better' before 'Response B'."""
        text = "The better response is Response B."
        assert extract_preference(text) == 'B'
    
    def test_extract_a_is_better(self):
        """Test extraction with 'A is better' phrasing."""
        text = "A is better overall."
        assert extract_preference(text) == 'A'
    
    def test_extract_b_seems_better(self):
        """Test extraction with 'B seems better' phrasing."""
        text = "B seems better for this use case."
        assert extract_preference(text) == 'B'
    
    def test_extract_with_positive_mentions(self):
        """Test extraction based on positive attribute counts."""
        text = "Response A is unclear. Response B is accurate, helpful, and clear."
        assert extract_preference(text) == 'B'
    
    def test_extract_with_negative_mentions(self):
        """Test extraction based on negative attribute counts."""
        text = "Response A is poor and inaccurate. Response B is adequate."
        assert extract_preference(text) == 'B'  # A has more negatives
    
    def test_extract_unclear_defaults_to_a(self):
        """Test that unclear preferences default to A."""
        text = "Both responses have merits and drawbacks."
        assert extract_preference(text) == 'A'
    
    def test_extract_empty_defaults_to_a(self):
        """Test that empty text defaults to A."""
        text = ""
        assert extract_preference(text) == 'A'
    
    def test_extract_case_insensitive(self):
        """Test that extraction is case-insensitive."""
        text = "RESPONSE B IS BETTER"
        assert extract_preference(text) == 'B'
    
    def test_extract_with_explanation(self):
        """Test extraction from text with detailed explanation."""
        text = """After careful analysis, Response A is better because:
        1. It is more accurate
        2. It provides specific examples
        3. It is easier to understand
        Therefore, A is the superior choice."""
        assert extract_preference(text) == 'A'
    
    def test_extract_prefer_with_distance(self):
        """Test extraction when 'prefer' and 'B' are separated by text."""
        text = "Given the context and requirements, I prefer, without hesitation, B."
        assert extract_preference(text) == 'B'


class TestGenerateComparison:
    """Test comparison generation function."""
    
    def test_generate_comparison_returns_dict(self):
        """Test that generate_comparison returns a properly formatted dictionary."""
        # Mock the model and tokenizer
        mock_model = Mock()
        mock_tokenizer = Mock()
        device = torch.device('cpu')
        
        # Mock generate_text to return a comparison
        with patch('src.safety.constitutional.preference_comparison.generate_text') as mock_gen:
            mock_gen.return_value = "Response B is better because it is more accurate."
            
            result = generate_comparison(
                prompt="What is AI?",
                response_a="AI is computers thinking.",
                response_b="AI is a field of computer science focused on creating systems that can perform tasks requiring human intelligence.",
                principles=["Be accurate", "Be helpful"],
                model=mock_model,
                tokenizer=mock_tokenizer,
                device=device
            )
            
            # Check result structure
            assert isinstance(result, dict)
            assert 'preferred' in result
            assert 'comparison_text' in result
            assert 'response_chosen' in result
            assert 'response_rejected' in result
    
    def test_generate_comparison_preference_a(self):
        """Test that comparison correctly identifies preference A."""
        mock_model = Mock()
        mock_tokenizer = Mock()
        device = torch.device('cpu')
        
        with patch('src.safety.constitutional.preference_comparison.generate_text') as mock_gen:
            mock_gen.return_value = "Response A is better."
            
            result = generate_comparison(
                prompt="Test prompt",
                response_a="Response A text",
                response_b="Response B text",
                principles=["Be helpful"],
                model=mock_model,
                tokenizer=mock_tokenizer,
                device=device
            )
            
            assert result['preferred'] == 'A'
            assert result['response_chosen'] == "Response A text"
            assert result['response_rejected'] == "Response B text"
    
    def test_generate_comparison_preference_b(self):
        """Test that comparison correctly identifies preference B."""
        mock_model = Mock()
        mock_tokenizer = Mock()
        device = torch.device('cpu')
        
        with patch('src.safety.constitutional.preference_comparison.generate_text') as mock_gen:
            mock_gen.return_value = "Response B is superior."
            
            result = generate_comparison(
                prompt="Test prompt",
                response_a="Response A text",
                response_b="Response B text",
                principles=["Be helpful"],
                model=mock_model,
                tokenizer=mock_tokenizer,
                device=device
            )
            
            assert result['preferred'] == 'B'
            assert result['response_chosen'] == "Response B text"
            assert result['response_rejected'] == "Response A text"
    
    def test_generate_comparison_uses_principles(self):
        """Test that comparison includes principles in prompt."""
        mock_model = Mock()
        mock_tokenizer = Mock()
        device = torch.device('cpu')
        
        principles = ["Be accurate", "Be helpful", "Prevent harm"]
        
        with patch('src.safety.constitutional.preference_comparison.generate_text') as mock_gen:
            mock_gen.return_value = "Response A is better."
            
            generate_comparison(
                prompt="Test",
                response_a="A",
                response_b="B",
                principles=principles,
                model=mock_model,
                tokenizer=mock_tokenizer,
                device=device
            )
            
            # Check that generate_text was called with a prompt containing principles
            call_args = mock_gen.call_args[0]
            prompt_used = call_args[2]  # Third argument is the prompt
            
            # Verify principles are in the prompt
            for principle in principles:
                assert principle in prompt_used


class TestGeneratePreferencePairs:
    """Test preference pair generation pipeline."""
    
    def test_generate_preference_pairs_returns_list(self):
        """Test that generate_preference_pairs returns a list."""
        mock_model = Mock()
        mock_tokenizer = Mock()
        device = torch.device('cpu')
        
        # Mock framework with principles
        mock_framework = Mock()
        mock_principle = Mock()
        mock_principle.description = "Be helpful"
        mock_framework.principles = {'helpful': mock_principle}
        
        with patch('src.safety.constitutional.preference_comparison.generate_text') as mock_gen:
            # Mock responses and comparisons
            mock_gen.side_effect = ["Response 1", "Response 2", "Response A is better"]
            
            result = generate_preference_pairs(
                prompts=["What is AI?"],
                model=mock_model,
                tokenizer=mock_tokenizer,
                framework=mock_framework,
                device=device,
                responses_per_prompt=2
            )
            
            assert isinstance(result, list)
    
    def test_generate_preference_pairs_correct_structure(self):
        """Test that preference pairs have correct structure."""
        mock_model = Mock()
        mock_tokenizer = Mock()
        device = torch.device('cpu')
        
        mock_framework = Mock()
        mock_principle = Mock()
        mock_principle.description = "Be helpful"
        mock_framework.principles = {'helpful': mock_principle}
        
        with patch('src.safety.constitutional.preference_comparison.generate_text') as mock_gen:
            mock_gen.side_effect = ["Response 1", "Response 2", "Response A is better"]
            
            result = generate_preference_pairs(
                prompts=["What is AI?"],
                model=mock_model,
                tokenizer=mock_tokenizer,
                framework=mock_framework,
                device=device,
                responses_per_prompt=2
            )
            
            assert len(result) == 1  # One prompt with 2 responses = 1 comparison
            assert 'prompt' in result[0]
            assert 'response_chosen' in result[0]
            assert 'response_rejected' in result[0]
            assert 'comparison_reasoning' in result[0]
    
    def test_generate_preference_pairs_multiple_prompts(self):
        """Test generating preferences for multiple prompts."""
        mock_model = Mock()
        mock_tokenizer = Mock()
        device = torch.device('cpu')
        
        mock_framework = Mock()
        mock_principle = Mock()
        mock_principle.description = "Be helpful"
        mock_framework.principles = {'helpful': mock_principle}
        
        with patch('src.safety.constitutional.preference_comparison.generate_text') as mock_gen:
            # 2 prompts x 2 responses each + 2 comparisons = 6 calls
            mock_gen.side_effect = [
                "R1", "R2", "A is better",  # Prompt 1
                "R3", "R4", "B is better"   # Prompt 2
            ]
            
            result = generate_preference_pairs(
                prompts=["Prompt 1", "Prompt 2"],
                model=mock_model,
                tokenizer=mock_tokenizer,
                framework=mock_framework,
                device=device,
                responses_per_prompt=2
            )
            
            assert len(result) == 2  # One comparison per prompt
    
    def test_generate_preference_pairs_handles_errors(self):
        """Test that errors in comparison don't stop the entire process."""
        mock_model = Mock()
        mock_tokenizer = Mock()
        device = torch.device('cpu')
        
        mock_framework = Mock()
        mock_principle = Mock()
        mock_principle.description = "Be helpful"
        mock_framework.principles = {'helpful': mock_principle}
        
        with patch('src.safety.constitutional.preference_comparison.generate_text') as mock_gen:
            # First prompt succeeds, second fails, third succeeds
            mock_gen.side_effect = [
                "R1", "R2", "A is better",  # Prompt 1 - success
                "R3", "R4", Exception("Model error"),  # Prompt 2 - fails
                "R5", "R6", "B is better"   # Prompt 3 - success
            ]
            
            # Should not raise exception, should continue processing
            result = generate_preference_pairs(
                prompts=["Prompt 1", "Prompt 2", "Prompt 3"],
                model=mock_model,
                tokenizer=mock_tokenizer,
                framework=mock_framework,
                device=device,
                responses_per_prompt=2
            )
            
            # Should have 2 successful comparisons (1 and 3)
            assert len(result) == 2


class TestPreferenceDataset:
    """Test PreferenceDataset class."""
    
    def test_dataset_initialization(self):
        """Test that dataset initializes correctly."""
        mock_tokenizer = Mock()
        data = [
            {
                'prompt': 'Test prompt',
                'response_chosen': 'Good response',
                'response_rejected': 'Bad response',
                'comparison_reasoning': 'Good is better'
            }
        ]
        
        dataset = PreferenceDataset(data, mock_tokenizer, max_length=512)
        
        assert len(dataset) == 1
        assert dataset.max_length == 512
    
    def test_dataset_length(self):
        """Test that __len__ returns correct length."""
        mock_tokenizer = Mock()
        data = [
            {'prompt': 'P1', 'response_chosen': 'C1', 'response_rejected': 'R1', 'comparison_reasoning': 'X'},
            {'prompt': 'P2', 'response_chosen': 'C2', 'response_rejected': 'R2', 'comparison_reasoning': 'Y'},
            {'prompt': 'P3', 'response_chosen': 'C3', 'response_rejected': 'R3', 'comparison_reasoning': 'Z'},
        ]
        
        dataset = PreferenceDataset(data, mock_tokenizer)
        assert len(dataset) == 3
    
    def test_dataset_getitem(self):
        """Test that __getitem__ returns properly formatted data."""
        mock_tokenizer = Mock()
        mock_tokenizer.return_value = {
            'input_ids': torch.tensor([[1, 2, 3]]),
            'attention_mask': torch.tensor([[1, 1, 1]])
        }
        
        data = [
            {
                'prompt': 'What is AI?',
                'response_chosen': 'Artificial Intelligence',
                'response_rejected': 'Computer stuff',
                'comparison_reasoning': 'First is better'
            }
        ]
        
        dataset = PreferenceDataset(data, mock_tokenizer, max_length=512)
        item = dataset[0]
        
        # Check that item has required keys
        assert 'chosen_input_ids' in item
        assert 'chosen_attention_mask' in item
        assert 'rejected_input_ids' in item
        assert 'rejected_attention_mask' in item
        
        # Check that tensors are properly shaped (squeezed)
        assert item['chosen_input_ids'].dim() == 1
        assert item['rejected_input_ids'].dim() == 1
    
    def test_dataset_combines_prompt_and_response(self):
        """Test that dataset combines prompt with responses."""
        mock_tokenizer = Mock()
        mock_tokenizer.return_value = {
            'input_ids': torch.tensor([[1, 2, 3]]),
            'attention_mask': torch.tensor([[1, 1, 1]])
        }
        
        data = [
            {
                'prompt': 'Question: ',
                'response_chosen': 'Good answer',
                'response_rejected': 'Bad answer',
                'comparison_reasoning': 'Reason'
            }
        ]
        
        dataset = PreferenceDataset(data, mock_tokenizer)
        _ = dataset[0]
        
        # Check tokenizer was called with combined text
        calls = mock_tokenizer.call_args_list
        assert len(calls) == 2  # Called for chosen and rejected
        
        # First call should be for chosen (prompt + chosen response)
        first_call_text = calls[0][0][0]
        assert 'Question: ' in first_call_text
        assert 'Good answer' in first_call_text
        
        # Second call should be for rejected (prompt + rejected response)
        second_call_text = calls[1][0][0]
        assert 'Question: ' in second_call_text
        assert 'Bad answer' in second_call_text


class TestComparisonTemplate:
    """Test the comparison template."""
    
    def test_template_has_required_fields(self):
        """Test that template contains all required placeholders."""
        assert '{prompt}' in COMPARISON_TEMPLATE
        assert '{response_a}' in COMPARISON_TEMPLATE
        assert '{response_b}' in COMPARISON_TEMPLATE
        assert '{principles_text}' in COMPARISON_TEMPLATE
    
    def test_template_formatting(self):
        """Test that template can be formatted correctly."""
        formatted = COMPARISON_TEMPLATE.format(
            prompt="What is AI?",
            response_a="AI is intelligent machines",
            response_b="AI is a computer science field",
            principles_text="1. Be accurate\n2. Be helpful"
        )
        
        assert "What is AI?" in formatted
        assert "AI is intelligent machines" in formatted
        assert "AI is a computer science field" in formatted
        assert "1. Be accurate" in formatted
        assert "2. Be helpful" in formatted
    
    def test_template_contains_evaluation_criteria(self):
        """Test that template includes key evaluation criteria."""
        assert 'helpful' in COMPARISON_TEMPLATE.lower()
        assert 'harm' in COMPARISON_TEMPLATE.lower()
        assert 'truthful' in COMPARISON_TEMPLATE.lower()
        assert 'fair' in COMPARISON_TEMPLATE.lower()
        assert 'autonomy' in COMPARISON_TEMPLATE.lower()


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
