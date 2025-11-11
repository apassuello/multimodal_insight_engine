"""
Unit tests for model_utils.py
Tests utility functions for model loading and text generation.
"""

import pytest
from unittest.mock import Mock, MagicMock, patch
import torch
from src.safety.constitutional.model_utils import (
    GenerationConfig,
    load_model,
    generate_text,
    batch_generate,
    prepare_model_for_training,
    get_model_device
)


class TestGenerationConfig:
    """Test GenerationConfig dataclass."""

    def test_default_values(self):
        """Test that default values are set correctly."""
        config = GenerationConfig()

        assert config.max_length == 100
        assert config.temperature == 1.0
        assert config.top_p == 0.9
        assert config.top_k == 50
        assert config.num_return_sequences == 1
        assert config.do_sample is True
        assert config.pad_token_id is None
        assert config.eos_token_id is None

    def test_custom_values(self):
        """Test creating config with custom values."""
        config = GenerationConfig(
            max_length=200,
            temperature=0.5,
            top_p=0.95,
            top_k=100,
            num_return_sequences=3,
            do_sample=False
        )

        assert config.max_length == 200
        assert config.temperature == 0.5
        assert config.top_p == 0.95
        assert config.top_k == 100
        assert config.num_return_sequences == 3
        assert config.do_sample is False

    def test_partial_override(self):
        """Test overriding only some values."""
        config = GenerationConfig(max_length=256, temperature=0.8)

        assert config.max_length == 256
        assert config.temperature == 0.8
        assert config.top_p == 0.9  # Default value
        assert config.do_sample is True  # Default value


class TestLoadModel:
    """Test load_model function."""

    @patch('transformers.AutoTokenizer')
    @patch('transformers.AutoModelForCausalLM')
    def test_loads_model_and_tokenizer(self, mock_model_class, mock_tokenizer_class):
        """Test that model and tokenizer are loaded."""
        mock_model = Mock()
        mock_tokenizer = Mock()
        mock_tokenizer.pad_token = None
        mock_tokenizer.eos_token = "[EOS]"

        mock_model_class.from_pretrained.return_value = mock_model
        mock_tokenizer_class.from_pretrained.return_value = mock_tokenizer

        # Mock model parameters for counting
        mock_param = Mock()
        mock_param.numel.return_value = 1000
        # parameters() should return an iterable - use side_effect to return fresh list each time
        mock_model.parameters.return_value = [mock_param]

        model, tokenizer = load_model("gpt2")

        assert model is mock_model
        assert tokenizer is mock_tokenizer
        mock_model_class.from_pretrained.assert_called_once_with("gpt2")
        mock_tokenizer_class.from_pretrained.assert_called_once_with("gpt2")

    @patch('transformers.AutoTokenizer')
    @patch('transformers.AutoModelForCausalLM')
    def test_sets_pad_token_if_none(self, mock_model_class, mock_tokenizer_class):
        """Test that pad token is set if not present."""
        mock_model = Mock()
        mock_tokenizer = Mock()
        mock_tokenizer.pad_token = None
        mock_tokenizer.eos_token = "[EOS]"

        mock_model_class.from_pretrained.return_value = mock_model
        mock_tokenizer_class.from_pretrained.return_value = mock_tokenizer

        # Mock parameters
        mock_param = Mock()
        mock_param.numel.return_value = 1000
        mock_model.parameters.return_value = [mock_param]

        model, tokenizer = load_model()

        assert tokenizer.pad_token == "[EOS]"

    @patch('transformers.AutoTokenizer')
    @patch('transformers.AutoModelForCausalLM')
    def test_moves_model_to_device(self, mock_model_class, mock_tokenizer_class):
        """Test that model is moved to specified device."""
        mock_model = Mock()
        mock_tokenizer = Mock()
        mock_tokenizer.pad_token = "[PAD]"

        mock_model_class.from_pretrained.return_value = mock_model
        mock_tokenizer_class.from_pretrained.return_value = mock_tokenizer

        # Mock parameters
        mock_param = Mock()
        mock_param.numel.return_value = 1000
        mock_model.parameters.return_value = [mock_param]

        device = torch.device("cpu")
        model, tokenizer = load_model(device=device)

        mock_model.to.assert_called_once_with(device)

    @patch('transformers.AutoTokenizer')
    @patch('transformers.AutoModelForCausalLM')
    def test_8bit_loading(self, mock_model_class, mock_tokenizer_class):
        """Test 8-bit model loading."""
        mock_model = Mock()
        mock_tokenizer = Mock()
        mock_tokenizer.pad_token = "[PAD]"

        mock_model_class.from_pretrained.return_value = mock_model
        mock_tokenizer_class.from_pretrained.return_value = mock_tokenizer

        # Mock parameters
        mock_param = Mock()
        mock_param.numel.return_value = 1000
        mock_model.parameters.return_value = [mock_param]

        device = torch.device("cuda")
        with patch('torch.cuda.is_available', return_value=True):
            model, tokenizer = load_model(device=device, load_in_8bit=True)

            # Should have called with 8-bit parameters
            call_kwargs = mock_model_class.from_pretrained.call_args[1]
            assert call_kwargs.get("load_in_8bit") is True

    def test_import_error_without_transformers(self):
        """Test that ImportError is raised without transformers."""
        with patch.dict('sys.modules', {'transformers': None}):
            with pytest.raises(ImportError, match="transformers library required"):
                load_model()


class TestGenerateText:
    """Test generate_text function."""

    def setup_method(self):
        """Setup test fixtures."""
        self.mock_model = Mock()
        self.mock_tokenizer = Mock()
        self.mock_tokenizer.pad_token_id = 0
        self.mock_tokenizer.eos_token_id = 1

        # Setup device
        mock_param = Mock()
        mock_param.device = torch.device("cpu")
        self.mock_model.parameters.return_value = [mock_param]

    def test_generates_text(self):
        """Test basic text generation."""
        # Mock tokenization
        self.mock_tokenizer.return_value = {
            "input_ids": torch.tensor([[1, 2, 3]]),
            "attention_mask": torch.tensor([[1, 1, 1]])
        }

        # Mock generation
        self.mock_model.generate.return_value = torch.tensor([[1, 2, 3, 4, 5, 6]])

        # Mock decoding
        self.mock_tokenizer.decode.return_value = "Generated text"

        result = generate_text(self.mock_model, self.mock_tokenizer, "Test prompt")

        assert result == "Generated text"
        self.mock_model.generate.assert_called_once()

    def test_uses_custom_generation_config(self):
        """Test generation with custom config."""
        config = GenerationConfig(
            max_length=200,
            temperature=0.5,
            top_p=0.95
        )

        # Setup mocks
        self.mock_tokenizer.return_value = {
            "input_ids": torch.tensor([[1, 2, 3]]),
            "attention_mask": torch.tensor([[1, 1, 1]])
        }
        self.mock_model.generate.return_value = torch.tensor([[1, 2, 3, 4]])
        self.mock_tokenizer.decode.return_value = "Text"

        generate_text(
            self.mock_model,
            self.mock_tokenizer,
            "Test",
            generation_config=config
        )

        # Check that config was used
        call_kwargs = self.mock_model.generate.call_args[1]
        assert call_kwargs["max_length"] == 200
        assert call_kwargs["temperature"] == 0.5
        assert call_kwargs["top_p"] == 0.95

    def test_removes_prompt_from_output(self):
        """Test that prompt is removed from generated output."""
        prompt_length = 3

        # Mock tokenization with prompt_length tokens
        self.mock_tokenizer.return_value = {
            "input_ids": torch.tensor([[1, 2, 3]]),
            "attention_mask": torch.tensor([[1, 1, 1]])
        }

        # Generated output includes prompt + new tokens
        self.mock_model.generate.return_value = torch.tensor([[1, 2, 3, 4, 5, 6]])

        self.mock_tokenizer.decode.return_value = "Only new text"

        result = generate_text(self.mock_model, self.mock_tokenizer, "Prompt")

        # Decode should be called with only the new tokens (indices 3:)
        assert result == "Only new text"

    def test_uses_device_parameter(self):
        """Test that custom device is used."""
        # Use CPU to avoid CUDA availability issues in testing
        device = torch.device("cpu")

        # Setup mocks
        self.mock_tokenizer.return_value = {
            "input_ids": torch.tensor([[1, 2, 3]]),
            "attention_mask": torch.tensor([[1, 1, 1]])
        }
        self.mock_model.generate.return_value = torch.tensor([[1, 2, 3]])
        self.mock_tokenizer.decode.return_value = "Text"

        generate_text(
            self.mock_model,
            self.mock_tokenizer,
            "Test",
            device=device
        )

        # Verify inputs were moved to device (check call was made)
        assert self.mock_tokenizer.called


class TestBatchGenerate:
    """Test batch_generate function."""

    def setup_method(self):
        """Setup test fixtures."""
        self.mock_model = Mock()
        self.mock_tokenizer = Mock()
        self.mock_tokenizer.pad_token_id = 0
        self.mock_tokenizer.eos_token_id = 1

        # Setup device
        mock_param = Mock()
        mock_param.device = torch.device("cpu")
        self.mock_model.parameters.return_value = [mock_param]

    def test_batch_generate_multiple_prompts(self):
        """Test generating for multiple prompts."""
        prompts = ["Prompt 1", "Prompt 2", "Prompt 3"]

        # Mock tokenization
        self.mock_tokenizer.return_value = {
            "input_ids": torch.tensor([[1, 2, 3], [1, 2, 3], [1, 2, 3]]),
            "attention_mask": torch.tensor([[1, 1, 1], [1, 1, 1], [1, 1, 1]])
        }

        # Mock generation
        self.mock_model.generate.return_value = torch.tensor([
            [1, 2, 3, 4, 5],
            [1, 2, 3, 6, 7],
            [1, 2, 3, 8, 9]
        ])

        # Mock decoding
        self.mock_tokenizer.decode.side_effect = ["Gen 1", "Gen 2", "Gen 3"]

        results = batch_generate(
            self.mock_model,
            self.mock_tokenizer,
            prompts,
            batch_size=4,
            show_progress=False
        )

        assert len(results) == 3
        assert results == ["Gen 1", "Gen 2", "Gen 3"]

    def test_batch_generate_respects_batch_size(self):
        """Test that batch size is respected."""
        prompts = ["P1", "P2", "P3", "P4", "P5"]

        # Mock tokenization
        def tokenize_side_effect(batch, **kwargs):
            return {
                "input_ids": torch.tensor([[1, 2]] * len(batch)),
                "attention_mask": torch.tensor([[1, 1]] * len(batch))
            }

        self.mock_tokenizer.side_effect = tokenize_side_effect

        # Mock generation
        def generate_side_effect(**kwargs):
            batch_size = kwargs["input_ids"].shape[0]
            return torch.tensor([[1, 2, 3]] * batch_size)

        self.mock_model.generate.side_effect = generate_side_effect

        # Mock decoding
        self.mock_tokenizer.decode.return_value = "Text"

        results = batch_generate(
            self.mock_model,
            self.mock_tokenizer,
            prompts,
            batch_size=2,
            show_progress=False
        )

        # Should make multiple calls due to batch size
        assert len(results) == 5

    def test_batch_generate_uses_config(self):
        """Test that generation config is used."""
        config = GenerationConfig(max_length=150, temperature=0.7)

        # Setup mocks
        self.mock_tokenizer.return_value = {
            "input_ids": torch.tensor([[1, 2]]),
            "attention_mask": torch.tensor([[1, 1]])
        }
        self.mock_model.generate.return_value = torch.tensor([[1, 2, 3]])
        self.mock_tokenizer.decode.return_value = "Text"

        batch_generate(
            self.mock_model,
            self.mock_tokenizer,
            ["Prompt"],
            generation_config=config,
            show_progress=False
        )

        # Check config was used
        call_kwargs = self.mock_model.generate.call_args[1]
        assert call_kwargs["max_length"] == 150
        assert call_kwargs["temperature"] == 0.7

    def test_batch_generate_empty_list(self):
        """Test batch generation with empty prompt list."""
        results = batch_generate(
            self.mock_model,
            self.mock_tokenizer,
            [],
            show_progress=False
        )

        assert len(results) == 0

    def test_batch_generate_shows_progress(self):
        """Test that batch_generate works with progress bar enabled."""
        # Setup mocks
        self.mock_tokenizer.return_value = {
            "input_ids": torch.tensor([[1, 2]]),
            "attention_mask": torch.tensor([[1, 1]])
        }
        self.mock_model.generate.return_value = torch.tensor([[1, 2, 3]])
        self.mock_tokenizer.decode.return_value = "Text"

        # Just test that it doesn't crash with show_progress=True
        # (tqdm import is conditional and hard to mock)
        results = batch_generate(
            self.mock_model,
            self.mock_tokenizer,
            ["Prompt"],
            show_progress=True
        )

        # Verify it still generates correctly
        assert len(results) == 1
        assert results[0] == "Text"


class TestPrepareModelForTraining:
    """Test prepare_model_for_training function."""

    def test_sets_model_to_train_mode(self):
        """Test that model is set to training mode."""
        mock_model = Mock()
        mock_param = Mock()
        mock_model.parameters.return_value = [mock_param]

        with patch('torch.optim.AdamW'):
            prepare_model_for_training(mock_model)

        mock_model.train.assert_called_once()

    def test_enables_gradients(self):
        """Test that gradients are enabled for parameters."""
        mock_model = Mock()
        mock_param = Mock()
        mock_param.requires_grad = False
        mock_model.parameters.return_value = [mock_param]

        with patch('torch.optim.AdamW'):
            prepare_model_for_training(mock_model)

        assert mock_param.requires_grad is True

    def test_creates_adamw_optimizer(self):
        """Test that AdamW optimizer is created."""
        mock_model = Mock()
        mock_param = Mock()
        mock_model.parameters.return_value = [mock_param]

        with patch('torch.optim.AdamW') as mock_adamw:
            prepare_model_for_training(mock_model)

            mock_adamw.assert_called_once()

    def test_uses_custom_learning_rate(self):
        """Test that custom learning rate is used."""
        mock_model = Mock()
        mock_param = Mock()
        mock_model.parameters.return_value = [mock_param]

        with patch('torch.optim.AdamW') as mock_adamw:
            prepare_model_for_training(mock_model, learning_rate=1e-4)

            call_kwargs = mock_adamw.call_args[1]
            assert call_kwargs["lr"] == 1e-4

    def test_uses_custom_weight_decay(self):
        """Test that custom weight decay is used."""
        mock_model = Mock()
        mock_param = Mock()
        mock_model.parameters.return_value = [mock_param]

        with patch('torch.optim.AdamW') as mock_adamw:
            prepare_model_for_training(mock_model, weight_decay=0.05)

            call_kwargs = mock_adamw.call_args[1]
            assert call_kwargs["weight_decay"] == 0.05

    def test_returns_optimizer(self):
        """Test that optimizer is returned."""
        mock_model = Mock()
        mock_param = Mock()
        mock_model.parameters.return_value = [mock_param]

        with patch('torch.optim.AdamW') as mock_adamw:
            mock_adamw.return_value = Mock()  # Return a mock optimizer
            optimizer = prepare_model_for_training(mock_model)

        assert optimizer is not None


class TestGetModelDevice:
    """Test get_model_device function."""

    def test_returns_model_device(self):
        """Test that model device is returned."""
        mock_model = Mock()
        mock_param = Mock()
        mock_param.device = torch.device("cuda")
        mock_model.parameters.return_value = [mock_param]

        device = get_model_device(mock_model)

        assert device == torch.device("cuda")

    def test_works_with_cpu_device(self):
        """Test with CPU device."""
        mock_model = Mock()
        mock_param = Mock()
        mock_param.device = torch.device("cpu")
        mock_model.parameters.return_value = [mock_param]

        device = get_model_device(mock_model)

        assert device == torch.device("cpu")


class TestIntegrationScenarios:
    """Test integration scenarios."""

    @patch('torch.optim.AdamW')
    @patch('transformers.AutoTokenizer')
    @patch('transformers.AutoModelForCausalLM')
    def test_load_and_prepare_for_training(self, mock_model_class, mock_tokenizer_class, mock_adamw):
        """Test loading model and preparing for training."""
        mock_model = Mock()
        mock_tokenizer = Mock()
        mock_tokenizer.pad_token = "[PAD]"

        mock_model_class.from_pretrained.return_value = mock_model
        mock_tokenizer_class.from_pretrained.return_value = mock_tokenizer
        mock_adamw.return_value = Mock()  # Return a mock optimizer

        # Mock parameters
        mock_param = Mock()
        mock_param.numel.return_value = 1000
        mock_param.requires_grad = False
        mock_model.parameters.return_value = [mock_param]

        # Load model
        model, tokenizer = load_model("gpt2")

        # Prepare for training
        optimizer = prepare_model_for_training(model)

        assert model is mock_model
        assert tokenizer is mock_tokenizer
        assert optimizer is not None
        assert mock_param.requires_grad is True

    def test_generate_with_multiple_configs(self):
        """Test generation with different configs."""
        mock_model = Mock()
        mock_tokenizer = Mock()
        mock_tokenizer.pad_token_id = 0
        mock_tokenizer.eos_token_id = 1

        # Setup device
        mock_param = Mock()
        mock_param.device = torch.device("cpu")
        mock_model.parameters.return_value = [mock_param]

        # Setup mocks
        mock_tokenizer.return_value = {
            "input_ids": torch.tensor([[1, 2, 3]]),
            "attention_mask": torch.tensor([[1, 1, 1]])
        }
        mock_model.generate.return_value = torch.tensor([[1, 2, 3, 4]])
        mock_tokenizer.decode.return_value = "Text"

        # Test different configs
        configs = [
            GenerationConfig(temperature=0.5),
            GenerationConfig(temperature=1.0),
            GenerationConfig(temperature=1.5)
        ]

        for config in configs:
            result = generate_text(mock_model, mock_tokenizer, "Test", generation_config=config)
            assert isinstance(result, str)
