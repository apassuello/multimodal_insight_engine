"""Tests for DataHandler.

Tests data preprocessing, device management, and batch preparation.
"""

import pytest
import torch
import torch.nn as nn

from src.training.trainers.multimodal.data_handler import DataHandler


class SimpleMultimodalModel(nn.Module):
    """Simple multimodal model for testing."""

    def __init__(self, vision_dim=10, text_dim=20, embed_dim=128):
        super().__init__()
        self.vision_model = nn.Linear(vision_dim, embed_dim)
        self.text_model = nn.Linear(text_dim, embed_dim)
        self.fusion_module = nn.Linear(embed_dim * 2, embed_dim)

    def forward(self, images=None, text_data=None):
        outputs = {}
        if images is not None:
            outputs["vision_features"] = self.vision_model(images)
        if text_data is not None:
            if isinstance(text_data, dict):
                # Handle tokenized text
                text_input = text_data.get("input_ids", text_data.get("tokens"))
                outputs["text_features"] = self.text_model(text_input.float())
            else:
                outputs["text_features"] = self.text_model(text_data)
        return outputs


@pytest.fixture
def device():
    """Get test device."""
    return torch.device("cpu")


@pytest.fixture
def model():
    """Create a multimodal model."""
    return SimpleMultimodalModel()


@pytest.fixture
def data_handler(model, device):
    """Create a DataHandler instance."""
    return DataHandler(
        model=model,
        device=device,
        enable_diagnostics=True,
    )


class TestDataHandler:
    """Tests for DataHandler class."""

    def test_initialization(self, data_handler, device):
        """Test DataHandler initialization."""
        assert data_handler.device == device
        assert data_handler.model is not None
        assert data_handler.enable_diagnostics is True
        assert data_handler.current_epoch == 0

    def test_to_device_simple_batch(self, data_handler):
        """Test moving simple batch to device."""
        batch = {
            "images": torch.randn(4, 10),
            "text": torch.randn(4, 20),
        }

        processed = data_handler.to_device(batch)

        assert processed["images"].device == data_handler.device
        assert processed["text_data"].device == data_handler.device  # Renamed from 'text'
        assert "text" not in processed  # Should be renamed

    def test_to_device_nested_dict(self, data_handler):
        """Test moving nested dictionary to device."""
        batch = {
            "images": torch.randn(4, 10),
            "text_data": {
                "input_ids": torch.randint(0, 1000, (4, 20)),
                "attention_mask": torch.ones(4, 20),
            },
        }

        processed = data_handler.to_device(batch)

        assert processed["images"].device == data_handler.device
        assert processed["text_data"]["input_ids"].device == data_handler.device
        assert processed["text_data"]["attention_mask"].device == data_handler.device

    def test_to_device_with_lists(self, data_handler):
        """Test moving batch with lists."""
        batch = {
            "images": torch.randn(4, 10),
            "labels": [0, 1, 2, 3],  # Non-tensor list
            "raw_text": ["hello", "world", "foo", "bar"],  # String list
        }

        processed = data_handler.to_device(batch)

        assert processed["images"].device == data_handler.device
        assert processed["labels"] == [0, 1, 2, 3]  # Unchanged
        assert processed["raw_text"] == ["hello", "world", "foo", "bar"]  # Unchanged

    def test_normalize_batch_keys(self, data_handler):
        """Test batch key normalization."""
        # Test image -> images
        batch = {"image": torch.randn(4, 10)}
        normalized = data_handler._normalize_batch_keys(batch)
        assert "images" in normalized
        assert "image" not in normalized

        # Test text -> text_data
        batch = {"text": torch.randn(4, 20)}
        normalized = data_handler._normalize_batch_keys(batch)
        assert "text_data" in normalized
        assert "text" not in normalized

    def test_prepare_model_inputs(self, data_handler):
        """Test preparing model inputs."""
        batch = {
            "images": torch.randn(4, 10),
            "text_data": torch.randn(4, 20),
            "idx": torch.arange(4),  # Should be excluded
            "raw_text": ["a", "b", "c", "d"],  # Should be excluded
        }

        model_inputs = data_handler.prepare_model_inputs(batch)

        assert "images" in model_inputs
        assert "text_data" in model_inputs
        assert "idx" not in model_inputs
        assert "raw_text" not in model_inputs

    def test_prepare_model_inputs_nested_text(self, data_handler):
        """Test preparing model inputs with nested text data."""
        batch = {
            "images": torch.randn(4, 10),
            "text_data": {
                "input_ids": torch.randint(0, 1000, (4, 20)),
                "attention_mask": torch.ones(4, 20),
            },
        }

        model_inputs = data_handler.prepare_model_inputs(batch)

        assert "images" in model_inputs
        assert "text_data" in model_inputs
        assert isinstance(model_inputs["text_data"], dict)
        assert "input_ids" in model_inputs["text_data"]

    def test_get_pooled_features_2d(self, data_handler):
        """Test pooling already-pooled features."""
        features = torch.randn(4, 128)
        pooled = data_handler.get_pooled_features(features)
        assert pooled.shape == (4, 128)
        assert torch.equal(pooled, features)

    def test_get_pooled_features_3d(self, data_handler):
        """Test pooling sequence features."""
        features = torch.randn(4, 10, 128)  # Batch x Seq x Dim
        pooled = data_handler.get_pooled_features(features)
        assert pooled.shape == (4, 128)

        # Verify mean pooling
        expected = features.mean(dim=1)
        assert torch.allclose(pooled, expected)

    def test_get_pooled_features_invalid_shape(self, data_handler):
        """Test pooling with invalid shape."""
        features = torch.randn(4, 10, 20, 128)  # 4D tensor
        with pytest.raises(ValueError):
            data_handler.get_pooled_features(features)

    def test_prepare_loss_inputs_base_features(self, data_handler, model):
        """Test preparing loss inputs with base features."""
        batch = {
            "images": torch.randn(4, 10),
            "text_data": torch.randn(4, 20),
        }

        outputs = model(images=batch["images"], text_data=batch["text_data"])
        loss_inputs = data_handler.prepare_loss_inputs(batch, outputs)

        assert "vision_features" in loss_inputs
        assert "text_features" in loss_inputs
        assert loss_inputs["vision_features"].shape[0] == 4
        assert loss_inputs["text_features"].shape[0] == 4

        # Check normalization
        vision_norms = torch.norm(loss_inputs["vision_features"], dim=1)
        text_norms = torch.norm(loss_inputs["text_features"], dim=1)
        assert torch.allclose(vision_norms, torch.ones(4), atol=1e-5)
        assert torch.allclose(text_norms, torch.ones(4), atol=1e-5)

    def test_prepare_loss_inputs_enhanced_features(self, data_handler):
        """Test loss inputs with enhanced features (priority)."""
        class EnhancedModel(nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, images=None, text_data=None):
                return {
                    "vision_features": torch.randn(4, 128),
                    "text_features": torch.randn(4, 128),
                    "vision_features_enhanced": torch.randn(4, 128) * 2,
                    "text_features_enhanced": torch.randn(4, 128) * 2,
                }

        handler = DataHandler(model=EnhancedModel(), device=data_handler.device)
        batch = {"images": torch.randn(4, 10)}
        outputs = {"vision_features_enhanced": torch.randn(4, 128),
                   "text_features_enhanced": torch.randn(4, 128)}

        loss_inputs = handler.prepare_loss_inputs(batch, outputs)

        # Should use enhanced features
        assert "vision_features" in loss_inputs
        assert "text_features" in loss_inputs

    def test_prepare_loss_inputs_fallback(self, data_handler):
        """Test loss inputs with fallback when features not found."""
        batch = {"images": torch.randn(4, 10)}
        outputs = {"logits": torch.randn(4, 10)}  # No features

        loss_inputs = data_handler.prepare_loss_inputs(batch, outputs)

        # Should create emergency fallback
        assert "vision_features" in loss_inputs
        assert "text_features" in loss_inputs
        assert loss_inputs["vision_features"].shape == (4, 768)

    def test_extract_features_priority(self, data_handler):
        """Test feature extraction priority order."""
        # Test enhanced features (highest priority)
        outputs = {
            "vision_features": torch.randn(4, 128),
            "text_features": torch.randn(4, 128),
            "vision_features_enhanced": torch.randn(4, 128),
            "text_features_enhanced": torch.randn(4, 128),
        }

        vision, text, source = data_handler._extract_features(outputs)
        assert source == "enhanced_features"

        # Test base features (second priority)
        outputs = {
            "vision_features": torch.randn(4, 128),
            "text_features": torch.randn(4, 128),
        }

        vision, text, source = data_handler._extract_features(outputs)
        assert source == "base_features"

        # Test image_features (third priority)
        outputs = {
            "image_features": torch.randn(4, 128),
            "text_features": torch.randn(4, 128),
        }

        vision, text, source = data_handler._extract_features(outputs)
        assert source == "image_text_features"

    def test_extract_features_not_found(self, data_handler):
        """Test feature extraction when features don't exist."""
        outputs = {"logits": torch.randn(4, 10)}

        vision, text, source = data_handler._extract_features(outputs)
        assert vision is None
        assert text is None
        assert source == "not_found"

    def test_ensure_model_on_device(self, model, device):
        """Test ensuring model is on correct device."""
        handler = DataHandler(model=model, device=device)
        handler.ensure_model_on_device()

        # Check all parameters are on correct device
        for param in model.parameters():
            assert param.device == device

    def test_compute_feature_stats(self, data_handler):
        """Test feature statistics computation."""
        features = torch.randn(10, 128)
        stats = data_handler._compute_feature_stats(features)

        assert "mean" in stats
        assert "std" in stats
        assert "min" in stats
        assert "max" in stats

        # Verify values are reasonable
        assert stats["min"] < stats["mean"] < stats["max"]
        assert stats["std"] > 0

    def test_verify_normalization(self, data_handler):
        """Test normalization verification."""
        # Create normalized features
        features = torch.randn(10, 128)
        vision_features = torch.nn.functional.normalize(features, p=2, dim=1)
        text_features = torch.nn.functional.normalize(features, p=2, dim=1)

        # Should not raise warnings (but we can't easily test logging)
        data_handler._verify_normalization(vision_features, text_features)

    def test_diagnose_features(self, data_handler):
        """Test feature diagnostics."""
        vision_features = torch.randn(10, 128)
        text_features = torch.randn(10, 128)

        # Should not raise errors
        data_handler._diagnose_features(vision_features, text_features, "test_source")

    def test_diagnose_features_with_nan(self, data_handler, capsys):
        """Test diagnostics with NaN values."""
        vision_features = torch.randn(10, 128)
        vision_features[0, 0] = float('nan')
        text_features = torch.randn(10, 128)

        data_handler._diagnose_features(vision_features, text_features, "test_source")

        # Check that error was logged (captured in stderr/stdout by logger)
        # We can't easily assert on log output in tests, but the method should not crash

    def test_set_current_epoch(self, data_handler):
        """Test setting current epoch."""
        data_handler.set_current_epoch(5)
        assert data_handler.current_epoch == 5

    def test_device_consistency_check(self, data_handler):
        """Test device consistency checking."""
        # Create batch on wrong device initially
        batch = {"images": torch.randn(4, 10)}  # CPU

        # Move to device
        processed = data_handler.to_device(batch)

        # Should be on correct device
        assert processed["images"].device == data_handler.device

    def test_process_item_recursive_complex(self, data_handler):
        """Test recursive processing of complex nested structures."""
        complex_item = {
            "level1": {
                "tensor": torch.randn(2, 3),
                "list": [torch.randn(2), torch.randn(3)],
                "nested": {
                    "deep_tensor": torch.randn(4, 5),
                },
            },
            "simple": torch.randn(6),
        }

        processed = data_handler._process_item_recursive(complex_item)

        # Check all tensors were processed
        assert processed["level1"]["tensor"].device == data_handler.device
        assert processed["level1"]["list"][0].device == data_handler.device
        assert processed["level1"]["nested"]["deep_tensor"].device == data_handler.device
        assert processed["simple"].device == data_handler.device

    def test_disable_diagnostics(self, model, device):
        """Test with diagnostics disabled."""
        handler = DataHandler(model=model, device=device, enable_diagnostics=False)

        batch = {"images": torch.randn(4, 10)}
        outputs = model(images=batch["images"])

        # Should work without diagnostics
        loss_inputs = handler.prepare_loss_inputs(batch, outputs)
        assert "vision_features" in loss_inputs

    def test_prepare_text_data_dict(self, data_handler):
        """Test preparing text data dictionary."""
        text_data = {
            "input_ids": torch.randint(0, 1000, (4, 20)),
            "attention_mask": torch.ones(4, 20),
        }

        prepared = data_handler._prepare_text_data(text_data)

        assert isinstance(prepared, dict)
        assert "input_ids" in prepared
        assert "attention_mask" in prepared
        assert prepared["input_ids"].device == data_handler.device

    def test_prepare_text_data_tensor(self, data_handler):
        """Test preparing text data as tensor."""
        text_data = torch.randn(4, 20)
        prepared = data_handler._prepare_text_data(text_data)
        assert torch.equal(prepared, text_data)

    def test_batch_without_images(self, data_handler, model):
        """Test handling batch without images."""
        batch = {"text_data": torch.randn(4, 20)}
        outputs = model(text_data=batch["text_data"])

        model_inputs = data_handler.prepare_model_inputs(batch)
        assert "images" not in model_inputs
        assert "text_data" in model_inputs

    def test_batch_without_text(self, data_handler, model):
        """Test handling batch without text."""
        batch = {"images": torch.randn(4, 10)}
        outputs = model(images=batch["images"])

        model_inputs = data_handler.prepare_model_inputs(batch)
        assert "images" in model_inputs
        assert "text_data" not in model_inputs


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
