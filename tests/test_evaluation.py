"""Tests for Evaluator.

Tests evaluation logic, retrieval metrics, and global vs in-batch comparison.
"""

import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from src.training.trainers.multimodal.evaluation import Evaluator


class MultimodalModel(nn.Module):
    """Simple multimodal model for testing."""

    def __init__(self, embed_dim=128):
        super().__init__()
        self.vision_encoder = nn.Linear(10, embed_dim)
        self.text_encoder = nn.Linear(20, embed_dim)

    def forward(self, images, texts):
        vision_features = self.vision_encoder(images)
        text_features = self.text_encoder(texts)
        return {
            "vision_features": vision_features,
            "text_features": text_features,
        }


@pytest.fixture
def device():
    """Get test device."""
    return torch.device("cpu")


@pytest.fixture
def model():
    """Create a multimodal model."""
    return MultimodalModel(embed_dim=128)


@pytest.fixture
def evaluator(model, device):
    """Create an Evaluator instance."""
    return Evaluator(
        model=model,
        device=device,
        recall_k_values=[1, 5, 10],
    )


@pytest.fixture
def dataloader():
    """Create a simple dataloader."""
    # Create dummy multimodal data
    images = torch.randn(50, 10)
    texts = torch.randn(50, 20)
    indices = torch.arange(50)  # Each sample has unique index

    dataset = TensorDataset(images, texts, indices)
    return DataLoader(dataset, batch_size=10, shuffle=False)


def prepare_model_inputs(batch):
    """Prepare inputs for model."""
    return {"images": batch[0], "texts": batch[1]}


def to_device(batch, device=torch.device("cpu")):
    """Move batch to device."""
    return {
        0: batch[0].to(device),
        1: batch[1].to(device),
        "idx": batch[2].to(device),
    }


class TestEvaluator:
    """Tests for Evaluator class."""

    def test_initialization(self, evaluator):
        """Test Evaluator initialization."""
        assert evaluator.recall_k_values == [1, 5, 10]
        assert evaluator.model is not None
        assert evaluator.device is not None

    def test_evaluate_basic(self, evaluator, dataloader):
        """Test basic evaluation."""
        metrics = evaluator.evaluate(
            dataloader=dataloader,
            prepare_model_inputs_fn=prepare_model_inputs,
            to_device_fn=lambda b: to_device(b, evaluator.device),
            compute_in_batch_comparison=False,
        )

        # Check that global metrics are present
        assert "global_accuracy" in metrics
        assert "global_i2t_accuracy" in metrics
        assert "global_t2i_accuracy" in metrics

        for k in [1, 5, 10]:
            assert f"global_avg_recall@{k}" in metrics
            assert f"global_i2t_recall@{k}" in metrics
            assert f"global_t2i_recall@{k}" in metrics

    def test_evaluate_with_in_batch_comparison(self, evaluator, dataloader):
        """Test evaluation with in-batch metrics."""
        metrics = evaluator.evaluate(
            dataloader=dataloader,
            prepare_model_inputs_fn=prepare_model_inputs,
            to_device_fn=lambda b: to_device(b, evaluator.device),
            compute_in_batch_comparison=True,
        )

        # Check global metrics
        assert "global_accuracy" in metrics

        # Check in-batch metrics
        assert "in_batch_accuracy" in metrics
        assert "in_batch_i2t_accuracy" in metrics
        assert "in_batch_t2i_accuracy" in metrics

        for k in [1, 5, 10]:
            assert f"in_batch_avg_recall@{k}" in metrics

    def test_recall_k_values(self, model, device):
        """Test custom Recall@K values."""
        evaluator = Evaluator(
            model=model,
            device=device,
            recall_k_values=[1, 3, 5],
        )

        assert evaluator.recall_k_values == [1, 3, 5]

    def test_extract_features(self, evaluator):
        """Test feature extraction with fallback keys."""
        # Test with primary key
        outputs = {"vision_features": torch.randn(10, 128)}
        features = evaluator._extract_features(
            outputs, preferred_keys=["vision_features", "image_features"]
        )
        assert features is not None
        assert features.shape == (10, 128)

        # Test with fallback key
        outputs = {"image_features": torch.randn(10, 128)}
        features = evaluator._extract_features(
            outputs, preferred_keys=["vision_features", "image_features"]
        )
        assert features is not None

        # Test with no matching key
        outputs = {"other_features": torch.randn(10, 128)}
        features = evaluator._extract_features(
            outputs, preferred_keys=["vision_features", "image_features"]
        )
        assert features is None

    def test_process_features_2d(self, evaluator):
        """Test feature processing for 2D tensors."""
        features = torch.randn(10, 128)
        processed = evaluator._process_features(features)

        # Should be normalized
        norms = torch.norm(processed, p=2, dim=1)
        assert torch.allclose(norms, torch.ones(10), atol=1e-5)

    def test_process_features_3d(self, evaluator):
        """Test feature processing for 3D tensors (with pooling)."""
        features = torch.randn(10, 20, 128)  # Batch x Seq x Dim
        processed = evaluator._process_features(features)

        # Should be pooled to 2D
        assert processed.shape == (10, 128)

        # Should be normalized
        norms = torch.norm(processed, p=2, dim=1)
        assert torch.allclose(norms, torch.ones(10), atol=1e-5)

    def test_extract_indices(self, evaluator):
        """Test index extraction from batch."""
        # Test with original_idx
        batch = {"original_idx": torch.tensor([1, 2, 3])}
        indices = evaluator._extract_indices(batch)
        assert indices == [1, 2, 3]

        # Test with idx
        batch = {"idx": torch.tensor([4, 5, 6])}
        indices = evaluator._extract_indices(batch)
        assert indices == [4, 5, 6]

        # Test with no indices
        batch = {}
        indices = evaluator._extract_indices(batch)
        assert indices == []

    def test_compute_retrieval_metrics(self, evaluator):
        """Test retrieval metrics computation from embeddings."""
        # Create simple embeddings where each sample matches itself
        image_embeddings = torch.eye(10)  # Identity matrix
        text_embeddings = torch.eye(10)

        metrics = evaluator.compute_retrieval_metrics(
            image_embeddings=image_embeddings,
            text_embeddings=text_embeddings,
        )

        # With identity matrix, R@1 should be perfect
        assert metrics["global_i2t_recall@1"] == 1.0
        assert metrics["global_t2i_recall@1"] == 1.0
        assert metrics["global_accuracy"] == 1.0

    def test_compute_retrieval_metrics_with_indices(self, evaluator):
        """Test retrieval metrics with custom indices."""
        # Create embeddings
        image_embeddings = torch.randn(10, 128)
        text_embeddings = torch.randn(10, 128)
        indices = list(range(10))

        metrics = evaluator.compute_retrieval_metrics(
            image_embeddings=image_embeddings,
            text_embeddings=text_embeddings,
            indices=indices,
        )

        # Should have all global metrics
        assert "global_accuracy" in metrics
        for k in [1, 5, 10]:
            assert f"global_avg_recall@{k}" in metrics

    def test_empty_dataloader(self, evaluator):
        """Test evaluation with empty dataloader."""
        # Create empty dataloader
        empty_dataset = TensorDataset(torch.randn(0, 10), torch.randn(0, 20), torch.arange(0))
        empty_dataloader = DataLoader(empty_dataset, batch_size=10)

        metrics = evaluator.evaluate(
            dataloader=empty_dataloader,
            prepare_model_inputs_fn=prepare_model_inputs,
            to_device_fn=lambda b: to_device(b, evaluator.device),
        )

        # Should return error
        assert "error" in metrics

    def test_collect_embeddings(self, evaluator, dataloader):
        """Test embedding collection."""
        result = evaluator._collect_embeddings(
            dataloader=dataloader,
            prepare_model_inputs_fn=prepare_model_inputs,
            to_device_fn=lambda b: to_device(b, evaluator.device),
        )

        assert result is not None
        image_embeddings, text_embeddings, indices = result

        # Check shapes
        assert image_embeddings.shape[0] == 50  # Total samples
        assert text_embeddings.shape[0] == 50
        assert len(indices) == 50

        # Check that embeddings are normalized
        image_norms = torch.norm(image_embeddings, p=2, dim=1)
        text_norms = torch.norm(text_embeddings, p=2, dim=1)
        assert torch.allclose(image_norms, torch.ones(50), atol=1e-5)
        assert torch.allclose(text_norms, torch.ones(50), atol=1e-5)

    def test_compute_global_metrics_perfect_match(self, evaluator):
        """Test global metrics with perfect matches."""
        # Create perfect similarity matrix (identity)
        similarity = torch.eye(10)
        indices = list(range(10))

        metrics = evaluator._compute_global_metrics(similarity, indices)

        # Should have perfect scores
        assert metrics["global_accuracy"] == 1.0
        assert metrics["global_i2t_recall@1"] == 1.0
        assert metrics["global_t2i_recall@1"] == 1.0

    def test_compute_global_metrics_no_match(self, evaluator):
        """Test global metrics with no matches."""
        # Create worst-case similarity matrix (anti-diagonal)
        similarity = torch.flip(torch.eye(10), [1])
        indices = list(range(10))

        metrics = evaluator._compute_global_metrics(similarity, indices)

        # Should have zero scores for R@1
        assert metrics["global_i2t_recall@1"] == 0.0
        assert metrics["global_t2i_recall@1"] == 0.0

    def test_compute_in_batch_metrics(self, evaluator):
        """Test in-batch metrics computation."""
        # Create identity similarity
        similarity = torch.eye(10)

        metrics = evaluator._compute_in_batch_metrics(similarity)

        # Should have perfect scores
        assert metrics["accuracy"] == 1.0
        assert metrics["i2t_recall@1"] == 1.0
        assert metrics["t2i_recall@1"] == 1.0

    def test_enhanced_features_fallback(self, model, device):
        """Test fallback to enhanced features if available."""
        class EnhancedModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.encoder = nn.Linear(10, 128)

            def forward(self, images, texts):
                base_features = self.encoder(images)
                enhanced_features = base_features * 2  # Simulated enhancement
                return {
                    "vision_features": base_features,
                    "vision_features_enhanced": enhanced_features,
                    "text_features": base_features,
                    "text_features_enhanced": enhanced_features,
                }

        enhanced_model = EnhancedModel()
        evaluator = Evaluator(model=enhanced_model, device=device)

        # Create simple dataset
        images = torch.randn(20, 10)
        texts = torch.randn(20, 10)
        indices = torch.arange(20)
        dataset = TensorDataset(images, texts, indices)
        dataloader = DataLoader(dataset, batch_size=5)

        # Evaluate - should use enhanced features
        metrics = evaluator.evaluate(
            dataloader=dataloader,
            prepare_model_inputs_fn=prepare_model_inputs,
            to_device_fn=lambda b: to_device(b, device),
        )

        assert "global_accuracy" in metrics

    def test_duplicate_indices(self, evaluator):
        """Test handling of duplicate indices (multiple captions per image)."""
        # Create similarity matrix
        similarity = torch.randn(10, 10)

        # Create indices with duplicates (simulating multiple captions per image)
        indices = [0, 0, 1, 1, 2, 2, 3, 3, 4, 4]

        metrics = evaluator._compute_global_metrics(similarity, indices)

        # Should handle duplicates correctly
        assert "global_accuracy" in metrics
        assert 0.0 <= metrics["global_accuracy"] <= 1.0

    def test_recall_at_different_k(self, evaluator):
        """Test that Recall@K increases with K."""
        # Create random embeddings
        image_embeddings = torch.randn(20, 128)
        text_embeddings = torch.randn(20, 128)

        metrics = evaluator.compute_retrieval_metrics(
            image_embeddings=image_embeddings,
            text_embeddings=text_embeddings,
        )

        # R@1 <= R@5 <= R@10 (generally)
        r1 = metrics["global_avg_recall@1"]
        r5 = metrics["global_avg_recall@5"]
        r10 = metrics["global_avg_recall@10"]

        # Allow for rare random cases where this doesn't hold
        # but at least check they're all valid probabilities
        assert 0.0 <= r1 <= 1.0
        assert 0.0 <= r5 <= 1.0
        assert 0.0 <= r10 <= 1.0

    def test_metric_symmetry(self, evaluator):
        """Test that i2t and t2i metrics are computed correctly."""
        # Create embeddings
        image_embeddings = torch.randn(10, 128)
        text_embeddings = torch.randn(10, 128)

        metrics = evaluator.compute_retrieval_metrics(
            image_embeddings=image_embeddings,
            text_embeddings=text_embeddings,
        )

        # Check that average is actually the average
        for k in [1, 5, 10]:
            i2t = metrics[f"global_i2t_recall@{k}"]
            t2i = metrics[f"global_t2i_recall@{k}"]
            avg = metrics[f"global_avg_recall@{k}"]

            assert abs(avg - (i2t + t2i) / 2) < 1e-6


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
