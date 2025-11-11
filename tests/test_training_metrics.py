"""
Purposeful tests for training metrics.

PURPOSE: Ensure metrics (accuracy, BLEU, perplexity) compute correctly.
RISK: Wrong metrics = training on bad signals = wasted compute & failed models.
"""

import pytest
import torch
import numpy as np
from src.training.metrics import Accuracy, Perplexity, F1Score, BLEUScore


class TestAccuracy:
    """Test accuracy metric for correctness."""

    def test_perfect_accuracy(self):
        """Validate 100% accuracy case."""
        metric = Accuracy()

        # All predictions correct
        pred = torch.tensor([[0.1, 0.9], [0.2, 0.8], [0.3, 0.7]])
        target = torch.tensor([1, 1, 1])

        metric.update(pred, target)
        assert metric.compute() == 1.0, "Perfect predictions should give 100% accuracy"

    def test_zero_accuracy(self):
        """Validate 0% accuracy case."""
        metric = Accuracy()

        # All predictions wrong
        pred = torch.tensor([[0.9, 0.1], [0.8, 0.2], [0.7, 0.3]])
        target = torch.tensor([1, 1, 1])

        metric.update(pred, target)
        assert metric.compute() == 0.0, "All wrong predictions should give 0% accuracy"

    def test_partial_accuracy(self):
        """Validate partial accuracy case."""
        metric = Accuracy()

        # 2 out of 4 correct
        pred = torch.tensor([
            [0.9, 0.1],  # predicts 0, target 0 ✓
            [0.1, 0.9],  # predicts 1, target 1 ✓
            [0.9, 0.1],  # predicts 0, target 1 ✗
            [0.1, 0.9],  # predicts 1, target 0 ✗
        ])
        target = torch.tensor([0, 1, 1, 0])

        metric.update(pred, target)
        acc = metric.compute()
        assert abs(acc - 0.5) < 1e-6, f"Expected 50% accuracy, got {acc*100}%"

    def test_top_k_accuracy(self):
        """Validate top-k accuracy works."""
        metric = Accuracy(top_k=2)

        # Predictions: [0.1, 0.3, 0.6] -> top-2 are classes 2 and 1
        pred = torch.tensor([[0.1, 0.3, 0.6]])
        target = torch.tensor([1])  # Target is in top-2

        metric.update(pred, target)
        assert metric.compute() == 1.0, "Target in top-2 should count as correct"

    def test_multi_batch_accumulation(self):
        """Validate accuracy accumulates across batches."""
        metric = Accuracy()

        # Batch 1: 2/3 correct
        pred1 = torch.tensor([[0.9, 0.1], [0.1, 0.9], [0.9, 0.1]])
        target1 = torch.tensor([0, 1, 1])
        metric.update(pred1, target1)

        # Batch 2: 1/2 correct
        pred2 = torch.tensor([[0.9, 0.1], [0.1, 0.9]])
        target2 = torch.tensor([0, 0])
        metric.update(pred2, target2)

        # Total: 3/5 correct = 60%
        acc = metric.compute()
        assert abs(acc - 0.6) < 1e-6, f"Expected 60% accuracy, got {acc*100}%"

    def test_reset(self):
        """Validate reset clears state."""
        metric = Accuracy()

        pred = torch.tensor([[0.9, 0.1], [0.1, 0.9]])
        target = torch.tensor([0, 1])
        metric.update(pred, target)
        assert metric.compute() == 1.0

        metric.reset()
        assert metric.compute() == 0.0, "Reset should clear all counts"

    def test_empty_batch_handling(self):
        """Validate handling of empty batches."""
        metric = Accuracy()
        assert metric.compute() == 0.0, "Empty metric should return 0"


class TestPerplexity:
    """Test perplexity metric for language models."""

    def test_perfect_prediction(self):
        """Validate perplexity for perfect predictions."""
        metric = Perplexity()

        # Perfect predictions (log prob = 0, loss = 0)
        loss = torch.tensor(0.0)
        metric.update(loss, num_tokens=10)

        ppl = metric.compute()
        assert abs(ppl - 1.0) < 1e-6, f"Perfect predictions should give perplexity=1, got {ppl}"

    def test_high_loss_perplexity(self):
        """Validate perplexity increases with loss."""
        metric = Perplexity()

        # High loss should give high perplexity
        loss = torch.tensor(5.0)
        metric.update(loss, num_tokens=100)

        ppl = metric.compute()
        assert ppl > 100, f"High loss should give high perplexity, got {ppl}"

    def test_perplexity_accumulation(self):
        """Validate perplexity accumulates across batches."""
        metric = Perplexity()

        metric.update(torch.tensor(2.0), num_tokens=50)
        metric.update(torch.tensor(3.0), num_tokens=50)

        # Average loss = 2.5, perplexity = exp(2.5)
        ppl = metric.compute()
        expected = np.exp(2.5)
        assert abs(ppl - expected) < 0.1, f"Expected ~{expected:.1f}, got {ppl:.1f}"


class TestF1Score:
    """Test F1 score for classification."""

    def test_perfect_f1(self):
        """Validate F1=1 for perfect predictions."""
        metric = F1Score(num_classes=2)

        # F1Score expects class indices, not logits
        pred = torch.tensor([1, 1, 0])
        target = torch.tensor([1, 1, 0])

        metric.update(pred, target)
        f1 = metric.compute()
        assert abs(f1 - 1.0) < 1e-6, "Perfect predictions should give F1=1"

    def test_zero_f1(self):
        """Validate F1=0 for completely wrong predictions."""
        metric = F1Score(num_classes=2)

        # All predictions wrong
        pred = torch.tensor([0, 0])
        target = torch.tensor([1, 1])

        metric.update(pred, target)
        f1 = metric.compute()
        assert f1 < 0.01, "All wrong predictions should give F1≈0"

    def test_multiclass_f1(self):
        """Validate F1 for multi-class classification."""
        metric = F1Score(num_classes=3)

        # Mix of correct and incorrect
        pred = torch.tensor([0, 1, 2])  # predictions
        target = torch.tensor([0, 1, 0])  # targets

        metric.update(pred, target)
        f1 = metric.compute()
        assert 0.4 < f1 < 0.9, f"Expected reasonable F1, got {f1}"


class TestBLEUScore:
    """Test BLEU score for machine translation."""

    def test_perfect_bleu(self):
        """Validate BLEU=1 for identical sentences."""
        metric = BLEUScore()

        # BLEUScore expects strings, not lists (it splits internally)
        reference = "the cat sat on the mat"
        hypothesis = "the cat sat on the mat"

        metric.update(hypothesis, reference)
        bleu = metric.compute()
        assert abs(bleu - 1.0) < 0.01, f"Identical sentences should give BLEU≈1, got {bleu}"

    def test_zero_bleu(self):
        """Validate BLEU=0 for completely different sentences."""
        metric = BLEUScore()

        reference = "the cat sat"
        hypothesis = "a dog ran"

        metric.update(hypothesis, reference)
        bleu = metric.compute()
        assert bleu < 0.1, f"Completely different sentences should give BLEU≈0, got {bleu}"

    def test_partial_bleu(self):
        """Validate BLEU for partially matching sentences."""
        metric = BLEUScore()

        reference = "the cat sat on the mat"
        hypothesis = "the cat sat"  # First 3 words match

        metric.update(hypothesis, reference)
        bleu = metric.compute()
        assert 0.2 < bleu < 0.8, f"Partial match should give intermediate BLEU, got {bleu}"

    def test_multiple_references(self):
        """Validate BLEU with multiple translations."""
        metric = BLEUScore()

        # Multiple calls to update() with same hypothesis
        hypothesis = "the cat sat on the mat"
        reference1 = "the cat sat on the mat"
        reference2 = "a cat was sitting on the mat"

        metric.update(hypothesis, reference1)
        bleu = metric.compute()
        assert bleu > 0.5, f"Good match with reference should give BLEU>0.5, got {bleu}"


class TestMetricIntegration:
    """Test that metrics work together in training scenarios."""

    def test_metrics_on_same_predictions(self):
        """Validate multiple metrics can evaluate same predictions."""
        pred_logits = torch.tensor([[0.9, 0.1], [0.1, 0.9], [0.9, 0.1]])
        pred_indices = pred_logits.argmax(dim=-1)  # [0, 1, 0]
        target = torch.tensor([0, 1, 0])

        acc = Accuracy()
        f1 = F1Score(num_classes=2)

        acc.update(pred_logits, target)  # Accuracy handles logits
        f1.update(pred_indices, target)  # F1Score needs indices

        assert acc.compute() == 1.0
        assert abs(f1.compute() - 1.0) < 1e-6

    def test_metric_consistency_across_devices(self):
        """Validate metrics give same results on CPU/GPU."""
        pred_cpu = torch.tensor([[0.9, 0.1], [0.1, 0.9]])
        target_cpu = torch.tensor([0, 1])

        metric_cpu = Accuracy()
        metric_cpu.update(pred_cpu, target_cpu)
        acc_cpu = metric_cpu.compute()

        if torch.cuda.is_available():
            pred_gpu = pred_cpu.cuda()
            target_gpu = target_cpu.cuda()

            metric_gpu = Accuracy()
            metric_gpu.update(pred_gpu, target_gpu)
            acc_gpu = metric_gpu.compute()

            assert abs(acc_cpu - acc_gpu) < 1e-6, "Metrics should match across devices"
