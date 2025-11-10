"""MODULE: evaluation.py
PURPOSE: Handles model evaluation with comprehensive retrieval metrics.

KEY COMPONENTS:
- Evaluator: Manages evaluation with global and in-batch metrics
- Supports multimodal retrieval evaluation (image-to-text, text-to-image)
- Computes Recall@K metrics for retrieval tasks
- Provides both global (correct) and in-batch (comparison) metrics

DEPENDENCIES:
- PyTorch (torch, torch.nn.functional)
- tqdm for progress tracking
- Python standard library (logging, collections)

SPECIAL NOTES:
- Uses global evaluation to avoid artificially high in-batch metrics
- Computes similarity across ALL samples, not just within batches
- Handles feature normalization and pooling automatically
- Tracks original indices for correct ground truth matching
"""

import logging
from collections import defaultdict
from typing import Dict, List, Optional, Tuple, Callable, Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

logger = logging.getLogger(__name__)


class Evaluator:
    """Handles model evaluation with comprehensive retrieval metrics."""

    def __init__(
        self,
        model: nn.Module,
        device: torch.device,
        recall_k_values: List[int] = None,
    ):
        """
        Initialize the evaluator.

        Args:
            model: The model to evaluate
            device: Device to run evaluation on
            recall_k_values: List of K values for Recall@K (default: [1, 5, 10])
        """
        self.model = model
        self.device = device
        self.recall_k_values = recall_k_values or [1, 5, 10]

    def evaluate(
        self,
        dataloader: torch.utils.data.DataLoader,
        prepare_model_inputs_fn: Callable,
        to_device_fn: Callable,
        compute_in_batch_comparison: bool = True,
    ) -> Dict[str, float]:
        """
        Evaluate the model using global comparison across all samples.

        This addresses the critical issue where in-batch metrics can give
        artificially high performance. It compares each image against ALL
        captions in the dataset.

        Args:
            dataloader: DataLoader for evaluation
            prepare_model_inputs_fn: Function to prepare model inputs from batch
            to_device_fn: Function to move batch to device
            compute_in_batch_comparison: Whether to compute in-batch metrics for comparison

        Returns:
            Dictionary with evaluation metrics
        """
        self.model.eval()

        # Collect all embeddings first
        embeddings = self._collect_embeddings(
            dataloader, prepare_model_inputs_fn, to_device_fn
        )

        if embeddings is None:
            logger.error("No embeddings collected during evaluation")
            return {"error": 1.0}

        # Unpack embeddings
        all_image_embeddings, all_text_embeddings, all_original_indices = embeddings

        # Compute global similarity matrix
        logger.info(f"Computing global similarity matrix of shape "
                   f"{all_image_embeddings.shape[0]}×{all_text_embeddings.shape[0]}...")
        similarity = torch.matmul(all_image_embeddings, all_text_embeddings.T)

        # Compute global retrieval metrics
        global_metrics = self._compute_global_metrics(
            similarity, all_original_indices
        )

        # Compute in-batch metrics for comparison
        if compute_in_batch_comparison:
            in_batch_metrics = self._compute_in_batch_metrics(similarity)

            # Print comparison
            self._print_metric_comparison(global_metrics, in_batch_metrics)

            # Add in-batch metrics to results
            for k, v in in_batch_metrics.items():
                global_metrics[f"in_batch_{k}"] = v

        return global_metrics

    def _collect_embeddings(
        self,
        dataloader: torch.utils.data.DataLoader,
        prepare_model_inputs_fn: Callable,
        to_device_fn: Callable,
    ) -> Optional[Tuple[torch.Tensor, torch.Tensor, List[int]]]:
        """
        Collect all embeddings from the dataset.

        Args:
            dataloader: DataLoader for evaluation
            prepare_model_inputs_fn: Function to prepare model inputs
            to_device_fn: Function to move batch to device

        Returns:
            Tuple of (image_embeddings, text_embeddings, original_indices) or None
        """
        all_image_embeddings = []
        all_text_embeddings = []
        all_original_indices = []

        logger.info(f"Collecting embeddings from all {len(dataloader)} batches...")

        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Extracting features"):
                # Move batch to device
                batch = to_device_fn(batch)

                # Prepare model inputs
                model_inputs = prepare_model_inputs_fn(batch)

                # Skip invalid batches
                if not model_inputs:
                    logger.warning("No valid data found in batch!")
                    continue

                # Forward pass
                outputs = self.model(**model_inputs)

                # Extract and process image features
                image_features = self._extract_features(
                    outputs,
                    preferred_keys=["vision_features_enhanced", "image_features", "vision_features"]
                )

                if image_features is not None:
                    image_features = self._process_features(image_features)
                    all_image_embeddings.append(image_features.cpu())

                # Extract and process text features
                text_features = self._extract_features(
                    outputs,
                    preferred_keys=["text_features_enhanced", "text_features"]
                )

                if text_features is not None:
                    text_features = self._process_features(text_features)
                    all_text_embeddings.append(text_features.cpu())

                # Track original indices
                indices = self._extract_indices(batch)
                if indices:
                    all_original_indices.extend(indices)

        # Check if we have embeddings
        if not all_image_embeddings or not all_text_embeddings:
            return None

        # Concatenate embeddings
        all_image_embeddings = torch.cat(all_image_embeddings, dim=0)
        all_text_embeddings = torch.cat(all_text_embeddings, dim=0)

        # Ensure we have indices for all samples
        if len(all_original_indices) != len(all_image_embeddings):
            logger.warning(f"Mismatch between indices ({len(all_original_indices)}) "
                          f"and embeddings ({len(all_image_embeddings)})")
            all_original_indices = list(range(len(all_image_embeddings)))

        return all_image_embeddings, all_text_embeddings, all_original_indices

    def _extract_features(
        self, outputs: Dict[str, torch.Tensor], preferred_keys: List[str]
    ) -> Optional[torch.Tensor]:
        """
        Extract features from model outputs with fallback keys.

        Args:
            outputs: Model outputs dictionary
            preferred_keys: List of keys to try in order

        Returns:
            Features tensor or None
        """
        for key in preferred_keys:
            if key in outputs:
                return outputs[key]
        return None

    def _process_features(self, features: torch.Tensor) -> torch.Tensor:
        """
        Process features: apply pooling if needed and normalize.

        Args:
            features: Feature tensor

        Returns:
            Processed features
        """
        # Apply mean pooling if features are sequences
        if len(features.shape) == 3:
            features = features.mean(dim=1)

        # Normalize features
        features = F.normalize(features, p=2, dim=1)

        return features

    def _extract_indices(self, batch: Dict[str, Any]) -> List[int]:
        """
        Extract original indices from batch.

        Args:
            batch: Batch dictionary

        Returns:
            List of indices
        """
        if "original_idx" in batch:
            return batch["original_idx"].cpu().tolist()
        elif "idx" in batch:
            return batch["idx"].cpu().tolist()
        return []

    def _compute_global_metrics(
        self, similarity: torch.Tensor, all_original_indices: List[int]
    ) -> Dict[str, float]:
        """
        Compute global retrieval metrics across all samples.

        Args:
            similarity: Global similarity matrix
            all_original_indices: Original sample indices for ground truth matching

        Returns:
            Dictionary with global metrics
        """
        metrics = {}
        recalls = {}

        correct_image_to_text = 0
        correct_text_to_image = 0

        # For each position, compute retrieval metrics
        for i, orig_idx in enumerate(all_original_indices):
            # Get positions of all samples with same original index
            matching_positions = [
                j for j, idx in enumerate(all_original_indices) if idx == orig_idx
            ]

            if not matching_positions:
                continue

            # Compute Recall@K for each K value
            for k in self.recall_k_values:
                k_adjusted = min(k, len(all_original_indices))

                # Image-to-text retrieval
                i2t_topk = torch.topk(similarity[i], k_adjusted, dim=0)[1].cpu().tolist()
                i2t_hit = any(pos in i2t_topk for pos in matching_positions)

                # Count R@1 hits for accuracy
                if k == 1 and i2t_hit:
                    correct_image_to_text += 1

                # Text-to-image retrieval
                t2i_topk = torch.topk(similarity[:, i], k_adjusted, dim=0)[1].cpu().tolist()
                t2i_hit = any(pos in t2i_topk for pos in matching_positions)

                # Count R@1 hits for accuracy
                if k == 1 and t2i_hit:
                    correct_text_to_image += 1

                # Accumulate hits
                recall_key = f"recall@{k}"
                if recall_key not in recalls:
                    recalls[recall_key] = {"i2t_hits": 0, "t2i_hits": 0, "total": 0}

                recalls[recall_key]["i2t_hits"] += int(i2t_hit)
                recalls[recall_key]["t2i_hits"] += int(t2i_hit)
                recalls[recall_key]["total"] += 1

        # Calculate final recall metrics
        for k in self.recall_k_values:
            recall_key = f"recall@{k}"
            bucket = recalls[recall_key]

            if bucket["total"] > 0:
                i2t_recall = bucket["i2t_hits"] / bucket["total"]
                t2i_recall = bucket["t2i_hits"] / bucket["total"]
                avg_recall = (i2t_recall + t2i_recall) / 2
            else:
                i2t_recall = t2i_recall = avg_recall = 0.0

            metrics[f"global_i2t_recall@{k}"] = i2t_recall
            metrics[f"global_t2i_recall@{k}"] = t2i_recall
            metrics[f"global_avg_recall@{k}"] = avg_recall

        # Calculate accuracy metrics
        total_samples = len(all_original_indices)
        if total_samples > 0:
            i2t_accuracy = correct_image_to_text / total_samples
            t2i_accuracy = correct_text_to_image / total_samples
            avg_accuracy = (i2t_accuracy + t2i_accuracy) / 2
        else:
            i2t_accuracy = t2i_accuracy = avg_accuracy = 0.0

        metrics["global_i2t_accuracy"] = i2t_accuracy
        metrics["global_t2i_accuracy"] = t2i_accuracy
        metrics["global_accuracy"] = avg_accuracy

        # Print global metrics
        self._print_global_metrics(metrics)

        return metrics

    def _compute_in_batch_metrics(self, similarity: torch.Tensor) -> Dict[str, float]:
        """
        Compute traditional in-batch metrics assuming diagonal matches.

        Args:
            similarity: Similarity matrix

        Returns:
            Dictionary with in-batch metrics
        """
        batch_size = min(similarity.shape[0], similarity.shape[1])
        targets = torch.arange(batch_size, device=similarity.device)

        # Image-to-text metrics
        i2t_sim = similarity[:batch_size, :batch_size]
        i2t_pred = torch.argmax(i2t_sim, dim=1)
        i2t_accuracy = (i2t_pred == targets).float().mean().item()

        # Text-to-image metrics
        t2i_sim = similarity[:batch_size, :batch_size].T
        t2i_pred = torch.argmax(t2i_sim, dim=1)
        t2i_accuracy = (t2i_pred == targets).float().mean().item()

        metrics = {
            "i2t_accuracy": i2t_accuracy,
            "t2i_accuracy": t2i_accuracy,
            "accuracy": (i2t_accuracy + t2i_accuracy) / 2,
        }

        # Compute Recall@K
        for k in self.recall_k_values:
            k_adjusted = min(k, batch_size)

            # Image-to-text
            i2t_topk = torch.topk(i2t_sim, k_adjusted, dim=1)[1]
            i2t_hits = torch.zeros(batch_size, dtype=torch.bool, device=similarity.device)
            for i in range(batch_size):
                i2t_hits[i] = (i2t_topk[i] == i).any()
            i2t_recall = i2t_hits.float().mean().item()

            # Text-to-image
            t2i_topk = torch.topk(t2i_sim, k_adjusted, dim=1)[1]
            t2i_hits = torch.zeros(batch_size, dtype=torch.bool, device=similarity.device)
            for i in range(batch_size):
                t2i_hits[i] = (t2i_topk[i] == i).any()
            t2i_recall = t2i_hits.float().mean().item()

            metrics[f"i2t_recall@{k}"] = i2t_recall
            metrics[f"t2i_recall@{k}"] = t2i_recall
            metrics[f"avg_recall@{k}"] = (i2t_recall + t2i_recall) / 2

        return metrics

    def _print_global_metrics(self, metrics: Dict[str, float]) -> None:
        """Print global evaluation metrics."""
        print("\n*** GLOBAL EVALUATION METRICS (USE THESE FOR FINAL RESULTS) ***")
        print(f"  Accuracy: {metrics['global_accuracy']:.4f} "
              f"(I2T: {metrics['global_i2t_accuracy']:.4f}, "
              f"T2I: {metrics['global_t2i_accuracy']:.4f})")

        for k in self.recall_k_values:
            print(f"  Recall@{k}: {metrics[f'global_avg_recall@{k}']:.4f} "
                  f"(I2T: {metrics[f'global_i2t_recall@{k}']:.4f}, "
                  f"T2I: {metrics[f'global_t2i_recall@{k}']:.4f})")

    def _print_metric_comparison(
        self, global_metrics: Dict[str, float], in_batch_metrics: Dict[str, float]
    ) -> None:
        """Print comparison between global and in-batch metrics."""
        print("\n⚠️ IN-BATCH VS GLOBAL METRICS COMPARISON ⚠️")
        print("WARNING: In-batch metrics are often misleadingly high!")

        # Compare accuracy
        global_acc = global_metrics['global_accuracy']
        in_batch_acc = in_batch_metrics['accuracy']
        acc_ratio = in_batch_acc / max(1e-5, global_acc)
        print(f"  Accuracy: In-Batch={in_batch_acc:.4f}, Global={global_acc:.4f}, "
              f"Ratio={acc_ratio:.1f}x higher (artificial)")

        # Compare Recall@K
        for k in self.recall_k_values:
            in_batch = in_batch_metrics[f'avg_recall@{k}']
            global_val = global_metrics[f'global_avg_recall@{k}']
            ratio = in_batch / max(1e-5, global_val)
            print(f"  Recall@{k}: In-Batch={in_batch:.4f}, Global={global_val:.4f}, "
                  f"Ratio={ratio:.1f}x higher (artificial)")

    def compute_retrieval_metrics(
        self,
        image_embeddings: torch.Tensor,
        text_embeddings: torch.Tensor,
        indices: Optional[List[int]] = None,
    ) -> Dict[str, float]:
        """
        Compute retrieval metrics from pre-computed embeddings.

        Args:
            image_embeddings: Image feature embeddings
            text_embeddings: Text feature embeddings
            indices: Optional original indices for ground truth matching

        Returns:
            Dictionary with retrieval metrics
        """
        # Normalize embeddings
        image_embeddings = F.normalize(image_embeddings, p=2, dim=1)
        text_embeddings = F.normalize(text_embeddings, p=2, dim=1)

        # Compute similarity
        similarity = torch.matmul(image_embeddings, text_embeddings.T)

        # Use position-based indices if not provided
        if indices is None:
            indices = list(range(len(image_embeddings)))

        # Compute metrics
        return self._compute_global_metrics(similarity, indices)
