"""
Tokenizer quality assessment metrics.

This module provides utilities for evaluating the quality of tokenization
and reporting metrics on tokenizer performance.
"""

import logging
import random
from collections import Counter
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


def calculate_tokenizer_metrics(
    tokenizer: Any,
    text_samples: List[str],
    sample_size: int = 20,
    verbose: bool = False,
) -> Dict[str, Any]:
    """
    Calculate various metrics to evaluate tokenizer quality.
    
    Args:
        tokenizer: The tokenizer to evaluate
        text_samples: List of text samples to tokenize
        sample_size: Number of samples to use for detailed analysis
        verbose: Whether to print detailed metrics
        
    Returns:
        Dictionary of tokenizer quality metrics
    """
    if not text_samples:
        return {"error": "No text samples provided"}

    # Select a subset of samples for detailed analysis
    if len(text_samples) > sample_size:
        analysis_samples = random.sample(text_samples, sample_size)
    else:
        analysis_samples = text_samples

    metrics = {}

    # 1. Token counts and sequence length statistics
    all_tokens = []
    sequence_lengths = []
    unknown_tokens = 0
    special_tokens = 0

    # Get special token IDs for identification
    special_token_ids = set()
    if hasattr(tokenizer, "special_tokens"):
        special_token_ids = set(tokenizer.special_tokens.values())

    # Get unknown token ID if available
    unk_token_id = None
    if hasattr(tokenizer, "special_tokens") and "unk_token_idx" in tokenizer.special_tokens:
        unk_token_id = tokenizer.special_tokens["unk_token_idx"]

    # Process each sample
    for text in analysis_samples:
        # Tokenize
        token_ids = tokenizer.encode(text)
        tokens = tokenizer.tokenize(text) if hasattr(tokenizer, "tokenize") else []

        # Count tokens
        all_tokens.extend(tokens)
        sequence_lengths.append(len(token_ids))

        # Count special tokens
        special_tokens += sum(1 for tid in token_ids if tid in special_token_ids)

        # Count unknown tokens
        if unk_token_id is not None:
            unknown_tokens += sum(1 for tid in token_ids if tid == unk_token_id)

    # Calculate token distribution metrics
    token_counter = Counter(all_tokens)
    total_tokens = len(all_tokens)
    unique_tokens = len(token_counter)

    # Basic metrics
    metrics["total_tokens"] = total_tokens
    metrics["unique_tokens"] = unique_tokens
    metrics["unique_token_ratio"] = unique_tokens / max(1, total_tokens)
    metrics["avg_sequence_length"] = sum(sequence_lengths) / max(1, len(sequence_lengths))
    metrics["unknown_token_percent"] = unknown_tokens / max(1, total_tokens) * 100
    metrics["special_token_percent"] = special_tokens / max(1, sum(sequence_lengths)) * 100

    # Top frequent tokens
    most_common = token_counter.most_common(10)
    metrics["top_tokens"] = [token for token, count in most_common]
    metrics["top_token_frequencies"] = [count / total_tokens for token, count in most_common]

    # 2. Test tokenization consistency
    consistency_scores = []

    for text in analysis_samples:
        # Original tokenization
        token_ids = tokenizer.encode(text)

        # Decode back to text
        decoded_text = tokenizer.decode(token_ids)

        # Re-encode the decoded text
        re_token_ids = tokenizer.encode(decoded_text)

        # Calculate consistency (Jaccard similarity of token ID sequences)
        # Only compare the non-padding part of the sequences
        if hasattr(tokenizer, "special_tokens") and "pad_token_idx" in tokenizer.special_tokens:
            pad_id = tokenizer.special_tokens["pad_token_idx"]
            # Remove padding tokens
            token_ids = [t for t in token_ids if t != pad_id]
            re_token_ids = [t for t in re_token_ids if t != pad_id]

        # Calculate Jaccard similarity
        token_ids_set = set(token_ids)
        re_token_ids_set = set(re_token_ids)

        intersection = len(token_ids_set.intersection(re_token_ids_set))
        union = len(token_ids_set.union(re_token_ids_set))

        if union > 0:
            jaccard = intersection / union
        else:
            jaccard = 1.0  # Empty sets considered perfectly consistent

        consistency_scores.append(jaccard)

    metrics["tokenization_consistency"] = sum(consistency_scores) / max(1, len(consistency_scores))

    # Print metrics if verbose
    if verbose:
        logger.info("=== Tokenizer Quality Metrics ===")
        logger.info(f"Vocabulary Coverage:")
        logger.info(f"  - Total tokens processed: {total_tokens}")
        logger.info(f"  - Unique tokens: {unique_tokens} ({metrics['unique_token_ratio']:.2%})")
        logger.info(f"  - Unknown tokens: {unknown_tokens} ({metrics['unknown_token_percent']:.2f}%)")
        logger.info(f"  - Special tokens: {special_tokens} ({metrics['special_token_percent']:.2f}%)")
        logger.info(f"  - Average sequence length: {metrics['avg_sequence_length']:.2f}")

        logger.info(f"Most common tokens:")
        for i, (token, freq) in enumerate(zip(metrics["top_tokens"], metrics["top_token_frequencies"])):
            logger.info(f"  {i+1}. '{token}': {freq:.2%}")

        logger.info(f"Tokenization consistency: {metrics['tokenization_consistency']:.4f}")

    return metrics


def calculate_semantic_token_metrics(
    tokenizer: Any,
    semantic_groups: Dict[str, List[str]],
    sample_size: int = 5,
    verbose: bool = False
) -> Dict[str, Any]:
    """
    Calculate metrics for tokenization of semantically related texts.
    
    Args:
        tokenizer: The tokenizer to evaluate
        semantic_groups: Dictionary mapping group IDs to lists of related texts
        sample_size: Number of semantic groups to analyze
        verbose: Whether to print detailed metrics
        
    Returns:
        Dictionary of semantic tokenization metrics
    """
    if not semantic_groups:
        return {"error": "No semantic groups provided"}

    # Select a subset of groups for analysis
    group_ids = list(semantic_groups.keys())
    if len(group_ids) > sample_size:
        analysis_groups = random.sample(group_ids, sample_size)
    else:
        analysis_groups = group_ids

    metrics = {}

    # Metrics across all groups
    token_overlap_scores = []
    token_diversity_scores = []

    # Process each group
    for group_id in analysis_groups:
        texts = semantic_groups[group_id]

        if len(texts) < 2:
            continue  # Need at least 2 texts for comparison

        # Tokenize all texts in the group
        tokenized_texts = []
        for text in texts:
            token_ids = tokenizer.encode(text)

            # Remove padding if possible
            if hasattr(tokenizer, "special_tokens") and "pad_token_idx" in tokenizer.special_tokens:
                pad_id = tokenizer.special_tokens["pad_token_idx"]
                token_ids = [t for t in token_ids if t != pad_id]

            tokenized_texts.append(set(token_ids))

        # Calculate token overlap (Jaccard similarity) between all pairs
        group_overlaps = []
        for i in range(len(tokenized_texts)):
            for j in range(i+1, len(tokenized_texts)):
                set1 = tokenized_texts[i]
                set2 = tokenized_texts[j]

                intersection = len(set1.intersection(set2))
                union = len(set1.union(set2))

                if union > 0:
                    jaccard = intersection / union
                else:
                    jaccard = 1.0  # Empty sets considered perfectly overlapping

                group_overlaps.append(jaccard)

        if group_overlaps:
            # Average overlap for this group
            avg_overlap = sum(group_overlaps) / len(group_overlaps)
            token_overlap_scores.append(avg_overlap)

            # Token diversity: how many unique tokens across all texts in group
            all_tokens = set()
            for token_set in tokenized_texts:
                all_tokens.update(token_set)

            # Diversity = unique tokens / average tokens per text
            avg_tokens = sum(len(token_set) for token_set in tokenized_texts) / len(tokenized_texts)
            diversity = len(all_tokens) / max(1, avg_tokens)
            token_diversity_scores.append(diversity)

    # Calculate overall metrics
    if token_overlap_scores:
        metrics["semantic_token_overlap"] = sum(token_overlap_scores) / len(token_overlap_scores)
    else:
        metrics["semantic_token_overlap"] = 0.0

    if token_diversity_scores:
        metrics["semantic_token_diversity"] = sum(token_diversity_scores) / len(token_diversity_scores)
    else:
        metrics["semantic_token_diversity"] = 0.0

    # Print metrics if verbose
    if verbose:
        logger.info("=== Semantic Tokenization Metrics ===")
        logger.info(f"Analyzed {len(token_overlap_scores)} semantic groups")
        logger.info(f"Average token overlap between related texts: {metrics['semantic_token_overlap']:.4f}")
        logger.info(f"Token diversity ratio within semantic groups: {metrics['semantic_token_diversity']:.4f}")

    return metrics


def log_tokenizer_evaluation(
    tokenizer: Any,
    text_data: List[str],
    match_ids: Optional[List[str]] = None,
    epoch: int = 0,
) -> Dict[str, Any]:
    """
    Evaluate tokenizer quality and log results.
    
    Args:
        tokenizer: Tokenizer to evaluate
        text_data: List of text samples
        match_ids: Optional list of match IDs for semantic grouping
        epoch: Current epoch (for logging)
        
    Returns:
        Combined metrics dictionary
    """
    logger.info(f"Evaluating tokenizer quality for epoch {epoch}")

    # Calculate basic metrics
    basic_metrics = calculate_tokenizer_metrics(
        tokenizer,
        text_data,
        sample_size=min(20, len(text_data)),
        verbose=True
    )

    # Calculate semantic metrics if match_ids are provided
    semantic_metrics = {}
    if match_ids is not None:
        # Group texts by match_id
        semantic_groups = {}
        for text, mid in zip(text_data, match_ids):
            if mid not in semantic_groups:
                semantic_groups[mid] = []
            semantic_groups[mid].append(text)

        # Get metrics for semantic groups
        semantic_metrics = calculate_semantic_token_metrics(
            tokenizer,
            semantic_groups,
            sample_size=min(5, len(semantic_groups)),
            verbose=True
        )

    # Combine metrics
    all_metrics = {**basic_metrics, **semantic_metrics}

    # Add additional insights based on metrics
    quality_score = 0.0
    num_factors = 0

    # 1. Contribution from consistency
    if "tokenization_consistency" in all_metrics:
        quality_score += all_metrics["tokenization_consistency"]
        num_factors += 1

    # 2. Contribution from unknown token percentage (inversely related)
    if "unknown_token_percent" in all_metrics:
        unk_quality = max(0, 1 - (all_metrics["unknown_token_percent"] / 100))
        quality_score += unk_quality
        num_factors += 1

    # 3. Contribution from semantic overlap
    if "semantic_token_overlap" in all_metrics:
        quality_score += all_metrics["semantic_token_overlap"]
        num_factors += 1

    # Calculate overall quality score (0-1 scale)
    if num_factors > 0:
        all_metrics["overall_quality_score"] = quality_score / num_factors

        # Log overall assessment
        quality_level = "Unknown"
        if all_metrics["overall_quality_score"] >= 0.9:
            quality_level = "Excellent"
        elif all_metrics["overall_quality_score"] >= 0.75:
            quality_level = "Good"
        elif all_metrics["overall_quality_score"] >= 0.5:
            quality_level = "Fair"
        elif all_metrics["overall_quality_score"] >= 0.25:
            quality_level = "Poor"
        else:
            quality_level = "Very Poor"

        all_metrics["quality_level"] = quality_level

        logger.info(f"Overall tokenizer quality: {quality_level} ({all_metrics['overall_quality_score']:.4f})")

    return all_metrics
