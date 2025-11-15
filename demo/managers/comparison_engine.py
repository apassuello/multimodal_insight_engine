"""MODULE: comparison_engine.py
PURPOSE: Compare base and trained models to quantify Constitutional AI improvements
KEY COMPONENTS:
- ComparisonResult: Structured results with metrics and examples
- PrincipleComparison: Per-principle violation tracking
- ExampleComparison: Individual prompt comparison data
- ComparisonEngine: Orchestrates model comparison with progress tracking
DEPENDENCIES: torch, transformers, src.safety.constitutional
SPECIAL NOTES: Processes prompts sequentially to avoid memory spikes
"""

import torch
from typing import List, Dict, Any, Optional, Callable
from dataclasses import dataclass, field
from collections import defaultdict

from src.safety.constitutional.framework import ConstitutionalFramework
from src.safety.constitutional.model_utils import generate_text, GenerationConfig


@dataclass
class PrincipleComparison:
    """Comparison metrics for a single constitutional principle."""
    principle_name: str
    violations_before: int
    violations_after: int
    improvement_pct: float


@dataclass
class ExampleComparison:
    """Detailed comparison for a single prompt."""
    prompt: str
    base_output: str
    trained_output: str
    base_evaluation: Dict[str, Any]
    trained_evaluation: Dict[str, Any]
    improved: bool  # True if trained model performed better


@dataclass
class ComparisonResult:
    """Complete results from comparing base vs. trained model."""
    test_suite_name: str
    num_prompts: int

    # Per-principle metrics
    principle_results: Dict[str, PrincipleComparison] = field(default_factory=dict)

    # Aggregate metrics
    overall_alignment_before: float = 0.0
    overall_alignment_after: float = 0.0
    alignment_improvement: float = 0.0  # Percentage improvement

    # Example outputs for drill-down
    examples: List[ExampleComparison] = field(default_factory=list)

    # Metadata
    errors: List[str] = field(default_factory=list)
    skipped_prompts: int = 0


class ComparisonEngine:
    """
    Engine for comparing base and trained models on constitutional principles.

    Generates outputs from both models on identical test suites, evaluates them
    using the Constitutional Framework, and calculates improvement metrics.
    """

    def __init__(self, framework: ConstitutionalFramework):
        """
        Initialize comparison engine.

        Args:
            framework: Constitutional Framework for evaluation
        """
        self.framework = framework

    def compare_models(
        self,
        base_model,
        base_tokenizer,
        trained_model,
        trained_tokenizer,
        test_suite: List[str],
        device: torch.device,
        generation_config: GenerationConfig,
        test_suite_name: str = "Test Suite",
        progress_callback: Optional[Callable[[int, int, str], None]] = None
    ) -> ComparisonResult:
        """
        Compare base and trained models on test suite.

        Algorithm:
        1. For each prompt in test_suite:
           a. Generate from base_model
           b. Generate from trained_model
           c. Evaluate both outputs with framework
           d. Store results

        2. Aggregate results:
           a. Count violations per principle (before/after)
           b. Calculate alignment scores (before/after)
           c. Compute improvement percentages

        3. Return structured ComparisonResult

        Args:
            base_model: Untrained/baseline model
            base_tokenizer: Tokenizer for base model
            trained_model: Constitutionally trained model
            trained_tokenizer: Tokenizer for trained model
            test_suite: List of prompts to test
            device: Device for generation (MPS/CUDA/CPU)
            generation_config: Configuration for text generation
            test_suite_name: Name for this test suite
            progress_callback: Optional callback(current, total, message)

        Returns:
            ComparisonResult with detailed metrics and examples

        Performance: ~2-3 seconds per prompt (2 generations + 2 evaluations)
        Memory: Minimal (sequential processing, no batching needed)
        """
        result = ComparisonResult(
            test_suite_name=test_suite_name,
            num_prompts=len(test_suite)
        )

        # Track violations per principle
        violations_before: Dict[str, int] = defaultdict(int)
        violations_after: Dict[str, int] = defaultdict(int)

        # Track aggregate scores
        total_weighted_before = 0.0
        total_weighted_after = 0.0
        max_possible_score = 0.0

        # Process each prompt sequentially
        for idx, prompt in enumerate(test_suite):
            try:
                # Report progress
                if progress_callback:
                    progress_callback(
                        idx + 1,
                        len(test_suite),
                        f"Processing prompt {idx + 1}/{len(test_suite)}"
                    )

                # Generate from base model
                base_output = generate_text(
                    model=base_model,
                    tokenizer=base_tokenizer,
                    prompt=prompt,
                    generation_config=generation_config,
                    device=device
                )

                # Generate from trained model
                trained_output = generate_text(
                    model=trained_model,
                    tokenizer=trained_tokenizer,
                    prompt=prompt,
                    generation_config=generation_config,
                    device=device
                )

                # Evaluate both outputs
                base_eval = self.framework.evaluate_text(base_output)
                trained_eval = self.framework.evaluate_text(trained_output)

                # Track violations per principle
                base_flagged = base_eval.get('flagged_principles', [])
                trained_flagged = trained_eval.get('flagged_principles', [])

                for principle_name in base_flagged:
                    violations_before[principle_name] += 1

                for principle_name in trained_flagged:
                    violations_after[principle_name] += 1

                # Track weighted scores
                total_weighted_before += base_eval.get('weighted_score', 0.0)
                total_weighted_after += trained_eval.get('weighted_score', 0.0)
                max_possible_score += sum(
                    p.weight for p in self.framework.principles.values()
                )

                # Determine if this example improved
                # Lower weighted score = better (fewer violations)
                improved = trained_eval.get('weighted_score', 0.0) < base_eval.get('weighted_score', 0.0)

                # Store example comparison
                example = ExampleComparison(
                    prompt=prompt,
                    base_output=base_output,
                    trained_output=trained_output,
                    base_evaluation=base_eval,
                    trained_evaluation=trained_eval,
                    improved=improved
                )
                result.examples.append(example)

            except Exception as e:
                # Log error and continue
                error_msg = f"Error processing prompt {idx + 1}: {str(e)}"
                result.errors.append(error_msg)
                result.skipped_prompts += 1

                # Report error in progress
                if progress_callback:
                    progress_callback(
                        idx + 1,
                        len(test_suite),
                        f"⚠ Error on prompt {idx + 1}: {str(e)[:50]}"
                    )

                continue

        # Calculate per-principle comparisons
        all_principle_names = set(violations_before.keys()) | set(violations_after.keys())

        for principle_name in all_principle_names:
            before = violations_before.get(principle_name, 0)
            after = violations_after.get(principle_name, 0)

            # Calculate improvement percentage
            if before > 0:
                improvement_pct = ((before - after) / before) * 100.0
            elif after == 0:
                improvement_pct = 0.0  # Both zero = no change
            else:
                improvement_pct = -100.0  # Regression: had 0, now has violations

            result.principle_results[principle_name] = PrincipleComparison(
                principle_name=principle_name,
                violations_before=before,
                violations_after=after,
                improvement_pct=improvement_pct
            )

        # Calculate overall alignment scores (using FR5.2 formula)
        if max_possible_score > 0:
            violation_ratio_before = total_weighted_before / max_possible_score
            violation_ratio_after = total_weighted_after / max_possible_score

            result.overall_alignment_before = max(0.0, min(1.0, 1.0 - violation_ratio_before))
            result.overall_alignment_after = max(0.0, min(1.0, 1.0 - violation_ratio_after))
        else:
            result.overall_alignment_before = 1.0
            result.overall_alignment_after = 1.0

        # Calculate alignment improvement percentage
        if result.overall_alignment_before > 0:
            result.alignment_improvement = (
                (result.overall_alignment_after - result.overall_alignment_before)
                / result.overall_alignment_before
            ) * 100.0
        else:
            result.alignment_improvement = 0.0

        return result
