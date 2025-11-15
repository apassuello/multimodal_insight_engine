"""MODULE: evaluation_manager.py
PURPOSE: Evaluation orchestration for Constitutional AI demo
KEY COMPONENTS:
- EvaluationManager: Handles text evaluation against constitutional principles
- Single text evaluation with AI or regex
- Side-by-side comparison (AI vs regex)
- Batch evaluation with statistics
DEPENDENCIES: typing, src.safety.constitutional
SPECIAL NOTES: Supports both AI-based and regex-based evaluation modes
"""

import time
from typing import Dict, List, Any, Optional, Tuple

from src.safety.constitutional.framework import ConstitutionalFramework
from src.safety.constitutional.principles import setup_default_framework


class EvaluationManager:
    """
    Manages constitutional principle evaluation for the demo.

    Handles single text evaluation, batch evaluation, and
    AI vs regex comparison modes.
    """

    def __init__(self):
        """Initialize evaluation manager."""
        self.framework_ai: Optional[ConstitutionalFramework] = None
        self.framework_regex: Optional[ConstitutionalFramework] = None
        self.last_evaluation: Optional[Dict[str, Any]] = None

    def initialize_frameworks(
        self,
        model=None,
        tokenizer=None,
        device=None
    ) -> Tuple[bool, str]:
        """
        Initialize both AI and regex evaluation frameworks.

        Args:
            model: Optional model for AI evaluation
            tokenizer: Optional tokenizer for AI evaluation
            device: Optional device for computation

        Returns:
            Tuple of (success: bool, message: str)
        """
        try:
            # Setup AI-based framework if model provided
            if model is not None and tokenizer is not None:
                self.framework_ai = setup_default_framework(
                    model=model,
                    tokenizer=tokenizer,
                    device=device
                )

            # Setup regex-based framework (always available as fallback)
            self.framework_regex = setup_default_framework(
                model=None,
                tokenizer=None,
                device=None
            )

            if self.framework_ai:
                return True, "✓ AI and regex frameworks initialized"
            else:
                return True, "✓ Regex framework initialized (no model for AI evaluation)"

        except Exception as e:
            return False, f"✗ Failed to initialize frameworks: {str(e)}"

    def evaluate_text(
        self,
        text: str,
        mode: str = "ai"
    ) -> Tuple[Dict[str, Any], bool, str]:
        """
        Evaluate text against constitutional principles.

        Args:
            text: Text to evaluate
            mode: Evaluation mode ("ai", "regex", or "both")

        Returns:
            Tuple of (results: dict, success: bool, message: str)
        """
        if not text or len(text.strip()) == 0:
            return {}, False, "✗ Please provide text to evaluate"

        try:
            start_time = time.time()

            if mode == "ai":
                if not self.framework_ai:
                    return {}, False, "✗ AI evaluation not available (no model loaded)"

                result = self.framework_ai.evaluate_text(text)
                result["mode"] = "ai"

            elif mode == "regex":
                if not self.framework_regex:
                    return {}, False, "✗ Regex evaluation not available"

                result = self.framework_regex.evaluate_text(text)
                result["mode"] = "regex"

            elif mode == "both":
                return self.evaluate_both(text)

            else:
                return {}, False, f"✗ Invalid evaluation mode: {mode}"

            elapsed_time = time.time() - start_time
            result["evaluation_time"] = elapsed_time

            # Format result for display
            formatted = self._format_evaluation_result(result)
            self.last_evaluation = result

            message = f"✓ Evaluation completed in {elapsed_time:.2f}s"
            return formatted, True, message

        except Exception as e:
            return {}, False, f"✗ Evaluation failed: {str(e)}"

    def evaluate_both(
        self,
        text: str
    ) -> Tuple[Dict[str, Any], bool, str]:
        """
        Evaluate text with both AI and regex methods for comparison.

        Args:
            text: Text to evaluate

        Returns:
            Tuple of (results: dict, success: bool, message: str)
        """
        try:
            start_time = time.time()

            # Evaluate with both methods
            ai_result = None
            regex_result = None

            if self.framework_ai:
                ai_result = self.framework_ai.evaluate_text(text)

            if self.framework_regex:
                regex_result = self.framework_regex.evaluate_text(text)

            if not ai_result and not regex_result:
                return {}, False, "✗ No evaluation framework available"

            # Compare results
            comparison = {
                "ai_evaluation": ai_result,
                "regex_evaluation": regex_result,
                "comparison": self._compare_evaluations(ai_result, regex_result),
                "mode": "both"
            }

            elapsed_time = time.time() - start_time
            comparison["evaluation_time"] = elapsed_time

            message = f"✓ Comparison completed in {elapsed_time:.2f}s"
            return comparison, True, message

        except Exception as e:
            return {}, False, f"✗ Comparison failed: {str(e)}"

    def _compare_evaluations(
        self,
        ai_result: Optional[Dict[str, Any]],
        regex_result: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Compare AI and regex evaluation results.

        Args:
            ai_result: AI evaluation result
            regex_result: Regex evaluation result

        Returns:
            Comparison statistics
        """
        if not ai_result or not regex_result:
            return {"error": "Missing evaluation results"}

        ai_flagged = set(ai_result.get("flagged_principles", []))
        regex_flagged = set(regex_result.get("flagged_principles", []))

        # Calculate differences
        only_ai = ai_flagged - regex_flagged
        only_regex = regex_flagged - ai_flagged
        both = ai_flagged & regex_flagged

        return {
            "ai_flagged_count": len(ai_flagged),
            "regex_flagged_count": len(regex_flagged),
            "both_flagged": list(both),
            "only_ai_detected": list(only_ai),
            "only_regex_detected": list(only_regex),
            "agreement": len(both) / max(len(ai_flagged | regex_flagged), 1),
            "ai_advantage": len(only_ai),
            "regex_advantage": len(only_regex)
        }

    def _format_evaluation_result(
        self,
        result: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Format evaluation result for display.

        Args:
            result: Raw evaluation result

        Returns:
            Formatted result dictionary
        """
        formatted = {
            "summary": {
                "any_flagged": result.get("any_flagged", False),
                "flagged_principles": result.get("flagged_principles", []),
                "weighted_score": result.get("weighted_score", 0.0),
                "num_principles_evaluated": result.get("num_principles_evaluated", 0),
                "evaluation_method": result.get("evaluation_method", "unknown"),
                "evaluation_time": result.get("evaluation_time", 0.0)
            },
            "principles": {}
        }

        # Format per-principle results
        principle_results = result.get("principle_results", {})
        for principle_name, principle_result in principle_results.items():
            formatted["principles"][principle_name] = {
                "flagged": principle_result.get("flagged", False),
                "weight": principle_result.get("weight", 1.0),
                "method": principle_result.get("method", "unknown"),
                "details": self._extract_principle_details(principle_result)
            }

        return formatted

    def _extract_principle_details(
        self,
        principle_result: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Extract relevant details from principle result.

        Args:
            principle_result: Raw principle evaluation result

        Returns:
            Formatted details dictionary
        """
        details = {}

        # Common fields
        if "reasoning" in principle_result:
            details["reasoning"] = principle_result["reasoning"]

        # Harm prevention specific
        if "explicit_harm_detected" in principle_result:
            details["explicit_harm"] = principle_result["explicit_harm_detected"]
        if "subtle_harm_score" in principle_result:
            details["subtle_harm_score"] = principle_result["subtle_harm_score"]

        # Truthfulness specific
        if "unsupported_claims" in principle_result:
            details["unsupported_claims"] = principle_result["unsupported_claims"]
        if "contradictions" in principle_result:
            details["contradictions"] = principle_result["contradictions"]
        if "misleading_statistics" in principle_result:
            details["misleading_statistics"] = principle_result["misleading_statistics"]

        # Fairness specific
        if "stereotypes" in principle_result:
            details["stereotypes"] = principle_result["stereotypes"]
        if "biased_language" in principle_result:
            details["biased_language"] = principle_result["biased_language"]

        # Autonomy specific
        if "coercive_language" in principle_result:
            details["coercive_language"] = principle_result["coercive_language"]
        if "manipulative_language" in principle_result:
            details["manipulative_language"] = principle_result["manipulative_language"]

        return details

    def batch_evaluate(
        self,
        texts: List[str],
        mode: str = "ai"
    ) -> Tuple[Dict[str, Any], bool, str]:
        """
        Evaluate multiple texts and provide aggregate statistics.

        Args:
            texts: List of texts to evaluate
            mode: Evaluation mode ("ai" or "regex")

        Returns:
            Tuple of (results: dict, success: bool, message: str)
        """
        if not texts or len(texts) == 0:
            return {}, False, "✗ No texts provided for batch evaluation"

        try:
            start_time = time.time()

            # Select framework
            if mode == "ai":
                if not self.framework_ai:
                    return {}, False, "✗ AI evaluation not available"
                framework = self.framework_ai
            else:
                if not self.framework_regex:
                    return {}, False, "✗ Regex evaluation not available"
                framework = self.framework_regex

            # Evaluate all texts
            results = []
            for text in texts:
                result = framework.evaluate_text(text)
                results.append(result)

            # Calculate aggregate statistics
            total_flagged = sum(1 for r in results if r.get("any_flagged", False))
            flagged_rate = total_flagged / len(texts) if texts else 0.0

            # Per-principle statistics
            principle_stats = {}
            for principle_name in framework.principles.keys():
                violations = sum(
                    1 for r in results
                    if principle_name in r.get("flagged_principles", [])
                )
                principle_stats[principle_name] = {
                    "violations": violations,
                    "rate": violations / len(texts) if texts else 0.0
                }

            elapsed_time = time.time() - start_time

            batch_result = {
                "num_texts": len(texts),
                "total_flagged": total_flagged,
                "flagged_rate": flagged_rate,
                "principle_statistics": principle_stats,
                "individual_results": results,
                "evaluation_time": elapsed_time,
                "mode": mode
            }

            message = f"✓ Batch evaluation completed: {total_flagged}/{len(texts)} texts flagged"
            return batch_result, True, message

        except Exception as e:
            return {}, False, f"✗ Batch evaluation failed: {str(e)}"

    def calculate_alignment_score(
        self,
        texts: List[str],
        mode: str = "ai"
    ) -> Tuple[float, bool, str]:
        """
        Calculate alignment score for a set of outputs.

        Alignment score = 1.0 - (total_weighted_violations / max_possible_violations)
        Range: 0.0 (completely misaligned) to 1.0 (perfectly aligned)

        Args:
            texts: List of texts to evaluate
            mode: Evaluation mode ("ai" or "regex")

        Returns:
            Tuple of (score: float, success: bool, message: str)
        """
        if not texts or len(texts) == 0:
            return 0.0, False, "✗ No texts provided"

        try:
            # Select framework
            if mode == "ai":
                if not self.framework_ai:
                    return 0.0, False, "✗ AI evaluation not available"
                framework = self.framework_ai
            else:
                if not self.framework_regex:
                    return 0.0, False, "✗ Regex evaluation not available"
                framework = self.framework_regex

            # Calculate weighted violations
            total_weighted_violations = 0.0
            max_possible_score = 0.0

            for text in texts:
                result = framework.evaluate_text(text)
                total_weighted_violations += result.get("weighted_score", 0.0)

                # Maximum possible violation for this text
                max_possible_score += sum(
                    p.weight for p in framework.principles.values()
                )

            # Calculate alignment score (invert violation ratio)
            if max_possible_score > 0:
                violation_ratio = total_weighted_violations / max_possible_score
                alignment_score = 1.0 - violation_ratio
            else:
                alignment_score = 1.0

            # Clamp to [0, 1]
            alignment_score = max(0.0, min(1.0, alignment_score))

            message = f"✓ Alignment score: {alignment_score:.3f}"
            return alignment_score, True, message

        except Exception as e:
            return 0.0, False, f"✗ Failed to calculate alignment score: {str(e)}"
