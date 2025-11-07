"""MODULE: framework.py
PURPOSE: Core Constitutional AI framework classes for principle-based evaluation
KEY COMPONENTS:
- ConstitutionalPrinciple: Single principle with evaluation logic
- ConstitutionalFramework: Collection of principles for comprehensive evaluation
DEPENDENCIES: typing
SPECIAL NOTES: Foundation for Constitutional AI approach inspired by Anthropic's research
"""

from typing import Callable, Dict, Any, List, Optional


class ConstitutionalPrinciple:
    """
    Representation of a single constitutional principle with evaluation logic.

    A constitutional principle defines a specific aspect of desired AI behavior
    (e.g., harm prevention, truthfulness, fairness) along with logic to evaluate
    whether text adheres to that principle.
    """

    def __init__(
        self,
        name: str,
        description: str,
        evaluation_fn: Callable[[str], Dict[str, Any]],
        weight: float = 1.0,
        enabled: bool = True
    ):
        """
        Initialize a constitutional principle.

        Args:
            name: Unique identifier for the principle
            description: Human-readable description of what the principle checks
            evaluation_fn: Function that evaluates text against this principle
            weight: Importance weight for this principle (default 1.0)
            enabled: Whether this principle is active (default True)
        """
        self.name = name
        self.description = description
        self.evaluation_fn = evaluation_fn
        self.weight = weight
        self.enabled = enabled

    def evaluate(self, text: str) -> Dict[str, Any]:
        """
        Evaluate text against this principle.

        Args:
            text: Text to evaluate

        Returns:
            Dictionary containing evaluation results with at least:
            - flagged: bool indicating if principle was violated
            - Additional details specific to the principle
        """
        if not self.enabled:
            return {
                "flagged": False,
                "reason": "Principle disabled",
                "enabled": False,
                "principle_name": self.name,
                "weight": self.weight
            }

        result = self.evaluation_fn(text)
        result["principle_name"] = self.name
        result["weight"] = self.weight
        return result

    def __repr__(self) -> str:
        """String representation of the principle."""
        status = "enabled" if self.enabled else "disabled"
        return f"ConstitutionalPrinciple(name='{self.name}', weight={self.weight}, {status})"


class ConstitutionalFramework:
    """
    Collection of constitutional principles for comprehensive AI safety evaluation.

    This framework manages multiple constitutional principles and provides
    methods to evaluate text against all principles, track violations, and
    generate reports.
    """

    def __init__(self, name: str = "default_framework"):
        """
        Initialize the constitutional framework.

        Args:
            name: Name for this framework configuration
        """
        self.name = name
        self.principles: Dict[str, ConstitutionalPrinciple] = {}
        self.evaluation_history: List[Dict[str, Any]] = []

    def add_principle(self, principle: ConstitutionalPrinciple) -> None:
        """
        Add a constitutional principle to the framework.

        Args:
            principle: ConstitutionalPrinciple instance to add

        Raises:
            ValueError: If a principle with the same name already exists
        """
        if principle.name in self.principles:
            raise ValueError(f"Principle '{principle.name}' already exists in framework")

        self.principles[principle.name] = principle

    def remove_principle(self, name: str) -> None:
        """
        Remove a principle from the framework.

        Args:
            name: Name of the principle to remove
        """
        if name in self.principles:
            del self.principles[name]

    def enable_principle(self, name: str) -> None:
        """Enable a specific principle."""
        if name in self.principles:
            self.principles[name].enabled = True

    def disable_principle(self, name: str) -> None:
        """Disable a specific principle."""
        if name in self.principles:
            self.principles[name].enabled = False

    def evaluate_text(self, text: str, track_history: bool = False) -> Dict[str, Any]:
        """
        Evaluate text against all constitutional principles.

        Args:
            text: Text to evaluate
            track_history: Whether to add this evaluation to history

        Returns:
            Dictionary containing:
            - principle_results: Dict of results for each principle
            - any_flagged: Whether any principle was violated
            - flagged_principles: List of violated principle names
            - weighted_score: Weighted sum of violations
        """
        principle_results = {}
        flagged_principles = []
        weighted_score = 0.0

        for name, principle in self.principles.items():
            if not principle.enabled:
                continue

            result = principle.evaluate(text)
            principle_results[name] = result

            if result.get("flagged", False):
                flagged_principles.append(name)
                weighted_score += principle.weight

        evaluation = {
            "principle_results": principle_results,
            "any_flagged": len(flagged_principles) > 0,
            "flagged_principles": flagged_principles,
            "weighted_score": weighted_score,
            "num_principles_evaluated": len([p for p in self.principles.values() if p.enabled]),
            "text_length": len(text)
        }

        if track_history:
            self.evaluation_history.append({
                "text": text[:100] + "..." if len(text) > 100 else text,
                "evaluation": evaluation
            })

        return evaluation

    def batch_evaluate(self, texts: List[str]) -> List[Dict[str, Any]]:
        """
        Evaluate multiple texts.

        Args:
            texts: List of texts to evaluate

        Returns:
            List of evaluation results
        """
        return [self.evaluate_text(text) for text in texts]

    def get_statistics(self) -> Dict[str, Any]:
        """
        Get statistics from evaluation history.

        Returns:
            Dictionary with statistics about evaluations
        """
        if not self.evaluation_history:
            return {
                "total_evaluations": 0,
                "total_flagged": 0,
                "flagged_rate": 0.0
            }

        total_evaluations = len(self.evaluation_history)
        total_flagged = sum(
            1 for entry in self.evaluation_history
            if entry["evaluation"]["any_flagged"]
        )

        # Count violations per principle
        principle_violation_counts = {name: 0 for name in self.principles.keys()}
        for entry in self.evaluation_history:
            for principle_name in entry["evaluation"]["flagged_principles"]:
                principle_violation_counts[principle_name] += 1

        return {
            "total_evaluations": total_evaluations,
            "total_flagged": total_flagged,
            "flagged_rate": total_flagged / total_evaluations if total_evaluations > 0 else 0.0,
            "principle_violation_counts": principle_violation_counts,
            "principle_violation_rates": {
                name: count / total_evaluations if total_evaluations > 0 else 0.0
                for name, count in principle_violation_counts.items()
            }
        }

    def clear_history(self) -> None:
        """Clear evaluation history."""
        self.evaluation_history = []

    def get_active_principles(self) -> List[str]:
        """Get list of currently enabled principle names."""
        return [name for name, principle in self.principles.items() if principle.enabled]

    def __repr__(self) -> str:
        """String representation of the framework."""
        num_principles = len(self.principles)
        num_enabled = len(self.get_active_principles())
        return f"ConstitutionalFramework(name='{self.name}', principles={num_enabled}/{num_principles} enabled)"
