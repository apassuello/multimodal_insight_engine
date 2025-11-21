"""MODULE: content_logger.py
PURPOSE: Content-visible logging for Constitutional AI pipeline transparency
KEY COMPONENTS:
- ContentLog: Structured log entry with content and metadata
- ContentLogger: Shows actual text at each pipeline stage (not just status)
DEPENDENCIES: typing, dataclasses, time, json
SPECIAL NOTES: Designed to answer "what's actually happening?" not just "what function is running?"
"""

import time
import json
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, asdict

# Maximum number of log entries to prevent unbounded memory growth
MAX_LOGS = 1000


@dataclass
class ContentLog:
    """Single log entry with content visibility."""
    stage: str
    content: str
    metadata: Dict[str, Any]
    timestamp: float


class ContentLogger:
    """
    Logger that shows actual content at each pipeline stage.

    Designed to expose what's really happening inside the Constitutional AI pipeline:
    - What text is being evaluated?
    - What did the AI actually generate?
    - Are critiques meaningful or generic?
    - Are revisions actually different from originals?
    - Is comparison comparing the right things?

    Verbosity levels:
    - 0: Off (no logging)
    - 1: Summary only (final results)
    - 2: Key stages (evaluation, training examples, comparisons) [DEFAULT]
    - 3: Full pipeline (every generation, every critique, every revision)
    """

    def __init__(self, verbosity: int = 2, max_logs: int = MAX_LOGS):
        """
        Initialize content logger.

        Args:
            verbosity: 0=off, 1=summary, 2=key stages (default), 3=full pipeline
            max_logs: Maximum number of logs to keep (default: 1000)
        """
        self.verbosity = verbosity
        self.max_logs = max_logs
        self.logs: List[ContentLog] = []

    def log_stage(
        self,
        stage: str,
        content: str,
        metadata: Optional[Dict[str, Any]] = None,
        truncate: int = 500,
        silent: bool = False
    ):
        """
        Log a pipeline stage with content visibility.

        Args:
            stage: Stage identifier (e.g., "EVAL-INPUT", "CRITIQUE-OUTPUT")
            content: The actual text content (not a status message!)
            metadata: Additional data (scores, timing, flags, etc.)
            truncate: Max characters to display (full content still stored)
            silent: If True, store log but don't print (for avoiding duplicates)
        """
        # Always store (even when verbosity=0 or silent=True)
        self.logs.append(ContentLog(
            stage=stage,
            content=content,
            metadata=metadata or {},
            timestamp=time.time()
        ))

        # Trim oldest entries if we exceed max_logs to prevent memory growth
        if len(self.logs) > self.max_logs:
            excess = len(self.logs) - self.max_logs
            self.logs = self.logs[excess:]

        # Skip display if silent or verbosity is 0
        if silent or self.verbosity == 0:
            return

        # Display with formatting (may be truncated)
        separator = "=" * 60
        print(f"\n[{stage}] {separator}")

        # Truncate for display if needed
        if len(content) > truncate:
            display_content = content[:truncate] + f"... (truncated, {len(content)} total chars)"
        else:
            display_content = content

        print(display_content)

        # Show metadata if verbosity >= 2
        if metadata and self.verbosity >= 2:
            print(f"\nMetadata:")
            for key, value in metadata.items():
                # Truncate long metadata values
                if isinstance(value, str) and len(value) > 100:
                    print(f"  {key}: {value[:100]}...")
                elif isinstance(value, (list, dict)) and len(str(value)) > 100:
                    print(f"  {key}: {str(value)[:100]}...")
                else:
                    print(f"  {key}: {value}")

    def log_comparison(
        self,
        label1: str,
        text1: str,
        label2: str,
        text2: str,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """
        Log side-by-side comparison of two texts.

        Args:
            label1: Label for first text (e.g., "BASE MODEL")
            text1: First text content
            label2: Label for second text (e.g., "TRAINED MODEL")
            text2: Second text content
            metadata: Comparison metrics (scores, improvement, etc.)
        """
        if self.verbosity == 0:
            return

        separator = "=" * 60
        print(f"\n[COMPARISON] {separator}")

        # Show both texts side-by-side (truncated for display)
        print(f"\n{label1}:")
        print(f"  {text1[:250]}")
        if len(text1) > 250:
            print(f"  ... ({len(text1)} total chars)")

        print(f"\n{label2}:")
        print(f"  {text2[:250]}")
        if len(text2) > 250:
            print(f"  ... ({len(text2)} total chars)")

        # Show comparison metrics
        if metadata:
            print(f"\nComparison Metrics:")
            for key, value in metadata.items():
                print(f"  {key}: {value}")

        # Highlight key differences
        self._highlight_differences(text1, text2)

        # Store in logs
        self.logs.append(ContentLog(
            stage="COMPARISON",
            content=f"{label1}: {text1}\n\n{label2}: {text2}",
            metadata=metadata or {},
            timestamp=time.time()
        ))
        
        # Trim oldest entries if we exceed max_logs to prevent memory growth
        if len(self.logs) > self.max_logs:
            excess = len(self.logs) - self.max_logs
            self.logs = self.logs[excess:]

    def _highlight_differences(self, text1: str, text2: str):
        """
        Automatically highlight common improvements between texts.

        Detects patterns like:
        - Prescriptive → Suggestive ("should" → "consider")
        - Mandatory → Optional ("must" → "might")
        - Absolute → Qualified ("all" → "some")
        """
        text1_lower = text1.lower()
        text2_lower = text2.lower()

        differences_found = False

        print(f"\n[DIFF] Auto-detected improvements:")

        # Prescriptive to suggestive
        if "should" in text1_lower and "consider" in text2_lower:
            print("  ✓ Changed prescriptive ('should') to suggestive ('consider')")
            differences_found = True

        # Mandatory to optional
        if "must" in text1_lower and ("might" in text2_lower or "could" in text2_lower):
            print("  ✓ Changed mandatory ('must') to optional ('might'/'could')")
            differences_found = True

        # Absolute to qualified
        if "all" in text1_lower and ("some" in text2_lower or "many" in text2_lower):
            print("  ✓ Changed absolute ('all') to qualified ('some'/'many')")
            differences_found = True

        # Negative to positive framing
        if "never" in text1_lower and "sometimes" in text2_lower:
            print("  ✓ Changed absolute negative ('never') to conditional")
            differences_found = True

        # Removed coercive urgency
        if "right now" in text1_lower and "right now" not in text2_lower:
            print("  ✓ Removed urgent/coercive language ('right now')")
            differences_found = True

        if "you'll regret" in text1_lower and "regret" not in text2_lower:
            print("  ✓ Removed fear-based manipulation ('you'll regret')")
            differences_found = True

        if not differences_found:
            print("  (No automatic patterns detected)")

    def get_summary(self) -> str:
        """
        Get summary of logged activity.

        Returns:
            String summary of pipeline activity
        """
        if not self.logs:
            return "No activity logged"

        # Count entries by stage type
        stages = {}
        for log in self.logs:
            # Extract stage category (before first hyphen)
            stage_type = log.stage.split('-')[0] if '-' in log.stage else log.stage
            stages[stage_type] = stages.get(stage_type, 0) + 1

        summary = "=" * 60 + "\n"
        summary += "CONTENT LOGGING SUMMARY\n"
        summary += "=" * 60 + "\n"
        summary += f"Total entries: {len(self.logs)}\n\n"
        summary += "Activity by stage:\n"

        for stage, count in sorted(stages.items()):
            summary += f"  {stage}: {count} entries\n"

        summary += "=" * 60

        return summary

    def export_logs(self, filepath: str):
        """
        Export full logs to JSON file for analysis.

        Args:
            filepath: Path to save logs (will create parent directories if needed)
        """
        from pathlib import Path

        # Create parent directories if needed
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)

        # Convert logs to JSON-serializable format
        logs_data = []
        for log in self.logs:
            logs_data.append({
                "stage": log.stage,
                "content": log.content,
                "metadata": log.metadata,
                "timestamp": log.timestamp
            })

        # Write to file with error handling for non-serializable objects
        try:
            with open(filepath, 'w') as f:
                json.dump(logs_data, f, indent=2, default=str)
        except (TypeError, ValueError) as e:
            # If serialization fails, try again with more aggressive fallback
            print(f"Warning: JSON serialization issue: {e}")
            with open(filepath, 'w') as f:
                # Convert all values to strings as last resort
                safe_logs_data = []
                for log in logs_data:
                    safe_log = {
                        "stage": str(log["stage"]),
                        "content": str(log["content"]),
                        "metadata": {k: str(v) for k, v in log["metadata"].items()},
                        "timestamp": log["timestamp"]
                    }
                    safe_logs_data.append(safe_log)
                json.dump(safe_logs_data, f, indent=2)

        print(f"\n{'='*60}")
        print(f"✓ Logs exported to: {filepath}")
        print(f"  Total entries: {len(self.logs)}")
        print(f"  File size: {Path(filepath).stat().st_size:,} bytes")
        print(f"{'='*60}")

    def clear_logs(self):
        """Clear all logged entries."""
        self.logs.clear()
        if self.verbosity > 0:
            print("\n[LOGGER] Logs cleared")

    def set_verbosity(self, level: int):
        """
        Change verbosity level.

        Args:
            level: 0=off, 1=summary, 2=key stages, 3=full pipeline
        """
        if level < 0 or level > 3:
            raise ValueError("Verbosity must be 0-3")

        self.verbosity = level
        if level > 0:
            print(f"\n[LOGGER] Verbosity set to {level}")
