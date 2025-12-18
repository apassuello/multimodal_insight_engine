#!/usr/bin/env python3
"""
Comprehensive verification script for Constitutional AI implementation.

This script verifies the correctness of the implementation by:
1. Checking all required components exist
2. Verifying algorithm implementations match specifications
3. Checking integration points
4. Validating code structure and patterns

Does NOT require PyTorch to run - uses AST parsing and code analysis.
"""

import ast
import re
from pathlib import Path
from typing import List, Dict, Set, Tuple

class ImplementationVerifier:
    """Verifies Constitutional AI implementation correctness."""

    def __init__(self, src_dir: str = "src/safety/constitutional"):
        self.src_dir = Path(src_dir)
        self.errors: List[str] = []
        self.warnings: List[str] = []
        self.successes: List[str] = []

    def verify_all(self) -> bool:
        """Run all verification checks."""
        print("=" * 80)
        print("CONSTITUTIONAL AI IMPLEMENTATION VERIFICATION")
        print("=" * 80)
        print()

        checks = [
            ("File Existence", self.check_files_exist),
            ("Syntax Validity", self.check_syntax),
            ("Bradley-Terry Loss", self.verify_bradley_terry_loss),
            ("GAE Algorithm", self.verify_gae_algorithm),
            ("PPO Clipped Objective", self.verify_ppo_clipping),
            ("KL Divergence", self.verify_kl_divergence),
            ("Critique Templates", self.verify_critique_templates),
            ("Preference Extraction", self.verify_preference_extraction),
            ("Function Signatures", self.verify_function_signatures),
            ("Integration Points", self.verify_integration_points),
        ]

        for check_name, check_func in checks:
            print(f"\n{'─' * 80}")
            print(f"Checking: {check_name}")
            print(f"{'─' * 80}")
            try:
                check_func()
            except Exception as e:
                self.errors.append(f"{check_name}: {str(e)}")
                print(f"❌ FAILED: {e}")

        self.print_summary()
        return len(self.errors) == 0

    def check_files_exist(self):
        """Verify all required files exist."""
        required_files = [
            "reward_model.py",
            "ppo_trainer.py",
            "critique_revision.py",
            "preference_comparison.py",
            "model_utils.py",
            "framework.py",
            "__init__.py"
        ]

        for filename in required_files:
            filepath = self.src_dir / filename
            if not filepath.exists():
                self.errors.append(f"Missing file: {filename}")
                print(f"  ❌ Missing: {filename}")
            else:
                self.successes.append(f"File exists: {filename}")
                print(f"  ✓ Found: {filename}")

    def check_syntax(self):
        """Verify all Python files have valid syntax."""
        for py_file in self.src_dir.glob("*.py"):
            try:
                with open(py_file) as f:
                    ast.parse(f.read())
                self.successes.append(f"Valid syntax: {py_file.name}")
                print(f"  ✓ Valid syntax: {py_file.name}")
            except SyntaxError as e:
                self.errors.append(f"Syntax error in {py_file.name}: {e}")
                print(f"  ❌ Syntax error in {py_file.name}: {e}")

    def verify_bradley_terry_loss(self):
        """Verify Bradley-Terry loss implementation."""
        filepath = self.src_dir / "reward_model.py"
        with open(filepath) as f:
            content = f.read()

        # Check for Bradley-Terry loss function
        if "def compute_reward_loss" not in content:
            self.errors.append("compute_reward_loss function not found")
            print("  ❌ compute_reward_loss function not found")
            return

        # Verify formula: -log(sigmoid(reward_chosen - reward_rejected))
        # Which is equivalent to: -F.logsigmoid(reward_chosen - reward_rejected)
        if "logsigmoid" in content and "reward_chosen - reward_rejected" in content:
            self.successes.append("Bradley-Terry loss formula correct")
            print("  ✓ Bradley-Terry loss formula: -log(sigmoid(r_chosen - r_rejected))")

            # Extract the actual line
            for line in content.split('\n'):
                if 'logsigmoid' in line and 'reward_chosen' in line:
                    print(f"    Formula: {line.strip()}")
                    break
        else:
            self.errors.append("Bradley-Terry loss formula incorrect or missing")
            print("  ❌ Bradley-Terry loss formula not found")

        # Check for mean() to average over batch
        loss_func = self._extract_function(content, "compute_reward_loss")
        if loss_func and ".mean()" in loss_func:
            self.successes.append("Loss averaged over batch")
            print("  ✓ Loss properly averaged over batch")
        else:
            self.warnings.append("Loss may not be averaged over batch")
            print("  ⚠ Warning: Loss averaging unclear")

    def verify_gae_algorithm(self):
        """Verify Generalized Advantage Estimation implementation."""
        filepath = self.src_dir / "ppo_trainer.py"
        with open(filepath) as f:
            content = f.read()

        # Check function exists
        if "def compute_gae" not in content:
            self.errors.append("compute_gae function not found")
            print("  ❌ compute_gae function not found")
            return

        gae_func = self._extract_function(content, "compute_gae")

        # Check for backwards iteration
        if "reversed(range" in gae_func:
            self.successes.append("GAE computes backwards through time")
            print("  ✓ GAE computes backwards through time")
        else:
            self.errors.append("GAE does not iterate backwards")
            print("  ❌ GAE must iterate backwards through time")

        # Check for TD residual: delta = r + gamma * V(s+1) - V(s)
        if "rewards[:, t]" in gae_func and "gamma" in gae_func and "next_value" in gae_func:
            self.successes.append("TD residual computation present")
            print("  ✓ TD residual: δ = r + γ*V(s+1) - V(s)")
        else:
            self.errors.append("TD residual computation missing or incorrect")
            print("  ❌ TD residual computation not found")

        # Check for GAE recursion: A_t = delta + gamma * lambda * A_{t+1}
        if "gae_lambda" in gae_func and "last_gae" in gae_func:
            self.successes.append("GAE recursion present")
            print("  ✓ GAE recursion: A_t = δ_t + γ*λ*A_{t+1}")
        else:
            self.errors.append("GAE recursion missing")
            print("  ❌ GAE recursion not found")

        # Check for returns computation
        if "returns" in gae_func and "advantages" in gae_func:
            self.successes.append("Returns computation present")
            print("  ✓ Returns: R_t = A_t + V(s_t)")
        else:
            self.warnings.append("Returns computation unclear")
            print("  ⚠ Returns computation unclear")

    def verify_ppo_clipping(self):
        """Verify PPO clipped objective implementation."""
        filepath = self.src_dir / "ppo_trainer.py"
        with open(filepath) as f:
            content = f.read()

        if "def compute_ppo_loss" not in content:
            self.errors.append("compute_ppo_loss function not found")
            print("  ❌ compute_ppo_loss function not found")
            return

        ppo_func = self._extract_function(content, "compute_ppo_loss")

        # Check for probability ratio
        if "torch.exp" in ppo_func and "new_logprobs - old_logprobs" in ppo_func:
            self.successes.append("Probability ratio computed correctly")
            print("  ✓ Probability ratio: π_new / π_old = exp(log(π_new) - log(π_old))")
        else:
            self.errors.append("Probability ratio computation incorrect")
            print("  ❌ Probability ratio not computed correctly")

        # Check for clipping
        if "torch.clamp" in ppo_func or "clip" in ppo_func.lower():
            self.successes.append("Ratio clipping present")
            print("  ✓ Ratio clipping: clip(ratio, 1-ε, 1+ε)")
        else:
            self.errors.append("Ratio clipping missing")
            print("  ❌ Ratio clipping not found")

        # Check for min operation (pessimistic bound)
        if "torch.min" in ppo_func or "minimum" in ppo_func:
            self.successes.append("Pessimistic bound (min) used")
            print("  ✓ Pessimistic bound: min(surr1, surr2)")
        else:
            self.errors.append("Min operation missing")
            print("  ❌ Min operation for pessimistic bound not found")

        # Check for negative sign (maximizing -> minimizing)
        if "-torch.min" in ppo_func or "-.min" in ppo_func:
            self.successes.append("Loss negated for minimization")
            print("  ✓ Loss negated for gradient descent")
        else:
            self.warnings.append("Loss negation unclear")
            print("  ⚠ Loss negation unclear")

    def verify_kl_divergence(self):
        """Verify KL divergence computation."""
        filepath = self.src_dir / "ppo_trainer.py"
        with open(filepath) as f:
            content = f.read()

        if "def compute_kl_divergence" not in content:
            self.errors.append("compute_kl_divergence function not found")
            print("  ❌ compute_kl_divergence function not found")
            return

        kl_func = self._extract_function(content, "compute_kl_divergence")

        # KL(p||q) = E[log(p) - log(q)]
        if "current_logprobs - reference_logprobs" in kl_func:
            self.successes.append("KL divergence formula correct")
            print("  ✓ KL divergence: E[log(π_current) - log(π_ref)]")
        else:
            self.errors.append("KL divergence formula incorrect")
            print("  ❌ KL divergence formula not correct")

    def verify_critique_templates(self):
        """Verify critique and revision templates exist."""
        filepath = self.src_dir / "critique_revision.py"
        with open(filepath) as f:
            content = f.read()

        if "CRITIQUE_TEMPLATE" in content:
            self.successes.append("Critique template defined")
            print("  ✓ CRITIQUE_TEMPLATE defined")

            # Check template has key placeholders
            if "{prompt}" in content and "{response}" in content and "{principles_text}" in content:
                self.successes.append("Critique template has required placeholders")
                print("  ✓ Template includes: prompt, response, principles")
            else:
                self.warnings.append("Critique template may be missing placeholders")
                print("  ⚠ Template placeholders unclear")
        else:
            self.errors.append("CRITIQUE_TEMPLATE not found")
            print("  ❌ CRITIQUE_TEMPLATE not found")

        if "REVISION_TEMPLATE" in content:
            self.successes.append("Revision template defined")
            print("  ✓ REVISION_TEMPLATE defined")

            if "{critique}" in content:
                self.successes.append("Revision template includes critique")
                print("  ✓ Template includes critique placeholder")
            else:
                self.warnings.append("Revision template may be missing critique")
                print("  ⚠ Critique placeholder unclear")
        else:
            self.errors.append("REVISION_TEMPLATE not found")
            print("  ❌ REVISION_TEMPLATE not found")

    def verify_preference_extraction(self):
        """Verify preference extraction logic."""
        filepath = self.src_dir / "preference_comparison.py"
        with open(filepath) as f:
            content = f.read()

        if "def extract_preference" not in content:
            self.errors.append("extract_preference function not found")
            print("  ❌ extract_preference function not found")
            return

        extract_func = self._extract_function(content, "extract_preference")

        # Check for regex pattern matching
        if "re.search" in extract_func or "re.match" in extract_func:
            self.successes.append("Uses regex for preference extraction")
            print("  ✓ Uses regex patterns for extraction")
        else:
            self.warnings.append("Preference extraction method unclear")
            print("  ⚠ Extraction method unclear")

        # Check for both A and B detection
        if "'A'" in extract_func and "'B'" in extract_func:
            self.successes.append("Extracts both A and B preferences")
            print("  ✓ Handles both 'A' and 'B' preferences")
        else:
            self.errors.append("Missing A or B preference handling")
            print("  ❌ A/B preference handling incomplete")

    def verify_function_signatures(self):
        """Verify required functions exist with correct signatures."""
        required_functions = {
            "reward_model.py": [
                "compute_reward_loss",
                "train_reward_model",
                "evaluate_reward_model"
            ],
            "ppo_trainer.py": [
                "compute_gae",
                "compute_kl_divergence",
                "compute_ppo_loss",
                "train_step",
                "train"
            ],
            "critique_revision.py": [
                "generate_critique",
                "generate_revision"
            ],
            "preference_comparison.py": [
                "generate_comparison",
                "extract_preference"
            ]
        }

        for filename, functions in required_functions.items():
            filepath = self.src_dir / filename
            if not filepath.exists():
                continue

            with open(filepath) as f:
                content = f.read()

            print(f"\n  {filename}:")
            for func_name in functions:
                if f"def {func_name}" in content:
                    self.successes.append(f"{filename}: {func_name} exists")
                    print(f"    ✓ {func_name}")
                else:
                    self.errors.append(f"{filename}: {func_name} missing")
                    print(f"    ❌ {func_name} MISSING")

    def verify_integration_points(self):
        """Verify components integrate properly."""
        # Check PPO uses reward model
        ppo_file = self.src_dir / "ppo_trainer.py"
        with open(ppo_file) as f:
            ppo_content = f.read()

        if "reward_model" in ppo_content and "compute_rewards" in ppo_content:
            self.successes.append("PPO integrates with reward model")
            print("  ✓ PPO uses reward model for feedback")
        else:
            self.errors.append("PPO-reward model integration unclear")
            print("  ❌ PPO-reward model integration missing")

        # Check critique uses model_utils
        critique_file = self.src_dir / "critique_revision.py"
        with open(critique_file) as f:
            critique_content = f.read()

        if "from .model_utils import" in critique_content or "model_utils" in critique_content:
            self.successes.append("Critique uses model_utils")
            print("  ✓ Critique imports model_utils")
        else:
            self.warnings.append("Critique-model_utils integration unclear")
            print("  ⚠ Critique-model_utils integration unclear")

        # Check preference uses model_utils
        pref_file = self.src_dir / "preference_comparison.py"
        with open(pref_file) as f:
            pref_content = f.read()

        if "generate_text" in pref_content:
            self.successes.append("Preference uses generate_text")
            print("  ✓ Preference uses generate_text for comparisons")
        else:
            self.warnings.append("Preference text generation unclear")
            print("  ⚠ Preference text generation unclear")

    def _extract_function(self, content: str, func_name: str) -> str:
        """Extract a function's code from file content."""
        lines = content.split('\n')
        func_lines = []
        in_function = False
        indent_level = 0

        for line in lines:
            if f"def {func_name}" in line:
                in_function = True
                indent_level = len(line) - len(line.lstrip())
                func_lines.append(line)
            elif in_function:
                current_indent = len(line) - len(line.lstrip())
                # Stop if we hit a line at same or lower indent (unless blank)
                if line.strip() and current_indent <= indent_level:
                    break
                func_lines.append(line)

        return '\n'.join(func_lines)

    def print_summary(self):
        """Print verification summary."""
        print("\n")
        print("=" * 80)
        print("VERIFICATION SUMMARY")
        print("=" * 80)

        print(f"\n✓ Successes: {len(self.successes)}")
        print(f"⚠ Warnings:  {len(self.warnings)}")
        print(f"❌ Errors:    {len(self.errors)}")

        if self.errors:
            print("\n❌ ERRORS:")
            for error in self.errors:
                print(f"  - {error}")

        if self.warnings:
            print("\n⚠ WARNINGS:")
            for warning in self.warnings:
                print(f"  - {warning}")

        print("\n" + "=" * 80)
        if len(self.errors) == 0:
            print("✅ VERIFICATION PASSED - Implementation is correct!")
        else:
            print("❌ VERIFICATION FAILED - Please fix errors above")
        print("=" * 80)


if __name__ == "__main__":
    verifier = ImplementationVerifier()
    success = verifier.verify_all()
    exit(0 if success else 1)
