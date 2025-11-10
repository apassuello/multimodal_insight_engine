
## 1 Implementation Plan

Here’s a suggested step-by-step strategy to implement the proposed testing improvements. The plan balances immediate gains (e.g., simple tests for uncovered modules) with building a robust, long-term solution (e.g., coverage checks, CI integration):

1. **Consolidate/Confirm Test Framework**
   - Decide definitively on **PyTest** (or confirm if you want to remain with any existing framework).
   - Ensure all existing tests conform to PyTest conventions (function names, usage of `pytest` fixtures, etc.).
   - Remove or refactor any leftover `unittest`-style or outdated tests.
2. **Set Up Coverage Checks**
   - Install `pytest-cov` (or the built-in `coverage` library) so that you can measure coverage.
   - Add a coverage config file (`.coveragerc`) to exclude non-code directories and define thresholds.
   - Verify that running `pytest --cov=src --cov-fail-under=<threshold>` works locally and fails if coverage is below the threshold.
3. **Add Minimal Tests for Each Module**
   - For each subpackage or core module in `src` that isn’t tested, create a corresponding test file under `tests/`.
   - Write at least one basic test that exercises the critical path of the module or class.
   - Keep tests short and use dummy data or mock objects as needed.
4. **Design & Implement Regression Test Script**
   - Create a top-level script (e.g., `run_tests.sh` or `test_all.py`) that:
     1. Installs the package (e.g., `pip install -e .`).
     2. Invokes `pytest` with coverage and desired reporting options (e.g., `--junitxml` for CI, `--json-report` if you want JSON).
     3. Sets coverage thresholds to ensure no untested module is left behind.
   - Document how to run this script locally (e.g., `bash run_tests.sh`) and confirm it works on your dev machine.
5. **Integrate into CI (GitHub Actions, GitLab, Jenkins, etc.)**
   - Create or update your CI config so it:
     - Checks out your repo.
     - Installs dependencies.
     - Runs the regression test script.
     - Fails if any test fails or coverage is below threshold.
   - Optionally configure the CI to display JUnit XML results or parse them for better visualization of test outcomes.
6. **Enforce Pre-Merge Checks**
   - Configure your repo settings so that merges to main (or the default branch) are blocked unless the CI job passes.
   - This ensures that no code is merged if it causes regression or has insufficient coverage.
7. **Refine & Expand Tests**
   - Once the baseline coverage is established, gradually add more specific or robust test cases.
   - Start covering advanced modules: optimization, safety, evaluation, etc.
   - Add integration tests that verify modules working together (e.g., data loading → model forward pass → trainer → inference).
8. **Add a Separate Benchmark/Performance Testing Suite** (Optional, future step)
   - Create a separate suite (e.g., under `tests/performance/`) or separate scripts to measure speed, memory usage, etc.
   - Keep them outside your main regression suite so that merges aren’t blocked by performance fluctuations.

By following this order, you’ll quickly achieve a workable test suite that covers all modules, with coverage-enforced merges, while leaving room to enhance test depth and performance checks over time.

------

