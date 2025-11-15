#!/usr/bin/env python3
"""
Phase 2 Implementation Validation Script

Tests:
1. All imports work correctly
2. Managers can be instantiated
3. Demo interface can be created
4. ComparisonEngine is properly integrated
"""

import sys
import traceback

def test_imports():
    """Test that all required imports work."""
    print("Testing imports...")
    try:
        # Core imports
        import gradio as gr
        import torch

        # Manager imports
        from demo.managers import (
            ModelManager,
            ModelStatus,
            EvaluationManager,
            TrainingManager,
            TrainingConfig,
            ComparisonEngine,
            ComparisonResult,
            PrincipleComparison,
            ExampleComparison
        )

        # Data imports
        from demo.data import (
            EVALUATION_EXAMPLES,
            TEST_SUITES,
            TRAINING_CONFIGS,
            get_training_prompts,
            get_adversarial_prompts
        )

        # Main import
        from demo import create_demo

        print("✓ All imports successful")
        return True
    except Exception as e:
        print(f"✗ Import failed: {e}")
        traceback.print_exc()
        return False


def test_manager_instantiation():
    """Test that managers can be instantiated."""
    print("\nTesting manager instantiation...")
    try:
        from demo.managers import (
            ModelManager,
            EvaluationManager,
            TrainingManager,
            ComparisonEngine
        )
        from src.safety.constitutional.principles import setup_default_framework

        # Instantiate managers
        model_manager = ModelManager()
        print("  ✓ ModelManager instantiated")

        evaluation_manager = EvaluationManager()
        print("  ✓ EvaluationManager instantiated")

        training_manager = TrainingManager()
        print("  ✓ TrainingManager instantiated")

        # Create framework for ComparisonEngine
        framework = setup_default_framework()
        comparison_engine = ComparisonEngine(framework)
        print("  ✓ ComparisonEngine instantiated")

        print("✓ All managers instantiated successfully")
        return True
    except Exception as e:
        print(f"✗ Manager instantiation failed: {e}")
        traceback.print_exc()
        return False


def test_demo_creation():
    """Test that demo interface can be created."""
    print("\nTesting demo creation...")
    try:
        from demo import create_demo

        demo = create_demo()
        print("  ✓ Demo interface created")

        # Check that demo is a Gradio Blocks object
        import gradio as gr
        if isinstance(demo, gr.Blocks):
            print("  ✓ Demo is valid Gradio Blocks instance")
        else:
            print(f"  ✗ Demo is not Gradio Blocks: {type(demo)}")
            return False

        print("✓ Demo creation successful")
        return True
    except Exception as e:
        print(f"✗ Demo creation failed: {e}")
        traceback.print_exc()
        return False


def test_test_suites():
    """Test that test suites are properly defined."""
    print("\nTesting test suites...")
    try:
        from demo.data import TEST_SUITES

        expected_suites = [
            "harmful_content",
            "stereotyping",
            "truthfulness",
            "autonomy_manipulation"
        ]

        for suite_name in expected_suites:
            if suite_name not in TEST_SUITES:
                print(f"  ✗ Missing test suite: {suite_name}")
                return False

            prompts = TEST_SUITES[suite_name]
            if not isinstance(prompts, list) or len(prompts) == 0:
                print(f"  ✗ Test suite {suite_name} is invalid")
                return False

            print(f"  ✓ Test suite '{suite_name}': {len(prompts)} prompts")

        print("✓ All test suites valid")
        return True
    except Exception as e:
        print(f"✗ Test suite validation failed: {e}")
        traceback.print_exc()
        return False


def main():
    """Run all validation tests."""
    print("=" * 60)
    print("Phase 2 Implementation Validation")
    print("=" * 60)

    tests = [
        ("Imports", test_imports),
        ("Manager Instantiation", test_manager_instantiation),
        ("Demo Creation", test_demo_creation),
        ("Test Suites", test_test_suites)
    ]

    results = []
    for test_name, test_func in tests:
        result = test_func()
        results.append((test_name, result))

    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)

    for test_name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{status}: {test_name}")

    all_passed = all(result for _, result in results)

    print("=" * 60)
    if all_passed:
        print("✓ ALL TESTS PASSED")
        print("\nPhase 2 implementation is ready for use!")
        print("Run: python demo/main.py")
        return 0
    else:
        print("✗ SOME TESTS FAILED")
        print("\nPlease fix the failing tests before proceeding.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
