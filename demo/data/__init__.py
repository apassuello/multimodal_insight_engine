"""Demo data package - test examples and training prompts."""

from demo.data.test_examples import (
    ADVERSARIAL_PROMPTS,
    EVALUATION_EXAMPLES,
    TEST_SUITES,
    TRAINING_CONFIGS,
    TRAINING_PROMPTS,
    get_adversarial_prompts,
    get_training_prompts,
)


__all__ = [
    "EVALUATION_EXAMPLES",
    "TRAINING_PROMPTS",
    "ADVERSARIAL_PROMPTS",
    "TEST_SUITES",
    "TRAINING_CONFIGS",
    "get_training_prompts",
    "get_adversarial_prompts",
]
