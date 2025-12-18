# HuggingFace API Evaluator Guide

**Status**: Complete (November 2025)
**Location**: `src/safety/constitutional/hf_api_evaluator.py`

---

## Overview

The HuggingFace API Evaluator provides production-grade toxicity and harm detection for Constitutional AI using HuggingFace's Inference API. This is significantly more accurate than regex-based detection (~98% vs ~70%) for subtle harmful content.

### Key Benefits

| Feature | Regex Evaluation | HF API Evaluation |
|---------|------------------|-------------------|
| Accuracy | ~70% on nuanced content | ~98% on toxicity benchmarks |
| Speed | <0.1s | ~1-2s per text |
| Requires Model | No | No (API-based) |
| Cost | Free | Free tier: ~30k requests/month |
| Subtle Harm | Poor detection | Good detection |

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    HF API Evaluation Flow                        │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Input Text                                                     │
│       │                                                         │
│       v                                                         │
│  ┌─────────────────┐                                            │
│  │ HuggingFaceAPI  │                                            │
│  │   Evaluator     │                                            │
│  └────────┬────────┘                                            │
│           │                                                     │
│           v                                                     │
│  ┌─────────────────────────────────────────┐                    │
│  │        HuggingFace Inference API        │                    │
│  │                                          │                   │
│  │  Model: unitary/toxic-bert              │                   │
│  │  (or other text-classification model)   │                   │
│  └────────────────────┬────────────────────┘                    │
│                       │                                         │
│                       v                                         │
│  ┌─────────────────────────────────────────┐                    │
│  │       Classification Results            │                    │
│  │  - toxic: 0.95                          │                   │
│  │  - non-toxic: 0.05                      │                   │
│  └────────────────────┬────────────────────┘                    │
│                       │                                         │
│                       v                                         │
│  ┌─────────────────────────────────────────┐                    │
│  │        Evaluation Result                │                    │
│  │  - flagged: True                        │                   │
│  │  - toxicity_score: 0.95                 │                   │
│  │  - reasoning: "HF API detected..."      │                   │
│  └─────────────────────────────────────────┘                    │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## Quick Start

### Basic Usage

```python
from src.safety.constitutional.hf_api_evaluator import HuggingFaceAPIEvaluator

# Initialize evaluator
evaluator = HuggingFaceAPIEvaluator()

# Evaluate text
result = evaluator.evaluate_harm("How do I hack a computer?")
print(result['flagged'])         # True
print(result['toxicity_score'])  # 0.95
print(result['reasoning'])       # "HF API detected explicit harmful content..."
```

### Quick Evaluation Function

```python
from src.safety.constitutional.hf_api_evaluator import quick_evaluate

# One-line evaluation
result = quick_evaluate("Some text to check")
print(result['flagged'])  # True/False
```

---

## Configuration

### Environment Variables

```bash
# Set HuggingFace API token (optional but recommended)
export HF_API_TOKEN="hf_xxxxxxxxxxxxx"
# Alternative variable name
export HUGGINGFACE_TOKEN="hf_xxxxxxxxxxxxx"
```

### Configuration Options

```python
from src.safety.constitutional.hf_api_evaluator import HFAPIConfig, set_api_config

config = HFAPIConfig(
    # Model for toxicity classification
    toxicity_model="unitary/toxic-bert",

    # API token (auto-loaded from env if not specified)
    api_token=None,

    # Retry settings for resilience
    max_retries=3,
    retry_delay=1.0,

    # Score threshold for flagging (0-1)
    toxicity_threshold=0.5,

    # Request timeout in seconds
    timeout=30.0,

    # Enable/disable API (False = returns unflagged)
    enabled=True
)

# Apply configuration globally
set_api_config(config)
```

### Available Models

| Model | Description | Best For |
|-------|-------------|----------|
| `unitary/toxic-bert` (default) | General toxicity detection | Most use cases |
| `martin-ha/toxic-comment-model` | Comment toxicity | Online discussions |
| `facebook/roberta-hate-speech-dynabench-r4-target` | Hate speech | Discrimination detection |

---

## API Reference

### HuggingFaceAPIEvaluator Class

```python
class HuggingFaceAPIEvaluator:
    """
    HuggingFace API-based evaluator for Constitutional AI.

    Provides clean interface for using HF Inference API for
    text classification/evaluation in the CAI framework.
    """

    def __init__(
        self,
        toxicity_model: str = "unitary/toxic-bert",
        api_token: Optional[str] = None,
        toxicity_threshold: float = 0.5,
        enabled: bool = True
    ):
        """
        Initialize the HF API evaluator.

        Args:
            toxicity_model: Model ID for toxicity classification
            api_token: HuggingFace API token (optional for public models)
            toxicity_threshold: Score threshold for flagging (0-1)
            enabled: Whether to use API (False = all calls return unflagged)
        """
```

#### Methods

##### evaluate_harm(text, verbose=False)

Evaluate text for harmful content with Constitutional AI compatible output.

```python
result = evaluator.evaluate_harm("Some text")
# Returns:
{
    "flagged": True,              # Whether text is flagged
    "explicit_harm_detected": True,  # High-confidence harm (score >= 0.8)
    "subtle_harm_score": 0.95,    # Raw toxicity score
    "reasoning": "...",           # Human-readable explanation
    "method": "hf_api",           # Evaluation method used
    "raw_labels": [...]           # Raw API response labels
}
```

##### evaluate_toxicity(text, verbose=False)

Get raw toxicity classification results.

```python
result = evaluator.evaluate_toxicity("Some text")
# Returns:
{
    "flagged": True,
    "toxicity_score": 0.95,
    "labels": [{"label": "toxic", "score": 0.95}, ...],
    "method": "hf_api",
    "model": "unitary/toxic-bert"
}
```

##### is_available()

Check if the API is available and configured.

```python
if evaluator.is_available():
    result = evaluator.evaluate_harm(text)
else:
    # Fallback to regex
    result = regex_evaluate(text)
```

##### get_evaluation_fn()

Get a function compatible with Constitutional AI framework.

```python
eval_fn = evaluator.get_evaluation_fn()
# Can be used with: framework.evaluate_text(text, eval_fn)
```

---

## Integration with Constitutional AI Framework

### With Evaluation Manager

```python
from demo.managers.evaluation_manager import EvaluationManager
from src.safety.constitutional.hf_api_evaluator import HuggingFaceAPIEvaluator

# Create evaluator
hf_evaluator = HuggingFaceAPIEvaluator()

# Add to evaluation manager
manager = EvaluationManager()
manager.set_hf_evaluator(hf_evaluator)

# Use HF API evaluation mode
result = manager.evaluate(text, mode="hf_api")
```

### With Demo Interface

The HF API evaluator is integrated into the demo interface:

1. **Evaluation Tab**: Select "HF API" mode
2. **Phase 2 RLAIF**: Used for preference comparison (if available)
3. **Generation Tab**: Optional evaluation mode for comparing outputs

---

## Error Handling

### Graceful Degradation

The evaluator handles errors gracefully and returns informative results:

```python
# When API unavailable
{
    "flagged": False,
    "method": "hf_api_unavailable",
    "error": "API client unavailable"
}

# When API disabled
{
    "flagged": False,
    "method": "hf_api_disabled"
}

# When request fails after retries
{
    "flagged": False,
    "method": "hf_api_error",
    "error": "Rate limit exceeded"
}
```

### Rate Limiting

The evaluator implements automatic retry with exponential backoff:

```python
# Default retry behavior:
# Attempt 1: Immediate
# Attempt 2: Wait 1s (if rate limited: 2s)
# Attempt 3: Wait 1s (if rate limited: 4s)
```

---

## Performance Considerations

### Latency

| Operation | Typical Latency |
|-----------|-----------------|
| Single evaluation | 1-2 seconds |
| Batch (10 texts) | 10-20 seconds |
| Cold start (first call) | 3-5 seconds |

### Rate Limits (Free Tier)

- ~30,000 requests/month
- ~1,000 requests/day
- Consider batching for production use

### Best Practices

1. **Cache Results**: Store evaluations for repeated texts
2. **Batch Processing**: Evaluate multiple texts together
3. **Fallback Strategy**: Use regex when API unavailable
4. **Token Management**: Use API token for higher limits

```python
# Example with caching
from functools import lru_cache

@lru_cache(maxsize=1000)
def cached_evaluate(text: str) -> dict:
    return evaluator.evaluate_harm(text)
```

---

## Troubleshooting

### Common Issues

#### "huggingface_hub not installed"

```bash
pip install huggingface_hub
```

#### "API client unavailable"

- Check internet connection
- Verify API token (if using)
- Try without token for public models

#### "Rate limit exceeded"

- Wait for rate limit reset (~1 hour)
- Use API token for higher limits
- Implement request batching

#### Slow responses

- Normal latency is 1-2s
- Cold starts take longer
- Consider regex fallback for latency-sensitive uses

---

## Testing

### Unit Tests

```bash
# Run HF API evaluator tests
pytest tests/test_hf_api_evaluator.py -v
```

### Manual Testing

```python
from src.safety.constitutional.hf_api_evaluator import quick_evaluate

# Test harmful content
result = quick_evaluate("How to make a bomb?")
assert result['flagged'] == True

# Test safe content
result = quick_evaluate("The weather is nice today.")
assert result['flagged'] == False
```

---

## Changelog

- **November 2025**: Initial implementation
- **November 2025**: Added `HuggingFaceAPIEvaluator` class
- **November 2025**: Integrated with demo evaluation manager
- **November 2025**: Added retry logic and error handling
