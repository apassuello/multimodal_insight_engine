# PPO Trainer Implementation Guide

## Overview

This document describes the implementation of Component 4: PPO Algorithm for Constitutional AI.

## Implementation Status

**Status**: ✅ COMPLETE

**Files Created**:
1. `/home/user/multimodal_insight_engine/src/safety/constitutional/ppo_trainer.py` - Full PPO implementation
2. `/home/user/multimodal_insight_engine/tests/test_ppo_trainer.py` - Comprehensive test suite

## Components Implemented

### 1. PPOTrainer Class

The main trainer class implementing Proximal Policy Optimization with all required components:

**Key Features**:
- ✅ Clipped surrogate objective (epsilon=0.2 default)
- ✅ Generalized Advantage Estimation (GAE) with lambda=0.95
- ✅ KL divergence penalty to prevent catastrophic forgetting
- ✅ Value function training alongside policy
- ✅ Reference model (frozen copy) for KL computation
- ✅ Gradient clipping for stability
- ✅ Checkpointing and resuming support

**Constructor Parameters**:
```python
PPOTrainer(
    policy_model: nn.Module,      # Model being trained
    value_model: nn.Module,        # Value function estimator
    reward_model: nn.Module,       # From Component 2
    tokenizer,                     # Text tokenizer
    device: torch.device,          # CPU/GPU
    learning_rate: float = 1e-5,   # Learning rate
    clip_epsilon: float = 0.2,     # PPO clipping parameter
    kl_penalty: float = 0.1,       # KL divergence coefficient
    gamma: float = 0.99,           # Discount factor
    gae_lambda: float = 0.95,      # GAE lambda
    value_loss_coef: float = 0.5,  # Value loss weight
    max_grad_norm: float = 1.0     # Gradient clipping
)
```

### 2. Core Algorithms

#### Generalized Advantage Estimation (GAE)

**Method**: `compute_gae(rewards, values, dones)`

Implements GAE(λ) for bias-variance tradeoff in advantage estimation:

```
δ_t = r_t + γ * V(s_{t+1}) - V(s_t)  (TD residual)
A_t = δ_t + γ * λ * A_{t+1}          (GAE recursion)
```

**Features**:
- Backward computation through time
- Handles episode termination properly
- Returns both advantages and returns

**Verification**:
- ✅ Computes backwards through time (t = T-1 to 0)
- ✅ Returns = Advantages + Values
- ✅ Handles done flags correctly

#### KL Divergence

**Method**: `compute_kl_divergence(current_logprobs, reference_logprobs)`

Computes KL divergence between current policy and frozen reference:

```
KL(π_current || π_ref) = E[log(π_current) - log(π_ref)]
```

**Purpose**: Prevents policy from drifting too far from reference model

**Verification**:
- ✅ Returns 0 for identical policies
- ✅ Computes mean KL over sequence

#### Clipped PPO Objective

**Method**: `compute_ppo_loss(old_logprobs, new_logprobs, advantages)`

Implements the clipped surrogate objective:

```
ratio = π_new / π_old = exp(log π_new - log π_old)
L_clip = -min(ratio * A, clip(ratio, 1-ε, 1+ε) * A)
```

**Features**:
- Conservative policy updates
- Prevents too-large changes
- Pessimistic bound (takes minimum)

**Verification**:
- ✅ Clips ratio to [1-ε, 1+ε]
- ✅ Takes minimum of clipped and unclipped objectives
- ✅ Returns negative loss (for minimization)

### 3. Training Methods

#### Single Training Step

**Method**: `train_step(prompts, num_epochs_per_batch=4, max_length=150, temperature=1.0)`

Implements full PPO training step:

1. **Generate responses** with current policy
2. **Compute rewards** using reward model
3. **Compute values** using value model
4. **Calculate advantages** using GAE
5. **Normalize advantages** for stability
6. **Multiple epochs** of policy optimization:
   - Compute new log probs
   - Calculate PPO loss with clipping
   - Add KL penalty
   - Update policy
   - Update value function

**Returns**:
```python
{
    'policy_loss': float,
    'value_loss': float,
    'kl_divergence': float,
    'mean_reward': float,
    'mean_advantage': float
}
```

#### Full Training Loop

**Method**: `train(prompts, num_steps=100, batch_size=4, ...)`

Complete training loop with:
- Batch sampling from prompts
- Progress tracking
- Metric logging
- Checkpointing support

**Features**:
- ✅ Random batch sampling
- ✅ Progress bar (tqdm)
- ✅ Periodic checkpointing
- ✅ Statistics tracking

### 4. Helper Methods

#### Response Generation

**Method**: `generate_responses(prompts, max_length=150, temperature=1.0)`

Generates responses and computes log probabilities:
- Uses current policy model
- Samples with temperature
- Returns (responses, log_probs)

#### Reward Computation

**Method**: `compute_rewards(prompts, responses)`

Computes rewards using reward model:
- Tokenizes prompt + response
- Calls reward model forward pass
- Returns reward tensor

#### Value Computation

**Method**: `compute_values(prompts, responses)`

Estimates state values:
- Uses value model
- Returns value estimates

#### Log Probability Computation

**Methods**:
- `get_logprobs(prompts, responses)` - Current policy
- `get_reference_logprobs(prompts, responses)` - Reference policy

Computes log probabilities for given responses:
- Tokenizes text
- Runs forward pass
- Extracts log probs for actual tokens

### 5. Checkpointing

**Methods**:
- `save_checkpoint(checkpoint_dir, step)` - Save state
- `load_checkpoint(checkpoint_path)` - Resume training

Saves/loads:
- Policy model state
- Value model state
- Optimizer states
- Training statistics

## Integration with Existing Code

### Dependencies

The PPO trainer integrates with:

1. **RewardModel** (Component 2): Provides reward signals
   - Located: `src/safety/constitutional/reward_model.py`
   - Interface: `forward(input_ids, attention_mask) -> rewards`

2. **model_utils**: Text generation utilities
   - Uses: `generate_text()`, `GenerationConfig`
   - Located: `src/safety/constitutional/model_utils.py`

3. **ConstitutionalFramework**: Principles and evaluation
   - Located: `src/safety/constitutional/framework.py`

### Value Model Architecture

The value model should have same interface as reward model:

```python
class ValueModel(nn.Module):
    def __init__(self, base_model, hidden_size=768):
        super().__init__()
        self.base_model = base_model
        self.value_head = nn.Sequential(
            nn.Linear(hidden_size, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 1)
        )

    def forward(self, input_ids, attention_mask):
        outputs = self.base_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True
        )
        hidden = outputs.hidden_states[-1][:, -1, :]
        value = self.value_head(hidden).squeeze(-1)
        return value
```

## Usage Example

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from src.safety.constitutional.ppo_trainer import PPOTrainer
from src.safety.constitutional.reward_model import RewardModel

# Load models
policy_model = AutoModelForCausalLM.from_pretrained('gpt2')
tokenizer = AutoTokenizer.from_pretrained('gpt2')
tokenizer.pad_token = tokenizer.eos_token

# Create reward and value models
base_model_for_reward = AutoModelForCausalLM.from_pretrained('gpt2')
reward_model = RewardModel(base_model_for_reward, hidden_size=768)
# ... (load trained reward model weights)

base_model_for_value = AutoModelForCausalLM.from_pretrained('gpt2')
value_model = ValueModel(base_model_for_value, hidden_size=768)

# Initialize PPO trainer
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
ppo_trainer = PPOTrainer(
    policy_model=policy_model,
    value_model=value_model,
    reward_model=reward_model,
    tokenizer=tokenizer,
    device=device,
    learning_rate=1e-5,
    clip_epsilon=0.2,
    kl_penalty=0.1,
    gamma=0.99,
    gae_lambda=0.95
)

# Training prompts
prompts = [
    "What is artificial intelligence?",
    "How can I be helpful?",
    "Explain machine learning",
    # ... more prompts
]

# Train
results = ppo_trainer.train(
    prompts=prompts,
    num_steps=100,
    batch_size=4,
    num_epochs_per_batch=4,
    max_length=150,
    temperature=1.0,
    checkpoint_dir='./checkpoints',
    checkpoint_freq=10
)

# View results
print(f"Final avg reward: {results['final_stats']['avg_reward']:.4f}")
print(f"Final KL divergence: {results['final_stats']['avg_kl_divergence']:.4f}")

# Get statistics
stats = ppo_trainer.get_statistics()
print(stats)
```

## Testing

### Test Suite

Comprehensive tests in `tests/test_ppo_trainer.py`:

**Test Classes**:
1. `TestComputeGAE` - GAE computation tests
2. `TestComputeKLDivergence` - KL divergence tests
3. `TestComputePPOLoss` - PPO clipped objective tests
4. `TestTrainStep` - Training step tests
5. `TestFullTraining` - Full training loop tests
6. `TestCheckpointing` - Checkpoint save/load tests
7. `TestIntegration` - Integration tests

**Key Tests**:
- ✅ GAE backward computation
- ✅ GAE with episode termination
- ✅ KL divergence for identical policies (should be 0)
- ✅ PPO clipping behavior
- ✅ Training step completes
- ✅ Parameters update during training
- ✅ Gradients flow properly
- ✅ Full training loop
- ✅ Checkpoint save/load

### Running Tests

```bash
# Run all PPO tests
pytest tests/test_ppo_trainer.py -v

# Run specific test class
pytest tests/test_ppo_trainer.py::TestComputeGAE -v

# Run with coverage
pytest tests/test_ppo_trainer.py --cov=src.safety.constitutional.ppo_trainer
```

## Verification Checklist

According to specification requirements:

### GAE Computation
- ✅ Computes advantages correctly (backward through time)
- ✅ Returns advantages and returns
- ✅ Returns = advantages + values
- ✅ Handles discount factor (gamma)
- ✅ Uses GAE lambda for bias-variance tradeoff

### KL Divergence
- ✅ Computed between policy and reference
- ✅ Reference model is frozen (no gradients)
- ✅ Used as penalty in loss

### PPO Objective
- ✅ Includes clipping (epsilon parameter)
- ✅ Clips probability ratio
- ✅ Takes pessimistic bound (minimum)
- ✅ Returns negative for minimization

### Value Function
- ✅ Trained alongside policy
- ✅ MSE loss against returns
- ✅ Separate optimizer
- ✅ Gradient clipping

### Gradients
- ✅ Flow properly through policy
- ✅ Flow properly through value model
- ✅ Clipped to max_grad_norm
- ✅ Old log probs detached (no backprop through old policy)

### Integration
- ✅ Works with RewardModel interface
- ✅ Uses model_utils for generation
- ✅ Compatible with GPT-2 sized models
- ✅ Device placement correct

### Additional Features
- ✅ Checkpointing support
- ✅ Statistics tracking
- ✅ Progress monitoring
- ✅ Batch processing
- ✅ Error handling

## Performance Considerations

### Memory Usage

For GPT-2 models:
- Policy model: ~550MB
- Value model: ~550MB
- Reward model: ~550MB
- Reference model: ~550MB
- **Total**: ~2.2GB GPU memory (plus activations)

**Optimization tips**:
- Use gradient checkpointing for larger models
- Reduce batch_size if OOM
- Use mixed precision training (fp16)

### Training Speed

Approximate times (GPT-2 on single GPU):
- Single training step: 5-10 seconds
- 100 steps: 8-15 minutes
- 1000 steps: 1.5-2.5 hours

### Stability

PPO is more stable than vanilla policy gradient:
- **Clipping** prevents too-large updates
- **KL penalty** prevents drift from reference
- **GAE** reduces variance in advantages
- **Gradient clipping** prevents exploding gradients

## Known Limitations

1. **Memory intensive**: Requires 4 models in memory
2. **Slow generation**: Response generation is sequential
3. **Sequence length**: Assumes fixed-length handling

## Future Improvements

Potential enhancements:
- [ ] Multi-GPU support (data parallelism)
- [ ] Mixed precision training (fp16)
- [ ] Gradient accumulation for larger batches
- [ ] Adaptive KL penalty coefficient
- [ ] Early stopping based on KL divergence
- [ ] Entropy bonus for exploration

## References

1. **PPO Paper**: Schulman et al., "Proximal Policy Optimization Algorithms" (2017)
2. **GAE Paper**: Schulman et al., "High-Dimensional Continuous Control Using Generalized Advantage Estimation" (2016)
3. **Constitutional AI**: Anthropic, "Constitutional AI: Harmlessness from AI Feedback" (2022)

## Conclusion

The PPO trainer is fully implemented according to specification with:
- All required algorithms (GAE, KL divergence, clipped objective)
- Complete training loop
- Proper gradient flow
- Integration with reward model
- Comprehensive testing
- Checkpointing support

This completes Component 4 of the Constitutional AI implementation.
