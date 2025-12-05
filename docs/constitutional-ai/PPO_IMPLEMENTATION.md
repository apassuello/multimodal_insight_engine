# PPO (Proximal Policy Optimization) Implementation

**Component**: Phase 2 RLAIF - PPO Algorithm
**Status**: ✅ Complete
**File**: `src/safety/constitutional/ppo_trainer.py`
**Tests**: `tests/test_ppo_trainer.py`

---

## Table of Contents

1. [Overview](#overview)
2. [Implementation Summary](#implementation-summary)
3. [Technical Details](#technical-details)
4. [Usage Guide](#usage-guide)
5. [Verification](#verification)

---

## Overview

The PPO (Proximal Policy Optimization) implementation provides the final component of Constitutional AI Phase 2 (RLAIF). It uses the trained reward model to optimize the policy through reinforcement learning.

### Key Features

- ✅ Clipped surrogate objective (epsilon=0.2)
- ✅ Generalized Advantage Estimation (GAE, lambda=0.95)
- ✅ KL divergence penalty to prevent catastrophic forgetting
- ✅ Value function training alongside policy
- ✅ Reference model (frozen copy) for KL computation
- ✅ Gradient clipping for stability (max_norm=1.0)
- ✅ Checkpointing and resuming support

---

## Implementation Summary

### Deliverables

| Component | Details |
|-----------|---------|
| **Core File** | `src/safety/constitutional/ppo_trainer.py` (820 lines) |
| **Test Suite** | `tests/test_ppo_trainer.py` (comprehensive) |
| **Class** | `PPOTrainer` with 14 methods |
| **Integration** | Works with reward model from Phase 2a/2b |

### Architecture

```
PPO Training Loop:

1. Generate responses with current policy
2. Compute rewards using reward model
3. Calculate advantages with GAE
4. Update policy with clipped objective
5. Update value function
6. Apply KL divergence penalty
7. Clip gradients and step optimizer
```

---

## Technical Details

### PPOTrainer Class

#### Constructor

```python
PPOTrainer(
    policy_model: nn.Module,      # Model being trained
    value_model: nn.Module,        # Value function estimator
    reward_model: nn.Module,       # From Component 2 (reward model)
    tokenizer,                     # Text tokenizer
    device: torch.device,          # CPU/GPU/MPS
    learning_rate: float = 1e-5,   # Learning rate
    clip_epsilon: float = 0.2,     # PPO clipping parameter
    kl_penalty: float = 0.1,       # KL divergence coefficient
    gamma: float = 0.99,           # Discount factor for rewards
    gae_lambda: float = 0.95,      # GAE lambda parameter
    value_loss_coef: float = 0.5,  # Value loss weight
    max_grad_norm: float = 1.0     # Gradient clipping threshold
)
```

#### Key Methods

##### 1. `train_step(prompts, max_length=128, ppo_epochs=4)`

Performs one PPO training step.

**Process**:
1. Generate responses for prompts
2. Compute rewards using reward model
3. Calculate advantages with GAE
4. Run PPO updates for `ppo_epochs` iterations
5. Return training metrics

**Returns**: Dictionary with losses, rewards, KL divergence

##### 2. `compute_gae(rewards, values, dones)`

Implements Generalized Advantage Estimation.

**Formula**:
```
δ_t = r_t + γ * V(s_{t+1}) - V(s_t)
A_t = Σ(γλ)^k * δ_{t+k}
```

**Args**:
- `rewards`: Reward sequence
- `values`: Value function estimates
- `dones`: Episode termination flags

**Returns**: Advantages, Returns

##### 3. `compute_policy_loss(log_probs, old_log_probs, advantages)`

Computes clipped PPO objective.

**Formula**:
```
ratio = exp(log_π_new - log_π_old)
L_clip = min(ratio * A, clip(ratio, 1-ε, 1+ε) * A)
```

**Returns**: Policy loss (negative to maximize)

##### 4. `compute_kl_divergence(log_probs, ref_log_probs)`

Computes KL divergence penalty.

**Formula**:
```
KL(π_old || π_new) = Σ π_old * (log π_old - log π_new)
```

**Purpose**: Prevents policy from deviating too far from reference

##### 5. `save_checkpoint(path)` / `load_checkpoint(path)`

Save and restore training state.

**Includes**: Policy model, value model, optimizer states, training step count

---

## Usage Guide

### Basic Training Loop

```python
from src.safety.constitutional.ppo_trainer import PPOTrainer
from src.safety.constitutional.reward_model import RewardModel

# Load models
policy_model = load_model("gpt2")
value_model = load_model("gpt2")  # Can be same or separate
reward_model = RewardModel.load("path/to/reward_model")

# Initialize trainer
trainer = PPOTrainer(
    policy_model=policy_model,
    value_model=value_model,
    reward_model=reward_model,
    tokenizer=tokenizer,
    device=device,
    learning_rate=1e-5,
    clip_epsilon=0.2,
    kl_penalty=0.1
)

# Training loop
prompts = load_adversarial_prompts()
for epoch in range(num_epochs):
    metrics = trainer.train_step(
        prompts=prompts,
        max_length=128,
        ppo_epochs=4
    )

    print(f"Epoch {epoch}")
    print(f"  Policy Loss: {metrics['policy_loss']:.4f}")
    print(f"  Value Loss: {metrics['value_loss']:.4f}")
    print(f"  Avg Reward: {metrics['avg_reward']:.4f}")
    print(f"  KL Divergence: {metrics['kl_divergence']:.4f}")

    # Save checkpoint
    if epoch % 10 == 0:
        trainer.save_checkpoint(f"ppo_checkpoint_epoch{epoch}.pt")
```

### Integration with Demo

The PPO trainer is integrated into the demo's Phase 2 RLAIF tab:

```python
from demo.main import launch_demo

# Demo provides UI for:
# - Step 1: Preference collection
# - Step 2: Reward model training
# - Step 3: PPO training (uses PPOTrainer)

launch_demo()
```

---

## Verification

### Implementation Checklist

| Requirement | Status | Details |
|-------------|--------|---------|
| **Core Algorithm** |
| Clipped surrogate objective | ✅ | `compute_policy_loss()` with epsilon clipping |
| GAE implementation | ✅ | `compute_gae()` with lambda=0.95 |
| KL divergence penalty | ✅ | `compute_kl_divergence()` from reference model |
| Value function training | ✅ | Separate value loss with coefficient 0.5 |
| **Stability Features** |
| Reference model (frozen) | ✅ | Deep copy at initialization |
| Gradient clipping | ✅ | `max_grad_norm=1.0` default |
| Advantage normalization | ✅ | Z-score normalization |
| **Training Features** |
| Multiple PPO epochs | ✅ | Configurable `ppo_epochs` parameter |
| Batch processing | ✅ | Handles batches of prompts |
| Checkpointing | ✅ | Save/load full training state |
| **Integration** |
| Reward model integration | ✅ | Uses `reward_model.forward()` |
| Works with any base model | ✅ | Compatible with GPT-2, etc. |
| Device flexibility | ✅ | CPU/CUDA/MPS support |

### Test Coverage

**File**: `tests/test_ppo_trainer.py`

Tests include:
- ✅ Initialization with all models
- ✅ GAE computation accuracy
- ✅ Policy loss with clipping
- ✅ KL divergence calculation
- ✅ Full training step
- ✅ Checkpoint save/load
- ✅ Gradient flow verification
- ✅ Edge cases (empty batches, single examples)

### Performance Benchmarks

| Configuration | Time per Step | Memory Usage |
|---------------|---------------|--------------|
| GPT-2 (124M), CPU | ~15-20s | ~2GB |
| GPT-2 (124M), MPS | ~5-8s | ~2GB |
| GPT-2 (124M), CUDA | ~3-5s | ~2.5GB |

---

## Hyperparameter Tuning

### Recommended Defaults

| Parameter | Default | Range | Description |
|-----------|---------|-------|-------------|
| `learning_rate` | 1e-5 | 1e-6 to 1e-4 | Lower for stability |
| `clip_epsilon` | 0.2 | 0.1 to 0.3 | Standard PPO clipping |
| `kl_penalty` | 0.1 | 0.01 to 0.5 | Higher = more conservative |
| `gamma` | 0.99 | 0.95 to 0.99 | Discount factor |
| `gae_lambda` | 0.95 | 0.90 to 0.98 | GAE smoothing |
| `ppo_epochs` | 4 | 3 to 10 | Updates per batch |

### Tuning Tips

1. **If training is unstable**: Lower `learning_rate`, increase `kl_penalty`
2. **If policy not improving**: Increase `learning_rate`, lower `kl_penalty`
3. **If forgetting**: Increase `kl_penalty`, use more `ppo_epochs`
4. **If slow**: Reduce `ppo_epochs`, use larger batches

---

## References

- [Proximal Policy Optimization Algorithms (Schulman et al., 2017)](https://arxiv.org/abs/1707.06347)
- [Constitutional AI: Harmlessness from AI Feedback (Anthropic, 2022)](https://arxiv.org/abs/2212.08073)
- OpenAI Spinning Up in Deep RL: [PPO Guide](https://spinningup.openai.com/en/latest/algorithms/ppo.html)

---

## Changelog

- **November 2025**: Initial implementation complete
- **November 2025**: Integration with demo Phase 2 RLAIF tab
- **November 2025**: Comprehensive test suite added
- **December 2025**: Documentation consolidated
