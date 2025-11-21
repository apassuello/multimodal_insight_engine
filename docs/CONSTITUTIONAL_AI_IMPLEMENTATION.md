# Constitutional AI Implementation Analysis & Roadmap

**Last Updated**: 2024
**Status**: Phase 1 (Supervised Learning) - Needs Fixes | Phase 2 (RLAIF) - Not Implemented

---

## Table of Contents

1. [Executive Summary](#executive-summary)
2. [Original CAI Paper Reference](#original-cai-paper-reference)
3. [Current Implementation Analysis](#current-implementation-analysis)
4. [Phase 1 Issues & Fixes](#phase-1-issues--fixes)
5. [Phase 2 RLAIF Implementation Plan](#phase-2-rlaif-implementation-plan)
6. [Model Recommendations](#model-recommendations)

---

## Executive Summary

### What Constitutional AI Is

Constitutional AI (CAI) is Anthropic's approach to training AI systems to be helpful, harmless, and honest. It consists of two phases:

| Phase | Name | Method | What Happens |
|-------|------|--------|--------------|
| **Phase 1** | Supervised Learning (SL) | Critique-Revision | Model critiques and revises its own responses, then trains on revisions |
| **Phase 2** | RLAIF | Reinforcement Learning | AI ranks responses, trains reward model, RL optimizes generation |

### Our Current State

| Component | Status | Notes |
|-----------|--------|-------|
| Phase 1 (SL) | Partially Working | Self-critique quality is poor with small models |
| Phase 2 (RLAIF) | Not Implemented | Requires PPO, reward model, preference collection |
| Evaluation | Working | Regex + optional AI hybrid evaluation |
| Training | Working | Standard supervised fine-tuning |

---

## Original CAI Paper Reference

**Paper**: "Constitutional AI: Harmlessness from AI Feedback" (Bai et al., 2022)
**arXiv**: 2212.08073
**GitHub**: https://github.com/anthropics/ConstitutionalHarmlessnessPaper

### Paper's Critique Templates (16 Principles)

The paper uses **16 different critique-revision principle pairs**, not a single generic template:

```
Principle 0 - General Harm:
  Critique: "Identify specific ways in which the assistant's last response
            is harmful, unethical, racist, sexist, toxic, dangerous, or illegal."
  Revision: "Please rewrite the assistant response to remove any and all
            harmful, unethical, racist, sexist, toxic, dangerous, or illegal content."

Principle 6 - Strongest Argument:
  Critique: "What's the strongest argument you could give that the assistant's
            last response was in some way harmful, unethical, racist, sexist,
            toxic, dangerous, or illegal."

Principle 7 - Child Safety:
  Critique: "Did the assistant's preceding response include anything that may
            be inappropriate for young children?"

Principle 12 - Empathy:
  Critique: "Discuss the extent to which the assistant's response is thoughtful
            and empathetic."
```

### Paper's Key Parameters

| Parameter | Paper's Value | Our Value | Notes |
|-----------|---------------|-----------|-------|
| Model size | 52B parameters | 125M-3B | Much smaller |
| Training examples | Thousands | 20-50 | Much fewer |
| Critique iterations | Multiple | 1-2 | Similar |
| Principles | 16 varied templates | 1 generic template | Less diverse |

---

## Current Implementation Analysis

### Architecture Overview

```
Current Flow (Phase 1 Only):

  Adversarial Prompt
        |
        v
  [Generation Model] --> Initial Response
        |
        v
  [Same Gen Model]   --> Critique (self-critique)  <-- PROBLEM: Small models bad at this
        |
        v
  [Same Gen Model]   --> Revised Response
        |
        v
  [Eval Model/Regex] --> Score (for filtering)
        |
        v
  improvement > 0? --YES--> Add to training data
        |
       NO --> Skip
        |
        v
  Supervised Fine-tuning on (prompt, revision) pairs
```

### Key Files

| Component | File |
|-----------|------|
| Critique/Revision | `src/safety/constitutional/critique_revision.py` |
| Evaluation | `src/safety/constitutional/principles.py` |
| Framework | `src/safety/constitutional/framework.py` |
| Model Utils | `src/safety/constitutional/model_utils.py` |

---

## Phase 1 Issues & Fixes

### Issue 1: Single Generic Template

**Problem**: We use one template for all critiques. Paper uses 16 diverse templates.

**Current**:
```python
CRITIQUE_TEMPLATE = """Identify any ways in which the assistant's response
is harmful, unethical, racist, sexist, toxic, dangerous, or illegal.
Specifically, evaluate against these principles: {principles_text}"""
```

**Fix**: Add multiple principle-specific templates and rotate through them.

### Issue 2: Small Models Can't Self-Critique Well

**Problem**: GPT-2 (125M params) critiquing itself produces garbage critiques.

**Options**:
1. Use external/larger model for critique (not true CAI but practical)
2. Use template-based critiques (deterministic, no AI needed)
3. Skip critique, use known-good revision templates

**Recommended for Demo**: Option 2 or 3 - don't rely on small model self-critique.

### Issue 3: Low Training Data Volume

**Problem**: 20-50 examples vs thousands in the paper.

**Fix**: Either generate more examples or accept this is a demo limitation.

### Issue 4: Training Data Format

**Current**: Simple concatenation `prompt + response`

**Better**: Use instruction format with clear separator:
```
<|user|>{prompt}<|assistant|>{response}<|end|>
```

---

## Phase 2 RLAIF Implementation Plan

### Overview

RLAIF (Reinforcement Learning from AI Feedback) is the second phase of CAI. It uses AI to generate preference data, trains a reward model, then uses RL to optimize.

### Architecture

```
Phase 2 (RLAIF) Flow:

  Prompt P
    |
    v
  Generate N responses: [R1, R2, R3, R4]
    |
    v
  AI Compares pairs:
    "Is R1 better than R2 according to principles?"
    "Is R3 better than R4 according to principles?"
    |
    v
  Collect preferences: [(R1, R2, R1>R2), (R3, R4, R4>R3), ...]
    |
    v
  Train Reward Model: R(prompt, response) --> scalar
    |
    v
  RL (PPO) to maximize: E[R(prompt, generated_response)]
```

### Implementation Steps

#### Step 1: Preference Data Collection

```python
# New file: src/safety/constitutional/preference_collection.py

def collect_preferences(
    prompts: List[str],
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    eval_model: PreTrainedModel,  # For ranking
    num_responses: int = 4,
    device: torch.device = None
) -> List[Dict]:
    """
    Generate multiple responses per prompt and have AI rank them.

    Returns:
        List of preference pairs: {
            'prompt': str,
            'chosen': str,      # Better response
            'rejected': str,    # Worse response
            'principle': str    # Which principle was used for comparison
        }
    """
    preferences = []

    for prompt in prompts:
        # Generate multiple responses
        responses = []
        for _ in range(num_responses):
            response = generate_text(model, tokenizer, prompt, config, device)
            responses.append(response)

        # Compare all pairs using AI
        for i in range(len(responses)):
            for j in range(i+1, len(responses)):
                # AI decides which is better
                comparison = compare_responses(
                    eval_model, tokenizer, prompt,
                    responses[i], responses[j], device
                )
                preferences.append({
                    'prompt': prompt,
                    'chosen': comparison['better'],
                    'rejected': comparison['worse'],
                    'principle': comparison['principle_used']
                })

    return preferences
```

#### Step 2: Reward Model Training

```python
# New file: src/safety/constitutional/reward_model.py

class RewardModel(nn.Module):
    """
    Reward model that scores (prompt, response) pairs.
    Trained on preference data to predict which response is better.
    """

    def __init__(self, base_model: PreTrainedModel):
        super().__init__()
        self.base_model = base_model
        self.reward_head = nn.Linear(base_model.config.hidden_size, 1)

    def forward(self, input_ids, attention_mask):
        outputs = self.base_model(input_ids, attention_mask=attention_mask)
        hidden_states = outputs.last_hidden_state[:, -1, :]  # Last token
        reward = self.reward_head(hidden_states)
        return reward

def train_reward_model(
    preferences: List[Dict],
    base_model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    epochs: int = 3,
    batch_size: int = 8
) -> RewardModel:
    """
    Train reward model on preference pairs using Bradley-Terry loss.
    """
    reward_model = RewardModel(base_model)
    optimizer = torch.optim.AdamW(reward_model.parameters(), lr=1e-5)

    for epoch in range(epochs):
        for batch in dataloader:
            # Get rewards for chosen and rejected
            chosen_reward = reward_model(batch['chosen_ids'], batch['chosen_mask'])
            rejected_reward = reward_model(batch['rejected_ids'], batch['rejected_mask'])

            # Bradley-Terry loss: -log(sigmoid(chosen - rejected))
            loss = -torch.log(torch.sigmoid(chosen_reward - rejected_reward)).mean()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    return reward_model
```

#### Step 3: PPO Training

```python
# New file: src/safety/constitutional/ppo_training.py

# Requires: pip install trl

from trl import PPOTrainer, PPOConfig

def train_with_ppo(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    reward_model: RewardModel,
    prompts: List[str],
    epochs: int = 4
):
    """
    Fine-tune model using PPO with the trained reward model.
    """
    ppo_config = PPOConfig(
        model_name="constitutional-ai",
        learning_rate=1e-5,
        batch_size=16,
        mini_batch_size=4,
        gradient_accumulation_steps=1,
        ppo_epochs=4,
        max_grad_norm=1.0,
    )

    ppo_trainer = PPOTrainer(
        config=ppo_config,
        model=model,
        tokenizer=tokenizer,
        dataset=prompts_dataset,
    )

    for epoch in range(epochs):
        for batch in ppo_trainer.dataloader:
            # Generate responses
            query_tensors = batch["input_ids"]
            response_tensors = ppo_trainer.generate(query_tensors)

            # Get rewards from reward model
            rewards = reward_model(
                torch.cat([query_tensors, response_tensors], dim=1),
                attention_mask=...
            )

            # PPO update
            stats = ppo_trainer.step(query_tensors, response_tensors, rewards)

    return model
```

### Dependencies for RLAIF

```
pip install trl>=0.7.0  # HuggingFace TRL library for PPO
pip install peft>=0.5.0  # For LoRA (optional but recommended)
```

### Comparison Prompt Template for RLAIF

```python
COMPARISON_TEMPLATE = """Consider the following prompt and two possible responses:

Prompt: {prompt}

Response A: {response_a}

Response B: {response_b}

According to the principle of "{principle}", which response is better?
The principle states: "{principle_description}"

Answer with just "A" or "B":"""
```

---

## Model Recommendations

### For Resource-Constrained Demo

| Role | Model | Size | Why |
|------|-------|------|-----|
| **Generation** | GPT-2 or DistilGPT-2 | 125-500MB | Small, trainable, "dumb" enough to show improvement |
| **Evaluation** | Regex only | 0MB | Fast, reliable for obvious cases |

### For Better Results (More Resources)

| Role | Model | Size | Why |
|------|-------|------|-----|
| **Generation** | GPT-2 or TinyLlama | 500MB-2GB | Trainable |
| **Evaluation** | toxic-bert or HuggingFace API | 500MB or API | 98% accuracy on toxicity |

### For Full RLAIF Implementation

| Role | Model | Size | Why |
|------|-------|------|-----|
| **Generation** | Phi-2 or larger | 2.5GB+ | Better quality generations |
| **Reward Model** | Same as generation | 2.5GB+ | Needs to understand quality |
| **Comparison AI** | Larger model or API | 7GB+ or API | Needs to reliably rank responses |

---

## Quick Reference: What To Fix First

### Priority 1: Make Phase 1 Work for Demo

1. **Don't rely on small model self-critique**
   - Use regex evaluation to detect violations
   - Use template-based revisions instead of model-generated ones

2. **Use GPT-2 for generation** (small, trainable)

3. **Focus on obvious adversarial prompts** (regex catches these well)

### Priority 2: Improve Phase 1 (Later)

1. Add multiple critique templates (paper's 16 principles)
2. Better training data format with separators
3. More training examples

### Priority 3: Implement RLAIF (Future)

1. Preference collection
2. Reward model training
3. PPO fine-tuning

---

## Changelog

- **2024-XX-XX**: Initial analysis and RLAIF plan documented
- **2024-XX-XX**: Phase 1 issues identified
