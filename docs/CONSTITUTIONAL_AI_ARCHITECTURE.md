# Constitutional AI System Architecture

This document provides comprehensive architecture and flow diagrams for the Constitutional AI implementation.

---

## Table of Contents
1. [Overall System Architecture](#overall-system-architecture)
2. [Component Architecture](#component-architecture)
3. [Phase 1: Supervised Learning Flow](#phase-1-supervised-learning-flow)
4. [Phase 2: RLAIF Flow](#phase-2-rlaif-flow)
5. [Data Flow Diagram](#data-flow-diagram)
6. [Training Pipeline](#training-pipeline)
7. [Model Relationships](#model-relationships)

---

## Overall System Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        CONSTITUTIONAL AI SYSTEM                              │
│                                                                               │
│  ┌───────────────────────────────────────────────────────────────────────┐  │
│  │                     PHASE 1: SUPERVISED LEARNING                       │  │
│  │                      (Critique-Revision Cycle)                         │  │
│  │                                                                         │  │
│  │  ┌──────────┐    ┌──────────┐    ┌──────────┐    ┌──────────┐       │  │
│  │  │ Original │───▶│ Critique │───▶│ Revision │───▶│   SFT    │       │  │
│  │  │ Response │    │Generation│    │Generation│    │ Training │       │  │
│  │  └──────────┘    └──────────┘    └──────────┘    └──────────┘       │  │
│  │                                                          │              │  │
│  │                                                          ▼              │  │
│  │                                                   ┌─────────────┐      │  │
│  │                                                   │ Fine-tuned  │      │  │
│  │                                                   │   Model     │      │  │
│  │                                                   └─────────────┘      │  │
│  └───────────────────────────────────────────────────────────│────────────┘  │
│                                                                │               │
│                                                                ▼               │
│  ┌───────────────────────────────────────────────────────────────────────┐  │
│  │                  PHASE 2: REINFORCEMENT LEARNING                       │  │
│  │                  (RLAIF - RL from AI Feedback)                         │  │
│  │                                                                         │  │
│  │  ┌──────────┐    ┌──────────┐    ┌──────────┐    ┌──────────┐       │  │
│  │  │Preference│───▶│  Reward  │───▶│   PPO    │───▶│ Aligned  │       │  │
│  │  │   Pairs  │    │  Model   │    │ Training │    │  Model   │       │  │
│  │  │Generation│    │ Training │    │          │    │          │       │  │
│  │  └──────────┘    └──────────┘    └──────────┘    └──────────┘       │  │
│  │                                                                         │  │
│  └─────────────────────────────────────────────────────────────────────────┘  │
│                                                                               │
│  ┌─────────────────────────────────────────────────────────────────────────┐  │
│  │                         SUPPORTING COMPONENTS                           │  │
│  │                                                                         │  │
│  │  • Constitutional Framework (principles.py, framework.py)              │  │
│  │  • Model Utilities (model_utils.py) - Text generation, loading        │  │
│  │  • Dataset Classes (constitutional_dataset.py)                         │  │
│  │  • Evaluation Framework (evaluator.py)                                 │  │
│  │  • Prompt Generation (generate_constitutional_prompts.py)              │  │
│  └─────────────────────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Component Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                          COMPONENT MODULES                                   │
└─────────────────────────────────────────────────────────────────────────────┘

┌────────────────────┐
│  framework.py      │  Constitutional principles and framework management
│  principles.py     │  - Harm prevention
└────────┬───────────┘  - Truthfulness
         │              - Fairness
         │              - Autonomy respect
         │
         ├─────────────────────────────────────────┐
         │                                         │
         ▼                                         ▼
┌────────────────────┐                   ┌────────────────────┐
│ critique_revision  │                   │ preference_        │
│       .py          │                   │  comparison.py     │
├────────────────────┤                   ├────────────────────┤
│ • generate_critique│                   │ • generate_        │
│ • generate_revision│                   │   comparison       │
│ • CRITIQUE_TEMPLATE│                   │ • extract_         │
│ • REVISION_TEMPLATE│                   │   preference       │
└────────┬───────────┘                   └────────┬───────────┘
         │                                         │
         │              ┌────────────────────┐     │
         └─────────────▶│   model_utils.py   │◀────┘
                        ├────────────────────┤
                        │ • load_model       │
                        │ • generate_text    │
                        │ • GenerationConfig │
                        └────────┬───────────┘
                                 │
                                 │
         ┌───────────────────────┼───────────────────────┐
         │                       │                       │
         ▼                       ▼                       ▼
┌────────────────────┐  ┌────────────────────┐  ┌────────────────────┐
│   reward_model.py  │  │  ppo_trainer.py    │  │ constitutional_    │
├────────────────────┤  ├────────────────────┤  │   dataset.py       │
│ • RewardModel      │  │ • PPOTrainer       │  ├────────────────────┤
│ • compute_reward_  │  │ • compute_gae      │  │ • PromptDataset    │
│   loss (Bradley-   │  │ • compute_ppo_loss │  │ • PromptResponse   │
│   Terry)           │  │ • compute_kl_      │  │   Dataset          │
│ • train_reward_    │  │   divergence       │  │ • Constitutional   │
│   model            │  │ • train_step       │  │   TrainingDataset  │
│ • RewardModel      │  │ • train            │  │                    │
│   Trainer          │  │                    │  │                    │
└────────────────────┘  └────────────────────┘  └────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────┐
│                           UTILITY MODULES                                    │
├─────────────────────────────────────────────────────────────────────────────┤
│  evaluator.py      - Model evaluation against constitutional principles     │
│  filter.py         - Response filtering based on principles                 │
│  trainer.py        - General training utilities                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Phase 1: Supervised Learning Flow

**Critique-Revision Cycle for Constitutional AI**

```
                              PHASE 1: SUPERVISED LEARNING
                              =============================

┌─────────────┐
│  Training   │
│  Prompts    │  (e.g., "What is photosynthesis?")
└──────┬──────┘
       │
       ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ STEP 1: Initial Response Generation                                         │
└─────────────────────────────────────────────────────────────────────────────┘
       │
       │  ┌──────────────────────────┐
       └─▶│  Base Language Model     │
          │  (e.g., GPT-2, LLaMA)    │
          └──────────┬───────────────┘
                     │
                     ▼
              ┌─────────────┐
              │  Original   │
              │  Response   │  "Photosynthesis is when plants make food"
              └──────┬──────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ STEP 2: Critique Generation                                                 │
│ (critique_revision.py: generate_critique)                                   │
└─────────────────────────────────────────────────────────────────────────────┘
                     │
                     │  ┌──────────────────────────────────────────┐
                     │  │ CRITIQUE_TEMPLATE:                       │
                     │  │ "Consider the response: {response}       │
                     │  │  Identify any ways it is:                │
                     │  │  - Harmful, unethical, toxic             │
                     │  │  - Racist, sexist, dangerous             │
                     │  │  Evaluate against principles:            │
                     │  │  {constitutional_principles}"            │
                     └─▶└──────────────┬───────────────────────────┘
                                       │
                                       ▼
                              ┌────────────────┐
                              │  Model generates│
                              │    Critique     │
                              └────────┬───────┘
                                       │
                                       ▼
                            ┌──────────────────┐
                            │ "The response is │
                            │ too simplistic   │
                            │ and lacks        │
                            │ scientific       │
                            │ accuracy"        │
                            └────────┬─────────┘
                                     │
                                     ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ STEP 3: Revision Generation                                                 │
│ (critique_revision.py: generate_revision)                                   │
└─────────────────────────────────────────────────────────────────────────────┘
                                     │
                                     │  ┌──────────────────────────────────┐
                                     │  │ REVISION_TEMPLATE:               │
                                     │  │ "Original: {response}            │
                                     │  │  Critique: {critique}            │
                                     │  │  Please rewrite to address       │
                                     └─▶│  these issues"                   │
                                        └──────────┬───────────────────────┘
                                                   │
                                                   ▼
                                          ┌────────────────┐
                                          │ Model generates│
                                          │  Revision      │
                                          └────────┬───────┘
                                                   │
                                                   ▼
                                        ┌──────────────────────┐
                                        │ "Photosynthesis is   │
                                        │ the biological       │
                                        │ process where plants │
                                        │ convert light energy,│
                                        │ water, and CO2 into  │
                                        │ glucose and oxygen"  │
                                        └──────────┬───────────┘
                                                   │
                                                   ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ STEP 4: Dataset Creation                                                    │
└─────────────────────────────────────────────────────────────────────────────┘
                                                   │
                                                   ▼
                                    ┌──────────────────────────────┐
                                    │ Training Pair:               │
                                    │ ┌──────────────────────────┐ │
                                    │ │ Prompt: "What is..."     │ │
                                    │ │ Target: [Revised Response]│ │
                                    │ └──────────────────────────┘ │
                                    └──────────┬───────────────────┘
                                               │
                                               ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ STEP 5: Supervised Fine-Tuning                                              │
└─────────────────────────────────────────────────────────────────────────────┘
                                               │
                                               ▼
                                    ┌──────────────────┐
                                    │  Fine-tune model │
                                    │  on revised      │
                                    │  responses using │
                                    │  standard SFT    │
                                    └────────┬─────────┘
                                             │
                                             ▼
                                    ┌─────────────────┐
                                    │  Phase 1 Model  │
                                    │  (Improved via  │
                                    │   Critique-     │
                                    │   Revision)     │
                                    └─────────────────┘
                                             │
                                             ▼
                                     [Ready for Phase 2]
```

---

## Phase 2: RLAIF Flow

**Reinforcement Learning from AI Feedback**

```
                            PHASE 2: REINFORCEMENT LEARNING
                            ================================

┌─────────────┐
│  Phase 1    │
│Fine-tuned   │  Model that has been improved via critique-revision
│   Model     │
└──────┬──────┘
       │
       ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ STEP 1: Preference Pair Generation                                          │
│ (preference_comparison.py: generate_preference_pairs)                       │
└─────────────────────────────────────────────────────────────────────────────┘
       │
       │  For each prompt, generate 2+ responses
       │
       ├──────────────────────┬──────────────────────┐
       ▼                      ▼                      ▼
┌─────────────┐       ┌─────────────┐       ┌─────────────┐
│ Response A  │       │ Response B  │       │ Response C  │
└──────┬──────┘       └──────┬──────┘       └──────┬──────┘
       │                     │                      │
       └─────────────────────┴──────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ STEP 2: AI Comparison                                                       │
│ (preference_comparison.py: generate_comparison)                             │
└─────────────────────────────────────────────────────────────────────────────┘
                             │
                             │  ┌──────────────────────────────────┐
                             │  │ COMPARISON_TEMPLATE:             │
                             │  │ "Response A: {response_a}        │
                             │  │  Response B: {response_b}        │
                             │  │  Which better follows:           │
                             └─▶│  {constitutional_principles}     │
                                │  Respond with 'A' or 'B'"        │
                                └──────────┬───────────────────────┘
                                           │
                                           ▼
                                  ┌────────────────┐
                                  │ Model evaluates│
                                  │ and selects    │
                                  └────────┬───────┘
                                           │
                                           ▼
                              ┌────────────────────────┐
                              │ Preference Pair:       │
                              │ ┌────────────────────┐ │
                              │ │ Prompt: "..."      │ │
                              │ │ Chosen: Response B │ │
                              │ │ Rejected: Response A│ │
                              │ └────────────────────┘ │
                              └────────┬───────────────┘
                                       │
                                       ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ STEP 3: Reward Model Training                                               │
│ (reward_model.py: train_reward_model)                                       │
└─────────────────────────────────────────────────────────────────────────────┘
                                       │
                                       │
        ┌──────────────────────────────┴──────────────────────────────┐
        │                                                              │
        │  For each preference pair:                                  │
        │  1. reward_chosen = RewardModel(prompt + chosen)            │
        │  2. reward_rejected = RewardModel(prompt + rejected)        │
        │  3. loss = -log(sigmoid(reward_chosen - reward_rejected))   │
        │  4. Backprop and update                                     │
        │                                                              │
        └──────────────────────────────┬──────────────────────────────┘
                                       │
                                       ▼
                              ┌─────────────────┐
                              │ Trained Reward  │
                              │     Model       │
                              │                 │
                              │ Can score any   │
                              │ (prompt,response)│
                              │ for constitutional│
                              │ alignment       │
                              └────────┬────────┘
                                       │
                                       ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ STEP 4: PPO Training                                                        │
│ (ppo_trainer.py: PPOTrainer.train)                                         │
└─────────────────────────────────────────────────────────────────────────────┘
                                       │
        ┌──────────────────────────────┴──────────────────────────────┐
        │                                                              │
        │  PPO Training Loop (for each batch):                        │
        │                                                              │
        │  1. Generate responses with current policy                  │
        │  2. Score with reward model                                 │
        │  3. Compute advantages using GAE                            │
        │  4. Update policy with PPO clipped objective                │
        │  5. Update value function                                   │
        │  6. Apply KL penalty vs reference model                     │
        │                                                              │
        └──────────────────────────────┬──────────────────────────────┘
                                       │
                                       ▼

        ┌───────────────────────────────────────────────────────┐
        │           PPO TRAINING STEP DETAIL                    │
        ├───────────────────────────────────────────────────────┤
        │                                                       │
        │  ┌──────────┐                                        │
        │  │ Prompt   │                                        │
        │  └────┬─────┘                                        │
        │       │                                              │
        │       ▼                                              │
        │  ┌──────────────┐    Generate with                  │
        │  │ Policy Model ├───▶ current policy                │
        │  └──────────────┘    (temperature sampling)         │
        │       │                                              │
        │       ▼                                              │
        │  ┌──────────┐                                        │
        │  │ Response │                                        │
        │  └────┬─────┘                                        │
        │       │                                              │
        │       ├────────┬─────────┬──────────┐               │
        │       ▼        ▼         ▼          ▼               │
        │  ┌────────┐ ┌─────┐  ┌──────┐  ┌────────┐          │
        │  │ Reward │ │Value│  │Ref   │  │Current │          │
        │  │ Model  │ │Model│  │Policy│  │Policy  │          │
        │  └────┬───┘ └──┬──┘  └──┬───┘  └───┬────┘          │
        │       │        │        │          │                │
        │       ▼        ▼        ▼          ▼                │
        │   rewards   values   ref_logp  curr_logp            │
        │       │        │        │          │                │
        │       └────┬───┴────────┴──────────┘                │
        │            ▼                                        │
        │       ┌─────────┐                                   │
        │       │   GAE   │  Compute advantages              │
        │       └────┬────┘  A_t = δ_t + γλA_{t+1}          │
        │            │                                        │
        │            ▼                                        │
        │       ┌─────────┐                                   │
        │       │   PPO   │  Clipped objective               │
        │       │  Loss   │  L = min(r·A, clip(r)·A)        │
        │       └────┬────┘                                   │
        │            │                                        │
        │            ▼                                        │
        │       ┌─────────┐                                   │
        │       │ Update  │  Gradient descent                │
        │       │ Policy  │  with gradient clipping          │
        │       └─────────┘                                   │
        │                                                      │
        └──────────────────────────────────────────────────────┘

                                       │
                                       ▼
                              ┌─────────────────┐
                              │ Aligned Model   │
                              │                 │
                              │ Optimized to    │
                              │ maximize reward │
                              │ while staying   │
                              │ close to        │
                              │ reference       │
                              └─────────────────┘
                                       │
                                       ▼
                                 [COMPLETE]
```

---

## Data Flow Diagram

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                          DATA FLOW THROUGH SYSTEM                            │
└─────────────────────────────────────────────────────────────────────────────┘

                            ┌──────────────┐
                            │   Prompts    │
                            │   Dataset    │
                            └──────┬───────┘
                                   │
                   ┌───────────────┴────────────────┐
                   │                                │
                   ▼                                ▼
           ┌───────────────┐              ┌────────────────┐
           │  PHASE 1 DATA │              │  PHASE 2 DATA  │
           └───────────────┘              └────────────────┘
                   │                                │
                   │                                │
      ┌────────────┴─────────────┐      ┌──────────┴──────────────┐
      ▼                          ▼      ▼                         ▼
┌───────────┐            ┌────────────┐  ┌──────────────┐  ┌───────────┐
│ Prompts   │            │  Critique- │  │  Preference  │  │  Prompt   │
│    +      │   Phase 1  │  Revision  │  │    Pairs     │  │    for    │
│ Original  │──────────▶ │   Pairs    │  │   Dataset    │  │    PPO    │
│ Responses │            └──────┬─────┘  └──────┬───────┘  └─────┬─────┘
└───────────┘                   │               │                │
                                │               │                │
                                ▼               ▼                │
                        ┌───────────────┐  ┌─────────────┐      │
                        │ ConstitutionalDataset          │      │
                        │ TrainingDataset│  │PreferenceDataset  │
                        └───────┬───────┘  └──────┬──────┘      │
                                │                 │              │
                                ▼                 ▼              │
                        ┌──────────────┐  ┌──────────────┐      │
                        │     SFT      │  │Reward Model  │      │
                        │   Training   │  │   Training   │      │
                        └───────┬──────┘  └──────┬───────┘      │
                                │                 │              │
                                ▼                 ▼              ▼
                        ┌──────────────┐  ┌──────────────┐  ┌─────────┐
                        │   Phase 1    │  │    Reward    │  │   PPO   │
                        │    Model     │──┼▶    Model    │──┤ Trainer │
                        └──────────────┘  └──────────────┘  └────┬────┘
                                                                  │
                                                                  ▼
                                                          ┌───────────────┐
                                                          │ Final Aligned │
                                                          │     Model     │
                                                          └───────────────┘

DATA FORMATS:
─────────────

1. Critique-Revision Pairs:
   {
     "prompt": "What is AI?",
     "original_response": "AI is computers",
     "critique": "Too simplistic...",
     "revised_response": "AI is the simulation of human intelligence..."
   }

2. Preference Pairs:
   {
     "prompt": "Explain gravity",
     "response_chosen": "Gravity is a fundamental force...",
     "response_rejected": "Gravity is stuff falling down",
     "comparison_reasoning": "Response A is more accurate..."
   }

3. Training Data:
   ConstitutionalTrainingDataset(
     prompts=[...],
     responses=[...]  # Revised responses from Phase 1
   )

4. Preference Data:
   PreferenceDataset(
     data=[
       {"prompt": ..., "response_chosen": ..., "response_rejected": ...}
     ]
   )
```

---

## Training Pipeline

```
FULL CONSTITUTIONAL AI TRAINING PIPELINE
=========================================

START
  │
  ▼
┌────────────────────────────────────────┐
│ 1. Prompt Generation                   │
│    • Use HuggingFace datasets          │
│    • Template-based generation         │
│    • Combined approach (recommended)   │
│    ▶ Output: 500-1000 prompts          │
└────────┬───────────────────────────────┘
         │
         ▼
┌────────────────────────────────────────┐
│ 2. Initial Response Generation         │
│    • Load base model (GPT-2/LLaMA)     │
│    • Generate responses for all prompts│
│    ▶ Output: Prompt-Response pairs     │
└────────┬───────────────────────────────┘
         │
         ▼
┌────────────────────────────────────────┐
│ 3. Phase 1: Critique-Revision          │
│    For each (prompt, response):        │
│    a) Generate critique                │
│    b) Generate revision                │
│    c) Store training pair              │
│    ▶ Output: Critique-revision dataset │
│    ▶ Time: ~2-4 hours                  │
└────────┬───────────────────────────────┘
         │
         ▼
┌────────────────────────────────────────┐
│ 4. Supervised Fine-Tuning              │
│    • Create ConstitutionalDataset      │
│    • Train on revised responses        │
│    • Validate on held-out set          │
│    ▶ Output: Phase 1 model             │
│    ▶ Time: ~3-6 hours                  │
└────────┬───────────────────────────────┘
         │
         ▼
┌────────────────────────────────────────┐
│ 5. Phase 2a: Preference Generation     │
│    For each prompt:                    │
│    a) Generate 2+ responses            │
│    b) Compare with AI                  │
│    c) Extract preference               │
│    d) Store preference pair            │
│    ▶ Output: Preference dataset        │
│    ▶ Time: ~3-5 hours                  │
└────────┬───────────────────────────────┘
         │
         ▼
┌────────────────────────────────────────┐
│ 6. Phase 2b: Reward Model Training     │
│    • Create PreferenceDataset          │
│    • Train with Bradley-Terry loss     │
│    • Target: >75% accuracy             │
│    ▶ Output: Reward model              │
│    ▶ Time: ~2-4 hours                  │
└────────┬───────────────────────────────┘
         │
         ▼
┌────────────────────────────────────────┐
│ 7. Phase 2c: PPO Training              │
│    • Initialize PPO trainer            │
│    • Train with reward model feedback  │
│    • Monitor KL divergence             │
│    • Checkpoint regularly              │
│    ▶ Output: Aligned model             │
│    ▶ Time: ~4-8 hours                  │
└────────┬───────────────────────────────┘
         │
         ▼
┌────────────────────────────────────────┐
│ 8. Evaluation                          │
│    • Test on held-out prompts          │
│    • Measure constitutional alignment  │
│    • Compare before/after metrics      │
│    ▶ Output: Evaluation report         │
└────────┬───────────────────────────────┘
         │
         ▼
       END

TOTAL TIME: ~15-30 hours (depends on dataset size and hardware)
TOTAL DATA: 500-1000 prompts recommended for good results
```

---

## Model Relationships

```
MODEL ARCHITECTURE AND RELATIONSHIPS
=====================================

┌──────────────────────────────────────────────────────────────────┐
│                        BASE MODEL                                 │
│                   (e.g., GPT-2, LLaMA)                           │
│                                                                   │
│              [Transformer Architecture]                          │
│              • Embeddings                                        │
│              • Self-attention layers                             │
│              • Feed-forward layers                               │
│              • Output projection                                 │
└────────────────────┬─────────────────────────────────────────────┘
                     │
                     │ Copy and fine-tune
                     │
     ┌───────────────┴────────────────────────────────┐
     │                                                 │
     ▼                                                 ▼
┌─────────────────────┐                      ┌────────────────────┐
│   PHASE 1 MODEL     │                      │  REWARD MODEL      │
│   (Fine-tuned SFT)  │                      │  (Score responses) │
├─────────────────────┤                      ├────────────────────┤
│ Base model +        │                      │ Base model +       │
│ Updated weights     │                      │ Reward head:       │
│ from critique-      │                      │                    │
│ revision training   │                      │ ┌────────────────┐ │
│                     │                      │ │ Linear(768→256)│ │
│                     │                      │ │ ReLU           │ │
│                     │                      │ │ Dropout(0.1)   │ │
│                     │                      │ │ Linear(256→1)  │ │
│                     │                      │ └────────────────┘ │
│                     │                      │                    │
│                     │                      │ Output: scalar     │
│                     │                      │ reward score       │
└──────────┬──────────┘                      └──────────┬─────────┘
           │                                            │
           │ Used in PPO                                │ Used in PPO
           │                                            │
           └──────────────┬─────────────────────────────┘
                          │
                          ▼
        ┌─────────────────────────────────────────────┐
        │          PPO TRAINER MODELS                 │
        ├─────────────────────────────────────────────┤
        │                                             │
        │  ┌──────────────┐  ┌──────────────┐       │
        │  │ Policy Model │  │ Value Model  │       │
        │  │  (trainable) │  │  (trainable) │       │
        │  │              │  │              │       │
        │  │ Generates    │  │ Estimates    │       │
        │  │ responses    │  │ state value  │       │
        │  └──────────────┘  └──────────────┘       │
        │                                             │
        │  ┌──────────────┐  ┌──────────────┐       │
        │  │Reward Model  │  │Reference Model│       │
        │  │  (frozen)    │  │   (frozen)   │       │
        │  │              │  │              │       │
        │  │ Scores       │  │ KL penalty   │       │
        │  │ responses    │  │ anchor       │       │
        │  └──────────────┘  └──────────────┘       │
        │                                             │
        └──────────────┬──────────────────────────────┘
                       │
                       ▼
              ┌────────────────┐
              │  FINAL ALIGNED │
              │     MODEL      │
              │                │
              │ • Better safety│
              │ • More helpful │
              │ • Constitutional│
              │   compliance   │
              └────────────────┘

PARAMETER FLOW:
───────────────

1. Base Model → Phase 1 Model:
   • All weights trainable
   • Standard supervised learning
   • Minimizes cross-entropy loss

2. Base Model → Reward Model:
   • Base weights frozen or fine-tuned
   • Reward head trainable
   • Minimizes Bradley-Terry loss

3. Phase 1 Model → Policy Model:
   • Initializes policy for PPO
   • Weights become trainable
   • Updated with PPO objective

4. Phase 1 Model → Reference Model:
   • Deep copy of Phase 1 model
   • All weights frozen
   • Never updated (KL anchor)

5. Policy Model → Final Model:
   • After PPO converges
   • Aligned with constitutional principles
   • Ready for deployment
```

---

## Summary

The Constitutional AI system consists of two main phases:

1. **Phase 1 (Supervised Learning)**: Improves responses through critique and revision
2. **Phase 2 (RLAIF)**: Optimizes the model using reinforcement learning with AI-generated preferences

All components are tightly integrated through:
- Shared constitutional framework
- Common model utilities for text generation
- Consistent data formats
- Clean interfaces between components

The system produces a final model that is:
- ✅ More helpful and accurate
- ✅ Less likely to produce harmful content
- ✅ Aligned with constitutional principles
- ✅ Better than the base model on safety metrics

**Total Training Time**: ~15-30 hours for full pipeline
**Required Data**: 500-1000 prompts recommended
**Hardware**: GPU recommended (works on CPU but slower)
