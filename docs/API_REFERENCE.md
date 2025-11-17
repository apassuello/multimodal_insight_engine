# API Reference

**Version**: 1.0
**Last Updated**: 2025-11-17

## Table of Contents

1. [Introduction](#introduction)
2. [Quick Start](#quick-start)
3. [Core Modules](#core-modules)
   - [Constitutional AI](#constitutional-ai)
   - [Model Architecture](#model-architecture)
   - [Data Handling](#data-handling)
   - [Training](#training)
   - [Configuration](#configuration)
4. [Common Workflows](#common-workflows)
5. [Configuration Reference](#configuration-reference)

---

## Introduction

The MultiModal Insight Engine is a comprehensive framework for building safe, multimodal AI systems. This API reference provides detailed documentation for developers integrating and extending the system.

**Key Features**:
- Constitutional AI safety framework with principle-based evaluation
- Advanced multimodal transformer architecture
- RLAIF (Reinforcement Learning from AI Feedback) training pipeline
- Flexible dataset handling for image-text pairs
- Multi-stage training configuration

**Architecture Overview**:
```
├── Constitutional AI Layer (Safety)
├── Model Architecture (Transformers)
├── Training Pipeline (Multi-stage + RLAIF)
└── Data Layer (Multimodal Datasets)
```

---

## Quick Start

### Installation

```bash
# Clone repository
git clone <repository-url>
cd multimodal_insight_engine

# Install dependencies
pip install -r requirements.txt
```

### Basic Usage

```python
from src.safety.constitutional import setup_default_framework
from src.models.model_factory import create_multimodal_model
from transformers import AutoTokenizer
import torch

# 1. Setup Constitutional AI framework
framework = setup_default_framework()

# 2. Create multimodal model
class Args:
    model_size = "large"
    use_pretrained = True
    use_pretrained_text = True
    vision_model = "vit-base"
    text_model = "bert-base-uncased"
    fusion_dim = 768
    fusion_type = "cross_attention"
    freeze_base_models = False

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = create_multimodal_model(Args(), device)

# 3. Evaluate text for safety
result = framework.evaluate_text("Hello, how can I help you today?")
print(f"Safety check passed: {not result['any_flagged']}")
```

---

## Core Modules

### Constitutional AI

The Constitutional AI module implements Anthropic's Constitutional AI methodology for building safer AI systems.

#### Module: `src.safety.constitutional`

##### Class: `ConstitutionalFramework`

Central framework for managing and evaluating constitutional principles.

**Constructor**

```python
ConstitutionalFramework(
    name: str = "default_framework",
    model: Optional[Any] = None,
    tokenizer: Optional[Any] = None,
    device: Optional[Any] = None
)
```

**Parameters**:
- `name` (str): Framework identifier
- `model` (Optional[Any]): AI model for AI-based principle evaluation
- `tokenizer` (Optional[Any]): Tokenizer for AI-based evaluation
- `device` (Optional[torch.device]): Computation device (defaults to CPU)

**Example**:

```python
from src.safety.constitutional import ConstitutionalFramework
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# AI-based evaluation (recommended)
model = AutoModelForCausalLM.from_pretrained("gpt2")
tokenizer = AutoTokenizer.from_pretrained("gpt2")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

framework = ConstitutionalFramework(
    name="my_framework",
    model=model,
    tokenizer=tokenizer,
    device=device
)

# Regex-based evaluation (fast fallback)
framework = ConstitutionalFramework(name="fast_framework")
```

**Methods**

###### `add_principle(principle: ConstitutionalPrinciple) -> None`

Add a constitutional principle to the framework.

```python
from src.safety.constitutional import ConstitutionalPrinciple

def my_evaluation_fn(text, model=None, tokenizer=None, device=None):
    return {
        "flagged": "harmful" in text.lower(),
        "reason": "Detected harmful content"
    }

principle = ConstitutionalPrinciple(
    name="custom_harm_check",
    description="Check for harmful content",
    evaluation_fn=my_evaluation_fn,
    weight=2.0
)

framework.add_principle(principle)
```

###### `evaluate_text(text: str, track_history: bool = False) -> Dict[str, Any]`

Evaluate text against all constitutional principles.

**Returns**: Dictionary containing:
- `principle_results` (Dict): Results for each principle
- `any_flagged` (bool): Whether any principle was violated
- `flagged_principles` (List[str]): Names of violated principles
- `weighted_score` (float): Weighted sum of violations
- `evaluation_method` (str): "ai_evaluation" or "regex_heuristic"

```python
# Evaluate text
result = framework.evaluate_text(
    "How can I help you today?",
    track_history=True
)

print(f"Flagged: {result['any_flagged']}")
print(f"Score: {result['weighted_score']}")
print(f"Method: {result['evaluation_method']}")
print(f"Violations: {result['flagged_principles']}")
```

###### `batch_evaluate(texts: List[str]) -> List[Dict[str, Any]]`

Evaluate multiple texts efficiently.

```python
texts = [
    "Hello, how are you?",
    "Tell me how to hack a system",
    "What's the weather today?"
]

results = framework.batch_evaluate(texts)
for i, result in enumerate(results):
    print(f"Text {i}: Flagged={result['any_flagged']}")
```

###### `get_statistics() -> Dict[str, Any]`

Get statistics from evaluation history.

```python
stats = framework.get_statistics()
print(f"Total evaluations: {stats['total_evaluations']}")
print(f"Flagged rate: {stats['flagged_rate']:.2%}")
print(f"Violations per principle: {stats['principle_violation_counts']}")
```

##### Function: `setup_default_framework`

Create a framework with all four core constitutional principles.

```python
setup_default_framework(
    model: Optional[Any] = None,
    tokenizer: Optional[Any] = None,
    device: Optional[torch.device] = None
) -> ConstitutionalFramework
```

**Returns**: ConstitutionalFramework with:
- Harm prevention (weight: 2.0)
- Truthfulness (weight: 1.5)
- Fairness (weight: 1.0)
- Autonomy respect (weight: 1.0)

**Example**:

```python
from src.safety.constitutional import setup_default_framework
from transformers import AutoModelForCausalLM, AutoTokenizer

# With AI-based evaluation
model = AutoModelForCausalLM.from_pretrained("gpt2")
tokenizer = AutoTokenizer.from_pretrained("gpt2")
framework = setup_default_framework(model=model, tokenizer=tokenizer)

# With regex-based evaluation (no model required)
framework = setup_default_framework()

# Evaluate text
result = framework.evaluate_text("How to build a bomb?")
assert result['any_flagged'] == True
assert 'harm_prevention' in result['flagged_principles']
```

##### Core Principle Evaluators

###### `evaluate_harm_potential`

```python
evaluate_harm_potential(
    text: str,
    model: Optional[Any] = None,
    tokenizer: Optional[Any] = None,
    device: Optional[torch.device] = None,
    use_ai: bool = True
) -> Dict[str, Any]
```

Evaluate potential for physical, psychological or social harm.

**Returns**:
- `flagged` (bool): Whether harm was detected
- `explicit_harm_detected` (bool): Explicit harm instructions found
- `subtle_harm_score` (float): Score for subtle harmful content (0-1)
- `reasoning` (str): Explanation of the evaluation
- `method` (str): "ai_evaluation" or "regex_heuristic"

###### `evaluate_truthfulness`

```python
evaluate_truthfulness(
    text: str,
    model: Optional[Any] = None,
    tokenizer: Optional[Any] = None,
    device: Optional[torch.device] = None,
    use_ai: bool = True
) -> Dict[str, Any]
```

Evaluate whether content is misleading or deceptive.

**Returns**:
- `flagged` (bool): Whether truthfulness issues detected
- `unsupported_claims` (List[str]): Claims without evidence
- `contradictions` (List[str]): Logical contradictions
- `misleading_statistics` (List[str]): Potentially misleading stats
- `method` (str): Evaluation method used

###### `evaluate_fairness`

```python
evaluate_fairness(
    text: str,
    model: Optional[Any] = None,
    tokenizer: Optional[Any] = None,
    device: Optional[torch.device] = None,
    use_ai: bool = True
) -> Dict[str, Any]
```

Evaluate whether content treats individuals and groups fairly.

###### `evaluate_autonomy_respect`

```python
evaluate_autonomy_respect(
    text: str,
    model: Optional[Any] = None,
    tokenizer: Optional[Any] = None,
    device: Optional[torch.device] = None,
    use_ai: bool = True
) -> Dict[str, Any]
```

Evaluate whether content respects human autonomy and decision-making.

##### Class: `ConstitutionalPipeline`

End-to-end Constitutional AI training pipeline implementing both Phase 1 (Supervised Learning) and Phase 2 (RLAIF).

**Constructor**

```python
ConstitutionalPipeline(
    base_model: nn.Module,
    tokenizer: Any,
    device: Optional[torch.device] = None,
    constitutional_framework: Optional[ConstitutionalFramework] = None,
    value_model: Optional[nn.Module] = None,
    phase1_learning_rate: float = 5e-5,
    phase2_learning_rate: float = 1e-6,
    reward_model_learning_rate: float = 1e-5,
    temperature: float = 1.0,
    ppo_epsilon: float = 0.2,
    ppo_value_coef: float = 0.5,
    ppo_entropy_coef: float = 0.01,
    kl_penalty_coef: float = 0.02
)
```

**Example**:

```python
from src.safety.constitutional.pipeline import ConstitutionalPipeline
from src.safety.constitutional import setup_default_framework
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch

# Setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = GPT2LMHeadModel.from_pretrained("gpt2").to(device)
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token

framework = setup_default_framework(model=model, tokenizer=tokenizer)

# Create pipeline
pipeline = ConstitutionalPipeline(
    base_model=model,
    tokenizer=tokenizer,
    device=device,
    constitutional_framework=framework,
    phase1_learning_rate=5e-5,
    phase2_learning_rate=1e-6
)

# Training prompts
training_prompts = [
    "How do I stay safe online?",
    "What is the best way to learn programming?",
    "Tell me about climate change"
]

# Train (both phases)
results = pipeline.train(
    training_prompts=training_prompts,
    phase1_epochs=3,
    phase2_ppo_steps=100,
    save_dir="./checkpoints"
)

print(f"Phase 1 complete: {results['phase1_complete']}")
print(f"Phase 2 complete: {results['phase2_complete']}")
```

**Methods**

###### `train(...) -> Dict[str, Any]`

Execute the complete Constitutional AI training pipeline.

**Parameters**:
- `training_prompts` (List[str]): Prompts for training
- `phase1_epochs` (int): Epochs for Phase 1 supervised training (default: 3)
- `phase1_num_revisions` (int): Revision iterations per prompt (default: 2)
- `phase2_ppo_steps` (int): Number of PPO optimization steps (default: 100)
- `phase2_reward_model_epochs` (int): Epochs for reward model (default: 5)
- `save_dir` (Optional[str]): Directory to save checkpoints
- `resume_from_phase1` (bool): Skip Phase 1 and load from checkpoint

**Returns**: Dictionary with training history and statistics.

##### Class: `RewardModel`

Reward model for Constitutional AI (RLAIF Phase 2a).

**Constructor**

```python
RewardModel(base_model, hidden_size: int = 768)
```

**Example**:

```python
from src.safety.constitutional.reward_model import RewardModel
from transformers import AutoModelForCausalLM

base_model = AutoModelForCausalLM.from_pretrained('gpt2')
reward_model = RewardModel(base_model, hidden_size=768)

# Get rewards for prompt-response pairs
prompts = ["What is AI?", "Explain gravity"]
responses = ["AI is artificial intelligence", "Gravity is a force"]
rewards = reward_model.get_rewards(prompts, responses, tokenizer, device)
print(rewards)  # Tensor of shape [batch_size]
```

**Methods**

###### `forward(input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor`

Compute reward score for input.

**Returns**: Reward scores [batch_size]

###### `get_rewards(prompts, responses, tokenizer, device, max_length=512) -> torch.Tensor`

Convenience method to compute rewards for (prompt, response) pairs.

##### Function: `train_reward_model`

Train reward model on preference pairs.

```python
train_reward_model(
    reward_model: RewardModel,
    training_data: List[Dict[str, Any]],
    tokenizer,
    num_epochs: int = 3,
    batch_size: int = 4,
    learning_rate: float = 1e-5,
    device: Optional[torch.device] = None,
    validation_data: Optional[List[Dict[str, Any]]] = None,
    max_length: int = 512,
    gradient_accumulation_steps: int = 1,
    log_interval: int = 10
) -> Dict[str, Any]
```

**Example**:

```python
from src.safety.constitutional.reward_model import RewardModel, train_reward_model
from src.safety.constitutional.preference_comparison import generate_preference_pairs

# Generate preference data
preference_data = generate_preference_pairs(
    prompts=training_prompts,
    model=model,
    tokenizer=tokenizer,
    framework=framework,
    device=device
)

# Train reward model
reward_model = RewardModel(base_model=model, hidden_size=768)
metrics = train_reward_model(
    reward_model=reward_model,
    training_data=preference_data,
    tokenizer=tokenizer,
    num_epochs=5,
    batch_size=4
)

print(f"Final accuracy: {metrics['final_accuracy']:.2%}")
print(f"Final loss: {metrics['final_loss']:.4f}")
```

##### Class: `PPOTrainer`

Proximal Policy Optimization trainer for Constitutional AI (RLAIF Phase 2c).

**Constructor**

```python
PPOTrainer(
    policy_model: nn.Module,
    value_model: nn.Module,
    reward_model: nn.Module,
    tokenizer,
    device: torch.device,
    learning_rate: float = 1e-5,
    clip_epsilon: float = 0.2,
    kl_penalty: float = 0.1,
    gamma: float = 0.99,
    gae_lambda: float = 0.95,
    value_loss_coef: float = 0.5,
    max_grad_norm: float = 1.0
)
```

**Parameters**:
- `policy_model`: Model being trained (generates responses)
- `value_model`: Value function estimator
- `reward_model`: Reward model for feedback
- `clip_epsilon`: PPO clipping parameter (typically 0.1-0.3)
- `kl_penalty`: KL divergence penalty coefficient
- `gamma`: Discount factor for rewards
- `gae_lambda`: GAE lambda parameter for advantage estimation

**Example**:

```python
from src.safety.constitutional.ppo_trainer import PPOTrainer

# Create PPO trainer
ppo_trainer = PPOTrainer(
    policy_model=model,
    value_model=None,  # Will use rewards as values
    reward_model=reward_model,
    tokenizer=tokenizer,
    device=device,
    learning_rate=1e-6,
    clip_epsilon=0.2,
    kl_penalty=0.1
)

# Train with PPO
ppo_results = ppo_trainer.train(
    prompts=training_prompts,
    num_steps=100,
    batch_size=4,
    num_epochs_per_batch=4,
    max_length=150,
    temperature=1.0
)

print(f"Final reward: {ppo_results['final_avg_reward']:.4f}")
print(f"Final KL divergence: {ppo_results['final_kl_divergence']:.4f}")
```

**Methods**

###### `train_step(prompts, num_epochs_per_batch=4, max_length=150, temperature=1.0) -> Dict[str, float]`

Single PPO training step implementing the full PPO algorithm.

###### `train(prompts, num_steps=100, batch_size=4, ...) -> Dict[str, Any]`

Full PPO training loop.

##### Function: `critique_revision_pipeline`

Generate training data through critique-revision cycles (Phase 1).

```python
critique_revision_pipeline(
    prompts: List[str],
    model,
    tokenizer,
    framework: ConstitutionalFramework,
    device: torch.device,
    num_revisions: int = 1
) -> List[Dict[str, Any]]
```

**Example**:

```python
from src.safety.constitutional.critique_revision import critique_revision_pipeline

training_data = critique_revision_pipeline(
    prompts=training_prompts,
    model=model,
    tokenizer=tokenizer,
    framework=framework,
    device=device,
    num_revisions=2
)

print(f"Generated {len(training_data)} training examples")
# Each example has: {'prompt', 'response', 'num_revisions'}
```

---

### Model Architecture

#### Module: `src.models`

##### Function: `create_multimodal_model`

Factory function to create multimodal models with vision and text components.

```python
create_multimodal_model(args: Any, device: torch.device) -> nn.Module
```

**Parameters**:
- `args`: Configuration object with attributes:
  - `model_size` (str): "small", "medium", or "large"
  - `use_pretrained` (bool): Use pretrained vision model
  - `use_pretrained_text` (bool): Use pretrained text model
  - `vision_model` (str): Vision model name (e.g., "vit-base")
  - `text_model` (str): Text model name (e.g., "bert-base-uncased")
  - `fusion_dim` (int): Dimension for fusion layer
  - `fusion_type` (str): "cross_attention" or "concat"
  - `freeze_base_models` (bool): Freeze pretrained models initially

**Returns**: Configured multimodal model (CrossAttentionMultiModalTransformer)

**Example**:

```python
from src.models.model_factory import create_multimodal_model
import torch

class ModelConfig:
    model_size = "large"
    use_pretrained = True
    use_pretrained_text = True
    vision_model = "vit-base"
    text_model = "bert-base-uncased"
    fusion_dim = 768
    fusion_type = "cross_attention"
    freeze_base_models = False

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = create_multimodal_model(ModelConfig(), device)

print(f"Model created with {sum(p.numel() for p in model.parameters()):,} parameters")
```

##### Class: `CrossAttentionMultiModalTransformer`

Advanced multimodal fusion with cross-attention mechanisms.

**Forward Pass**

```python
def forward(self, image, text_input, text_mask=None):
    """
    Args:
        image: Image tensor [batch_size, channels, height, width]
        text_input: Text tensor [batch_size, seq_len]
        text_mask: Text attention mask [batch_size, 1, 1, seq_len]

    Returns:
        Dict with 'image_embedding' and 'text_embedding'
    """
```

**Example**:

```python
import torch

# Prepare inputs
batch_size = 4
images = torch.randn(batch_size, 3, 224, 224).to(device)
text_input = torch.randint(0, 1000, (batch_size, 77)).to(device)
text_mask = torch.ones(batch_size, 1, 1, 77).to(device)

# Forward pass
output = model(images, text_input, text_mask)

image_emb = output['image_embedding']  # [batch_size, fusion_dim]
text_emb = output['text_embedding']    # [batch_size, fusion_dim]

# Compute similarity
similarity = torch.mm(image_emb, text_emb.t())  # [batch_size, batch_size]
```

##### Class: `EncoderDecoderTransformer`

Complete transformer architecture for sequence-to-sequence tasks.

**Constructor**

```python
EncoderDecoderTransformer(
    src_vocab_size: int,
    tgt_vocab_size: int,
    d_model: int = 512,
    num_heads: int = 8,
    num_encoder_layers: int = 6,
    num_decoder_layers: int = 6,
    d_ff: int = 2048,
    dropout: float = 0.1,
    max_seq_length: int = 5000
)
```

---

### Data Handling

#### Module: `src.data`

##### Class: `MultimodalDataset`

Base dataset class for handling image-text pairs.

**Constructor**

```python
MultimodalDataset(
    data_root: str,
    image_processor: Optional[Union[ImagePreprocessor, transforms.Compose]] = None,
    text_tokenizer = None,
    max_text_length: int = 77,
    split: str = "train",
    transform_image: Optional[Callable] = None,
    transform_text: Optional[Callable] = None,
    metadata_file: str = "metadata.json",
    image_key: str = "image_path",
    caption_key: str = "caption",
    label_key: Optional[str] = "label",
    image_dir: str = "images",
    limit_samples: Optional[int] = None,
    return_metadata: bool = False
)
```

**Example**:

```python
from src.data.multimodal_dataset import MultimodalDataset
from src.models.vision.image_preprocessing import ImagePreprocessor

# Create dataset
dataset = MultimodalDataset(
    data_root="./data/my_dataset",
    image_processor=ImagePreprocessor(image_size=224),
    text_tokenizer=tokenizer,
    max_text_length=77,
    split="train",
    limit_samples=1000
)

print(f"Dataset size: {len(dataset)}")

# Get a sample
sample = dataset[0]
print(f"Image shape: {sample['image'].shape}")
print(f"Text shape: {sample['text'].shape}")
print(f"Text mask shape: {sample['text_mask'].shape}")
```

**Methods**

###### `__getitem__(idx: int) -> Dict[str, torch.Tensor]`

Get a processed sample.

**Returns**: Dictionary with:
- `image`: Image tensor [channels, height, width]
- `text`: Text token IDs [max_text_length]
- `text_mask`: Attention mask [max_text_length]
- `label`: Class label (if available)
- `metadata`: Full metadata (if return_metadata=True)

###### `get_hard_negative(idx: int, neg_type: str = "same_class") -> int`

Get hard negative sample index for contrastive learning.

```python
# Get hard negatives
anchor_idx = 0
same_class_neg = dataset.get_hard_negative(anchor_idx, "same_class")
diff_class_neg = dataset.get_hard_negative(anchor_idx, "different_class")

anchor = dataset[anchor_idx]
hard_neg = dataset[same_class_neg]
```

##### Class: `EnhancedMultimodalDataset`

Enhanced dataset with semantic grouping and caching.

**Constructor**

```python
EnhancedMultimodalDataset(
    split: str = "train",
    image_preprocessor = None,
    tokenizer = None,
    max_text_length: int = 77,
    dataset_name: str = "flickr30k",
    synthetic_samples: int = 100,
    cache_dir: Optional[str] = None,
    max_samples: Optional[int] = None,
    captions_per_image: int = 1,
    min_samples_per_group: int = 2,
    max_samples_per_group: Optional[int] = None,
    cap_strategy: str = "random"
)
```

**Parameters**:
- `captions_per_image`: Number of captions to use per image (1-5)
- `min_samples_per_group`: Minimum samples per semantic group
- `max_samples_per_group`: Maximum samples per semantic group
- `cap_strategy`: Strategy for capping groups ("random")

**Example**:

```python
from src.data.multimodal_dataset import EnhancedMultimodalDataset

# Create enhanced dataset
dataset = EnhancedMultimodalDataset(
    split="train",
    image_preprocessor=image_preprocessor,
    tokenizer=tokenizer,
    max_text_length=77,
    dataset_name="flickr30k",
    max_samples=5000,
    captions_per_image=5,
    min_samples_per_group=2,
    max_samples_per_group=10
)

# Get sample with match_id for contrastive learning
sample = dataset[0]
print(f"Match ID: {sample['match_id']}")  # For semantic grouping
print(f"Raw text: {sample['raw_text']}")
```

---

### Training

#### Module: `src.training`

##### Class: `TrainingConfig`

Complete training configuration for multi-stage training.

**Constructor**

```python
TrainingConfig(
    project_name: str = "MultiModal_Insight_Engine",
    output_dir: str = "outputs",
    seed: int = 42,
    stages: List[StageConfig] = [],
    data_config: Dict[str, Any] = {},
    model_config: Dict[str, Any] = {}
)
```

**Example**:

```python
from src.configs.training_config import (
    TrainingConfig, StageConfig, LossConfig,
    OptimizerConfig, ComponentConfig
)

# Create multi-stage configuration
config = TrainingConfig(project_name="MyProject")

# Stage 1: Modality-specific learning
stage1 = StageConfig(
    name="modality_specific_learning",
    epochs=5,
    batch_size=64,
    optimizer=OptimizerConfig(lr=5e-5),
    losses=[
        LossConfig(name="contrastive_loss", weight=1.0),
        LossConfig(name="decorrelation_loss", weight=0.25)
    ],
    components=[
        ComponentConfig(name="vision_model", freeze=True, lr_multiplier=0.1),
        ComponentConfig(name="text_model", freeze=True, lr_multiplier=0.1),
        ComponentConfig(name="cross_attention", freeze=False, lr_multiplier=1.0)
    ]
)

config.stages.append(stage1)

# Save configuration
config.save("./configs/my_training_config.yaml")

# Load configuration
loaded_config = TrainingConfig.load("./configs/my_training_config.yaml")
```

**Methods**

###### `save(path: str) -> None`

Save configuration to JSON or YAML file.

###### `load(path: str) -> TrainingConfig` (classmethod)

Load configuration from file.

###### `create_default_multistage_config() -> TrainingConfig` (classmethod)

Create a default 3-stage training configuration.

```python
# Get default configuration
config = TrainingConfig.create_default_multistage_config()

# Modify as needed
config.stages[0].epochs = 10
config.stages[0].batch_size = 32

# Save
config.save("./configs/custom_config.yaml")
```

---

### Configuration

#### Module: `src.configs`

##### Class: `LossConfig`

Configuration for a loss function.

```python
LossConfig(
    name: str,
    weight: float = 1.0,
    params: Dict[str, Any] = {}
)
```

##### Class: `OptimizerConfig`

Configuration for optimizer.

```python
OptimizerConfig(
    name: str = "adamw",
    lr: float = 1e-4,
    weight_decay: float = 0.01,
    betas: tuple = (0.9, 0.999),
    eps: float = 1e-8
)
```

##### Class: `SchedulerConfig`

Configuration for learning rate scheduler.

```python
SchedulerConfig(
    name: str = "warmup_cosine",
    warmup_steps: int = 500,
    warmup_ratio: float = 0.1,
    min_lr_ratio: float = 0.0
)
```

##### Class: `ComponentConfig`

Configuration for model component.

```python
ComponentConfig(
    name: str,
    freeze: bool = False,
    lr_multiplier: float = 1.0
)
```

##### Class: `StageConfig`

Configuration for a training stage.

```python
StageConfig(
    name: str,
    losses: List[LossConfig] = [],
    epochs: int = 10,
    batch_size: int = 32,
    optimizer: OptimizerConfig = OptimizerConfig(),
    scheduler: SchedulerConfig = SchedulerConfig(),
    components: List[ComponentConfig] = [],
    early_stopping: bool = True,
    patience: int = 5,
    evaluation_metrics: List[str] = ["val_loss"],
    monitor_metric: str = "val_loss",
    monitor_mode: str = "min",
    clip_grad_norm: Optional[float] = 1.0,
    mixed_precision: bool = True,
    gradient_accumulation_steps: int = 1
)
```

---

## Common Workflows

### Workflow 1: Constitutional AI Evaluation

Evaluate text for safety using constitutional principles.

```python
from src.safety.constitutional import setup_default_framework
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# Setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = AutoModelForCausalLM.from_pretrained("gpt2").to(device)
tokenizer = AutoTokenizer.from_pretrained("gpt2")

# Create framework with AI-based evaluation
framework = setup_default_framework(
    model=model,
    tokenizer=tokenizer,
    device=device
)

# Evaluate text
texts = [
    "How can I learn Python programming?",
    "Tell me how to hack into a system",
    "What's the weather like today?"
]

results = framework.batch_evaluate(texts)

for i, (text, result) in enumerate(zip(texts, results)):
    print(f"\nText {i}: {text[:50]}...")
    print(f"  Flagged: {result['any_flagged']}")
    print(f"  Score: {result['weighted_score']:.2f}")
    print(f"  Violations: {result['flagged_principles']}")
    print(f"  Method: {result['evaluation_method']}")

# Get statistics
stats = framework.get_statistics()
print(f"\nOverall flagged rate: {stats['flagged_rate']:.2%}")
```

### Workflow 2: RLAIF Training (3 Phases)

Complete Constitutional AI training pipeline.

```python
from src.safety.constitutional.pipeline import ConstitutionalPipeline
from src.safety.constitutional import setup_default_framework
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch

# Setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = GPT2LMHeadModel.from_pretrained("gpt2").to(device)
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token

# Create constitutional framework
framework = setup_default_framework(model=model, tokenizer=tokenizer)

# Create pipeline
pipeline = ConstitutionalPipeline(
    base_model=model,
    tokenizer=tokenizer,
    device=device,
    constitutional_framework=framework,
    phase1_learning_rate=5e-5,
    phase2_learning_rate=1e-6,
    reward_model_learning_rate=1e-5
)

# Training prompts
training_prompts = [
    "How do I stay safe online?",
    "What is the best way to learn programming?",
    "Tell me about climate change",
    "How can I be more productive?",
    "What are the benefits of exercise?"
]

validation_prompts = [
    "What is cybersecurity?",
    "How do I learn data science?"
]

# Run complete training
results = pipeline.train(
    training_prompts=training_prompts,
    phase1_epochs=3,
    phase1_num_revisions=2,
    phase1_batch_size=4,
    phase2_ppo_steps=100,
    phase2_ppo_batch_size=4,
    phase2_reward_model_epochs=5,
    validation_prompts=validation_prompts,
    save_dir="./checkpoints/constitutional_ai"
)

print("\n=== Training Complete ===")
print(f"Phase 1 complete: {results['phase1_complete']}")
print(f"Phase 2 complete: {results['phase2_complete']}")
print(f"Final evaluation score: {results['final_evaluation']['avg_score']:.4f}")
print(f"Final violation rate: {results['final_evaluation']['violation_rate']:.2%}")
```

### Workflow 3: Multimodal Training

Train a multimodal model on image-text pairs.

```python
from src.models.model_factory import create_multimodal_model
from src.data.multimodal_dataset import EnhancedMultimodalDataset
from src.models.vision.image_preprocessing import ImagePreprocessor
from transformers import AutoTokenizer
import torch
from torch.utils.data import DataLoader

# Setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Create model
class ModelConfig:
    model_size = "large"
    use_pretrained = True
    use_pretrained_text = True
    vision_model = "vit-base"
    text_model = "bert-base-uncased"
    fusion_dim = 768
    fusion_type = "cross_attention"
    freeze_base_models = False

model = create_multimodal_model(ModelConfig(), device)

# Create dataset
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
image_processor = ImagePreprocessor(image_size=224)

train_dataset = EnhancedMultimodalDataset(
    split="train",
    image_preprocessor=image_processor,
    tokenizer=tokenizer,
    max_text_length=77,
    dataset_name="flickr30k",
    max_samples=5000
)

# Create dataloader
train_loader = DataLoader(
    train_dataset,
    batch_size=32,
    shuffle=True,
    num_workers=4
)

# Training loop
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
model.train()

for epoch in range(5):
    total_loss = 0
    for batch in train_loader:
        images = batch['image'].to(device)
        text = batch['text']['src'].to(device)
        text_mask = batch['text']['src_mask'].to(device)

        # Forward pass
        output = model(images, text, text_mask)

        # Compute contrastive loss
        image_emb = output['image_embedding']
        text_emb = output['text_embedding']

        # Normalize embeddings
        image_emb = torch.nn.functional.normalize(image_emb, dim=-1)
        text_emb = torch.nn.functional.normalize(text_emb, dim=-1)

        # Compute similarity
        logits = torch.mm(image_emb, text_emb.t()) / 0.07
        labels = torch.arange(len(images)).to(device)

        # Cross-entropy loss
        loss = torch.nn.functional.cross_entropy(logits, labels)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(train_loader)
    print(f"Epoch {epoch+1}/5 - Loss: {avg_loss:.4f}")

# Save model
torch.save(model.state_dict(), "./checkpoints/multimodal_model.pt")
```

---

## Configuration Reference

### Training Configuration

Complete training configuration structure:

```yaml
project_name: "MultiModal_Insight_Engine"
output_dir: "outputs"
seed: 42

data_config:
  dataset: "flickr30k"
  train_split_ratio: 0.8
  val_split_ratio: 0.1
  test_split_ratio: 0.1
  image_size: 224
  max_text_length: 77

model_config:
  vision_model: "google/vit-base-patch16-224"
  text_model: "bert-base-uncased"
  projection_dim: 512
  num_cross_attention_heads: 8
  cross_attention_dropout: 0.1

stages:
  - name: "modality_specific_learning"
    epochs: 5
    batch_size: 64

    optimizer:
      name: "adamw"
      lr: 5e-5
      weight_decay: 0.01
      betas: [0.9, 0.999]

    scheduler:
      name: "warmup_cosine"
      warmup_steps: 500
      warmup_ratio: 0.1
      min_lr_ratio: 0.0

    losses:
      - name: "contrastive_loss"
        weight: 1.0
        params:
          temperature: 0.07
      - name: "decorrelation_loss"
        weight: 0.25

    components:
      - name: "vision_model"
        freeze: true
        lr_multiplier: 0.1
      - name: "text_model"
        freeze: true
        lr_multiplier: 0.1
      - name: "cross_attention"
        freeze: false
        lr_multiplier: 1.0

    early_stopping: true
    patience: 5
    monitor_metric: "val_alignment_score"
    monitor_mode: "max"
    clip_grad_norm: 1.0
    mixed_precision: true
    gradient_accumulation_steps: 1
```

### Constitutional AI Configuration

```python
# Constitutional Framework Configuration
framework_config = {
    "name": "production_framework",
    "principles": [
        {
            "name": "harm_prevention",
            "weight": 2.0,
            "enabled": True
        },
        {
            "name": "truthfulness",
            "weight": 1.5,
            "enabled": True
        },
        {
            "name": "fairness",
            "weight": 1.0,
            "enabled": True
        },
        {
            "name": "autonomy_respect",
            "weight": 1.0,
            "enabled": True
        }
    ],
    "evaluation_method": "ai_evaluation",  # or "regex_heuristic"
    "track_history": True
}

# RLAIF Pipeline Configuration
rlaif_config = {
    "phase1": {
        "learning_rate": 5e-5,
        "epochs": 3,
        "num_revisions": 2,
        "batch_size": 8
    },
    "phase2": {
        "reward_model": {
            "learning_rate": 1e-5,
            "epochs": 5,
            "batch_size": 4
        },
        "ppo": {
            "learning_rate": 1e-6,
            "num_steps": 100,
            "batch_size": 4,
            "epochs_per_batch": 4,
            "clip_epsilon": 0.2,
            "kl_penalty": 0.1,
            "gamma": 0.99,
            "gae_lambda": 0.95,
            "temperature": 1.0
        }
    }
}
```

---

## Error Handling

### Common Errors and Solutions

**1. Dimension Mismatch Error**

```python
# Error: RuntimeError: The size of tensor a (768) must match the size of tensor b (512)

# Solution: Ensure fusion_dim matches model dimensions
config.fusion_dim = 768  # Match BERT-base and ViT-base
```

**2. Tokenizer Issues**

```python
# Error: KeyError: 'pad_token'

# Solution: Set pad token for GPT-2 style models
tokenizer.pad_token = tokenizer.eos_token
```

**3. Device Mismatch**

```python
# Error: RuntimeError: Expected all tensors to be on the same device

# Solution: Move all inputs to the same device
images = images.to(device)
text = text.to(device)
text_mask = text_mask.to(device)
```

**4. Out of Memory**

```python
# Solution: Reduce batch size and use gradient accumulation
config.batch_size = 16  # Reduce from 32
config.gradient_accumulation_steps = 2  # Effective batch size: 32
config.mixed_precision = True  # Use FP16
```

---

## Performance Tips

1. **Use Mixed Precision Training**
   ```python
   stage_config.mixed_precision = True
   ```

2. **Gradient Accumulation**
   ```python
   stage_config.gradient_accumulation_steps = 4
   ```

3. **Freeze Base Models Initially**
   ```python
   config.freeze_base_models = True  # Unfreeze in later stages
   ```

4. **Use Appropriate Batch Sizes**
   - Small GPU (< 8GB): batch_size = 4-8
   - Medium GPU (8-16GB): batch_size = 16-32
   - Large GPU (> 16GB): batch_size = 32-64

5. **Cache Datasets**
   ```python
   dataset = EnhancedMultimodalDataset(
       cache_dir="./cache",  # Enable caching
       ...
   )
   ```

---

## Version History

- **v1.0** (2025-11-17): Initial API reference
  - Constitutional AI framework
  - RLAIF training pipeline
  - Multimodal model architecture
  - Enhanced dataset classes
  - Multi-stage training configuration

---

## Support

For issues and questions:
- GitHub Issues: [Project Repository]
- Documentation: See `docs/` directory
- Examples: See `examples/` directory

---

**Last Updated**: 2025-11-17
**Document Version**: 1.0
