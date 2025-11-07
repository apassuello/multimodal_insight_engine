# Architecture Diagrams

Visual representations of the MultiModal Insight Engine architecture.

---

## Current Architecture (As-Is)

### System Overview

```mermaid
graph TD
    A[Application Layer<br/>30+ Demo Files] --> B[Orchestration Layer<br/>8 Trainer Types]
    B --> C[Domain Layer<br/>Models + Losses + Optimization]
    C --> D[Data Layer<br/>24+ Dataset Types]
    D --> E[Infrastructure<br/>Config + Utils + Safety]

    style B fill:#ffcccc
    style C fill:#ffffcc

    note1[⚠️ multimodal_trainer.py: 2,927 lines]
    note2[⚠️ 21 loss classes with duplication]

    B -.-> note1
    C -.-> note2
```

### Loss Function Architecture (Current - Problematic)

```mermaid
graph TD
    subgraph "20 Loss Files"
        A[contrastive_loss.py<br/>47KB]
        B[decoupled_contrastive_loss.py<br/>13KB]
        C[contrastive_learning.py<br/>25KB]
        D[vicreg_loss.py<br/>10KB]
        E[barlow_twins_loss.py<br/>11KB]
        F[memory_queue_contrastive_loss.py<br/>16KB]
        G[dynamic_temperature_contrastive_loss.py<br/>6KB]
        H[...]
    end

    I[loss_factory.py<br/>30KB] --> A
    I --> B
    I --> C
    I --> D
    I --> E
    I --> F
    I --> G

    J[SimpleContrastiveLoss<br/>187 lines] -.->|ANTI-PATTERN<br/>Defined inside factory!| I
    K[DecoupledContrastiveLoss] -.->|DUPLICATION<br/>Exists in TWO files!| B
    K -.-> C

    style I fill:#ffcccc
    style J fill:#ff6666
    style K fill:#ff6666
```

### Trainer Architecture (Current - God Object)

```mermaid
classDiagram
    class MultimodalTrainer {
        -model
        -optimizer
        -scheduler
        -loss_fn
        -train_dataloader
        -val_dataloader
        -checkpoint_dir
        -log_dir
        -metrics_tracker
        -early_stopping
        +train()
        +evaluate()
        +_log_metrics()
        +_save_checkpoint()
        +_load_checkpoint()
        +_validate()
        +_compute_metrics()
        +_balance_gradients()
        +_visualize_embeddings()
        ... 30+ more methods
        2,927 LINES!
    }

    class TransformerTrainer {
        ... similar structure
        1,025 lines
    }

    class VisionTransformerTrainer {
        ... similar structure
        454 lines
    }

    class LanguageModelTrainer {
        ... similar structure
        437 lines
    }

    note for MultimodalTrainer "❌ Violates SRP\n❌ No inheritance\n❌ 60% duplication\n❌ Hard to test"
```

### Configuration Management (Current - Chaos)

```mermaid
graph LR
    subgraph "Configuration Sources"
        A[training_config.py<br/>Dataclasses]
        B[argparse<br/>CLI args]
        C[Dict configs<br/>In factories]
        D[Hard-coded<br/>In code]
    end

    E[Model Creation] --> A
    E --> B
    E --> C
    E --> D

    F[Loss Creation] --> A
    F --> B
    F --> C
    F --> D

    G[Trainer Creation] --> A
    G --> B
    G --> C
    G --> D

    H{Which config<br/>takes precedence?} -.->|Unclear!| E

    style A fill:#ccffcc
    style B fill:#ffffcc
    style C fill:#ffcccc
    style D fill:#ff6666
```

---

## Proposed Architecture (To-Be)

### System Overview (Improved)

```mermaid
graph TD
    A[Application Layer<br/>Clean demos] --> B[Orchestration Layer<br/>BaseTrainer + Strategies]
    B --> C[Domain Layer<br/>Models + Losses + Optimization]
    C --> D[Data Layer<br/>Repository Pattern]
    D --> E[Infrastructure<br/>Unified Config]

    style B fill:#ccffcc
    style C fill:#ccffcc
    style E fill:#ccffcc

    note1[✅ Trainers < 500 lines]
    note2[✅ Loss hierarchy with 67% less code]
    note3[✅ Single source of truth]

    B -.-> note1
    C -.-> note2
    E -.-> note3
```

### Loss Function Architecture (Proposed)

```mermaid
classDiagram
    class BaseLoss {
        <<abstract>>
        +forward()
        +compute_similarity()
        #normalize_features()
        #temperature_scaling()
    }

    class ContrastiveLossBase {
        -temperature
        -sampling_strategy
        +forward()
        +sample_negatives()
        #compute_infonce()
    }

    class RegularizationLossBase {
        +forward()
        #compute_regularization()
    }

    class InfoNCELoss {
        +forward()
    }

    class DecoupledLoss {
        +forward()
    }

    class VICRegLoss {
        -sim_coeff
        -var_coeff
        -cov_coeff
        +forward()
    }

    class BarlowTwinsLoss {
        -lambda_coeff
        +forward()
    }

    class MemoryQueueDecorator {
        -queue_size
        +wrap_loss()
    }

    class HardNegativeMiningDecorator {
        -mining_ratio
        +wrap_loss()
    }

    BaseLoss <|-- ContrastiveLossBase
    BaseLoss <|-- RegularizationLossBase

    ContrastiveLossBase <|-- InfoNCELoss
    ContrastiveLossBase <|-- DecoupledLoss

    RegularizationLossBase <|-- VICRegLoss
    RegularizationLossBase <|-- BarlowTwinsLoss

    InfoNCELoss <.. MemoryQueueDecorator : decorates
    InfoNCELoss <.. HardNegativeMiningDecorator : decorates

    note for BaseLoss "✅ Shared logic extracted\n✅ 80% code reuse\n✅ Consistent interface"
```

### Trainer Architecture (Proposed - Template Method)

```mermaid
classDiagram
    class BaseTrainer {
        <<abstract>>
        -model
        -optimizer
        -device
        -checkpointer
        -logger
        +train() Template Method
        +train_epoch()* abstract
        +validate_epoch()* abstract
        #on_epoch_begin()
        #on_epoch_end()
        #should_stop_early()
    }

    class MultimodalTrainer {
        -loss_fn
        -modality_balancer
        +train_epoch()
        +validate_epoch()
        ~500 lines
    }

    class TransformerTrainer {
        -sequence_processor
        +train_epoch()
        +validate_epoch()
        ~400 lines
    }

    class CheckpointManager {
        +save()
        +load()
        +get_best()
    }

    class MetricsLogger {
        +log()
        +plot()
        +export()
    }

    class TrainingScheduler {
        +update_lr()
        +check_early_stop()
    }

    BaseTrainer <|-- MultimodalTrainer
    BaseTrainer <|-- TransformerTrainer

    BaseTrainer --> CheckpointManager : uses
    BaseTrainer --> MetricsLogger : uses
    BaseTrainer --> TrainingScheduler : uses

    note for BaseTrainer "✅ Template method pattern\n✅ 60% less duplication\n✅ Consistent interface"
    note for CheckpointManager "✅ Single Responsibility\n✅ Testable in isolation"
```

### Configuration Management (Proposed - Unified)

```mermaid
graph TD
    A[SystemConfig<br/>Pydantic BaseSettings] --> B[ModelConfig]
    A --> C[TrainingConfig]
    A --> D[DataConfig]
    A --> E[LossConfig]

    F[Environment Variables] --> A
    G[YAML/JSON Files] --> A
    H[CLI Arguments] --> A

    A --> I{Validation}
    I -->|Valid| J[Immutable Config Object]
    I -->|Invalid| K[Error with clear message]

    J --> L[Model Creation]
    J --> M[Trainer Creation]
    J --> N[Loss Creation]

    style A fill:#ccffcc
    style J fill:#ccffcc

    note1[✅ Single source of truth<br/>✅ Type-safe<br/>✅ Immutable<br/>✅ Validated]
    A -.-> note1
```

---

## Dependency Architecture

### Current Dependencies (Tightly Coupled)

```mermaid
graph LR
    A[Trainers] -->|direct import| B[Losses]
    A -->|direct import| C[Models]
    A -->|direct import| D[Datasets]

    B -->|dimension detection| C
    C -->|config reading| E[Configs]
    D -->|config reading| E

    F[Factories] -->|instantiate| A
    F -->|instantiate| B
    F -->|instantiate| C

    G[Utils] -.->|used by| A
    G -.->|used by| B
    G -.->|used by| C

    style A fill:#ffcccc
    style B fill:#ffcccc
    style C fill:#ffcccc

    note1[❌ Tight coupling<br/>❌ Hard to test<br/>❌ Circular dependencies]
```

### Proposed Dependencies (Loosely Coupled)

```mermaid
graph TD
    A[Application Layer] --> B[Interface Layer]
    B --> C[Domain Layer]
    C --> D[Infrastructure Layer]

    subgraph "Interface Layer"
        E[TrainerInterface]
        F[LossInterface]
        G[DataRepositoryInterface]
    end

    subgraph "Domain Layer"
        H[Trainers]
        I[Losses]
        J[Models]
    end

    subgraph "Infrastructure Layer"
        K[Config]
        L[Logging]
        M[Checkpointing]
    end

    H -.->|implements| E
    I -.->|implements| F
    J -.->|uses| G

    H --> F
    H --> G

    style E fill:#ccffcc
    style F fill:#ccffcc
    style G fill:#ccffcc

    note1[✅ Dependency Inversion<br/>✅ Testable with mocks<br/>✅ No circular dependencies]
```

---

## Data Flow Architecture

### Training Pipeline Flow

```mermaid
sequenceDiagram
    participant App as Application
    participant Config as Configuration
    participant Factory as Factory
    participant Trainer as Trainer
    participant Model as Model
    participant Loss as Loss
    participant Data as DataLoader

    App->>Config: Load config
    Config-->>App: Validated config

    App->>Factory: Create trainer
    Factory->>Model: Create model
    Factory->>Loss: Create loss
    Factory->>Data: Create dataloaders
    Factory-->>App: Configured trainer

    App->>Trainer: train()

    loop Each Epoch
        Trainer->>Data: Get batch
        Data-->>Trainer: batch

        Trainer->>Model: forward(batch)
        Model-->>Trainer: features

        Trainer->>Loss: compute(features)
        Loss-->>Trainer: loss

        Trainer->>Trainer: backward()
        Trainer->>Trainer: optimizer.step()

        Trainer->>Trainer: log_metrics()
        Trainer->>Trainer: save_checkpoint()
    end

    Trainer-->>App: Training complete
```

### Multi-Stage Training Flow

```mermaid
stateDiagram-v2
    [*] --> Stage1: Initialize

    state Stage1 {
        [*] --> FreezeEncoders
        FreezeEncoders --> TrainProjections
        TrainProjections --> ContrastiveLearning
        ContrastiveLearning --> [*]
    }

    Stage1 --> Stage2: Transition

    state Stage2 {
        [*] --> UnfreezeEncoders
        UnfreezeEncoders --> TrainCrossAttention
        TrainCrossAttention --> MemoryBankLearning
        MemoryBankLearning --> [*]
    }

    Stage2 --> Stage3: Transition

    state Stage3 {
        [*] --> FullFineTuning
        FullFineTuning --> HardNegativeMining
        HardNegativeMining --> [*]
    }

    Stage3 --> [*]: Complete

    note right of Stage1
        Focus: Alignment
        Loss: Simple Contrastive
        LR: 5e-5
    end note

    note right of Stage2
        Focus: Fusion
        Loss: Memory Queue
        LR: 1e-4
    end note

    note right of Stage3
        Focus: Fine-tuning
        Loss: Hard Negative Mining
        LR: 5e-6
    end note
```

---

## Testing Architecture

### Current Testing Structure

```mermaid
graph TD
    A[tests/] --> B[test_models/]
    A --> C[test_training/]
    A --> D[test_data/]

    E[Source Code] -.->|~60% coverage| A

    style A fill:#ffffcc

    note1[⚠️ Coverage below target<br/>⚠️ No integration tests<br/>⚠️ Limited mocking]
```

### Proposed Testing Structure

```mermaid
graph TD
    A[tests/] --> B[unit/]
    A --> C[integration/]
    A --> D[characterization/]

    B --> E[test_models/]
    B --> F[test_losses/]
    B --> G[test_trainers/]
    B --> H[test_data/]

    C --> I[test_training_pipeline/]
    C --> J[test_multimodal_integration/]

    D --> K[test_baseline_behavior/]

    L[Mocks & Fixtures] -.->|injected into| B
    L -.->|injected into| C

    M[Source Code] -.->|>85% coverage| A

    style A fill:#ccffcc
    style L fill:#ccffcc

    note1[✅ Comprehensive coverage<br/>✅ Integration tests<br/>✅ Proper mocking<br/>✅ Regression protection]
```

---

## Deployment Architecture

### Training Infrastructure

```mermaid
graph TD
    A[Training Script] --> B{Device Selection}

    B -->|Single GPU| C[CUDA Training]
    B -->|Multi GPU| D[Distributed Training]
    B -->|CPU/MPS| E[CPU/Apple Silicon]

    C --> F[Mixed Precision]
    D --> F

    F --> G[Gradient Accumulation]
    G --> H[Checkpoint Saving]

    H --> I[(Model Registry)]

    J[TensorBoard] -.->|monitors| F
    K[Weights & Biases] -.->|monitors| F

    style D fill:#ffffcc
    note1[⚠️ Not yet implemented]
    D -.-> note1
```

### Model Serving (Future)

```mermaid
graph LR
    A[Client] --> B[API Gateway]
    B --> C[Model Server]

    C --> D[(Model Cache)]
    C --> E[Inference Engine]

    E --> F[Preprocessing]
    E --> G[Model Forward]
    E --> H[Postprocessing]

    I[(Checkpoint Storage)] -.->|loads| C

    style B fill:#ffffcc
    style C fill:#ffffcc
    style E fill:#ffffcc

    note1[🔵 Future enhancement]
```

---

## Component Interaction Map

```mermaid
graph TB
    subgraph "Application Layer"
        A1[Training Scripts]
        A2[Demo Scripts]
        A3[Evaluation Scripts]
    end

    subgraph "Orchestration Layer"
        B1[BaseTrainer]
        B2[MultimodalTrainer]
        B3[MultistageTrainer]
        B4[Training Strategies]
    end

    subgraph "Domain Layer"
        C1[Models]
        C2[Losses]
        C3[Optimizers]
    end

    subgraph "Data Layer"
        D1[Datasets]
        D2[Tokenizers]
        D3[Augmentation]
    end

    subgraph "Infrastructure Layer"
        E1[Configuration]
        E2[Logging]
        E3[Checkpointing]
        E4[Metrics]
    end

    A1 --> B2
    A2 --> B2
    A3 --> B2

    B2 -.->|inherits| B1
    B3 -.->|inherits| B1
    B2 --> B4

    B1 --> C1
    B1 --> C2
    B1 --> C3

    B1 --> D1

    B1 --> E1
    B1 --> E2
    B1 --> E3
    B1 --> E4

    C1 --> D2
    D1 --> D2
    D1 --> D3
```

---

## Migration Path Visualization

### Phase 1: Foundation (Weeks 1-2)

```mermaid
gantt
    title Foundation Phase
    dateFormat  YYYY-MM-DD
    section Critical Fixes
    Remove DecoupledContrastiveLoss duplication    :done, 2025-11-08, 1d
    Extract SimpleContrastiveLoss from factory     :done, 2025-11-08, 1d
    Create BaseTrainer class                       :active, 2025-11-09, 3d
    Document current architecture                  :2025-11-10, 2d
```

### Phase 2: Consolidation (Weeks 3-5)

```mermaid
gantt
    title Consolidation Phase
    dateFormat  YYYY-MM-DD
    section Refactoring
    Refactor loss function hierarchy               :2025-11-11, 14d
    Decompose MultimodalTrainer                    :2025-11-18, 14d
    Unify configuration                            :2025-11-25, 7d
```

### Phase 3: Enhancement (Weeks 6-8)

```mermaid
gantt
    title Enhancement Phase
    dateFormat  YYYY-MM-DD
    section Modern Patterns
    Implement template method for trainers         :2025-12-02, 7d
    Add repository pattern                         :2025-12-09, 7d
    Create callback system                         :2025-12-16, 3d
    Add ADRs                                       :2025-12-19, 1d
```

---

## Success Metrics Dashboard

```mermaid
graph LR
    A[Code Metrics] --> A1[File Size]
    A --> A2[Duplication]
    A --> A3[Complexity]

    A1 -.->|Current: 2927 lines| B1[❌]
    A1 -.->|Target: <800 lines| C1[Target]

    A2 -.->|Current: 35%| B2[❌]
    A2 -.->|Target: <10%| C2[Target]

    A3 -.->|Current: >50| B3[❌]
    A3 -.->|Target: <15| C3[Target]

    D[Quality Metrics] --> D1[Test Coverage]
    D --> D2[Code Review Time]
    D --> D3[Bug Rate]

    D1 -.->|Current: 60%| E1[⚠️]
    D1 -.->|Target: >85%| F1[Target]

    D2 -.->|Current: 2 hours| E2[⚠️]
    D2 -.->|Target: <30 min| F2[Target]

    style B1 fill:#ff6666
    style B2 fill:#ff6666
    style B3 fill:#ff6666
    style E1 fill:#ffcc66
    style E2 fill:#ffcc66
```

---

## Glossary

| Symbol | Meaning |
|--------|---------|
| ✅ | Implemented and working well |
| ⚠️ | Partially implemented or needs improvement |
| ❌ | Not implemented or problematic |
| 🔴 | Critical priority |
| 🟡 | High priority |
| 🟢 | Medium priority |
| 🔵 | Low priority / Future |

---

**Last Updated**: 2025-11-07
**See Also**:
- `ARCHITECTURE_REVIEW.md` - Detailed analysis
- `ARCHITECTURE_QUICK_FIXES.md` - Action items
- `docs/adr/` - Architecture decisions
