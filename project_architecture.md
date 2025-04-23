# MultiModal Insight Engine Architecture
## Project Overview
This document provides an overview of the project structure, key components, and architecture.
Generated from 84 source files.

## Directory Structure
```
📁 src/
    📁 optimization/
        📄 mixed_precision.py
        📄 quantization.py
        📄 pruning.py
        📄 benchmarking.py
    📁 training/
        📄 loss_factory.py
        📄 metrics.py
        📄 contrastive_learning.py
        📄 transformer_utils.py
        📄 vision_transformer_trainer.py
        📄 optimizers.py
        📄 language_model_trainer.py
        📄 transformer_trainer.py
        📄 losses.py
        📄 trainer.py
        📄 multimodal_trainer.py
        📄 joint_bpe_training.py
    📁 utils/
        📄 model_utils.py
        📄 logging.py
        📄 argument_configs.py
        📄 config.py
        📄 visualization.py
        📄 list_models.py
        📄 profiling.py
    📁 models/
        📄 attention.py
        📄 activations.py
        📄 feed_forward.py
        📁 vision/
            📄 image_preprocessing.py
            📄 multimodal_integration.py
            📄 vision_transformer.py
            📄 cross_modal_attention.py
            📄 patch_embedding.py
        📄 base_model.py
        📁 multimodal/
            📄 fusion.py
        📄 embeddings.py
        📄 transformer.py
        📄 text_generation.py
        📄 layers.py
        📄 model_factory.py
        📁 pretrained/
            📄 vision_transformer.py
            📄 huggingface_wrapper.py
            📄 clip_model.py
            📄 base_wrapper.py
            📄 model_registry.py
            📄 adapters.py
        📄 positional.py
    📁 safety/
        📄 harness.py
        📄 integration.py
        📄 utils.py
        📄 filter.py
        📄 evaluator.py
        📁 red_teaming/
            📄 framework.py
            📄 generators.py
            📄 model_loader.py
            📄 prompt_injection.py
            📄 evaluator.py
    📁 evaluation/
        📄 translation_metrics.py
        📄 language_model_evaluation.py
        📄 inference_demo.py
    📁 data/
        📄 wmt_dataset.py
        📄 curriculum_dataset.py
        📄 image_dataset.py
        📄 wmt_dataloader.py
        📄 combined_dataset.py
        📄 iwslt_dataset.py
        📄 language_modeling.py
        📄 europarl_dataset.py
        📄 combined_wmt_translation_dataset.py
        📄 combined_translation_dataset.py
        📄 multimodal_data_utils.py
        📁 tokenization/
            📄 optimized_bpe_tokenizer.py
            📄 base_tokenizer.py
            📄 utils.py
            📄 preprocessing.py
            📄 bpe_tokenizer.py
            📄 vocabulary.py
            📄 turbo_bpe_preprocessor.py
            📄 simple_tokenizer.py
        📄 multimodal_dataset.py
        📄 preprocessing.py
        📄 sequence_data.py
        📄 opensubtitles_dataset.py
        📄 dataloader.py
        📄 dataset_wrapper.py
        📄 wikipedia_dataset.py
```

## Module Summary
| Module | Purpose | Complexity |
|--------|---------|------------|
| `src/data/combined_dataset.py` | No metadata function available | N/A |
| `src/data/combined_translation_dataset.py` | Implements dataset class for combining multiple translation datasets with configurable sampling | 2 |
| `src/data/combined_wmt_translation_dataset.py` | No metadata function available | N/A |
| `src/data/curriculum_dataset.py` | Implements curriculum learning for translation datasets, gradually increasing difficulty during t... | 7 |
| `src/data/dataloader.py` | Provides utilities for handling multimodal data with PyTorch DataLoader | 3 |
| `src/data/dataset_wrapper.py` | Provides dataset wrappers to standardize interfaces for different dataset types | 3 |
| `src/data/europarl_dataset.py` | Provides a dataset class for loading and preprocessing Europarl parallel corpus data | 4 |
| `src/data/image_dataset.py` | Provides dataset functionality for loading and preprocessing image data for vision transformer mo... | 4 |
| `src/data/iwslt_dataset.py` | Provides a dataset class for loading and preprocessing IWSLT dataset for machine translation | 5 |
| `src/data/language_modeling.py` | Implements dataset and dataloaders for language modeling tasks with efficient tokenization and ba... | 5 |
| `src/data/multimodal_data_utils.py` | Utilities for multimodal data loading and processing | 7 |
| `src/data/multimodal_dataset.py` | No metadata function available | N/A |
| `src/data/opensubtitles_dataset.py` | Provides a dataset class for loading and preprocessing OpenSubtitles parallel corpus data for mac... | 4 |
| `src/data/preprocessing.py` | Provides data preprocessing utilities for time series data and machine learning datasets | 4 |
| `src/data/sequence_data.py` | Provides dataset and dataloader utilities for transformer sequence-to-sequence tasks | 6 |
| `src/data/tokenization/base_tokenizer.py` | Defines the abstract base class for all tokenizers in the system with standard interface | 2 |
| `src/data/tokenization/bpe_tokenizer.py` | Implements Byte Pair Encoding tokenizer for subword tokenization with merge operations | 7 |
| `src/data/tokenization/optimized_bpe_tokenizer.py` | Implements an optimized Byte Pair Encoding tokenizer with smart caching and batch processing | 9 |
| `src/data/tokenization/preprocessing.py` | Provides text preprocessing utilities for tokenization including Unicode normalization and text c... | 3 |
| `src/data/tokenization/simple_tokenizer.py` | No metadata function available | N/A |
| `src/data/tokenization/turbo_bpe_preprocessor.py` | No metadata function available | N/A |
| `src/data/tokenization/utils.py` | No metadata function available | N/A |
| `src/data/tokenization/vocabulary.py` | Implements a flexible vocabulary system for mapping between tokens and indices with special token... | 6 |
| `src/data/wikipedia_dataset.py` | Provides a dataset class for loading and preprocessing Wikipedia Web2M data from TFRecord format | 5 |
| `src/data/wmt_dataloader.py` | Provides a data loader for WMT (Workshop on Machine Translation) parallel corpus | 3 |
| `src/data/wmt_dataset.py` | Provides a dataset class for loading and preprocessing WMT dataset for machine translation | 4 |
| `src/evaluation/inference_demo.py` | Inference demo utilities for multimodal models | 7 |
| `src/evaluation/language_model_evaluation.py` | Provides comprehensive evaluation utilities for language models, including perplexity calculation... | 7 |
| `src/evaluation/translation_metrics.py` | Implements standard evaluation metrics for machine translation tasks including BLEU and TER scoring | 3 |
| `src/models/activations.py` | Implements various activation functions used in the transformer architecture | 1 |
| `src/models/attention.py` | Implements various attention mechanisms for transformer architectures | 8 |
| `src/models/base_model.py` | Provides the foundational base class for all neural network models in the MultiModal Insight Engine | 3 |
| `src/models/embeddings.py` | Implements token embedding layers for transformer models with proper initialization and scaling | 2 |
| `src/models/feed_forward.py` | Implements various feed-forward neural network architectures with modern features for flexible mo... | 6 |
| `src/models/layers.py` | Implements fundamental neural network layers with advanced features for transformer architectures | 4 |
| `src/models/model_factory.py` | Factory functions for creating and configuring different types of models | 8 |
| `src/models/multimodal/fusion.py` | No metadata function available | N/A |
| `src/models/positional.py` | Implements various positional encoding schemes for transformer models to handle sequence order in... | 7 |
| `src/models/pretrained/adapters.py` | Implements adapter layers for fine-tuning frozen pretrained models efficiently | 3 |
| `src/models/pretrained/base_wrapper.py` | Provides a base wrapper class for pretrained models with consistent interface | 4 |
| `src/models/pretrained/clip_model.py` | Provides a wrapper for OpenAI CLIP multimodal models with standardized interface | 5 |
| `src/models/pretrained/huggingface_wrapper.py` | No metadata function available | N/A |
| `src/models/pretrained/model_registry.py` | Implements a registry pattern for accessing and instantiating pretrained models | 3 |
| `src/models/pretrained/vision_transformer.py` | Provides a wrapper for Hugging Face Vision Transformer models with standardized interface | 3 |
| `src/models/text_generation.py` | Provides utilities for text generation using language models | 7 |
| `src/models/transformer.py` | Implements transformer models for sequence processing tasks | 9 |
| `src/models/vision/cross_modal_attention.py` | No metadata function available | N/A |
| `src/models/vision/image_preprocessing.py` | Provides utilities for preprocessing images for vision transformer models | 6 |
| `src/models/vision/multimodal_integration.py` | Implements models for combining vision and text modalities in a unified architecture | 7 |
| `src/models/vision/patch_embedding.py` | Implements patch embedding for Vision Transformer (ViT) models | 5 |
| `src/models/vision/vision_transformer.py` | Implements Vision Transformer (ViT) architecture for image classification tasks | 8 |
| `src/optimization/benchmarking.py` | Provides a framework for measuring and comparing model optimization techniques. | 6 |
| `src/optimization/mixed_precision.py` | Converts models to use mixed precision formats for training and inference. | 5 |
| `src/optimization/pruning.py` | Implements various pruning techniques for neural networks. | 7 |
| `src/optimization/quantization.py` | Implements various quantization techniques for neural networks. | 8 |
| `src/safety/evaluator.py` | Provides a comprehensive framework for evaluating model outputs for safety concerns with configur... | 8 |
| `src/safety/filter.py` | Implements safety filtering mechanisms for validating model inputs and filtering outputs based on... | 8 |
| `src/safety/harness.py` | Provides a test harness for evaluating model safety on benchmark test cases, including test suite... | 9 |
| `src/safety/integration.py` | Provides integration layer for augmenting models with safety mechanisms, including input validati... | 8 |
| `src/safety/red_teaming/evaluator.py` | Provides evaluation tools for measuring model robustness against adversarial attacks | 8 |
| `src/safety/red_teaming/framework.py` | Provides a framework for conducting red teaming exercises on language models | 7 |
| `src/safety/red_teaming/generators.py` | Provides strategies for generating adversarial inputs to test model robustness and safety | 7 |
| `src/safety/red_teaming/model_loader.py` | No metadata function available | N/A |
| `src/safety/red_teaming/prompt_injection.py` | No metadata function available | N/A |
| `src/safety/utils.py` | Provides utility functions and constants for safety evaluation, including pattern matching, scori... | 9 |
| `src/training/contrastive_learning.py` | No metadata function available | N/A |
| `src/training/joint_bpe_training.py` | Implements joint BPE tokenizer training for multilingual text processing in machine translation t... | 3 |
| `src/training/language_model_trainer.py` | Implements a specialized trainer for language modeling tasks with support for causal language mod... | 8 |
| `src/training/loss_factory.py` | Factory functions for creating and configuring loss functions | 7 |
| `src/training/losses.py` | Implements custom loss functions for model training with support for label smoothing and weighted... | 4 |
| `src/training/metrics.py` | Implements common training metrics for model evaluation with support for various tasks | 7 |
| `src/training/multimodal_trainer.py` | No metadata function available | N/A |
| `src/training/optimizers.py` | Implements custom optimizers and learning rate schedulers for model training | 8 |
| `src/training/trainer.py` | Provides a generic, flexible training loop for PyTorch models with support for callbacks and earl... | 6 |
| `src/training/transformer_trainer.py` | Implements a specialized trainer for transformer models with support for encoder-decoder architec... | 8 |
| `src/training/transformer_utils.py` | Provides utility functions and classes for transformer model training, including attention maskin... | 6 |
| `src/training/vision_transformer_trainer.py` | Provides a specialized trainer for Vision Transformer models with advanced training techniques | 8 |
| `src/utils/argument_configs.py` | Argument configuration utilities for multimodal training scripts | 4 |
| `src/utils/config.py` | Provides configuration management utilities with file loading and environment support | 3 |
| `src/utils/list_models.py` | Provides utilities for listing and retrieving information about available models | 4 |
| `src/utils/logging.py` | Provides custom logging functionality with configurable file and console output | 4 |
| `src/utils/model_utils.py` | No metadata function available | N/A |
| `src/utils/profiling.py` | Provides utilities for profiling and benchmarking PyTorch models with comprehensive performance a... | 9 |
| `src/utils/visualization.py` | Provides visualization utilities for model performance, attention patterns, embeddings, and multi... | 7 |


## Complexity Analysis
**Top 10 Most Complex Modules:**

```
src/utils/profiling.py                           | ██████████████████████████████ (9)
src/models/transformer.py                        | ██████████████████████████████ (9)
src/safety/harness.py                            | ██████████████████████████████ (9)
src/safety/utils.py                              | ██████████████████████████████ (9)
src/data/tokenization/optimized_bpe_tokenizer.py | ██████████████████████████████ (9)
src/optimization/quantization.py                 | ██████████████████████████ (8)
src/training/vision_transformer_trainer.py       | ██████████████████████████ (8)
src/training/optimizers.py                       | ██████████████████████████ (8)
src/training/language_model_trainer.py           | ██████████████████████████ (8)
src/training/transformer_trainer.py              | ██████████████████████████ (8)
```

**Average Module Complexity:** 5.64

## Dependencies
**External Dependencies:**

| Library | Usage Count |

|---------|-------------|

| torch | 47 |
| numpy | 20 |
| matplotlib | 9 |
| tqdm | 9 |
| logging | 8 |
| json | 8 |
| random | 5 |
| typing | 4 |
| re | 4 |
| os | 3 |
| seaborn | 3 |
| nltk | 2 |
| argparse | 2 |
| sklearn | 2 |
| psutil | 2 |
| transformers | 2 |
| PIL | 2 |
| datasets | 2 |
| collections.Counter | 2 |
| time | 1 |
| src.data.tokenization | 1 |
| huggingface_hub | 1 |
| pandas | 1 |
| timm | 1 |
| torchvision | 1 |
| math | 1 |
| open_clip | 1 |
| datetime | 1 |
| collections | 1 |
| pathlib | 1 |
| torch.utils.data | 1 |
| tensorflow | 1 |
| threading | 1 |
| abc | 1 |
| unicodedata | 1 |
| html | 1 |
| multiprocessing.Pool | 1 |

## Key Components
### MixedPrecisionConverter (`optimization/mixed_precision.py`)
Converts models to use mixed precision formats.

**Inherits from:** ``

**Dependencies:**
- `torch`
- `typing`
- `logging`

**Key Methods:**
- `__init__`: Initialize the mixed precision converter.
- `convert_to_mixed_precision`: Convert the model to use mixed precision.
- `restore_original_precision`: Restore the model to its original precision.


### MixedPrecisionWrapper (`optimization/mixed_precision.py`)
Wrapper for mixed precision inference with autocast.

**Inherits from:** `nn.Module`

**Dependencies:**
- `torch`
- `typing`
- `logging`

**Key Methods:**
- `__init__`: Initialize the mixed precision wrapper.
- `forward`: Forward pass with automatic mixed precision.
- `__getattr__`: Delegate attribute access to the wrapped model.


### QuantizationConfig (`optimization/quantization.py`)
Configuration class for model quantization settings.

**Inherits from:** ``

**Dependencies:**
- `torch`
- `typing`

**Key Methods:**
- `__init__`: Initialize quantization configuration.
- `__str__`: String representation of the configuration.


### ModelOptimizer (`optimization/quantization.py`)
Base class for model optimization techniques.

**Inherits from:** ``

**Dependencies:**
- `torch`
- `typing`
- `logging`

**Key Methods:**
- `__init__`: Initialize the model optimizer.
- `optimize`: Apply optimization to the model.
- `restore_original`: Restore the model to its original state.
- `get_size_info`: Get information about model size before and after optimization.


### DynamicQuantizer (`optimization/quantization.py`)
Implements dynamic quantization for PyTorch models.

**Inherits from:** `ModelOptimizer`

**Dependencies:**
- `torch`
- `typing`
- `logging`

**Key Methods:**
- `__init__`: Initialize the dynamic quantizer.
- `optimize`: Apply dynamic quantization to the model.
- `_fuse_modules`: Fuse modules for improved quantization where applicable.


### StaticQuantizer (`optimization/quantization.py`)
Implements static quantization for PyTorch models.

**Inherits from:** `ModelOptimizer`

**Dependencies:**
- `torch`
- `typing`
- `logging`

**Key Methods:**
- `__init__`: Initialize the static quantizer.
- `optimize`: Apply static quantization to the model.
- `_calibrate_model`: Calibrate the model for static quantization.
- `_fuse_modules`: Fuse modules for improved quantization where applicable.


### PruningConfig (`optimization/pruning.py`)
Configuration class for model pruning settings.

**Inherits from:** ``

**Dependencies:**
- `torch`
- `typing`

**Key Methods:**
- `__init__`: Initialize pruning configuration.
- `__str__`: String representation of the configuration.


### ModelPruner (`optimization/pruning.py`)
Implements various pruning techniques for neural networks.

**Inherits from:** ``

**Dependencies:**
- `torch`
- `typing`
- `logging`

**Key Methods:**
- `__init__`: Initialize the model pruner.
- `prune_model`: Apply pruning to the model.
- `restore_model`: Restore the model to its original unpruned state.
- `get_pruning_info`: Get information about pruning results.


### OptimizationBenchmark (`optimization/benchmarking.py`)
Framework for measuring and comparing model optimization techniques.

**Inherits from:** ``

**Dependencies:**
- `torch`
- `typing`
- `json`
- `os`
- `numpy`
- `matplotlib`

**Key Methods:**
- `__init__`: Initialize the optimization benchmark.
- `benchmark_original_model`: Benchmark the original unoptimized model.
- `benchmark_optimized_model`: Benchmark an optimized model.
- `compare_optimizations`: Compare all benchmarked optimizations.
- `save_results`: Save benchmark results to a file.
- `generate_report`: Generate a report of the benchmark results.


### Accuracy (`training/metrics.py`)
Computes classification accuracy with support for top-k accuracy

**Dependencies:**
- `torch`

**Key Methods:**
- `update`: Updates the accuracy metric with new predictions and targets
- `compute`: Computes the current accuracy value


### Perplexity (`training/metrics.py`)
Computes perplexity for language models

**Dependencies:**
- `torch`
- `numpy`

**Key Methods:**
- `update`: Updates the perplexity metric with new loss values
- `compute`: Computes the current perplexity value


### F1Score (`training/metrics.py`)
Computes F1 score for classification tasks

**Dependencies:**
- `torch`

**Key Methods:**
- `update`: Updates the F1 score metric with new predictions
- `compute`: Computes the current F1 score


### BLEUScore (`training/metrics.py`)
Computes BLEU score for machine translation

**Dependencies:**
- `nltk`

**Key Methods:**
- `update`: Updates the BLEU score metric with new translations
- `compute`: Computes the current BLEU score


### LabelSmoothing (`training/transformer_utils.py`)
Implements label smoothing loss for improved model generalization

**Inherits from:** `nn.Module`

**Dependencies:**
- `torch`
- `torch.nn`

**Key Methods:**
- `forward`: Computes smoothed loss with proper handling of padding tokens


### VisionTransformerTrainer (`training/vision_transformer_trainer.py`)
Specialized trainer for Vision Transformer models with mixup/cutmix augmentation support

**Dependencies:**
- `torch`
- `torch.nn`
- `VisionTransformer`
- `matplotlib`
- `numpy`
- `tqdm`

**Key Methods:**
- `train_epoch`: Train the model for one epoch with mixup/cutmix augmentation support
- `validate`: Validate the model on validation dataset
- `train`: Train the model for specified number of epochs with early stopping
- `save_checkpoint`: Save a checkpoint of the model and training state
- `load_checkpoint`: Load a checkpoint of the model and training state
- `_mixup_data`: Perform mixup data augmentation
- `_cutmix_data`: Perform cutmix data augmentation


### AdamW (`training/optimizers.py`)
AdamW optimizer with improved weight decay handling and gradient clipping

**Inherits from:** `optim.AdamW`

**Dependencies:**
- `torch`
- `torch.optim`

**Key Methods:**
- `step`: Performs a single optimization step with optional gradient clipping


### OneCycleLR (`training/optimizers.py`)
One-cycle learning rate scheduler for fast training

**Inherits from:** `_LRScheduler`

**Dependencies:**
- `torch`
- `torch.optim.lr_scheduler`

**Key Methods:**
- `get_lr`: Computes learning rates based on the one-cycle policy
- `step`: Performs a scheduler step and updates learning rates


### CosineAnnealingLR (`training/optimizers.py`)
Cosine annealing learning rate scheduler with warm restarts

**Inherits from:** `_LRScheduler`

**Dependencies:**
- `torch`
- `torch.optim.lr_scheduler`

**Key Methods:**
- `get_lr`: Computes learning rates based on cosine annealing


### LanguageModelTrainer (`training/language_model_trainer.py`)
Main trainer class for language model training with comprehensive training and evaluation capabilities

**Dependencies:**
- `torch`
- `torch.nn`
- `torch.nn.functional`
- `numpy`
- `matplotlib`
- `tqdm`

**Key Methods:**
- `train`: Main training loop with support for validation and checkpointing
- `evaluate`: Evaluates the model on validation data and returns loss and perplexity
- `save_model`: Saves the model and training state to disk
- `load_model`: Loads a saved model and training state from disk
- `plot_training_curves`: Visualizes training metrics including loss, perplexity, and learning rate


### TransformerTrainer (`training/transformer_trainer.py`)
Main trainer class for transformer model training with comprehensive training and evaluation capabilities

**Dependencies:**
- `torch`
- `torch.nn`
- `numpy`
- `matplotlib`
- `tqdm`
- `transformer_utils`

**Key Methods:**
- `train`: Main training loop with support for validation and early stopping
- `train_epoch`: Trains the model for a single epoch with progress tracking
- `validate`: Evaluates the model on validation data and returns loss metrics
- `get_lr_scheduler`: Creates learning rate scheduler with warmup and decay strategies
- `save_checkpoint`: Saves model checkpoint with training state
- `load_checkpoint`: Loads model checkpoint and training state
- `plot_learning_rate`: Visualizes the learning rate schedule
- `plot_training_history`: Visualizes training and validation metrics over time
- `plot_epoch_metrics`: Visualizes detailed metrics for a specific epoch


### CrossEntropyLoss (`training/losses.py`)
Cross-entropy loss with label smoothing for classification tasks

**Inherits from:** `nn.Module`

**Dependencies:**
- `torch`
- `torch.nn`
- `torch.nn.functional`

**Key Methods:**
- `forward`: Computes cross-entropy loss with label smoothing and optional sample weights


### MeanSquaredError (`training/losses.py`)
Mean squared error loss with support for weighted samples and gradient clipping

**Inherits from:** `nn.Module`

**Dependencies:**
- `torch`
- `torch.nn`

**Key Methods:**
- `forward`: Computes MSE loss with optional sample weights and gradient clipping


### LogManager (`utils/logging.py`)
Central logging manager that configures and provides logger instances

**Dependencies:**
- `logging`
- `os`
- `sys`

**Key Methods:**
- `get_logger`: Creates or retrieves a logger with the specified name and level
- `configure_file_logging`: Sets up file logging to the specified directory


### ConfigManager (`utils/config.py`)
Manages application configuration with support for file and environment loading

**Dependencies:**
- `os`
- `json`

**Key Methods:**
- `load_from_file`: Load configuration from a JSON file
- `get`: Get a configuration value with optional default
- `set`: Set a configuration value
- `save_to_file`: Save current configuration to a JSON file


### ModelProfiler (`utils/profiling.py`)
Utility for profiling PyTorch models with execution time, memory usage, and layer-wise analysis

**Inherits from:** ``

**Dependencies:**
- `torch`
- `numpy`
- `matplotlib`
- `psutil`
- `pandas`
- `seaborn`

**Key Methods:**
- `__init__`: Initialize the profiler with a model and target device
- `measure_execution_time`: Measure the execution time of a forward pass
- `measure_memory_usage`: Measure the memory usage during a forward pass
- `generate_report`: Generate a comprehensive profiling report with all metrics
- `plot_metrics`: Create visualization plots for performance metrics
- `profile_with_pytorch_profiler`: Profile the model using PyTorch's built-in profiler
- `trace_memory_by_layer`: Trace memory usage by layer in the model
- `benchmark_model`: Benchmark the model across different batch sizes and sequence lengths
- `monitor_hardware_utilization`: Monitor CPU, GPU, and memory utilization during model execution


### ModelBenchmarkSuite (`utils/profiling.py`)
Comprehensive suite for benchmarking and comparing multiple models with visualization

**Inherits from:** ``

**Dependencies:**
- `torch`
- `numpy`
- `matplotlib`
- `pandas`
- `seaborn`

**Key Methods:**
- `__init__`: Initialize the benchmark suite with output directory
- `benchmark_model`: Run a comprehensive benchmark on a model
- `compare_models`: Compare performance metrics across multiple models
- `generate_optimization_recommendations`: Generate optimization recommendations based on profiling results


### ScaledDotProductAttention (`models/attention.py`)
Core attention mechanism with scaling and masking support

**Inherits from:** `nn.Module`

**Dependencies:**
- `torch`
- `torch.nn`
- `torch.nn.functional`
- `math`

**Key Methods:**
- `forward`: Compute attention scores and context vectors


### MultiHeadAttention (`models/attention.py`)
Multi-head attention mechanism for parallel attention computation

**Inherits from:** `nn.Module`

**Dependencies:**
- `torch`
- `torch.nn`
- `torch.nn.functional`

**Key Methods:**
- `forward`: Compute multi-head attention with optional rotary embeddings
- `split_heads`: Split input tensor into multiple attention heads
- `combine_heads`: Combine multiple attention heads into a single tensor


### GroupedQueryAttention (`models/attention.py`)
Efficient attention mechanism with grouped query heads for reduced computation

**Inherits from:** `nn.Module`

**Dependencies:**
- `torch`
- `torch.nn`

**Key Methods:**
- `forward`: Compute grouped query attention


### ALiBiAttention (`models/attention.py`)
Attention with linear biases for better sequence length extrapolation

**Inherits from:** `nn.Module`

**Dependencies:**
- `torch`
- `torch.nn`
- `math`

**Key Methods:**
- `forward`: Compute attention with ALiBi biases
- `_get_slopes`: Compute attention slopes for ALiBi


### GELU (`models/activations.py`)
Gaussian Error Linear Unit activation function for transformer architectures

**Inherits from:** `nn.Module`

**Dependencies:**
- `torch`
- `torch.nn`
- `torch.nn.functional`

**Key Methods:**
- `__init__`: Initialize the GELU activation layer
- `forward`: Apply the GELU activation function to the input tensor


### FeedForwardNN (`models/feed_forward.py`)
Base class for configurable feed-forward neural networks with optional layer normalization and residual connections

**Inherits from:** `BaseModel`

**Dependencies:**
- `torch`
- `torch.nn`
- `.base_model`
- `.layers`

**Key Methods:**
- `__init__`: Initializes a configurable feed-forward neural network with given architecture
- `forward`: Performs forward pass through the network layers


### FeedForwardClassifier (`models/feed_forward.py`)
Specialized feed-forward classifier with training utilities and prediction methods

**Inherits from:** `FeedForwardNN`

**Dependencies:**
- `torch`
- `torch.nn.functional`

**Key Methods:**
- `predict`: Makes class predictions by selecting highest probability class
- `predict_proba`: Returns class probabilities using softmax on logits
- `training_step`: Performs a single training step with loss calculation and metrics


### MultiLayerPerceptron (`models/feed_forward.py`)
Traditional MLP implementation with modern features like layer normalization and residual connections

**Inherits from:** `nn.Module`

**Dependencies:**
- `torch`
- `torch.nn`

**Key Methods:**
- `forward`: Passes input through all network layers with optional skip connections


### BaseModel (`models/base_model.py`)
Abstract base class providing common model functionality like saving/loading, parameter counting, and device management

**Inherits from:** `nn.Module`

**Dependencies:**
- `torch`
- `torch.nn`
- `os`
- `typing`

**Key Methods:**
- `forward`: Abstract forward pass method that must be implemented by subclasses
- `save`: Save model weights and training state to a file
- `load`: Load model weights from a file


### TokenEmbedding (`models/embeddings.py`)
Neural network layer that converts token indices to dense vector representations with proper scaling

**Inherits from:** `nn.Module`

**Dependencies:**
- `torch`
- `torch.nn`
- `math`

**Key Methods:**
- `__init__`: Initialize the embedding layer with Xavier uniform initialization
- `forward`: Convert token indices to scaled embeddings


### TransformerEncoderLayer (`models/transformer.py`)
Implements a single transformer encoder layer with self-attention and feed-forward networks

**Inherits from:** `nn.Module`

**Dependencies:**
- `torch`
- `torch.nn`
- `.attention`
- `.layers`
- `.positional`

**Key Methods:**
- `forward`: Forward pass through the encoder layer with self-attention and feed-forward


### TransformerEncoder (`models/transformer.py`)
Implements the encoder part of the transformer with multiple layers and positional encoding

**Inherits from:** `nn.Module`

**Dependencies:**
- `torch`
- `torch.nn`
- `.embeddings`
- `.positional`

**Key Methods:**
- `forward`: Forward pass through the encoder with token embeddings and positional encoding


### TransformerDecoderLayer (`models/transformer.py`)
Implements a single transformer decoder layer with self-attention, cross-attention, and feed-forward networks

**Inherits from:** `nn.Module`

**Dependencies:**
- `torch`
- `torch.nn`
- `.attention`
- `.layers`
- `.positional`

**Key Methods:**
- `forward`: Forward pass through the decoder layer with self-attention, cross-attention and feed-forward


### TransformerDecoder (`models/transformer.py`)
Implements the decoder part of the transformer with multiple layers and positional encoding

**Inherits from:** `nn.Module`

**Dependencies:**
- `torch`
- `torch.nn`
- `.embeddings`
- `.positional`

**Key Methods:**
- `forward`: Forward pass through the decoder with token embeddings, positional encoding and encoder memory


### Transformer (`models/transformer.py`)
Implements a complete transformer model with encoder-only architecture

**Inherits from:** `BaseModel`

**Dependencies:**
- `torch`
- `torch.nn`
- `.base_model`

**Key Methods:**
- `forward`: Forward pass through the transformer
- `configure_optimizers`: Configure the optimizer for training


### EncoderDecoderTransformer (`models/transformer.py`)
Implements the full transformer architecture with both encoder and decoder

**Inherits from:** `BaseModel`

**Dependencies:**
- `torch`
- `torch.nn`
- `.base_model`

**Key Methods:**
- `forward`: Forward pass through the encoder-decoder transformer
- `encode`: Encode source sequence
- `decode`: Decode target sequence given encoder memory
- `generate`: Generate output sequences using the trained model
- `generate_square_subsequent_mask`: Generate a square mask for preventing attending to future tokens
- `clone`: Create a deep copy of the transformer model
- `configure_optimizers`: Configure optimizer and learning rate scheduler for training


### TextGenerator (`models/text_generation.py`)
Text generation utilities with various sampling strategies and optimizations

**Dependencies:**
- `torch`
- `torch.nn.functional`
- `numpy`

**Key Methods:**
- `generate`: Generate text from a prompt using various sampling strategies
- `_generate_with_kv_cache`: Generate text with key-value caching for faster inference
- `batch_generate`: Generate text for multiple prompts in parallel


### LinearLayer (`models/layers.py`)
Enhanced linear layer with configurable initialization, dropout, and layer normalization

**Inherits from:** `nn.Module`

**Dependencies:**
- `torch`
- `torch.nn`
- `torch.nn.functional`

**Key Methods:**
- `__init__`: Initialize the enhanced linear layer with optional features
- `_init_weights`: Initialize weights using specified method (kaiming/xavier)
- `forward`: Apply linear transformation with optional normalization and dropout


### FeedForwardBlock (`models/layers.py`)
Flexible feed-forward block with optional residual connections and multiple activation options

**Inherits from:** `nn.Module`

**Dependencies:**
- `torch`
- `torch.nn`
- `torch.nn.functional`

**Key Methods:**
- `__init__`: Initialize the feed-forward block with configurable architecture
- `forward`: Apply feed-forward transformation with optional residual connection


### PositionalEncoding (`models/positional.py`)
Implements both fixed sinusoidal and learnable positional encodings for transformers

**Inherits from:** `nn.Module`

**Dependencies:**
- `torch`
- `torch.nn`
- `matplotlib.pyplot`
- `numpy`

**Key Methods:**
- `__init__`: Initializes positional encoding with configurable parameters
- `forward`: Adds positional information to input embeddings
- `visualize_encodings`: Visualizes the positional encodings as a heatmap for analysis


### RotaryPositionEncoding (`models/positional.py`)
Implements Rotary Position Embedding (RoPE) for enhanced relative position handling

**Inherits from:** `nn.Module`

**Dependencies:**
- `torch`
- `torch.nn`
- `math`

**Key Methods:**
- `__init__`: Initializes rotary embeddings with given dimensions
- `forward`: Applies rotary position encoding to query and key tensors
- `visualize_rotation`: Visualizes the rotation effects on different sequence positions


### ImagePreprocessor (`models/vision/image_preprocessing.py`)
Handles image resizing, normalization, and conversion to tensor format for vision models

**Dependencies:**
- `torchvision.transforms`
- `PIL.Image`
- `numpy`
- `torch.nn.functional`

**Key Methods:**
- `preprocess`: Processes a single image from various input formats to a standardized tensor
- `batch_preprocess`: Processes multiple images into a batch tensor


### PatchExtractor (`models/vision/image_preprocessing.py`)
Efficiently extracts fixed-size patches from image tensors using unfold operations

**Inherits from:** `nn.Module`

**Dependencies:**
- `torch`
- `torch.nn`

**Key Methods:**
- `forward`: Extracts patches from a batch of images and returns patch dimensions


### MultiModalTransformer (`models/vision/multimodal_integration.py`)
Combines vision and text transformer models with projection layers to a common embedding space

**Inherits from:** `BaseModel`

**Dependencies:**
- `torch`
- `torch.nn`
- `torch.nn.functional`
- `..base_model`
- `..transformer`
- `.vision_transformer`

**Key Methods:**
- `encode_image`: Projects image features to multimodal embedding space
- `encode_text`: Projects text features to multimodal embedding space
- `forward`: Processes image and/or text inputs and computes similarity if both are provided


### PatchEmbed (`models/vision/vision_transformer.py`)
Converts images into sequences of patch embeddings using efficient convolution

**Inherits from:** `nn.Module`

**Dependencies:**
- `torch`
- `torch.nn`

**Key Methods:**
- `forward`: Projects image to patch embeddings


### Attention (`models/vision/vision_transformer.py`)
Implements multi-head self-attention mechanism with combined QKV projection

**Inherits from:** `nn.Module`

**Dependencies:**
- `torch`
- `torch.nn`

**Key Methods:**
- `forward`: Performs multi-head attention operation


### Block (`models/vision/vision_transformer.py`)
Transformer block with attention, MLP, and residual connections

**Inherits from:** `nn.Module`

**Dependencies:**
- `torch`
- `torch.nn`

**Key Methods:**
- `forward`: Processes input through attention and MLP layers with residual connections


### VisionTransformer (`models/vision/vision_transformer.py`)
Complete Vision Transformer model for image classification

**Inherits from:** `BaseModel`

**Dependencies:**
- `torch`
- `torch.nn`
- `..base_model`

**Key Methods:**
- `forward`: Forward pass through the model to get class logits and optionally features
- `forward_features`: Forward pass to extract features before classification head
- `extract_features`: Convenience method to extract features for external use
- `configure_optimizers`: Creates optimizer with weight decay excluded from bias and norm parameters


### PatchEmbedding (`models/vision/patch_embedding.py`)
Extracts image patches and projects them to an embedding space with positional information

**Inherits from:** `nn.Module`

**Dependencies:**
- `torch`
- `torch.nn`

**Key Methods:**
- `forward`: Transforms images into sequences of embedded patches with positional information


### VisionTransformerWrapper (`models/pretrained/vision_transformer.py`)
Wrapper for Hugging Face Vision Transformer models with simplified interface

**Inherits from:** `PretrainedModelWrapper`

**Dependencies:**
- `torch`
- `transformers`

**Key Methods:**
- `__init__`: Initialize with specific ViT model
- `load_model`: Load a pretrained Vision Transformer model
- `forward`: Process images through the Vision Transformer


### CLIPModelWrapper (`models/pretrained/clip_model.py`)
Wrapper for OpenAI CLIP models with image-text similarity functionality

**Inherits from:** `PretrainedModelWrapper`

**Dependencies:**
- `torch`
- `torch.nn`
- `open_clip`

**Key Methods:**
- `__init__`: Initialize with specific CLIP model variant and weights
- `load_model`: Load a pretrained CLIP model with transforms and tokenizer
- `encode_image`: Encode images to the multimodal embedding space
- `encode_text`: Encode text to the multimodal embedding space
- `forward`: Process images and/or text through CLIP and compute similarities


### PretrainedModelWrapper (`models/pretrained/base_wrapper.py`)
Base wrapper for pretrained models with standardized interface and utilities

**Inherits from:** `nn.Module`

**Dependencies:**
- `torch`
- `torch.nn`
- `os`
- `typing`

**Key Methods:**
- `__init__`: Initialize the wrapper with model name or model instance
- `load_model`: Abstract method for loading a pretrained model
- `forward`: Abstract forward pass method
- `save`: Save wrapper configuration and model weights
- `load`: Load wrapper configuration and model weights


### ModelRegistry (`models/pretrained/model_registry.py`)
Registry that provides centralized access to all available models

**Dependencies:**
- `torch.nn`
- `.vision_transformer`
- `.clip_model`

**Key Methods:**
- `get_model`: Instantiate a model by type name with optional configuration
- `register_model`: Register a new model type in the registry


### ModelAdapter (`models/pretrained/adapters.py`)
Adapter that adds small trainable components to a frozen pretrained model

**Inherits from:** `nn.Module`

**Dependencies:**
- `torch`
- `torch.nn`

**Key Methods:**
- `__init__`: Initialize the adapter with the base model and adapter dimension
- `_get_output_dim`: Get the output dimension of the base model
- `forward`: Forward pass with residual adapter connection


### SafetyTestHarness (`safety/harness.py`)
Main class for safety testing and evaluation of models against benchmark test cases

**Dependencies:**
- `os`
- `json`
- `typing`
- `datetime`
- `evaluator`
- `utils`

**Key Methods:**
- `create_test_suite`: Creates a basic test suite with examples for each safety category
- `load_test_cases`: Loads test cases from disk, optionally filtered by category
- `evaluate_model`: Evaluates a model against safety test cases and tracks performance metrics
- `generate_report`: Generates detailed safety evaluation reports with performance metrics


### SafetyAugmentedModel (`safety/integration.py`)
Wrapper class that adds safety checks and filtering to base models

**Dependencies:**
- `typing`
- `datetime`
- `filter`

**Key Methods:**
- `predict`: Main method for safe model inference with input validation and output filtering
- `_generate_rejection_message`: Generates appropriate rejection messages based on safety violations
- `_log_safety_event`: Logs safety-related events for monitoring


### SafetyFilter (`safety/filter.py`)
Main class for validating inputs and filtering outputs based on safety concerns

**Dependencies:**
- `re`
- `typing`
- `evaluator`
- `utils`

**Key Methods:**
- `validate_input`: Validates input text for safety concerns with optional override
- `filter_output`: Filters output text to ensure safety and remove unsafe content
- `_redact_unsafe_content`: Redacts unsafe content from text based on safety evaluation


### SafetyEvaluator (`safety/evaluator.py`)
Main class for safety evaluation with configurable thresholds and sensitivity levels

**Dependencies:**
- `typing`
- `re`
- `json`
- `os`
- `.utils`

**Key Methods:**
- `evaluate_text`: Evaluates text for safety concerns across multiple categories and returns detailed results
- `log_evaluation`: Records safety evaluation results for analysis and tracking
- `set_sensitivity`: Adjusts the sensitivity level of safety checks based on application requirements
- `get_safety_summary`: Provides aggregate statistics on past evaluations and current settings


### RedTeamingFramework (`safety/red_teaming/framework.py`)
Framework for organizing, executing, and analyzing adversarial testing strategies

**Inherits from:** ``

**Dependencies:**
- `os`
- `json`
- `datetime`
- `typing`

**Key Methods:**
- `__init__`: Initialize the red teaming framework with output configuration
- `register_attack_strategy`: Register an attack strategy function for generating adversarial inputs
- `generate_adversarial_inputs`: Generate adversarial inputs using registered strategies
- `evaluate_model_robustness`: Evaluate model robustness against adversarial inputs
- `generate_report`: Generate a human-readable report from evaluation results


### AdversarialInputGenerator (`safety/red_teaming/generators.py`)
Collection of methods for creating adversarial prompts to test model boundaries

**Dependencies:**
- `re`
- `random`

**Key Methods:**
- `directive_smuggling`: Generate inputs that attempt to smuggle harmful directives into prompts
- `prompt_injection`: Generate inputs that attempt to inject malicious instructions
- `context_manipulation`: Generate inputs that manipulate the context to elicit problematic outputs
- `goal_hijacking`: Generate inputs that attempt to hijack the model's goal


### AdversarialRobustnessEvaluator (`safety/red_teaming/evaluator.py`)
Evaluates model robustness against various adversarial inputs and attacks

**Dependencies:**
- `src.safety.evaluator.SafetyEvaluator`
- `re`
- `json`

**Key Methods:**
- `evaluate`: Evaluate model robustness based on prompt-response pair
- `_evaluate_jailbreak_success`: Evaluate whether a jailbreak attempt was successful
- `_evaluate_harmful_content`: Evaluate whether the response contains harmful content
- `_evaluate_response_consistency`: Evaluate whether the response is consistent or shows signs of manipulation
- `_evaluate_instruction_following`: Evaluate whether the model improperly followed harmful instructions


### LanguageModelEvaluator (`evaluation/language_model_evaluation.py`)
Evaluation class for language models with metrics and visualization capabilities

**Dependencies:**
- `torch`
- `numpy`
- `matplotlib`
- `seaborn`

**Key Methods:**
- `calculate_perplexity`: Calculates perplexity score for a given text under the model
- `calculate_batch_perplexity`: Calculate perplexity for a batch of texts with optimized processing
- `visualize_attention`: Visualizes attention patterns for a given text at specified layer and head
- `visualize_attention_patterns`: Visualizes attention patterns across all layers and heads
- `analyze_token_probabilities`: Analyzes token probabilities in a text to identify high and low confidence predictions
- `evaluate_on_dataset`: Evaluates model performance on a dataset of texts
- `plot_perplexity_distribution`: Plot the distribution of perplexities as a histogram with statistics


### WMTDataset (`data/wmt_dataset.py`)
Handles loading and preprocessing parallel text data from the WMT dataset

**Dependencies:**
- `os`
- `random`
- `tqdm`

**Key Methods:**
- `__init__`: Initialize the dataset with source/target languages and processing options
- `load_data`: Load and preprocess parallel corpora from WMT dataset


### CurriculumTranslationDataset (`data/curriculum_dataset.py`)
Dataset that implements curriculum learning strategies for translation tasks

**Inherits from:** `Dataset`

**Dependencies:**
- `torch.utils.data.Dataset`
- `numpy`
- `collections.Counter`

**Key Methods:**
- `_calculate_difficulties`: Calculate difficulty scores for all examples based on selected strategy
- `update_stage`: Update the curriculum stage to expose more complex examples
- `__getitem__`: Get an item from the dataset based on curriculum stage
- `get_curriculum_stats`: Get statistics about the current curriculum stage


### ImageDataset (`data/image_dataset.py`)
Dataset for loading and preprocessing images for vision transformer models

**Inherits from:** `Dataset`

**Dependencies:**
- `torch.utils.data.Dataset`
- `PIL.Image`
- `ImagePreprocessor`

**Key Methods:**
- `_get_class_idx`: Get class index from class name using mapping or dynamic creation
- `__getitem__`: Load, preprocess and return an image with its label and path


### WMTDataLoader (`data/wmt_dataloader.py`)
Loads and preprocesses WMT parallel corpus data with batching capability

**Dependencies:**
- `os`
- `random`

**Key Methods:**
- `__init__`: Initialize the WMT data loader with configurable batch size and filtering
- `load_data`: Load and preprocess parallel data
- `__iter__`: Yield batches of source and target sentences


### IWSLTDataset (`data/iwslt_dataset.py`)
Handles loading and preprocessing parallel text data from the IWSLT dataset

**Dependencies:**
- `os`
- `random`
- `requests`
- `tqdm`
- `tarfile`
- `io`

**Key Methods:**
- `__init__`: Initialize the dataset with source/target languages and processing options
- `download_data`: Download and prepare the IWSLT dataset for a specific year with fallback to synthetic data generation
- `load_data`: Load and preprocess parallel corpora, combining data from multiple years if needed


### LanguageModelingDataset (`data/language_modeling.py`)
Dataset for causal language modeling with next-token prediction setup

**Inherits from:** `Dataset`

**Dependencies:**
- `torch`
- `torch.utils.data`
- `tokenization.BPETokenizer`

**Key Methods:**
- `__init__`: Initializes dataset and tokenizes all texts upfront for efficiency
- `__getitem__`: Creates input-target pairs for next-token prediction tasks


### EuroparlDataset (`data/europarl_dataset.py`)
Handles loading and preprocessing parallel text data from the Europarl corpus

**Dependencies:**
- `os`
- `random`

**Key Methods:**
- `__init__`: Initialize the dataset with language pair and optional filtering
- `load_data`: Load and preprocess parallel data with multiple file pattern detection


### CombinedTranslationDataset (`data/combined_translation_dataset.py`)
Combines samples from multiple translation datasets for unified training

**Inherits from:** ``

**Dependencies:**
- `.europarl_dataset`
- `.opensubtitles_dataset`

**Key Methods:**
- `__init__`: Initialize the combined dataset with configurable sources and sample counts


### DataPreprocessor (`data/preprocessing.py`)
Handles data preprocessing operations like standardization and normalization

**Dependencies:**
- `torch`
- `numpy`
- `sklearn.preprocessing`

**Key Methods:**
- `fit`: Fit the preprocessor on the data
- `transform`: Transform the data using the fitted preprocessor
- `fit_transform`: Fit the preprocessor and transform the data
- `inverse_transform`: Inverse transform data back to original scale


### TransformerDataset (`data/sequence_data.py`)
Dataset for handling tokenized source and target sequences for transformer models

**Inherits from:** `Dataset`

**Dependencies:**
- `torch.utils.data.Dataset`

**Key Methods:**
- `__getitem__`: Get source and target sequences with proper BOS/EOS tokens and truncation


### TransformerCollator (`data/sequence_data.py`)
Collator class for batching transformer sequences with padding

**Key Methods:**
- `__call__`: Collate a batch of data with proper padding


### TransformerDataModule (`data/sequence_data.py`)
Complete data module for handling loading, preprocessing, and batching transformer data

**Dependencies:**
- `torch.utils.data.DataLoader`
- `numpy`

**Key Methods:**
- `_setup`: Set up datasets and dataloaders with train/validation splits
- `get_train_dataloader`: Get the training dataloader
- `get_val_dataloader`: Get the validation dataloader
- `_collate_fn`: Collate function to create batches with padding
- `update_curriculum_stage`: Update curriculum stage based on epoch
- `get_curriculum_stats`: Get statistics about the current curriculum stage
- `estimate_steps_per_epoch`: Estimate the number of steps per epoch


### OpenSubtitlesDataset (`data/opensubtitles_dataset.py`)
Handles loading and preprocessing parallel text data from the OpenSubtitles corpus

**Dependencies:**
- `os`
- `random`
- `typing`

**Key Methods:**
- `__init__`: Initialize the dataset with source/target languages and processing options
- `load_data`: Load and preprocess parallel corpora with support for multiple file patterns


### MultimodalDataset (`data/dataloader.py`)
Dataset class for handling multiple modalities of data with consistent length validation

**Inherits from:** `Dataset`

**Dependencies:**
- `torch.utils.data.Dataset`

**Key Methods:**
- `__getitem__`: Get an item from each modality at the specified index
- `__len__`: Return the length of the dataset


### DictionaryDataset (`data/dataset_wrapper.py`)
Wrapper for datasets that return tuples, converting them to dictionaries for a standardized interface

**Inherits from:** `Dataset`

**Dependencies:**
- `torch.utils.data.Dataset`

**Key Methods:**
- `__getitem__`: Get a sample from the dataset as a dictionary with standardized keys


### WikipediaDataset (`data/wikipedia_dataset.py`)
Handles loading and preprocessing multimodal (image-text) data from WikiWeb2M TFRecords

**Dependencies:**
- `tensorflow`
- `torch`
- `numpy`
- `tqdm`

**Key Methods:**
- `__init__`: Initialize the dataset with data split and processing options
- `load_data`: Load and preprocess data from TFRecord files with caching capability
- `to_pytorch_dataset`: Convert to a PyTorch dataset compatible with MultimodalDataset


### LRUCache (`data/tokenization/optimized_bpe_tokenizer.py`)
LRU (Least Recently Used) Cache with expiration for efficient tokenizer caching

**Dependencies:**
- `collections.OrderedDict`
- `threading`

**Key Methods:**
- `get`: Get a value from the cache with expiration checking
- `put`: Add or update a value in the cache with timestamp
- `_cleanup_expired`: Remove expired items from cache


### OptimizedBPETokenizer (`data/tokenization/optimized_bpe_tokenizer.py`)
High-performance BPE tokenizer with caching, vectorization, and memory efficiency

**Inherits from:** `BaseTokenizer`

**Dependencies:**
- `.base_tokenizer`
- `.vocabulary`
- `torch`
- `psutil`

**Key Methods:**
- `tokenize`: Convert text to tokens with caching and validation
- `encode`: Convert text to token indices
- `batch_encode_optimized`: Encode a batch of texts with optimized processing
- `train`: Train the BPE tokenizer on a corpus of texts


### BaseTokenizer (`data/tokenization/base_tokenizer.py`)
Abstract base class that defines the standard interface for all tokenizer implementations

**Inherits from:** `ABC`

**Dependencies:**
- `abc.ABC`

**Key Methods:**
- `tokenize`: Convert text into tokens
- `encode`: Convert text to token indices
- `decode`: Convert token indices back to text
- `batch_encode`: Encode multiple texts efficiently
- `vocab_size`: Get the size of the vocabulary
- `special_tokens`: Get the special tokens used by this tokenizer


### BPETokenizer (`data/tokenization/bpe_tokenizer.py`)
Tokenizer that implements Byte Pair Encoding algorithm for subword tokenization

**Inherits from:** `BaseTokenizer`

**Dependencies:**
- `.base_tokenizer`
- `.vocabulary`
- `.preprocessing`

**Key Methods:**
- `preprocess`: Preprocess text before tokenization
- `train`: Train the BPE tokenizer on a corpus of texts by iteratively merging frequent character pairs
- `_tokenize_word`: Tokenize a single word using BPE merge operations
- `tokenize`: Convert text into subword tokens based on learned merge operations
- `encode`: Convert text to token indices using the vocabulary
- `batch_encode`: Encode a batch of texts into token indices
- `decode`: Convert token indices back to text
- `save_pretrained`: Save tokenizer configuration, vocabulary and merges to disk
- `from_pretrained`: Load a tokenizer from a saved directory
- `vocab_size`: Get the size of the tokenizer vocabulary (property)
- `special_tokens`: Get the special token IDs (property)


### Vocabulary (`data/tokenization/vocabulary.py`)
Manages token-to-index and index-to-token mappings with special token handling

**Dependencies:**
- `json`
- `collections.Counter`
- `logging`

**Key Methods:**
- `add_token`: Add a token to the vocabulary and return its index
- `token_to_index`: Convert a token to its index with validation
- `index_to_token`: Convert an index to its token with validation
- `tokens_to_indices`: Convert a list of tokens to their indices
- `indices_to_tokens`: Convert a list of indices to their tokens
- `save`: Save vocabulary to a JSON file
- `load`: Load vocabulary from a JSON file
- `build_from_texts`: Build a vocabulary from a list of texts using the provided tokenizer



## Statistics
- Total Python modules: 84
- Modules with metadata: 70
- Modules without metadata: 14
- External dependencies: 37

---
Generated automatically from source code metadata