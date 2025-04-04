#!/usr/bin/env python
# Language Model Training & Optimization Demo

import os
import sys
import torch
import numpy as np
import random
import matplotlib.pyplot as plt
from tqdm import tqdm
import argparse
import json
from typing import List, Dict, Optional, Union, Any

# Add project root to path (adjust if needed)
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import modules from our implementation
from src.models.transformer import EncoderDecoderTransformer
from src.training.language_model_trainer import LanguageModelTrainer
from src.data.language_modeling import LanguageModelingDataset, create_lm_dataloaders
from src.data.tokenization import OptimizedBPETokenizer
from src.models.text_generation import TextGenerator
from src.evaluation.language_model_evaluation import LanguageModelEvaluator
from src.optimization.mixed_precision import MixedPrecisionConverter
from src.optimization.quantization import DynamicQuantizer, StaticQuantizer, QuantizationConfig
from src.optimization.benchmarking import OptimizationBenchmark

def set_seed(seed):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def load_dataset(file_path, limit=None):
    """Load a text dataset from file."""
    print(f"Loading dataset from: {file_path}")
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        # Clean up lines
        texts = [line.strip() for line in lines if line.strip()]
        
        if limit:
            print(f"Limiting dataset to {limit} samples")
            texts = texts[:limit]
            
        print(f"Loaded {len(texts)} text samples")
        return texts
    except FileNotFoundError:
        print(f"Dataset file not found: {file_path}")
        # Generate a small synthetic dataset for demonstration
        print("Generating synthetic dataset...")
        return generate_synthetic_dataset(num_samples=limit or 1000)

def generate_synthetic_dataset(num_samples=1000, min_len=5, max_len=50):
    """Generate a synthetic text dataset for demonstration."""
    # Basic vocabulary for generating simple sentences
    subjects = ["The model", "The system", "The algorithm", "The transformer", "The network", 
               "Our approach", "The method", "This architecture", "The solution"]
    verbs = ["processes", "analyzes", "generates", "transforms", "improves", 
            "handles", "calculates", "optimizes", "learns from"]
    objects = ["the input data", "complex sequences", "text patterns", "language models", 
              "attention mechanisms", "diverse datasets", "natural language", "training examples"]
    adjectives = ["efficiently", "accurately", "quickly", "robustly", "effectively", 
                "systematically", "intelligently", "precisely", "reliably"]
    
    texts = []
    for _ in range(num_samples):
        # Generate a random sentence using our vocabulary
        length = random.randint(1, 4)  # Number of sentences in each sample
        sentences = []
        
        for _ in range(length):
            subject = random.choice(subjects)
            verb = random.choice(verbs)
            obj = random.choice(objects)
            adj = random.choice(adjectives)
            
            # Randomly order components with different patterns
            pattern = random.choice([
                f"{subject} {verb} {obj} {adj}.",
                f"{adj.capitalize()}, {subject.lower()} {verb} {obj}.",
                f"{subject} {adj} {verb} {obj}.",
                f"When {subject.lower()} {verb} {obj}, it does so {adj}."
            ])
            
            sentences.append(pattern)
        
        texts.append(" ".join(sentences))
    
    return texts

def train_tokenizer(texts, vocab_size=8000, output_dir="models/tokenizer", device=None):
    """Train a BPE tokenizer on the dataset."""
    print(f"Training BPE tokenizer with vocab size {vocab_size}...")
    
    # Create tokenizer with device support
    tokenizer = OptimizedBPETokenizer(device=device)
    
    # Train on texts
    tokenizer.train(
        texts=texts,
        vocab_size=vocab_size,
        min_frequency=2,
        show_progress=True
    )
    
    # Save tokenizer
    os.makedirs(output_dir, exist_ok=True)
    tokenizer.save_pretrained(output_dir)
    
    print(f"Tokenizer trained and saved to {output_dir}")
    return tokenizer

def load_or_train_tokenizer(texts, vocab_size=8000, tokenizer_path="models/tokenizer", device=None):
    """Load a pre-existing tokenizer or train a new one."""
    # Check if tokenizer exists
    if os.path.exists(tokenizer_path) and os.path.isdir(tokenizer_path):
        try:
            print(f"Loading pre-trained tokenizer from {tokenizer_path}")
            tokenizer = OptimizedBPETokenizer.from_pretrained(tokenizer_path)
            # Set device after loading
            if device is not None:
                tokenizer.device = device
            print(f"Loaded tokenizer with vocab size: {tokenizer.vocab_size}")
            return tokenizer
        except Exception as e:
            print(f"Error loading tokenizer: {e}")
            print("Training new tokenizer instead...")
    
    # Train tokenizer
    return train_tokenizer(texts, vocab_size, tokenizer_path, device)

def create_or_load_model(vocab_size, model_path=None, device=None):
    """Create a new transformer model or load a pre-trained one."""
    # Model configuration
    config = {
        "src_vocab_size": vocab_size,
        "tgt_vocab_size": vocab_size,
        "d_model": 512,
        "num_heads": 8,
        "num_encoder_layers": 6,
        "num_decoder_layers": 6,
        "d_ff": 2048,
        "dropout": 0.1,
        "max_seq_length": 512,
        "positional_encoding": "sinusoidal",
        # Share embeddings for encoder and decoder in language modeling
        "share_embeddings": True  
    }
    
    print("Creating transformer model...")
    model = EncoderDecoderTransformer(**config)
    
    # Move to device
    if device:
        model.to(device)
    
    # Load pre-trained weights if provided
    if model_path and os.path.exists(model_path):
        print(f"Loading pre-trained model from {model_path}")
        model.load(model_path)
    
    print(f"Model parameter count: {sum(p.numel() for p in model.parameters())}")
    return model

def create_input_generator(tokenizer, model, sample_text="The transformer architecture has revolutionized natural language processing.", max_len=128):
    """Create an input generator function for benchmarking."""
    # Check if we're working with EncoderDecoderTransformer
    is_encoder_decoder = hasattr(model, 'encoder') and hasattr(model, 'decoder')
    
    def generate_input(batch_size):
        """Generate a batch of inputs for benchmarking."""
        # Tokenize sample text
        token_ids = tokenizer.encode(sample_text)
        
        # Add special tokens
        bos_idx = tokenizer.special_tokens["bos_token_idx"]
        eos_idx = tokenizer.special_tokens["eos_token_idx"]
        token_ids = [bos_idx] + token_ids + [eos_idx]
        
        # Pad or truncate to max_len
        if len(token_ids) > max_len:
            token_ids = token_ids[:max_len]
        else:
            token_ids = token_ids + [tokenizer.special_tokens["pad_token_idx"]] * (max_len - len(token_ids))
        
        # Create tensor with batch dimension
        inputs = torch.tensor(token_ids, dtype=torch.long).unsqueeze(0)
        inputs = inputs.repeat(batch_size, 1)
        
        if is_encoder_decoder:
            # For encoder-decoder models, return both src and tgt
            # Create attention masks that match expected dimensions
            src_len = inputs.size(1)
            
            # Create a square mask (batch_size, src_len, src_len) where each position can attend to all others
            # This works with most attention mechanisms expecting a square mask
            src_mask = torch.ones(batch_size, src_len, src_len, dtype=torch.bool)
            
            # Create a causal/triangular mask for the target to prevent attending to future tokens
            tgt_mask = torch.tril(torch.ones(src_len, src_len, dtype=torch.bool)).unsqueeze(0).repeat(batch_size, 1, 1)
            
            return {
                "src": inputs,
                "tgt": inputs.clone(),
                "src_mask": src_mask,
                "tgt_mask": tgt_mask
            }
        else:
            # For standard models
            return inputs
    
    return generate_input

def main(args):
    # Set random seeds for reproducibility
    set_seed(args.seed)
    
    # Determine device
    device = torch.device("cuda" if torch.cuda.is_available() else 
                          "mps" if torch.backends.mps.is_available() else 
                          "cpu")
    print(f"Using device: {device}")
    
    # Create output directories
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, "models"), exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, "results"), exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, "figures"), exist_ok=True)
    
    # Load dataset
    texts = load_dataset(args.dataset, limit=args.dataset_limit)
    
    # Load or train tokenizer
    tokenizer = load_or_train_tokenizer(
        texts, 
        vocab_size=args.vocab_size, 
        tokenizer_path=os.path.join(args.output_dir, "tokenizer"),
        device=device
    )
    
    # Create dataloaders
    train_dataloader, val_dataloader = create_lm_dataloaders(
        texts=texts,
        tokenizer=tokenizer,
        batch_size=args.batch_size,
        max_length=args.max_seq_length,
        val_split=args.val_split,
        seed=args.seed
    )
    
    print(f"Created dataloaders with batch size {args.batch_size}")
    print(f"Training samples: {len(train_dataloader.dataset)}")
    print(f"Validation samples: {len(val_dataloader.dataset)}")
    
    # Create or load model
    model_path = os.path.join(args.output_dir, "models", "language_model_best.pt")
    model_exists = os.path.exists(model_path)
    
    model = create_or_load_model(
        vocab_size=tokenizer.vocab_size,
        model_path=model_path if model_exists and not args.force_train else None,
        device=device
    )
    
    # Only train if model doesn't exist or forced training
    if not model_exists or args.force_train:
        print("\n=== Training Language Model ===")
        
        # Create trainer
        trainer = LanguageModelTrainer(
            model=model,
            train_dataloader=train_dataloader,
            val_dataloader=val_dataloader,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            warmup_steps=args.warmup_steps,
            max_grad_norm=args.max_grad_norm,
            device=device,
            log_dir=os.path.join(args.output_dir, "logs")
        )
        
        # Train model
        history = trainer.train(
            num_epochs=args.epochs,
            save_dir=os.path.join(args.output_dir, "models"),
            model_name="language_model"
        )
        
        # Plot training curves
        print("Plotting training metrics...")
        trainer.plot_training_curves(save_path=os.path.join(args.output_dir, "figures", "training_curves.png"))
        
        # Save training history
        with open(os.path.join(args.output_dir, "results", "training_history.json"), 'w') as f:
            # Convert tensors and other non-serializable types to Python native types
            serializable_history = {}
            for k, v in history.items():
                if isinstance(v, list) and all(isinstance(item, (int, float)) for item in v):
                    serializable_history[k] = v
                elif isinstance(v, list):
                    serializable_history[k] = [float(item) if hasattr(item, 'item') else item for item in v]
                elif hasattr(v, 'item'):
                    serializable_history[k] = float(v)
                else:
                    serializable_history[k] = v
            
            json.dump(serializable_history, f, indent=2)
    
    # Ensure model is in evaluation mode
    model.eval()
    
    # Create text generator
    print("\n=== Text Generation Examples ===")
    text_generator = TextGenerator(
        model=model,
        tokenizer=tokenizer,
        device=device
    )
    
    # Example prompts for generation
    prompts = [
        "The transformer architecture",
        "In recent years, natural language processing has",
        "The key advantage of attention mechanisms is",
        "Learning to generate realistic text requires"
    ]
    
    # Generate with different settings
    print("\nGreedy Decoding:")
    for prompt in prompts[:2]:  # Use first two prompts
        generated = text_generator.generate(
            prompt=prompt,
            max_new_tokens=30,
            do_sample=False
        )
        print(f"\nPrompt: {prompt}")
        print(f"Generated: {generated}")
    
    print("\nSampling with Temperature:")
    for prompt in prompts[2:]:  # Use last two prompts
        generated = text_generator.generate(
            prompt=prompt,
            max_new_tokens=30,
            temperature=0.8,
            do_sample=True
        )
        print(f"\nPrompt: {prompt}")
        print(f"Generated: {generated}")
    
    print("\nTop-K Sampling:")
    generated = text_generator.generate(
        prompt="Attention mechanisms allow models to",
        max_new_tokens=40,
        temperature=0.9,
        top_k=50,
        do_sample=True
    )
    print(f"Generated: {generated}")
    
    print("\nTop-P (Nucleus) Sampling:")
    generated = text_generator.generate(
        prompt="The future of language models will",
        max_new_tokens=40,
        temperature=0.9,
        top_p=0.92,
        do_sample=True
    )
    print(f"Generated: {generated}")
    
    print("\nBatch Generation:")
    batch_prompts = [
        "Transformers work by",
        "Language models can be used for"
    ]
    batch_generated = text_generator.batch_generate(
        prompts=batch_prompts,
        max_new_tokens=20,
        temperature=0.7,
        do_sample=True
    )
    for prompt, generated in zip(batch_prompts, batch_generated):
        print(f"\nPrompt: {prompt}")
        print(f"Generated: {generated}")
    
    # Model evaluation
    print("\n=== Model Evaluation ===")
    evaluator = LanguageModelEvaluator(
        model=model,
        tokenizer=tokenizer,
        device=device
    )
    
    # Calculate perplexity on sample text
    sample_texts = texts[:5]  # Use first 5 texts for demonstration
    try:
        perplexity_results = evaluator.calculate_batch_perplexity(sample_texts)
        print(f"Overall Perplexity: {perplexity_results['perplexity']:.2f}")
        print(f"Per-sample perplexities: {', '.join(f'{p:.2f}' for p in perplexity_results['per_sequence_perplexity'])}")
    except Exception as e:
        print(f"Error calculating batch perplexity: {e}")
        print("Falling back to individual calculations")
        # Fall back to calculating perplexity for individual samples
        perplexities = []
        for text in sample_texts[:2]:  # Just try a couple samples
            try:
                # Analyze token probabilities instead, which should be more compatible
                token_analysis = evaluator.analyze_token_probabilities(text)
                avg_prob = token_analysis['average_probability']
                perplexity = 1.0 / max(avg_prob, 1e-10)  # Simple perplexity approximation
                perplexities.append(perplexity)
                print(f"Sample perplexity (approx): {perplexity:.2f}")
            except Exception as e:
                print(f"Could not analyze sample text: {e}")
    
    # Analyze token probabilities for a sample
    sample_text = "The model generates text based on patterns learned during training."
    try:
        token_analysis = evaluator.analyze_token_probabilities(sample_text)
        
        print(f"\nToken Probability Analysis for: '{sample_text}'")
        print(f"Average token probability: {token_analysis['average_probability']:.4f}")
        print(f"Lowest probability token: '{token_analysis['min_probability_token']}' ({token_analysis['min_probability']:.4f})")
        print(f"Highest probability token: '{token_analysis['max_probability_token']}' ({token_analysis['max_probability']:.4f})")
    except Exception as e:
        print(f"Error analyzing token probabilities: {e}")
    
    # Skip attention visualization if having issues
    try:
        print("\nGenerating attention visualization...")
        fig = evaluator.visualize_attention(
            sample_text,
            layer=-1,  # Last layer
            head=0     # First attention head
        )
        fig.savefig(os.path.join(args.output_dir, "figures", "attention_visualization.png"))
        plt.close(fig)
        print(f"Attention visualization saved to {os.path.join(args.output_dir, 'figures', 'attention_visualization.png')}")
    except Exception as e:
        print(f"Error generating attention visualization: {e}")
    
    # Model optimization
    if args.run_optimization:
        print("\n=== Model Optimization ===")
        print("Creating input generator for benchmarking...")
        input_generator = create_input_generator(tokenizer, model)
        
        # Create benchmark
        benchmark = OptimizationBenchmark(
            model=model,
            input_generator=input_generator,
            batch_sizes=[1, 4, 16, 32],
            save_dir=os.path.join(args.output_dir, "results")
        )
        
        # Benchmark original model
        print("Benchmarking original model...")
        original_results = benchmark.benchmark_original_model()
        
        # Apply mixed precision
        print("\nApplying mixed precision...")
        # Create a copy of the model by creating a new instance and loading state dict
        if hasattr(model, 'clone'):
            mp_model_base = model.clone()
        else:
            # Handle encoder-decoder model that might not have clone method
            # Get model parameters for reconstruction
            model_params = {}
            if hasattr(model, 'encoder') and hasattr(model, 'decoder'):
                # For EncoderDecoderTransformer
                src_vocab_size = model.encoder.token_embedding.embedding.weight.size(0)
                tgt_vocab_size = src_vocab_size if hasattr(model, 'share_embeddings') and model.share_embeddings else model.decoder.token_embedding.embedding.weight.size(0)
                
                # Get feed forward dimension by checking the linear1 layer inside the feed_forward block
                d_ff = model.encoder.layers[0].feed_forward.linear1.linear.in_features
                
                model_params = {
                    'src_vocab_size': src_vocab_size,
                    'tgt_vocab_size': tgt_vocab_size,
                    'd_model': model.d_model,
                    'num_heads': model.encoder.layers[0].self_attn.num_heads,
                    'num_encoder_layers': len(model.encoder.layers),
                    'num_decoder_layers': len(model.decoder.layers),
                    'd_ff': d_ff,
                    'share_embeddings': model.share_embeddings if hasattr(model, 'share_embeddings') else False
                }
            
            # Create a new instance with same architecture
            mp_model_base = model.__class__(**model_params)
            # Copy weights from current model to the new one
            mp_model_base.load_state_dict(model.state_dict())
            
        mp_converter = MixedPrecisionConverter(
            model=mp_model_base,
            dtype=torch.float16,
            use_auto_cast=True
        )
        mp_model = mp_converter.convert_to_mixed_precision()
        mp_results = benchmark.benchmark_optimized_model(mp_model, "mixed_precision")
        
        # Apply dynamic quantization
        print("\nApplying dynamic quantization...")
        # Create another copy of the model
        if hasattr(model, 'clone'):
            dq_model_base = model.clone()
        else:
            # Handle encoder-decoder model that might not have clone method
            # Get model parameters for reconstruction
            model_params = {}
            if hasattr(model, 'encoder') and hasattr(model, 'decoder'):
                # For EncoderDecoderTransformer
                src_vocab_size = model.encoder.token_embedding.embedding.weight.size(0)
                tgt_vocab_size = src_vocab_size if hasattr(model, 'share_embeddings') and model.share_embeddings else model.decoder.token_embedding.embedding.weight.size(0)
                
                # Get feed forward dimension by checking the linear1 layer inside the feed_forward block
                d_ff = model.encoder.layers[0].feed_forward.linear1.linear.in_features
                
                model_params = {
                    'src_vocab_size': src_vocab_size,
                    'tgt_vocab_size': tgt_vocab_size,
                    'd_model': model.d_model,
                    'num_heads': model.encoder.layers[0].self_attn.num_heads,
                    'num_encoder_layers': len(model.encoder.layers),
                    'num_decoder_layers': len(model.decoder.layers),
                    'd_ff': d_ff,
                    'share_embeddings': model.share_embeddings if hasattr(model, 'share_embeddings') else False
                }
            
            # Create a new instance with same architecture
            dq_model_base = model.__class__(**model_params)
            # Copy weights from current model to the new one
            dq_model_base.load_state_dict(model.state_dict())
            
        dq_config = QuantizationConfig(
            quantization_type="dynamic",
            dtype=torch.qint8,
            quantize_weights=True,
            quantize_activations=False
        )
        dq_quantizer = DynamicQuantizer(
            model=dq_model_base,
            config=dq_config
        )
        dq_model = dq_quantizer.optimize()
        dq_results = benchmark.benchmark_optimized_model(dq_model, "dynamic_quantization")
        
        # Compare optimizations
        print("\nComparing optimization techniques...")
        comparison = benchmark.compare_optimizations(save_plot=True)
        
        # Generate optimization report
        print("Generating optimization report...")
        report = benchmark.generate_report()
        
        # Save benchmark results
        benchmark.save_results()
        
        # Print optimization summary
        print("\n=== Optimization Summary ===")
        for name, metrics in comparison["optimizations"].items():
            print(f"{name}:")
            print(f"  Average Speedup: {metrics['avg_speedup']:.2f}x")
            if metrics["avg_memory_reduction"] is not None:
                print(f"  Average Memory Reduction: {metrics['avg_memory_reduction']:.2f}x")
            print()
    
    print("\nDemo completed successfully!")
    return 0

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Language Model Training & Optimization Demo")
    
    # Dataset parameters
    parser.add_argument("--dataset", type=str, default="data/wiki/wikiweb2m-train.tfrecord.gz-00000-of-00005", 
                        help="Path to dataset file")
    parser.add_argument("--dataset_limit", type=int, default=1000000, 
                        help="Maximum number of samples to use from dataset")
    parser.add_argument("--val_split", type=float, default=0.1, 
                        help="Fraction of data to use for validation")
    
    # Model parameters
    parser.add_argument("--vocab_size", type=int, default=32000, 
                        help="Vocabulary size for tokenizer")
    parser.add_argument("--max_seq_length", type=int, default=128, 
                        help="Maximum sequence length")
    parser.add_argument("--force_train", action="store_true", 
                        help="Force training even if pre-trained model exists")
    
    # Training parameters
    parser.add_argument("--batch_size", type=int, default=64, 
                        help="Training batch size")
    parser.add_argument("--epochs", type=int, default=10, 
                        help="Number of training epochs")
    parser.add_argument("--learning_rate", type=float, default=5e-5, 
                        help="Peak learning rate after warmup")
    parser.add_argument("--weight_decay", type=float, default=0.01, 
                        help="Weight decay for regularization")
    parser.add_argument("--warmup_steps", type=int, default=4000, 
                        help="Learning rate warmup steps")
    parser.add_argument("--max_grad_norm", type=float, default=1.0, 
                        help="Maximum gradient norm for clipping")
    
    # Optimization parameters
    parser.add_argument("--run_optimization", action="store_true", 
                        help="Run optimization benchmarks")
    
    # Misc parameters
    parser.add_argument("--seed", type=int, default=42, 
                        help="Random seed for reproducibility")
    parser.add_argument("--output_dir", type=str, default="output", 
                        help="Output directory for results")
    
    args = parser.parse_args()
    
    # For interactive environments without argparse
    if 'ipykernel' in sys.modules:
        class Args:
            dataset = "data/text_corpus.txt"
            dataset_limit = 5000
            val_split = 0.1
            vocab_size = 8000
            max_seq_length = 128
            force_train = True
            batch_size = 32
            epochs = 2
            learning_rate = 5e-5
            weight_decay = 0.01
            warmup_steps = 500
            max_grad_norm = 1.0
            run_optimization = True
            seed = 42
            output_dir = "output"
        args = Args()
    
    sys.exit(main(args))