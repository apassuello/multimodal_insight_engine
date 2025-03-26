#!/usr/bin/env python
# demos/hardware_profiling_translation_demo.py

import os
import sys
import time
import torch
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from typing import Dict, List, Tuple, Optional, Union

# Add parent directory to path to import local modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import our model implementations
from src.models.transformer import EncoderDecoderTransformer
from src.utils.profiling import ModelProfiler, ModelBenchmarkSuite
from src.data.europarl_dataset import EuroparlDataset
from src.data.tokenization import BPETokenizer
from src.training.transformer_trainer import TransformerTrainer

def load_or_create_tokenizers():
    """
    Load pretrained BPE tokenizers or create them if they don't exist.
    
    Returns:
        Tuple of (source_tokenizer, target_tokenizer)
    """
    tokenizer_dir = "models/tokenizers"
    src_tokenizer_dir = f"{tokenizer_dir}/de"
    tgt_tokenizer_dir = f"{tokenizer_dir}/en"
    
    # Check if tokenizers already exist
    if (os.path.exists(src_tokenizer_dir) and os.path.exists(tgt_tokenizer_dir)):
        print("Loading existing tokenizers...")
        de_tokenizer = BPETokenizer.from_pretrained(src_tokenizer_dir)
        en_tokenizer = BPETokenizer.from_pretrained(tgt_tokenizer_dir)
        print(f"Loaded German tokenizer with vocab size: {de_tokenizer.vocab_size}")
        print(f"Loaded English tokenizer with vocab size: {en_tokenizer.vocab_size}")
        return de_tokenizer, en_tokenizer
    
    # If tokenizers don't exist, we need to create them
    print("Tokenizers not found. Creating new tokenizers...")
    os.makedirs(src_tokenizer_dir, exist_ok=True)
    os.makedirs(tgt_tokenizer_dir, exist_ok=True)
    
    # Load a sample of the dataset to train tokenizers
    print("Loading Europarl dataset for tokenizer training...")
    train_dataset = EuroparlDataset(
        data_dir="data/europarl",
        src_lang="de",  # German source
        tgt_lang="en",  # English target
        max_examples=5000  # Small subset for tokenizer training
    )
    
    # Train tokenizers
    print("Training German tokenizer...")
    de_tokenizer = BPETokenizer(num_merges=8000-256)
    de_tokenizer.train(
        texts=train_dataset.src_data,
        vocab_size=8000,
        min_frequency=2,
        show_progress=True,
    )
    
    print("Training English tokenizer...")
    en_tokenizer = BPETokenizer(num_merges=8000-256)
    en_tokenizer.train(
        texts=train_dataset.tgt_data,
        vocab_size=8000,
        min_frequency=2,
        show_progress=True,
    )
    
    # Save tokenizers
    print("Saving tokenizers...")
    de_tokenizer.save_pretrained(src_tokenizer_dir)
    en_tokenizer.save_pretrained(tgt_tokenizer_dir)
    
    return de_tokenizer, en_tokenizer

def load_or_create_translation_model(de_tokenizer, en_tokenizer):
    """
    Load a pretrained translation model or create one if it doesn't exist.
    
    Args:
        de_tokenizer: German tokenizer
        en_tokenizer: English tokenizer
        
    Returns:
        EncoderDecoderTransformer model
    """
    model_path = "models/de_en_translation_model.pt"
    
    # Check if model already exists
    if os.path.exists(model_path):
        print(f"Loading existing translation model from {model_path}...")
        device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        checkpoint = torch.load(model_path, map_location=device)
        
        # Create model with the same architecture
        model = EncoderDecoderTransformer(
            src_vocab_size=de_tokenizer.vocab_size,
            tgt_vocab_size=en_tokenizer.vocab_size,
            d_model=512,
            num_heads=8,
            num_encoder_layers=6,
            num_decoder_layers=6,
            d_ff=2048,
            dropout=0.1,
            max_seq_length=100,
            positional_encoding="sinusoidal",
            share_embeddings=False,
        )
        
        # Load weights
        model.load_state_dict(checkpoint["model_state_dict"])
        model.to(device)
        return model
    
    # Create a new model
    print("Creating new translation model...")
    model = EncoderDecoderTransformer(
        src_vocab_size=de_tokenizer.vocab_size,
        tgt_vocab_size=en_tokenizer.vocab_size,
        d_model=512,  # Standard dimension
        num_heads=8,  # Standard number of attention heads
        num_encoder_layers=6,  # Standard number of encoder layers
        num_decoder_layers=6,  # Standard number of decoder layers
        d_ff=2048,    # Standard feed-forward dimension
        dropout=0.1,
        max_seq_length=100,
        positional_encoding="sinusoidal",
        share_embeddings=False,
    )
    
    # Return the untrained model
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model.to(device)
    return model

def preprocess_data_for_training(dataset, de_tokenizer, en_tokenizer, max_length=100):
    """
    Preprocess dataset for training the transformer.
    
    Args:
        dataset: EuroparlDataset with parallel texts
        de_tokenizer: German tokenizer
        en_tokenizer: English tokenizer
        max_length: Maximum sequence length
        
    Returns:
        Dictionary with preprocessed data for model input
    """
    print("Preprocessing data for training...")
    
    # Restrict sample size for profiling
    sample_size = min(100, len(dataset.src_data))
    src_texts = dataset.src_data[:sample_size]
    tgt_texts = dataset.tgt_data[:sample_size]
    
    # Special token indices
    src_pad_idx = de_tokenizer.special_tokens["pad_token_idx"]
    tgt_pad_idx = en_tokenizer.special_tokens["pad_token_idx"]
    src_bos_idx = de_tokenizer.special_tokens["bos_token_idx"]
    tgt_bos_idx = en_tokenizer.special_tokens["bos_token_idx"]
    src_eos_idx = de_tokenizer.special_tokens["eos_token_idx"]
    tgt_eos_idx = en_tokenizer.special_tokens["eos_token_idx"]
    
    # Tokenize and encode data
    src_sequences = []
    tgt_sequences = []
    
    for src_text, tgt_text in zip(src_texts, tgt_texts):
        # Encode source and target texts
        src_ids = de_tokenizer.encode(src_text)
        tgt_ids = en_tokenizer.encode(tgt_text)
        
        # Add special tokens
        src_ids = [src_bos_idx] + src_ids + [src_eos_idx]
        tgt_ids = [tgt_bos_idx] + tgt_ids + [tgt_eos_idx]
        
        # Truncate if needed
        if len(src_ids) > max_length:
            src_ids = src_ids[:max_length-1] + [src_eos_idx]
        if len(tgt_ids) > max_length:
            tgt_ids = tgt_ids[:max_length-1] + [tgt_eos_idx]
        
        src_sequences.append(src_ids)
        tgt_sequences.append(tgt_ids)
    
    # Create batches for profiling (typically this would be done with a DataLoader)
    batch_size = 4
    batches = []
    
    for i in range(0, len(src_sequences), batch_size):
        # Get batch sequences
        batch_src = src_sequences[i:i+batch_size]
        batch_tgt = tgt_sequences[i:i+batch_size]
        
        # Pad sequences to max length in batch
        max_src_len = max(len(seq) for seq in batch_src)
        max_tgt_len = max(len(seq) for seq in batch_tgt)
        
        # Pad source sequences
        padded_src = [seq + [src_pad_idx] * (max_src_len - len(seq)) for seq in batch_src]
        
        # Pad target sequences (input and output)
        padded_tgt_input = [seq[:-1] + [tgt_pad_idx] * (max_tgt_len - len(seq) + 1) for seq in batch_tgt]  # Remove last token (EOS)
        padded_tgt_output = [seq[1:] + [tgt_pad_idx] * (max_tgt_len - len(seq) + 1) for seq in batch_tgt]  # Remove first token (BOS)
        
        # Convert to tensors
        src_tensor = torch.tensor(padded_src, dtype=torch.long)
        tgt_input_tensor = torch.tensor(padded_tgt_input, dtype=torch.long)
        tgt_output_tensor = torch.tensor(padded_tgt_output, dtype=torch.long)
        
        # Create source mask (1 = attend to, 0 = ignore)
        src_mask = (src_tensor != src_pad_idx).unsqueeze(1).unsqueeze(2)
        
        # Create target mask (causal + padding)
        tgt_len = tgt_input_tensor.size(1)
        tgt_padding_mask = (tgt_input_tensor != tgt_pad_idx).unsqueeze(1).unsqueeze(2)
        tgt_causal_mask = torch.tril(torch.ones((tgt_len, tgt_len), dtype=torch.bool)).unsqueeze(0).unsqueeze(0)
        tgt_mask = tgt_padding_mask & tgt_causal_mask
        
        batch = {
            "src": src_tensor,
            "tgt_input": tgt_input_tensor,
            "tgt_output": tgt_output_tensor,
            "src_mask": src_mask,
            "tgt_mask": tgt_mask,
        }
        
        batches.append(batch)
    
    return batches

def create_translation_input_generator(dataset, de_tokenizer, en_tokenizer):
    """
    Create an input generator function for translation model.
    
    Args:
        dataset: EuroparlDataset
        de_tokenizer: German tokenizer
        en_tokenizer: English tokenizer
        
    Returns:
        Function that generates input data for different batch sizes and sequence lengths
    """
    # Preprocess the entire dataset
    sample_size = min(1000, len(dataset.src_data))
    src_texts = dataset.src_data[:sample_size]
    tgt_texts = dataset.tgt_data[:sample_size]
    
    # Special token indices
    src_pad_idx = de_tokenizer.special_tokens["pad_token_idx"]
    tgt_pad_idx = en_tokenizer.special_tokens["pad_token_idx"]
    src_bos_idx = de_tokenizer.special_tokens["bos_token_idx"]
    tgt_bos_idx = en_tokenizer.special_tokens["bos_token_idx"]
    src_eos_idx = de_tokenizer.special_tokens["eos_token_idx"]
    tgt_eos_idx = en_tokenizer.special_tokens["eos_token_idx"]
    
    # Encode all texts once
    all_src_sequences = []
    all_tgt_sequences = []
    
    for src_text, tgt_text in zip(src_texts, tgt_texts):
        src_ids = de_tokenizer.encode(src_text)
        tgt_ids = en_tokenizer.encode(tgt_text)
        
        # Add special tokens
        src_ids = [src_bos_idx] + src_ids + [src_eos_idx]
        tgt_ids = [tgt_bos_idx] + tgt_ids + [tgt_eos_idx]
        
        all_src_sequences.append(src_ids)
        all_tgt_sequences.append(tgt_ids)
    
    def input_generator(batch_size, seq_length):
        """Generate input data for translation model with given batch size and sequence length."""
        device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        
        # Select random subset of sequences for batch
        batch_indices = np.random.choice(len(all_src_sequences), size=batch_size, replace=False)
        
        # Get batch sequences
        batch_src = [all_src_sequences[i][:seq_length] for i in batch_indices]
        batch_tgt = [all_tgt_sequences[i][:seq_length] for i in batch_indices]
        
        # Ensure EOS token is present if sequences were truncated
        batch_src = [seq[:-1] + [src_eos_idx] if len(seq) == seq_length else seq for seq in batch_src]
        batch_tgt = [seq[:-1] + [tgt_eos_idx] if len(seq) == seq_length else seq for seq in batch_tgt]
        
        # Pad sequences
        padded_src = [seq + [src_pad_idx] * (seq_length - len(seq)) for seq in batch_src]
        padded_tgt = [seq[:-1] + [tgt_pad_idx] * (seq_length - len(seq) + 1) for seq in batch_tgt]  # Remove EOS for input
        
        # Convert to tensors
        src_tensor = torch.tensor(padded_src, dtype=torch.long, device=device)
        tgt_tensor = torch.tensor(padded_tgt, dtype=torch.long, device=device)
        
        # Create source mask
        src_mask = (src_tensor != src_pad_idx).unsqueeze(1).unsqueeze(2)
        
        # Create target mask (causal + padding)
        tgt_len = tgt_tensor.size(1)
        tgt_padding_mask = (tgt_tensor != tgt_pad_idx).unsqueeze(1).unsqueeze(2)
        tgt_causal_mask = torch.tril(torch.ones((tgt_len, tgt_len), dtype=torch.bool, device=device)).unsqueeze(0).unsqueeze(0)
        tgt_mask = tgt_padding_mask & tgt_causal_mask
        
        return {
            "src": src_tensor,
            "tgt": tgt_tensor,
            "src_mask": src_mask,
            "tgt_mask": tgt_mask
        }
    
    return input_generator

def profile_translation_model(model, batches):
    """
    Profile the translation model with real data.
    
    Args:
        model: The translation model
        batches: List of preprocessed batches
        
    Returns:
        ModelProfiler instance
    """
    print("\n" + "="*80)
    print("Profiling Translation Model with Real Data")
    print("="*80)
    
    # Create a wrapper for the model to handle dictionary input
    class ModelWrapper(torch.nn.Module):
        def __init__(self, model):
            super().__init__()
            self.model = model
            
        def forward(self, x):
            # Unpack the dictionary to pass as separate arguments
            return self.model(x["src"], x["tgt"], 
                            src_mask=x.get("src_mask"), 
                            tgt_mask=x.get("tgt_mask"))
    
    # Wrap the model
    wrapped_model = ModelWrapper(model)
    
    # Create profiler with the wrapped model
    device = next(model.parameters()).device
    profiler = ModelProfiler(wrapped_model, device)
    
    # Measure execution time on a batch
    print("\nMeasuring translation inference time...")
    batch = batches[0]  # Use the first batch
    time_metrics = profiler.measure_execution_time(
        input_data={"src": batch["src"], "tgt": batch["tgt_input"], 
                    "src_mask": batch["src_mask"], "tgt_mask": batch["tgt_mask"]},
        iterations=5,
        warmup=2
    )
    print(f"Average inference time: {time_metrics['avg_time']:.6f}s")
    print(f"Translations per second: {time_metrics['iterations_per_second']:.2f}")
    
    # Measure memory usage
    print("\nMeasuring translation memory usage...")
    memory_metrics = profiler.measure_memory_usage(
        input_data={"src": batch["src"], "tgt": batch["tgt_input"], 
                    "src_mask": batch["src_mask"], "tgt_mask": batch["tgt_mask"]}
    )
    print(f"Memory usage: {memory_metrics['memory_used_mb']:.2f} MB")
    
    # Profile with PyTorch profiler if available
    try:
        print("\nRunning PyTorch profiler on translation model...")
        profiler.profile_with_pytorch_profiler(
            input_data={"src": batch["src"], "tgt": batch["tgt_input"], 
                        "src_mask": batch["src_mask"], "tgt_mask": batch["tgt_mask"]},
            use_mps=torch.backends.mps.is_available(),
            num_steps=3,
            save_path="profiling_results/translation_trace.json"
        )
    except Exception as e:
        print(f"PyTorch profiler not available or error: {e}")
    
    # Trace memory by layer
    try:
        print("\nTracing memory usage by layer in translation model...")
        profiler.trace_memory_by_layer(
            input_data={"src": batch["src"], "tgt": batch["tgt_input"], 
                        "src_mask": batch["src_mask"], "tgt_mask": batch["tgt_mask"]},
            save_path="profiling_results/translation_memory_trace.csv"
        )
    except Exception as e:
        print(f"Memory tracing error: {e}")
    
    # Generate report
    report = profiler.generate_report(save_path="profiling_results/translation_report.md")
    print(f"Translation model profiling report saved to profiling_results/translation_report.md")
    
    return profiler

def benchmark_translation_model(model, input_generator):
    """
    Benchmark the translation model across different batch sizes and sequence lengths.
    
    Args:
        model: The translation model
        input_generator: Function that generates inputs
        
    Returns:
        ModelBenchmarkSuite instance
    """
    print("\n" + "="*80)
    print("Benchmarking Translation Model")
    print("="*80)
    
    # Create a wrapper for the model to handle dictionary input
    class ModelWrapper(torch.nn.Module):
        def __init__(self, model):
            super().__init__()
            self.model = model
            
        def forward(self, x):
            # Unpack the dictionary to pass as separate arguments
            return self.model(x["src"], x["tgt"], 
                            src_mask=x.get("src_mask"), 
                            tgt_mask=x.get("tgt_mask"))
    
    # Wrap the model
    wrapped_model = ModelWrapper(model)
    
    # Create benchmark suite
    suite = ModelBenchmarkSuite(save_dir="benchmark_results")
    
    # Define benchmark parameters
    batch_sizes = [1, 2, 4, 8, 64, 128]
    sequence_lengths = [16, 32, 64, 96, 180]
    
    print("\nRunning comprehensive benchmark...")
    try:
        suite.benchmark_model(
            model=wrapped_model,
            model_name="translation_model",
            input_generator=input_generator,
            batch_sizes=batch_sizes,
            sequence_lengths=sequence_lengths,
            num_iterations=3,  # Reduced for demo
            profile_with_pytorch=True,
            trace_memory=True
        )
    except Exception as e:
        print(f"Error during benchmark: {e}")
    
    print("\nGenerating optimization recommendations...")
    try:
        recommendations = suite.generate_optimization_recommendations("translation_model")
        print("Recommendations sample:")
        print("\n".join(recommendations.split("\n")[:10]) + "\n...")
    except Exception as e:
        print(f"Error generating recommendations: {e}")
    
    return suite

def monitor_translation_training(model, batches):
    """
    Monitor hardware utilization during translation model training.
    
    Args:
        model: The translation model
        batches: List of preprocessed batches
    """
    print("\n" + "="*80)
    print("Monitoring Hardware During Translation Model Training")
    print("="*80)
    
    # Get device
    device = next(model.parameters()).device
    
    # Create a wrapper for the model to handle dictionary input
    class ModelWrapper(torch.nn.Module):
        def __init__(self, model):
            super().__init__()
            self.model = model
            
        def forward(self, x):
            # Unpack the dictionary to pass as separate arguments
            return self.model(x["src"], x["tgt"], 
                            src_mask=x.get("src_mask"), 
                            tgt_mask=x.get("tgt_mask"))
    
    # Wrap the model
    wrapped_model = ModelWrapper(model)
    
    # Create profiler
    profiler = ModelProfiler(wrapped_model, device)
    
    # Create training function
    def training_function():
        print("Starting translation model training...")
        
        # Set model to training mode
        model.train()
        
        # Create optimizer
        optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
        
        # Define loss function
        loss_fn = torch.nn.CrossEntropyLoss(ignore_index=0)  # Ignore padding
        
        # Mini training loop (5 batches, 2 epochs)
        for epoch in range(2):
            print(f"Epoch {epoch+1}/2")
            for i, batch in enumerate(batches[:5]):
                # Move data to device (should already be there, but just to be safe)
                src = batch["src"].to(device)
                tgt_input = batch["tgt_input"].to(device) 
                tgt_output = batch["tgt_output"].to(device)
                src_mask = batch["src_mask"].to(device)
                tgt_mask = batch["tgt_mask"].to(device)
                
                # Forward pass
                outputs = model(src, tgt_input, src_mask=src_mask, tgt_mask=tgt_mask)
                
                # Reshape outputs and targets for loss calculation
                outputs = outputs.view(-1, outputs.size(-1))
                targets = tgt_output.view(-1)
                
                # Calculate loss
                loss = loss_fn(outputs, targets)
                
                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                print(f"  Batch {i+1}/5, Loss: {loss.item():.4f}")
                time.sleep(0.5)  # Add delay to make monitoring more obvious
    
    # Monitor hardware utilization during training
    try:
        print("\nMonitoring hardware during translation model training...")
        hardware_df = profiler.monitor_hardware_utilization(
            train_fn=training_function,
            duration=60,  # Max seconds to monitor
            interval=0.2,  # Sampling interval
            save_path="profiling_results/translation_training_hardware"
        )
        print("Hardware monitoring completed.")
    except Exception as e:
        print(f"Error monitoring hardware: {e}")

def profile_model_components(model):
    """
    Profile different components of the model separately.
    
    Args:
        model: The translation model
    """
    print("\n" + "="*80)
    print("Profiling Individual Model Components")
    print("="*80)
    
    # Create a sample input
    device = next(model.parameters()).device
    batch_size = 64
    seq_length = 100
    
    # Create dummy inputs
    src = torch.randint(0, 1000, (batch_size, seq_length), device=device)
    tgt = torch.randint(0, 1000, (batch_size, seq_length), device=device)
    
    # Create masks
    src_mask = torch.ones((batch_size, 1, 1, seq_length), dtype=torch.bool, device=device)
    tgt_mask = torch.tril(torch.ones((batch_size, 1, seq_length, seq_length), dtype=torch.bool, device=device))
    
    # Profile encoder
    print("\nProfiling encoder component...")
    encoder_profiler = ModelProfiler(model.encoder, device)
    
    encoder_time = encoder_profiler.measure_execution_time(src, iterations=10, warmup=2)
    encoder_memory = encoder_profiler.measure_memory_usage(src)
    
    print(f"Encoder average time: {encoder_time['avg_time']:.6f}s")
    print(f"Encoder memory usage: {encoder_memory['memory_used_mb']:.2f} MB")
    
    # Profile decoder
    print("\nProfiling decoder component...")
    
    # Create a wrapper for decoder since it requires encoder output
    class DecoderWrapper(torch.nn.Module):
        def __init__(self, encoder, decoder):
            super().__init__()
            self.encoder = encoder
            self.decoder = decoder
            
        def forward(self, x):
            tgt, memory = x
            return self.decoder(tgt, memory, tgt_mask=tgt_mask)
    
    # First get encoder output
    with torch.no_grad():
        memory = model.encoder(src, mask=src_mask)
    
    # Create decoder wrapper
    decoder_wrapper = DecoderWrapper(model.encoder, model.decoder)
    decoder_profiler = ModelProfiler(decoder_wrapper, device)
    
    decoder_time = decoder_profiler.measure_execution_time((tgt, memory), iterations=10, warmup=2)
    decoder_memory = decoder_profiler.measure_memory_usage((tgt, memory))
    
    print(f"Decoder average time: {decoder_time['avg_time']:.6f}s")
    print(f"Decoder memory usage: {decoder_memory['memory_used_mb']:.2f} MB")
    
    # Compare encoder vs decoder
    print("\nEncoder vs Decoder Performance Comparison:")
    print(f"Time ratio (Decoder/Encoder): {decoder_time['avg_time']/encoder_time['avg_time']:.2f}x")
    print(f"Memory ratio (Decoder/Encoder): {decoder_memory['memory_used_mb']/encoder_memory['memory_used_mb']:.2f}x")
    
    # Create component comparison visualization
    labels = ['Encoder', 'Decoder']
    times = [encoder_time['avg_time'], decoder_time['avg_time']]
    memories = [encoder_memory['memory_used_mb'], decoder_memory['memory_used_mb']]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Time comparison
    ax1.bar(labels, times)
    ax1.set_ylabel('Time (s)')
    ax1.set_title('Component Execution Time')
    
    # Memory comparison
    ax2.bar(labels, memories)
    ax2.set_ylabel('Memory (MB)')
    ax2.set_title('Component Memory Usage')
    
    fig.tight_layout()
    fig.savefig("profiling_results/component_comparison.png")
    print("Component comparison plot saved to profiling_results/component_comparison.png")

def main():
    """Run all the demo components with a real translation model."""
    # Create output directories
    os.makedirs("profiling_results", exist_ok=True)
    os.makedirs("benchmark_results", exist_ok=True)
    os.makedirs("models/tokenizers", exist_ok=True)
    
    # Print system info
    print("\n" + "="*80)
    print("TRANSLATION MODEL PROFILING DEMO")
    print("="*80)
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"MPS available: {torch.backends.mps.is_available()}")
    print(f"Using device: {'mps' if torch.backends.mps.is_available() else 'cpu'}")
    
    # Load or create tokenizers
    de_tokenizer, en_tokenizer = load_or_create_tokenizers()
    
    # Load dataset
    print("\nLoading Europarl dataset...")
    dataset = EuroparlDataset(
        data_dir="data/europarl",
        src_lang="de",  # German source
        tgt_lang="en",  # English target
        max_examples=100000  # Limit for profiling demo
    )
    
    # Load or create model
    model = load_or_create_translation_model(de_tokenizer, en_tokenizer)
    
    # Preprocess data for training
    batches = preprocess_data_for_training(dataset, de_tokenizer, en_tokenizer)
    
    # Create input generator for benchmarking
    input_generator = create_translation_input_generator(dataset, de_tokenizer, en_tokenizer)
    
    # Profile translation model
    profiler = profile_translation_model(model, batches)
    
    # Benchmark translation model
    suite = benchmark_translation_model(model, input_generator)
    
    # Monitor hardware during training
    monitor_translation_training(model, batches)
    
    # Profile model components
    profile_model_components(model)
    
    print("\n" + "="*80)
    print("Translation model profiling demo completed!")
    print("="*80)
    print("Check the 'profiling_results' and 'benchmark_results' directories for outputs.")
    
    return profiler, suite

if __name__ == "__main__":
    profiler, suite = main()
