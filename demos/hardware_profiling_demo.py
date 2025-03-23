#!/usr/bin/env python
# demos/hardware_profiling_demo.py

import os
import sys
import time
import torch
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# Add parent directory to path to import local modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import our model implementations
from src.models.transformer import EncoderDecoderTransformer, TransformerEncoder
from src.utils.profiling import ModelProfiler, ModelBenchmarkSuite
from src.models.feed_forward import FeedForwardNN

def create_small_transformer():
    """Create a small transformer encoder model for benchmarking."""
    print("Creating a small transformer encoder model...")
    model = TransformerEncoder(
        vocab_size=10000,  # Small vocabulary for testing
        d_model=256,       # Smaller dimension for quicker testing
        num_heads=4,       # Reduced number of attention heads
        num_layers=2,      # Reduced number of layers
        d_ff=512,          # Smaller feed-forward dimension
        dropout=0.1,
        max_seq_length=512,
        positional_encoding="sinusoidal"
    )
    return model

def create_medium_transformer():
    """Create a medium-sized transformer encoder model for benchmarking."""
    print("Creating a medium-sized transformer encoder model...")
    model = TransformerEncoder(
        vocab_size=30000,  # Medium vocabulary
        d_model=512,       # Standard dimension
        num_heads=8,       # Standard number of attention heads
        num_layers=6,      # Standard number of layers
        d_ff=2048,         # Standard feed-forward dimension
        dropout=0.1,
        max_seq_length=512,
        positional_encoding="sinusoidal"
    )
    return model

def create_feedforward_baseline():
    """Create a feed-forward neural network as a baseline."""
    print("Creating a feed-forward neural network baseline...")
    model = FeedForwardNN(
        input_size=512,
        hidden_sizes=[1024, 1024, 512],
        output_size=512,
        activation="relu",
        dropout=0.1
    )
    return model

def create_transformer_input_generator(vocab_size):
    """Create an input generator function for transformer models."""
    def generate_input(batch_size, seq_length):
        # Generate random token indices
        token_indices = torch.randint(0, vocab_size, (batch_size, seq_length), dtype=torch.long)
        return token_indices
    return generate_input

def create_feedforward_input_generator():
    """Create an input generator function for feed-forward models."""
    def generate_input(batch_size, seq_length):
        # For feed-forward, we'll create a tensor of shape [batch_size, input_size]
        return torch.randn(batch_size, 512)
    return generate_input

def test_model_profiler():
    """Test the ModelProfiler class on a small transformer."""
    print("\n" + "="*80)
    print("Testing ModelProfiler on a small transformer")
    print("="*80)
    
    # Create a model
    model = create_medium_transformer()
    
    # Determine device
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model.to(device)
    
    # Create profiler
    profiler = ModelProfiler(model, device)
    
    # Generate test input
    input_generator = create_transformer_input_generator(10000)
    test_input = input_generator(64, 100)  # batch_size=2, seq_length=64
    
    # Measure execution time
    print("\nMeasuring execution time...")
    time_metrics = profiler.measure_execution_time(test_input, iterations=5, warmup=2)
    print(f"Average execution time: {time_metrics['avg_time']:.6f}s")
    print(f"Iterations per second: {time_metrics['iterations_per_second']:.2f}")
    
    # Measure memory usage
    print("\nMeasuring memory usage...")
    memory_metrics = profiler.measure_memory_usage(test_input)
    print(f"Memory usage: {memory_metrics['memory_used_mb']:.2f} MB")
    
    # Generate report
    print("\nGenerating profiling report...")
    report = profiler.generate_report(save_path="profiling_results/small_transformer_report.md")
    print(f"Report saved to profiling_results/small_transformer_report.md")
    
    # Test PyTorch profiler if available
    try:
        print("\nRunning PyTorch profiler (if available)...")
        profiler.profile_with_pytorch_profiler(
            test_input,
            use_mps=torch.backends.mps.is_available(),
            num_steps=3,
            save_path="profiling_results/small_transformer_trace.json"
        )
    except Exception as e:
        print(f"PyTorch profiler not available or error: {e}")
    
    # Trace memory by layer
    try:
        print("\nTracing memory usage by layer...")
        profiler.trace_memory_by_layer(
            test_input,
            save_path="profiling_results/small_transformer_memory_trace.csv"
        )
    except Exception as e:
        print(f"Memory tracing error: {e}")
    
    # Plot metrics
    print("\nPlotting metrics...")
    figures = profiler.plot_metrics(save_dir="profiling_results")
    
    return profiler

def test_benchmark_suite():
    """Test the ModelBenchmarkSuite class with multiple models."""
    print("\n" + "="*80)
    print("Testing ModelBenchmarkSuite with multiple models")
    print("="*80)
    
    # Create benchmark suite
    suite = ModelBenchmarkSuite(save_dir="benchmark_results")
    
    # Define common benchmark parameters
    batch_sizes = [1, 8, 64, 128]
    sequence_lengths = [32, 64, 128]
    
    # Benchmark small transformer
    print("\nBenchmarking small transformer...")
    small_transformer = create_medium_transformer()
    small_transformer.to(torch.device("mps" if torch.backends.mps.is_available() else "cpu"))
    small_input_gen = create_transformer_input_generator(10000)
    
    try:
        suite.benchmark_model(
            model=small_transformer,
            model_name="small_transformer",
            input_generator=small_input_gen,
            batch_sizes=batch_sizes,
            sequence_lengths=sequence_lengths,
            num_iterations=3,  # Reduced for demo
            profile_with_pytorch=True,
            trace_memory=True
        )
    except Exception as e:
        print(f"Error benchmarking small transformer: {e}")
    
    # Benchmark feed-forward baseline
    print("\nBenchmarking feed-forward baseline...")
    feedforward = create_feedforward_baseline()
    feedforward.to(torch.device("mps" if torch.backends.mps.is_available() else "cpu"))
    ff_input_gen = create_feedforward_input_generator()
    
    try:
        suite.benchmark_model(
            model=feedforward,
            model_name="feedforward_baseline",
            input_generator=ff_input_gen,
            batch_sizes=batch_sizes,
            sequence_lengths=[1],  # Only one "sequence length" for feed-forward
            num_iterations=3,  # Reduced for demo
            profile_with_pytorch=True,
            trace_memory=True
        )
    except Exception as e:
        print(f"Error benchmarking feed-forward: {e}")
    
    # Perform model comparison
    try:
        print("\nComparing models...")
        comparison_df = suite.compare_models(
            model_names=["small_transformer", "feedforward_baseline"],
            metric="avg_time",
            save_path="benchmark_results/model_comparison.png"
        )
        print("Model comparison completed.")
    except Exception as e:
        print(f"Error comparing models: {e}")
    
    # Generate optimization recommendations
    try:
        print("\nGenerating optimization recommendations...")
        recommendations = suite.generate_optimization_recommendations("small_transformer")
        print("Recommendations sample:")
        print("\n".join(recommendations.split("\n")[:10]) + "\n...")
    except Exception as e:
        print(f"Error generating recommendations: {e}")
    
    return suite

def test_hardware_monitoring():
    """Test hardware monitoring during a simulated training loop."""
    print("\n" + "="*80)
    print("Testing hardware monitoring during simulated training")
    print("="*80)
    
    # Create a model
    model = create_medium_transformer()
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model.to(device)
    
    # Create profiler
    profiler = ModelProfiler(model, device)
    
    # Create a simulated training function
    def simulated_training():
        print("Starting simulated training...")
        optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
        input_generator = create_transformer_input_generator(1000000)
        
        # Simulated training loop
        for i in range(10):
            # Generate random batch
            batch_size = 64
            seq_length = 100
            inputs = input_generator(batch_size, seq_length)
            inputs = inputs.to(device)
            
            # Forward pass
            outputs = model(inputs)
            
            # Simulated loss
            loss = outputs.sum()
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            print(f"Iteration {i+1}/10 completed")
            time.sleep(0.5)  # Add delay to make monitoring more obvious
    
    # Monitor hardware during training
    try:
        print("\nMonitoring hardware utilization during training...")
        hardware_df = profiler.monitor_hardware_utilization(
            train_fn=simulated_training,
            duration=20,  # Max seconds to monitor
            interval=0.2,  # Sampling interval
            save_path="profiling_results/hardware_utilization"
        )
        print("Hardware monitoring completed.")
    except Exception as e:
        print(f"Error monitoring hardware: {e}")

def main():
    """Run all the demo components."""
    # Create output directories
    os.makedirs("profiling_results", exist_ok=True)
    os.makedirs("benchmark_results", exist_ok=True)
    
    # Print system info
    print("\n" + "="*80)
    print("HARDWARE PROFILING DEMO")
    print("="*80)
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"MPS available: {torch.backends.mps.is_available()}")
    print(f"Using device: {'mps' if torch.backends.mps.is_available() else 'cpu'}")
    
    # Test components
    profiler = test_model_profiler()
    suite = test_benchmark_suite()
    test_hardware_monitoring()
    
    print("\n" + "="*80)
    print("Hardware profiling demo completed!")
    print("="*80)
    print("Check the 'profiling_results' and 'benchmark_results' directories for outputs.")
    
    return profiler, suite

if __name__ == "__main__":
    profiler, suite = main()