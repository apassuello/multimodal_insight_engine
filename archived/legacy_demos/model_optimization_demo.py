#!/usr/bin/env python
# demos/model_optimization_demo.py

import os
import sys
import argparse
import torch
import numpy as np
import time
import copy
from typing import Dict, List, Optional, Union, Any, Callable
import matplotlib.pyplot as plt

# Add parent directory to path to import project modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import our model implementations
from src.models.transformer import TransformerEncoder
from src.optimization.pruning import ModelPruner, PruningConfig

def set_seed(seed: int = 42):
    """Set random seeds for reproducibility."""
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def get_device():
    """Get the appropriate device for testing."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")

def create_test_transformer(size="medium"):
    """Create a test transformer model for optimization experiments."""
    print(f"Creating a {size} transformer model for testing...")
    
    if size == "small":
        model = TransformerEncoder(
            vocab_size=10000,  # Small vocabulary for testing
            d_model=128,       # Smaller dimension for quicker testing
            num_heads=4,       # Reduced number of attention heads
            num_layers=2,      # Reduced number of layers
            d_ff=512,          # Smaller feed-forward dimension
            dropout=0.1,
            max_seq_length=128,
            positional_encoding="sinusoidal"
        )
    elif size == "medium":
        model = TransformerEncoder(
            vocab_size=30000,  # Medium vocabulary
            d_model=512,       # Standard dimension
            num_heads=8,       # Standard number of attention heads
            num_layers=4,      # Standard number of layers
            d_ff=1024,         # Standard feed-forward dimension
            dropout=0.1,
            max_seq_length=256,
            positional_encoding="sinusoidal"
        )
    else:
        raise ValueError(f"Unknown model size: {size}")
    
    device = get_device()
    model.to(device)
    
    # Print model size info
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model has {total_params:,} parameters")
    
    return model

def direct_magnitude_pruning(model, amount=0.3):
    """
    Apply magnitude pruning directly by zeroing out weights.
    
    Args:
        model: The model to prune
        amount: The proportion of weights to prune
        
    Returns:
        Pruned model and sparsity information
    """
    print(f"Applying direct magnitude pruning ({amount*100:.0f}%)...")
    
    total_params = 0
    zero_params = 0
    
    for name, param in model.named_parameters():
        if 'weight' in name:  # Only prune weights, not biases
            # Get tensor size
            tensor_size = param.data.numel()
            total_params += tensor_size
            
            # Calculate number of weights to prune
            n_prune = int(tensor_size * amount)
            
            # Find the magnitude threshold
            threshold = param.data.abs().flatten().sort()[0][n_prune]
            
            # Create a mask for small weights
            mask = param.data.abs() <= threshold
            
            # Count zeros before pruning (for verification)
            initial_zeros = (param.data == 0).sum().item()
            
            # Apply the mask by setting weights below threshold to zero
            param.data[mask] = 0
            
            # Count zeros after pruning
            pruned_zeros = (param.data == 0).sum().item()
            
            # Add to total zero count
            zero_params += pruned_zeros
            
            print(f"  {name}: Pruned {pruned_zeros - initial_zeros} of {tensor_size} weights")
    
    # Calculate final sparsity
    sparsity = zero_params / total_params if total_params > 0 else 0
    print(f"Final sparsity: {sparsity*100:.2f}%")
    
    return model, {"total_params": total_params, "zero_params": zero_params, "sparsity": sparsity}

def create_model_copy(model):
    """
    Create a proper copy of the model with the same architecture and parameters.
    This avoids issues with simple type(model)() instantiation.
    """
    # For TransformerEncoder, we need to recreate it with the same parameters
    if isinstance(model, TransformerEncoder):
        # Extract parameters from the model
        vocab_size = model.token_embedding.embedding.num_embeddings if hasattr(model, 'token_embedding') else None
        d_model = model.d_model
        # Get number of heads from the first layer
        num_heads = model.layers[0].self_attn.query_projection.out_features // (d_model // len(model.layers[0].self_attn.query_projection.bias))
        num_layers = len(model.layers)
        # Get d_ff from the first layer's feed-forward network
        d_ff = model.layers[0].feed_forward.linear1.linear.out_features
        dropout = model.layers[0].dropout1.p if hasattr(model.layers[0], 'dropout1') else 0.1
        # Use the max_seq_length from positional encoding
        max_seq_length = model.positional_encoding.max_seq_length if hasattr(model, 'positional_encoding') else 5000
        # Determine positional encoding type
        pos_encoding = "sinusoidal"
        if hasattr(model, 'positional_encoding_type'):
            pos_encoding = model.positional_encoding_type
        
        # Create a new model with the same parameters
        copied_model = TransformerEncoder(
            vocab_size=vocab_size,
            d_model=d_model,
            num_heads=num_heads,
            num_layers=num_layers,
            d_ff=d_ff,
            dropout=dropout,
            max_seq_length=max_seq_length,
            positional_encoding=pos_encoding
        )
        
        # Move to the same device
        copied_model.to(model.get_device() if hasattr(model, 'get_device') else next(model.parameters()).device)
        
        # Copy parameters
        copied_model.load_state_dict(model.state_dict())
        
        return copied_model
    
    # For any other model type, use deepcopy as a fallback
    return copy.deepcopy(model)

def compare_model_outputs(original_model, pruned_model, input_tensor):
    """
    Compare the outputs of the original and pruned models.
    
    Args:
        original_model: The original model before pruning
        pruned_model: The pruned model
        input_tensor: The input tensor to test
    
    Returns:
        A dictionary with comparison metrics
    """
    original_model.eval()
    pruned_model.eval()
    
    with torch.no_grad():
        original_output = original_model(input_tensor)
        pruned_output = pruned_model(input_tensor)
    
    # Calculate the difference between outputs
    output_difference = torch.norm(original_output - pruned_output).item()
    
    print(f"Output difference (L2 norm): {output_difference:.4f}")
    
    return {
        "output_difference": output_difference
    }

def test_weight_pruning(model, method="magnitude", amount=0.3):
    """Test weight pruning on the model."""
    print("\n" + "="*80)
    print(f"TESTING DIRECT MAGNITUDE WEIGHT PRUNING ({amount*100:.0f}%)")
    print("="*80)
    
    # Clone the model for pruning
    model_copy = create_model_copy(model)
    
    # Perform direct magnitude pruning
    start_time = time.time()
    pruned_model, pruning_info = direct_magnitude_pruning(model_copy, amount=amount)
    end_time = time.time()
    
    print(f"Pruning completed in {end_time - start_time:.2f} seconds")
    
    # Display pruning information
    print("\nPruning Information:")
    print(f"Final sparsity: {pruning_info['sparsity']*100:.2f}%")
    print(f"Total parameters: {pruning_info['total_params']:,}")
    print(f"Zero parameters: {pruning_info['zero_params']:,}")
    
    # Test inference with the pruned model
    device = get_device()
    vocab_size = 10000 if hasattr(model, 'vocab_size') else 1000
    input_tensor = torch.randint(0, vocab_size, (1, 64), dtype=torch.long).to(device)
    
    # Ensure the model still runs
    print("\nTesting inference with pruned model...")
    with torch.no_grad():
        output = pruned_model(input_tensor)
    
    print(f"Inference successful. Output shape: {output.shape}")
    
    # Compare outputs
    compare_model_outputs(model, pruned_model, input_tensor)
    
    return pruned_model

def direct_iterative_pruning(model, amount=0.5, iterations=5):
    """
    Apply iterative magnitude pruning directly.
    
    Args:
        model: The model to prune
        amount: Final sparsity target (0.0-1.0)
        iterations: Number of pruning iterations
        
    Returns:
        Pruned model and pruning history
    """
    print(f"Applying iterative magnitude pruning to {amount*100:.0f}% over {iterations} iterations...")
    
    # Calculate per-iteration sparsity targets
    sparsity_targets = []
    for i in range(iterations):
        # Cubic schedule for smoother progression
        progress = (i + 1) / iterations
        target = amount * (progress ** 3)
        sparsity_targets.append(target)
    
    # Initialize tracking variables
    pruning_history = []
    current_sparsity = 0.0
    
    # Apply pruning in iterations
    for i, target_sparsity in enumerate(sparsity_targets):
        print(f"\nIteration {i+1}/{iterations} - Target sparsity: {target_sparsity*100:.2f}%")
        
        # Calculate the additional sparsity needed
        if i == 0:
            # First iteration - prune to the first target
            iteration_sparsity = target_sparsity
        else:
            # Calculate additional sparsity needed
            remaining_weights = 1.0 - current_sparsity
            additional_sparsity = (target_sparsity - current_sparsity) / remaining_weights
            iteration_sparsity = additional_sparsity
        
        # Apply pruning for this iteration
        total_params = 0
        zero_params = 0
        
        for name, param in model.named_parameters():
            if 'weight' in name:
                tensor_size = param.data.numel()
                total_params += tensor_size
                
                # Count current zeros
                current_zeros = (param.data == 0).sum().item()
                
                # Only prune non-zero weights
                non_zero_weights = param.data[param.data != 0]
                
                # If no non-zero weights, skip this parameter
                if len(non_zero_weights) == 0:
                    zero_params += current_zeros
                    continue
                
                # Calculate pruning threshold for this iteration
                n_to_prune = int(len(non_zero_weights) * iteration_sparsity)
                if n_to_prune == 0:
                    zero_params += current_zeros
                    continue
                    
                threshold = non_zero_weights.abs().flatten().sort()[0][n_to_prune]
                
                # Create mask for small weights (but don't prune already pruned weights)
                mask = (param.data.abs() <= threshold) & (param.data != 0)
                
                # Apply mask
                param.data[mask] = 0
                
                # Count zeros after pruning
                new_zeros = (param.data == 0).sum().item()
                zero_params += new_zeros
                
                print(f"  {name}: Pruned {new_zeros - current_zeros} weights")
        
        # Calculate new sparsity
        current_sparsity = zero_params / total_params if total_params > 0 else 0
        
        # Record pruning stats for this iteration
        pruning_history.append({
            "iteration": i + 1,
            "target_sparsity": target_sparsity,
            "actual_sparsity": current_sparsity,
            "total_params": total_params,
            "zero_params": zero_params
        })
        
        print(f"  Iteration {i+1} complete - Actual sparsity: {current_sparsity*100:.2f}%")
    
    return model, {"history": pruning_history, "final_sparsity": current_sparsity}

def test_iterative_pruning(model, amount=0.5, iterations=5):
    """Test iterative magnitude pruning on the model."""
    print("\n" + "="*80)
    print(f"TESTING DIRECT ITERATIVE MAGNITUDE PRUNING ({amount*100:.0f}% over {iterations} iterations)")
    print("="*80)
    
    # Clone the model for pruning
    model_copy = create_model_copy(model)
    
    # Perform direct iterative pruning
    start_time = time.time()
    pruned_model, pruning_info = direct_iterative_pruning(model_copy, amount=amount, iterations=iterations)
    end_time = time.time()
    
    print(f"Pruning completed in {end_time - start_time:.2f} seconds")
    
    # Get pruning information
    print("\nPruning Information:")
    print(f"Final sparsity: {pruning_info['final_sparsity']*100:.2f}%")
    
    # Show sparsity progression
    print("\nSparsity progression:")
    for stats in pruning_info['history']:
        print(f"Iteration {stats['iteration']}: {stats['actual_sparsity']*100:.2f}% sparsity")
    
    # Plot sparsity progression
    plt.figure(figsize=(10, 6))
    iterations = [stats['iteration'] for stats in pruning_info['history']]
    sparsities = [stats['actual_sparsity'] * 100 for stats in pruning_info['history']]
    target_sparsities = [stats['target_sparsity'] * 100 for stats in pruning_info['history']]
    
    plt.plot(iterations, sparsities, marker='o', label='Actual Sparsity')
    plt.plot(iterations, target_sparsities, marker='x', linestyle='--', label='Target Sparsity')
    plt.title('Sparsity Progression During Iterative Pruning')
    plt.xlabel('Iteration')
    plt.ylabel('Sparsity (%)')
    plt.grid(True)
    plt.legend()
    plt.savefig('pruning_progression.png')
    plt.close()
    print("\nSparsity progression plot saved as 'pruning_progression.png'")
    
    # Test inference with the pruned model
    device = get_device()
    vocab_size = 10000 if hasattr(model, 'vocab_size') else 1000
    input_tensor = torch.randint(0, vocab_size, (1, 64), dtype=torch.long).to(device)
    
    # Ensure the model still runs
    print("\nTesting inference with pruned model...")
    with torch.no_grad():
        output = pruned_model(input_tensor)
    
    print(f"Inference successful. Output shape: {output.shape}")
    
    # Compare outputs
    compare_model_outputs(model, pruned_model, input_tensor)
    
    return pruned_model

def test_structured_pruning(model, amount=0.3, dim=0):
    """Test structured pruning (removing entire neurons/filters)."""
    print("\n" + "="*80)
    print(f"TESTING STRUCTURED PRUNING ({amount*100:.0f}%, dim={dim})")
    print("="*80)
    
    # Clone the model for pruning
    model_copy = create_model_copy(model)
    
    # Create configuration
    config = PruningConfig(
        method="structured",
        amount=amount,
        pruning_dims=["weight"],
        dim=dim  # 0 for output neurons, 1 for input connections
    )
    
    # Create pruner
    pruner = ModelPruner(model_copy, config)
    
    # Perform pruning
    print(f"Applying structured pruning with {amount*100:.0f}% sparsity along dimension {dim}...")
    start_time = time.time()
    try:
        pruned_model = pruner.prune_model()
        end_time = time.time()
        
        print(f"Pruning completed in {end_time - start_time:.2f} seconds")
        
        # Get pruning information
        pruning_info = pruner.get_pruning_info()
        print("\nPruning Information:")
        print(f"Final sparsity: {pruning_info['final_sparsity']*100:.2f}%")
        
        # Test inference with the pruned model
        device = get_device()
        vocab_size = 10000 if hasattr(model, 'vocab_size') else 1000
        input_tensor = torch.randint(0, vocab_size, (1, 64), dtype=torch.long).to(device)
        
        # Ensure the model still runs
        print("\nTesting inference with pruned model...")
        with torch.no_grad():
            output = pruned_model(input_tensor)
        
        print(f"Inference successful. Output shape: {output.shape}")
        
        return pruned_model
    except Exception as e:
        print(f"Error during structured pruning: {e}")
        print("Falling back to unstructured pruning...")
        return test_weight_pruning(model, method="magnitude", amount=amount)

def measure_performance(model, input_generator, batch_sizes=[1, 2, 4, 8, 16], num_iterations=50):
    """
    Measure inference performance with different batch sizes.
    
    Args:
        model: Model to benchmark
        input_generator: Function to generate inputs
        batch_sizes: List of batch sizes to test
        num_iterations: Number of iterations for each measurement
        
    Returns:
        Dictionary of performance metrics
    """
    print("\n" + "="*80)
    print("PERFORMANCE MEASUREMENT")
    print("="*80)
    
    device = next(model.parameters()).device
    model.eval()
    
    results = {}
    
    for batch_size in batch_sizes:
        print(f"\nTesting batch size {batch_size}...")
        # Generate input
        inputs = input_generator(batch_size).to(device)
        
        # Warm up
        with torch.no_grad():
            for _ in range(10):
                _ = model(inputs)
        
        # Measure inference time
        latencies = []
        with torch.no_grad():
            for _ in range(num_iterations):
                torch.cuda.synchronize() if device.type == "cuda" else None
                start_time = time.time()
                _ = model(inputs)
                torch.cuda.synchronize() if device.type == "cuda" else None
                end_time = time.time()
                latencies.append(end_time - start_time)
        
        # Calculate statistics
        avg_latency = sum(latencies) / len(latencies)
        min_latency = min(latencies)
        max_latency = max(latencies)
        p95_latency = sorted(latencies)[int(0.95 * len(latencies))]
        throughput = batch_size / avg_latency
        
        print(f"  Average latency: {avg_latency*1000:.2f} ms")
        print(f"  Throughput: {throughput:.2f} examples/second")
        
        results[batch_size] = {
            "avg_latency": avg_latency,
            "min_latency": min_latency,
            "max_latency": max_latency,
            "p95_latency": p95_latency,
            "throughput": throughput
        }
    
    return results


def compare_model_performance(models_dict, input_generator, batch_sizes=[1, 2, 4, 8, 16]):
    """
    Compare performance between original and optimized models.
    
    Args:
        models_dict: Dictionary mapping model names to model instances
        input_generator: Function to generate inputs
        batch_sizes: List of batch sizes to test
        
    Returns:
        Dictionary of comparative results
    """
    print("\n" + "="*80)
    print("MODEL PERFORMANCE COMPARISON")
    print("="*80)
    
    results = {}
    
    # Measure performance for each model
    for name, model in models_dict.items():
        print(f"\nMeasuring performance for {name}...")
        results[name] = measure_performance(model, input_generator, batch_sizes)
    
    # Calculate speedups relative to base model
    base_model = list(models_dict.keys())[0]  # First model is the base
    speedups = {}
    
    for name in list(models_dict.keys())[1:]:  # Skip the base model
        speedups[name] = {}
        for batch_size in batch_sizes:
            base_latency = results[base_model][batch_size]["avg_latency"]
            model_latency = results[name][batch_size]["avg_latency"]
            speedup = base_latency / model_latency
            speedups[name][batch_size] = speedup
            print(f"{name} speedup at batch size {batch_size}: {speedup:.2f}x")
    
    # Plot comparison
    plt.figure(figsize=(12, 8))
    
    for name in models_dict.keys():
        throughputs = [results[name][bs]["throughput"] for bs in batch_sizes]
        plt.plot(batch_sizes, throughputs, marker='o', label=name)
    
    plt.title("Model Throughput Comparison")
    plt.xlabel("Batch Size")
    plt.ylabel("Throughput (examples/second)")
    plt.grid(True)
    plt.legend()
    plt.savefig("model_comparison.png")
    plt.close()
    
    # Plot speedups
    if speedups:
        plt.figure(figsize=(12, 8))
        
        for name, speedup_dict in speedups.items():
            speed_values = [speedup_dict[bs] for bs in batch_sizes]
            plt.plot(batch_sizes, speed_values, marker='o', label=name)
        
        plt.axhline(y=1.0, color='r', linestyle='--', label='Baseline')
        plt.title("Speedup Comparison")
        plt.xlabel("Batch Size")
        plt.ylabel("Speedup (relative to original)")
        plt.grid(True)
        plt.legend()
        plt.savefig("speedup_comparison.png")
        plt.close()
    
    print("\nComparison plots saved as 'model_comparison.png' and 'speedup_comparison.png'")
    
    return results, speedups

def create_input_generator(vocab_size, max_seq_length=128):
    """Create an input generator function for transformer models."""
    def generate_input(batch_size):
        # Generate random token indices
        token_indices = torch.randint(0, vocab_size, (batch_size, max_seq_length), dtype=torch.long)
        return token_indices
    return generate_input

def main():
    """Run the model optimization demo."""
    parser = argparse.ArgumentParser(description="Model Optimization Demo")
    
    # Add command line arguments
    parser.add_argument("--model", type=str, default="medium", choices=["small", "medium"],
                      help="Model size to use for testing")
    parser.add_argument("--output_dir", type=str, default="optimization_results",
                      help="Directory to save benchmark results")
    parser.add_argument("--seed", type=int, default=42,
                      help="Random seed for reproducibility")
    parser.add_argument("--compare", action="store_true",
                      help="Run performance comparison between original and optimized models")
    
    args = parser.parse_args()
    
    # Set random seed
    set_seed(args.seed)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Print system info
    print("\n" + "="*80)
    print("MODEL OPTIMIZATION DEMO")
    print("="*80)
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"MPS available: {torch.backends.mps.is_available()}")
    print(f"Using device: {get_device()}")
    
    # Create test model
    original_model = create_test_transformer(size=args.model)
    
    # Create models dictionary for comparison
    models = {"Original": original_model}
    
    # Run pruning tests with direct implementations
    pruned_model = test_weight_pruning(original_model, amount=0.3)
    models["Pruned (30%)"] = pruned_model
    
    # Compare outputs
    input_tensor = torch.randint(0, 10000, (1, 64), dtype=torch.long).to(get_device())
    compare_model_outputs(original_model, pruned_model, input_tensor)
    
    iterative_model = test_iterative_pruning(original_model, amount=0.5, iterations=5)
    models["Iteratively Pruned (50%)"] = iterative_model
    
    # Compare outputs
    compare_model_outputs(original_model, iterative_model, input_tensor)
    
    # Run performance comparison if requested
    if args.compare:
        # Create input generator
        vocab_size = 10000 if hasattr(original_model, 'vocab_size') else 1000
        input_gen = create_input_generator(vocab_size)
        
        # Determine batch sizes based on available memory
        device = get_device()
        if device.type == "cuda" and torch.cuda.get_device_properties(0).total_memory > 8e9:
            batch_sizes = [1, 4, 16, 32, 64]
        else:
            # MPS or lower memory device
            batch_sizes = [1, 2, 4, 8, 16]
        
        # Compare performance
        results, speedups = compare_model_performance(models, input_gen, batch_sizes)
    
    print("\n" + "="*80)
    print("Model optimization demo completed!")
    print("="*80)
    
    return 0

if __name__ == "__main__":
    sys.exit(main())