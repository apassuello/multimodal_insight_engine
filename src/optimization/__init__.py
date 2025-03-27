# src/optimization/__init__.py
import torch
from .quantization import QuantizationConfig, DynamicQuantizer, StaticQuantizer
from .mixed_precision import MixedPrecisionConverter
from .pruning import PruningConfig, ModelPruner
from .benchmarking import OptimizationBenchmark

# Convenience functions for easy access
def quantize_model(
    model,
    quantization_type="dynamic",
    dtype=None,
    bits=8,
    calibration_loader=None,
    **kwargs
):
    """
    Quantize a model using the specified approach.
    
    Args:
        model: The model to quantize
        quantization_type: Type of quantization ("dynamic", "static", or "qat")
        dtype: Target data type for quantization
        bits: Bit width for quantization (8 or 16)
        calibration_loader: DataLoader providing calibration data (for static quantization)
        **kwargs: Additional arguments for the quantizer
        
    Returns:
        Quantized model
    """
    config = QuantizationConfig(
        quantization_type=quantization_type,
        dtype=dtype,
        bits=bits,
        **kwargs
    )
    
    if quantization_type == "dynamic":
        quantizer = DynamicQuantizer(model, config)
    elif quantization_type == "static":
        quantizer = StaticQuantizer(model, config, calibration_loader)
    else:
        raise ValueError(f"Unsupported quantization type: {quantization_type}")
    
    return quantizer.optimize()

def prune_model(
    model,
    method="magnitude",
    amount=0.2,
    **kwargs
):
    """
    Prune a model using the specified approach.
    
    Args:
        model: The model to prune
        method: Pruning method ("magnitude", "structured", "iterative_magnitude")
        amount: Amount to prune (percentage)
        **kwargs: Additional arguments for the pruner
        
    Returns:
        Pruned model
    """
    config = PruningConfig(
        method=method,
        amount=amount,
        **kwargs
    )
    
    pruner = ModelPruner(model, config)
    return pruner.prune_model()

def convert_to_mixed_precision(
    model,
    dtype=None,
    use_auto_cast=True
):
    """
    Convert a model to use mixed precision.
    
    Args:
        model: The model to convert
        dtype: Target data type (torch.float16 or torch.bfloat16)
        use_auto_cast: Whether to use automatic mixed precision
        
    Returns:
        Model with mixed precision
    """
    # Set default dtype based on available hardware
    if dtype is None:
        if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
            dtype = torch.bfloat16
        else:
            dtype = torch.float16
    
    converter = MixedPrecisionConverter(model, dtype, use_auto_cast)
    return converter.convert_to_mixed_precision()

def benchmark_optimizations(
    model,
    input_generator,
    optimizations=[],
    **kwargs
):
    """
    Benchmark a model with different optimizations.
    
    Args:
        model: The original model to benchmark
        input_generator: Function to generate inputs
        optimizations: List of optimization functions to apply
        **kwargs: Additional arguments for the benchmark
        
    Returns:
        OptimizationBenchmark instance with results
    """
    benchmark = OptimizationBenchmark(model, input_generator, **kwargs)
    
    # Benchmark original model
    benchmark.benchmark_original_model()
    
    # Apply and benchmark each optimization
    for i, (opt_fn, opt_kwargs, name) in enumerate(optimizations):
        # Apply optimization
        optimized_model = opt_fn(model, **opt_kwargs)
        
        # Benchmark optimized model
        benchmark.benchmark_optimized_model(optimized_model, name)
    
    # Generate comparison
    benchmark.compare_optimizations()
    
    # Generate report
    benchmark.generate_report()
    
    return benchmark