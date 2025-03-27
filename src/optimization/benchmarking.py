# src/optimization/benchmarking.py
import time
from typing import Dict, List, Any, Callable
import json
import os
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

class OptimizationBenchmark:
    """
    Framework for measuring and comparing model optimization techniques.
    
    This class provides utilities for benchmarking models before and after
    various optimization techniques like quantization, pruning, and mixed precision.
    """
    
    def __init__(
        self,
        model: nn.Module,
        input_generator: Callable[[int], torch.Tensor],
        batch_sizes: List[int] = [1, 4, 16, 32],
        precision: float = 0.001,  # Desired time measurement precision
        save_dir: str = "benchmark_results",
    ):
        """
        Initialize the optimization benchmark.
        
        Args:
            model: The original model to benchmark
            input_generator: Function to generate inputs of specified batch size
            batch_sizes: List of batch sizes to benchmark
            precision: Desired time measurement precision
            save_dir: Directory to save benchmark results
        """
        self.model = model
        self.input_generator = input_generator
        self.batch_sizes = batch_sizes
        self.precision = precision
        self.save_dir = save_dir
        
        # Ensure save directory exists
        os.makedirs(save_dir, exist_ok=True)
        
        # Track optimized models
        self.optimized_models = {}
        
        # Results storage
        self.results = {}
    
    def benchmark_original_model(self) -> Dict[str, Any]:
        """
        Benchmark the original unoptimized model.
        
        Returns:
            Dictionary with benchmark results
        """
        # Store the original model's results
        return self._benchmark_model(self.model, "original")
    
    def benchmark_optimized_model(
        self,
        model: nn.Module,
        name: str,
    ) -> Dict[str, Any]:
        """
        Benchmark an optimized model.
        
        Args:
            model: The optimized model to benchmark
            name: Name to identify this optimization
            
        Returns:
            Dictionary with benchmark results
        """
        # Store the optimized model
        self.optimized_models[name] = model
        
        # Benchmark the optimized model
        return self._benchmark_model(model, name)
    
    def _benchmark_model(
        self,
        model: nn.Module,
        name: str,
    ) -> Dict[str, Any]:
        """
        Benchmark a model's performance.
        
        Args:
            model: The model to benchmark
            name: Name to identify this model
            
        Returns:
            Dictionary with benchmark results
        """
        device = next(model.parameters()).device
        model.eval()
        
        results = {"name": name, "batch_sizes": {}}
        
        for batch_size in self.batch_sizes:
            # Generate input
            inputs = self.input_generator(batch_size).to(device)
            
            # Warm up
            with torch.no_grad():
                for _ in range(10):
                    _ = model(inputs)
            
            # Measure memory usage
            torch.cuda.reset_peak_memory_stats(device)
            torch.cuda.empty_cache()
            
            # Benchmark timing
            times = []
            with torch.no_grad():
                # Determine number of iterations needed for precision
                # Start with a small number and adjust
                n_iterations = 10
                
                while True:
                    start = time.time()
                    for _ in range(n_iterations):
                        _ = model(inputs)
                    end = time.time()
                    
                    elapsed = end - start
                    if elapsed < 0.1:
                        # Too short, increase iterations
                        n_iterations *= 10
                    else:
                        # Record the average time
                        times = [elapsed / n_iterations]
                        break
                
                # Now run the actual benchmark
                for _ in range(5):
                    start = time.time()
                    for _ in range(n_iterations):
                        _ = model(inputs)
                    elapsed = (time.time() - start) / n_iterations
                    times.append(elapsed)
            
            # Calculate memory usage
            if device.type == "cuda":
                memory_used = torch.cuda.max_memory_allocated(device) / (1024 * 1024)  # MB
            else:
                memory_used = None
            
            # Calculate statistics
            avg_time = np.mean(times)
            std_time = np.std(times)
            
            # Store results
            batch_results = {
                "avg_time": avg_time,
                "std_time": std_time,
                "iterations_per_second": 1.0 / avg_time,
                "memory_mb": memory_used,
            }
            
            results["batch_sizes"][batch_size] = batch_results
        
        # Store in results dictionary
        self.results[name] = results
        
        return results
    
    def compare_optimizations(self, save_plot: bool = True) -> Dict[str, Any]:
        """
        Compare all benchmarked optimizations.
        
        Args:
            save_plot: Whether to save comparison plots
            
        Returns:
            Dictionary with comparison results
        """
        if "original" not in self.results:
            self.benchmark_original_model()
        
        comparison = {
            "baseline": "original",
            "optimizations": {},
        }
        
        for name, results in self.results.items():
            if name == "original":
                continue
            
            # Compare to original model
            speedups = {}
            memory_reductions = {}
            
            for batch_size, metrics in results["batch_sizes"].items():
                original_metrics = self.results["original"]["batch_sizes"][batch_size]
                
                # Calculate speedup
                speedup = original_metrics["avg_time"] / metrics["avg_time"]
                speedups[batch_size] = speedup
                
                # Calculate memory reduction if available
                if metrics["memory_mb"] is not None and original_metrics["memory_mb"] is not None:
                    memory_reduction = original_metrics["memory_mb"] / metrics["memory_mb"]
                    memory_reductions[batch_size] = memory_reduction
            
            # Store comparison
            comparison["optimizations"][name] = {
                "speedups": speedups,
                "memory_reductions": memory_reductions,
                "avg_speedup": np.mean(list(speedups.values())),
                "avg_memory_reduction": np.mean(list(memory_reductions.values())) if memory_reductions else None,
            }
        
        # Save comparison plots
        if save_plot:
            self._plot_comparison(comparison)
        
        return comparison
    
    def _plot_comparison(self, comparison: Dict[str, Any]):
        """
        Plot comparison of optimization techniques.
        
        Args:
            comparison: Comparison results from compare_optimizations
        """
        # Create speedup plot
        plt.figure(figsize=(12, 8))
        
        # Extract data for plotting
        x = self.batch_sizes
        lines = []
        labels = []
        
        for name, metrics in comparison["optimizations"].items():
            speedups = [metrics["speedups"].get(bs, 1.0) for bs in x]
            line, = plt.plot(x, speedups, marker='o', linestyle='-', linewidth=2)
            lines.append(line)
            labels.append(name)
        
        plt.axhline(y=1.0, color='r', linestyle='--', label='Baseline')
        
        plt.xlabel('Batch Size')
        plt.ylabel('Speedup (x)')
        plt.title('Optimization Speedup by Batch Size')
        plt.legend(lines + [Line2D([0], [0], color='r', linestyle='--')], 
                 labels + ['Baseline'])
        plt.grid(True)
        
        # Set x-axis to logarithmic scale if wide range of batch sizes
        if max(x) / min(x) > 10:
            plt.xscale('log', base=2)
        
        # Save the plot
        plt.savefig(os.path.join(self.save_dir, 'optimization_speedup.png'))
        
        # Create memory reduction plot if data is available
        memory_data = False
        for metrics in comparison["optimizations"].values():
            if metrics["avg_memory_reduction"] is not None:
                memory_data = True
                break
        
        if memory_data:
            plt.figure(figsize=(12, 8))
            
            lines = []
            labels = []
            
            for name, metrics in comparison["optimizations"].items():
                if metrics["avg_memory_reduction"] is None:
                    continue
                    
                reductions = [metrics["memory_reductions"].get(bs, 1.0) for bs in x]
                line, = plt.plot(x, reductions, marker='o', linestyle='-', linewidth=2)
                lines.append(line)
                labels.append(name)
            
            plt.axhline(y=1.0, color='r', linestyle='--', label='Baseline')
            
            plt.xlabel('Batch Size')
            plt.ylabel('Memory Reduction (x)')
            plt.title('Optimization Memory Reduction by Batch Size')
            plt.legend(lines + [Line2D([0], [0], color='r', linestyle='--')], 
                     labels + ['Baseline'])
            plt.grid(True)
            
            # Set x-axis to logarithmic scale if wide range of batch sizes
            if max(x) / min(x) > 10:
                plt.xscale('log', base=2)
            
            # Save the plot
            plt.savefig(os.path.join(self.save_dir, 'optimization_memory.png'))
    
    def save_results(self, filename: str = "optimization_benchmark.json"):
        """
        Save benchmark results to a file.
        
        Args:
            filename: Name of the file to save results to
        """
        # Save results as JSON
        with open(os.path.join(self.save_dir, filename), 'w') as f:
            json.dump(self.results, f, indent=2)
    
    def generate_report(self) -> str:
        """
        Generate a human-readable report of benchmark results.
        
        Returns:
            Markdown report as string
        """
        comparison = self.compare_optimizations(save_plot=False)
        
        # Create markdown report
        report = [
            "# Model Optimization Benchmark Report\n",
            "## Summary\n",
            f"- Original model: {type(self.model).__name__}\n",
            f"- Device: {next(self.model.parameters()).device}\n",
            f"- Batch sizes: {self.batch_sizes}\n",
            f"- Optimizations tested: {len(comparison['optimizations'])}\n\n",
            "## Optimization Results\n",
        ]
        
        # Add table header
        report.append("| Optimization | Avg Speedup | Avg Memory Reduction | Best For |\n")
        report.append("|-------------|------------|---------------------|----------|\n")
        
        # Add data rows
        for name, metrics in comparison["optimizations"].items():
            # Determine what this optimization is best for
            best_for = []
            
            if metrics["avg_speedup"] > 1.5:
                best_for.append("Inference speed")
            
            if metrics["avg_memory_reduction"] and metrics["avg_memory_reduction"] > 1.5:
                best_for.append("Memory efficiency")
            
            if not best_for:
                best_for.append("Minimal impact")
            
            # Format row
            memory_reduction_str = f"{metrics['avg_memory_reduction']:.2f}x" if metrics['avg_memory_reduction'] is not None else 'N/A'
            report.append(
                f"| {name} | {metrics['avg_speedup']:.2f}x | {memory_reduction_str} | "
                f"{', '.join(best_for)} |\n"
            )
        
        report.append("\n## Detailed Results by Batch Size\n")
        
        # Add detailed results
        for name, results in self.results.items():
            report.append(f"### {name}\n\n")
            
            # Add table header
            report.append("| Batch Size | Time (ms) | Iterations/sec | Memory (MB) |\n")
            report.append("|------------|----------|--------------|------------|\n")
            
            # Add data rows
            for batch_size, metrics in results["batch_sizes"].items():
                memory_str = f"{metrics['memory_mb']:.1f}" if metrics['memory_mb'] is not None else "N/A"
                report.append(
                    f"| {batch_size} | {metrics['avg_time']*1000:.2f} ± {metrics['std_time']*1000:.2f} | "
                    f"{metrics['iterations_per_second']:.1f} | {memory_str} |\n"
                )
            
            report.append("\n")
        
        report.append("## Recommendations\n\n")
        
        # Generate recommendations
        recommendations = self._generate_recommendations(comparison)
        report.extend(recommendations)
        
        # Save report
        report_text = "".join(report)
        with open(os.path.join(self.save_dir, "optimization_report.md"), 'w') as f:
            f.write(report_text)
        
        return report_text
    
    def _generate_recommendations(self, comparison: Dict[str, Any]) -> List[str]:
        """
        Generate optimization recommendations based on benchmark results.
        
        Args:
            comparison: Comparison results from compare_optimizations
            
        Returns:
            List of recommendation strings
        """
        recommendations = []
        
        # Find the best overall optimization
        best_overall = None
        best_speedup = 1.0
        
        for name, metrics in comparison["optimizations"].items():
            if metrics["avg_speedup"] > best_speedup:
                best_speedup = metrics["avg_speedup"]
                best_overall = name
        
        if best_overall:
            recommendations.append(f"- **Best overall optimization**: {best_overall} with {best_speedup:.2f}x average speedup\n")
        
        # Find best for different scenarios
        best_for_small_batch = None
        best_small_speedup = 1.0
        
        best_for_large_batch = None
        best_large_speedup = 1.0
        
        min_batch = min(self.batch_sizes)
        max_batch = max(self.batch_sizes)
        
        for name, metrics in comparison["optimizations"].items():
            small_speedup = metrics["speedups"].get(min_batch, 1.0)
            if small_speedup > best_small_speedup:
                best_small_speedup = small_speedup
                best_for_small_batch = name
                
            large_speedup = metrics["speedups"].get(max_batch, 1.0)
            if large_speedup > best_large_speedup:
                best_large_speedup = large_speedup
                best_for_large_batch = name
        
        if best_for_small_batch:
            recommendations.append(f"- **Best for small batches**: {best_for_small_batch} with {best_small_speedup:.2f}x speedup at batch size {min_batch}\n")
            
        if best_for_large_batch:
            recommendations.append(f"- **Best for large batches**: {best_for_large_batch} with {best_large_speedup:.2f}x speedup at batch size {max_batch}\n")
        
        # Check for memory-constrained scenarios
        best_memory_reduction = None
        best_reduction = 1.0
        
        for name, metrics in comparison["optimizations"].items():
            if metrics["avg_memory_reduction"] and metrics["avg_memory_reduction"] > best_reduction:
                best_reduction = metrics["avg_memory_reduction"]
                best_memory_reduction = name
        
        if best_memory_reduction:
            recommendations.append(f"- **Best for memory-constrained devices**: {best_memory_reduction} with {best_reduction:.2f}x memory reduction\n")
        
        # Add general recommendations
        recommendations.append("\n### Implementation Recommendations\n")
        
        # Dynamic vs. static quantization
        dynamic_better = False
        static_better = False
        
        for name, metrics in comparison["optimizations"].items():
            if "dynamic" in name.lower() and metrics["avg_speedup"] > 1.2:
                dynamic_better = True
            if "static" in name.lower() and metrics["avg_speedup"] > 1.2:
                static_better = True
        
        if dynamic_better and static_better:
            recommendations.append("- Consider using **static quantization** for server deployment and **dynamic quantization** for mobile/edge devices.\n")
        elif dynamic_better:
            recommendations.append("- **Dynamic quantization** shows good results and is easier to implement than static quantization.\n")
        elif static_better:
            recommendations.append("- **Static quantization** provides the best performance but requires calibration data.\n")
        
        # Pruning recommendations
        pruning_effective = False
        for name, metrics in comparison["optimizations"].items():
            if "pruning" in name.lower() and metrics["avg_speedup"] > 1.2:
                pruning_effective = True
                break
        
        if pruning_effective:
            recommendations.append("- **Weight pruning** is effective for this model. Consider incorporating it into your training pipeline with fine-tuning.\n")
        else:
            found_pruning = False
            for name in comparison["optimizations"]:
                if "pruning" in name.lower():
                    found_pruning = True
                    break
                    
            if found_pruning:
                recommendations.append("- **Weight pruning** did not significantly improve performance. Consider focusing on other optimization techniques.\n")
        
        # Mixed precision recommendations
        fp16_effective = False
        for name, metrics in comparison["optimizations"].items():
            if "fp16" in name.lower() or "mixed" in name.lower():
                if metrics["avg_speedup"] > 1.2:
                    fp16_effective = True
                    break
        
        if fp16_effective:
            recommendations.append("- **Mixed precision (FP16)** provides good speedup with minimal accuracy impact. Highly recommended for your hardware.\n")
        
        # Apple Silicon specific
        device = next(self.model.parameters()).device
        if device.type == "mps":
            recommendations.append("- For **Apple Silicon**, make sure to enable MPS acceleration and use FP16 precision for optimal performance.\n")
        
        return recommendations