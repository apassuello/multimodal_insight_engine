# src/utils/profiling.py         
import time
import torch
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Optional, Union, Callable, Any
import os
import psutil
import pandas as pd
import seaborn as sns

class ModelProfiler:
    """
    A utility for profiling PyTorch models.
    
    This class provides methods for measuring execution time, memory usage,
    and other performance metrics for PyTorch models.
    """
    
    def __init__(self, model: torch.nn.Module, device: Optional[torch.device] = None):
        """
        Initialize the profiler.
        
        Args:
            model: The PyTorch model to profile
            device: The device to run profiling on (defaults to model's device)
        """
        self.model = model
# src/utils/profiling.py
        # Get device from model if not specified
        if device is None:
            # Get the device of the first parameter
            self.device = next(model.parameters()).device
        else:
            self.device = device

        self.device = device or torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    
        # Initialize metrics storage
        self.metrics = {
            'execution_time': [],
            'memory_usage': [],
            'parameter_count': self._count_parameters(),
        }
        
        # Storage for detailed profiling results
        self.detailed_metrics = {}
        
        # Create results directory
        os.makedirs('profiling_results', exist_ok=True)
    
    def _count_parameters(self) -> Dict[str, int]:
        """
        Count the number of parameters in the model.
        
        Returns:
            Dictionary with parameter counts
        """
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        return {
            'total': total_params,
            'trainable': trainable_params,
            'non_trainable': total_params - trainable_params
        }
    
    def measure_execution_time(
        self, 
        input_data: Union[torch.Tensor, Dict[str, torch.Tensor]], 
        iterations: int = 10,
        warmup: int = 2
    ) -> Dict[str, float]:
        """
        Measure the execution time of a forward pass.
        
        Args:
            input_data: Input tensor or dictionary of tensors
            iterations: Number of iterations to average over
            warmup: Number of warmup iterations (not measured)
            
        Returns:
            Dictionary with timing metrics
        """
        # Move input data to the correct device
        if isinstance(input_data, torch.Tensor):
            input_data = input_data.to(self.device)
        elif isinstance(input_data, dict):
            input_data = {k: v.to(self.device) if torch.is_tensor(v) else v 
                          for k, v in input_data.items()}
        
        # Set model to evaluation mode
        self.model.eval()
        
        # Perform warmup iterations
        with torch.no_grad():
            for _ in range(warmup):
                _ = self.model(input_data)
        
        # Measure execution time
        start_time = time.time()
        with torch.no_grad():
            for _ in range(iterations):
                _ = self.model(input_data)
        end_time = time.time()
        
        # Calculate metrics
        total_time = end_time - start_time
        avg_time = total_time / iterations
        iterations_per_second = iterations / total_time
        
        # Store results
        timing_metrics = {
            'total_time': total_time,
            'avg_time': avg_time,
            'iterations_per_second': iterations_per_second
        }
        
        self.metrics['execution_time'].append(timing_metrics)
        
        return timing_metrics
    
    def measure_memory_usage(
        self, 
        input_data: Union[torch.Tensor, Dict[str, torch.Tensor]]
    ) -> Dict[str, float]:
        """
        Measure the memory usage during a forward pass.
        
        Args:
            input_data: Input tensor or dictionary of tensors
            
        Returns:
            Dictionary with memory metrics
        """
        # Move input data to the correct device
        if isinstance(input_data, torch.Tensor):
            input_data = input_data.to(self.device)
        elif isinstance(input_data, dict):
            input_data = {k: v.to(self.device) if torch.is_tensor(v) else v 
                          for k, v in input_data.items()}
        
        # Set model to evaluation mode
        self.model.eval()
        
        # Measure memory before forward pass
        torch.mps.empty_cache() if hasattr(torch.mps, 'empty_cache') else None
        process = psutil.Process(os.getpid())
        memory_before = process.memory_info().rss / (1024 * 1024)  # Convert to MB
        
        # Perform forward pass
        with torch.no_grad():
            _ = self.model(input_data)
        
        # Measure memory after forward pass
        memory_after = process.memory_info().rss / (1024 * 1024)  # Convert to MB
        
        # Calculate metrics
        memory_usage = memory_after - memory_before
        
        # Store results
        memory_metrics = {
            'memory_before_mb': memory_before,
            'memory_after_mb': memory_after,
            'memory_used_mb': memory_usage
        }
        
        self.metrics['memory_usage'].append(memory_metrics)
        
        return memory_metrics
    
    def generate_report(self, save_path: Optional[str] = None) -> str:
        """
        Generate a human-readable report of the profiling results.
        
        Args:
            save_path: Optional path to save the report
            
        Returns:
            Report string
        """
        report = "# Model Profiling Report\n\n"
        
        # Add parameter counts
        params = self.metrics['parameter_count']
        report += "## Model Parameters\n\n"
        report += f"- Total parameters: {params['total']:,}\n"
        report += f"- Trainable parameters: {params['trainable']:,} ({params['trainable']/params['total']*100:.2f}%)\n"
        report += f"- Non-trainable parameters: {params['non_trainable']:,} ({params['non_trainable']/params['total']*100:.2f}%)\n\n"
        
        # Add execution time metrics
        if self.metrics['execution_time']:
            report += "## Execution Time\n\n"
            
            # Calculate statistics across all measurements
            times = [m['avg_time'] for m in self.metrics['execution_time']]
            iterations = [m['iterations_per_second'] for m in self.metrics['execution_time']]
            
            report += f"- Average execution time: {np.mean(times):.6f}s (±{np.std(times):.6f}s)\n"
            report += f"- Median execution time: {np.median(times):.6f}s\n"
            report += f"- Min execution time: {np.min(times):.6f}s\n"
            report += f"- Max execution time: {np.max(times):.6f}s\n"
            report += f"- Average iterations per second: {np.mean(iterations):.2f}\n\n"
        
        # Add memory usage metrics
        if self.metrics['memory_usage']:
            report += "## Memory Usage\n\n"
            
            # Calculate statistics across all measurements
            memory = [m['memory_used_mb'] for m in self.metrics['memory_usage']]
            
            report += f"- Average memory usage: {np.mean(memory):.2f}MB (±{np.std(memory):.2f}MB)\n"
            report += f"- Median memory usage: {np.median(memory):.2f}MB\n"
            report += f"- Min memory usage: {np.min(memory):.2f}MB\n"
            report += f"- Max memory usage: {np.max(memory):.2f}MB\n\n"
        
        # Add detailed metrics if available
        if self.detailed_metrics:
            report += "## Detailed Metrics\n\n"
            
            for name, metrics in self.detailed_metrics.items():
                report += f"### {name}\n\n"
                
                # Format metrics as a table if they're a dictionary
                if isinstance(metrics, dict):
                    report += "| Metric | Value |\n"
                    report += "| ------ | ----- |\n"
                    for metric, value in metrics.items():
                        report += f"| {metric} | {value} |\n"
                else:
                    report += f"{metrics}\n"
                
                report += "\n"
        
        # Save report if path is provided
        if save_path:
            with open(save_path, 'w') as f:
                f.write(report)
            print(f"Report saved to {save_path}")
        
        return report
    
    def plot_metrics(self, save_dir: Optional[str] = None) -> Dict[str, plt.Figure]:
        """
        Plot various metrics from the profiling results.
        
        Args:
            save_dir: Optional directory to save the plots
            
        Returns:
            Dictionary of matplotlib figures
        """
        figures = {}
        
        # Create save directory if provided
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
        
        # Plot execution time
        if self.metrics['execution_time']:
            fig, ax = plt.subplots(figsize=(10, 6))
            times = [m['avg_time'] for m in self.metrics['execution_time']]
            iterations = list(range(1, len(times) + 1))
            
            ax.plot(iterations, times, 'o-', label='Average execution time')
            ax.set_xlabel('Measurement')
            ax.set_ylabel('Time (s)')
            ax.set_title('Model Execution Time')
            ax.grid(True)
            
            figures['execution_time'] = fig
            
            if save_dir:
                fig.savefig(f"{save_dir}/execution_time.png")
        
        # Plot memory usage
        if self.metrics['memory_usage']:
            fig, ax = plt.subplots(figsize=(10, 6))
            memory = [m['memory_used_mb'] for m in self.metrics['memory_usage']]
            iterations = list(range(1, len(memory) + 1))
            
            ax.plot(iterations, memory, 'o-', label='Memory usage')
            ax.set_xlabel('Measurement')
            ax.set_ylabel('Memory (MB)')
            ax.set_title('Model Memory Usage')
            ax.grid(True)
            
            figures['memory_usage'] = fig
            
            if save_dir:
                fig.savefig(f"{save_dir}/memory_usage.png")
        
        return figures

    def profile_with_pytorch_profiler(
        self, 
        input_data: Union[torch.Tensor, Dict[str, torch.Tensor]],
        use_mps: bool = True,
        num_steps: int = 10,
        warmup: int = 3,
        activities: Optional[List[str]] = None,
        record_shapes: bool = True,
        profile_memory: bool = True,
        save_path: Optional[str] = None
    ) -> None:
        """
        Profile the model using PyTorch's built-in profiler.
        
        Args:
            input_data: Input tensor or dictionary of tensors
            use_mps: Whether to use MPS profiling
            num_steps: Number of steps to profile
            warmup: Number of warmup steps
            activities: List of activities to profile
            record_shapes: Whether to record tensor shapes
            profile_memory: Whether to profile memory usage
            save_path: Path to save the trace
        """
        # Import inside function to avoid dependency issues
        from torch.profiler import profile, record_function, ProfilerActivity
        
        # Set default activities if not provided
        if activities is None:
            if use_mps and torch.backends.mps.is_available():
                activities = [ProfilerActivity.CPU, ProfilerActivity.MPS]
            else:
                activities = [ProfilerActivity.CPU]
        
        # Move input data to the correct device
        if isinstance(input_data, torch.Tensor):
            input_data = input_data.to(self.device)
        elif isinstance(input_data, dict):
            input_data = {k: v.to(self.device) if torch.is_tensor(v) else v 
                        for k, v in input_data.items()}
        
        # Set model to evaluation mode
        self.model.eval()
        
        # Create the profiler
        with profile(
            activities=activities,
            record_shapes=record_shapes,
            profile_memory=profile_memory,
            with_stack=True
        ) as prof:
            # Warmup iterations
            for _ in range(warmup):
                with torch.no_grad():
                    _ = self.model(input_data)
                    
            # Profiled iterations
            for step in range(num_steps):
                with record_function(f"step_{step}"):
                    with torch.no_grad():
                        _ = self.model(input_data)
                prof.step()
        
        # Store profiling results
        self.detailed_metrics['pytorch_profiler'] = {
            'table': prof.key_averages().table(sort_by="cpu_time_total"),
            'total_cpu_time': prof.key_averages().total_average().cpu_time_total,
            'total_self_cpu_time': prof.key_averages().total_average().self_cpu_time_total,
        }
        
        # Save trace if path is provided
        if save_path:
            prof.export_chrome_trace(save_path)
            print(f"Trace saved to {save_path}")
        
        # Print summary
        print(prof.key_averages().table(sort_by="cpu_time_total"))
        
        # Analyze by operator type
        op_table = prof.key_averages(group_by_input_shape=True).table(sort_by="cpu_time_total")
        self.detailed_metrics['pytorch_profiler_ops'] = op_table
        print("\nProfile by operator type:")
        print(op_table)
        
        # Extract data for plotting
        df = pd.DataFrame(
            [(evt.key, evt.cpu_time_total, evt.self_cpu_time_total) 
            for evt in prof.key_averages()],
            columns=['Operation', 'CPU Time', 'Self CPU Time']
        )
        df = df.sort_values('CPU Time', ascending=False).head(15)
        
        # Plot the results
        fig, ax = plt.subplots(figsize=(12, 8))
        df.plot.barh(x='Operation', y=['CPU Time', 'Self CPU Time'], ax=ax)
        ax.set_xlabel('Time (μs)')
        ax.set_title('Top 15 Operations by CPU Time')
        
        # Save plot if directory is provided
        if save_path:
            save_dir = os.path.dirname(save_path)
            fig.savefig(f"{save_dir}/op_profile.png")
        
        # Store the figure
        self.detailed_metrics['pytorch_profiler_fig'] = fig

    def benchmark_model(
        self,
        input_generator: Callable[[int, int], Union[torch.Tensor, Dict[str, torch.Tensor]]],
        batch_sizes: List[int],
        sequence_lengths: List[int],
        num_iterations: int = 5,
        save_dir: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Benchmark the model across different batch sizes and sequence lengths.
        
        Args:
            input_generator: Function that generates inputs given batch_size and seq_length
            batch_sizes: List of batch sizes to test
            sequence_lengths: List of sequence lengths to test
            num_iterations: Number of iterations for each configuration
            save_dir: Directory to save results
            
        Returns:
            DataFrame with benchmarking results
        """
        results = []
        
        total_configurations = len(batch_sizes) * len(sequence_lengths)
        current_config = 0
        
        for batch_size in batch_sizes:
            for seq_length in sequence_lengths:
                current_config += 1
                print(f"Benchmarking configuration {current_config}/{total_configurations}: "
                    f"batch_size={batch_size}, seq_length={seq_length}")
                
                # Generate input data
                input_data = input_generator(batch_size, seq_length)
                
                # Measure execution time
                time_metrics = self.measure_execution_time(
                    input_data, iterations=num_iterations
                )
                
                # Measure memory usage
                memory_metrics = self.measure_memory_usage(input_data)
                
                # Store results
                result = {
                    'batch_size': batch_size,
                    'sequence_length': seq_length,
                    'avg_time': time_metrics['avg_time'],
                    'iterations_per_second': time_metrics['iterations_per_second'],
                    'memory_used_mb': memory_metrics['memory_used_mb']
                }
                
                results.append(result)
        
        # Create DataFrame
        df = pd.DataFrame(results)
        
        # Generate heatmaps
        if len(batch_sizes) > 1 and len(sequence_lengths) > 1:
            # Reshape data for heatmaps
            avg_time_pivot = df.pivot(index='batch_size', columns='sequence_length', values='avg_time')
            memory_pivot = df.pivot(index='batch_size', columns='sequence_length', values='memory_used_mb')
            
            # Create time heatmap
            fig, ax = plt.subplots(figsize=(12, 10))
            sns.heatmap(avg_time_pivot, annot=True, fmt=".4f", cmap="YlGnBu", ax=ax)
            ax.set_title("Execution Time (s)")
            ax.set_xlabel("Sequence Length")
            ax.set_ylabel("Batch Size")
            if save_dir:
                os.makedirs(save_dir, exist_ok=True)
                fig.savefig(f"{save_dir}/time_heatmap.png")
                
            # Create memory heatmap
            fig, ax = plt.subplots(figsize=(12, 10))
            sns.heatmap(memory_pivot, annot=True, fmt=".2f", cmap="YlGnBu", ax=ax)
            ax.set_title("Memory Usage (MB)")
            ax.set_xlabel("Sequence Length")
            ax.set_ylabel("Batch Size")
            
            if save_dir:
                fig.savefig(f"{save_dir}/memory_heatmap.png")
        
        # Save raw data
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            df.to_csv(f"{save_dir}/benchmark_results.csv", index=False)
        
        return df

    def trace_memory_by_layer(
        self, 
        input_data: Union[torch.Tensor, Dict[str, torch.Tensor]],
        save_path: Optional[str] = None
    ) -> Dict[str, float]:
        """
        Trace memory usage by layer in the model.
        
        Args:
            input_data: Input tensor or dictionary of tensors
            save_path: Path to save the trace
            
        Returns:
            Dictionary with memory usage by layer
        """
        # Move input data to the correct device
        if isinstance(input_data, torch.Tensor):
            input_data = input_data.to(self.device)
        elif isinstance(input_data, dict):
            input_data = {k: v.to(self.device) if torch.is_tensor(v) else v 
                        for k, v in input_data.items()}
        
        # Set model to evaluation mode
        self.model.eval()
        
        # Register hooks to track memory usage
        memory_by_layer = {}
        hooks = []
        
        def hook_fn(name):
            def _hook(module, input, output):
                # Calculate memory for input
                input_size = sum(x.numel() * x.element_size() 
                            for x in input if isinstance(x, torch.Tensor))
                
                # Calculate memory for output
                if isinstance(output, torch.Tensor):
                    output_size = output.numel() * output.element_size()
                elif isinstance(output, tuple) and isinstance(output[0], torch.Tensor):
                    output_size = sum(x.numel() * x.element_size() 
                                    for x in output if isinstance(x, torch.Tensor))
                else:
                    output_size = 0
                
                # Store memory usage
                memory_by_layer[name] = {
                    'input_memory_mb': input_size / (1024 * 1024),
                    'output_memory_mb': output_size / (1024 * 1024),
                    'total_memory_mb': (input_size + output_size) / (1024 * 1024)
                }
            
            return _hook
        
        # Register hooks for each module
        for name, module in self.model.named_modules():
            if name:  # Skip the root module
                hook = module.register_forward_hook(hook_fn(name))
                hooks.append(hook)
        
        # Forward pass
        with torch.no_grad():
            _ = self.model(input_data)
        
        # Remove hooks
        for hook in hooks:
            hook.remove()
        
        # Create DataFrame for analysis
        data = []
        for name, metrics in memory_by_layer.items():
            data.append({
                'layer': name,
                'input_memory_mb': metrics['input_memory_mb'],
                'output_memory_mb': metrics['output_memory_mb'],
                'total_memory_mb': metrics['total_memory_mb']
            })
        
        df = pd.DataFrame(data)
        df = df.sort_values('total_memory_mb', ascending=False).reset_index(drop=True)
        
        # Store results
        self.detailed_metrics['memory_by_layer'] = df
        
        # Plot the results
        fig, ax = plt.subplots(figsize=(12, min(len(df), 20)))
        df.head(15).plot.barh(x='layer', y=['input_memory_mb', 'output_memory_mb'], 
                        stacked=True, ax=ax)
        ax.set_xlabel('Memory (MB)')
        ax.set_title('Top 15 Layers by Memory Usage')
        
        # Save results if path is provided
        if save_path:
            save_dir = os.path.dirname(save_path)
            os.makedirs(save_dir, exist_ok=True)
            df.to_csv(f"{save_dir}/memory_by_layer.csv", index=False)
            fig.savefig(f"{save_dir}/memory_by_layer.png")
        
        return memory_by_layer

    def monitor_hardware_utilization(
        self,
        train_fn: Callable,
        duration: int = 60,
        interval: float = 0.5,
        save_path: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Monitor hardware utilization during model training or inference.
        
        Args:
            train_fn: Function that runs the training or inference process
            duration: Maximum duration to monitor (in seconds)
            interval: Sampling interval (in seconds)
            save_path: Path to save the results
            
        Returns:
            DataFrame with hardware utilization metrics
        """
        import threading
        import psutil
        
        # Initialize metrics storage
        metrics = []
        stop_monitoring = threading.Event()
        
        def monitor_thread():
            """Thread function to monitor hardware utilization."""
            start_time = time.time()
            
            while not stop_monitoring.is_set() and (time.time() - start_time) < duration:
                # CPU usage
                cpu_percent = psutil.cpu_percent(interval=None)
                
                # Memory usage
                memory = psutil.virtual_memory()
                memory_used = memory.used / (1024 * 1024)  # MB
                memory_percent = memory.percent
                
                # Process-specific metrics
                process = psutil.Process(os.getpid())
                process_cpu = process.cpu_percent(interval=None)
                process_memory = process.memory_info().rss / (1024 * 1024)  # MB
                
                # Apple silicon-specific metrics
                apple_silicon = torch.backends.mps.is_available()
                gpu_utilization = None
                mps_tensors = None
                # MPS doesn't provide direct GPU utilization metrics, but we can track model's device
                if apple_silicon:
                    # Check how many tensors are on MPS device as a proxy
                    mps_tensors = sum(1 for p in self.model.parameters() if p.device.type == 'mps')
                    
                # Store metrics
                metrics.append({
                    'timestamp': time.time() - start_time,
                    'cpu_percent': cpu_percent,
                    'memory_used_mb': memory_used,
                    'memory_percent': memory_percent,
                    'process_cpu_percent': process_cpu,
                    'process_memory_mb': process_memory,
                    'is_apple_silicon': apple_silicon,
                    'mps_tensors': None if not apple_silicon else mps_tensors
                })
                
                # Sleep for the specified interval
                time.sleep(interval)
        
        # Start monitoring thread
        monitor = threading.Thread(target=monitor_thread)
        monitor.start()
        
        # Run the training or inference function
        try:
            train_fn()
        finally:
            # Stop monitoring
            stop_monitoring.set()
            monitor.join()
        
        # Convert metrics to DataFrame
        df = pd.DataFrame(metrics)
        
        # Store results
        self.detailed_metrics['hardware_utilization'] = df
        
        # Plot the results
        fig, axs = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
        
        # CPU utilization
        axs[0].plot(df['timestamp'], df['cpu_percent'], label='System CPU')
        axs[0].plot(df['timestamp'], df['process_cpu_percent'], label='Process CPU')
        axs[0].set_ylabel('CPU Utilization (%)')
        axs[0].set_title('CPU Utilization During Training/Inference')
        axs[0].legend()
        axs[0].grid(True)
        
        # Memory utilization
        axs[1].plot(df['timestamp'], df['memory_used_mb'], label='System Memory')
        axs[1].plot(df['timestamp'], df['process_memory_mb'], label='Process Memory')
        axs[1].set_xlabel('Time (s)')
        axs[1].set_ylabel('Memory Usage (MB)')
        axs[1].set_title('Memory Usage During Training/Inference')
        axs[1].legend()
        axs[1].grid(True)
        
        plt.tight_layout()
        
        # Save results if path is provided
        if save_path:
            save_dir = os.path.dirname(save_path)
            os.makedirs(save_dir, exist_ok=True)
            df.to_csv(f"{save_path}_hardware_utilization.csv", index=False)
            fig.savefig(f"{save_path}_hardware_utilization.png")
        
        return df

class ModelBenchmarkSuite:
    """
    A comprehensive suite for benchmarking and profiling transformer models.
    
    This class provides methods for:
    1. Measuring performance across configurations
    2. Profiling memory and computation
    3. Comparing different model variants
    4. Visualizing results
    """
    
    def __init__(self, save_dir: str = "benchmark_results"):
        """
        Initialize the benchmark suite.
        
        Args:
            save_dir: Directory to save benchmark results
        """
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
        
        # Store all benchmark results
        self.results = {}
        
        # For comparing multiple models
        self.model_comparisons = {}
    
    def benchmark_model(
        self,
        model: torch.nn.Module,
        model_name: str,
        input_generator: Callable[[int, int], torch.Tensor],
        batch_sizes: List[int] = [1, 2, 4, 8],
        sequence_lengths: List[int] = [16, 32, 64, 128, 256],
        num_iterations: int = 5,
        profile_with_pytorch: bool = True,
        trace_memory: bool = True,
        device: Optional[torch.device] = None
    ) -> Dict[str, Any]:
        """
        Run a comprehensive benchmark on a model.
        
        Args:
            model: The PyTorch model to benchmark
            model_name: Name identifier for the model
            input_generator: Function to generate inputs
            batch_sizes: List of batch sizes to test
            sequence_lengths: List of sequence lengths to test
            num_iterations: Number of iterations for each configuration
            profile_with_pytorch: Whether to use PyTorch profiler
            trace_memory: Whether to trace memory by layer
            device: Device to run on (default: model's device)
            
        Returns:
            Dictionary with benchmark results
        """
        # Create model-specific directory
        model_dir = os.path.join(self.save_dir, model_name)
        os.makedirs(model_dir, exist_ok=True)
        
        # Create profiler
        profiler = ModelProfiler(model, device)
        
        # Run benchmark across configurations
        benchmark_df = profiler.benchmark_model(
            input_generator=input_generator,
            batch_sizes=batch_sizes,
            sequence_lengths=sequence_lengths,
            num_iterations=num_iterations,
            save_dir=model_dir
        )
        
        # Get median configuration for detailed profiling
        median_idx = len(benchmark_df) // 2
        median_config = benchmark_df.iloc[median_idx]
        median_batch = median_config['batch_size']
        median_seq = median_config['sequence_length']
        
        # Generate input for detailed profiling
        median_input = input_generator(median_batch, median_seq)
        
        # Run PyTorch profiler
        if profile_with_pytorch:
            profiler.profile_with_pytorch_profiler(
                input_data=median_input,
                use_mps=torch.backends.mps.is_available(),
                save_path=os.path.join(model_dir, "pytorch_profile.json")
            )
        
        # Trace memory by layer
        if trace_memory:
            profiler.trace_memory_by_layer(
                input_data=median_input,
                save_path=os.path.join(model_dir, "memory_trace.csv")
            )
        
        # Generate report
        report = profiler.generate_report(
            save_path=os.path.join(model_dir, "benchmark_report.md")
        )
        
        # Plot metrics
        profiler.plot_metrics(save_dir=model_dir)
        
        # Store results
        results = {
            'model_name': model_name,
            'benchmark_df': benchmark_df,
            'profiler': profiler,
            'report': report,
            'model_dir': model_dir
        }
        
        self.results[model_name] = results
        
        return results
    
    def compare_models(
        self,
        model_names: List[str] = None,
        metric: str = 'avg_time',
        save_path: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Compare performance metrics across multiple models.
        
        Args:
            model_names: List of model names to compare (None = all)
            metric: Metric to compare ('avg_time', 'memory_used_mb', etc.)
            save_path: Path to save comparison results
            
        Returns:
            DataFrame with comparison results
        """
        if model_names is None:
            model_names = list(self.results.keys())
        
        if not model_names:
            raise ValueError("No models to compare. Run benchmark_model first.")
        
        # Collect data for each model
        comparison_data = []
        
        for model_name in model_names:
            if model_name not in self.results:
                print(f"Warning: Model '{model_name}' not found in results.")
                continue
                
            # Get model's benchmark DataFrame
            df = self.results[model_name]['benchmark_df']
            
            # Add model_name column
            df = df.copy()
            df['model_name'] = model_name
            
            comparison_data.append(df)
        
        # Combine all data
        if not comparison_data:
            raise ValueError("No valid models found for comparison.")
            
        comparison_df = pd.concat(comparison_data, ignore_index=True)
        
        # Create comparison visualizations
        self._plot_model_comparison(comparison_df, metric, save_path)
        
        # Store comparison
        comparison_id = "_vs_".join(model_names)
        self.model_comparisons[comparison_id] = comparison_df
        
        return comparison_df
    
    def _plot_model_comparison(
        self,
        comparison_df: pd.DataFrame,
        metric: str = 'avg_time',
        save_path: Optional[str] = None
    ) -> None:
        """
        Plot comparison between models.
        
        Args:
            comparison_df: DataFrame with comparison data
            metric: Metric to compare
            save_path: Path to save the plot
        """
        # Check if we have multiple batch sizes and sequence lengths
        batch_sizes = comparison_df['batch_size'].unique()
        seq_lengths = comparison_df['sequence_length'].unique()
        model_names = comparison_df['model_name'].unique()
        
        # Different visualization depending on the data
        if len(seq_lengths) > 1 and len(batch_sizes) == 1:
            # Create line plot across sequence lengths
            fig, ax = plt.subplots(figsize=(12, 8))
            
            for model in model_names:
                model_data = comparison_df[comparison_df['model_name'] == model]
                ax.plot(model_data['sequence_length'], model_data[metric], 
                        'o-', label=model)
            
            ax.set_xlabel('Sequence Length')
            ax.set_ylabel(metric)
            ax.set_title(f'Model Comparison by {metric} vs Sequence Length')
            ax.legend()
            ax.grid(True)
            
        elif len(batch_sizes) > 1 and len(seq_lengths) == 1:
            # Create line plot across batch sizes
            fig, ax = plt.subplots(figsize=(12, 8))
            
            for model in model_names:
                model_data = comparison_df[comparison_df['model_name'] == model]
                ax.plot(model_data['batch_size'], model_data[metric], 
                        'o-', label=model)
            
            ax.set_xlabel('Batch Size')
            ax.set_ylabel(metric)
            ax.set_title(f'Model Comparison by {metric} vs Batch Size')
            ax.legend()
            ax.grid(True)
            
        elif len(batch_sizes) > 1 and len(seq_lengths) > 1:
            # Create heatmap of differences between models
            # This is more complex, so we'll focus on comparing just 2 models
            if len(model_names) == 2:
                model1, model2 = model_names
                
                # Pivot data for each model
                pivot1 = comparison_df[comparison_df['model_name'] == model1].pivot(
                    index='batch_size', columns='sequence_length', values=metric)
                pivot2 = comparison_df[comparison_df['model_name'] == model2].pivot(
                    index='batch_size', columns='sequence_length', values=metric)
                
                # Calculate relative difference (model2/model1 - 1)
                diff = (pivot2 / pivot1 - 1) * 100  # as percentage
                
                # Plot heatmap
                fig, ax = plt.subplots(figsize=(12, 10))
                sns.heatmap(diff, annot=True, fmt=".1f", cmap="RdBu_r", center=0, ax=ax)
                ax.set_title(f'Relative Difference in {metric} (%) ({model2} vs {model1})')
                ax.set_xlabel('Sequence Length')
                ax.set_ylabel('Batch Size')
            else:
                # For more than 2 models, use a grouped bar chart for a subset of the data
                # Take median sequence length for each batch size
                median_seq = np.median(seq_lengths).astype(int)
                subset = comparison_df[comparison_df['sequence_length'] == median_seq]
                
                fig, ax = plt.subplots(figsize=(12, 8))
                sns.barplot(x='batch_size', y=metric, hue='model_name', data=subset, ax=ax)
                ax.set_title(f'Model Comparison by {metric} (Sequence Length = {median_seq})')
                ax.set_xlabel('Batch Size')
                ax.set_ylabel(metric)
        else:
            # Simple bar chart
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.barplot(x='model_name', y=metric, data=comparison_df, ax=ax)
            ax.set_title(f'Model Comparison by {metric}')
            ax.set_xlabel('Model')
            ax.set_ylabel(metric)
        
        # Save plot if path provided
        if save_path:
            fig.savefig(save_path)
        
        plt.tight_layout()
        
    def generate_optimization_recommendations(
        self, 
        model_name: str
    ) -> str:
        """
        Generate optimization recommendations based on profiling results.
        
        Args:
            model_name: Name of the model to analyze
            
        Returns:
            String with optimization recommendations
        """
        if model_name not in self.results:
            raise ValueError(f"Model '{model_name}' not found in results.")
        
        profiler = self.results[model_name]['profiler']
        benchmark_df = self.results[model_name]['benchmark_df']
        
        recommendations = "# Optimization Recommendations\n\n"
        
        # Check if we have memory by layer data
        if 'memory_by_layer' in profiler.detailed_metrics:
            memory_df = profiler.detailed_metrics['memory_by_layer']
            top_memory_layers = memory_df.head(5)
            
            recommendations += "## Memory Optimization\n\n"
            recommendations += "Top 5 layers by memory usage:\n\n"
            
            for _, row in top_memory_layers.iterrows():
                layer = row['layer']
                memory = row['total_memory_mb']
                recommendations += f"- **{layer}**: {memory:.2f} MB\n"
            
            recommendations += "\nRecommendations:\n\n"
            
            # Check if attention layers are in top memory users
            if any('attention' in layer.lower() for layer in top_memory_layers['layer']):
                recommendations += "- Consider implementing **FlashAttention** or other memory-efficient attention variants\n"
                recommendations += "- Experiment with **attention sparsity** techniques\n"
            
            # Check if linear layers are in top memory users
            if any('linear' in layer.lower() for layer in top_memory_layers['layer']):
                recommendations += "- Try **weight factorization** techniques for large linear layers\n"
                recommendations += "- Explore **mixed precision training** to reduce memory footprint\n"
            
            # General recommendations
            recommendations += "- Implement **gradient checkpointing** to trade computation for memory\n"
            recommendations += "- Consider **model pruning** for less critical components\n"
        
        # Batch size and sequence length optimization
        if len(benchmark_df) > 1:
            recommendations += "\n## Configuration Optimization\n\n"
            
            # Find optimal batch size
            if len(benchmark_df['batch_size'].unique()) > 1:
                optimal_batch = benchmark_df.groupby('batch_size')['avg_time'].mean().idxmin()
                recommendations += f"- Optimal batch size for speed: **{optimal_batch}**\n"
            
            # Find optimal sequence length
            if len(benchmark_df['sequence_length'].unique()) > 1:
                optimal_seq = benchmark_df.groupby('sequence_length')['avg_time'].mean().idxmin()
                recommendations += f"- Optimal sequence length for speed: **{optimal_seq}**\n"
            
            # Find memory vs speed sweet spot
            try:
                # Normalize metrics between 0 and 1
                df_norm = benchmark_df.copy()
                df_norm['time_norm'] = (df_norm['avg_time'] - df_norm['avg_time'].min()) / (df_norm['avg_time'].max() - df_norm['avg_time'].min())
                df_norm['memory_norm'] = (df_norm['memory_used_mb'] - df_norm['memory_used_mb'].min()) / (df_norm['memory_used_mb'].max() - df_norm['memory_used_mb'].min())
                
                # Calculate balanced score (lower is better)
                df_norm['balanced_score'] = df_norm['time_norm'] + df_norm['memory_norm']
                best_config = df_norm.loc[df_norm['balanced_score'].idxmin()]
                
                recommendations += f"- Best balanced configuration (speed vs memory):\n"
                recommendations += f"  - Batch size: **{best_config['batch_size']}**\n"
                recommendations += f"  - Sequence length: **{best_config['sequence_length']}**\n"
            except:
                # If any error occurs, skip this recommendation
                pass
        
        # PyTorch profiler recommendations
        if 'pytorch_profiler' in profiler.detailed_metrics:
            op_table = profiler.detailed_metrics.get('pytorch_profiler_ops', '')
            
            recommendations += "\n## Computational Optimization\n\n"
            
            # Check which operations are most expensive
            if 'matmul' in op_table or 'mm' in op_table:
                recommendations += "- Use **optimized matrix multiplication libraries** (MKL, LAPACK)\n"
                recommendations += "- Consider **quantization** techniques for matrix operations\n"
            
            if 'attention' in op_table or 'self_attention' in op_table:
                recommendations += "- Implement **optimized attention algorithms** (FlashAttention)\n"
                recommendations += "- Research **sparse attention mechanisms** for your use case\n"
            
            # General recommendations
            recommendations += "- Implement **kernel fusion** for consecutive operations\n"
            recommendations += "- Consider **mixed precision inference** for faster computation\n"
            recommendations += "- Optimize **memory access patterns** for better cache utilization\n"
        
        # Save recommendations
        model_dir = self.results[model_name]['model_dir']
        with open(os.path.join(model_dir, "optimization_recommendations.md"), 'w') as f:
            f.write(recommendations)
        
        return recommendations

def extract_file_metadata(file_path=__file__):
    """
    Extract structured metadata about this module.
    
    Args:
        file_path: Path to the source file (defaults to current file)
        
    Returns:
        dict: Structured metadata about the module's purpose and components
    """
    return {
        "filename": os.path.basename(file_path),
        "module_purpose": "Provides utilities for profiling and benchmarking PyTorch models with comprehensive performance analysis",
        "key_classes": [
            {
                "name": "ModelProfiler",
                "purpose": "Utility for profiling PyTorch models with execution time, memory usage, and layer-wise analysis",
                "key_methods": [
                    {
                        "name": "__init__",
                        "signature": "__init__(self, model: torch.nn.Module, device: Optional[torch.device] = None)",
                        "brief_description": "Initialize the profiler with a model and target device"
                    },
                    {
                        "name": "measure_execution_time",
                        "signature": "measure_execution_time(self, input_data: Union[torch.Tensor, Dict[str, torch.Tensor]], iterations: int = 10, warmup: int = 2) -> Dict[str, float]",
                        "brief_description": "Measure the execution time of a forward pass"
                    },
                    {
                        "name": "measure_memory_usage",
                        "signature": "measure_memory_usage(self, input_data: Union[torch.Tensor, Dict[str, torch.Tensor]]) -> Dict[str, float]",
                        "brief_description": "Measure the memory usage during a forward pass"
                    },
                    {
                        "name": "generate_report",
                        "signature": "generate_report(self, save_path: Optional[str] = None) -> str",
                        "brief_description": "Generate a comprehensive profiling report with all metrics"
                    },
                    {
                        "name": "plot_metrics",
                        "signature": "plot_metrics(self, save_dir: Optional[str] = None) -> Dict[str, plt.Figure]",
                        "brief_description": "Create visualization plots for performance metrics"
                    },
                    {
                        "name": "profile_with_pytorch_profiler",
                        "signature": "profile_with_pytorch_profiler(self, input_data: Union[torch.Tensor, Dict[str, torch.Tensor]], use_mps: bool = True, num_steps: int = 10, warmup: int = 3, activities: Optional[List[str]] = None, record_shapes: bool = True, profile_memory: bool = True, save_path: Optional[str] = None) -> None",
                        "brief_description": "Profile the model using PyTorch's built-in profiler"
                    },
                    {
                        "name": "trace_memory_by_layer",
                        "signature": "trace_memory_by_layer(self, input_data: Union[torch.Tensor, Dict[str, torch.Tensor]], save_path: Optional[str] = None) -> Dict[str, float]",
                        "brief_description": "Trace memory usage by layer in the model"
                    },
                    {
                        "name": "benchmark_model",
                        "signature": "benchmark_model(self, input_generator: Callable[[int, int], Union[torch.Tensor, Dict[str, torch.Tensor]]], batch_sizes: List[int], sequence_lengths: List[int], num_iterations: int = 5, save_dir: Optional[str] = None) -> pd.DataFrame",
                        "brief_description": "Benchmark the model across different batch sizes and sequence lengths"
                    },
                    {
                        "name": "monitor_hardware_utilization",
                        "signature": "monitor_hardware_utilization(self, train_fn: Callable, duration: int = 60, interval: float = 0.5, save_path: Optional[str] = None) -> pd.DataFrame",
                        "brief_description": "Monitor CPU, GPU, and memory utilization during model execution"
                    }
                ],
                "inheritance": "",
                "dependencies": ["torch", "numpy", "matplotlib", "psutil", "pandas", "seaborn"]
            },
            {
                "name": "ModelBenchmarkSuite",
                "purpose": "Comprehensive suite for benchmarking and comparing multiple models with visualization",
                "key_methods": [
                    {
                        "name": "__init__",
                        "signature": "__init__(self, save_dir: str = \"benchmark_results\")",
                        "brief_description": "Initialize the benchmark suite with output directory"
                    },
                    {
                        "name": "benchmark_model",
                        "signature": "benchmark_model(self, model: torch.nn.Module, model_name: str, input_generator: Callable[[int, int], torch.Tensor], batch_sizes: List[int] = [1, 2, 4, 8], sequence_lengths: List[int] = [16, 32, 64, 128, 256], num_iterations: int = 5, profile_with_pytorch: bool = True, trace_memory: bool = True, device: Optional[torch.device] = None) -> Dict[str, Any]",
                        "brief_description": "Run a comprehensive benchmark on a model"
                    },
                    {
                        "name": "compare_models",
                        "signature": "compare_models(self, model_names: List[str] = None, metric: str = 'avg_time', save_path: Optional[str] = None) -> pd.DataFrame",
                        "brief_description": "Compare performance metrics across multiple models"
                    },
                    {
                        "name": "generate_optimization_recommendations",
                        "signature": "generate_optimization_recommendations(self, model_name: str) -> str",
                        "brief_description": "Generate optimization recommendations based on profiling results"
                    }
                ],
                "inheritance": "",
                "dependencies": ["torch", "numpy", "matplotlib", "pandas", "seaborn"]
            }
        ],
        "external_dependencies": ["torch", "numpy", "pandas", "matplotlib", "seaborn", "psutil"],
        "complexity_score": 9  # High complexity due to comprehensive profiling capabilities
    }