def profile_memory_efficiency(self, 
                           input_sizes: List[Tuple[int, ...]], 
                           num_runs: int = 3) -> Dict[str, Any]:
    """
    Profile memory efficiency across different input sizes.
    
    Args:
        input_sizes: List of input tensor shapes to test
        num_runs: Number of profiling runs for each size
        
    Returns:
        Dictionary with memory efficiency results
    """
    results = {
        "device": str(self.device),
        "input_sizes": input_sizes,
        "memory_metrics": {},
    }
    
    # Check if CUDA is available for memory profiling
    if self.device.type != "cuda":
        print("Memory efficiency profiling requires CUDA. Skipping.")
        return results
    
    print(f"Profiling memory efficiency across {len(input_sizes)} input sizes...")
    
    for input_shape in input_sizes:
        print(f"Profiling input shape: {input_shape}")
        
        # Generate random input data
        input_data = torch.rand(*input_shape, device=self.device)
        
        # Profile memory usage
        size_results = {
            "theoretical_activations_mb": 0,
            "peak_memory_mb": [],
            "memory_efficiency": [],
        }
        
        # Calculate theoretical activation memory
        # This is a rough estimate based on input and output shapes
        # A more accurate estimate would require tracing the model
        with torch.no_grad():
            sample_output = self.model(input_data)
            
            # Simple estimate: input + output tensors
            input_size_mb = input_data.nelement() * input_data.element_size() / (1024 * 1024)
            
            if isinstance(sample_output, torch.Tensor):
                output_size_mb = sample_output.nelement() * sample_output.element_size() / (1024 * 1024)
            else:
                # Handle case where output is not a tensor
                output_size_mb = 0
                
            # Very rough estimate - in practice, there are many intermediate activations
            size_results["theoretical_activations_mb"] = input_size_mb + output_size_mb
        
        # Perform memory profiling runs
        for i in range(num_runs):
            # Clear cache
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            
            # Record initial memory
            initial_memory = torch.cuda.memory_allocated() / (1024 * 1024)
            
            # Forward pass
            with torch.no_grad():
                _ = self.model(input_data)
                
            # Record peak memory
            peak_memory = torch.cuda.max_memory_allocated() / (1024 * 1024)
            used_memory = peak_memory - initial_memory
            
            # Calculate a simple efficiency metric
            # (theoretical / actual) - lower is better
            if used_memory > 0:
                efficiency = size_results["theoretical_activations_mb"] / used_memory
            else:
                efficiency = 0
                
            size_results["peak_memory_mb"].append(used_memory)
            size_results["memory_efficiency"].append(efficiency)
        
        # Calculate statistics
        size_results["mean_peak_memory_mb"] = np.mean(size_results["peak_memory_mb"])
        size_results["mean_efficiency"] = np.mean(size_results["memory_efficiency"])
        
        # Store results for this input size
        results["memory_metrics"][str(input_shape)] = size_results
    
    # Visualize memory efficiency
    self._visualize_memory_efficiency(results)
    
    # Save results
    self.profiling_results["memory_efficiency"] = results
    
    return results

def _visualize_memory_efficiency(self, results: Dict[str, Any]) -> None:
    """
    Create visualizations for memory efficiency profiling.
    
    Args:
        results: Memory efficiency profiling results
    """
    # Extract data for plotting
    input_labels = []
    peak_memories = []
    efficiencies = []
    theoretical_mems = []
    
    for input_shape, metrics in results["memory_metrics"].items():
        # Create a label based on input shape
        # Show total elements and dimensions
        shape = eval(input_shape)  # Convert string back to tuple
        total_elements = np.prod(shape)
        label = f"{input_shape}\n({total_elements:,} elements)"
        
        input_labels.append(label)
        peak_memories.append(metrics["mean_peak_memory_mb"])
        efficiencies.append(metrics["mean_efficiency"])
        theoretical_mems.append(metrics["theoretical_activations_mb"])
    
    # Create a figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Plot memory usage
    x = np.arange(len(input_labels))
    width = 0.35
    
    ax1.bar(x - width/2, peak_memories, width, label='Actual Memory')
    ax1.bar(x + width/2, theoretical_mems, width, label='Theoretical Minimum')
    
    ax1.set_xlabel('Input Shape')
    ax1.set_ylabel('Memory Usage (MB)')
    ax1.set_title('Memory Usage by Input Size')
    ax1.set_xticks(x)
    ax1.set_xticklabels(input_labels)
    ax1.legend()
    
    # Plot memory efficiency
    ax2.bar(x, efficiencies, width)
    ax2.set_xlabel('Input Shape')
    ax2.set_ylabel('Efficiency Ratio (theoretical/actual)')
    ax2.set_title('Memory Efficiency by Input Size')
    ax2.set_xticks(x)
    ax2.set_xticklabels(input_labels)
    
    # Add a line at 1.0 for reference (perfect efficiency)
    ax2.axhline(y=1.0, color='r', linestyle='--', alpha=0.5, label='Ideal')
    ax2.legend()
    
    plt.tight_layout()
    
    # Save the figure
    plt.savefig(os.path.join(self.output_dir, "memory_efficiency.png"), dpi=150, bbox_inches="tight")
    plt.close()

def compare_optimizations(self, 
                        optimized_models: Dict[str, torch.nn.Module],
                        input_data: torch.Tensor,
                        num_runs: int = 10) -> Dict[str, Any]:
    """
    Compare performance of different optimized versions of the model.
    
    Args:
        optimized_models: Dictionary mapping optimization names to model instances
        input_data: Input data for testing
        num_runs: Number of profiling runs for each model
        
    Returns:
        Dictionary with optimization comparison results
    """
    # Make sure base model is included
    if self.model is not None:
        all_models = {"baseline": self.model, **optimized_models}
    else:
        all_models = optimized_models
    
    results = {
        "device": str(self.device),
        "input_shape": list(input_data.shape),
        "optimization_metrics": {},
    }
    
    print(f"Comparing {len(all_models)} model optimizations...")
    
    # Profile each model
    for name, model in all_models.items():
        print(f"Profiling optimization: {name}")
        
        # Move model to device
        model.to(self.device)
        model.eval()
        
        # Move input to device
        data = input_data.to(self.device)
        
        # Set up metrics
        opt_results = {
            "parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
            "model_size_mb": sum(p.numel() * p.element_size() for p in model.parameters()) / (1024 * 1024),
            "run_times_ms": [],
            "peak_memory_mb": [] if self.device.type == "cuda" else None,
        }
        
        # Warm-up run
        with torch.no_grad():
            _ = model(data)
        
        # CUDA profiling
        if self.device.type == "cuda":
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            
            torch.cuda.synchronize()
            
            with torch.no_grad():
                for _ in range(num_runs):
                    # Clear cache and record initial memory
                    torch.cuda.empty_cache()
                    torch.cuda.reset_peak_memory_stats()
                    initial_memory = torch.cuda.memory_allocated() / (1024 * 1024)
                    
                    # Time the forward pass
                    start_event.record()
                    output = model(data)
                    end_event.record()
                    
                    torch.cuda.synchronize()
                    
                    # Record metrics
                    elapsed_time_ms = start_event.elapsed_time(end_event)
                    opt_results["run_times_ms"].append(elapsed_time_ms)
                    
                    # Calculate peak memory used
                    peak_memory = torch.cuda.max_memory_allocated() / (1024 * 1024)
                    opt_results["peak_memory_mb"].append(peak_memory - initial_memory)
        else:
            # CPU profiling
            with torch.no_grad():
                for _ in range(num_runs):
                    # Time the forward pass
                    start_time = time.time()
                    output = model(data)
                    end_time = time.time()
                    
                    # Record metrics
                    elapsed_time_ms = (end_time - start_time) * 1000
                    opt_results["run_times_ms"].append(elapsed_time_ms)
        
        # Calculate statistics
        opt_results["mean_time_ms"] = np.mean(opt_results["run_times_ms"])
        opt_results["std_time_ms"] = np.std(opt_results["run_times_ms"])
        
        if opt_results["peak_memory_mb"]:
            opt_results["mean_peak_memory_mb"] = np.mean(opt_results["peak_memory_mb"])
        
        # Store results for this optimization
        results["optimization_metrics"][name] = opt_results
    
    # Calculate relative performance
    if "baseline" in results["optimization_metrics"]:
        baseline_time = results["optimization_metrics"]["baseline"]["mean_time_ms"]
        for name, metrics in results["optimization_metrics"].items():
            if name != "baseline":
                metrics["speedup_vs_baseline"] = baseline_time / metrics["mean_time_ms"]
                
        if self.device.type == "cuda" and "mean_peak_memory_mb" in results["optimization_metrics"]["baseline"]:
            baseline_memory = results["optimization_metrics"]["baseline"]["mean_peak_memory_mb"]
            for name, metrics in results["optimization_metrics"].items():
                if name != "baseline" and "mean_peak_memory_mb" in metrics:
                    metrics["memory_reduction_vs_baseline"] = baseline_memory / metrics["mean_peak_memory_mb"]
    
    # Visualize optimization comparison
    self._visualize_optimization_comparison(results)
    
    # Save results
    self.profiling_results["optimization_comparison"] = results
    
    return results

def _visualize_optimization_comparison(self, results: Dict[str, Any]) -> None:
    """
    Create visualizations for optimization comparison.
    
    Args:
        results: Optimization comparison results
    """
    # Extract data for plotting
    optimization_names = list(results["optimization_metrics"].keys())
    mean_times = [results["optimization_metrics"][name]["mean_time_ms"] for name in optimization_names]
    model_sizes = [results["optimization_metrics"][name]["model_size_mb"] for name in optimization_names]
    
    # Check if we have memory data
    has_memory = all("mean_peak_memory_mb" in results["optimization_metrics"][name] for name in optimization_names)
    if has_memory:
        mean_memories = [results["optimization_metrics"][name]["mean_peak_memory_mb"] for name in optimization_names]
    
    # Create figure with appropriate number of subplots
    if has_memory:
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))
    else:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Plot inference time
    x = np.arange(len(optimization_names))
    width = 0.6
    
    bars1 = ax1.bar(x, mean_times, width)
    ax1.set_xlabel('Optimization')
    ax1.set_ylabel('Mean Inference Time (ms)')
    ax1.set_title('Inference Time by Optimization')
    ax1.set_xticks(x)
    ax1.set_xticklabels(optimization_names, rotation=45, ha="right")
    
    # Add time values on bars
    for bar in bars1:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                f'{height:.2f}',
                ha='center', va='bottom', rotation=0)
    
    # Plot model size
    bars2 = ax2.bar(x, model_sizes, width)
    ax2.set_xlabel('Optimization')
    ax2.set_ylabel('Model Size (MB)')
    ax2.set_title('Model Size by Optimization')
    ax2.set_xticks(x)
    ax2.set_xticklabels(optimization_names, rotation=45, ha="right")
    
    # Add size values on bars
    for bar in bars2:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                f'{height:.2f}',
                ha='center', va='bottom', rotation=0)
    
    # Plot memory usage if available
    if has_memory:
        bars3 = ax3.bar(x, mean_memories, width)
        ax3.set_xlabel('Optimization')
        ax3.set_ylabel('Peak Memory Usage (MB)')
        ax3.set_title('Memory Usage by Optimization')
        ax3.set_xticks(x)
        ax3.set_xticklabels(optimization_names, rotation=45, ha="right")
        
        # Add memory values on bars
        for bar in bars3:
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                    f'{height:.2f}',
                    ha='center', va='bottom', rotation=0)
    
    plt.tight_layout()
    
    # Save the figure
    plt.savefig(os.path.join(self.output_dir, "optimization_comparison.png"), dpi=150, bbox_inches="tight")
    plt.close()
    
    # If we have a baseline, create speedup and efficiency plots
    if "baseline" in results["optimization_metrics"] and len(optimization_names) > 1:
        # Create a new figure for relative metrics
        if has_memory:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        else:
            fig, ax1 = plt.subplots(1, 1, figsize=(8, 6))
            ax2 = None
        
        # Only include non-baseline optimizations
        rel_names = [name for name in optimization_names if name != "baseline"]
        rel_x = np.arange(len(rel_names))
        
        # Get speedup values
        speedups = [results["optimization_metrics"][name].get("speedup_vs_baseline", 1.0) for name in rel_names]
        
        # Plot speedups
        bars1 = ax1.bar(rel_x, speedups, width)
        ax1.set_xlabel('Optimization')
        ax1.set_ylabel('Speedup vs Baseline')
        ax1.set_title('Performance Improvement vs Baseline')
        ax1.set_xticks(rel_x)
        ax1.set_xticklabels(rel_names, rotation=45, ha="right")
        
        # Add reference line at 1.0 (baseline performance)
        ax1.axhline(y=1.0, color='r', linestyle='--', alpha=0.7)
        
        # Add speedup values on bars
        for bar in bars1:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 0.05,
                    f'{height:.2f}x',
                    ha='center', va='bottom', rotation=0)
        
        # Plot memory reduction if available
        if has_memory and ax2 is not None:
            memory_reductions = [results["optimization_metrics"][name].get("memory_reduction_vs_baseline", 1.0) for name in rel_names]
            
            bars2 = ax2.bar(rel_x, memory_reductions, width)
            ax2.set_xlabel('Optimization')
            ax2.set_ylabel('Memory Reduction vs Baseline')
            ax2.set_title('Memory Efficiency vs Baseline')
            ax2.set_xticks(rel_x)
            ax2.set_xticklabels(rel_names, rotation=45, ha="right")
            
            # Add reference line at 1.0 (baseline memory)
            ax2.axhline(y=1.0, color='r', linestyle='--', alpha=0.7)
            
            # Add reduction values on bars
            for bar in bars2:
                height = bar.get_height()
                ax2.text(bar.get_x() + bar.get_width()/2., height + 0.05,
                        f'{height:.2f}x',
                        ha='center', va='bottom', rotation=0)
        
        plt.tight_layout()
        
        # Save the figure
        plt.savefig(os.path.join(self.output_dir, "optimization_relative_performance.png"), dpi=150, bbox_inches="tight")
        plt.close()

def profile_inference_scaling(self, 
                           model_family: Dict[str, torch.nn.Module],
                           input_data: torch.Tensor,
                           num_runs: int = 5) -> Dict[str, Any]:
    """
    Profile scaling behavior across different model sizes in a model family.
    
    Args:
        model_family: Dictionary mapping model size names to model instances
        input_data: Input data for testing
        num_runs: Number of profiling runs for each model
        
    Returns:
        Dictionary with scaling analysis results
    """
    results = {
        "device": str(self.device),
        "input_shape": list(input_data.shape),
        "model_metrics": {},
    }
    
    print(f"Profiling scaling across {len(model_family)} model variants...")
    
    # Get parameter counts for each model
    param_counts = {}
    for name, model in model_family.items():
        param_counts[name] = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    # Sort models by parameter count
    sorted_models = sorted(model_family.items(), key=lambda x: param_counts[x[0]])
    
    # Profile each model
    for name, model in sorted_models:
        print(f"Profiling model: {name} ({param_counts[name]:,} parameters)")
        
        # Move model to device
        model.to(self.device)
        model.eval()
        
        # Move input to device
        data = input_data.to(self.device)
        
        # Set up metrics
        model_results = {
            "parameters": param_counts[name],
            "model_size_mb": sum(p.numel() * p.element_size() for p in model.parameters()) / (1024 * 1024),
            "run_times_ms": [],
            "peak_memory_mb": [] if self.device.type == "cuda" else None,
        }
        
        # Warm-up run
        with torch.no_grad():
            _ = model(data)
        
        # CUDA profiling
        if self.device.type == "cuda":
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            
            torch.cuda.synchronize()
            
            with torch.no_grad():
                for _ in range(num_runs):
                    # Clear cache and record initial memory
                    torch.cuda.empty_cache()
                    torch.cuda.reset_peak_memory_stats()
                    initial_memory = torch.cuda.memory_allocated() / (1024 * 1024)
                    
                    # Time the forward pass
                    start_event.record()
                    output = model(data)
                    end_event.record()
                    
                    torch.cuda.synchronize()
                    
                    # Record metrics
                    elapsed_time_ms = start_event.elapsed_time(end_event)
                    model_results["run_times_ms"].append(elapsed_time_ms)
                    
                    # Calculate peak memory used
                    peak_memory = torch.cuda.max_memory_allocated() / (1024 * 1024)
                    model_results["peak_memory_mb"].append(peak_memory - initial_memory)
        else:
            # CPU profiling
            with torch.no_grad():
                for _ in range(num_runs):
                    # Time the forward pass
                    start_time = time.time()
                    output = model(data)
                    end_time = time.time()
                    
                    # Record metrics
                    elapsed_time_ms = (end_time - start_time) * 1000
                    model_results["run_times_ms"].append(elapsed_time_ms)
        
        # Calculate statistics
        model_results["mean_time_ms"] = np.mean(model_results["run_times_ms"])
        model_results["std_time_ms"] = np.std(model_results["run_times_ms"])
        
        if model_results["peak_memory_mb"]:
            model_results["mean_peak_memory_mb"] = np.mean(model_results["peak_memory_mb"])
        
        # Store results for this model
        results["model_metrics"][name] = model_results
    
    # Calculate scaling factors
    smallest_model = sorted_models[0][0]
    smallest_params = param_counts[smallest_model]
    smallest_time = results["model_metrics"][smallest_model]["mean_time_ms"]
    
    for name, metrics in results["model_metrics"].items():
        # Parameter scale factor
        metrics["param_scale"] = metrics["parameters"] / smallest_params
        
        # Time scale factor
        metrics["time_scale"] = metrics["mean_time_ms"] / smallest_time
        
        # Calculate scaling efficiency (ideally would be 1.0 for linear scaling)
        metrics["scaling_efficiency"] = metrics["time_scale"] / metrics["param_scale"]
    
    # Visualize scaling behavior
    self._visualize_scaling_behavior(results, param_counts)
    
    # Save results
    self.profiling_results["scaling_behavior"] = results
    
    return results

def _visualize_scaling_behavior(self, results: Dict[str, Any], param_counts: Dict[str, int]) -> None:
    """
    Create visualizations for model scaling behavior.
    
    Args:
        results: Scaling analysis results
        param_counts: Parameter counts for each model
    """
    # Sort model names by parameter count
    model_names = sorted(results["model_metrics"].keys(), key=lambda x: param_counts[x])
    
    # Extract parameter counts and inference times
    params = [results["model_metrics"][name]["parameters"] for name in model_names]
    times = [results["model_metrics"][name]["mean_time_ms"] for name in model_names]
    
    # Check if we have memory data
    has_memory = all("mean_peak_memory_mb" in results["model_metrics"][name] for name in model_names)
    if has_memory:
        memories = [results["model_metrics"][name]["mean_peak_memory_mb"] for name in model_names]
    
    # Create figure with appropriate number of subplots
    if has_memory:
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))
    else:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Plot inference time vs parameters
    ax1.plot(params, times, 'o-', linewidth=2, markersize=8)
    ax1.set_xlabel('Parameters')
    ax1.set_ylabel('Inference Time (ms)')
    ax1.set_title('Inference Time vs Model Size')
    ax1.set_xscale('log')
    ax1.set_yscale('log')
    ax1.grid(True, which="both", ls="--", alpha=0.7)
    
    # Add model names as annotations
    for i, name in enumerate(model_names):
        ax1.annotate(name, (params[i], times[i]), 
                    textcoords="offset points", 
                    xytext=(0, 10), 
                    ha='center')
    
    # Plot time scale vs parameter scale
    param_scales = [results["model_metrics"][name]["param_scale"] for name in model_names]
    time_scales = [results["model_metrics"][name]["time_scale"] for name in model_names]
    
    ax2.plot(param_scales, time_scales, 'o-', linewidth=2, markersize=8, color="green")
    
    # Add reference line for linear scaling
    max_scale = max(param_scales)
    ax2.plot([1, max_scale], [1, max_scale], '--', color="red", alpha=0.7, label="Linear Scaling")
    
    ax2.set_xlabel('Parameter Scale Factor')
    ax2.set_ylabel('Time Scale Factor')
    ax2.set_title('Scaling Behavior (relative to smallest model)')
    ax2.set_xscale('log')
    ax2.set_yscale('log')
    ax2.grid(True, which="both", ls="--", alpha=0.7)
    ax2.legend()
    
    # Add model names as annotations
    for i, name in enumerate(model_names):
        ax2.annotate(name, (param_scales[i], time_scales[i]), 
                    textcoords="offset points", 
                    xytext=(0, 10), 
                    ha='center')
    
    # Plot memory usage if available
    if has_memory:
        ax3.plot(params, memories, 'o-', linewidth=2, markersize=8, color="purple")
        ax3.set_xlabel('Parameters')
        ax3.set_ylabel('Peak Memory Usage (MB)')
        ax3.set_title('Memory Usage vs Model Size')
        ax3.set_xscale('log')
        ax3.set_yscale('log')
        ax3.grid(True, which="both", ls="--", alpha=0.7)
        
        # Add model names as annotations
        for i, name in enumerate(model_names):
            ax3.annotate(name, (params[i], memories[i]), 
                        textcoords="offset points", 
                        xytext=(0, 10), 
                        ha='center')
    
    plt.tight_layout()
    
    # Save the figure
    plt.savefig(os.path.join(self.output_dir, "scaling_behavior.png"), dpi=150, bbox_inches="tight")
    plt.close()
    
    # Create a scaling efficiency figure
    plt.figure(figsize=(10, 6))
    
    efficiencies = [results["model_metrics"][name]["scaling_efficiency"] for name in model_names]
    
    plt.plot(params, efficiencies, 'o-', linewidth=2, markersize=8, color="blue")
    plt.xlabel('Parameters')
    plt.ylabel('Scaling Efficiency (Time Scale / Param Scale)')
    plt.title('Computational Efficiency vs Model Size')
    plt.xscale('log')
    plt.grid(True, which="both", ls="--", alpha=0.7)
    
    # Add reference line for perfect scaling efficiency
    plt.axhline(y=1.0, color='r', linestyle='--', alpha=0.7, label="Perfect Scaling")
    
    # Add model names as annotations
    for i, name in enumerate(model_names):
        plt.annotate(name, (params[i], efficiencies[i]), 
                    textcoords="offset points", 
                    xytext=(0, 10), 
                    ha='center')
    
    plt.legend()
    plt.tight_layout()
    
    # Save the figure
    plt.savefig(os.path.join(self.output_dir, "scaling_efficiency.png"), dpi=150, bbox_inches="tight")
    plt.close()