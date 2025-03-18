import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx

class CircuitAnalyzer:
    """
    Analyze information flow paths (circuits) in transformer models
    to understand how information propagates through the network.
    """
    def __init__(self, model):
        self.model = model
        self.activation_records = {}
        self.hooks = []
        
    def register_activation_hooks(self):
        """Register hooks to record activations"""
        def hook_fn(name):
            def hook(module, input, output):
                self.activation_records[name] = output.detach().cpu()
            return hook
            
        # Clear existing hooks
        self.remove_hooks()
        
        # Register hooks for key components
        for name, module in self.model.named_modules():
            if isinstance(module, (nn.MultiheadAttention, nn.Linear, nn.LayerNorm)):
                self.hooks.append(
                    module.register_forward_hook(hook_fn(name))
                )
                
    def remove_hooks(self):
        """Remove registered hooks"""
        for hook in self.hooks:
            hook.remove()
        self.hooks = []
        
    def analyze_circuit(self, input_data, output_neuron):
        """Analyze information flow to a specific output neuron"""
        # Register hooks
        self.register_activation_hooks()
        
        # Forward pass
        with torch.no_grad():
            output = self.model(input_data)
            
        # Extract output neuron activation
        target_activation = output[0, output_neuron].item()
        
        # Perform circuit analysis
        circuit_paths = self.trace_information_flow(output_neuron)
        
        # Clean up
        self.remove_hooks()
        
        return {
            "target_activation": target_activation,
            "circuit_paths": circuit_paths,
            "activation_records": self.activation_records
        }
        
    def trace_information_flow(self, output_neuron):
        """Trace information flow to the output neuron"""
        # This would implement circuit tracing algorithms
        # Simplified for illustration
        circuit_paths = []
        
        # In practice, would analyze patterns in activation records
        # to identify paths that strongly influence the output neuron
        
        return circuit_paths
        
    def ablation_study(self, input_data, component_name, ablation_type="zero"):
        """Perform ablation study on a model component"""
        # Register forward hooks that perform ablation
        def ablation_hook(module, input, output):
            if ablation_type == "zero":
                return torch.zeros_like(output)
            elif ablation_type == "mean":
                return torch.ones_like(output) * output.mean()
            elif ablation_type == "random":
                return torch.randn_like(output) * output.std() + output.mean()
                
        # Find target module
        target_module = None
        for name, module in self.model.named_modules():
            if name == component_name:
                target_module = module
                break
                
        if target_module is None:
            raise ValueError(f"Component {component_name} not found")
            
        # Run model with ablation
        handle = target_module.register_forward_hook(ablation_hook)
        
        with torch.no_grad():
            ablated_output = self.model(input_data)
            
        handle.remove()
        
        # Run model normally for comparison
        with torch.no_grad():
            normal_output = self.model(input_data)
            
        # Compare outputs
        return {
            "normal_output": normal_output,
            "ablated_output": ablated_output,
            "difference": (normal_output - ablated_output).abs().mean().item()
        }
        
    def visualize_circuit(self, circuit_paths, figsize=(12, 10)):
        """Visualize the identified circuit pathways as a graph"""
        G = nx.DiGraph()
        
        # Add nodes and edges
        for path in circuit_paths:
            for i in range(len(path) - 1):
                G.add_edge(path[i], path[i+1])
        
        # Draw the graph
        plt.figure(figsize=figsize)
        pos = nx.spring_layout(G)
        nx.draw(G, pos, with_labels=True, node_size=1000, node_color="skyblue", 
                font_size=10, font_weight="bold", arrows=True, 
                arrowsize=15, width=2.0)
        
        plt.title("Information Flow Circuit")
        plt.tight_layout()
        return plt.gcf()