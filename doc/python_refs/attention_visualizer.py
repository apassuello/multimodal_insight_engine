import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

class AttentionVisualizer:
    """
    Visualize attention patterns in transformer models
    to support interpretability research and debugging.
    """
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
        self.attention_hooks = []
        self.attention_patterns = {}
        
    def register_hooks(self):
        """Register hooks to capture attention patterns"""
        def hook_fn(name):
            def hook(module, input, output):
                # For MultiHeadAttention, output may include attention weights
                if isinstance(output, tuple) and len(output) > 1:
                    self.attention_patterns[name] = output[1].detach().cpu()
            return hook
            
        # Clear existing hooks
        self.remove_hooks()
        
        # Register new hooks
        for name, module in self.model.named_modules():
            if isinstance(module, nn.MultiheadAttention):
                self.attention_hooks.append(
                    module.register_forward_hook(hook_fn(name))
                )
                
    def remove_hooks(self):
        """Remove registered hooks"""
        for hook in self.attention_hooks:
            hook.remove()
        self.attention_hooks = []
        
    def visualize_attention(self, text_input, layer=None, head=None):
        """Generate attention visualizations"""
        # Tokenize input
        tokens = self.tokenizer(text_input, return_tensors='pt')
        token_ids = tokens['input_ids'][0]
        
        # Register hooks and run model
        self.register_hooks()
        with torch.no_grad():
            self.model(**tokens)
            
        # Get token strings for visualization
        token_strings = [self.tokenizer.decode(t) for t in token_ids]
        
        # Create visualizations
        visualizations = {}
        
        for name, attention in self.attention_patterns.items():
            # Filter by layer if specified
            if layer is not None and f"layer.{layer}." not in name:
                continue
                
            # Process attention weights
            # Shape: [batch_size, num_heads, seq_len, seq_len]
            attn = attention[0]  # First batch item
            
            if head is not None:
                # Visualize specific head
                if head < attn.size(0):
                    visualizations[f"{name}_head_{head}"] = {
                        "attention": attn[head].numpy(),
                        "tokens": token_strings
                    }
            else:
                # Visualize all heads
                for h in range(attn.size(0)):
                    visualizations[f"{name}_head_{h}"] = {
                        "attention": attn[h].numpy(),
                        "tokens": token_strings
                    }
                
                # Also include head-averaged attention
                visualizations[f"{name}_avg"] = {
                    "attention": attn.mean(0).numpy(),
                    "tokens": token_strings
                }
                
        self.remove_hooks()
        return visualizations
        
    def plot_attention_heatmap(self, attention_data, title=None, figsize=(10, 8)):
        """Plot attention heatmap for visualization"""
        attention_matrix = attention_data["attention"]
        tokens = attention_data["tokens"]
        
        # Create figure
        plt.figure(figsize=figsize)
        
        # Plot heatmap
        ax = sns.heatmap(
            attention_matrix,
            xticklabels=tokens,
            yticklabels=tokens,
            cmap="viridis",
            vmin=0,
            vmax=np.max(attention_matrix),
            cbar_kws={"label": "Attention Weight"}
        )
        
        # Add labels and title
        plt.xlabel("Attended to")
        plt.ylabel("Attention from")
        if title:
            plt.title(title)
            
        # Rotate x labels for readability
        plt.xticks(rotation=45, ha="right")
        plt.yticks(rotation=0)
        
        plt.tight_layout()
        return plt.gcf()