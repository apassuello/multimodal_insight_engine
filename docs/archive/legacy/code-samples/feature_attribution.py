import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np

class FeatureAttributionTool:
    """
    Tools for attributing model outputs to input features,
    helping explain which inputs most influence model decisions.
    """
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
        
    def integrated_gradients(self, input_text, target_class, steps=50):
        """
        Implement integrated gradients attribution method.
        
        Integrated gradients measures the importance of input features
        by accumulating gradients along a straight-line path from a baseline
        to the input, which satisfies certain desirable axioms.
        """
        # Tokenize
        tokens = self.tokenizer(input_text, return_tensors='pt')
        input_ids = tokens['input_ids']
        
        # Create embedding baseline (zeros)
        embedding_layer = self.get_embedding_layer()
        baseline_embeds = torch.zeros_like(embedding_layer(input_ids))
        input_embeds = embedding_layer(input_ids)
        
        # Compute integrated gradients
        step_sizes = torch.linspace(0, 1, steps=steps)
        accumulated_grads = torch.zeros_like(input_embeds)
        
        for alpha in step_sizes:
            # Interpolate between baseline and input
            interp_embeds = baseline_embeds + alpha * (input_embeds - baseline_embeds)
            interp_embeds.requires_grad_(True)
            
            # Forward pass with the interpolated embeddings
            logits = self.forward_from_embeddings(interp_embeds)
            
            # Compute gradients
            if len(logits.shape) > 2:  # For sequence outputs
                score = logits[0, :, target_class].sum()
            else:  # For classification outputs
                score = logits[0, target_class]
                
            self.model.zero_grad()
            score.backward()
            
            # Accumulate gradients
            accumulated_grads += interp_embeds.grad
            
        # Scale gradients by input difference
        attributions = (input_embeds - baseline_embeds) * accumulated_grads / steps
        
        # Average across embedding dimension for token-level attributions
        token_attributions = attributions.sum(dim=-1)[0]
        
        # Map to tokens
        tokens_list = [self.tokenizer.decode(t) for t in input_ids[0]]
        
        return {
            "tokens": tokens_list,
            "attributions": token_attributions.detach().cpu().numpy()
        }
        
    def get_embedding_layer(self):
        """Find the embedding layer in the model"""
        for module in self.model.modules():
            if isinstance(module, nn.Embedding):
                return module
        raise ValueError("Embedding layer not found")
        
    def forward_from_embeddings(self, embeddings):
        """Forward pass starting from embeddings"""
        # This would depend on the specific model architecture
        # Simplified for illustration
        return self.model.forward_from_embeddings(embeddings)
        
    def visualize_attributions(self, attributions, figsize=(12, 4)):
        """Visualize feature attributions as a bar chart"""
        tokens = attributions["tokens"]
        scores = attributions["attributions"]
        
        # Create figure
        plt.figure(figsize=figsize)
        
        # Plot bar chart
        plt.bar(range(len(tokens)), scores)
        plt.xticks(range(len(tokens)), tokens, rotation=45, ha="right")
        
        # Add labels and styling
        plt.xlabel("Tokens")
        plt.ylabel("Attribution Score")
        plt.title("Feature Attributions")
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        
        # Handle special tokens if needed
        for i, token in enumerate(tokens):
            if token.startswith('[') and token.endswith(']'):
                plt.gca().get_xticklabels()[i].set_color('red')
                
        plt.tight_layout()
        return plt.gcf()