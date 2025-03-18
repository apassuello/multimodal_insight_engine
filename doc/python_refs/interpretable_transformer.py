import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Tuple, Optional, List, Any, Union

class InterpretableMultiHeadAttention(nn.Module):
    """
    Multi-head attention mechanism with built-in interpretability features
    that record and expose attention patterns.
    """
    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        
        # Ensure dimensions are compatible
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        
        # Linear projections
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.output_proj = nn.Linear(d_model, d_model)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
        # For interpretability
        self.record_attention = True
        
    def forward(self, 
                query: torch.Tensor, 
                key: torch.Tensor, 
                value: torch.Tensor, 
                mask: Optional[torch.Tensor] = None,
                return_attention: bool = True) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass with attention pattern recording.
        
        Args:
            query: Query tensor [batch_size, seq_len_q, d_model]
            key: Key tensor [batch_size, seq_len_k, d_model]
            value: Value tensor [batch_size, seq_len_v, d_model]
            mask: Optional mask tensor [batch_size, seq_len_q, seq_len_k]
            return_attention: Whether to return attention patterns
            
        Returns:
            Tuple of (output tensor, attention patterns)
        """
        batch_size = query.size(0)
        
        # Linear projections and reshape for multi-head attention
        q = self.q_proj(query).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(key).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(value).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Compute scaled dot-product attention
        # [batch_size, num_heads, seq_len_q, head_dim] x [batch_size, num_heads, head_dim, seq_len_k]
        # -> [batch_size, num_heads, seq_len_q, seq_len_k]
        scores = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)
        
        # Apply mask if provided
        if mask is not None:
            # Add extra dimensions to match scores shape if needed
            if mask.dim() == 3 and scores.dim() == 4:
                mask = mask.unsqueeze(1)  # [batch_size, 1, seq_len_q, seq_len_k]
            scores = scores.masked_fill(mask == 0, -1e9)
        
        # Apply softmax to get attention weights
        attention_weights = torch.softmax(scores, dim=-1)
        attention_weights_dropout = self.dropout(attention_weights)
        
        # Apply attention weights to values
        # [batch_size, num_heads, seq_len_q, seq_len_k] x [batch_size, num_heads, seq_len_v, head_dim]
        # -> [batch_size, num_heads, seq_len_q, head_dim]
        context = torch.matmul(attention_weights_dropout, v)
        
        # Reshape back to original dimensions
        # [batch_size, num_heads, seq_len_q, head_dim] -> [batch_size, seq_len_q, d_model]
        context = context.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        
        # Final linear projection
        output = self.output_proj(context)
        
        # Return output and attention patterns for interpretability
        if return_attention and self.record_attention:
            return output, attention_weights
        else:
            return output, None

class InterpretableTransformerLayer(nn.Module):
    """
    Transformer layer with built-in interpretability features that
    record activations and attention patterns for analysis.
    """
    def __init__(self, d_model: int, num_heads: int, dim_feedforward: int, dropout: float = 0.1):
        super().__init__()
        
        # Self-attention with interpretability
        self.self_attn = InterpretableMultiHeadAttention(d_model, num_heads, dropout)
        
        # Feed-forward network
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model)
        )
        
        # Layer normalization
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
        # Activation recording for interpretability
        self.attn_records = None
        self.intermediate_records = None
        self.record_activations = False
    
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass with activation recording.
        
        Args:
            x: Input tensor [batch_size, seq_len, d_model]
            mask: Optional attention mask
            
        Returns:
            Output tensor
        """
        # Self-attention with residual connection
        attn_output, attn_patterns = self.self_attn(
            self.norm1(x), self.norm1(x), self.norm1(x), mask=mask
        )
        x = x + self.dropout(attn_output)
        
        # Record attention patterns if requested
        if self.record_activations and attn_patterns is not None:
            self.attn_records = attn_patterns.detach().cpu()
        
        # Feed-forward with residual connection
        ff_output = self.feed_forward(self.norm2(x))
        output = x + self.dropout(ff_output)
        
        # Record intermediate activations if requested
        if self.record_activations:
            self.intermediate_records = output.detach().cpu()
            
        return output

class FeatureAttributionSystem:
    """
    System for attributing model outputs to input features.
    Helps explain which input elements influenced the model's decisions.
    """
    def __init__(self, model: nn.Module):
        """
        Initialize the feature attribution system.
        
        Args:
            model: Model to analyze
        """
        self.model = model
        self.gradients = {}
        self.hooks = []
        
    def register_hooks(self):
        """Register gradient hooks for all relevant layers"""
        def save_gradients(name: str):
            def hook(grad: torch.Tensor):
                self.gradients[name] = grad.detach().cpu()
            return hook
            
        # Register hooks for each layer
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.hooks.append(param.register_hook(save_gradients(name)))
                
    def integrated_gradients(self, 
                             input_ids: torch.Tensor, 
                             target_index: int, 
                             steps: int = 50) -> torch.Tensor:
        """
        Implement integrated gradients method for attribution.
        
        Args:
            input_ids: Input token IDs
            target_index: Target output index to analyze
            steps: Number of interpolation steps
            
        Returns:
            Attribution scores for each input token
        """
        # Get model's embedding layer
        embedding_layer = self._get_embedding_layer()
        if embedding_layer is None:
            raise ValueError("Could not find embedding layer in model")
            
        # Create baseline (zero embedding)
        input_embeds = embedding_layer(input_ids)
        baseline_embeds = torch.zeros_like(input_embeds)
        
        # Enable gradient recording
        input_embeds.requires_grad_(True)
        
        # Interpolate between baseline and input
        gradients = []
        for alpha in torch.linspace(0, 1, steps):
            # Create interpolated input
            interp_embeds = baseline_embeds + alpha * (input_embeds - baseline_embeds)
            interp_embeds.requires_grad_(True)
            
            # Forward pass
            output = self._forward_from_embeddings(interp_embeds)
            
            # Select target output
            if output.dim() > 2:  # For sequence outputs
                target = output[0, -1, target_index]  # Last token prediction
            else:  # For classification
                target = output[0, target_index]
                
            # Backward pass
            self.model.zero_grad()
            target.backward(retain_graph=True)
            
            # Get gradients for embeddings
            if interp_embeds.grad is not None:
                gradients.append(interp_embeds.grad.clone())
            
        # Stack and average gradients
        if not gradients:
            raise ValueError("No gradients were captured")
            
        all_gradients = torch.stack(gradients)
        avg_gradients = all_gradients.mean(dim=0)
        
        # Calculate attribution scores
        attributions = (input_embeds - baseline_embeds) * avg_gradients
        
        # Sum across embedding dimension
        token_attributions = attributions.sum(dim=-1)
        
        return token_attributions
        
    def _get_embedding_layer(self) -> Optional[nn.Module]:
        """Find the embedding layer in the model"""
        for module in self.model.modules():
            if isinstance(module, nn.Embedding):
                return module
        return None
        
    def _forward_from_embeddings(self, embeddings: torch.Tensor) -> torch.Tensor:
        """Forward pass starting from embeddings"""
        # This would be implemented specifically for your model architecture
        # Simplified placeholder that would need to be customized
        return self.model.forward_from_embeddings(embeddings)

class CircuitTracker:
    """
    Tracks information flow through the network as 'circuits',
    enabling the identification of functional pathways.
    """
    def __init__(self, model: nn.Module):
        """
        Initialize the circuit tracker.
        
        Args:
            model: Model to analyze
        """
        self.model = model
        self.activation_dict = {}
        self.hooks = []
        
    def register_activation_hooks(self):
        """Register hooks to record activations through the model"""
        def hook_fn(name: str):
            def hook(module, input, output):
                self.activation_dict[name] = output.detach().cpu()
            return hook
            
        # Register hooks for each module
        for name, module in self.model.named_modules():
            if isinstance(module, (nn.MultiheadAttention, nn.Linear)):
                self.hooks.append(module.register_forward_hook(hook_fn(name)))
                
    def clear_hooks(self):
        """Remove all registered hooks"""
        for hook in self.hooks:
            hook.remove()
        self.hooks = []
        
    def identify_circuits(self, 
                          input_data: torch.Tensor, 
                          target_neuron: Union[int, Tuple[int, ...]], 
                          threshold: float = 0.5) -> List[Dict[str, Any]]:
        """
        Identify potential circuits based on activation patterns.
        
        Args:
            input_data: Input data to analyze
            target_neuron: Target neuron(s) to trace
            threshold: Activation threshold for circuit identification
            
        Returns:
            List of identified circuits with activation information
        """
        # Register hooks and run forward pass
        self.register_activation_hooks()
        self.activation_dict = {}
        with torch.no_grad():
            output = self.model(input_data)
            
        # Get target neuron activation
        if isinstance(target_neuron, int):
            target_activation = output[0, target_neuron].item()
        else:
            # Handle multi-dimensional indices (batch, seq, neuron)
            idx = target_neuron
            target_activation = output[idx].item()
            
        # Identify circuits by tracing backward from the target
        # This is a simplified example - a real implementation would
        # involve more sophisticated analysis of activation patterns
        circuits = []
        
        # Sort modules by activation influence (simplified)
        influential_modules = []
        for name, activation in self.activation_dict.items():
            # Calculate influence score (simplified)
            if activation.dim() > 1:
                influence = activation.abs().mean().item()
            else:
                influence = activation.abs().item()
                
            if influence > threshold:
                influential_modules.append({
                    "module": name,
                    "influence": influence,
                    "activation_shape": list(activation.shape)
                })
                
        # Sort by influence
        influential_modules.sort(key=lambda x: x["influence"], reverse=True)
        
        # Create circuit paths (simplified)
        if influential_modules:
            circuits.append({
                "target_activation": target_activation,
                "influential_modules": influential_modules,
                "threshold": threshold
            })
        
        # Clean up
        self.clear_hooks()
        
        return circuits