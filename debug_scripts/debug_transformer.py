#!/usr/bin/env python3
# debug_transformer.py
# Module for debugging transformer model training

import torch
import numpy as np
from typing import Dict, List, Any, Tuple, Optional
import math
import os

class TransformerDebugger:
    """Helper class to debug transformer model training."""
    
    def __init__(
        self, 
        log_dir: str = "logs",
        print_every: int = 50,
        num_examples: int = 5,
        num_top_tokens: int = 5,
        src_tokenizer = None,
        tgt_tokenizer = None
    ):
        """
        Initialize the transformer debugger.
        
        Args:
            log_dir: Directory to save log files
            print_every: How often to print debug info (in steps)
            num_examples: Number of examples to debug
            num_top_tokens: Number of top predicted tokens to show
            src_tokenizer: Source tokenizer for decoding tokens
            tgt_tokenizer: Target tokenizer for decoding tokens
        """
        self.log_dir = log_dir
        self.print_every = print_every
        self.num_examples = num_examples
        self.num_top_tokens = num_top_tokens
        self.src_tokenizer = src_tokenizer
        self.tgt_tokenizer = tgt_tokenizer
        self.step = 0
        
        # Create log directory if it doesn't exist
        os.makedirs(log_dir, exist_ok=True)
        
        # Open the log file
        self.log_file = open(f"{log_dir}/transformer_debug.log", "w")
        self.log_file.write("===== Transformer Debugging Log =====\n\n")
        self.log_file.flush()
    
    def __del__(self):
        """Close the log file when the debugger is destroyed."""
        if hasattr(self, 'log_file') and self.log_file:
            self.log_file.close()
    
    def log(self, message: str):
        """Write a message to the log file and print it."""
        print(message)
        self.log_file.write(message + "\n")
        self.log_file.flush()
    
    def debug_model_outputs(
        self, 
        model: torch.nn.Module, 
        batch: Dict[str, torch.Tensor],
        outputs: torch.Tensor,
        loss: torch.Tensor,
        step: Optional[int] = None
    ):
        """
        Debug model outputs and predictions.
        
        Args:
            model: The transformer model
            batch: The input batch (with src, tgt, src_mask, tgt_mask)
            outputs: Model outputs (logits)
            loss: The loss value
            step: Current training step (if None, use internal counter)
        """
        if step is not None:
            self.step = step
        else:
            self.step += 1
            
        # Only debug at specified intervals
        if self.step % self.print_every != 0:
            return
            
        # Extract batch components
        src = batch.get("src")
        tgt = batch.get("tgt")
        tgt_y = batch.get("tgt_y", tgt[:, 1:] if tgt is not None else None)  # Shifted target for next token prediction
        
        if src is None or tgt is None or tgt_y is None:
            self.log("Warning: Missing required batch components for debugging")
            return
            
        # Get batch size and limit examples to actual batch size
        batch_size = src.size(0)
        num_examples = min(self.num_examples, batch_size)
        
        # Start logging
        self.log(f"\n===== Debug Step {self.step} =====")
        self.log(f"Loss: {loss.item():.4f}, Perplexity: {math.exp(loss.item()):.2f}")
        
        # Sample some examples from the batch
        for i in range(num_examples):
            # Get source and target sequences
            src_seq = src[i].cpu()
            tgt_seq = tgt[i].cpu()
            tgt_y_seq = tgt_y[i].cpu()
            
            # Get model predictions for this example
            if len(outputs.shape) == 3:  # [batch_size, seq_len, vocab_size]
                logits = outputs[i].detach().cpu()
                
                # Apply softmax to get probabilities
                probs = torch.nn.functional.softmax(logits, dim=-1)
                
                # Get the top predicted tokens and their probabilities
                top_probs, top_indices = torch.topk(probs, k=self.num_top_tokens, dim=-1)
                
                # Log the example
                self.log(f"\nExample {i+1}:")
                
                # Decode source and target if tokenizers are available
                if self.src_tokenizer:
                    src_text = self.src_tokenizer.decode(src_seq.tolist())
                    self.log(f"Source: {src_text}")
                else:
                    self.log(f"Source indices: {src_seq.tolist()}")
                    
                if self.tgt_tokenizer:
                    tgt_text = self.tgt_tokenizer.decode(tgt_seq.tolist())
                    self.log(f"Target: {tgt_text}")
                else:
                    self.log(f"Target indices: {tgt_seq.tolist()}")
                
                # Log predictions for a few positions
                num_positions = min(5, len(tgt_y_seq))
                for pos in range(num_positions):
                    self.log(f"  Position {pos}:")
                    
                    # Get the true target token at this position
                    true_token_idx = tgt_y_seq[pos].item()
                    if self.tgt_tokenizer:
                        true_token = self.tgt_tokenizer.decode([true_token_idx])
                        self.log(f"    True token: {true_token} (id={true_token_idx})")
                    else:
                        self.log(f"    True token id: {true_token_idx}")
                    
                    # Get the probability of the true token
                    true_prob = probs[pos, true_token_idx].item()
                    self.log(f"    True token probability: {true_prob:.6f}")
                    
                    # Print top predicted tokens
                    self.log(f"    Top {self.num_top_tokens} predictions:")
                    for j in range(self.num_top_tokens):
                        token_idx = top_indices[pos, j].item()
                        token_prob = top_probs[pos, j].item()
                        
                        if self.tgt_tokenizer:
                            token = self.tgt_tokenizer.decode([token_idx])
                            self.log(f"      {j+1}. {token} (id={token_idx}): {token_prob:.6f}")
                        else:
                            self.log(f"      {j+1}. Token id {token_idx}: {token_prob:.6f}")
            else:
                self.log(f"Warning: Unexpected output shape: {outputs.shape}")
    
    def check_regularization(self, model: torch.nn.Module, optimizer: torch.optim.Optimizer):
        """
        Check for excessive regularization in the model and optimizer.
        
        Args:
            model: The transformer model
            optimizer: The optimizer
        """
        self.log("\n===== Regularization Analysis =====")
        
        # Check for weight decay in optimizer
        weight_decay = None
        if isinstance(optimizer, torch.optim.AdamW):
            weight_decay = optimizer.param_groups[0].get('weight_decay', 0)
            self.log(f"AdamW weight decay: {weight_decay}")
        elif isinstance(optimizer, torch.optim.Adam):
            weight_decay = optimizer.param_groups[0].get('weight_decay', 0)
            self.log(f"Adam weight decay: {weight_decay}")
        else:
            self.log(f"Optimizer type: {type(optimizer).__name__}")
        
        if weight_decay and weight_decay > 0.01:
            self.log("WARNING: Weight decay may be too high!")
        
        # Check for dropout in the model
        dropout_values = []
        for name, module in model.named_modules():
            if isinstance(module, torch.nn.Dropout):
                dropout_values.append((name, module.p))
        
        if dropout_values:
            self.log("\nDropout layers:")
            for name, p in dropout_values:
                self.log(f"  {name}: p={p}")
                if p > 0.3:
                    self.log(f"  WARNING: Dropout rate for {name} may be too high!")
        else:
            self.log("No dropout layers found in the model.")
        
        # Check for gradient norms
        total_norm = 0
        for p in model.parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
        total_norm = total_norm ** 0.5
        
        self.log(f"\nTotal gradient norm: {total_norm}")
        if total_norm > 10.0:
            self.log("WARNING: Gradient norm is very high!")
        elif total_norm < 0.01:
            self.log("WARNING: Gradient norm is very low!")
        
        # Check weight magnitudes for specific layers
        self.log("\nWeight magnitudes:")
        tiny_weights_count = 0
        large_weights_count = 0
        
        for name, param in model.named_parameters():
            if 'weight' in name:
                weight_mean = param.data.abs().mean().item()
                weight_std = param.data.std().item()
                
                if 'layernorm' in name.lower() or 'layer_norm' in name.lower():
                    # LayerNorm weights are typically around 1.0
                    pass
                elif weight_mean < 0.01:
                    tiny_weights_count += 1
                    self.log(f"  {name}: mean={weight_mean:.6f}, std={weight_std:.6f} - TINY!")
                elif weight_mean > 1.0:
                    large_weights_count += 1
                    self.log(f"  {name}: mean={weight_mean:.6f}, std={weight_std:.6f} - LARGE!")
                    
        self.log(f"\nFound {tiny_weights_count} layers with tiny weights")
        self.log(f"Found {large_weights_count} layers with large weights")
        
        # Final recommendations
        self.log("\nRecommendations:")
        if weight_decay and weight_decay > 0.01:
            self.log("- Reduce weight decay (try 0.01 or lower)")
        
        high_dropout = any(p > 0.3 for _, p in dropout_values)
        if high_dropout:
            self.log("- Reduce dropout rates (try 0.1 or 0.2)")
            
        if tiny_weights_count > 3:
            self.log("- Check initialization, weights might be too small")
            
        if large_weights_count > 3:
            self.log("- Check for exploding gradients, weights might be too large")
            
        if total_norm > 10.0:
            self.log("- Consider gradient clipping")
        elif total_norm < 0.01:
            self.log("- Increase learning rate or check for vanishing gradients")


def attach_debugger_to_trainer(trainer, src_tokenizer=None, tgt_tokenizer=None, print_every=50):
    """
    Helper function to attach a debugger to a transformer trainer.
    
    Args:
        trainer: The TransformerTrainer instance
        src_tokenizer: Source tokenizer
        tgt_tokenizer: Target tokenizer
        print_every: How often to print debug info
        
    Returns:
        The debugger instance
    """
    debugger = TransformerDebugger(
        src_tokenizer=src_tokenizer,
        tgt_tokenizer=tgt_tokenizer,
        print_every=print_every
    )
    
    # Check available methods in the trainer
    if hasattr(trainer, '_training_step'):
        # Store original methods to patch
        original_method = trainer._training_step
        
        # Define a patched method
        def patched_training_step(batch):
            # Call the original method
            loss, outputs = original_method(batch)
            
            # Debug the outputs
            debugger.debug_model_outputs(trainer.model, batch, outputs, loss)
            
            # Every 200 steps, check regularization
            if debugger.step % 200 == 0:
                debugger.check_regularization(trainer.model, trainer.optimizer)
                
            return loss, outputs
        
        # Patch the method
        trainer._training_step = patched_training_step
    
    elif hasattr(trainer, 'train_step'):
        # Store original methods to patch
        original_method = trainer.train_step
        
        # Define a patched method
        def patched_train_step(batch):
            # Call the original method
            result = original_method(batch)
            
            # Extract loss and outputs from result
            if isinstance(result, tuple) and len(result) >= 2:
                loss, outputs = result[0], result[1]
            else:
                loss, outputs = result, None
                
            # Debug the outputs if available
            if outputs is not None:
                debugger.debug_model_outputs(trainer.model, batch, outputs, loss)
            
            # Every 200 steps, check regularization
            if debugger.step % 200 == 0:
                debugger.check_regularization(trainer.model, trainer.optimizer)
                
            return result
        
        # Patch the method
        trainer.train_step = patched_train_step
    
    elif hasattr(trainer, 'forward_backward_pass'):
        # Store original methods to patch
        original_method = trainer.forward_backward_pass
        
        # Define a patched method
        def patched_forward_backward_pass(batch):
            # Call the original method
            loss, outputs = original_method(batch)
            
            # Debug the outputs
            debugger.debug_model_outputs(trainer.model, batch, outputs, loss)
            
            # Every 200 steps, check regularization
            if debugger.step % 200 == 0:
                debugger.check_regularization(trainer.model, trainer.optimizer)
                
            return loss, outputs
        
        # Patch the method
        trainer.forward_backward_pass = patched_forward_backward_pass
    
    else:
        print("Warning: Could not find a suitable method to patch in the trainer.")
        print("Available methods:", [method for method in dir(trainer) if not method.startswith('_') or method == '_training_step'])
        print("Debugging will not be automatically attached.")
    
    return debugger


def debug_sample_batch(model, batch, tokenizer):
    """
    Quick debug function to analyze a single batch of data.
    
    Args:
        model: The transformer model
        batch: A batch of data (dictionary with src, tgt, etc.)
        tokenizer: Target tokenizer for decoding
    
    Returns:
        A dictionary with debug information
    """
    # Make sure we're in eval mode
    model.eval()
    
    # Check batch content
    batch_debug = {}
    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            batch_debug[key] = {
                "shape": list(value.shape),
                "dtype": str(value.dtype),
                "device": str(value.device),
                "min": float(value.min().item()) if value.numel() > 0 else None,
                "max": float(value.max().item()) if value.numel() > 0 else None,
                "mean": float(value.float().mean().item()) if value.numel() > 0 else None,
                "has_nan": bool(torch.isnan(value).any().item()) if value.numel() > 0 else None
            }
    
    # Try a forward pass
    with torch.no_grad():
        try:
            outputs = model(
                src=batch.get("src"),
                tgt=batch.get("tgt"),
                src_mask=batch.get("src_mask"),
                tgt_mask=batch.get("tgt_mask")
            )
            
            output_debug = {
                "shape": list(outputs.shape),
                "dtype": str(outputs.dtype),
                "min": float(outputs.min().item()),
                "max": float(outputs.max().item()),
                "mean": float(outputs.float().mean().item()),
                "has_nan": bool(torch.isnan(outputs).any().item())
            }
            
            # Check if outputs look reasonable
            if output_debug["has_nan"]:
                print("WARNING: NaN values in model outputs!")
                
            if abs(output_debug["mean"]) < 1e-6:
                print("WARNING: Model outputs are close to zero!")
                
            if abs(output_debug["min"]) > 100 or abs(output_debug["max"]) > 100:
                print("WARNING: Model outputs have very large values!")
                
            # Compute logits for a sample sequence
            sample_idx = 0
            if tokenizer and "tgt" in batch and batch["tgt"] is not None:
                tgt_seq = batch["tgt"][sample_idx].cpu()
                decoded = tokenizer.decode(tgt_seq.tolist())
                print(f"Sample target sequence: {decoded}")
                
                # Get predictions for this sample
                sample_outputs = outputs[sample_idx]
                probs = torch.nn.functional.softmax(sample_outputs, dim=-1)
                top_probs, top_indices = torch.topk(probs, k=5, dim=-1)
                
                # Print predictions for first few positions
                print("\nTop predictions for first 3 positions:")
                for pos in range(min(3, sample_outputs.size(0))):
                    print(f"Position {pos}:")
                    for j in range(5):  # Top 5 predictions
                        token_idx = top_indices[pos, j].item()
                        token_prob = top_probs[pos, j].item()
                        token = tokenizer.decode([token_idx])
                        print(f"  {j+1}. {token} (id={token_idx}): {token_prob:.6f}")
            
            batch_debug["forward_pass_successful"] = True
            batch_debug["outputs"] = output_debug
            
        except Exception as e:
            print(f"Error in forward pass: {str(e)}")
            batch_debug["forward_pass_successful"] = False
            batch_debug["error"] = str(e)
    
    return batch_debug 