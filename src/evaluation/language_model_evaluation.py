# src/evaluation/language_model_evaluation.py
import torch
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict, Tuple, Optional, Union, Any
import seaborn as sns
import math
import os
import json
from collections import Counter
from torch.nn import functional as F
from matplotlib.figure import Figure

class LanguageModelEvaluator:
    """
    Evaluation utilities for language models.
    
    This class provides methods for evaluating language model performance
    using metrics like perplexity, and visualizing generation results.
    """
    
    def __init__(
        self,
        model: torch.nn.Module,
        tokenizer,
        device: Optional[torch.device] = None,
    ):
        """
        Initialize the evaluator.
        
        Args:
            model: Language model to evaluate
            tokenizer: Tokenizer for encoding/decoding text
            device: Device to run on
        """
        self.model = model
        self.tokenizer = tokenizer
        
        # Set device
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else 
                                     "mps" if torch.backends.mps.is_available() else 
                                     "cpu")
        else:
            self.device = device
        
        # Move model to device
        self.model.to(self.device)
        
        # Put model in evaluation mode
        self.model.eval()
        
        # Get special token indices
        self.pad_idx = tokenizer.special_tokens["pad_token_idx"]
        self.bos_idx = tokenizer.special_tokens["bos_token_idx"]
        self.eos_idx = tokenizer.special_tokens["eos_token_idx"]
    
    def calculate_perplexity(self, text: str) -> float:
        """
        Calculate perplexity of a text under the model.
        
        Perplexity is a measure of how well a model predicts a sample.
        Lower perplexity indicates better prediction.
        
        Args:
            text: Text to evaluate
            
        Returns:
            Perplexity score
        """
        # Encode text
        input_ids = self.tokenizer.encode(text)
        
        # Add special tokens
        input_ids = [self.bos_idx] + input_ids + [self.eos_idx]
        
        # Create tensor
        input_ids = torch.tensor([input_ids], dtype=torch.long).to(self.device)
        
        # Calculate perplexity
        with torch.no_grad():
            # Forward pass
            outputs = self.model(input_ids=input_ids)
            
            # Get logits
            logits = outputs.logits if hasattr(outputs, "logits") else outputs
            
            # Shift labels and logits for next-token prediction
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = input_ids[..., 1:].contiguous()
            
            # Calculate loss
            loss_fct = torch.nn.CrossEntropyLoss(ignore_index=self.pad_idx)
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), 
                          shift_labels.view(-1))
            
            # Calculate perplexity
            perplexity = math.exp(loss.item())
        
        return perplexity
    
    def calculate_batch_perplexity(self, texts: List[str]) -> Dict[str, Union[float, List[float]]]:
        """
        Calculate perplexity for a batch of texts.
        
        Args:
            texts: List of texts to evaluate
            
        Returns:
            Dictionary with perplexity metrics
        """
        # Encode all texts
        encoded_texts = []
        max_length = 0
        
        for text in texts:
            input_ids = self.tokenizer.encode(text)
            input_ids = [self.bos_idx] + input_ids + [self.eos_idx]
            encoded_texts.append(input_ids)
            max_length = max(max_length, len(input_ids))
        
        # Pad sequences to max length
        padded_texts = []
        for input_ids in encoded_texts:
            padded = input_ids + [self.pad_idx] * (max_length - len(input_ids))
            padded_texts.append(padded)
        
        # Create tensors
        input_ids = torch.tensor(padded_texts, dtype=torch.long).to(self.device)
        attention_mask = (input_ids != self.pad_idx).long()
        
        # Check if we're working with EncoderDecoderTransformer
        is_encoder_decoder = hasattr(self.model, 'encoder') and hasattr(self.model, 'decoder')
        
        # Calculate perplexity
        with torch.no_grad():
            # Forward pass
            if is_encoder_decoder:
                # For encoder-decoder models, use src/tgt parameters
                # For causal language modeling, tgt is shifted input_ids
                src = input_ids
                tgt = input_ids.clone()  # Use same sequence for source and target
                src_mask = attention_mask
                
                outputs = self.model(
                    src=src,
                    tgt=tgt,
                    src_mask=src_mask
                )
                
                # Already have log probs outputs
                logits = outputs
            else:
                # For standard models
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask
                )
                
                # Get logits
                logits = outputs.logits if hasattr(outputs, "logits") else outputs
            
            # Shift labels and logits for next-token prediction
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = input_ids[..., 1:].contiguous()
            
            # Calculate loss
            loss_fct = torch.nn.CrossEntropyLoss(ignore_index=self.pad_idx, reduction='none')
            losses = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), 
                           shift_labels.view(-1))
            
            # Reshape losses to match input shape
            losses = losses.view(shift_labels.size())
            
            # Mask padded positions
            mask = (shift_labels != self.pad_idx).float()
            masked_losses = losses * mask
            
            # Calculate sum and count of losses for each sequence
            seq_sum_loss = masked_losses.sum(dim=1)
            seq_token_count = mask.sum(dim=1)
            
            # Calculate per-sequence average loss and perplexity
            seq_avg_loss = seq_sum_loss / seq_token_count
            seq_perplexity = torch.exp(seq_avg_loss)
            
            # Calculate total average loss and perplexity
            avg_loss = masked_losses.sum() / mask.sum()
            perplexity = math.exp(avg_loss.item())
        
        # Collect results
        perplexities = seq_perplexity.cpu().tolist()
        
        return {
            "perplexity": float(perplexity),
            "per_sequence_perplexity": perplexities,
            "min_perplexity": float(min(perplexities, default=0.0)),
            "max_perplexity": float(max(perplexities, default=0.0)),
            "median_perplexity": float(np.median(perplexities) if perplexities else 0.0),
        }
    
    def analyze_token_probabilities(self, text: str) -> Dict[str, Any]:
        """
        Analyze token probabilities in a text.
        
        Args:
            text: Text to analyze
            
        Returns:
            Dictionary with token probability analysis
        """
        # Encode text
        input_ids = self.tokenizer.encode(text)
        
        # Add special tokens
        input_ids = [self.bos_idx] + input_ids + [self.eos_idx]
        
        # Create tensor
        input_ids = torch.tensor([input_ids], dtype=torch.long).to(self.device)
        
        # Check if we're working with EncoderDecoderTransformer
        is_encoder_decoder = hasattr(self.model, 'encoder') and hasattr(self.model, 'decoder')
        
        # Analyze token probabilities
        with torch.no_grad():
            # Forward pass
            if is_encoder_decoder:
                # For encoder-decoder models, use src/tgt parameters
                src = input_ids
                tgt = input_ids.clone()
                src_mask = torch.ones_like(input_ids, dtype=torch.long)
                
                outputs = self.model(
                    src=src,
                    tgt=tgt,
                    src_mask=src_mask
                )
                
                # Already have probabilities
                probs = outputs
            else:
                # For standard models
                outputs = self.model(input_ids=input_ids)
                
                # Get logits
                logits = outputs.logits if hasattr(outputs, "logits") else outputs
                
                # Calculate probabilities
                probs = F.softmax(logits, dim=-1)
            
            probs = probs[0]  # Ensure probs is a 2D tensor for indexing
            
            # Get probabilities for actual next tokens
            next_token_probs = []
            for i in range(input_ids.size(1) - 1):
                next_token = input_ids[0, i + 1].item()
                next_token_prob = probs[i][int(next_token)].item()  # Ensure correct indexing
                next_token_probs.append(next_token_prob)
            
            # Calculate statistics
            avg_prob = np.mean(next_token_probs)
            min_prob = np.min(next_token_probs)
            max_prob = np.max(next_token_probs)
            
            # Find min and max probability tokens
            min_prob_idx = np.argmin(next_token_probs)
            max_prob_idx = np.argmax(next_token_probs)
            
            min_prob_token = self.tokenizer.decode([input_ids[0, min_prob_idx + 1].item()])
            max_prob_token = self.tokenizer.decode([input_ids[0, max_prob_idx + 1].item()])
            
            # Create token-by-token analysis
            tokens = [self.tokenizer.decode([token_id]) for token_id in input_ids[0, 1:].tolist()]
            token_analysis = [
                {"token": token, "probability": prob}
                for token, prob in zip(tokens, next_token_probs)
            ]
        
        return {
            "average_probability": float(avg_prob),
            "min_probability": float(min_prob),
            "max_probability": float(max_prob),
            "min_probability_token": min_prob_token,
            "max_probability_token": max_prob_token,
            "token_analysis": token_analysis,
        }
    
    def visualize_attention(
        self,
        text: str,
        layer: int = -1,
        head: int = 0,
        attention_type: str = "self",
        cmap: str = "viridis",
    ) -> Figure:
        """
        Visualize attention patterns for a given text.
        
        Args:
            text: Input text
            layer: Layer index to visualize (-1 for last layer)
            head: Attention head to visualize
            attention_type: Type of attention to visualize (self, cross)
            cmap: Colormap for the heatmap
            
        Returns:
            Matplotlib figure with attention visualization
        """
        # Encode text
        input_ids = self.tokenizer.encode(text)
        
        # Add special tokens
        input_ids = [self.bos_idx] + input_ids + [self.eos_idx]
        
        # Get token strings for labels
        tokens = [self.tokenizer.decode([token_id]) for token_id in input_ids]
        
        # Create tensor
        input_ids = torch.tensor([input_ids], dtype=torch.long).to(self.device)
        
        # Check if we're working with EncoderDecoderTransformer
        is_encoder_decoder = hasattr(self.model, 'encoder') and hasattr(self.model, 'decoder')
        
        # Enable attention output
        if hasattr(self.model, 'output_attentions'):
            self.model.output_attentions = True
            
        # Get attention weights
        with torch.no_grad():
            # Forward pass
            if is_encoder_decoder:
                # For encoder-decoder models
                src = input_ids
                tgt = input_ids.clone()
                src_mask = torch.ones_like(input_ids, dtype=torch.long)
                
                # Need to access attention directly from the model components
                # First run encoder
                memory = self.model.encoder(src, mask=src_mask)
                
                # Extract encoder self-attention if that's what we want
                if attention_type == "self" and hasattr(self.model.encoder, 'layers'):
                    # Get the requested layer
                    actual_layer = layer if layer >= 0 else len(self.model.encoder.layers) + layer
                    if 0 <= actual_layer < len(self.model.encoder.layers):
                        # Get self-attention module from the specified layer
                        attn_layer = self.model.encoder.layers[actual_layer]
                        if hasattr(attn_layer, 'self_attn'):
                            # Get normalized input
                            norm_x = attn_layer.norm1(src)
                            # Calculate query, key, value
                            q, k, v = attn_layer.self_attn.prepare_qkv(norm_x, norm_x, norm_x)
                            # Get attention weights
                            _, attention_weights = attn_layer.self_attn.attention(q, k, v, None, self.device)
                            attention_weights = attention_weights.cpu().numpy()
                
                # Run decoder with memory from encoder to get cross-attention
                outputs = self.model.decoder(tgt, memory, tgt_mask=None, memory_mask=None)
                
                # For cross-attention between encoder and decoder
                if attention_type == "cross" and hasattr(self.model.decoder, 'layers'):
                    # Get the requested layer
                    actual_layer = layer if layer >= 0 else len(self.model.decoder.layers) + layer
                    if 0 <= actual_layer < len(self.model.decoder.layers):
                        # Get cross-attention module from the specified layer
                        attn_layer = self.model.decoder.layers[actual_layer]
                        if hasattr(attn_layer, 'encoder_attn'):
                            # Get normalized input for cross-attention
                            norm_x = attn_layer.norm2(tgt)
                            # Calculate query, key, value
                            q = attn_layer.encoder_attn.prepare_query(norm_x)
                            k, v = attn_layer.encoder_attn.prepare_key_value(memory, memory)
                            # Get attention weights
                            _, attention_weights = attn_layer.encoder_attn.attention(q, k, v, None, self.device)
                            attention_weights = attention_weights.cpu().numpy()
            else:
                # For standard models
                outputs = self.model(input_ids=input_ids)
                
                # Extract attention weights
                if hasattr(outputs, 'attentions') and outputs.attentions:
                    # Use the specified layer
                    layer_idx = layer if layer >= 0 else len(outputs.attentions) + layer
                    attention_weights = outputs.attentions[layer_idx][0, head].cpu().numpy()
                else:
                    # If model doesn't output attention, create a dummy pattern
                    token_length = input_ids.size(1)
                    attention_weights = np.eye(token_length)
        
        # Create figure for visualization
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Plot attention heatmap
        im = ax.imshow(attention_weights, cmap=cmap)
        
        # Set labels
        ax.set_xticks(np.arange(len(tokens)))
        ax.set_yticks(np.arange(len(tokens)))
        ax.set_xticklabels(tokens)
        ax.set_yticklabels(tokens)
        
        # Rotate x labels for better readability
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
        
        # Add colorbar
        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label("Attention Weight")
        
        # Add title
        attention_name = "Self" if attention_type == "self" else "Cross"
        ax.set_title(f"{attention_name}-Attention Weights (Layer {layer}, Head {head})")
        
        # Add grid lines
        ax.set_xticks(np.arange(-.5, len(tokens), 1), minor=True)
        ax.set_yticks(np.arange(-.5, len(tokens), 1), minor=True)
        ax.grid(which="minor", color="gray", linestyle="-", linewidth=0.5)
        
        # Ensure layout fits
        fig.tight_layout()
        
        return fig
    
    def visualize_attention_patterns(
        self,
        text: str,
        save_dir: Optional[str] = None,
    ) -> List[Figure]:
        """
        Visualize attention patterns across all layers and heads.
        
        Args:
            text: Text to visualize attention for
            save_dir: Directory to save visualizations
            
        Returns:
            List of Matplotlib figures
        """
        # Encode text
        input_ids = self.tokenizer.encode(text)
        
        # Add special tokens
        input_ids = [self.bos_idx] + input_ids + [self.eos_idx]
        
        # Create tensor
        input_ids = torch.tensor([input_ids], dtype=torch.long).to(self.device)
        
        # Set model output_attentions to True
        self.model.config.output_attentions = True
        
        # Get attention weights
        with torch.no_grad():
            outputs = self.model(input_ids=input_ids)
            
            # Ensure we have attention outputs
            if not hasattr(outputs, "attentions") and not isinstance(outputs.attentions, tuple):
                raise ValueError("Model does not output attention weights")
            
            # Get all attention weights
            attentions = outputs.attentions
            
            # Get tokens
            tokens = [self.tokenizer.decode([token_id]) for token_id in input_ids[0].tolist()]
        
        # Create directory if needed
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
        
        # Create figures
        figures = []
        
        # Loop through layers and heads
        for layer_idx, layer_attn in enumerate(attentions):
            for head_idx in range(layer_attn.size(1)):
                # Get attention weights for this head
                head_attn = layer_attn[0, head_idx].cpu().numpy()
                
                # Create figure
                fig, ax = plt.subplots(figsize=(10, 8))
                im = ax.imshow(head_attn, cmap="viridis")
                
                # Add colorbar
                cbar = ax.figure.colorbar(im, ax=ax)
                cbar.ax.set_ylabel("Attention Weight", rotation=-90, va="bottom")
                
                # Set ticks and labels
                ax.set_xticks(np.arange(len(tokens)))
                ax.set_yticks(np.arange(len(tokens)))
                ax.set_xticklabels(tokens, rotation=45, ha="right", rotation_mode="anchor")
                ax.set_yticklabels(tokens)
                
                # Set grid and labels
                ax.set_xticks(np.arange(-.5, len(tokens), 1), minor=True)
                ax.set_yticks(np.arange(-.5, len(tokens), 1), minor=True)
                ax.grid(which="minor", color="w", linestyle='-', linewidth=2)
                ax.tick_params(which="minor", bottom=False, left=False)
                
                # Set title
                ax.set_title(f"Attention Weights (Layer {layer_idx+1}, Head {head_idx+1})")
                ax.set_xlabel("Key Position")
                ax.set_ylabel("Query Position")
                
                # Save figure if directory is provided
                if save_dir:
                    plt.savefig(f"{save_dir}/attention_L{layer_idx+1}_H{head_idx+1}.png", 
                              bbox_inches="tight")
                
                figures.append(fig)
        
        return figures
    
    def evaluate_on_dataset(
        self,
        texts: List[str],
        save_path: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Evaluate the model on a dataset of texts.
        
        Args:
            texts: List of texts to evaluate
            save_path: Path to save the evaluation results
            
        Returns:
            Dictionary with evaluation results
        """
        # Calculate perplexity for each text
        perplexities = []
        
        print(f"Evaluating on {len(texts)} texts...")
        for i, text in enumerate(texts):
            if i % 10 == 0:
                print(f"Progress: {i}/{len(texts)}")
            
            try:
                perplexity = self.calculate_perplexity(text)
                perplexities.append(perplexity)
            except Exception as e:
                print(f"Error evaluating text {i}: {e}")
        
        # Calculate statistics
        avg_perplexity = np.mean(perplexities)
        median_perplexity = np.median(perplexities)
        min_perplexity = np.min(perplexities)
        max_perplexity = np.max(perplexities)
        std_perplexity = np.std(perplexities)
        
        # Create results dictionary
        results = {
            "num_texts": len(texts),
            "average_perplexity": float(avg_perplexity),
            "median_perplexity": float(median_perplexity),
            "min_perplexity": float(min(perplexities, default=0.0)),
            "max_perplexity": float(max(perplexities, default=0.0)),
            "std_perplexity": float(std_perplexity),
            "per_text_perplexity": [float(p) for p in perplexities],
        }
        
        # Save results if path is provided
        if save_path:
            with open(save_path, "w") as f:
                json.dump(results, f, indent=2)
        
        return results
    
    def plot_perplexity_distribution(
        self,
        perplexities: List[float],
        save_path: Optional[str] = None,
    ) -> Figure:
        """
        Plot the distribution of perplexities.
        
        Args:
            perplexities: List of perplexity values
            save_path: Path to save the plot
            
        Returns:
            Matplotlib figure
        """
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Plot histogram
        ax.hist(perplexities, bins=30, alpha=0.7, color="skyblue")
        
        # Add a kernel density estimate
        sns.kdeplot(perplexities, ax=ax, color="darkblue")
        
        # Add statistics
        avg_perplexity = float(np.mean(perplexities))
        median_perplexity = float(np.median(perplexities))
        
        ax.axvline(avg_perplexity, color="red", linestyle="--", 
                 label=f"Mean: {avg_perplexity:.2f}")
        ax.axvline(median_perplexity, color="green", linestyle=":", 
                 label=f"Median: {median_perplexity:.2f}")
        
        # Add labels and title
        ax.set_xlabel("Perplexity")
        ax.set_ylabel("Frequency")
        ax.set_title("Distribution of Perplexity Scores")
        ax.legend()
        
        # Add grid
        ax.grid(True, alpha=0.3)
        
        # Save figure if path is provided
        if save_path:
            plt.savefig(save_path, bbox_inches="tight")
        
        return fig

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
        "module_purpose": "Provides comprehensive evaluation utilities for language models, including perplexity calculation, attention visualization, and token probability analysis",
        "key_classes": [
            {
                "name": "LanguageModelEvaluator",
                "purpose": "Evaluation class for language models with metrics and visualization capabilities",
                "key_methods": [
                    {
                        "name": "calculate_perplexity",
                        "signature": "calculate_perplexity(self, text: str) -> float",
                        "brief_description": "Calculates perplexity score for a given text under the model"
                    },
                    {
                        "name": "calculate_batch_perplexity",
                        "signature": "calculate_batch_perplexity(self, texts: List[str]) -> Dict[str, Union[float, List[float]]]",
                        "brief_description": "Calculate perplexity for a batch of texts with optimized processing"
                    },
                    {
                        "name": "visualize_attention",
                        "signature": "visualize_attention(self, text: str, layer: int = -1, head: int = 0, attention_type: str = 'self', cmap: str = 'viridis') -> Figure",
                        "brief_description": "Visualizes attention patterns for a given text at specified layer and head"
                    },
                    {
                        "name": "visualize_attention_patterns",
                        "signature": "visualize_attention_patterns(self, text: str, save_dir: Optional[str] = None) -> List[Figure]",
                        "brief_description": "Visualizes attention patterns across all layers and heads"
                    },
                    {
                        "name": "analyze_token_probabilities",
                        "signature": "analyze_token_probabilities(self, text: str) -> Dict[str, Any]",
                        "brief_description": "Analyzes token probabilities in a text to identify high and low confidence predictions"
                    },
                    {
                        "name": "evaluate_on_dataset",
                        "signature": "evaluate_on_dataset(self, texts: List[str], save_path: Optional[str] = None) -> Dict[str, Any]",
                        "brief_description": "Evaluates model performance on a dataset of texts"
                    },
                    {
                        "name": "plot_perplexity_distribution",
                        "signature": "plot_perplexity_distribution(self, perplexities: List[float], save_path: Optional[str] = None) -> Figure",
                        "brief_description": "Plot the distribution of perplexities as a histogram with statistics"
                    }
                ],
                "inheritance": "object",
                "dependencies": ["torch", "numpy", "matplotlib", "seaborn"]
            }
        ],
        "external_dependencies": ["torch", "numpy", "matplotlib", "seaborn"],
        "complexity_score": 7  # High complexity due to visualization features and comprehensive evaluation metrics
    }