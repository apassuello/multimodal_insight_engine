#!/usr/bin/env python3
# debug_training_issue.py
# Script to debug training issues in the TransformerTrainer

import torch
import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict, Optional
import math
import random
from tqdm import tqdm
import argparse

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models.transformer import EncoderDecoderTransformer
from src.data.sequence_data import TransformerDataModule
from src.training.transformer_trainer import TransformerTrainer
from src.data.tokenization import BPETokenizer, OptimizedBPETokenizer
from src.data.europarl_dataset import EuroparlDataset
from src.data.opensubtitles_dataset import OpenSubtitlesDataset

class DiagnosticTrainer:
    """
    Wrapper around TransformerTrainer to diagnose training issues.
    This class intercepts key methods and adds detailed logging.
    """
    
    def __init__(
        self,
        trainer: TransformerTrainer,
        src_tokenizer=None,
        tgt_tokenizer=None,
        verbose=True,
        log_file="training_debug.log"
    ):
        """
        Initialize the diagnostic trainer wrapper.
        
        Args:
            trainer: TransformerTrainer instance to wrap
            src_tokenizer: Source tokenizer for decoding examples
            tgt_tokenizer: Target tokenizer for decoding examples
            verbose: Whether to print debugging info
            log_file: File to write debug logs to
        """
        self.trainer = trainer
        self.src_tokenizer = src_tokenizer
        self.tgt_tokenizer = tgt_tokenizer
        self.verbose = verbose
        self.step = 0
        self.loss_history = []
        
        # Create log file
        os.makedirs(os.path.dirname(log_file) if os.path.dirname(log_file) else ".", exist_ok=True)
        self.log_file = open(log_file, "w")
        self.log("=== Transformer Training Diagnostics ===\n")
        
        # Store original methods to intercept
        self._patch_methods()
    
    def __del__(self):
        """Close the log file when the object is destroyed."""
        if hasattr(self, 'log_file') and self.log_file:
            self.log_file.close()
            
    def _patch_methods(self):
        """Patch key methods of the trainer to add diagnostics."""
        # Store original methods
        self._original_train_epoch = self.trainer.train_epoch
        self.trainer.train_epoch = lambda: self._wrap_train_epoch()
        
        # The _calculate_loss method doesn't exist in TransformerTrainer, so we'll patch the criterion instead
        self._original_criterion = self.trainer.criterion
        self._criterion_forward = self._original_criterion.forward
        self._original_criterion.forward = lambda *args, **kwargs: self._wrap_criterion(*args, **kwargs)
        
        # The forward_backward_pass doesn't exist, so we'll monitor the train_epoch method
        
        # Monkey patch train method
        self._original_train = self.trainer.train
        self.trainer.train = lambda epochs, save_path=None: self._wrap_train(epochs, save_path)
        
        # Log that we've patched the methods
        self.log("Patched trainer methods:")
        self.log(f"  - train_epoch")
        self.log(f"  - criterion (for loss calculation)")
        self.log(f"  - train")
    
    def log(self, message):
        """Log a message to the file and optionally print it."""
        if self.verbose:
            print(message)
        self.log_file.write(f"{message}\n")
        self.log_file.flush()
    
    def _wrap_train(self, epochs, save_path=None):
        """Wrap the train method to add diagnostics."""
        self.log(f"\n=== Starting Training for {epochs} Epochs ===")
        self.log(f"Model architecture: {self.trainer.model.__class__.__name__}")
        self.log(f"Trainable parameters: {sum(p.numel() for p in self.trainer.model.parameters() if p.requires_grad)}")
        
        # Log training parameters
        self.log(f"Learning rate: {self.trainer.optimizer.param_groups[0]['lr']}")
        self.log(f"Warmup steps: {self.trainer.warmup_steps}")
        if hasattr(self.trainer.criterion, 'smoothing'):
            self.log(f"Label smoothing: {self.trainer.criterion.smoothing}")
        if hasattr(self.trainer, 'clip_grad'):
            self.log(f"Gradient clipping: {self.trainer.clip_grad}")
        
        # Call original train method
        return self._original_train(epochs, save_path)
    
    def _wrap_train_epoch(self):
        """Wrap the train_epoch method to add diagnostics."""
        self.log(f"\n=== Starting Epoch {self.trainer.current_epoch} ===")
        
        # Call original train_epoch
        result = self._original_train_epoch()
        
        # Log overall epoch results
        self.log(f"Epoch {self.trainer.current_epoch} completed - loss: {result:.4f}")
        
        return result
    
    def _wrap_criterion(self, outputs, targets):
        """Wrap the criterion forward method to diagnose loss calculation issues."""
        # Log details about the outputs and targets
        if self.step % 50 == 0:
            self.log("\n=== Loss Calculation Details ===")
            self.log(f"Outputs shape: {outputs.shape}")
            self.log(f"Targets shape: {targets.shape}")
            
            # Check for potential issues
            if outputs.shape[:-1] != targets.shape:
                self.log(f"WARNING: Output and target shapes don't match!")
            
            # Sample a few predictions for the first sequence
            if outputs.dim() == 3:  # [batch, seq, vocab]
                position = 0  # First position
                batch_idx = 0  # First batch item
                
                # Get top predictions for this position
                probs = torch.nn.functional.softmax(outputs[batch_idx, position], dim=-1)
                top_probs, top_indices = torch.topk(probs, 5)
                
                # Log the true target and top predictions
                true_idx = targets[batch_idx, position].item()
                true_prob = probs[true_idx].item()
                
                self.log(f"Sample prediction (batch {batch_idx}, pos {position}):")
                if self.tgt_tokenizer:
                    true_token = self.tgt_tokenizer.decode([true_idx])
                    self.log(f"  True token: '{true_token}' (id={true_idx}), prob: {true_prob:.6f}")
                    
                    self.log(f"  Top 5 predictions:")
                    for i, (idx, prob) in enumerate(zip(top_indices.tolist(), top_probs.tolist())):
                        token = self.tgt_tokenizer.decode([idx])
                        self.log(f"    {i+1}. '{token}' (id={idx}), prob: {prob:.6f}")
                else:
                    self.log(f"  True token ID: {true_idx}, prob: {true_prob:.6f}")
                    self.log(f"  Top 5 prediction IDs: {top_indices.tolist()}")
                    self.log(f"  Top 5 probabilities: {top_probs.tolist()}")
        
        # Call original criterion method
        loss = self._criterion_forward(outputs, targets)
        
        # Store loss for later analysis
        self.loss_history.append(loss.item())
        
        # Log detailed loss information periodically
        if self.step % 50 == 0:
            self.log(f"Loss value: {loss.item():.4f}")
            self.log(f"Perplexity: {math.exp(loss.item()):.2f}")
            
            # Check if loss is suspiciously high
            if loss.item() > 6.0:
                vocab_size = outputs.size(-1)
                random_guess_loss = math.log(vocab_size)
                self.log(f"WARNING: Loss is very high! Current: {loss.item():.4f}")
                self.log(f"Random guessing would give loss of ~{random_guess_loss:.4f}")
        
        self.step += 1
        return loss
    
    def plot_loss_history(self, filename="loss_history.png"):
        """Plot the loss history."""
        if not self.loss_history:
            self.log("No loss history to plot")
            return
        
        plt.figure(figsize=(10, 6))
        plt.plot(self.loss_history)
        plt.title('Loss History During Training')
        plt.xlabel('Steps')
        plt.ylabel('Loss')
        plt.grid(True)
        
        # Calculate statistics
        mean_loss = sum(self.loss_history) / len(self.loss_history)
        min_loss = min(self.loss_history)
        max_loss = max(self.loss_history)
        
        # Add loss statistics to plot
        plt.axhline(y=mean_loss, color='r', linestyle='--', label=f'Mean: {mean_loss:.4f}')
        
        # Add text with statistics
        plt.text(0.02, 0.95, f"Mean Loss: {mean_loss:.4f}\nMin Loss: {min_loss:.4f}\nMax Loss: {max_loss:.4f}",
                 transform=plt.gca().transAxes, fontsize=10, verticalalignment='top',
                 bbox=dict(boxstyle='round', facecolor='white', alpha=0.5))
        
        # Add random guess line if we can estimate it
        # Since we don't have access to vocab size, we'll use 8000 as an estimate based on the tokenizer
        estimated_vocab_size = 8000
        random_guess_loss = math.log(estimated_vocab_size)
        plt.axhline(y=random_guess_loss, color='g', linestyle='--', 
                    label=f'Random Guess: {random_guess_loss:.4f}')
        
        plt.legend()
        plt.tight_layout()
        plt.savefig(filename)
        self.log(f"Loss history plot saved to {filename}")


def preprocess_data_with_bpe(dataset, src_tokenizer, tgt_tokenizer):
    """
    Preprocess the dataset for training using BPE tokenizers.
    This function is copied from the working version of the code.
    
    Args:
        dataset: Dataset with src_data and tgt_data attributes
        src_tokenizer: Source language tokenizer (BPE)
        tgt_tokenizer: Target language tokenizer (BPE)
        
    Returns:
        Tuple of (src_sequences, tgt_sequences)
    """
    src_sequences = []
    tgt_sequences = []
    
    for src_text, tgt_text in zip(dataset.src_data, dataset.tgt_data):
        # Tokenize with BPE
        src_ids = src_tokenizer.encode(src_text)
        tgt_ids = tgt_tokenizer.encode(tgt_text)
        
        # Add special tokens
        src_ids = [src_tokenizer.special_tokens["bos_token_idx"]] + src_ids + [src_tokenizer.special_tokens["eos_token_idx"]]
        tgt_ids = [tgt_tokenizer.special_tokens["bos_token_idx"]] + tgt_ids + [tgt_tokenizer.special_tokens["eos_token_idx"]]
        
        src_sequences.append(src_ids)
        tgt_sequences.append(tgt_ids)
    
    return src_sequences, tgt_sequences


def main():
    """Main function to run the debugging."""
    parser = argparse.ArgumentParser(description="Debug transformer training issues")
    
    # Dataset options
    parser.add_argument("--dataset", type=str, choices=["europarl", "opensubtitles"], default="europarl",
                        help="Dataset to use for training")
    parser.add_argument("--max_train_examples", type=int, default=1000,
                        help="Maximum number of training examples to use")
    parser.add_argument("--max_val_examples", type=int, default=200,
                        help="Maximum number of validation examples to use")
    parser.add_argument("--src_lang", type=str, default="de",
                        help="Source language code")
    parser.add_argument("--tgt_lang", type=str, default="en",
                        help="Target language code")
    
    # Model options
    parser.add_argument("--tokenizer_type", type=str, choices=["bpe", "optimized_bpe"], default="bpe",
                        help="Type of tokenizer to use")
    parser.add_argument("--d_model", type=int, default=512,
                        help="Model dimension")
    parser.add_argument("--num_heads", type=int, default=8,
                        help="Number of attention heads")
    parser.add_argument("--num_layers", type=int, default=4,
                        help="Number of layers in encoder and decoder")
    parser.add_argument("--d_ff", type=int, default=2048,
                        help="Feed-forward network dimension")
    parser.add_argument("--dropout", type=float, default=0.1,
                        help="Dropout rate")
    
    # Training options
    parser.add_argument("--batch_size", type=int, default=32,
                        help="Batch size")
    parser.add_argument("--learning_rate", type=float, default=0.0001,
                        help="Learning rate")
    parser.add_argument("--warmup_steps", type=int, default=2000,
                        help="Number of warmup steps")
    parser.add_argument("--label_smoothing", type=float, default=0.1,
                        help="Label smoothing factor")
    parser.add_argument("--epochs", type=int, default=3,
                        help="Number of epochs to train")
    
    args = parser.parse_args()
    
    # Set random seed for reproducibility
    torch.manual_seed(42)
    random.seed(42)
    np.random.seed(42)
    
    # Set device
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load dataset
    print(f"Loading {args.dataset} dataset...")
    if args.dataset == "europarl":
        train_dataset = EuroparlDataset(
            src_lang=args.src_lang,
            tgt_lang=args.tgt_lang,
            max_examples=args.max_train_examples
        )
        
        val_dataset = EuroparlDataset(
            src_lang=args.src_lang,
            tgt_lang=args.tgt_lang,
            max_examples=args.max_val_examples
        )
    else:  # opensubtitles
        train_dataset = OpenSubtitlesDataset(
            data_dir="data/os",
            src_lang=args.src_lang,
            tgt_lang=args.tgt_lang,
            max_examples=args.max_train_examples
        )
        
        val_dataset = OpenSubtitlesDataset(
            data_dir="data/os",
            src_lang=args.src_lang,
            tgt_lang=args.tgt_lang,
            max_examples=args.max_val_examples
        )
    
    # Load tokenizers based on type
    print(f"Loading {args.tokenizer_type} tokenizers...")
    if args.tokenizer_type == "bpe":
        src_tokenizer = BPETokenizer.from_pretrained(f"models/tokenizers/{args.src_lang}")
        tgt_tokenizer = BPETokenizer.from_pretrained(f"models/tokenizers/{args.tgt_lang}")
    else:  # optimized_bpe
        src_tokenizer = OptimizedBPETokenizer.from_pretrained(f"models/tokenizers/{args.src_lang}")
        tgt_tokenizer = OptimizedBPETokenizer.from_pretrained(f"models/tokenizers/{args.tgt_lang}")
    
    print(f"Loaded {args.src_lang} tokenizer with vocab size: {src_tokenizer.vocab_size}")
    print(f"Loaded {args.tgt_lang} tokenizer with vocab size: {tgt_tokenizer.vocab_size}")
    
    # Get special token indices
    src_pad_idx = src_tokenizer.special_tokens["pad_token_idx"]
    tgt_pad_idx = tgt_tokenizer.special_tokens["pad_token_idx"]
    src_bos_idx = src_tokenizer.special_tokens["bos_token_idx"]
    tgt_bos_idx = tgt_tokenizer.special_tokens["bos_token_idx"]
    src_eos_idx = src_tokenizer.special_tokens["eos_token_idx"]
    tgt_eos_idx = tgt_tokenizer.special_tokens["eos_token_idx"]
    
    # Preprocess data with BPE tokenizers using the exact function from the working code
    print("Preprocessing training data...")
    train_src_sequences, train_tgt_sequences = preprocess_data_with_bpe(
        train_dataset, src_tokenizer, tgt_tokenizer
    )
    
    print("Preprocessing validation data...")
    val_src_sequences, val_tgt_sequences = preprocess_data_with_bpe(
        val_dataset, src_tokenizer, tgt_tokenizer
    )
    
    # Create data module
    print("Creating data module...")
    data_module = TransformerDataModule(
        source_sequences=train_src_sequences,
        target_sequences=train_tgt_sequences,
        batch_size=args.batch_size,
        max_src_len=100,
        max_tgt_len=100,
        pad_idx=src_pad_idx,
        bos_idx=src_bos_idx,
        eos_idx=src_eos_idx,
        val_split=0.0,
        shuffle=True,
        num_workers=0  # Use 0 for debugging
    )
    
    # Create a separate validation data module
    val_data_module = TransformerDataModule(
        source_sequences=val_src_sequences,
        target_sequences=val_tgt_sequences,
        batch_size=args.batch_size,
        max_src_len=100,
        max_tgt_len=100,
        pad_idx=src_pad_idx,
        bos_idx=src_bos_idx,
        eos_idx=src_eos_idx,
        val_split=0.0,
        shuffle=False,
        num_workers=0  # Use 0 for debugging
    )
    
    # Create transformer model
    print("Creating transformer model...")
    model = EncoderDecoderTransformer(
        src_vocab_size=src_tokenizer.vocab_size,
        tgt_vocab_size=tgt_tokenizer.vocab_size,
        d_model=args.d_model,
        num_heads=args.num_heads,
        num_encoder_layers=args.num_layers,
        num_decoder_layers=args.num_layers,
        d_ff=args.d_ff,
        dropout=args.dropout,
        max_seq_length=100,
        positional_encoding="sinusoidal",
        share_embeddings=False,
    )
    model.to(device)
    
    # Create trainer exactly as in the working code
    print("Creating trainer...")
    trainer = TransformerTrainer(
        model=model,
        train_dataloader=data_module.get_train_dataloader(),
        val_dataloader=val_data_module.get_train_dataloader(),
        pad_idx=src_pad_idx,
        lr=args.learning_rate,
        warmup_steps=args.warmup_steps,
        label_smoothing=args.label_smoothing,
        clip_grad=1.0,
        early_stopping_patience=5,
        device=device,
        track_perplexity=True,
    )
    
    # Wrap with diagnostic trainer
    diagnostic_trainer = DiagnosticTrainer(
        trainer=trainer,
        src_tokenizer=src_tokenizer,
        tgt_tokenizer=tgt_tokenizer,
        verbose=True,
        log_file="training_diagnostic.log"
    )
    
    # Train for a few epochs to diagnose issues
    print(f"Training for {args.epochs} epochs to diagnose issues...")
    history = trainer.train(epochs=args.epochs)
    
    # Plot loss history
    diagnostic_trainer.plot_loss_history()
    
    # Analyze the results
    print("\n=== TRAINING DIAGNOSTIC RESULTS ===")
    
    # Calculate statistics from the loss history
    if diagnostic_trainer.loss_history:
        mean_loss = sum(diagnostic_trainer.loss_history) / len(diagnostic_trainer.loss_history)
        min_loss = min(diagnostic_trainer.loss_history)
        max_loss = max(diagnostic_trainer.loss_history)
        
        print(f"Loss Analysis:")
        print(f"  - Mean: {mean_loss:.4f}")
        print(f"  - Min: {min_loss:.4f}")
        print(f"  - Max: {max_loss:.4f}")
        
        # Estimate loss from random guessing (using tokenizer vocab size)
        if args.tokenizer_type in ['bpe', 'optimized_bpe']:
            vocab_size = 8005  # Based on loaded tokenizer size
            random_guess = math.log(vocab_size)
            print(f"\nRandom guessing would give loss of {random_guess:.4f}")
            
            # Analyze how close we are to random guessing
            if abs(mean_loss - random_guess) < 0.5:
                print("\nCRITICAL ISSUE: The model is essentially making random predictions!")
                print("This indicates a fundamental problem in the training process:")
                print("  1. Check that target shifting is correct for teacher forcing")
                print("  2. Verify loss function is correctly excluding padding tokens")
                print("  3. Ensure model initialization is not causing uniform predictions")
                print("  4. Verify gradient flow through the model")
        
        # Check if loss decreased during training
        if len(diagnostic_trainer.loss_history) > 5:
            first_losses = diagnostic_trainer.loss_history[:5]
            last_losses = diagnostic_trainer.loss_history[-5:]
            
            first_mean = sum(first_losses) / len(first_losses)
            last_mean = sum(last_losses) / len(last_losses)
            
            print(f"\nLoss Change Analysis:")
            print(f"  - First 5 batches mean loss: {first_mean:.4f}")
            print(f"  - Last 5 batches mean loss: {last_mean:.4f}")
            print(f"  - Change: {last_mean - first_mean:.4f}")
            
            if last_mean >= first_mean - 0.1:
                print("\nWARNING: Loss is not decreasing significantly during training!")
                print("This suggests the model is not learning effectively.")
    
    print("\nRecommended Next Steps:")
    print("1. Review the training loop implementation carefully")
    print("2. Check how target sequences are processed for teacher forcing")
    print("3. Ensure loss calculation is correctly handling padding tokens")
    print("4. Verify gradient flow through the model with a simpler example")
    print("5. Try training with a much smaller learning rate to see if that helps")
    
    print("Diagnostic training completed. Check logs for details.")


if __name__ == "__main__":
    main() 