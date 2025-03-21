# examples/translation_example_fixed.py
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict, Tuple
import os
import sys
import time
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import requests
import zipfile
import io
import random
from tqdm import tqdm
import re
from collections import Counter

# Add parent directory to path to import local modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models.transformer import EncoderDecoderTransformer
from src.data.sequence_data import TransformerDataModule
from src.training.transformer_trainer import TransformerTrainer
from src.training.transformer_utils import create_padding_mask, create_causal_mask

class SimpleTokenizer:
    """A simple tokenizer that splits on spaces and punctuation."""
    
    def __init__(self, language="en"):
        self.language = language
        
        # Patterns to use for tokenization
        self.patterns = [
            r'(?<=[.,!?;:])(?=[^\s])',  # Split after punctuation
            r'\s+',                      # Split on whitespace
        ]
    
    def __call__(self, text):
        """Tokenize text into words."""
        # Simple preprocessing
        text = text.lower().strip()
        
        # Apply patterns
        for pattern in self.patterns:
            text = re.sub(pattern, ' ', text)
        
        # Split and filter empty tokens
        tokens = [token for token in text.split() if token]
        
        return tokens

class Vocabulary:
    """A vocabulary class for managing word-to-index mappings."""
    
    def __init__(self, specials=None):
        """
        Initialize vocabulary.
        
        Args:
            specials: List of special tokens to include
        """
        self.word2idx = {}
        self.idx2word = []
        self.word_counts = Counter()
        
        # Add special tokens if provided
        if specials:
            for token in specials:
                self.add_token(token, count=float('inf'))  # Special tokens have infinite count
    
    def add_token(self, token, count=1):
        """Add a token to the vocabulary or update its count."""
        self.word_counts[token] += count
        
        # Add to vocabulary if not already present
        if token not in self.word2idx:
            self.word2idx[token] = len(self.idx2word)
            self.idx2word.append(token)
    
    def build(self, min_freq=1):
        """Build vocabulary from collected counts, filtering by minimum frequency."""
        # Reset and re-add special tokens first
        special_tokens = [word for word, count in self.word_counts.items() 
                         if count == float('inf')]
        
        # Reset vocabulary
        self.word2idx = {}
        self.idx2word = []
        
        # Re-add special tokens
        for token in special_tokens:
            self.word2idx[token] = len(self.idx2word)
            self.idx2word.append(token)
        
        # Add regular tokens that meet minimum frequency
        for word, count in self.word_counts.items():
            if count >= min_freq and word not in self.word2idx:
                self.word2idx[word] = len(self.idx2word)
                self.idx2word.append(word)
        
        return self
    
    def __getitem__(self, token):
        """Get index for a token, returning the UNK index if not found."""
        return self.word2idx.get(token, self.word2idx.get("<unk>", 0))
    
    def __len__(self):
        """Get vocabulary size."""
        return len(self.idx2word)
    
    def lookup_token(self, idx):
        """Get token for an index."""
        if idx < len(self.idx2word):
            return self.idx2word[idx]
        return "<unk>"

class IWSLTDataset:
    """Dataset class for the IWSLT translation dataset."""
    
    def __init__(self, 
                 src_lang="en", 
                 tgt_lang="de", 
                 year="2017", 
                 split="train",
                 max_examples=None):
        """
        Initialize the IWSLT dataset.
        
        Args:
            src_lang: Source language code
            tgt_lang: Target language code
            year: Dataset year
            split: Data split (train, valid, test)
            max_examples: Maximum number of examples to use (None = use all)
        """
        self.src_lang = src_lang
        self.tgt_lang = tgt_lang
        self.year = year
        self.split = split
        self.max_examples = max_examples
        
        # Paths
        self.data_dir = "data/iwslt"
        os.makedirs(self.data_dir, exist_ok=True)
        
        # Download and process data
        self.download_data()
        self.src_data, self.tgt_data = self.load_data()
        
        # Print dataset info
        print(f"Loaded {len(self.src_data)} {split} examples")
    
    def download_data(self):
        """Download IWSLT dataset if not already present."""
        # Define data paths
        self.src_file = f"{self.data_dir}/iwslt{self.year}.{self.src_lang}-{self.tgt_lang}.{self.split}.{self.src_lang}"
        self.tgt_file = f"{self.data_dir}/iwslt{self.year}.{self.src_lang}-{self.tgt_lang}.{self.split}.{self.tgt_lang}"
        
        # Check if data exists
        if os.path.exists(self.src_file) and os.path.exists(self.tgt_file):
            print(f"IWSLT {self.year} {self.src_lang}-{self.tgt_lang} {self.split} data already exists")
            return
        
        # Download data
        print(f"Downloading IWSLT {self.year} {self.src_lang}-{self.tgt_lang} {self.split} data...")
        
        # Use HuggingFace datasets as a more reliable source
        try:
            from datasets import load_dataset
            
            # Load the IWSLT dataset
            dataset = load_dataset("iwslt2017", f"{self.src_lang}-{self.tgt_lang}", split=self.split)
            
            # Extract source and target texts
            src_texts = [example['translation'][self.src_lang] for example in dataset]
            tgt_texts = [example['translation'][self.tgt_lang] for example in dataset]
            
            # Save to files
            os.makedirs(self.data_dir, exist_ok=True)
            
            with open(self.src_file, 'w', encoding='utf-8') as f:
                f.write('\n'.join(src_texts))
            
            with open(self.tgt_file, 'w', encoding='utf-8') as f:
                f.write('\n'.join(tgt_texts))
                
            print("Download completed")
            
        except Exception as e:
            print(f"Error downloading data: {e}")
            
            # Fallback: Create small sample dataset if download fails
            print("Creating sample dataset instead...")
            en_samples = [
                "Hello, how are you?",
                "I am learning machine translation.",
                "Transformers are powerful models for NLP.",
                "This is an example of English to German translation.",
                "The weather is nice today.",
                "I love programming and artificial intelligence.",
                "Neural networks can learn complex patterns.",
                "Please translate this sentence to German.",
                "The cat is sleeping on the couch.",
                "We are working on a challenging project."
            ]
            de_samples = [
                "Hallo, wie geht es dir?",
                "Ich lerne maschinelle Übersetzung.",
                "Transformer sind leistungsstarke Modelle für NLP.",
                "Dies ist ein Beispiel für die Übersetzung von Englisch nach Deutsch.",
                "Das Wetter ist heute schön.",
                "Ich liebe Programmierung und künstliche Intelligenz.",
                "Neuronale Netze können komplexe Muster lernen.",
                "Bitte übersetzen Sie diesen Satz ins Deutsche.",
                "Die Katze schläft auf dem Sofa.",
                "Wir arbeiten an einem anspruchsvollen Projekt."
            ]
            
            with open(self.src_file, 'w', encoding='utf-8') as f:
                f.write('\n'.join(en_samples))
            
            with open(self.tgt_file, 'w', encoding='utf-8') as f:
                f.write('\n'.join(de_samples))
            
            print("Created sample dataset")
    
    def load_data(self):
        """Load and preprocess the data."""
        # Read data files
        with open(self.src_file, 'r', encoding='utf-8') as f:
            src_data = f.read().strip().split('\n')
        
        with open(self.tgt_file, 'r', encoding='utf-8') as f:
            tgt_data = f.read().strip().split('\n')
        
        # Ensure same length
        assert len(src_data) == len(tgt_data), "Source and target data must have same length"
        
        # Limit dataset size if specified
        if self.max_examples is not None and self.max_examples < len(src_data):
            # Use a fixed random seed for reproducibility
            random.seed(42)
            indices = random.sample(range(len(src_data)), self.max_examples)
            src_data = [src_data[i] for i in indices]
            tgt_data = [tgt_data[i] for i in indices]
        
        return src_data, tgt_data

def build_vocabs(dataset, src_tokenizer, tgt_tokenizer, min_freq=2):
    """
    Build vocabularies for source and target languages.
    
    Args:
        dataset: IWSLT dataset
        src_tokenizer: Source language tokenizer
        tgt_tokenizer: Target language tokenizer
        min_freq: Minimum frequency for tokens to be included
        
    Returns:
        Source vocabulary, target vocabulary
    """
    # Create vocabularies with special tokens
    special_tokens = ["<pad>", "<bos>", "<eos>", "<unk>"]
    src_vocab = Vocabulary(specials=special_tokens)
    tgt_vocab = Vocabulary(specials=special_tokens)
    
    # Process all source texts
    for text in tqdm(dataset.src_data, desc="Building source vocabulary"):
        tokens = src_tokenizer(text)
        for token in tokens:
            src_vocab.add_token(token)
    
    # Process all target texts
    for text in tqdm(dataset.tgt_data, desc="Building target vocabulary"):
        tokens = tgt_tokenizer(text)
        for token in tokens:
            tgt_vocab.add_token(token)
    
    # Build final vocabularies with minimum frequency filter
    src_vocab.build(min_freq=min_freq)
    tgt_vocab.build(min_freq=min_freq)
    
    return src_vocab, tgt_vocab

def preprocess_data(dataset, src_tokenizer, tgt_tokenizer, src_vocab, tgt_vocab):
    """
    Preprocess the dataset for training.
    
    Args:
        dataset: IWSLT dataset
        src_tokenizer: Source language tokenizer
        tgt_tokenizer: Target language tokenizer
        src_vocab: Source vocabulary
        tgt_vocab: Target vocabulary
        
    Returns:
        Tokenized source sequences, tokenized target sequences
    """
    src_sequences = []
    tgt_sequences = []
    
    for src_text, tgt_text in zip(dataset.src_data, dataset.tgt_data):
        # Tokenize
        src_tokens = src_tokenizer(src_text)
        tgt_tokens = tgt_tokenizer(tgt_text)
        
        # Convert to indices
        src_indices = [src_vocab[token] for token in src_tokens]
        tgt_indices = [tgt_vocab[token] for token in tgt_tokens]
        
        src_sequences.append(src_indices)
        tgt_sequences.append(tgt_indices)
    
    return src_sequences, tgt_sequences

def calculate_bleu(hypotheses, references):
    """
    Calculate a simplified BLEU score.
    
    Args:
        hypotheses: List of generated translations (token lists)
        references: List of reference translations (token lists)
        
    Returns:
        BLEU score
    """
    # This is a very simplified BLEU implementation
    # In practice, you would use a library like NLTK or sacrebleu
    
    def count_ngrams(sentence, n):
        """Count n-grams in a sentence."""
        ngrams = {}
        for i in range(len(sentence) - n + 1):
            ngram = tuple(sentence[i:i+n])
            ngrams[ngram] = ngrams.get(ngram, 0) + 1
        return ngrams
    
    def modified_precision(hypothesis, reference, n):
        """Calculate modified precision for n-grams."""
        hyp_ngrams = count_ngrams(hypothesis, n)
        ref_ngrams = count_ngrams(reference, n)
        
        if not hyp_ngrams:
            return 0
        
        # Count matches
        matches = 0
        for ngram, count in hyp_ngrams.items():
            matches += min(count, ref_ngrams.get(ngram, 0))
        
        return matches / sum(hyp_ngrams.values())
    
    # Calculate n-gram precisions
    precisions = []
    for n in range(1, 5):
        precision_sum = 0
        for hyp, ref in zip(hypotheses, references):
            precision_sum += modified_precision(hyp, ref, n)
        
        if len(hypotheses) > 0:
            precisions.append(precision_sum / len(hypotheses))
        else:
            precisions.append(0)
    
    # Apply weights (equal weights for simplicity)
    weighted_precision = sum(0.25 * p for p in precisions) if all(p > 0 for p in precisions) else 0
    
    # Calculate brevity penalty
    hyp_length = sum(len(h) for h in hypotheses)
    ref_length = sum(len(r) for r in references)
    
    if hyp_length < ref_length:
        bp = np.exp(1 - ref_length / hyp_length) if hyp_length > 0 else 0
    else:
        bp = 1
    
    # Calculate BLEU
    bleu = bp * np.exp(weighted_precision)
    
    return bleu

def early_stopping(history, patience=10):
    """
    Determine whether to stop training based on validation loss.
    
    Args:
        history: Dictionary containing training history
        patience: Number of epochs to wait for improvement
    
    Returns:
        Boolean indicating whether to stop training
    """
    val_losses = history.get('val_loss', [])
    
    # Not enough history to apply early stopping
    if len(val_losses) < patience:
        return False
    
    # Get recent validation losses
    recent_losses = val_losses[-patience:]
    
    # Find the best (lowest) loss in the recent window
    best_loss = min(recent_losses)
    
    # Stop if no significant improvement for 'patience' epochs
    # Using a 1% improvement threshold to prevent stopping too early
    return recent_losses[-1] > best_loss * 0.99


def main():
    # Set random seed for reproducibility
    torch.manual_seed(42)
    random.seed(42)
    np.random.seed(42)

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")
    # # Set device
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # print(f"Using device: {device}")
    
    # Load dataset
    print("Loading IWSLT dataset...")
    train_dataset = IWSLTDataset(
        src_lang="en",
        tgt_lang="de",
        year="2016",  # Using 2016 version
        split="train",
        max_examples=1000000  # Increased from 5000 to 1000000
    )
    
    val_dataset = IWSLTDataset(
        src_lang="en",
        tgt_lang="de",
        year="2016",
        split="valid",
        max_examples=30000  # Increased from 500 to 30000
    )
    
    # Initialize simple tokenizers
    src_tokenizer = SimpleTokenizer(language="en")
    tgt_tokenizer = SimpleTokenizer(language="de")
    
    # Build vocabularies
    print("Building vocabularies...")
    src_vocab, tgt_vocab = build_vocabs(
        train_dataset,
        src_tokenizer,
        tgt_tokenizer,
        min_freq=1  # Reduced from 2 to 1 to include more vocabulary
    )
    
    print(f"Source vocabulary size: {len(src_vocab)}")
    print(f"Target vocabulary size: {len(tgt_vocab)}")
    
    # Preprocess data
    print("Preprocessing training data...")
    train_src_sequences, train_tgt_sequences = preprocess_data(
        train_dataset,
        src_tokenizer,
        tgt_tokenizer,
        src_vocab,
        tgt_vocab
    )
    
    print("Preprocessing validation data...")
    val_src_sequences, val_tgt_sequences = preprocess_data(
        val_dataset,
        src_tokenizer,
        tgt_tokenizer,
        src_vocab,
        tgt_vocab
    )
    
    # Create data module
    print("Creating data module...")
    data_module = TransformerDataModule(
        source_sequences=train_src_sequences,
        target_sequences=train_tgt_sequences,
        batch_size=128,  # Increased batch size for training efficiency
        max_src_len=100,  # Limit sequence length for memory efficiency
        max_tgt_len=100,
        pad_idx=src_vocab["<pad>"],
        bos_idx=src_vocab["<bos>"],
        eos_idx=src_vocab["<eos>"],
        val_split=0.0,  # We already have a separate validation set
        shuffle=True,
        num_workers=os.cpu_count()
    )
    
    # Create a separate validation data module
    val_data_module = TransformerDataModule(
        source_sequences=val_src_sequences,
        target_sequences=val_tgt_sequences,
        batch_size=128,
        max_src_len=100,
        max_tgt_len=100,
        pad_idx=src_vocab["<pad>"],
        bos_idx=src_vocab["<bos>"],
        eos_idx=src_vocab["<eos>"],
        val_split=0.0,  # All data is for validation
        shuffle=False,
        num_workers=4,
    )
    
    # Create model with larger capacity
    print("Creating transformer model...")
    model = EncoderDecoderTransformer(
        src_vocab_size=len(src_vocab),
        tgt_vocab_size=len(tgt_vocab),
        d_model=512,  # Increased from 256 to 512
        num_heads=8,
        num_encoder_layers=6,  # Increased from 3 to 6
        num_decoder_layers=6,  # Increased from 3 to 6
        d_ff=2048,  # Increased from 512 to 2048
        dropout=0.1,
        max_seq_length=100,
        positional_encoding="sinusoidal",
        share_embeddings=True,  # Changed to True to reduce parameters and improve generalization
    )
    
    model.to(device)
    
    # Create trainer with adjusted parameters
    print("Creating trainer...")
    trainer = TransformerTrainer(
        model=model,
        train_dataloader=data_module.get_train_dataloader(),
        val_dataloader=val_data_module.get_train_dataloader(),
        pad_idx=src_vocab["<pad>"],
        lr=0.0001,  # Reduced from 0.0002 for more stable training
        warmup_steps=4000,  # Increased from 2000 to 4000
        label_smoothing=0.1,
        clip_grad=1.0,
        early_stopping_patience=10,  # Increased from 5 to 10
        device=device,
        track_perplexity=True  # Add this to track perplexity
    )
    
    # Create save directory
    os.makedirs("models", exist_ok=True)
    
    # Train for more epochs
    print("Training model...")
    history = trainer.train(epochs=200, save_path="models/iwslt_en_de")  # Increased from 10 to 50 epochs
    
    # Plot training history
    trainer.plot_training_history()
    plt.savefig("iwslt_training_history.png")
    plt.close()
    
    # Plot learning rate schedule
    trainer.plot_learning_rate()
    plt.savefig("iwslt_learning_rate_schedule.png")
    plt.close()
    
    # Function to translate text
    def translate(text, max_len=100):
        """
        Translate English text to German.
        
        Args:
            text: English text to translate
            max_len: Maximum length of generated translation
            
        Returns:
            German translation
        """
        model.eval()
        
        # Tokenize source text
        src_tokens = src_tokenizer(text)
        src_indices = [src_vocab[token] for token in src_tokens]
        src_tensor = torch.tensor([src_indices], dtype=torch.long).to(device)
        
        # Set start token
        tgt = torch.tensor([[tgt_vocab["<bos>"]]], dtype=torch.long).to(device)
        
        # Create source mask
        src_mask = create_padding_mask(src_tensor, pad_idx=src_vocab["<pad>"])
        
        # Encode source
        memory = model.encode(src_tensor, src_mask=src_mask)
        
        # Generate translation auto-regressively
        for i in range(max_len - 1):
            # Create target mask to prevent attending to future tokens
            tgt_mask = create_causal_mask(tgt.size(1), device)
            
            # Predict next token
            output = model.decode(tgt, memory, tgt_mask=tgt_mask)
            next_token_logits = output[:, -1, :]
            next_token = next_token_logits.argmax(dim=-1, keepdim=True)
            
            # Append to output sequence
            tgt = torch.cat([tgt, next_token], dim=1)
            
            # Stop if end token is generated
            if next_token.item() == tgt_vocab["<eos>"]:
                break
        
        # Convert token indices to words (skip BOS, include EOS)
        tgt_indices = tgt[0].cpu().tolist()
        tgt_tokens = []
        for idx in tgt_indices[1:]:  # Skip BOS token
            if idx == tgt_vocab["<eos>"]:
                break
            tgt_tokens.append(tgt_vocab.lookup_token(idx))
        
        return " ".join(tgt_tokens)
    
    # Test translation on some examples
    test_sentences = [
        "Hello, how are you?",
        "I am learning machine translation.",
        "Transformers are powerful models for NLP.",
        "This is an example of English to German translation."
    ]
    
    print("\n=== Testing Translation ===")
    for sentence in test_sentences:
        translation = translate(sentence)
        print(f"Source: {sentence}")
        print(f"Translation: {translation}")
        print()
    
    # Calculate BLEU score
    print("Calculating BLEU score...")
    candidate_corpus = []
    reference_corpus = []
    
    # Use a smaller subset for evaluation to save time
    for src_indices, tgt_indices in tqdm(list(zip(val_src_sequences, val_tgt_sequences))[:100], desc="Translating"):
        # Create source tensor
        src_tensor = torch.tensor([src_indices], dtype=torch.long).to(device)
        
        # Set start token
        tgt = torch.tensor([[tgt_vocab["<bos>"]]], dtype=torch.long).to(device)
        
        # Create source mask
        src_mask = create_padding_mask(src_tensor, pad_idx=src_vocab["<pad>"])
        
        # Encode source
        memory = model.encode(src_tensor, src_mask=src_mask)
        
        # Generate translation auto-regressively
        for i in range(100):  # Max length of 100
            # Create target mask
            tgt_mask = create_causal_mask(tgt.size(1), device)
            
            # Predict next token
            output = model.decode(tgt, memory, tgt_mask=tgt_mask)
            next_token_logits = output[:, -1, :]
            next_token = next_token_logits.argmax(dim=-1, keepdim=True)
            
            # Append to output sequence
            tgt = torch.cat([tgt, next_token], dim=1)
            
            # Stop if end token is generated
            if next_token.item() == tgt_vocab["<eos>"]:
                break
        
        # Convert token indices to words
        predicted_indices = tgt[0].cpu().tolist()
        predicted_tokens = []
        for idx in predicted_indices[1:]:  # Skip BOS token
            if idx == tgt_vocab["<eos>"]:
                break
            predicted_tokens.append(tgt_vocab.lookup_token(idx))
        
        # Convert target indices to words
        target_tokens = []
        for idx in tgt_indices:
            if idx == tgt_vocab["<eos>"]:
                break
            target_tokens.append(tgt_vocab.lookup_token(idx))
        
        candidate_corpus.append(predicted_tokens)
        reference_corpus.append(target_tokens)
    
    # Calculate BLEU score
    bleu = calculate_bleu(candidate_corpus, reference_corpus)
    print(f"BLEU score on validation set: {bleu:.4f}")
    

if __name__ == "__main__":
    main()

    # Visualize Training History
    plt.figure(figsize=(15, 5))

    # Loss Subplot
    plt.subplot(1, 2, 1)
    plt.title('Training and Validation Loss')
    plt.plot(history['train_loss'], label='Training Loss')
    plt.plot(history['val_loss'], label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    # Perplexity Subplot
    plt.subplot(1, 2, 2)
    plt.title('Training and Validation Perplexity')
    plt.plot(history['train_ppl'], label='Training Perplexity')
    plt.plot(history['val_ppl'], label='Validation Perplexity')
    plt.xlabel('Epochs')
    plt.ylabel('Perplexity')
    plt.legend()

    plt.tight_layout()
    plt.savefig('training_history.png')
    plt.close()