# demos/language_model_demo.py
import os
import sys
import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
import random
import time
from pathlib import Path
from typing import List, Dict, Optional, Union, Any
from tqdm import tqdm

# Add parent directory to path to import project modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import modules from our project
from src.models.transformer import EncoderDecoderTransformer
from src.data.tokenization import BPETokenizer
from src.data.language_modeling import (
    LanguageModelingDataset,
    create_lm_dataloaders
)
from src.training.language_model_trainer import LanguageModelTrainer
from src.models.text_generation import TextGenerator
from src.evaluation.language_model_evaluation import LanguageModelEvaluator
from src.data.dataloader import create_dataloader, MultimodalDataset
from src.data.europarl_dataset import EuroparlDataset

def set_seed(seed: int = 42):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def get_device():
    """Get the appropriate device for training."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")

class DatasetProvider:
    """Provides datasets for language modeling."""
    
    @staticmethod
    def get_synthetic_dataset(num_samples: int = 1000, seq_length: int = 20):
        """Generate a synthetic dataset for testing."""
        vocabulary = ["the", "quick", "brown", "fox", "jumps", "over", "lazy", "dog",
                     "hello", "world", "language", "model", "training", "artificial",
                     "intelligence", "machine", "learning", "transformer", "attention"]
        
        texts = []
        for _ in range(num_samples):
            # Generate a random text
            length = random.randint(5, seq_length)
            text = " ".join(random.choices(vocabulary, k=length))
            texts.append(text)
        
        return texts
    
    @staticmethod
    def load_text_files(directory: str):
        """Load text files from a directory."""
        texts = []
        
        # Get all text files in the directory
        files = list(Path(directory).glob("*.txt"))
        print(f"Found {len(files)} text files in {directory}")
        
        for file_path in tqdm(files, desc="Loading text files"):
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    text = f.read()
                
                # Split into paragraphs and add each as a separate example
                paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
                texts.extend(paragraphs)
            except Exception as e:
                print(f"Error loading {file_path}: {e}")
        
        print(f"Loaded {len(texts)} texts from {directory}")
        
        return texts
    
    @staticmethod
    def download_wikitext(save_dir: str = "data", dataset_name: str = "wikitext-2-raw-v1"):
        """Download a subset of the WikiText dataset."""
        try:
            from datasets import load_dataset
            
            # Create directory if it doesn't exist
            os.makedirs(save_dir, exist_ok=True)
            
            # Load dataset
            print(f"Downloading {dataset_name}...")
            dataset = load_dataset(dataset_name)
            
            # Extract texts from each split
            train_texts = dataset["train"]["text"]
            val_texts = dataset["validation"]["text"]
            test_texts = dataset["test"]["text"]
            
            # Filter out empty lines
            train_texts = [text for text in train_texts if text.strip()]
            val_texts = [text for text in val_texts if text.strip()]
            test_texts = [text for text in test_texts if text.strip()]
            
            # Combine consecutive lines to create longer examples
            train_examples = []
            current_example = ""
            
            for text in train_texts:
                if not text.strip():
                    if current_example:
                        train_examples.append(current_example)
                        current_example = ""
                else:
                    current_example += text + " "
            
            if current_example:
                train_examples.append(current_example)
            
            # Do the same for validation
            val_examples = []
            current_example = ""
            
            for text in val_texts:
                if not text.strip():
                    if current_example:
                        val_examples.append(current_example)
                        current_example = ""
                else:
                    current_example += text + " "
            
            if current_example:
                val_examples.append(current_example)
            
            # Report statistics
            print(f"Extracted {len(train_examples)} training examples")
            print(f"Extracted {len(val_examples)} validation examples")
            
            return {
                "train": train_examples,
                "validation": val_examples,
                "test": test_texts
            }
            
        except ImportError:
            print("Could not import 'datasets' library. Falling back to synthetic data.")
            train_texts = DatasetProvider.get_synthetic_dataset(10000, 50)
            val_texts = DatasetProvider.get_synthetic_dataset(1000, 50)
            
            return {
                "train": train_texts,
                "validation": val_texts,
                "test": val_texts[:100]
            }

class ModelProvider:
    """Provides model configurations for language modeling."""
    
    @staticmethod
    def get_small_config():
        """Get a small model configuration for testing."""
        return {
            "d_model": 128,
            "num_heads": 4,
            "num_encoder_layers": 3,
            "num_decoder_layers": 3,
            "d_ff": 512,
            "dropout": 0.1,
            "max_seq_length": 128
        }
    
    @staticmethod
    def get_medium_config():
        """Get a medium model configuration."""
        return {
            "d_model": 256,
            "num_heads": 8,
            "num_encoder_layers": 6,
            "num_decoder_layers": 6,
            "d_ff": 1024,
            "dropout": 0.1,
            "max_seq_length": 256
        }
    
    @staticmethod
    def get_large_config():
        """Get a large model configuration."""
        return {
            "d_model": 512,
            "num_heads": 8,
            "num_encoder_layers": 6,
            "num_decoder_layers": 6,
            "d_ff": 2048,
            "dropout": 0.1,
            "max_seq_length": 512
        }
    
    @staticmethod
    def create_model(vocab_size: int, config_name: str = "small"):
        """Create a transformer model for language modeling."""
        if config_name == "small":
            config = ModelProvider.get_small_config()
        elif config_name == "medium":
            config = ModelProvider.get_medium_config()
        elif config_name == "large":
            config = ModelProvider.get_large_config()
        else:
            raise ValueError(f"Unknown config: {config_name}")
        
        # Create model
        model = EncoderDecoderTransformer(
            src_vocab_size=vocab_size,
            tgt_vocab_size=vocab_size,
            d_model=config["d_model"],
            num_heads=config["num_heads"],
            num_encoder_layers=config["num_encoder_layers"],
            num_decoder_layers=config["num_decoder_layers"],
            d_ff=config["d_ff"],
            dropout=config["dropout"],
            max_seq_length=config["max_seq_length"],
            positional_encoding="sinusoidal",
            share_embeddings=True
        )
        
        return model

def train_language_model(
    tokenizer_path: str,
    train_texts: List[str],
    val_texts: List[str],
    model_config: str = "small",
    batch_size: int = 16,
    max_seq_length: int = 128,
    num_epochs: int = 5,
    learning_rate: float = 5e-5,
    save_dir: str = "models"
):
    """Train a language model on the given texts."""
    # Load tokenizer
    tokenizer = BPETokenizer.from_pretrained(tokenizer_path)
    print(f"Loaded tokenizer with vocabulary size: {tokenizer.vocab_size}")
    
    # Create dataloaders
    train_dataloader, val_dataloader = create_lm_dataloaders(
        texts=train_texts,
        tokenizer=tokenizer,
        batch_size=batch_size,
        max_length=max_seq_length,
        val_split=0.0  # We already have separate validation texts
    )
    
    # Create validation dataloader separately
    val_dataset = LanguageModelingDataset(
        texts=val_texts,
        tokenizer=tokenizer,
        max_length=max_seq_length,
        pad_idx=tokenizer.special_tokens["pad_token_idx"],
        bos_idx=tokenizer.special_tokens["bos_token_idx"],
        eos_idx=tokenizer.special_tokens["eos_token_idx"],
    )
    
    val_dataloader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=lambda batch: lm_collate_fn(batch, tokenizer.special_tokens["pad_token_idx"]),
        num_workers=2,    # Use multiple workers (but not too many)
        pin_memory=True,  # Use pinned memory
        prefetch_factor=2  # Prefetch batches
    )
    
    # Create model
    model = ModelProvider.create_model(
        vocab_size=tokenizer.vocab_size,
        config_name=model_config
    )
    
    # Get device
    device = get_device()
    print(f"Using device: {device}")
    
    # Create trainer
    trainer = LanguageModelTrainer(
        model=model,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        learning_rate=learning_rate,
        warmup_steps=100,
        max_grad_norm=1.0,
        device=device,
        log_dir=os.path.join(save_dir, "logs")
    )
    
    # Train model
    print(f"Starting training for {num_epochs} epochs...")
    training_stats = trainer.train(
        num_epochs=num_epochs,
        save_dir=save_dir,
        model_name="language_model"
    )
    
    # Plot training curves
    trainer.plot_training_curves(save_path=os.path.join(save_dir, "training_curves.png"))
    
    # Save final model
    trainer.save_model(os.path.join(save_dir, "language_model_final.pt"))
    
    return model, training_stats

def lm_collate_fn(batch, pad_idx):
    """Collate function for language modeling batches."""
    # Get max sequence length in this batch
    max_input_length = max(example["input_ids"].size(0) for example in batch)
    max_label_length = max(example["labels"].size(0) for example in batch)
    
    # Initialize padded tensors
    input_ids = torch.full((len(batch), max_input_length), pad_idx, dtype=torch.long)
    labels = torch.full((len(batch), max_label_length), -100, dtype=torch.long)  # -100 is ignored in loss
    attention_mask = torch.zeros((len(batch), max_input_length), dtype=torch.bool)
    
    # Fill tensors with data
    for i, example in enumerate(batch):
        input_seq_len = example["input_ids"].size(0)
        label_seq_len = example["labels"].size(0)
        
        input_ids[i, :input_seq_len] = example["input_ids"]
        labels[i, :label_seq_len] = example["labels"]
        attention_mask[i, :input_seq_len] = 1  # 1 means attended to, 0 means masked
    
    return {
        "input_ids": input_ids,
        "labels": labels,
        "attention_mask": attention_mask
    }

def evaluate_and_generate(
    model_path: str,
    tokenizer_path: str,
    test_texts: List[str],
    prompts: List[str],
    save_dir: str = "evaluation"
):
    """Evaluate a trained model and generate text from prompts."""
    # Load tokenizer
    tokenizer = BPETokenizer.from_pretrained(tokenizer_path)
    
    # Get device
    device = get_device()
    
    # Create model
    model = ModelProvider.create_model(
        vocab_size=tokenizer.vocab_size,
        config_name="small"  # This will be overridden by loaded weights
    )
    
    # Load model weights
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    
    # Create generator
    generator = TextGenerator(
        model=model,
        tokenizer=tokenizer,
        device=device
    )
    
    # Create evaluator
    evaluator = LanguageModelEvaluator(
        model=model,
        tokenizer=tokenizer,
        device=device
    )
    
    # Create save directory
    os.makedirs(save_dir, exist_ok=True)
    
    # Evaluate on test texts
    print(f"Evaluating on {len(test_texts)} test texts...")
    eval_results = evaluator.evaluate_on_dataset(
        texts=test_texts[:100],  # Limit to 100 for efficiency
        save_path=os.path.join(save_dir, "evaluation_results.json")
    )
    
    # Plot perplexity distribution
    evaluator.plot_perplexity_distribution(
        perplexities=eval_results["per_text_perplexity"],
        save_path=os.path.join(save_dir, "perplexity_distribution.png")
    )
    
    # Print evaluation summary
    print("\nEvaluation Results:")
    print(f"Average Perplexity: {eval_results['average_perplexity']:.2f}")
    print(f"Median Perplexity: {eval_results['median_perplexity']:.2f}")
    print(f"Min Perplexity: {eval_results['min_perplexity']:.2f}")
    print(f"Max Perplexity: {eval_results['max_perplexity']:.2f}")
    
    # Generate text from prompts
    print("\nGenerating text from prompts:")
    generated_texts = []
    
    for i, prompt in enumerate(prompts):
        print(f"\nPrompt {i+1}: {prompt}")
        
        # Generate with different settings
        greedy_text = generator.generate(
            prompt=prompt,
            max_new_tokens=50,
            do_sample=False,
            num_return_sequences=1
        )
        
        temp_text = generator.generate(
            prompt=prompt,
            max_new_tokens=50,
            temperature=0.7,
            do_sample=True,
            num_return_sequences=1
        )
        
        nucleus_text = generator.generate(
            prompt=prompt,
            max_new_tokens=50,
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
            num_return_sequences=1
        )
        
        # Print generation results
        print(f"Greedy: {greedy_text[0]}")
        print(f"Temperature (0.7): {temp_text[0]}")
        print(f"Nucleus (T=0.7, P=0.9): {nucleus_text[0]}")
        
        # Analyze token probabilities for the best generation
        token_analysis = evaluator.analyze_token_probabilities(prompt)
        
        # Store results
        generated_texts.append({
            "prompt": prompt,
            "greedy": greedy_text[0],
            "temperature": temp_text[0],
            "nucleus": nucleus_text[0],
            "token_analysis": token_analysis
        })
    
    # Save generation results
    with open(os.path.join(save_dir, "generation_results.json"), "w") as f:
        import json
        json.dump(generated_texts, f, indent=2)
    
    # Visualize attention for a sample prompt
    if prompts:
        attention_dir = os.path.join(save_dir, "attention_visualizations")
        os.makedirs(attention_dir, exist_ok=True)
        
        # Visualize attention patterns for the first prompt
        evaluator.visualize_attention_patterns(
            text=prompts[0],
            save_dir=attention_dir
        )
    
    return generated_texts, eval_results

def main():
    parser = argparse.ArgumentParser(description="Language Model Training and Evaluation")
    
    # Dataset options
    parser.add_argument("--dataset", type=str, default="synthetic",
                      choices=["synthetic", "wikitext", "custom"],
                      help="Dataset to use for training")
    parser.add_argument("--data_dir", type=str, default="data",
                      help="Directory for data storage")
    parser.add_argument("--custom_data_dir", type=str, default=None,
                      help="Directory containing custom text files")
    
    # Tokenizer options
    parser.add_argument("--tokenizer_path", type=str, default="models/tokenizers/en",
                      help="Path to the tokenizer")
    
    # Model options
    parser.add_argument("--model_config", type=str, default="small",
                      choices=["small", "medium", "large"],
                      help="Model size configuration")
    
    # Training options
    parser.add_argument("--batch_size", type=int, default=16,
                      help="Training batch size")
    parser.add_argument("--max_seq_length", type=int, default=128,
                      help="Maximum sequence length")
    parser.add_argument("--num_epochs", type=int, default=5,
                      help="Number of training epochs")
    parser.add_argument("--learning_rate", type=float, default=5e-5,
                      help="Learning rate")
    
    # Output options
    parser.add_argument("--save_dir", type=str, default="models/language_model",
                      help="Directory to save model and results")
    parser.add_argument("--eval_only", action="store_true",
                      help="Only run evaluation on a trained model")
    parser.add_argument("--model_path", type=str, default=None,
                      help="Path to trained model for evaluation")
    
    # Seed
    parser.add_argument("--seed", type=int, default=42,
                      help="Random seed")
    
    args = parser.parse_args()
    
    # Set random seed
    set_seed(args.seed)
    
    # Create save directory
    os.makedirs(args.save_dir, exist_ok=True)
    
    # Load or create dataset
    if args.dataset == "synthetic":
        print("Using synthetic dataset")
        train_texts = DatasetProvider.get_synthetic_dataset(10000, 50)
        val_texts = DatasetProvider.get_synthetic_dataset(1000, 50)
        test_texts = DatasetProvider.get_synthetic_dataset(100, 50)
    elif args.dataset == "wikitext":
        print("Loading WikiText dataset")
        dataset = DatasetProvider.download_wikitext(args.data_dir)
        train_texts = dataset["train"]
        val_texts = dataset["validation"]
        test_texts = dataset["test"]
    elif args.dataset == "custom" and args.custom_data_dir:
        print(f"Loading custom dataset from {args.custom_data_dir}")
        europarl_dataset = EuroparlDataset(data_dir=args.custom_data_dir)
        all_texts = europarl_dataset.src_data
        random.shuffle(all_texts)

        # Define split sizes
        val_size = len(all_texts) // 10
        train_size = len(all_texts) - 2 * val_size

        # Split the data
        train_texts = all_texts[:train_size]
        val_texts = all_texts[train_size:train_size + val_size]
        test_texts = all_texts[train_size + val_size:]
    else:
        raise ValueError("Invalid dataset choice or missing custom_data_dir")
    
    # Sample test prompts
    test_prompts = [
        "Once upon a time",
        "The most important thing to remember",
        "In the field of artificial intelligence",
        "The transformer architecture",
    ]
    
    # Either train or evaluate
    if args.eval_only and args.model_path:
        # Evaluate existing model
        print(f"Evaluating model from {args.model_path}")
        evaluate_and_generate(
            model_path=args.model_path,
            tokenizer_path=args.tokenizer_path,
            test_texts=test_texts,
            prompts=test_prompts,
            save_dir=os.path.join(args.save_dir, "evaluation")
        )
    else:
        # Train model
        print("Training new model")
        model, stats = train_language_model(
            tokenizer_path=args.tokenizer_path,
            train_texts=train_texts,
            val_texts=val_texts,
            model_config=args.model_config,
            batch_size=args.batch_size,
            max_seq_length=args.max_seq_length,
            num_epochs=args.num_epochs,
            learning_rate=args.learning_rate,
            save_dir=args.save_dir
        )
        
        # Evaluate trained model
        print("Evaluating trained model")
        evaluate_and_generate(
            model_path=os.path.join(args.save_dir, "language_model_final.pt"),
            tokenizer_path=args.tokenizer_path,
            test_texts=test_texts,
            prompts=test_prompts,
            save_dir=os.path.join(args.save_dir, "evaluation")
        )

if __name__ == "__main__":
    main()