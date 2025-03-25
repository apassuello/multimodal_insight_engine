import os
import sys
import torch
import random
import numpy as np
from typing import List, Dict, Tuple, Optional
from tqdm import tqdm
import matplotlib.pyplot as plt

# Add parent directory to path to import local modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import our tokenization modules
from src.data.tokenization import BPETokenizer
from src.data.opensubtitles_dataset import OpenSubtitlesDataset

def set_seed(seed=42):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def analyze_tokenization(
    tokenizer: BPETokenizer,
    texts: List[str],
    sample_size: int = 1000,
    lang: str = "unknown",
    device: str = "mps"
):
    """
    Analyze tokenization statistics.
    
    Args:
        tokenizer: The trained tokenizer
        texts: List of texts to analyze
        sample_size: Number of examples to analyze
        lang: Language being analyzed
        device: Device to use for tensor operations
    """
    # Sample texts
    if len(texts) > sample_size:
        texts = random.sample(texts, sample_size)
    
    # Collect statistics
    lengths = []
    oov_counts = []
    
    for text in tqdm(texts, desc=f"Analyzing {lang} tokenization"):
        tokens = tokenizer.encode(text)
        lengths.append(len(tokens))
        
        # Count OOV tokens
        oov_count = sum(1 for t in tokens if t >= tokenizer.vocab_size)
        oov_counts.append(oov_count)
    
    # Calculate statistics
    avg_length = np.mean(lengths)
    std_length = np.std(lengths)
    max_length = max(lengths)
    avg_oov = np.mean(oov_counts)
    
    print(f"\n{lang} Tokenization Statistics:")
    print(f"Average sequence length: {avg_length:.2f} ± {std_length:.2f}")
    print(f"Maximum sequence length: {max_length}")
    print(f"Average OOV tokens per sequence: {avg_oov:.2f}")
    
    # Plot length distribution
    plt.figure(figsize=(10, 5))
    plt.hist(lengths, bins=50, alpha=0.7)
    plt.title(f"{lang} Sequence Length Distribution")
    plt.xlabel("Length")
    plt.ylabel("Count")
    plt.savefig(f"tokenization_length_distribution_{lang.lower()}.png")
    plt.close()

def demonstrate_oov_handling(tokenizer: BPETokenizer, lang: str, device: str = "mps"):
    """Demonstrate how the tokenizer handles out-of-vocabulary words."""
    print(f"\n{lang} OOV Handling Demonstration:")
    
    # Test cases with potential OOV words
    if lang == "German":
        test_cases = [
            "Die künstliche Intelligenz verändert unsere Welt.",
            "Transformer-Modelle haben die natürliche Sprachverarbeitung revolutioniert.",
            "Der Zug fährt um 15 Uhr vom Hauptbahnhof ab.",
            "Wir sollten mehr Wert auf Nachhaltigkeit legen.",
            "Ich glaube, dass maschinelles Lernen in Zukunft noch wichtiger wird."
        ]
    else:  # English
        test_cases = [
            "The neural network architecture uses transformer blocks.",
            "Machine learning models require large datasets for training.",
            "Natural language processing is a challenging field.",
            "Deep learning has revolutionized artificial intelligence.",
            "The model uses attention mechanisms for translation."
        ]
    
    for text in test_cases:
        tokens = tokenizer.encode(text)
        decoded = tokenizer.decode(tokens)
        
        # Normalize text for comparison
        normalized_original = ''.join(text.lower().split())
        normalized_decoded = ''.join(decoded.lower().split())
        
        print(f"\nOriginal: {text}")
        print(f"Decoded:  {decoded}")
        print(f"Tokens:   {tokens}")
        print(f"Match:    {'✓' if normalized_original == normalized_decoded else '✗'}")
        if normalized_original != normalized_decoded:
            print(f"Normalized original: {normalized_original}")
            print(f"Normalized decoded:  {normalized_decoded}")

def verify_tokenizer(tokenizer: BPETokenizer, test_texts: List[str], lang: str):
    """
    Verify that a tokenizer works correctly by testing encode/decode operations.
    
    Args:
        tokenizer: The tokenizer to verify
        test_texts: List of test texts
        lang: Language name for logging
    """
    print(f"\nVerifying {lang} tokenizer...")
    
    # Test a few examples
    for i, text in enumerate(test_texts[:5]):
        # Encode and decode
        tokens = tokenizer.encode(text)
        decoded = tokenizer.decode(tokens)
        
        # Normalize text for comparison (remove spaces and convert to lowercase)
        normalized_original = ''.join(text.lower().split())
        normalized_decoded = ''.join(decoded.lower().split())
        
        # Check if encode/decode roundtrip works
        is_correct = normalized_original == normalized_decoded
        print(f"\nTest {i+1}:")
        print(f"Original: {text}")
        print(f"Decoded:  {decoded}")
        print(f"Tokens:   {tokens}")
        print(f"Match:    {'✓' if is_correct else '✗'}")
        if not is_correct:
            print("WARNING: Tokenizer roundtrip failed!")
            print(f"Normalized original: {normalized_original}")
            print(f"Normalized decoded:  {normalized_decoded}")
    
    # Check if merges were saved correctly
    print(f"\n{lang} tokenizer merges:")
    print(f"Number of merges: {len(tokenizer.merges)}")
    if tokenizer.merges:
        print("First 5 merges:", tokenizer.merges[:5])
    else:
        print("WARNING: No merges found!")

def main():
    """Train separate tokenizers for German and English using OpenSubtitles dataset."""
    # Set seed for reproducibility
    set_seed(42)
    
    # Set up device
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")
    
    print("Loading OpenSubtitles dataset...")
    dataset = OpenSubtitlesDataset(
        src_lang="de",
        tgt_lang="en",
        max_examples=15000000
    )
    
    print(f"Loaded {len(dataset.src_data)} parallel sentences")
    
    # Create save directory
    os.makedirs("models/tokenizers", exist_ok=True)
    
    # Train German tokenizer
    print("\nTraining German tokenizer...")
    de_tokenizer = BPETokenizer(num_merges=8000-256)
    de_tokenizer.train(
        texts=dataset.src_data,
        vocab_size=16000,
        min_frequency=2,
        show_progress=True,
    )
    print(f"Number of merges created: {len(de_tokenizer.merges)}")
    print("First few merges:", de_tokenizer.merges[:5])
    
    # Train English tokenizer
    print("\nTraining English tokenizer...")
    en_tokenizer = BPETokenizer(num_merges=8000-256)
    en_tokenizer.train(
        texts=dataset.tgt_data,
        vocab_size=16000,
        min_frequency=2,
        show_progress=True,
    )
    print(f"Number of merges created: {len(en_tokenizer.merges)}")
    print("First few merges:", en_tokenizer.merges[:5])
    
    # Save tokenizers
    print("\nSaving tokenizers...")
    de_tokenizer.save_pretrained("models/tokenizers/german")
    en_tokenizer.save_pretrained("models/tokenizers/english")
    print(f"Saved German tokenizer with vocab size: {de_tokenizer.vocab_size}")
    print(f"Saved English tokenizer with vocab size: {en_tokenizer.vocab_size}")
    
    # Verify saved tokenizers
    print("\nVerifying saved tokenizers...")
    
    # Load and verify German tokenizer
    print("\nLoading German tokenizer...")
    de_tokenizer_loaded = BPETokenizer.from_pretrained("models/tokenizers/german")
    verify_tokenizer(de_tokenizer_loaded, dataset.src_data, "German")
    
    # Load and verify English tokenizer
    print("\nLoading English tokenizer...")
    en_tokenizer_loaded = BPETokenizer.from_pretrained("models/tokenizers/english")
    verify_tokenizer(en_tokenizer_loaded, dataset.tgt_data, "English")
    
    # Analyze tokenization for both languages
    print("\nAnalyzing tokenization...")
    analyze_tokenization(
        tokenizer=de_tokenizer,
        texts=dataset.src_data,
        sample_size=1000,
        lang="German",
        device=str(device)
    )
    analyze_tokenization(
        tokenizer=en_tokenizer,
        texts=dataset.tgt_data,
        sample_size=1000,
        lang="English",
        device=str(device)
    )
    
    # Demonstrate OOV handling for both languages
    demonstrate_oov_handling(de_tokenizer, "German", str(device))
    demonstrate_oov_handling(en_tokenizer, "English", str(device))

if __name__ == "__main__":
    main() 