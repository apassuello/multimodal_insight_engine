# examples/translation_example_fixed.py
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict, Tuple
import os
import sys
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import requests
import random
from tqdm import tqdm
import re
from collections import Counter
import argparse

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models.transformer import EncoderDecoderTransformer
from src.data.sequence_data import TransformerDataModule
from src.training.transformer_trainer import TransformerTrainer
from src.training.transformer_utils import create_padding_mask, create_causal_mask
from src.data.tokenization import OptimizedBPETokenizer
import unicodedata
from src.optimization.mixed_precision import MixedPrecisionConverter
from src.data.europarl_dataset import EuroparlDataset
from src.data.opensubtitles_dataset import OpenSubtitlesDataset
from debug_transformer import attach_debugger_to_trainer, debug_sample_batch

class IWSLTDataset:
    """Dataset class for the IWSLT translation dataset with enhanced fallback to synthetic data."""
    
    def __init__(self, 
                 src_lang="de", 
                 tgt_lang="en", 
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
        files_exist = os.path.exists(self.src_file) and os.path.exists(self.tgt_file)
        if files_exist:
            # Verify the files are not empty and contain adequate data
            try:
                with open(self.src_file, 'r', encoding='utf-8') as f:
                    src_content = f.read().strip()
                with open(self.tgt_file, 'r', encoding='utf-8') as f:
                    tgt_content = f.read().strip()
                    
                # Check if we have enough data
                src_lines = src_content.count('\n') + 1
                tgt_lines = tgt_content.count('\n') + 1
                
                min_examples = 500000 if self.split == "train" else 100000
                
                if src_lines >= min_examples and tgt_lines >= min_examples:
                    print(f"IWSLT {self.year} {self.src_lang}-{self.tgt_lang} {self.split} data already exists with {src_lines} examples")
                    return
                else:
                    print(f"IWSLT files exist but only contain {src_lines} examples. Need at least {min_examples}. Recreating...")
            except Exception as e:
                print(f"Error reading existing files: {e}. Recreating...")
        
        # Attempt to download from official sources
        try:
            self._download_from_huggingface()
        except Exception as e:
            print(f"Error downloading from HuggingFace: {e}")
            try:
                self._download_from_official_source()
            except Exception as e2:
                print(f"Error downloading from official source: {e2}")
                self._create_large_synthetic_dataset()
    
    def _download_from_huggingface(self):
        """Download dataset from HuggingFace datasets."""
        print(f"Downloading IWSLT {self.year} {self.src_lang}-{self.tgt_lang} {self.split} data from HuggingFace...")
        
        from datasets import load_dataset
        
        # Load the IWSLT dataset
        dataset = load_dataset("iwslt2017", f"{self.src_lang}-{self.tgt_lang}", split=self.split)
        
        # Extract source and target texts
        src_texts = [example['translation'][self.src_lang] for example in dataset]
        tgt_texts = [example['translation'][self.tgt_lang] for example in dataset]
        
        # Save to files
        with open(self.src_file, 'w', encoding='utf-8') as f:
            f.write('\n'.join(src_texts))
        
        with open(self.tgt_file, 'w', encoding='utf-8') as f:
            f.write('\n'.join(tgt_texts))
            
        print(f"Downloaded {len(src_texts)} examples from HuggingFace")
    
    def _download_from_official_source(self):
        """Attempt to download from the official IWSLT website."""
        print(f"Downloading IWSLT {self.year} {self.src_lang}-{self.tgt_lang} {self.split} data from official source...")
        
        import requests
        import tarfile
        import io
        
        # This is a simplified example - the actual URL structure would need to be adjusted
        # based on the specific IWSLT release
        base_url = f"https://wit3.fbk.eu/archive/{self.year}/texts/{self.src_lang}/{self.tgt_lang}"
        tarball_url = f"{base_url}/{self.src_lang}-{self.tgt_lang}.tgz"
        
        try:
            # Download tarball
            response = requests.get(tarball_url)
            response.raise_for_status()
            
            # Extract from tarball
            with tarfile.open(fileobj=io.BytesIO(response.content)) as tar:
                # Extract relevant files (this would need to be adjusted based on the archive structure)
                src_file_in_tar = f"train.{self.src_lang}"
                tgt_file_in_tar = f"train.{self.tgt_lang}"
                
                # Extract and save files
                with open(self.src_file, 'wb') as f:
                    f.write(tar.extractfile(src_file_in_tar).read())
                
                with open(self.tgt_file, 'wb') as f:
                    f.write(tar.extractfile(tgt_file_in_tar).read())
                
            print("Downloaded from official source")
            
        except Exception as e:
            print(f"Official source download failed: {e}")
            raise
    
    def _create_large_synthetic_dataset(self):
        """Create a large synthetic dataset when downloads fail."""
        print("Creating large synthetic dataset instead...")
        
        # Base examples that will be modified to create variations
        base_examples_en = [
            "Hello, how are you?",
            "I am learning machine translation.",
            "Transformers are powerful models for natural language processing.",
            "This is an example of English to German translation.",
            "The weather is nice today.",
            "I love programming and artificial intelligence.",
            "Neural networks can learn complex patterns from data.",
            "Please translate this sentence to German.",
            "The cat is sleeping on the couch.",
            "We are working on a challenging project.",
            "Machine learning is transforming our world.",
            "The transformer architecture revolutionized natural language processing.",
            "Deep learning models require lots of data to train effectively.",
            "Attention mechanisms help models focus on important parts of the input.",
            "Transfer learning reduces the need for large datasets in some cases.",
            "Python is a popular programming language for machine learning.",
            "The model generates text based on the input it receives.",
            "The translation quality depends on the training data.",
            "Neural machine translation has improved significantly in recent years.",
            "Large language models can understand and generate human-like text.",
        ]
        
        base_examples_de = [
            "Hallo, wie geht es dir?",
            "Ich lerne maschinelle Übersetzung.",
            "Transformer sind leistungsstarke Modelle für die Verarbeitung natürlicher Sprache.",
            "Dies ist ein Beispiel für die Übersetzung von Englisch nach Deutsch.",
            "Das Wetter ist heute schön.",
            "Ich liebe Programmierung und künstliche Intelligenz.",
            "Neuronale Netze können komplexe Muster aus Daten lernen.",
            "Bitte übersetze diesen Satz ins Deutsche.",
            "Die Katze schläft auf dem Sofa.",
            "Wir arbeiten an einem anspruchsvollen Projekt.",
            "Maschinelles Lernen verändert unsere Welt.",
            "Die Transformer-Architektur revolutionierte die Verarbeitung natürlicher Sprache.",
            "Deep-Learning-Modelle benötigen viele Daten, um effektiv zu trainieren.",
            "Aufmerksamkeitsmechanismen helfen Modellen, sich auf wichtige Teile der Eingabe zu konzentrieren.",
            "Transfer Learning reduziert in einigen Fällen den Bedarf an großen Datensätzen.",
            "Python ist eine beliebte Programmiersprache für maschinelles Lernen.",
            "Das Modell generiert Text basierend auf der Eingabe, die es erhält.",
            "Die Übersetzungsqualität hängt von den Trainingsdaten ab.",
            "Neuronale maschinelle Übersetzung hat sich in den letzten Jahren deutlich verbessert.",
            "Große Sprachmodelle können menschenähnlichen Text verstehen und generieren.",
        ]
        
        # Additional vocabulary to use in generating variations
        subjects_en = [
            "The model", "The system", "The algorithm", "The network", "The approach", 
            "The method", "The user", "The programmer", "The developer", "The researcher",
            "The student", "The teacher", "The engineer", "The professor", "The scientist",
            "The translator", "The computer", "The machine", "The person", "The expert"
        ]
        
        subjects_de = [
            "Das Modell", "Das System", "Der Algorithmus", "Das Netzwerk", "Der Ansatz",
            "Die Methode", "Der Benutzer", "Der Programmierer", "Der Entwickler", "Der Forscher",
            "Der Student", "Der Lehrer", "Der Ingenieur", "Der Professor", "Der Wissenschaftler",
            "Der Übersetzer", "Der Computer", "Die Maschine", "Die Person", "Der Experte"
        ]
        
        verbs_en = [
            "processes", "analyzes", "understands", "generates", "translates",
            "learns", "computes", "predicts", "transforms", "evaluates",
            "improves", "creates", "develops", "builds", "designs",
            "implements", "optimizes", "utilizes", "applies", "interprets"
        ]
        
        verbs_de = [
            "verarbeitet", "analysiert", "versteht", "generiert", "übersetzt",
            "lernt", "berechnet", "sagt voraus", "transformiert", "bewertet",
            "verbessert", "erstellt", "entwickelt", "baut", "gestaltet",
            "implementiert", "optimiert", "nutzt", "wendet an", "interpretiert"
        ]
        
        objects_en = [
            "the input data", "the text", "the sentences", "the language", "the translation",
            "the information", "the patterns", "the features", "the representations", "the embeddings",
            "the words", "the sequences", "the documents", "the meanings", "the concepts",
            "the queries", "the results", "the output", "the model", "the system"
        ]
        
        objects_de = [
            "die Eingabedaten", "den Text", "die Sätze", "die Sprache", "die Übersetzung",
            "die Information", "die Muster", "die Merkmale", "die Darstellungen", "die Einbettungen",
            "die Wörter", "die Sequenzen", "die Dokumente", "die Bedeutungen", "die Konzepte",
            "die Anfragen", "die Ergebnisse", "die Ausgabe", "das Modell", "das System"
        ]
        
        adverbs_en = [
            "efficiently", "accurately", "rapidly", "effectively", "automatically",
            "precisely", "correctly", "quickly", "reliably", "consistently",
            "intelligently", "appropriately", "successfully", "carefully", "thoroughly",
            "easily", "directly", "clearly", "properly", "immediately"
        ]
        
        adverbs_de = [
            "effizient", "genau", "schnell", "effektiv", "automatisch",
            "präzise", "korrekt", "rasch", "zuverlässig", "konsistent",
            "intelligent", "angemessen", "erfolgreich", "sorgfältig", "gründlich",
            "leicht", "direkt", "klar", "ordnungsgemäß", "sofort"
        ]
        
        # Generate variations for both source and target
        import random
        random.seed(42)  # For reproducibility
        
        # Target number of examples
        target_examples = 50000 if self.split == "train" else 10000
        
        # Generate the synthetic dataset
        en_sentences = []
        de_sentences = []
        
        # First, add all base examples
        en_sentences.extend(base_examples_en)
        de_sentences.extend(base_examples_de)
        
        # Generate sentence variations until we reach target size
        pattern_templates_en = [
            "{subject} {verb} {object} {adverb}.",
            "{adverb}, {subject} {verb} {object}.",
            "{subject} {adverb} {verb} {object}.",
            "When {subject} {verb} {object}, it does so {adverb}.",
            "The {object} is {adverb} {verb}ed by {subject}.",
            "{subject} can {verb} {object} more {adverb}.",
            "To {verb} {object}, {subject} proceeds {adverb}.",
            "It is important that {subject} {verb} {object} {adverb}.",
            "{subject} should {verb} {object} {adverb} to improve results.",
            "The ability to {verb} {object} {adverb} is crucial for {subject}."
        ]
        
        pattern_templates_de = [
            "{subject} {verb} {object} {adverb}.",
            "{adverb} {verb} {subject} {object}.",
            "{subject} {verb} {adverb} {object}.",
            "Wenn {subject} {object} {verb}, tut es dies {adverb}.",
            "Das {object} wird {adverb} von {subject} {verb}.",
            "{subject} kann {object} {adverb}er {verb}.",
            "Um {object} zu {verb}, geht {subject} {adverb} vor.",
            "Es ist wichtig, dass {subject} {object} {adverb} {verb}.",
            "{subject} sollte {object} {adverb} {verb}, um Ergebnisse zu verbessern.",
            "Die Fähigkeit, {object} {adverb} zu {verb}, ist entscheidend für {subject}."
        ]
        
        while len(en_sentences) < target_examples:
            # Choose a random template
            template_idx = random.randrange(0, len(pattern_templates_en))
            template_en = pattern_templates_en[template_idx]
            template_de = pattern_templates_de[template_idx]
            
            # Fill in the template with random words
            subject_idx = random.randrange(0, len(subjects_en))
            verb_idx = random.randrange(0, len(verbs_en))
            object_idx = random.randrange(0, len(objects_en))
            adverb_idx = random.randrange(0, len(adverbs_en))
            
            # Create sentences
            en_sentence = template_en.format(
                subject=subjects_en[subject_idx],
                verb=verbs_en[verb_idx],
                object=objects_en[object_idx],
                adverb=adverbs_en[adverb_idx]
            )
            
            de_sentence = template_de.format(
                subject=subjects_de[subject_idx],
                verb=verbs_de[verb_idx],
                object=objects_de[object_idx],
                adverb=adverbs_de[adverb_idx]
            )
            
            # Add to dataset
            en_sentences.append(en_sentence)
            de_sentences.append(de_sentence)
        
        # Additional variations from base examples
        if len(en_sentences) < target_examples:
            for i in range(min(len(base_examples_en), (target_examples - len(en_sentences)) // 10)):
                # Generate variations of each base example
                for j in range(10):  # 10 variations per base example
                    # Create a variation by adding adjectives, changing tense, etc.
                    base_en = base_examples_en[i]
                    base_de = base_examples_de[i]
                    
                    # Simple variation: add a prefix
                    prefix_en = ["In my opinion, ", "I believe that ", "It's clear that ", 
                               "Experts say that ", "According to research, ",
                               "As we know, ", "Interestingly, ", "Consider this: ",
                               "To put it simply, ", "In technical terms, "]
                    
                    prefix_de = ["Meiner Meinung nach ", "Ich glaube, dass ", "Es ist klar, dass ", 
                               "Experten sagen, dass ", "Nach Forschungsergebnissen ",
                               "Wie wir wissen, ", "Interessanterweise ", "Betrachten Sie dies: ",
                               "Einfach ausgedrückt, ", "In technischer Hinsicht "]
                    
                    prefix_idx = random.randrange(0, len(prefix_en))
                    variation_en = prefix_en[prefix_idx] + base_en.lower()
                    variation_de = prefix_de[prefix_idx] + base_de.lower()
                    
                    en_sentences.append(variation_en)
                    de_sentences.append(variation_de)
        
        # Write to files
        with open(self.src_file, 'w', encoding='utf-8') as f:
            f.write('\n'.join(en_sentences))
        
        with open(self.tgt_file, 'w', encoding='utf-8') as f:
            f.write('\n'.join(de_sentences))
        
        print(f"Created synthetic dataset with {len(en_sentences)} examples")
    
    def load_data(self):
        """Load and preprocess the data."""
        # Read data files
        with open(self.src_file, 'r', encoding='utf-8') as f:
            src_data = f.read().strip().split('\n')
        
        with open(self.tgt_file, 'r', encoding='utf-8') as f:
            tgt_data = f.read().strip().split('\n')
        
        # Skip empty lines
        src_data = [line for line in src_data if line.strip()]
        tgt_data = [line for line in tgt_data if line.strip()]
        
        # Ensure same length
        min_len = min(len(src_data), len(tgt_data))
        if min_len < max(len(src_data), len(tgt_data)):
            print(f"Warning: Source and target data have different lengths. Truncating to {min_len} examples.")
            src_data = src_data[:min_len]
            tgt_data = tgt_data[:min_len]
        
        assert len(src_data) == len(tgt_data), "Source and target data must have same length"
        
        # Limit dataset size if specified
        if self.max_examples is not None and self.max_examples < len(src_data):
            # Use a fixed random seed for reproducibility
            random.seed(42)
            indices = random.sample(range(len(src_data)), self.max_examples)
            src_data = [src_data[i] for i in indices]
            tgt_data = [tgt_data[i] for i in indices]
        
        return src_data, tgt_data


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

def preprocess_data_with_bpe(dataset, de_tokenizer, en_tokenizer):
    """
    Preprocess the dataset for training using BPE tokenizers.
    
    Args:
        dataset: IWSLT dataset or EuroparlDataset
        de_tokenizer: German BPE tokenizer
        en_tokenizer: English BPE tokenizer
        
    Returns:
        Lists of tokenized source and target sequences
    """
    src_sequences = []
    tgt_sequences = []
    
    for src_text, tgt_text in zip(dataset.src_data, dataset.tgt_data):
        # Tokenize with BPE
        src_ids = de_tokenizer.encode(src_text)
        tgt_ids = en_tokenizer.encode(tgt_text)
        # Add special tokens
        src_ids = [de_tokenizer.special_tokens["bos_token_idx"]] + src_ids + [de_tokenizer.special_tokens["eos_token_idx"]]
        tgt_ids = [en_tokenizer.special_tokens["bos_token_idx"]] + tgt_ids + [en_tokenizer.special_tokens["eos_token_idx"]]
        
        src_sequences.append(src_ids)
        tgt_sequences.append(tgt_ids)
    
    return src_sequences, tgt_sequences

def count_parameters(model):
    """Count the number of trainable parameters in a model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def main(args):
    # Set random seed for reproducibility
    torch.manual_seed(42)
    random.seed(42)
    np.random.seed(42)

    # Set device
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load dataset based on user selection
    print(f"Loading {args.dataset.capitalize()} dataset...")
    
    if args.dataset == "europarl":
        # Load Europarl dataset
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
        # Load OpenSubtitles dataset
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
    
    # Load the pre-trained BPE tokenizers
    print("Loading pre-trained BPE tokenizers...")
    de_tokenizer = OptimizedBPETokenizer.from_pretrained(f"models/tokenizers/{args.src_lang}")
    en_tokenizer = OptimizedBPETokenizer.from_pretrained(f"models/tokenizers/{args.tgt_lang}")
    print(f"Loaded {args.src_lang} tokenizer with vocab size: {de_tokenizer.vocab_size}")
    print(f"Loaded {args.tgt_lang} tokenizer with vocab size: {en_tokenizer.vocab_size}")

    # Get special token indices for the transformer
    src_pad_idx = de_tokenizer.special_tokens["pad_token_idx"]
    tgt_pad_idx = en_tokenizer.special_tokens["pad_token_idx"]
    src_bos_idx = de_tokenizer.special_tokens["bos_token_idx"]
    tgt_bos_idx = en_tokenizer.special_tokens["bos_token_idx"]
    src_eos_idx = de_tokenizer.special_tokens["eos_token_idx"]
    tgt_eos_idx = en_tokenizer.special_tokens["eos_token_idx"]
    
    # Preprocess data with BPE tokenizers
    print("Preprocessing training data...")
    train_src_sequences, train_tgt_sequences = preprocess_data_with_bpe(
        train_dataset, de_tokenizer, en_tokenizer
    )
    
    print("Preprocessing validation data...")
    val_src_sequences, val_tgt_sequences = preprocess_data_with_bpe(
        val_dataset, de_tokenizer, en_tokenizer
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
        shuffle=False,
        num_workers=4
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
        num_workers=4,
    )
    
    # Create transformer model using BPE vocabulary sizes
    print("Creating transformer model...")
    model = EncoderDecoderTransformer(
        src_vocab_size=de_tokenizer.vocab_size,
        tgt_vocab_size=en_tokenizer.vocab_size,
        d_model=args.d_model,
        num_heads=args.num_heads,
        num_encoder_layers=args.num_encoder_layers,
        num_decoder_layers=args.num_decoder_layers,
        d_ff=args.d_ff,
        dropout=args.dropout,
        max_seq_length=100,
        positional_encoding="sinusoidal",
        share_embeddings=False,
    )
    
    # Print model parameter count
    num_params = count_parameters(model)
    print(f"Model created with {num_params:,} trainable parameters")
    
    # Apply mixed precision if requested
    if args.use_mixed_precision:
        print("Using mixed precision training (can sometimes cause instability)")
        mp_converter = MixedPrecisionConverter(
            model=model,
            dtype=torch.float16,
            use_auto_cast=True
        )
        model = mp_converter.convert_to_mixed_precision()
        
        # Check if using mixed precision wrapper
        if hasattr(model, 'model'):
            print(f"Using mixed precision with {model.dtype} for computation (parameters stored in FP32)")
    else:
        print("Using full precision training (more stable but slower)")
        
    model.to(device)
    
    # After model creation
    for name, param in model.named_parameters():
        if 'weight' in name:
            print(f"{name}: mean={param.data.mean().item():.6f}, std={param.data.std().item():.6f}")
            if param.data.std().item() < 0.01 or param.data.std().item() > 1.0:
                print(f"Warning: Unusual initialization for {name}")
    
    # Create the trainer
    trainer = TransformerTrainer(
        model=model,
        train_dataloader=data_module.get_train_dataloader(),
        val_dataloader=val_data_module.get_val_dataloader() if val_data_module else None,
        pad_idx=src_pad_idx,
        lr=args.learning_rate,
        warmup_steps=args.warmup_steps,
        label_smoothing=args.label_smoothing,
        device=device,
        track_perplexity=True,
        use_gradient_scaling=args.use_gradient_scaling
    )
    
    # Add debugging - this will automatically attach to the trainer
    if args.debug:
        print("Enabling debugging features...")
        debugger = attach_debugger_to_trainer(
            trainer=trainer,
            src_tokenizer=de_tokenizer,
            tgt_tokenizer=en_tokenizer,
            print_every=args.debug_frequency
        )
        
        # Optionally debug a sample batch before training
        if args.debug_sample_batch:
            print("\n===== Debugging Sample Batch =====")
            # Get a sample batch from the training data
            sample_batch = next(iter(data_module.get_train_dataloader()))
            debug_info = debug_sample_batch(model, sample_batch, en_tokenizer)
            print("\nSample batch properties:")
            for key, info in debug_info.items():
                if key != "outputs":
                    print(f"  {key}: {info}")
            if "outputs" in debug_info:
                print(f"  outputs shape: {debug_info['outputs']['shape']}")
                print(f"  outputs range: {debug_info['outputs']['min']} to {debug_info['outputs']['max']}")
    
    # Create save directory
    os.makedirs("models", exist_ok=True)
    
    # Prefix for model file based on dataset and languages
    model_prefix = f"{args.dataset}_{args.src_lang}_{args.tgt_lang}"
    
    # Train model
    print("Training model...")
    trainer_history = trainer.train(epochs=args.epochs, save_path=f"models/{model_prefix}_translation")
    
    # Plot training history
    trainer.plot_training_history()
    plt.savefig(f"{model_prefix}_training_history.png")
    plt.close()
    
    # Plot learning rate schedule
    trainer.plot_learning_rate()
    plt.savefig(f"{model_prefix}_learning_rate_schedule.png")
    plt.close()
    
    return trainer_history

if __name__ == "__main__":
    # Parse arguments
    parser = argparse.ArgumentParser(description="Train a transformer model for machine translation")
    
    # Dataset options
    parser.add_argument("--dataset", type=str, choices=["europarl", "opensubtitles"], default="europarl",
                        help="Dataset to use for training")
    parser.add_argument("--max_train_examples", type=int, default=100000,
                        help="Maximum number of training examples to use")
    parser.add_argument("--max_val_examples", type=int, default=20000,
                        help="Maximum number of validation examples to use")
    parser.add_argument("--src_lang", type=str, default="de",
                        help="Source language code")
    parser.add_argument("--tgt_lang", type=str, default="en",
                        help="Target language code")
    
    # Model options
    parser.add_argument("--d_model", type=int, default=512,
                        help="Dimension of model embeddings")
    parser.add_argument("--num_heads", type=int, default=8,
                        help="Number of attention heads")
    parser.add_argument("--num_encoder_layers", type=int, default=4,
                        help="Number of encoder layers")
    parser.add_argument("--num_decoder_layers", type=int, default=4,
                        help="Number of decoder layers")
    parser.add_argument("--d_ff", type=int, default=2048,
                        help="Dimension of feed-forward network")
    parser.add_argument("--dropout", type=float, default=0.1,
                        help="Dropout rate")
    
    # Training options
    parser.add_argument("--batch_size", type=int, default=128,
                        help="Batch size for training")
    parser.add_argument("--learning_rate", type=float, default=0.0001,
                        help="Initial learning rate")
    parser.add_argument("--label_smoothing", type=float, default=0.1,
                        help="Label smoothing factor for training")
    parser.add_argument("--warmup_steps", type=int, default=4000,
                        help="Number of warmup steps for learning rate scheduler")
    parser.add_argument("--use_gradient_scaling", action="store_true",
                        help="Whether to use gradient scaling for mixed precision training")
    parser.add_argument("--use_mixed_precision", action="store_true",
                        help="Whether to use mixed precision training")
    parser.add_argument("--debug", action="store_true",
                        help="Enable debug mode with more verbose output")
    parser.add_argument("--debug_frequency", type=int, default=100,
                        help="How often to print debug information (in training steps)")
    parser.add_argument("--debug_sample_batch", action="store_true",
                        help="Debug a sample batch before training")
    parser.add_argument("--epochs", type=int, default=3,
                        help="Number of epochs to train for")
    
    args = parser.parse_args()
    
    # Set random seed for reproducibility
    torch.manual_seed(42)
    random.seed(42)
    np.random.seed(42)
    
    # Check for mixed precision conflicts
    if args.use_mixed_precision and args.use_gradient_scaling:
        print("Warning: Using both mixed precision and gradient scaling.")
        print("Gradient scaling is typically used alongside mixed precision.")
    
    # Ensure gradient scaling is enabled with mixed precision
    if args.use_mixed_precision and not args.use_gradient_scaling:
        print("Enabling gradient scaling since mixed precision is enabled.")
        args.use_gradient_scaling = True
    
    # Set up debugging
    debugger = None
    if args.debug:
        debugger = create_translation_debugger(print_every=args.debug_frequency)
    
    # Run the training process
    main(args)

    # Visualize Training History
    plt.figure(figsize=(15, 5))

    # Loss Subplot
    plt.subplot(1, 2, 1)
    plt.title('Training and Validation Loss')
    plt.plot(trainer_history['train_loss'], label='Training Loss')
    plt.plot(trainer_history['val_loss'], label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    # Perplexity Subplot
    plt.subplot(1, 2, 2)
    plt.title('Training and Validation Perplexity')
    plt.plot(trainer_history['train_ppl'], label='Training Perplexity')
    plt.plot(trainer_history['val_ppl'], label='Validation Perplexity')
    plt.xlabel('Epochs')
    plt.ylabel('Perplexity')
    plt.legend()

    plt.tight_layout()
    plt.savefig('training_history.png')
    plt.close()