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

# Add parent directory to path to import local modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models.transformer import EncoderDecoderTransformer
from src.data.sequence_data import TransformerDataModule
from src.training.transformer_trainer import TransformerTrainer
from src.training.transformer_utils import create_padding_mask, create_causal_mask
from src.data.tokenization import BPETokenizer
import unicodedata

class IWSLTDataset:
    """Dataset class for the IWSLT translation dataset with enhanced fallback to synthetic data."""
    
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


def main():
    # Set random seed for reproducibility
    torch.manual_seed(42)
    random.seed(42)
    np.random.seed(42)

    # Set device
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load dataset - can use either IWSLT or Europarl
    print("Loading translation dataset...")
    train_dataset = IWSLTDataset(
        src_lang="de",  # German source
        tgt_lang="en",  # English target
        year="2016",
        split="train",
        max_examples=50000
    )
    
    val_dataset = IWSLTDataset(
        src_lang="de",  # German source
        tgt_lang="en",  # English target
        year="2016",
        split="valid",
        max_examples=10000
    )
    
    # Load the pre-trained BPE tokenizers
    print("Loading pre-trained BPE tokenizers...")
    de_tokenizer = BPETokenizer.from_pretrained("models/tokenizers/de")
    en_tokenizer = BPETokenizer.from_pretrained("models/tokenizers/en")
    print(f"Loaded German tokenizer with vocab size: {de_tokenizer.vocab_size}")
    print(f"Loaded English tokenizer with vocab size: {en_tokenizer.vocab_size}")

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
        batch_size=64,
        max_src_len=100,
        max_tgt_len=100,
        pad_idx=src_pad_idx,
        bos_idx=src_bos_idx,
        eos_idx=src_eos_idx,
        val_split=0.0,
        shuffle=True,
        num_workers=4
    )
    
    # Create a separate validation data module
    val_data_module = TransformerDataModule(
        source_sequences=val_src_sequences,
        target_sequences=val_tgt_sequences,
        batch_size=64,
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
        d_model=512,
        num_heads=8,
        num_encoder_layers=6,
        num_decoder_layers=6,
        d_ff=2048,
        dropout=0.1,
        max_seq_length=100,
        positional_encoding="sinusoidal",
        share_embeddings=False,
    )
    
    model.to(device)
    
    # Create trainer
    print("Creating trainer...")
    trainer = TransformerTrainer(
        model=model,
        train_dataloader=data_module.get_train_dataloader(),
        val_dataloader=val_data_module.get_train_dataloader(),
        pad_idx=src_pad_idx,
        lr=0.0001,
        warmup_steps=4000,
        label_smoothing=0.1,
        clip_grad=1.0,
        early_stopping_patience=10,
        device=device,
        track_perplexity=True
    )
    
    # Create save directory
    os.makedirs("models", exist_ok=True)
    
    # Train model
    print("Training model...")
    history = trainer.train(epochs=30, save_path="models/de_en_translation")
    
    # Plot training history
    trainer.plot_training_history()
    plt.savefig("de_en_training_history.png")
    plt.close()
    
    # Plot learning rate schedule
    trainer.plot_learning_rate()
    plt.savefig("de_en_learning_rate_schedule.png")
    plt.close()
    
    # Test translation on some examples
    test_sentences = [
        # Basic sentences
        ("Hallo, wie geht es dir?", "Hello, how are you?"),
        ("Ich lerne maschinelle Übersetzung.", "I am learning machine translation."),
        ("Das Wetter ist heute schön.", "The weather is nice today."),
        ("Ich komme aus Deutschland.", "I come from Germany."),
        
        # Medium complexity
        ("Die künstliche Intelligenz verändert unsere Welt.", "Artificial intelligence is changing our world."),
        ("Transformer-Modelle haben die natürliche Sprachverarbeitung revolutioniert.", 
         "Transformer models have revolutionized natural language processing."),
        ("Der Zug fährt um 15 Uhr vom Hauptbahnhof ab.", "The train departs from the main station at 3 PM."),
        ("Wir sollten mehr Wert auf Nachhaltigkeit legen.", "We should place more value on sustainability."),
        
        # Complex sentences with subordinate clauses
        ("Ich glaube, dass maschinelles Lernen in Zukunft noch wichtiger wird.", 
         "I believe that machine learning will become even more important in the future."),
        ("Obwohl es regnet, möchte ich spazieren gehen.", "Although it's raining, I want to go for a walk."),
        ("Nachdem wir das Projekt abgeschlossen hatten, gingen wir alle zusammen essen.", 
         "After we had completed the project, we all went to eat together."),
        
        # Questions
        ("Wann wurde der Transformer-Architekt veröffentlicht?", "When was the Transformer architecture published?"),
        ("Wie funktioniert ein neuronales Netzwerk?", "How does a neural network work?"),
        ("Warum ist Datenschutz so wichtig für KI-Systeme?", "Why is data protection so important for AI systems?"),
        
        # Technical content
        ("Die Aufmerksamkeitsmechanismen ermöglichen es dem Modell, auf relevante Informationen zu fokussieren.", 
         "The attention mechanisms allow the model to focus on relevant information."),
        ("Gradientenabstieg ist ein Optimierungsalgorithmus zum Trainieren neuronaler Netze.", 
         "Gradient descent is an optimization algorithm for training neural networks."),
        ("Tokenisierung ist der erste Schritt bei der Verarbeitung von Texteingaben.", 
         "Tokenization is the first step in processing text inputs."),
        
        # Idiomatic expressions
        ("Das ist ein Kinderspiel.", "That is child's play (easy)."),
        ("Es ist mir Wurst.", "I don't care."),
        ("Ich verstehe nur Bahnhof.", "It's all Greek to me."),
        
        # Long sentences
        ("Die Implementierung eines maschinellen Übersetzungssystems erfordert tiefes Verständnis von Sprachmodellen, Aufmerksamkeitsmechanismen und Tokenisierungsalgorithmen.", 
         "Implementing a machine translation system requires a deep understanding of language models, attention mechanisms, and tokenization algorithms."),
        ("Der Europarl-Datensatz enthält Übersetzungen von Debatten des Europäischen Parlaments und wird häufig zum Trainieren von Übersetzungsmodellen verwendet.", 
         "The Europarl dataset contains translations of European Parliament debates and is frequently used to train translation models."),
        
        # Sentences with numbers and named entities
        ("Berlin ist die Hauptstadt Deutschlands mit etwa 3,7 Millionen Einwohnern.", 
         "Berlin is the capital of Germany with about 3.7 million inhabitants."),
        ("Die Konferenz für maschinelles Lernen findet am 15. Mai 2023 in München statt.", 
         "The machine learning conference takes place on May 15, 2023, in Munich."),
        
        # Sentences with compound words (challenging for BPE)
        ("Datenschutzgrundverordnung ist ein langes deutsches Wort.", 
         "General Data Protection Regulation is a long German word."),
        ("Maschinelles Lernen und Computerlinguistik sind verwandte Forschungsgebiete.", 
         "Machine learning and computational linguistics are related research fields."),
        
        # Sentences with different tenses
        ("Ich habe gestern ein neues Buch gekauft.", "I bought a new book yesterday."),
        ("Sie werden morgen nach Berlin reisen.", "They will travel to Berlin tomorrow."),
        ("Wir hatten das Problem bereits gelöst, bevor der Chef davon erfuhr.", 
         "We had already solved the problem before the boss found out about it."),
        
        # Domain-specific sentence (IT/programming)
        ("Die Funktion gibt einen Fehler zurück, wenn die Eingabe ungültig ist.", 
         "The function returns an error if the input is invalid.")
    ]
    def translate(text, max_len=100):
        """
        Translate German text to English using BPE tokenization.
        
        Args:
            text: German text to translate
            max_len: Maximum length of generated translation
            
        Returns:
            English translation
        """
        model.eval()
        
        # Tokenize source text with BPE
        src_ids = de_tokenizer.encode(text)
        src_ids = [src_bos_idx] + src_ids + [src_eos_idx]
        src_tensor = torch.tensor([src_ids], dtype=torch.long).to(device)
        
        # Create source mask
        src_mask = create_padding_mask(src_tensor, pad_idx=src_pad_idx)
        
        # Set start token
        tgt = torch.tensor([[tgt_bos_idx]], dtype=torch.long).to(device)
        
        # Generate translation auto-regressively
        for i in range(max_len - 1):
            # Create target mask
            tgt_mask = create_causal_mask(tgt.size(1), device)
            
            # Predict next token
            output = model.decode(tgt, memory, tgt_mask=tgt_mask)
            next_token_logits = output[:, -1, :]
            next_token = next_token_logits.argmax(dim=-1, keepdim=True)
            
            # Append to output sequence
            tgt = torch.cat([tgt, next_token], dim=1)
            
            # Stop if end token is generated
            if next_token.item() == tgt_eos_idx:
                break
        
        # Convert token indices to words (skip BOS, include EOS)
        tgt_indices = tgt[0].cpu().tolist()
        
        # Decode using BPE tokenizer
        # Skip first token (BOS) and stop at EOS if present
        decoded_indices = tgt_indices[1:]
        if tgt_eos_idx in decoded_indices:
            decoded_indices = decoded_indices[:decoded_indices.index(tgt_eos_idx)]
        
        translation = en_tokenizer.decode(decoded_indices)
        
        return translation
    
    print("\n=== Testing Translation ===")
    for i, (source, reference) in enumerate(test_sentences):
        # Translate the sentence
        generated = translate(source)
        
        # Print the results
        print(f"Example {i+1}:")
        print(f"Source:     {source}")
        print(f"Reference:  {reference}")
        print(f"Generated:  {generated}")
        print("-" * 80)  # Separator for better readability
    return history
    

if __name__ == "__main__":
    history = main()

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