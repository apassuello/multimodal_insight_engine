# src/data/tokenization/bpe_tokenizer.py
from typing import List, Dict, Tuple, Optional, Set, Counter as CounterType
from collections import Counter
import re
import os
import json
from tqdm import tqdm

from .base_tokenizer import BaseTokenizer
from .vocabulary import Vocabulary
from .preprocessing import clean_text

class BPETokenizer(BaseTokenizer):
    """
    A Byte Pair Encoding (BPE) tokenizer.
    
    BPE is a subword tokenization algorithm that starts with characters and
    iteratively merges the most frequent pairs of adjacent tokens.
    """
    
    def __init__(
        self,
        vocab: Optional[Vocabulary] = None,
        merges: Optional[List[Tuple[str, str]]] = None,
        num_merges: int = 10000,
        lower_case: bool = True,
    ):
        """
        Initialize the BPE tokenizer.
        
        Args:
            vocab: Optional vocabulary to use
            merges: Optional list of merge operations
            num_merges: Maximum number of merge operations
            lower_case: Whether to convert text to lowercase
        """
        self.lower_case = lower_case
        self.num_merges = num_merges
        
        # Initialize vocabulary
        if vocab is None:
            self.vocab = Vocabulary()
        else:
            self.vocab = vocab
        
        # Initialize merge operations
        self.merges = merges or []
        self.merges_dict = {pair: i for i, pair in enumerate(self.merges)}
    
    def preprocess(self, text: str) -> str:
        """
        Preprocess text before tokenization.
        
        Args:
            text: Input text
            
        Returns:
            Preprocessed text
        """
        return clean_text(text, lower=self.lower_case)
    
    def train(
        self,
        texts: List[str],
        vocab_size: Optional[int] = None,
        min_frequency: int = 2,
        show_progress: bool = True,
    ) -> None:
        """
        Train the BPE tokenizer on a corpus of texts.
        
        Args:
            texts: List of training texts
            vocab_size: Target vocabulary size
            min_frequency: Minimum frequency for a token to be included
            show_progress: Whether to show a progress bar
        """
        # Calculate target vocabulary size and number of merges
        if vocab_size is None:
            vocab_size = self.num_merges + 256  # Base character vocab + merges
        
        # Preprocess texts
        processed_texts = [self.preprocess(text) for text in texts]
        
        # Initialize with character vocabulary
        word_freqs = Counter()
        for text in processed_texts:
            word_freqs.update(text.split())
        
        # Filter by frequency
        word_freqs = Counter({
            word: freq for word, freq in word_freqs.items()
            if freq >= min_frequency
        })
        
        # Initialize with characters
        vocab = set()
        for word in word_freqs:
            for char in word:
                vocab.add(char)
        
        # Initialize vocabulary with characters
        self.vocab = Vocabulary(tokens=list(vocab))
        
        # Initialize word splits (each word is a list of characters)
        splits = {word: list(word) for word in word_freqs}
        
        # Perform BPE training
        merges = []
        vocab_size_target = vocab_size - len(vocab)
        
        iterator = range(min(vocab_size_target, self.num_merges))
        if show_progress:
            iterator = tqdm(iterator, desc="Training BPE")
        
        for _ in iterator:
            # Count pairs
            pair_freqs = Counter()
            for word, freq in word_freqs.items():
                word_pieces = splits[word]
                for i in range(len(word_pieces) - 1):
                    pair = (word_pieces[i], word_pieces[i + 1])
                    pair_freqs[pair] += freq
            
            # Find most frequent pair
            if not pair_freqs:
                break
                
            best_pair = max(pair_freqs, key=pair_freqs.get)
            merges.append(best_pair)
            
            # Create new token for this pair
            new_token = ''.join(best_pair)
            self.vocab.add_token(new_token)
            
            # Update word splits
            new_splits = {}
            for word, word_pieces in splits.items():
                new_pieces = []
                i = 0
                while i < len(word_pieces):
                    if (i < len(word_pieces) - 1 and 
                        (word_pieces[i], word_pieces[i + 1]) == best_pair):
                        new_pieces.append(new_token)
                        i += 2
                    else:
                        new_pieces.append(word_pieces[i])
                        i += 1
                new_splits[word] = new_pieces
            
            splits = new_splits
        
        # Store merges
        self.merges = merges
        self.merges_dict = {pair: i for i, pair in enumerate(self.merges)}
    
    def _tokenize_word(self, word: str) -> List[str]:
        """
        Tokenize a single word using BPE.
        
        Args:
            word: The word to tokenize
            
        Returns:
            List of BPE tokens
        """
        # Start with characters
        pieces = list(word)
        
        # Apply merges
        while len(pieces) > 1:
            # Find the best merge
            best_idx = -1
            best_rank = float('inf')
            
            for i in range(len(pieces) - 1):
                pair = (pieces[i], pieces[i + 1])
                rank = self.merges_dict.get(pair, float('inf'))
                
                if rank < best_rank:
                    best_rank = rank
                    best_idx = i
            
            # No more merges found
            if best_idx == -1:
                break
                
            # Apply the merge
            pieces[best_idx] = pieces[best_idx] + pieces[best_idx + 1]
            pieces.pop(best_idx + 1)
        
        return pieces
    
    def tokenize(self, text: str) -> List[str]:
        """
        Convert a text string into a list of tokens.
        
        Args:
            text: The input text to tokenize
            
        Returns:
            A list of tokens
        """
        # Preprocess text
        text = self.preprocess(text)
        
        # Split into words
        words = text.split()
        
        # Tokenize each word
        tokens = []
        for word in words:
            tokens.extend(self._tokenize_word(word))
        
        return tokens
    
    def encode(self, text: str) -> List[int]:
        """
        Convert a text string into a list of token indices.
        
        Args:
            text: The input text to encode
            
        Returns:
            A list of token indices
        """
        tokens = self.tokenize(text)
        return self.vocab.tokens_to_indices(tokens)
    
    def decode(self, token_ids: List[int]) -> str:
        """
        Convert a list of token indices back into a text string.
        
        Args:
            token_ids: The token indices to decode
            
        Returns:
            The reconstructed text
        """
        tokens = self.vocab.indices_to_tokens(token_ids)
        # Simple concatenation - in a real BPE implementation, this would need
        # to handle special splitting characters if used during training
        return ''.join(tokens)
    
    def batch_encode(self, texts: List[str]) -> List[List[int]]:
        """
        Encode a batch of texts into token indices.
        
        Args:
            texts: List of input texts to encode
            
        Returns:
            List of token index lists
        """
        return [self.encode(text) for text in texts]
    
    @property
    def vocab_size(self) -> int:
        """
        Get the vocabulary size of this tokenizer.
        
        Returns:
            The number of tokens in the vocabulary
        """
        return len(self.vocab)
    
    @property
    def special_tokens(self) -> Dict[str, int]:
        """
        Get the special tokens used by this tokenizer.
        
        Returns:
            Dictionary mapping special token names to their indices
        """
        return self.vocab.special_token_indices
    
    def save_pretrained(self, path: str) -> None:
        """
        Save the tokenizer to a directory.
        
        Args:
            path: Directory path to save to
        """
        os.makedirs(path, exist_ok=True)
        
        # Save vocabulary
        self.vocab.save(f"{path}/vocab.json")
        
        # Save merges - convert tuples to lists for JSON serialization
        merges_list = [list(pair) for pair in self.merges]
        with open(f"{path}/merges.json", "w", encoding="utf-8") as f:
            json.dump(merges_list, f, ensure_ascii=False)
        
        # Save config
        config = {
            "num_merges": self.num_merges,
            "lower_case": self.lower_case,
        }
        with open(f"{path}/config.json", "w", encoding="utf-8") as f:
            json.dump(config, f, ensure_ascii=False)

    @classmethod
    def from_pretrained(cls, path: str) -> "BPETokenizer":
        """
        Load a tokenizer from a saved directory.
        
        Args:
            path: Directory path to load from
            
        Returns:
            Loaded tokenizer
        """
        # Load vocabulary
        vocab = Vocabulary.load(f"{path}/vocab.json")
        
        # Load merges and convert from lists to tuples
        with open(f"{path}/merges.json", "r", encoding="utf-8") as f:
            merges_list = json.load(f)
            # Convert lists to tuples for hashing
            merges = [tuple(pair) for pair in merges_list]
        
        # Load config
        with open(f"{path}/config.json", "r", encoding="utf-8") as f:
            config = json.load(f)
        
        # Create tokenizer
        tokenizer = cls(
            vocab=vocab,
            merges=merges,
            num_merges=config["num_merges"],
            lower_case=config["lower_case"],
        )
        
        return tokenizer

def preprocess_data_with_optimized_bpe(
    dataset, 
    de_tokenizer, 
    en_tokenizer, 
    batch_size=4000,  # Larger batch size for better GPU utilization
    use_multiprocessing=False,
    num_workers=4
):
    """
    Optimized preprocessing function for translation datasets.
    
    Args:
        dataset: Dataset with src_data and tgt_data attributes
        de_tokenizer: German BPE tokenizer
        en_tokenizer: English BPE tokenizer
        batch_size: Size of batches for processing
        use_multiprocessing: Whether to use multiprocessing for CPU parallelism
        num_workers: Number of worker processes when using multiprocessing
        
    Returns:
        Lists of tokenized source and target sequences
    """
    if use_multiprocessing:
        return _preprocess_with_multiprocessing(
            dataset, de_tokenizer, en_tokenizer, batch_size, num_workers
        )
    else:
        return _preprocess_without_multiprocessing(
            dataset, de_tokenizer, en_tokenizer, batch_size
        )

def _preprocess_without_multiprocessing(dataset, de_tokenizer, en_tokenizer, batch_size):
    """Process without multiprocessing (better for GPU utilization)."""
    src_sequences = []
    tgt_sequences = []
    
    # Get special token indices
    src_bos_idx = de_tokenizer.special_tokens["bos_token_idx"]
    src_eos_idx = de_tokenizer.special_tokens["eos_token_idx"]
    tgt_bos_idx = en_tokenizer.special_tokens["bos_token_idx"]
    tgt_eos_idx = en_tokenizer.special_tokens["eos_token_idx"]
    
    # Process in batches
    total_batches = (len(dataset.src_data) + batch_size - 1) // batch_size
    
    for i in tqdm(range(0, len(dataset.src_data), batch_size), 
                 total=total_batches, 
                 desc="Preprocessing batches"):
        # Get batch
        batch_end = min(i + batch_size, len(dataset.src_data))
        batch_src = dataset.src_data[i:batch_end]
        batch_tgt = dataset.tgt_data[i:batch_end]
        
        # Process source and target texts using optimized batch encoding
        src_token_ids = de_tokenizer.batch_encode_optimized(batch_src)
        tgt_token_ids = en_tokenizer.batch_encode_optimized(batch_tgt)
        
        # Add special tokens efficiently
        for src_ids, tgt_ids in zip(src_token_ids, tgt_token_ids):
            src_sequences.append([src_bos_idx] + src_ids + [src_eos_idx])
            tgt_sequences.append([tgt_bos_idx] + tgt_ids + [tgt_eos_idx])
    
    return src_sequences, tgt_sequences

def _preprocess_with_multiprocessing(dataset, de_tokenizer, en_tokenizer, batch_size, num_workers):
    """Process with multiprocessing (better for CPU-bound tasks)."""
    from multiprocessing import Pool
    
    # Prepare batches
    batches = []
    for i in range(0, len(dataset.src_data), batch_size):
        batch_end = min(i + batch_size, len(dataset.src_data))
        batches.append((
            dataset.src_data[i:batch_end],
            dataset.tgt_data[i:batch_end],
        ))
    
    # Get special token indices
    special_tokens = {
        'src_bos': de_tokenizer.special_tokens["bos_token_idx"],
        'src_eos': de_tokenizer.special_tokens["eos_token_idx"],
        'tgt_bos': en_tokenizer.special_tokens["bos_token_idx"],
        'tgt_eos': en_tokenizer.special_tokens["eos_token_idx"],
    }
    
    # Function to process a single batch
    def process_batch(batch_data):
        src_batch, tgt_batch = batch_data
        
        # Create CPU-only tokenizers for multiprocessing
        cpu_de_tokenizer = OptimizedBPETokenizer(
            vocab=de_tokenizer.vocab,
            merges=de_tokenizer.merges,
            device="mps"  # Force CPU for multiprocessing compatibility
        )
        
        cpu_en_tokenizer = OptimizedBPETokenizer(
            vocab=en_tokenizer.vocab,
            merges=en_tokenizer.merges,
            device="mps"  # Force CPU for multiprocessing compatibility
        )
        
        # Process batch
        src_token_ids = cpu_de_tokenizer.batch_encode_optimized(src_batch)
        tgt_token_ids = cpu_en_tokenizer.batch_encode_optimized(tgt_batch)
        
        # Add special tokens
        src_sequences = []
        tgt_sequences = []
        
        for src_ids, tgt_ids in zip(src_token_ids, tgt_token_ids):
            src_sequences.append([special_tokens['src_bos']] + src_ids + [special_tokens['src_eos']])
            tgt_sequences.append([special_tokens['tgt_bos']] + tgt_ids + [special_tokens['tgt_eos']])
        
        return src_sequences, tgt_sequences
    
    # Process batches in parallel
    with Pool(processes=num_workers) as pool:
        results = list(
            tqdm(
                pool.imap(process_batch, batches),
                total=len(batches),
                desc="Preprocessing batches (multiprocessing)"
            )
        )
    
    # Combine results
    src_sequences = []
    tgt_sequences = []
    
    for src_batch, tgt_batch in results:
        src_sequences.extend(src_batch)
        tgt_sequences.extend(tgt_batch)
    
    return src_sequences, tgt_sequences



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
        "module_purpose": "Implements Byte Pair Encoding tokenizer for subword tokenization with merge operations",
        "key_classes": [
            {
                "name": "BPETokenizer",
                "purpose": "Tokenizer that implements Byte Pair Encoding algorithm for subword tokenization",
                "key_methods": [
                    {
                        "name": "train",
                        "signature": "train(self, texts: List[str], vocab_size: Optional[int] = None, min_frequency: int = 2, show_progress: bool = True) -> None",
                        "brief_description": "Train the BPE tokenizer on a corpus of texts by iteratively merging frequent character pairs"
                    },
                    {
                        "name": "tokenize",
                        "signature": "tokenize(self, text: str) -> List[str]",
                        "brief_description": "Convert text into subword tokens based on learned merge operations"
                    },
                    {
                        "name": "encode",
                        "signature": "encode(self, text: str) -> List[int]",
                        "brief_description": "Convert text to token indices using the vocabulary"
                    },
                    {
                        "name": "decode",
                        "signature": "decode(self, token_ids: List[int]) -> str",
                        "brief_description": "Convert token indices back to text"
                    },
                    {
                        "name": "save_pretrained",
                        "signature": "save_pretrained(self, path: str) -> None",
                        "brief_description": "Save tokenizer configuration, vocabulary and merges to disk"
                    },
                    {
                        "name": "from_pretrained",
                        "signature": "from_pretrained(cls, path: str) -> 'BPETokenizer'",
                        "brief_description": "Load a tokenizer from a saved directory"
                    }
                ],
                "inheritance": "BaseTokenizer",
                "dependencies": [".base_tokenizer", ".vocabulary", ".preprocessing"]
            }
        ],
        "external_dependencies": ["json", "tqdm", "collections.Counter"],
        "complexity_score": 7,  # High complexity due to BPE training algorithm and merging logic
    }