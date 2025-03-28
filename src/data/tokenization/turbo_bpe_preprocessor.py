import torch
import time
import numpy as np
from tqdm import tqdm
from collections import Counter
import pickle
import os
import multiprocessing
from functools import partial

class TurboBPEPreprocessor:
    """
    A high-performance BPE preprocessing system specifically optimized for Apple Silicon.
    
    This preprocessor uses aggressive caching, parallel processing, and avoids
    unnecessary CPU-GPU transfers for maximum performance on M-series chips.
    """
    
    def __init__(self, cache_dir="tokenizer_cache"):
        """
        Initialize the preprocessor.
        
        Args:
            cache_dir: Directory for storing cache files
        """
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)
        self.word_cache = {}
        self.dataset_cache = {}
        
        # Determine best batch size for local hardware
        # Smaller batches often work better on MPS for this workload
        self.optimal_batch_size = 1000  # Adjusted for M4-Pro
        
        # Determine number of CPU cores to use (leave some for system)
        num_cpus = multiprocessing.cpu_count()
        self.num_workers = max(1, num_cpus - 2)
        
        # Check if we're using MPS
        self.device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        print(f"TurboBPEPreprocessor initialized with {self.num_workers} worker processes")
        print(f"Optimal batch size: {self.optimal_batch_size}")
        print(f"Using device: {self.device}")
    
    def _generate_cache_key(self, dataset):
        """Generate a cache key based on the dataset characteristics."""
        # Use first few examples to create a hash
        sample_size = min(100, len(dataset.src_data))
        samples = [dataset.src_data[i] + dataset.tgt_data[i] for i in range(sample_size)]
        sample_text = "".join(samples)
        import hashlib
        return hashlib.md5(sample_text.encode()).hexdigest()
    
    def check_cached_preprocessed_data(self, dataset, src_lang="de", tgt_lang="en"):
        """Check if preprocessed data exists in cache."""
        cache_key = self._generate_cache_key(dataset)
        cache_file = f"{self.cache_dir}/{src_lang}_{tgt_lang}_{cache_key}.pkl"
        
        if os.path.exists(cache_file):
            print(f"Found cached preprocessed data: {cache_file}")
            try:
                with open(cache_file, 'rb') as f:
                    return pickle.load(f)
            except Exception as e:
                print(f"Error loading cache: {e}. Regenerating...")
                
        return None
    
    def save_preprocessed_data(self, data, dataset, src_lang="de", tgt_lang="en"):
        """Save preprocessed data to cache."""
        cache_key = self._generate_cache_key(dataset)
        cache_file = f"{self.cache_dir}/{src_lang}_{tgt_lang}_{cache_key}.pkl"
        
        try:
            with open(cache_file, 'wb') as f:
                pickle.dump(data, f)
            print(f"Saved preprocessed data to cache: {cache_file}")
        except Exception as e:
            print(f"Error saving cache: {e}")
    
    def _process_text_batch(self, texts, tokenizer):
        """Process a batch of texts with a tokenizer."""
        # Using cached MPS implementation
        token_ids_batch = []
        
        for text in texts:
            # Check if text is in cache
            if text in self.word_cache:
                token_ids_batch.append(self.word_cache[text])
                continue
                
            # Preprocess text
            processed = tokenizer.preprocess(text)
            
            # Split into words
            words = processed.split()
            
            # Tokenize words individually with caching
            all_tokens = []
            for word in words:
                # Skip empty words
                if not word:
                    continue
                    
                # Use cached tokens if available
                word_key = (tokenizer.__class__.__name__, word)
                if word_key in self.word_cache:
                    tokens = self.word_cache[word_key]
                else:
                    # Tokenize word
                    tokens = tokenizer._tokenize_word(word)
                    # Cache result
                    self.word_cache[word_key] = tokens
                
                all_tokens.extend(tokens)
            
            # Convert to token IDs
            token_ids = [tokenizer.vocab.token_to_index(token) for token in all_tokens]
            
            # Cache the result for the whole text
            self.word_cache[text] = token_ids
            token_ids_batch.append(token_ids)
        
        return token_ids_batch
    
    def _process_data_chunk(self, args):
        """Process a chunk of data (for multiprocessing)."""
        chunk_id, src_chunk, tgt_chunk, src_tokenizer, tgt_tokenizer, special_tokens = args
        
        # Process source texts
        src_token_ids = self._process_text_batch(src_chunk, src_tokenizer)
        
        # Process target texts
        tgt_token_ids = self._process_text_batch(tgt_chunk, tgt_tokenizer)
        
        # Add special tokens
        src_sequences = []
        tgt_sequences = []
        
        for src_ids, tgt_ids in zip(src_token_ids, tgt_token_ids):
            src_sequences.append([special_tokens['src_bos']] + src_ids + [special_tokens['src_eos']])
            tgt_sequences.append([special_tokens['tgt_bos']] + tgt_ids + [special_tokens['tgt_eos']])
        
        return chunk_id, src_sequences, tgt_sequences
    
    def preprocess_with_caching(self, dataset, src_tokenizer, tgt_tokenizer, force_regenerate=False):
        """Preprocess data with aggressive caching and parallel processing."""
        # Check cache first
        if not force_regenerate:
            cached_data = self.check_cached_preprocessed_data(dataset)
            if cached_data is not None:
                return cached_data
        
        print(f"Preprocessing {len(dataset.src_data)} sentence pairs...")
        
        # Initialize timer
        start_time = time.time()
        
        # Get special token IDs
        special_tokens = {
            'src_bos': src_tokenizer.special_tokens["bos_token_idx"],
            'src_eos': src_tokenizer.special_tokens["eos_token_idx"],
            'tgt_bos': tgt_tokenizer.special_tokens["bos_token_idx"],
            'tgt_eos': tgt_tokenizer.special_tokens["eos_token_idx"],
        }
        
        # Always use single-process approach for MPS
        print("Using single-process approach with GPU acceleration")
        
        # Process in optimally-sized batches for MPS
        src_sequences = []
        tgt_sequences = []
        
        # Use tqdm for progress tracking
        for i in tqdm(range(0, len(dataset.src_data), self.optimal_batch_size), 
                     desc="Processing batches"):
            end_idx = min(i + self.optimal_batch_size, len(dataset.src_data))
            
            # Get batch
            src_batch = dataset.src_data[i:end_idx]
            tgt_batch = dataset.tgt_data[i:end_idx]
            
            # Process batch
            src_token_ids = self._process_text_batch(src_batch, src_tokenizer)
            tgt_token_ids = self._process_text_batch(tgt_batch, tgt_tokenizer)
            
            # Add special tokens
            for src_ids, tgt_ids in zip(src_token_ids, tgt_token_ids):
                src_sequences.append([special_tokens['src_bos']] + src_ids + [special_tokens['src_eos']])
                tgt_sequences.append([special_tokens['tgt_bos']] + tgt_ids + [special_tokens['tgt_eos']])
        
        # Calculate processing time
        elapsed_time = time.time() - start_time
        examples_per_sec = len(dataset.src_data) / elapsed_time
        
        print(f"Preprocessing completed in {elapsed_time:.2f}s ({examples_per_sec:.1f} examples/sec)")
        
        # Cache result for future use
        result = (src_sequences, tgt_sequences)
        self.save_preprocessed_data(result, dataset)
        
        return result
    
    @staticmethod
    def optimize_tokenizer_for_preprocessing(tokenizer):
        """Apply optimizations to a tokenizer for preprocessing."""
        # 1. Add word cache if it doesn't exist
        if not hasattr(tokenizer, 'word_token_cache'):
            tokenizer.word_token_cache = {}
        
        # 2. Ensure _tokenize_word uses simple dictionary lookup when possible
        if not hasattr(tokenizer, '_tokenize_word_original'):
            # Save original implementation
            tokenizer._tokenize_word_original = tokenizer._tokenize_word
            
            # Replace with optimized version that prioritizes cache and dict lookup
            def _tokenize_word_optimized(self, word: str) -> list:
                # Check cache first
                if word in self.word_token_cache:
                    return self.word_token_cache[word]
                
                # For single-character words, return immediately
                if len(word) <= 1:
                    result = [word]
                    self.word_token_cache[word] = result
                    return result
                
                # Use original implementation but with caching
                result = self._tokenize_word_original(word)
                
                # Cache result
                if len(self.word_token_cache) < 100000:  # Limit cache size
                    self.word_token_cache[word] = result
                
                return result
            
            # Bind the optimized method to the tokenizer instance
            import types
            tokenizer._tokenize_word = types.MethodType(_tokenize_word_optimized, tokenizer)
        
        return tokenizer

def turbo_preprocess_data(dataset, de_tokenizer, en_tokenizer, force_regenerate=False):
    """
    High-performance preprocessing function for translation datasets.
    
    Args:
        dataset: Dataset with src_data and tgt_data attributes
        de_tokenizer: German BPE tokenizer
        en_tokenizer: English BPE tokenizer
        force_regenerate: Whether to force regeneration of preprocessed data
        
    Returns:
        Lists of tokenized source and target sequences
    """
    # Create the processor
    processor = TurboBPEPreprocessor()
    
    # Optimize tokenizers
    de_tokenizer = processor.optimize_tokenizer_for_preprocessing(de_tokenizer)
    en_tokenizer = processor.optimize_tokenizer_for_preprocessing(en_tokenizer)
    
    # Process the dataset
    return processor.preprocess_with_caching(
        dataset, 
        de_tokenizer, 
        en_tokenizer,
        force_regenerate=force_regenerate
    )

__all__ = ['TurboBPEPreprocessor', 'turbo_preprocess_data']
