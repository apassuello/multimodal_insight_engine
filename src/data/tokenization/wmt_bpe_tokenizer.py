import os
from typing import List, Tuple, Optional
from collections import Counter
import torch

class WMTBPETokenizer:
    def __init__(self, vocab: Optional[dict] = None, merges: Optional[List[Tuple[str, str]]] = None, num_merges: int = 10000, lower_case: bool = True, device: str = "cpu", cache_size: int = 100000):
        self.lower_case = lower_case
        self.num_merges = num_merges
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.cache_size = cache_size
        self.vocab = vocab or {}
        self.merges = merges or []
        self.merges_dict = {pair: i for i, pair in enumerate(self.merges)}
        self.token_cache = {}
        self.word_token_cache = {}
        self._create_tensor_lookup()

    def _create_tensor_lookup(self):
        """Create tensor-based lookup tables for faster tokenization."""
        if self.merges:
            single_char_merges = [(i, pair) for i, pair in enumerate(self.merges) if len(pair[0]) == 1 and len(pair[1]) == 1]
            if single_char_merges:
                indices, pairs = zip(*single_char_merges)
                self.single_char_merge_indices = torch.tensor(indices, device=self.device)
                self.single_char_merge_pairs = torch.tensor(
                    [[ord(p[0]), ord(p[1])] for p in pairs], 
                    device=self.device
                )
            else:
                self.single_char_merge_indices = torch.empty(0, dtype=torch.long, device=self.device)
                self.single_char_merge_pairs = torch.empty((0, 2), dtype=torch.long, device=self.device)

    def preprocess(self, text: str) -> str:
        """Preprocess text before tokenization."""
        if self.lower_case:
            text = text.lower()
        return text

    def batch_tokenize(self, texts: List[str]) -> List[List[str]]:
        """Tokenize a batch of texts."""
        return [self.tokenize(text) for text in texts]

    def batch_encode(self, texts: List[str]) -> List[List[int]]:
        """Encode a batch of texts to token IDs."""
        return [self.encode(text) for text in texts]

    def _tokenize_word_optimized(self, word: str) -> List[str]:
        """Tokenize a single word using BPE with optimized dictionary lookups."""
        if word in self.word_token_cache:
            return self.word_token_cache[word]

        pieces = list(word)
        if len(pieces) <= 1:
            self.word_token_cache[word] = pieces
            return pieces

        active_pieces = pieces.copy()
        while len(active_pieces) > 1:
            best_pair = None
            best_rank = float('inf')
            for i in range(len(active_pieces) - 1):
                pair = (active_pieces[i], active_pieces[i+1])
                if pair in self.merges_dict:
                    rank = self.merges_dict[pair]
                    if rank < best_rank:
                        best_rank = rank
                        best_pair = (i, i+1, pair)
            if best_pair is None:
                break
            i, j, (first, second) = best_pair
            merged = first + second
            active_pieces = active_pieces[:i] + [merged] + active_pieces[j+1:]

        if len(self.word_token_cache) < self.cache_size:
            self.word_token_cache[word] = active_pieces

        return active_pieces

    def tokenize(self, text: str) -> List[str]:
        """Convert a text string into a list of tokens."""
        if text in self.token_cache:
            return self.token_cache[text]

        text = self.preprocess(text)
        words = text.split()
        tokens = []
        for word in words:
            tokens.extend(self._tokenize_word_optimized(word))

        if len(self.token_cache) < self.cache_size:
            self.token_cache[text] = tokens

        return tokens

    def encode(self, text: str) -> List[int]:
        """Encode text to token IDs."""
        tokens = self.tokenize(text)
        return [self.vocab.get(token, self.vocab.get('<unk>')) for token in tokens]

    def decode(self, token_ids: List[int]) -> str:
        """Decode token IDs back into text."""
        reverse_vocab = {v: k for k, v in self.vocab.items()}
        return ' '.join(reverse_vocab.get(id, '<unk>') for id in token_ids)

# Example usage:
# tokenizer = WMTBPETokenizer(vocab={'hello': 0, 'world': 1, '<unk>': 2}, merges=[('h', 'e'), ('l', 'l'), ('o', 'w'), ('o', 'r'), ('l', 'd')])
# tokens = tokenizer.tokenize('hello world')
# print(tokens)
# token_ids = tokenizer.encode('hello world')
# print(token_ids)
# text = tokenizer.decode(token_ids)
# print(text) 