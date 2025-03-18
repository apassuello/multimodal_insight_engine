import os

import spacy
import torch
from torchtext.datasets import Multi30k
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator

def tokenize_de(text):
    return [tok.text for tok in spacy_de.tokenizer(text)]

def tokenize_en(text):
    return [tok.text for tok in spacy_en.tokenizer(text)]

def yield_tokens(data_iter, tokenizer):
    for de_sentence, en_sentence in data_iter:
        yield tokenizer(de_sentence)

def yield_tokens_en(data_iter, tokenizer):
    for de_sentence, en_sentence in data_iter:
        yield tokenizer(en_sentence)

# Initialize tokenizers
spacy_de = get_tokenizer('spacy', language='de_core_news_sm')
spacy_en = get_tokenizer('spacy', language='en_core_web_sm')

# Load dataset
train, val, test = Multi30k(split=('train', 'valid', 'test'),
                           language_pair=('de', 'en'),
                           root='.data')

# Define special tokens
UNK_TOKEN = "<unk>"
PAD_TOKEN = "<pad>"
BOS_TOKEN = "<s>"
EOS_TOKEN = "</s>"
special_tokens = [UNK_TOKEN, PAD_TOKEN, BOS_TOKEN, EOS_TOKEN]

# Build vocabularies
print("Building German Vocabulary ...")
src_vocab = build_vocab_from_iterator(
    yield_tokens(train, tokenize_de),
    min_freq=2,
    specials=special_tokens,
    special_first=True,
)

print("Building English Vocabulary ...")
tgt_vocab = build_vocab_from_iterator(
    yield_tokens_en(train, tokenize_en),
    min_freq=2,
    specials=special_tokens,
    special_first=True,
)

# Set default indices
src_vocab.set_default_index(0)
tgt_vocab.set_default_index(0)

# Save vocabularies
print("Saving vocabularies to vocab.pt...")
torch.save((src_vocab, tgt_vocab), "vocab.pt")
print("Done!") 