import torch
from torchtext.vocab import build_vocab_from_iterator
import re

# Define special tokens
UNK_TOKEN = "<unk>"
PAD_TOKEN = "<pad>"
BOS_TOKEN = "<s>"
EOS_TOKEN = "</s>"
special_tokens = [UNK_TOKEN, PAD_TOKEN, BOS_TOKEN, EOS_TOKEN]

# Simple tokenization function
def tokenize(text):
    # Split on whitespace and remove punctuation
    return re.findall(r'\w+', text.lower())

# Create minimal vocabularies with just special tokens
def yield_tokens():
    # Add some dummy tokens to ensure we have a basic vocabulary
    dummy_text = "hello world this is a test"
    yield tokenize(dummy_text)
    yield special_tokens

print("Building minimal vocabularies...")
src_vocab = build_vocab_from_iterator(
    yield_tokens(),
    min_freq=1,
    specials=special_tokens,
    special_first=True
)

tgt_vocab = build_vocab_from_iterator(
    yield_tokens(),
    min_freq=1,
    specials=special_tokens,
    special_first=True
)

# Set default indices
src_vocab.set_default_index(0)
tgt_vocab.set_default_index(0)

# Save vocabularies
print("Saving minimal vocabularies to vocab.pt...")
torch.save((src_vocab, tgt_vocab), "vocab.pt")
print("Done!") 