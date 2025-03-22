import os
import sys
import torch
import random
import numpy as np
from typing import List, Dict, Tuple, Optional
from torch.utils.data import Dataset, DataLoader

# Add parent directory to path to import local modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import our tokenization modules
from src.data.tokenization import (
    BPETokenizer,
    create_transformer_dataloaders,
)

# Import the transformer and training components
from src.models.transformer import EncoderDecoderTransformer
from src.training.transformer_trainer import TransformerTrainer
from src.training.transformer_utils import create_padding_mask, create_causal_mask


class TranslationDataset(Dataset):
    """
    Dataset for translation tasks with BPE tokenizers.
    
    This dataset handles parallel text data for sequence-to-sequence tasks.
    """
    
    def __init__(
        self,
        source_texts: List[str],
        target_texts: List[str],
        source_tokenizer: BPETokenizer,
        target_tokenizer: BPETokenizer,
        max_source_length: Optional[int] = None,
        max_target_length: Optional[int] = None,
    ):
        """
        Initialize the dataset.
        
        Args:
            source_texts: List of source language texts
            target_texts: List of target language texts
            source_tokenizer: Tokenizer for source language
            target_tokenizer: Tokenizer for target language
            max_source_length: Maximum source sequence length
            max_target_length: Maximum target sequence length
        """
        assert len(source_texts) == len(target_texts), "Source and target must have same length"
        
        self.source_texts = source_texts
        self.target_texts = target_texts
        self.source_tokenizer = source_tokenizer
        self.target_tokenizer = target_tokenizer
        self.max_source_length = max_source_length
        self.max_target_length = max_target_length
        
        # Get special token indices
        self.src_pad_idx = source_tokenizer.special_tokens.get("pad_token_idx", 0)
        self.tgt_pad_idx = target_tokenizer.special_tokens.get("pad_token_idx", 0)
        self.src_bos_idx = source_tokenizer.special_tokens.get("bos_token_idx", 1)
        self.tgt_bos_idx = target_tokenizer.special_tokens.get("bos_token_idx", 1)
        self.src_eos_idx = source_tokenizer.special_tokens.get("eos_token_idx", 2)
        self.tgt_eos_idx = target_tokenizer.special_tokens.get("eos_token_idx", 2)
    
    def __len__(self):
        return len(self.source_texts)
    
    def __getitem__(self, idx):
        source_text = self.source_texts[idx]
        target_text = self.target_texts[idx]
        
        # Tokenize and encode
        source_ids = self.source_tokenizer.encode(source_text)
        target_ids = self.target_tokenizer.encode(target_text)
        
        # Add special tokens
        source_ids = [self.src_bos_idx] + source_ids + [self.src_eos_idx]
        target_ids = [self.tgt_bos_idx] + target_ids + [self.tgt_eos_idx]
        
        # Truncate if needed
        if self.max_source_length is not None and len(source_ids) > self.max_source_length:
            source_ids = source_ids[:self.max_source_length-1] + [self.src_eos_idx]  # Keep EOS token
            
        if self.max_target_length is not None and len(target_ids) > self.max_target_length:
            target_ids = target_ids[:self.max_target_length-1] + [self.tgt_eos_idx]  # Keep EOS token
        
        # Create source and target input/output
        src = torch.tensor(source_ids, dtype=torch.long)
        
        # Target input is shifted right (remove last token)
        tgt_input = torch.tensor(target_ids[:-1], dtype=torch.long)
        
        # Target output is shifted left (remove first token)
        tgt_output = torch.tensor(target_ids[1:], dtype=torch.long)
        
        return {
            "src": src,
            "tgt_input": tgt_input,
            "tgt_output": tgt_output,
        }


def collate_translation_batch(batch):
    """
    Collate function for translation batches.
    
    Args:
        batch: List of dictionaries with "src", "tgt_input", and "tgt_output"
        
    Returns:
        Dictionary with batched tensors
    """
    # Get source and target sequences
    src_sequences = [item["src"] for item in batch]
    tgt_input_sequences = [item["tgt_input"] for item in batch]
    tgt_output_sequences = [item["tgt_output"] for item in batch]
    
    # Get max lengths
    src_max_len = max(seq.size(0) for seq in src_sequences)
    tgt_input_max_len = max(seq.size(0) for seq in tgt_input_sequences)
    tgt_output_max_len = max(seq.size(0) for seq in tgt_output_sequences)
    
    # Get padding indices (assuming same across batch)
    src_pad_idx = 0  # Default padding index
    tgt_pad_idx = 0  # Default padding index
    
    # Pad sequences
    src_padded = torch.stack([
        torch.cat([seq, torch.full((src_max_len - seq.size(0),), src_pad_idx, dtype=torch.long)])
        if seq.size(0) < src_max_len else seq
        for seq in src_sequences
    ])
    
    tgt_input_padded = torch.stack([
        torch.cat([seq, torch.full((tgt_input_max_len - seq.size(0),), tgt_pad_idx, dtype=torch.long)])
        if seq.size(0) < tgt_input_max_len else seq
        for seq in tgt_input_sequences
    ])
    
    tgt_output_padded = torch.stack([
        torch.cat([seq, torch.full((tgt_output_max_len - seq.size(0),), tgt_pad_idx, dtype=torch.long)])
        if seq.size(0) < tgt_output_max_len else seq
        for seq in tgt_output_sequences
    ])
    
    # Create masks
    src_padding_mask = (src_padded != src_pad_idx).unsqueeze(1).unsqueeze(2)  # [batch, 1, 1, src_len]
    tgt_padding_mask = (tgt_input_padded != tgt_pad_idx).unsqueeze(1).unsqueeze(2)  # [batch, 1, 1, tgt_len]
    
    # Create causal mask for target
    device = tgt_input_padded.device
    tgt_causal_mask = torch.tril(torch.ones((tgt_input_max_len, tgt_input_max_len), device=device)).bool()
    tgt_causal_mask = tgt_causal_mask.unsqueeze(0).unsqueeze(0)  # [1, 1, tgt_len, tgt_len]
    
    # Combine padding and causal mask for target
    tgt_mask = tgt_padding_mask & tgt_causal_mask
    
    return {
        "src": src_padded,
        "tgt_input": tgt_input_padded,
        "tgt_output": tgt_output_padded,
        "src_mask": src_padding_mask,
        "tgt_mask": tgt_mask,
        "src_padding_mask": src_padded == src_pad_idx,  # For loss calculation
        "tgt_padding_mask": tgt_output_padded == tgt_pad_idx,  # For loss calculation
    }


def create_translation_dataloaders(
    source_texts: List[str],
    target_texts: List[str],
    source_tokenizer: BPETokenizer,
    target_tokenizer: BPETokenizer,
    batch_size: int = 32,
    max_source_length: Optional[int] = None,
    max_target_length: Optional[int] = None,
    shuffle: bool = True,
    validation_split: float = 0.1,
    random_seed: int = 42,
) -> Tuple[DataLoader, Optional[DataLoader]]:
    """
    Create dataloaders for translation training.
    
    Args:
        source_texts: List of source language texts
        target_texts: List of target language texts
        source_tokenizer: Tokenizer for source language
        target_tokenizer: Tokenizer for target language
        batch_size: Batch size
        max_source_length: Maximum source sequence length
        max_target_length: Maximum target sequence length
        shuffle: Whether to shuffle the training data
        validation_split: Fraction of data to use for validation
        random_seed: Random seed for reproducibility
        
    Returns:
        Tuple of (train_dataloader, val_dataloader)
    """
    # Set random seed for reproducibility
    random.seed(random_seed)
    
    # Create indices for splitting
    indices = list(range(len(source_texts)))
    if shuffle:
        random.shuffle(indices)
    
    # Calculate split
    val_size = int(len(indices) * validation_split)
    train_indices = indices[val_size:]
    val_indices = indices[:val_size]
    
    # Split data
    train_source = [source_texts[i] for i in train_indices]
    train_target = [target_texts[i] for i in train_indices]
    
    val_source = [source_texts[i] for i in val_indices]
    val_target = [target_texts[i] for i in val_indices]
    
    # Create datasets
    train_dataset = TranslationDataset(
        source_texts=train_source,
        target_texts=train_target,
        source_tokenizer=source_tokenizer,
        target_tokenizer=target_tokenizer,
        max_source_length=max_source_length,
        max_target_length=max_target_length,
    )
    
    val_dataset = TranslationDataset(
        source_texts=val_source,
        target_texts=val_target,
        source_tokenizer=source_tokenizer,
        target_tokenizer=target_tokenizer,
        max_source_length=max_source_length,
        max_target_length=max_target_length,
    )
    
    # Create dataloaders
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collate_translation_batch,
    )
    
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_translation_batch,
    )
    
    return train_dataloader, val_dataloader


def train_translation_model(
    train_dataloader: DataLoader,
    val_dataloader: DataLoader,
    src_vocab_size: int,
    tgt_vocab_size: int,
    pad_idx: int = 0,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    d_model: int = 512,
    num_heads: int = 8,
    num_encoder_layers: int = 6,
    num_decoder_layers: int = 6,
    d_ff: int = 2048,
    dropout: float = 0.1,
    epochs: int = 10,
    save_path: str = "models/translation_model.pt",
):
    """
    Train a translation model.
    
    Args:
        train_dataloader: Training data loader
        val_dataloader: Validation data loader
        src_vocab_size: Source vocabulary size
        tgt_vocab_size: Target vocabulary size
        pad_idx: Padding token index
        device: Device to train on
        d_model: Dimension of model embeddings
        num_heads: Number of attention heads
        num_encoder_layers: Number of encoder layers
        num_decoder_layers: Number of decoder layers
        d_ff: Dimension of feed-forward networks
        dropout: Dropout probability
        epochs: Number of epochs to train for
        save_path: Path to save the model
    """
    # Create model
    model = EncoderDecoderTransformer(
        src_vocab_size=src_vocab_size,
        tgt_vocab_size=tgt_vocab_size,
        d_model=d_model,
        num_heads=num_heads,
        num_encoder_layers=num_encoder_layers,
        num_decoder_layers=num_decoder_layers,
        d_ff=d_ff,
        dropout=dropout,
        max_seq_length=1000,  # Adjust based on your dataset
        positional_encoding="sinusoidal",
        share_embeddings=False,  # Set to True to share embeddings between encoder and decoder
    )
    
    # Define a custom train function for the transformer trainer
    def custom_train_step(model, batch):
        # Forward pass
        outputs = model(
            src=batch["src"],
            tgt=batch["tgt_input"],
            src_mask=batch["src_mask"],
            tgt_mask=batch["tgt_mask"],
        )
        
        # Calculate loss (cross-entropy)
        loss_func = torch.nn.CrossEntropyLoss(ignore_index=pad_idx)
        
        # Reshape outputs and targets for loss calculation
        outputs = outputs.view(-1, outputs.size(-1))
        targets = batch["tgt_output"].view(-1)
        
        loss = loss_func(outputs, targets)
        
        # Calculate accuracy
        predictions = outputs.argmax(dim=-1)
        mask = (targets != pad_idx)
        correct = (predictions == targets) & mask
        total = mask.sum().item()
        accuracy = correct.sum().item() / total if total > 0 else 0.0
        
        return {
            "loss": loss,
            "accuracy": torch.tensor(accuracy, device=loss.device),
        }
    
    # Create trainer
    trainer = TransformerTrainer(
        model=model,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        pad_idx=pad_idx,
        lr=0.0001,
        warmup_steps=4000,
        label_smoothing=0.1,
        clip_grad=1.0,
        early_stopping_patience=5,
        device=device,
    )
    
    # Train model
    model.train_step = custom_train_step
    model.validation_step = custom_train_step
    
    history = trainer.train(epochs=epochs, save_path=save_path)
    
    return model, history


def translate(
    model: EncoderDecoderTransformer,
    source_tokenizer: BPETokenizer,
    target_tokenizer: BPETokenizer,
    text: str,
    max_length: int = 100,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
):
    """
    Translate text using a trained model.
    
    Args:
        model: Trained translation model
        source_tokenizer: Source language tokenizer
        target_tokenizer: Target language tokenizer
        text: Text to translate
        max_length: Maximum generation length
        device: Device to use
        
    Returns:
        Translated text
    """
    model.eval()
    
    # Tokenize source text
    source_ids = source_tokenizer.encode(text)
    source_ids = [source_tokenizer.special_tokens["bos_token_idx"]] + source_ids + [source_tokenizer.special_tokens["eos_token_idx"]]
    source_tensor = torch.tensor([source_ids], dtype=torch.long).to(device)
    
    # Create source mask
    src_mask = (source_tensor != source_tokenizer.special_tokens["pad_token_idx"]).unsqueeze(1).unsqueeze(2)
    
    # Initialize target with BOS token
    tgt_bos = target_tokenizer.special_tokens["bos_token_idx"]
    target_tensor = torch.tensor([[tgt_bos]], dtype=torch.long).to(device)
    
    # Generate translation
    for _ in range(max_length):
        # Create target mask (causal)
        tgt_len = target_tensor.size(1)
        tgt_mask = torch.tril(torch.ones((tgt_len, tgt_len), device=device)).bool().unsqueeze(0).unsqueeze(0)
        
        # Forward pass
        output = model(
            src=source_tensor,
            tgt=target_tensor,
            src_mask=src_mask,
            tgt_mask=tgt_mask,
        )
        
        # Get the next token prediction
        next_token_logits = output[:, -1, :]
        next_token = next_token_logits.argmax(dim=-1, keepdim=True)
        
        # Append the next token
        target_tensor = torch.cat([target_tensor, next_token], dim=1)
        
        # Stop if EOS token is generated
        if next_token.item() == target_tokenizer.special_tokens["eos_token_idx"]:
            break
    
    # Decode the generated tokens
    generated_ids = target_tensor[0, 1:].tolist()  # Skip the BOS token
    
    # Stop at EOS token if present
    if target_tokenizer.special_tokens["eos_token_idx"] in generated_ids:
        generated_ids = generated_ids[:generated_ids.index(target_tokenizer.special_tokens["eos_token_idx"])]
    
    translation = target_tokenizer.decode(generated_ids)
    return translation


def main():
    """Example of integrating BPE tokenizers with transformer models."""
    # Load tokenizers
    en_tokenizer = BPETokenizer.from_pretrained("models/tokenizers/en")
    de_tokenizer = BPETokenizer.from_pretrained("models/tokenizers/de")
    
    print(f"Loaded English tokenizer with vocab size: {en_tokenizer.vocab_size}")
    print(f"Loaded German tokenizer with vocab size: {de_tokenizer.vocab_size}")
    
    # Load a small subset of data for demonstration
    from demos.translation_example import IWSLTDataset
    
    # Set random seed for reproducibility
    np.random.seed(42)
    torch.manual_seed(42)
    random.seed(42)
    
    # Load dataset
    train_dataset = IWSLTDataset(
        src_lang="en",
        tgt_lang="de",
        year="2016",
        split="train",
        max_examples=5000  # Small subset for demonstration
    )
    
    # Create dataloaders
    train_dataloader, val_dataloader = create_translation_dataloaders(
        source_texts=train_dataset.src_data,
        target_texts=train_dataset.tgt_data,
        source_tokenizer=en_tokenizer,
        target_tokenizer=de_tokenizer,
        batch_size=32,
        max_source_length=128,
        max_target_length=128,
    )
    
    print(f"Created dataloaders with {len(train_dataloader)} training batches and {len(val_dataloader)} validation batches")
    
    # Display a batch example
    batch = next(iter(train_dataloader))
    print(f"Source shape: {batch['src'].shape}")
    print(f"Target input shape: {batch['tgt_input'].shape}")
    print(f"Target output shape: {batch['tgt_output'].shape}")
    
    # Example of how to train the model (commented out to avoid actual training)
    """
    model, history = train_translation_model(
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        src_vocab_size=en_tokenizer.vocab_size,
        tgt_vocab_size=de_tokenizer.vocab_size,
        pad_idx=en_tokenizer.special_tokens["pad_token_idx"],
        d_model=512,
        num_heads=8,
        num_encoder_layers=6,
        num_decoder_layers=6,
        epochs=5,
        save_path="models/translation_model.pt",
    )
    
    # Example of translating text
    source_text = "Hello, how are you?"
    translation = translate(
        model=model,
        source_tokenizer=en_tokenizer,
        target_tokenizer=de_tokenizer,
        text=source_text,
    )
    
    print(f"Source: {source_text}")
    print(f"Translation: {translation}")
    """
    
    print("\nIntegration with Transformer Model Complete!")
    print("To train the model and perform translations:")
    print("1. Uncomment the training code block in the script")
    print("2. Run the script to train the model")
    print("3. Use the trained model to translate text")


if __name__ == "__main__":
    main()
