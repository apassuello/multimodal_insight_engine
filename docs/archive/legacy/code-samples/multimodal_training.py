import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

def train_multimodal_model(model, text_image_dataloader, epochs=10, lr=1e-4,
                           device=None, validation_dataloader=None,
                           early_stopping_patience=3):
    """
    Training loop for multimodal model with text and image inputs
    
    Args:
        model: The multimodal model to train
        text_image_dataloader: DataLoader providing text and image batches
        epochs: Number of training epochs
        lr: Learning rate
        device: Device to train on (will use CUDA if available if not specified)
        validation_dataloader: Optional validation DataLoader
        early_stopping_patience: Number of epochs to wait before early stopping
    
    Returns:
        Trained model and training history
    """
    # Set device
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    # Setup optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    
    # Loss function
    criterion = nn.CrossEntropyLoss()
    
    # For tracking metrics
    history = {
        "train_loss": [],
        "val_loss": []
    }
    
    # Early stopping variables
    best_val_loss = float('inf')
    patience_counter = 0
    
    # Training loop
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        
        # Progress bar
        progress_bar = tqdm(text_image_dataloader, desc=f"Epoch {epoch+1}/{epochs}")
        
        for batch in progress_bar:
            # Extract data and move to device
            text_ids = batch['text_ids'].to(device)
            images = batch['images'].to(device)
            labels = batch['labels'].to(device)
            
            # Forward pass
            outputs = model(text_ids, images)
            
            # Calculate loss (assuming outputs are [batch, seq_len, vocab_size])
            loss = criterion(
                outputs.view(-1, outputs.size(-1)), 
                labels.view(-1)
            )
            
            # Backward pass and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Update progress bar
            progress_bar.set_postfix(loss=loss.item())
            total_loss += loss.item()
        
        # Calculate average training loss
        avg_train_loss = total_loss / len(text_image_dataloader)
        history["train_loss"].append(avg_train_loss)
        
        # Validation if dataloader provided
        if validation_dataloader is not None:
            model.eval()
            val_loss = 0
            
            with torch.no_grad():
                for batch in validation_dataloader:
                    # Extract data and move to device
                    text_ids = batch['text_ids'].to(device)
                    images = batch['images'].to(device)
                    labels = batch['labels'].to(device)
                    
                    # Forward pass
                    outputs = model(text_ids, images)
                    
                    # Calculate loss
                    batch_loss = criterion(
                        outputs.view(-1, outputs.size(-1)), 
                        labels.view(-1)
                    )
                    val_loss += batch_loss.item()
            
            # Calculate average validation loss
            avg_val_loss = val_loss / len(validation_dataloader)
            history["val_loss"].append(avg_val_loss)
            
            print(f"Epoch {epoch+1}/{epochs}, Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")
            
            # Early stopping check
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= early_stopping_patience:
                    print(f"Early stopping triggered after {epoch+1} epochs")
                    break
        else:
            print(f"Epoch {epoch+1}/{epochs}, Train Loss: {avg_train_loss:.4f}")
    
    return model, history