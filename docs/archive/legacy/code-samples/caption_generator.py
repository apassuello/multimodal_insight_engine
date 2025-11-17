import torch
import torchvision.transforms as transforms
from PIL import Image

def generate_caption(model, image, tokenizer, prompt="This image shows ", 
                    max_length=50, device=None, temperature=1.0):
    """
    Generate text caption based on image and optional prompt.
    
    Args:
        model: Multimodal model for text generation
        image: Input image (PIL.Image or file path)
        tokenizer: Tokenizer for text processing
        prompt: Optional text prompt to start generation
        max_length: Maximum caption length
        device: Device to run inference on
        temperature: Sampling temperature (higher = more random)
        
    Returns:
        Generated caption text
    """
    # Set device
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Handle image input
    if isinstance(image, str):
        image = Image.open(image).convert('RGB')
        
    # Preprocess image
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])
    image_tensor = transform(image).unsqueeze(0).to(device)
    
    # Tokenize initial prompt
    tokens = tokenizer(prompt, return_tensors='pt').input_ids.to(device)
    
    # Generate text auto-regressively
    model.eval()
    with torch.no_grad():
        for _ in range(max_length):
            # Get predictions
            outputs = model(tokens, image_tensor)
            
            # Get next token prediction (last token's prediction)
            next_token_logits = outputs[:, -1, :] / temperature
            
            # Apply optional sampling strategies
            # For simplicity, we use basic argmax here
            next_token = torch.argmax(next_token_logits, dim=-1).unsqueeze(0)
            
            # Append prediction to existing tokens
            tokens = torch.cat([tokens, next_token], dim=1)
            
            # Check if we've generated an end token
            if next_token.item() == tokenizer.eos_token_id:
                break
    
    # Decode the generated text
    generated_text = tokenizer.decode(tokens[0], skip_special_tokens=True)
    return generated_text

def generate_caption_with_beam_search(model, image, tokenizer, prompt="This image shows ",
                                     max_length=50, beam_size=5, device=None):
    """
    Generate caption using beam search for better quality.
    
    Args:
        model: Multimodal model for text generation
        image: Input image
        tokenizer: Tokenizer for text processing
        prompt: Optional text prompt to start generation
        max_length: Maximum caption length
        beam_size: Beam search width
        device: Device to run inference on
        
    Returns:
        Best generated caption according to model
    """
    # Set device
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Handle image input
    if isinstance(image, str):
        image = Image.open(image).convert('RGB')
        
    # Preprocess image
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])
    image_tensor = transform(image).unsqueeze(0).to(device)
    
    # Tokenize initial prompt
    start_tokens = tokenizer(prompt, return_tensors='pt').input_ids.to(device)
    
    # Initialize beam search
    model.eval()
    with torch.no_grad():
        # Initial forward pass
        outputs = model(start_tokens, image_tensor)
        
        # Get initial probabilities for first token
        next_token_logits = outputs[:, -1, :]
        next_token_probs = torch.nn.functional.log_softmax(next_token_logits, dim=-1)
        
        # Get top-k initial tokens
        topk_probs, topk_indices = next_token_probs.topk(beam_size, dim=1)
        
        # Initialize beams with top-k tokens
        beams = []
        for i in range(beam_size):
            token_id = topk_indices[0, i].item()
            log_prob = topk_probs[0, i].item()
            beam_tokens = torch.cat([start_tokens, topk_indices[:, i:i+1]], dim=1)
            beams.append((beam_tokens, log_prob))
            
        # Beam search loop
        completed_beams = []
        
        for _ in range(max_length - 1):
            if len(beams) == 0:
                break
                
            new_beams = []
            
            # Extend each beam
            for beam_tokens, beam_log_prob in beams:
                # Check if beam is already completed
                if beam_tokens[0, -1].item() == tokenizer.eos_token_id:
                    completed_beams.append((beam_tokens, beam_log_prob))
                    continue
                    
                # Forward pass with current beam
                outputs = model(beam_tokens, image_tensor)
                
                # Get probabilities for next token
                next_token_logits = outputs[:, -1, :]
                next_token_probs = torch.nn.functional.log_softmax(next_token_logits, dim=-1)
                
                # Get top-k next tokens
                topk_probs, topk_indices = next_token_probs.topk(beam_size, dim=1)
                
                # Add new candidates to beam
                for i in range(beam_size):
                    token_id = topk_indices[0, i].item()
                    log_prob = topk_probs[0, i].item() + beam_log_prob
                    new_beam_tokens = torch.cat([beam_tokens, topk_indices[:, i:i+1]], dim=1)
                    new_beams.append((new_beam_tokens, log_prob))
            
            # Keep only top beams
            beams = sorted(new_beams, key=lambda x: x[1], reverse=True)[:beam_size]
            
            # Check if all beams are completed
            if all(beam[0][0, -1].item() == tokenizer.eos_token_id for beam in beams):
                completed_beams.extend(beams)
                break
                
        # Add any remaining beams to completed list
        completed_beams.extend(beams)
        
        # Get best beam
        best_beam = max(completed_beams, key=lambda x: x[1])
        best_tokens = best_beam[0][0]
        
        # Decode the generated text
        generated_text = tokenizer.decode(best_tokens, skip_special_tokens=True)
        return generated_text