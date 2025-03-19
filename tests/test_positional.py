import torch
import matplotlib.pyplot as plt
import sys
import os
import numpy as np
import math

# Add the src directory to the Python path
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from src.models.positional import PositionalEncoding, RotaryPositionEncoding

def test_positional_encoding():
    """Test and visualize positional encoding implementations."""
    print("\n=== Testing Positional Encoding ===")
    
    # Test parameters
    d_model = 128
    max_seq_length = 100
    batch_size = 2
    seq_length = 50
    
    # Create dummy input embeddings
    embeddings = torch.randn(batch_size, seq_length, d_model)
    
    # Test sinusoidal encoding
    print("\nTesting Sinusoidal Positional Encoding...")
    pos_encoding = PositionalEncoding(
        d_model=d_model,
        max_seq_length=max_seq_length,
        encoding_type="sinusoidal"
    )
    
    # Apply positional encoding
    encoded_embeddings = pos_encoding(embeddings)
    
    # Check shapes
    print(f"Input shape: {embeddings.shape}")
    print(f"Encoded shape: {encoded_embeddings.shape}")
    
    # Confirm the shapes match
    assert embeddings.shape == encoded_embeddings.shape, "Shape mismatch after encoding"
    
    # Visualize the encodings
    fig = pos_encoding.visualize_encodings(seq_length=50)
    plt.savefig("sinusoidal_positional_encoding.png")
    plt.close(fig)
    print("Sinusoidal encoding visualization saved as 'sinusoidal_positional_encoding.png'")
    
    # Test learned encoding
    print("\nTesting Learned Positional Encoding...")
    learned_pos_encoding = PositionalEncoding(
        d_model=d_model,
        max_seq_length=max_seq_length,
        encoding_type="learned"
    )
    
    # Apply learned positional encoding
    learned_encoded_embeddings = learned_pos_encoding(embeddings)
    
    # Check shapes
    print(f"Input shape: {embeddings.shape}")
    print(f"Encoded shape: {learned_encoded_embeddings.shape}")
    
    # Confirm the shapes match
    assert embeddings.shape == learned_encoded_embeddings.shape, "Shape mismatch after encoding"
    
    # Visualize the learned encodings (initial state before training)
    fig = learned_pos_encoding.visualize_encodings(seq_length=50)
    plt.savefig("learned_positional_encoding.png")
    plt.close(fig)
    print("Learned encoding visualization saved as 'learned_positional_encoding.png'")
    
    # Instead of returning, assert the expected properties
    assert isinstance(pos_encoding, PositionalEncoding), "Invalid sinusoidal encoding type"
    assert isinstance(learned_pos_encoding, PositionalEncoding), "Invalid learned encoding type"
    assert pos_encoding.encoding_type == "sinusoidal", "Wrong encoding type"
    assert learned_pos_encoding.encoding_type == "learned", "Wrong encoding type"
    assert pos_encoding.d_model == d_model, "Wrong model dimension"
    assert learned_pos_encoding.d_model == d_model, "Wrong model dimension"

def test_rotary_position_encoding():
    """Test and visualize rotary position embeddings."""
    print("\n=== Testing Rotary Position Encoding ===")
    
    # Test parameters
    d_model = 128  # Must be divisible by 2
    max_seq_length = 100
    batch_size = 2
    seq_length = 50
    num_heads = 4
    head_dim = d_model // num_heads
    
    # Create a rotary encoding instance
    rotary_encoding = RotaryPositionEncoding(
        head_dim=head_dim,  # RoPE is applied to each head separately
        max_seq_length=max_seq_length
    )
    
    # Create dummy query and key tensors [batch, seq, heads, head_dim]
    q = torch.randn(batch_size, seq_length, num_heads, head_dim)
    k = torch.randn(batch_size, seq_length, num_heads, head_dim)
    
    # Apply rotary encoding
    q_rot, k_rot = rotary_encoding(q, k)
    
    # Check shapes
    print(f"Input shapes: q={q.shape}, k={k.shape}")
    print(f"Rotated shapes: q_rot={q_rot.shape}, k_rot={k_rot.shape}")
    
    # Confirm the shapes match
    assert q.shape == q_rot.shape, "Shape mismatch after rotary encoding for q"
    assert k.shape == k_rot.shape, "Shape mismatch after rotary encoding for k"
    
    # Visualize the rotary effect
    fig = rotary_encoding.visualize_rotation(seq_length=10)
    plt.savefig("rotary_position_encoding.png")
    plt.close(fig)
    print("Rotary encoding visualization saved as 'rotary_position_encoding.png'")
    
    # Instead of returning, assert the expected properties
    assert isinstance(rotary_encoding, RotaryPositionEncoding), "Invalid rotary encoding type"
    assert rotary_encoding.head_dim == head_dim, "Wrong head dimension"
    assert rotary_encoding.max_seq_length == max_seq_length, "Wrong max sequence length"
    assert q_rot.shape == q.shape, "Shape mismatch in rotary-encoded queries"
    assert k_rot.shape == k.shape, "Shape mismatch in rotary-encoded keys"

def validate_positional_properties():
    """Validate key properties of positional encodings."""
    print("\n=== Validating Positional Encoding Properties ===")
    
    # Create sinusoidal positional encoding
    d_model = 128
    max_seq_length = 1000
    pos_encoding = PositionalEncoding(d_model, max_seq_length)
    
    # Get the encodings
    encodings = pos_encoding.pe[0, :, :].detach().cpu().numpy()
    
    # Property 1: Check if PE(pos+k) can be represented as a linear function of PE(pos)
    # This is a key property of sinusoidal encodings that allows the model to generalize to longer sequences
    print("\nTesting linear relationship property...")
    
    # Take positions 100 and 200
    pos1 = 100
    pos2 = 200
    
    # Calculate PE(pos1) and PE(pos2)
    pe_pos1 = encodings[pos1, :]
    pe_pos2 = encodings[pos2, :]
    
    # For sinusoidal encodings, we expect a linear relationship
    # Specifically, we can show that in certain cases, 
    # the relationship between PE(pos) and PE(pos+k) is determined by 
    # a rotation in each 2D subspace corresponding to each frequency
    
    # We'll demonstrate this for the first few dimensions
    dim_pairs = 3  # Number of dimension pairs to check
    
    for i in range(dim_pairs):
        # Get a pair of dimensions (2i, 2i+1)
        dim1, dim2 = 2*i, 2*i+1
        
        # Get the values for each position
        pos1_pair = encodings[pos1, [dim1, dim2]]
        pos2_pair = encodings[pos2, [dim1, dim2]]
        
        # Calculate the angle between them
        dot_product = np.dot(pos1_pair, pos2_pair)
        norm_product = np.linalg.norm(pos1_pair) * np.linalg.norm(pos2_pair)
        
        # Avoid division by zero
        if norm_product > 1e-10:
            angle = np.arccos(np.clip(dot_product / norm_product, -1.0, 1.0))
            angle_degrees = np.degrees(angle)
            print(f"Pair {i}: Angle between positions {pos1} and {pos2} in dimensions ({dim1},{dim2}): {angle_degrees:.2f}°")
    
    # Property 2: Check if encodings for different positions are orthogonal
    # This is another useful property of sinusoidal encodings
    print("\nTesting orthogonality property...")
    
    # Calculate dot products between different positions
    position_samples = [10, 50, 100, 500]
    
    for i, pos_i in enumerate(position_samples):
        for j, pos_j in enumerate(position_samples):
            if i < j:  # Only check unique pairs
                pe_i = encodings[pos_i, :]
                pe_j = encodings[pos_j, :]
                
                # Calculate dot product
                dot_product = np.dot(pe_i, pe_j)
                
                # Normalize by vector lengths
                norm_i = np.linalg.norm(pe_i)
                norm_j = np.linalg.norm(pe_j)
                normalized_dot = dot_product / (norm_i * norm_j)
                
                print(f"Normalized dot product between positions {pos_i} and {pos_j}: {normalized_dot:.4f}")
    
    print("\nValidation complete.")

def main():
    """Run all tests."""
    # Set random seed for reproducibility
    torch.manual_seed(42)
    
    # Test standard positional encodings
    test_positional_encoding()
    
    # Test rotary position encodings
    test_rotary_position_encoding()
    
    # Validate mathematical properties
    validate_positional_properties()
    
    print("\nAll tests completed successfully!")


if __name__ == "__main__":
    main()