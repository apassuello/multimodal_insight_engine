import torch
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import sys
import os
import math
import torch.nn.functional as F

# Add the src directory to the Python path
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from src.models.attention import (
    ScaledDotProductAttention,
    SimpleAttention,
    MultiHeadAttention,
)


def test_scaled_dot_product_attention():
    """Test the ScaledDotProductAttention implementation."""
    print("\n=== Testing ScaledDotProductAttention ===")

    # Create a simple batch of sequences for testing
    batch_size = 2
    seq_length = 4
    d_model = 8

    # Generate random query, key, value tensors
    query = torch.randn(batch_size, seq_length, d_model)
    key = torch.randn(batch_size, seq_length, d_model)
    value = torch.randn(batch_size, seq_length, d_model)

    # Initialize attention layer with no dropout for testing
    attention = ScaledDotProductAttention(dropout=0.0)  # Changed to 0.0
    attention.eval()  # Explicitly set to evaluation mode
    print(f"Training mode: {attention.training}")  # Should print False

    # Forward pass
    output, attention_weights = attention(query, key, value)

    # Print shapes
    print(f"Input shapes: query={query.shape}, key={key.shape}, value={value.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Attention weights shape: {attention_weights.shape}")

    # Check that raw softmax sums to 1 (add this debugging step)
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
    raw_weights = F.softmax(scores, dim=-1)
    raw_sum = raw_weights.sum(dim=-1)
    print(f"Raw softmax sum: {raw_sum[0][0].item()}")  # Should be very close to 1.0

    # Check output dimensions
    assert output.shape == (
        batch_size,
        seq_length,
        d_model,
    ), "Output shape is incorrect"
    assert attention_weights.shape == (
        batch_size,
        seq_length,
        seq_length,
    ), "Attention weights shape is incorrect"

    # Check that attention weights sum to 1 along the right dimension
    attn_sum = attention_weights.sum(dim=-1)
    print(f"Attention sum: min={attn_sum.min().item()}, max={attn_sum.max().item()}")
    assert torch.allclose(
        attn_sum, torch.ones_like(attn_sum), atol=1e-6
    ), "Attention weights don't sum to 1"

    print("ScaledDotProductAttention tests passed!")

    # Visualize attention weights for the first batch
    # This heatmap shows the raw scaled dot-product attention weights without any learned parameters.
    # Since we're using random data, you should see a somewhat random pattern of attention weights.
    # Each cell shows how much a query at position i (row) attends to a key at position j (column).
    # The weights in each row sum to 1.0 due to the softmax normalization.
    # Look for any positions that receive notably high attention across multiple queries.

    visualize_attention_weights(attention_weights[0], "Scaled Dot-Product Attention")

    # Instead of returning, assert the expected properties
    assert attention_weights.shape == (batch_size, seq_length, seq_length), "Incorrect attention weights shape"
    assert torch.all(attention_weights >= 0) and torch.all(attention_weights <= 1), "Attention weights should be between 0 and 1"


def test_simple_attention():
    """Test the SimpleAttention implementation."""
    print("\n=== Testing SimpleAttention ===")

    # Create test data
    batch_size = 2
    seq_length = 4
    input_dim = 8
    attention_dim = 16

    # Generate random input tensors
    query = torch.randn(batch_size, seq_length, input_dim)

    # Initialize attention layer
    attention = SimpleAttention(
        input_dim=input_dim, attention_dim=attention_dim, dropout=0.1
    )

    # Test self-attention
    output, attention_weights = attention(query)

    # Print shapes
    print(f"Input shape: {query.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Attention weights shape: {attention_weights.shape}")

    # Check output dimensions
    assert output.shape == (
        batch_size,
        seq_length,
        input_dim,
    ), "Output shape is incorrect"
    assert attention_weights.shape == (
        batch_size,
        seq_length,
        seq_length,
    ), "Attention weights shape is incorrect"

    # Test cross-attention
    key = torch.randn(
        batch_size, seq_length + 2, input_dim
    )  # Different sequence length
    value = torch.randn(batch_size, seq_length + 2, input_dim)

    cross_output, cross_attention_weights = attention(query, key, value)

    print(f"Cross-attention output shape: {cross_output.shape}")
    print(f"Cross-attention weights shape: {cross_attention_weights.shape}")

    # Check output dimensions for cross-attention
    assert cross_output.shape == (
        batch_size,
        seq_length,
        input_dim,
    ), "Cross-attention output shape is incorrect"
    assert cross_attention_weights.shape == (
        batch_size,
        seq_length,
        seq_length + 2,
    ), "Cross-attention weights shape is incorrect"

    print("SimpleAttention tests passed!")

    # Visualize attention weights for the first batch
    # This heatmap shows attention after applying learned projections to query, key, and value vectors.
    # Unlike the basic scaled dot-product attention, these weights reflect the effect of the projection matrices.
    # The pattern may still look somewhat random with untrained weights, but should differ from
    # the scaled dot-product attention due to the additional transformations.
    # This is similar to what one attention head in a transformer would compute.
    # Look for how the projection matrices change the attention distribution compared to the basic attention.
    visualize_attention_weights(attention_weights[0], "Simple Attention")

    # Instead of returning, assert the expected properties
    assert attention_weights.shape == (batch_size, seq_length, seq_length), "Incorrect attention weights shape"
    assert torch.all(attention_weights >= 0) and torch.all(attention_weights <= 1), "Attention weights should be between 0 and 1"


def test_multi_head_attention():
    """Test the MultiHeadAttention implementation."""
    print("\n=== Testing MultiHeadAttention ===")

    # Create test data
    batch_size = 2
    seq_length = 4
    input_dim = 64  # Must be divisible by num_heads
    num_heads = 8

    # Generate random input tensors
    query = torch.randn(batch_size, seq_length, input_dim)

    # Initialize attention layer
    attention = MultiHeadAttention(
        input_dim=input_dim, num_heads=num_heads, dropout=0.1
    )

    # Test self-attention
    output, attention_weights = attention(query)

    # Print shapes
    print(f"Input shape: {query.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Attention weights shape: {attention_weights.shape}")

    # Check output dimensions
    assert output.shape == (
        batch_size,
        seq_length,
        input_dim,
    ), "Output shape is incorrect"
    assert attention_weights.shape == (
        batch_size,
        seq_length,
        seq_length,
    ), "Attention weights shape is incorrect"

    # Test cross-attention
    key = torch.randn(
        batch_size, seq_length + 2, input_dim
    )  # Different sequence length
    value = torch.randn(batch_size, seq_length + 2, input_dim)

    cross_output, cross_attention_weights = attention(query, key, value)

    print(f"Cross-attention output shape: {cross_output.shape}")
    print(f"Cross-attention weights shape: {cross_attention_weights.shape}")

    # Check output dimensions for cross-attention
    assert cross_output.shape == (
        batch_size,
        seq_length,
        input_dim,
    ), "Cross-attention output shape is incorrect"
    assert cross_attention_weights.shape == (
        batch_size,
        seq_length,
        seq_length + 2,
    ), "Cross-attention weights shape is incorrect"

    # Test with a mask
    mask = torch.ones(batch_size, seq_length, seq_length)
    # Create a simple causal mask (lower triangular)
    for i in range(seq_length):
        for j in range(seq_length):
            if j > i:
                mask[:, i, j] = 0

    masked_output, masked_attention_weights = attention(query, mask=mask)

    # Verify mask was applied correctly - upper triangular elements should be close to zero
    for i in range(seq_length):
        for j in range(seq_length):
            if j > i:
                assert torch.all(
                    masked_attention_weights[:, i, j] < 0.01
                ), f"Mask didn't work at position ({i}, {j})"

    print("MultiHeadAttention tests passed!")

    # Visualize attention weights for the first batch
    # This heatmap shows the average attention pattern across all heads in multi-head attention.
    # In a trained model, each head would specialize in different aspects of the relationships between positions.
    # Since we're using random weights, you'll likely see a more diffuse pattern than in single-head attention,
    # as this represents the average of multiple different attention patterns.
    # In a real transformer, different heads might focus on syntactic relationships, semantic similarities,
    # or other linguistic patterns, but those specializations emerge during training.

    visualize_attention_weights(
        attention_weights[0], "Multi-Head Attention (average over heads)"
    )
    # This heatmap shows multi-head attention with a causal (triangular) mask applied.
    # You should see a distinct triangular pattern where:
    # - The lower triangle (including diagonal) contains attention weights
    # - The upper triangle contains zeros or very small values (appearing white/light)
    # This pattern ensures each position can only attend to itself and previous positions,
    # which is crucial for autoregressive models (like language models) to prevent "seeing the future".
    # In a real transformer decoder, this prevents information leakage during training and generation.

    visualize_attention_weights(
        masked_attention_weights[0], "Multi-Head Attention with Causal Mask"
    )

    # Instead of returning, assert the expected properties
    assert output.shape == (batch_size, seq_length, input_dim), "Incorrect output shape"
    assert attention_weights.shape == (batch_size, seq_length, seq_length), "Incorrect attention weights shape"
    assert torch.all(attention_weights >= 0) and torch.all(attention_weights <= 1), "Attention weights should be between 0 and 1"


def visualize_attention_weights(attention_weights, title):
    """Visualize attention weights as a heatmap."""
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        attention_weights.detach().cpu().numpy(), annot=True, fmt=".2f", cmap="viridis"
    )
    plt.title(title)
    plt.xlabel("Key Position")
    plt.ylabel("Query Position")
    plt.tight_layout()
    plt.savefig(f"{title.replace(' ', '_').lower()}.png")
    plt.close()


def test_attention_training():
    """Test if attention mechanisms can be trained."""
    print("\n=== Testing Attention Training ===")

    # Create a simple sequence-to-sequence task
    batch_size = 4
    seq_length = 6
    input_dim = 32

    # Generate random input and target
    inputs = torch.randn(batch_size, seq_length, input_dim)
    targets = torch.randn(batch_size, seq_length, input_dim)

    # Create model with attention
    model = torch.nn.Sequential(
        torch.nn.Linear(input_dim, input_dim),
        torch.nn.ReLU(),
    )

    # Add attention layer
    attention = SimpleAttention(input_dim=input_dim)

    # Define optimizer and loss
    optimizer = torch.optim.Adam(
        list(model.parameters()) + list(attention.parameters()), lr=0.01
    )
    criterion = torch.nn.MSELoss()

    # Simple training loop
    initial_loss = None
    final_loss = None

    for epoch in range(10):
        # Forward pass through model
        feature_vectors = model(inputs)

        # Apply attention
        attended_outputs, _ = attention(feature_vectors)

        # Compute loss
        loss = criterion(attended_outputs, targets)

        if epoch == 0:
            initial_loss = loss.item()

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print(f"Epoch {epoch+1}, Loss: {loss.item():.6f}")

        if epoch == 9:
            final_loss = loss.item()

    assert final_loss < initial_loss, "Loss did not decrease during training"

    print("Attention training test passed!")


def main():
    """Run all tests."""
    # Set random seed for reproducibility
    torch.manual_seed(42)

    # Test all attention mechanisms
    test_scaled_dot_product_attention()
    test_simple_attention()
    test_multi_head_attention()

    # Test if attention can be trained
    test_attention_training()

    print("\nAll tests passed!")


if __name__ == "__main__":
    main()
