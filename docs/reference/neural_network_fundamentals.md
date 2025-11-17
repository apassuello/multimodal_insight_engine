# Neural Network Fundamentals Reference Guide

## Basic Architecture

Neural networks are computational systems inspired by biological neural networks, composed of:

- **Neurons**: Computational units that process input signals
- **Layers**: Collections of neurons that transform data
- **Weights & Biases**: Learnable parameters that determine network behavior
- **Activation Functions**: Non-linear transformations that enable complex pattern learning

## Forward Propagation

The process of passing input data through the network to generate predictions:

```
Z[l] = W[l] · A[l-1] + b[l]
A[l] = g(Z[l])
```

Where:
- `Z[l]`: Weighted input to layer l
- `W[l]`: Weight matrix for layer l
- `A[l-1]`: Activation from previous layer
- `b[l]`: Bias vector
- `g()`: Activation function
- `A[l]`: Output activation of layer l

## Backpropagation

Algorithm for computing gradients of the loss function with respect to network parameters:

1. Calculate error at output layer
2. Propagate error backward through network using chain rule
3. Calculate gradients for weights and biases
4. Update parameters using an optimization algorithm

```
∂L/∂W[l] = ∂L/∂Z[l] · ∂Z[l]/∂W[l]
∂L/∂b[l] = ∂L/∂Z[l]
```

## Activation Functions

| Function | Formula | Pros | Cons | Use Cases |
|----------|---------|------|------|-----------|
| **ReLU** | max(0,x) | Fast computation, Reduces vanishing gradient, Induces sparsity | Dying ReLU problem (neurons that always output 0) | Default for hidden layers in many networks |
| **Leaky ReLU** | max(αx,x) where α is small | Prevents dying neurons | Extra hyperparameter | Alternative to ReLU when dead neurons are a concern |
| **GELU** | x·Φ(x) where Φ is standard normal CDF | Smooth transitions, Works well in transformers | Computationally expensive | Transformers, including Anthropic's models |
| **Sigmoid** | 1/(1+e^(-x)) | Outputs between 0 and 1 | Vanishing gradient, Not zero-centered | Binary classification outputs, Gates in RNNs |
| **Tanh** | (e^x-e^(-x))/(e^x+e^(-x)) | Zero-centered outputs | Vanishing gradient at extremes | RNN hidden states |
| **Softmax** | e^(xi)/Σe^(xj) | Outputs sum to 1 (probability distribution) | Only used in output layer | Multi-class classification |

**Selection Criteria**:
- Consider gradient properties (vanishing/exploding)
- Consider output range requirements
- Balance computational efficiency vs. performance
- Consider the specific layer's role in the network

## Loss Functions

| Function | Formula | Best Used For | Properties |
|----------|---------|---------------|------------|
| **MSE** | (1/n)∑(y-ŷ)² | Regression | Heavily penalizes outliers, Assumes normal distribution |
| **MAE** | (1/n)∑\|y-ŷ\| | Regression with outliers | More robust to outliers, Non-smooth at zero |
| **Binary Cross-Entropy** | -(y·log(ŷ)+(1-y)·log(1-ŷ)) | Binary classification | Works with probabilities, Good gradients |
| **Categorical Cross-Entropy** | -∑y·log(ŷ) | Multi-class classification | Used with softmax, Standard for classification |
| **KL Divergence** | ∑P(x)·log(P(x)/Q(x)) | Comparing distributions | Used in VAEs and generative models |

**Selection Criteria**:
- Match to the problem type (regression vs. classification)
- Consider robustness requirements
- Consider the implicit statistical assumptions
- Consider computational efficiency

## Optimization Algorithms

### Gradient Descent
```
θ = θ - α · ∇J(θ)
```

### Stochastic Gradient Descent (SGD)
```
θ = θ - α · ∇J(θ; x(i), y(i))
```

### RMSprop
```
v_t = β · v_{t-1} + (1-β) · g_t^2
θ_t = θ_{t-1} - (α / sqrt(v_t + ε)) · g_t
```
- Maintains moving average of squared gradients
- Divides learning rate by square root of this average
- Adapts learning rate per parameter
- Helps navigate ravines in the loss landscape
- Typical values: β = 0.9, ε = 10^-8

### Adam
```
m_t = β1 · m_{t-1} + (1-β1) · g_t
v_t = β2 · v_{t-1} + (1-β2) · g_t^2
m̂_t = m_t / (1-β1^t)
v̂_t = v_t / (1-β2^t)
θ_t = θ_{t-1} - (α / sqrt(v̂_t + ε)) · m̂_t
```
- Combines momentum and RMSprop
- Adaptation per parameter
- Bias correction terms
- Typical values: β1 = 0.9, β2 = 0.999, ε = 10^-8

**Selection Criteria**:
- Dataset size (SGD for very large datasets)
- Computational constraints
- Need for adaptive learning rates
- Problem structure (ravines, saddle points)

## Implementation in PyTorch

Basic neural network implementation pattern:

```python
class SimpleNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.activation = nn.ReLU()  # Or another activation
        self.fc2 = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.activation(x)
        x = self.fc2(x)
        return x
```

Training loop structure:
```python
# Initialize
model = SimpleNN(input_size, hidden_size, output_size)
criterion = nn.CrossEntropyLoss()  # Or another loss function
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
for epoch in range(num_epochs):
    for batch_x, batch_y in data_loader:
        # Forward pass
        outputs = model(batch_x)
        loss = criterion(outputs, batch_y)
        
        # Backward pass and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
```

## Connection to Transformers

These neural network fundamentals translate to transformers in the following ways:

- **Attention Mechanism**: Special neural network layer using weighted connections
- **Multi-Head Attention**: Multiple attention mechanisms in parallel
- **Feed-Forward Networks**: Standard neural layers within transformer blocks
- **Layer Normalization**: Stabilizes training by normalizing activations
- **Residual Connections**: Helps with gradient flow in deep networks

## Anthropic Connection

Anthropic's Claude models likely use:

- **GELU activations** in transformer blocks
- **Layer normalization** for stability
- **Cross-entropy loss** for next-token prediction
- **Custom reward modeling losses** for RLHF training
- **Safety-specific loss terms** for alignment

## Practical Tips

- Start with simpler architectures and grow complexity
- Monitor gradients for vanishing/exploding issues
- Use learning rate schedules for better convergence
- Implement proper weight initialization
- Apply regularization techniques (dropout, weight decay)
- Batch normalization or layer normalization for stability
- Validate with appropriate metrics beyond just loss
