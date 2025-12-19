import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import time

# Check PyTorch version and device availability
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Create synthetic data
np.random.seed(42)
x = np.random.rand(100, 1)
y = 1 + 2 * x + 0.1 * np.random.randn(100, 1)  # y = 1 + 2x + noise

# Convert to PyTorch tensors
x_tensor = torch.from_numpy(x).float().to(device)
y_tensor = torch.from_numpy(y).float().to(device)

# Define the neural network
class SimpleNet(nn.Module):
    def __init__(self):
        super(SimpleNet, self).__init__()
        self.fc1 = nn.Linear(1, 10)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(10, 1)
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

# Instantiate the model
model = SimpleNet().to(device)
print(model)

# Define loss function and optimizer
criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.1)

# Train the model
start_time = time.time()
losses = []

for epoch in range(100):
    # Forward pass
    outputs = model(x_tensor)
    loss = criterion(outputs, y_tensor)
    losses.append(loss.item())
    
    # Backward and optimize
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    if (epoch+1) % 10 == 0:
        print(f'Epoch [{epoch+1}/100], Loss: {loss.item():.4f}')

print(f"Training time: {time.time() - start_time:.2f} seconds")

# Test the model
model.eval()
with torch.no_grad():
    predicted = model(x_tensor).cpu().numpy()

# Plot the results
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.plot(losses)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss')

plt.subplot(1, 2, 2)
plt.scatter(x, y, label='Original data')
plt.plot(x, predicted, 'r', label='Fitted line')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Linear Regression Result')
plt.legend()

plt.tight_layout()
plt.savefig('pytorch_test_result.png')
plt.show()

# Check model parameters vs true parameters
print("Learned parameters:")
# The model.fc2.weight is a 1x10 tensor, so we need to access it differently
final_layer_weights = model.fc2.weight.detach().cpu().numpy()
final_layer_bias = model.fc2.bias.detach().cpu().item()
print(f"Weight shape: {final_layer_weights.shape}")
print(f"Weight values: {final_layer_weights}")
print(f"Bias: {final_layer_bias:.4f}")
print("True parameters: Weight: 2.0000, Bias: 1.0000")