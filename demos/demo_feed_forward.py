# # Example usage in a script or notebook

# import torch
# from torch.utils.data import DataLoader, TensorDataset
# from src.models.feed_forward import FeedForwardClassifier
# from src.training.trainer import train_model

# # Create some dummy data
# num_samples = 1000000
# input_size = 20
# num_classes = 10

# # Generate random data
# X = torch.randn(num_samples, input_size)
# y = torch.randint(0, num_classes, (num_samples,))

# # Split into train and validation sets
# train_size = int(0.8 * num_samples)
# X_train, X_val = X[:train_size], X[train_size:]
# y_train, y_val = y[:train_size], y[train_size:]

# # Create datasets and dataloaders
# train_dataset = TensorDataset(X_train, y_train)
# val_dataset = TensorDataset(X_val, y_val)

# # Create dataloaders
# train_dataloader = DataLoader(
#     train_dataset,
#     batch_size=32,
#     shuffle=True,
#     collate_fn=lambda batch: {
#         "inputs": torch.stack([x[0] for x in batch]),
#         "targets": torch.stack([x[1] for x in batch]),
#     },
# )

# val_dataloader = DataLoader(
#     val_dataset,
#     batch_size=32,
#     collate_fn=lambda batch: {
#         "inputs": torch.stack([x[0] for x in batch]),
#         "targets": torch.stack([x[1] for x in batch]),
#     },
# )

# # Create the model
# model = FeedForwardClassifier(
#     input_size=input_size,
#     hidden_sizes=[128, 64],
#     num_classes=num_classes,
#     activation="relu",
#     dropout=0.2,
#     use_layer_norm=True,
# )

# # Train the model
# history = train_model(
#     model=model,
#     train_dataloader=train_dataloader,
#     val_dataloader=val_dataloader,
#     epochs=10,
#     learning_rate=0.001,
#     early_stopping_patience=3,
# )

# # Save the trained model
# model.save("models/feed_forward_classifier.pt")

# # Example of making predictions
# with torch.no_grad():
#     # Get some test data
#     test_inputs = X_val[:5]

#     # Make predictions
#     predicted_classes = model.predict(test_inputs)
#     predicted_probs = model.predict_proba(test_inputs)

#     print("Predicted classes:", predicted_classes)
#     print("Predicted probabilities:", predicted_probs)

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from src.models.feed_forward import FeedForwardClassifier
from src.training.trainer import train_model

# Set random seed for reproducibility
torch.manual_seed(42)

# Define transformations for MNIST
transform = transforms.Compose(
    [
        transforms.ToTensor(),  # Convert images to tensors
        transforms.Normalize(
            (0.1307,), (0.3081,)
        ),  # Normalize using MNIST mean and std
    ]
)

# Load MNIST datasets
train_dataset = datasets.MNIST("./data", train=True, download=True, transform=transform)
test_dataset = datasets.MNIST("./data", train=False, download=True, transform=transform)


# Create dataloaders with the dictionary format our trainer expects
def collate_fn(batch):
    images = torch.stack([item[0] for item in batch])
    labels = torch.tensor([item[1] for item in batch])
    # Flatten the images from (B, 1, 28, 28) to (B, 784)
    images = images.view(images.size(0), -1)
    return {"inputs": images, "targets": labels}


train_dataloader = DataLoader(
    train_dataset, batch_size=64, shuffle=True, collate_fn=collate_fn
)

val_dataloader = DataLoader(test_dataset, batch_size=64, collate_fn=collate_fn)

# Create the model
input_size = 28 * 28  # MNIST images are 28x28 pixels
hidden_sizes = [128, 64]
num_classes = 10

model = FeedForwardClassifier(
    input_size=input_size,
    hidden_sizes=hidden_sizes,
    num_classes=num_classes,
    activation="relu",
    dropout=0.2,
    use_layer_norm=True,
)

print(f"Model has {model.count_parameters():,} trainable parameters")

# Train the model
history = train_model(
    model=model,
    train_dataloader=train_dataloader,
    val_dataloader=val_dataloader,
    epochs=10,
    learning_rate=0.001,
    early_stopping_patience=3,
)

# Save the trained model
model.save("models/mnist_classifier.pt")

# Visualize some predictions
import matplotlib.pyplot as plt
import numpy as np

# Get a batch of test data
for batch in val_dataloader:
    test_images = batch["inputs"][:5]  # Get 5 test images
    test_labels = batch["targets"][:5]  # Get corresponding labels
    break

# Make predictions
predicted_classes = model.predict(test_images)
predicted_probs = model.predict_proba(test_images)

# Display results
fig, axes = plt.subplots(1, 5, figsize=(15, 3))
for i in range(5):
    # Reshape image back to 28x28
    img = test_images[i].reshape(28, 28).cpu().numpy()
    # Display the image
    axes[i].imshow(img, cmap="gray")
    true_label = test_labels[i].item()
    pred_label = predicted_classes[i].item()
    # Set title with true and predicted labels
    axes[i].set_title(f"True: {true_label}\nPred: {pred_label}")
    axes[i].axis("off")

plt.tight_layout()
plt.savefig("mnist_predictions.png")
plt.close()

print("Sample predictions:")
for i in range(5):
    true_label = test_labels[i].item()
    pred_label = predicted_classes[i].item()
    prob = predicted_probs[i][pred_label].item() * 100
    print(
        f"Image {i+1}: True: {true_label}, Predicted: {pred_label} (Confidence: {prob:.2f}%)"
    )
