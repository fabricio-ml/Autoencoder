# Import necessary PyTorch modules
import torch
import torch.nn as nn
import torch.optim as optim
# Import datasets and transforms from torchvision
from torchvision import datasets, transforms
# Import DataLoader for batch processing
from torch.utils.data import DataLoader
# Import matplotlib for visualization
import matplotlib.pyplot as plt

# Set the device to GPU if available, otherwise CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define the Autoencoder class
class Autoencoder(nn.Module):
    def __init__(self):
        # Call the parent class constructor
        super(Autoencoder, self).__init__()
        
        # Define the encoder layers
        self.encoder = nn.Sequential(
            nn.Linear(28 * 28, 128),  # Input layer: 784 (28x28) to 128
            nn.ReLU(True),            # ReLU activation
            nn.Linear(128, 64),       # 128 to 64
            nn.ReLU(True),
            nn.Linear(64, 12),        # 64 to 12
            nn.ReLU(True),
            nn.Linear(12, 3)          # 12 to 3 (latent space)
        )
        
        # Define the decoder layers
        self.decoder = nn.Sequential(
            nn.Linear(3, 12),         # 3 (latent space) to 12
            nn.ReLU(True),
            nn.Linear(12, 64),        # 12 to 64
            nn.ReLU(True),
            nn.Linear(64, 128),       # 64 to 128
            nn.ReLU(True),
            nn.Linear(128, 28 * 28),  # 128 to 784 (28x28)
            nn.Tanh()                 # Tanh activation for output
        )
        
    # Define the forward pass
    def forward(self, x):
        x = self.encoder(x)  # Encode the input
        x = self.decoder(x)  # Decode the encoded input
        return x

# Create an instance of the Autoencoder and move it to the specified device
model = Autoencoder().to(device)

# Define data transformations
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])

# Load the MNIST dataset
dataset = datasets.MNIST(root='./data', train=True, transform=transform, download=True)

# Create a DataLoader for batch processing
dataloader = DataLoader(dataset, batch_size=128, shuffle=True)

# Define the loss function (Mean Squared Error)
criterion = nn.MSELoss()

# Define the optimizer (Adam)
optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)

# Set the number of training epochs
num_epochs = 20

# Initialize a list to store outputs for visualization
outputs = []

# Training loop
for epoch in range(num_epochs):
    for data in dataloader:
        img, _ = data  # Get the image data (ignore labels)
        img = img.view(img.size(0), -1).to(device)  # Flatten the image and move to device
        output = model(img)  # Forward pass
        loss = criterion(output, img)  # Calculate loss
        
        optimizer.zero_grad()  # Zero the gradients
        loss.backward()  # Backpropagation
        optimizer.step()  # Update weights
    
    # Print epoch results
    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')
    outputs.append((epoch, img, output))  # Store results for visualization

print("Training complete.")

# Visualize results
for k in range(0, num_epochs, 5):
    plt.figure(figsize=(9, 2))
    plt.gray()
    
    # Get original and reconstructed images
    imgs = outputs[k][1].cpu().data.numpy()
    recon = outputs[k][2].cpu().data.numpy()
    
    # Plot original images
    for i, item in enumerate(imgs):
        if i >= 9: break
        plt.subplot(2, 9, i + 1)
        plt.imshow(item.reshape(28, 28))
        
    # Plot reconstructed images
    for i, item in enumerate(recon):
        if i >= 9: break
        plt.subplot(2, 9, 9 + i + 1)
        plt.imshow(item.reshape(28, 28))
    
    plt.show()  # Display the plot
