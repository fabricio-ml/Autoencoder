# Autoencoder

This PyTorch script implements an autoencoder for MNIST digit images. It defines a neural network that compresses 28x28 pixel images to a 3D latent space and reconstructs them. The code trains the model over 20 epochs, using Adam optimizer and MSE loss. It visualizes original and reconstructed images to show the autoencoder's progress in learning to compress and reconstruct the digits effectively.

## Prerequisites

To run this autoencoder code, you'll need:

1. Python (3.x recommended)
2. PyTorch
3. torchvision
4. matplotlib

## Additionally, you'll need:

- CUDA-capable GPU (optional, for faster training)

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/fabricio-ml/Autoencoder.git
   cd Autoencoder

2. Install the dependencies:
   ```bash
   pip install torch torchvision matplotlib
   
3. Run it!
   ```bash
   python autoencoder.py
