import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import numpy as np
import struct
import os

# Set random seed for reproducibility
torch.manual_seed(3)

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Hyperparameters
batch_size = 128
num_epochs = 50
learning_rate = 0.001

# Data Preprocessing
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

class MNISTDataset(Dataset):
    def __init__(self, images_file, labels_file, transform=None):
        self.images = self.read_idx_file(images_file)
        self.labels = self.read_idx_file(labels_file)
        self.transform = transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        image = self.images[idx].reshape(28, 28).astype(np.float32) / 255.0
        label = int(self.labels[idx])
        
        if self.transform:
            image = self.transform(image)
        
        return image, label

    def read_idx_file(self, filename):
        with open(filename, 'rb') as f:
            zero, data_type, dims = struct.unpack('>HBB', f.read(4))
            shape = tuple(struct.unpack('>I', f.read(4))[0] for d in range(dims))
            return np.frombuffer(f.read(), dtype=np.uint8).reshape(shape)

# Create datasets
train_dataset = MNISTDataset(
    images_file='./data/mnist/train-images.idx3-ubyte',
    labels_file='./data/mnist/train-labels.idx1-ubyte',
    transform=transform
)

test_dataset = MNISTDataset(
    images_file='./data/mnist/t10k-images.idx3-ubyte',
    labels_file='./data/mnist/t10k-labels.idx1-ubyte',
    transform=transform
)

# Create data loaders
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Add noise function
def add_noise(img):
    noise = torch.randn_like(img) * 0.1
    noisy_img = img + noise
    return noisy_img.clamp(0, 1)

# Autoencoder Model
class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()
        
        # Encoder layers
        self.encode_conv1 = nn.Parameter(torch.randn(16, 1, 3, 3))
        self.encode_bias1 = nn.Parameter(torch.zeros(16))
        self.encode_conv2 = nn.Parameter(torch.randn(32, 16, 3, 3))
        self.encode_bias2 = nn.Parameter(torch.zeros(32))
        self.encode_conv3 = nn.Parameter(torch.randn(64, 32, 3, 3))
        self.encode_bias3 = nn.Parameter(torch.zeros(64))
        
        # Decoder layers
        self.decode_conv1 = nn.Parameter(torch.randn(64, 32, 3, 3))
        self.decode_bias1 = nn.Parameter(torch.zeros(32))
        self.decode_conv2 = nn.Parameter(torch.randn(32, 16, 3, 3))
        self.decode_bias2 = nn.Parameter(torch.zeros(16))
        self.decode_conv3 = nn.Parameter(torch.randn(16, 1, 3, 3))
        self.decode_bias3 = nn.Parameter(torch.zeros(1))

    def encode(self, x):
        x = F.relu(F.conv2d(x, weight=self.encode_conv1, bias=self.encode_bias1, padding=1))
        x = F.relu(F.conv2d(x, weight=self.encode_conv2, bias=self.encode_bias2, padding=1))
        x = F.relu(F.conv2d(x, weight=self.encode_conv3, bias=self.encode_bias3, padding=1))
        return x
    
    def decode(self, x):
        x = F.relu(F.conv_transpose2d(x, weight=self.decode_conv1, bias=self.decode_bias1, padding=1))
        x = F.relu(F.conv_transpose2d(x, weight=self.decode_conv2, bias=self.decode_bias2, padding=1))
        x = torch.sigmoid(F.conv_transpose2d(x, weight=self.decode_conv3, bias=self.decode_bias3, padding=1))
        return x
    
    def forward(self, x):
        x = self.encode(x)
        x = self.decode(x)
        return x

# Initialize model, loss, and optimizer
model = Autoencoder().to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Training loop
train_losses = []
test_losses = []

for epoch in range(num_epochs):
    model.train()
    train_loss = 0
    for batch_idx, (data, _) in enumerate(train_loader):
        data = data.to(device)
        noisy_data = add_noise(data)
        
        optimizer.zero_grad()
        outputs = model(noisy_data)
        loss = criterion(outputs, data)
        loss.backward()
        optimizer.step()
        
        train_loss += loss.item()
    
    train_loss /= len(train_loader)
    train_losses.append(train_loss)
    
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for data, _ in test_loader:
            data = data.to(device)
            noisy_data = add_noise(data)
            outputs = model(noisy_data)
            loss = criterion(outputs, data)
            test_loss += loss.item()
    
    test_loss /= len(test_loader)
    test_losses.append(test_loss)
    
    print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f}')

# Plot training curves
plt.figure(figsize=(10, 5))
plt.plot(train_losses, label='Train Loss')
plt.plot(test_losses, label='Test Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.title('Training and Test Loss Curves')
plt.show()

# Visualize results
model.eval()
with torch.no_grad():
    test_images = next(iter(test_loader))[0][:5].to(device)
    noisy_images = add_noise(test_images)
    reconstructed = model(noisy_images)

    fig, axes = plt.subplots(3, 5, figsize=(15, 9))
    for i in range(5):
        axes[0, i].imshow(test_images[i].cpu().squeeze(), cmap='gray')
        axes[0, i].axis('off')
        axes[1, i].imshow(noisy_images[i].cpu().squeeze(), cmap='gray')
        axes[1, i].axis('off')
        axes[2, i].imshow(reconstructed[i].cpu().squeeze(), cmap='gray')
        axes[2, i].axis('off')
    
    axes[0, 0].set_ylabel('Original')
    axes[1, 0].set_ylabel('Noisy')
    axes[2, 0].set_ylabel('Reconstructed')
    plt.tight_layout()
    plt.show()

# Save the model
# torch.save(model.state_dict(), 'autoencoder_model.pth')
