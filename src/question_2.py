import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import struct

# Set random seed for reproducibility
torch.manual_seed(3)

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Hyperparameters
batch_size = 128
num_epochs = 50
learning_rate = 0.001

# Data Preprocessing
transform = lambda x: torch.tensor(x).unsqueeze(0)

class FashionMNISTDataset(Dataset):
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
train_dataset = FashionMNISTDataset(
    images_file='./data/fashion/train-images-idx3-ubyte',
    labels_file='./data/fashion/train-labels-idx1-ubyte',
    transform=transform
)

test_dataset = FashionMNISTDataset(
    images_file='./data/fashion/t10k-images-idx3-ubyte',
    labels_file='./data/fashion/t10k-labels-idx1-ubyte',
    transform=transform
)

# Create data loaders
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

class Autoencoder(nn.Module):
    def __init__(self, encoded_dim):
        super(Autoencoder, self).__init__()
        self.encoded_dim = encoded_dim
        
        # Encoder
        self.encoder_conv1 = nn.Conv2d(1, 16, 3, stride=2, padding=1)
        self.encoder_conv2 = nn.Conv2d(16, 32, 3, stride=2, padding=1)
        self.encoder_fc = nn.Linear(32 * 7 * 7, encoded_dim)
        
        # Decoder
        self.decoder_fc = nn.Linear(encoded_dim, 32 * 7 * 7)
        self.decoder_conv1 = nn.ConvTranspose2d(32, 16, 3, stride=2, padding=1, output_padding=1)
        self.decoder_conv2 = nn.ConvTranspose2d(16, 1, 3, stride=2, padding=1, output_padding=1)

    def encode(self, x):
        x = F.relu(self.encoder_conv1(x))
        x = F.relu(self.encoder_conv2(x))
        x = x.view(x.size(0), -1)
        x = self.encoder_fc(x)
        return x

    def decode(self, x):
        x = self.decoder_fc(x)
        x = x.view(x.size(0), 32, 7, 7)
        x = F.relu(self.decoder_conv1(x))
        x = torch.sigmoid(self.decoder_conv2(x))
        return x

    def forward(self, x):
        x = self.encode(x)
        x = self.decode(x)
        return x

def train_autoencoder(model, train_loader, test_loader, num_epochs, optimizer, criterion):
    train_losses = []
    test_losses = []

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0
        for data, _ in train_loader:
            data = data.to(device)
            optimizer.zero_grad()
            outputs = model(data)
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
                outputs = model(data)
                loss = criterion(outputs, data)
                test_loss += loss.item()
        
        test_loss /= len(test_loader)
        test_losses.append(test_loss)
        
        print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f}')
    
    return train_losses, test_losses

def visualize_results(original, reconstructed, method, dim):
    fig, axes = plt.subplots(2, 5, figsize=(12, 5))
    for i in range(5):
        axes[0, i].imshow(original[i].reshape(28, 28), cmap='gray')
        axes[0, i].axis('off')
        axes[1, i].imshow(reconstructed[i].reshape(28, 28), cmap='gray')
        axes[1, i].axis('off')
    plt.suptitle(f'{method} Reconstruction (dim={dim})')
    plt.tight_layout()
    plt.show()

# Main execution
for dim in [1, 2, 3]:
    print(f"\nDimensionality: {dim}")
    
    # PCA
    pca = PCA(n_components=dim)
    train_data = train_dataset.images.reshape(-1, 28*28) / 255.0
    test_data = test_dataset.images.reshape(-1, 28*28) / 255.0
    
    pca_train = pca.fit_transform(train_data)
    pca_test = pca.transform(test_data)
    
    pca_reconstructed = pca.inverse_transform(pca_test)
    
    print("PCA Reconstruction Error:", np.mean((test_data - pca_reconstructed)**2))
    
    visualize_results(test_data[:5], pca_reconstructed[:5], "PCA", dim)
    
    # Autoencoder
    model = Autoencoder(dim).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    train_losses, test_losses = train_autoencoder(model, train_loader, test_loader, num_epochs, optimizer, criterion)
    
    plt.plot(train_losses, label='Train Loss')
    plt.plot(test_losses, label='Test Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title(f'Autoencoder Training (dim={dim})')
    plt.show()
    
    model.eval()
    with torch.no_grad():
        test_samples = next(iter(test_loader))[0][:5].to(device)
        reconstructed = model(test_samples)
    
    visualize_results(test_samples.cpu(), reconstructed.cpu(), "Autoencoder", dim)

# Save the final model
# torch.save(model.state_dict(), f'autoencoder_model_dim{dim}.pth')
