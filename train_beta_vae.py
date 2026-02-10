import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import pandas as pd
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

# Check GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")

# Dataset
class FaceDataset(Dataset):
    def __init__(self, csv_file, transform=None):
        self.df = pd.read_csv(csv_file)
        self.transform = transform
        
        # Create label mapping
        self.labels = sorted(self.df['label'].unique())
        self.label_to_idx = {label: idx for idx, label in enumerate(self.labels)}
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        img_path = self.df.iloc[idx]['image_path']
        label = self.df.iloc[idx]['label']
        
        # Load image
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        label_idx = self.label_to_idx[label]
        
        return image, label_idx

# β-VAE Model
class BetaVAE(nn.Module):
    def __init__(self, latent_dim=128, beta=4, img_size=128):
        super().__init__()
        self.latent_dim = latent_dim
        self.beta = beta
        
        # Encoder
        self.encoder = nn.Sequential(
            # 128x128 -> 64x64
            nn.Conv2d(3, 32, 4, 2, 1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            
            # 64x64 -> 32x32
            nn.Conv2d(32, 64, 4, 2, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            
            # 32x32 -> 16x16
            nn.Conv2d(64, 128, 4, 2, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            
            # 16x16 -> 8x8
            nn.Conv2d(128, 256, 4, 2, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            
            # 8x8 -> 4x4
            nn.Conv2d(256, 512, 4, 2, 1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
        )
        
        # Latent space
        self.fc_mu = nn.Linear(512 * 4 * 4, latent_dim)
        self.fc_logvar = nn.Linear(512 * 4 * 4, latent_dim)
        
        # Decoder
        self.fc_decode = nn.Linear(latent_dim, 512 * 4 * 4)
        
        self.decoder = nn.Sequential(
            # 4x4 -> 8x8
            nn.ConvTranspose2d(512, 256, 4, 2, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            
            # 8x8 -> 16x16
            nn.ConvTranspose2d(256, 128, 4, 2, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            
            # 16x16 -> 32x32
            nn.ConvTranspose2d(128, 64, 4, 2, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            
            # 32x32 -> 64x64
            nn.ConvTranspose2d(64, 32, 4, 2, 1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            
            # 64x64 -> 128x128
            nn.ConvTranspose2d(32, 3, 4, 2, 1),
            nn.Sigmoid()
        )
    
    def encode(self, x):
        h = self.encoder(x)
        h = h.view(h.size(0), -1)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar
    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def decode(self, z):
        h = self.fc_decode(z)
        h = h.view(-1, 512, 4, 4)
        return self.decoder(h)
    
    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z)
        return recon, mu, logvar

def beta_vae_loss(recon_x, x, mu, logvar, beta=4):
    """β-VAE loss function"""
    # Reconstruction loss (MSE)
    recon_loss = F.mse_loss(recon_x, x, reduction='sum')
    
    # KL divergence
    kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    
    # Total loss with β weighting
    return recon_loss + beta * kld, recon_loss, kld

# Training function
def train_epoch(model, dataloader, optimizer, device, beta):
    model.train()
    total_loss = 0
    total_recon = 0
    total_kld = 0
    
    pbar = tqdm(dataloader, desc='Training')
    for batch_idx, (data, _) in enumerate(pbar):
        data = data.to(device)
        optimizer.zero_grad()
        
        # Forward pass
        recon, mu, logvar = model(data)
        loss, recon_loss, kld = beta_vae_loss(recon, data, mu, logvar, beta)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # Track metrics
        total_loss += loss.item()
        total_recon += recon_loss.item()
        total_kld += kld.item()
        
        # Update progress bar
        pbar.set_postfix({
            'loss': loss.item() / len(data),
            'recon': recon_loss.item() / len(data),
            'kld': kld.item() / len(data)
        })
    
    n_samples = len(dataloader.dataset)
    return total_loss / n_samples, total_recon / n_samples, total_kld / n_samples

def validate(model, dataloader, device, beta):
    model.eval()
    total_loss = 0
    total_recon = 0
    total_kld = 0
    
    with torch.no_grad():
        for data, _ in dataloader:
            data = data.to(device)
            recon, mu, logvar = model(data)
            loss, recon_loss, kld = beta_vae_loss(recon, data, mu, logvar, beta)
            
            total_loss += loss.item()
            total_recon += recon_loss.item()
            total_kld += kld.item()
    
    n_samples = len(dataloader.dataset)
    return total_loss / n_samples, total_recon / n_samples, total_kld / n_samples

# Main training script
def main():
    # Hyperparameters (optimized for RTX 4070 8GB)
    LATENT_DIM = 128
    BETA = 4
    IMG_SIZE = 128
    BATCH_SIZE = 64  # Increased from 32 to better utilize 8GB VRAM
    NUM_EPOCHS = 50
    LEARNING_RATE = 1e-4
    
    # Data transforms
    transform = transforms.Compose([
        transforms.Resize(IMG_SIZE),
        transforms.CenterCrop(IMG_SIZE),
        transforms.ToTensor(),
    ])
    
    # Load datasets (using augmented data with synthetic faces)
    train_dataset = FaceDataset('data/train_augmented.csv', transform=transform)
    val_dataset = FaceDataset('data/val.csv', transform=transform)
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=True, 
        num_workers=8,  # Increased for faster data loading
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=False, 
        num_workers=8,  # Increased for faster data loading
        pin_memory=True
    )
    
    print(f"Train set: {len(train_dataset)} images")
    print(f"Val set: {len(val_dataset)} images")
    print(f"Number of classes: {len(train_dataset.labels)}")
    print(f"Classes: {train_dataset.labels}")
    
    # Create model
    model = BetaVAE(latent_dim=LATENT_DIM, beta=BETA, img_size=IMG_SIZE).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    # Training history
    history = {
        'train_loss': [], 'train_recon': [], 'train_kld': [],
        'val_loss': [], 'val_recon': [], 'val_kld': []
    }
    
    best_val_loss = float('inf')
    
    # Training loop
    for epoch in range(NUM_EPOCHS):
        print(f"\nEpoch {epoch+1}/{NUM_EPOCHS}")
        
        # Train
        train_loss, train_recon, train_kld = train_epoch(
            model, train_loader, optimizer, device, BETA
        )
        
        # Validate
        val_loss, val_recon, val_kld = validate(
            model, val_loader, device, BETA
        )
        
        # Save history
        history['train_loss'].append(train_loss)
        history['train_recon'].append(train_recon)
        history['train_kld'].append(train_kld)
        history['val_loss'].append(val_loss)
        history['val_recon'].append(val_recon)
        history['val_kld'].append(val_kld)
        
        print(f"Train - Loss: {train_loss:.4f}, Recon: {train_recon:.4f}, KLD: {train_kld:.4f}")
        print(f"Val   - Loss: {val_loss:.4f}, Recon: {val_recon:.4f}, KLD: {val_kld:.4f}")
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
            }, 'models/best_beta_vae.pth')
            print(" Saved best model")
        
        # Save checkpoint every 10 epochs
        if (epoch + 1) % 10 == 0:
            torch.save(model.state_dict(), f'models/beta_vae_epoch_{epoch+1}.pth')
    
    # Save final model
    torch.save(model.state_dict(), 'models/beta_vae_final.pth')
    
    # Plot training curves
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    axes[0].plot(history['train_loss'], label='Train')
    axes[0].plot(history['val_loss'], label='Val')
    axes[0].set_title('Total Loss')
    axes[0].legend()
    
    axes[1].plot(history['train_recon'], label='Train')
    axes[1].plot(history['val_recon'], label='Val')
    axes[1].set_title('Reconstruction Loss')
    axes[1].legend()
    
    axes[2].plot(history['train_kld'], label='Train')
    axes[2].plot(history['val_kld'], label='Val')
    axes[2].set_title('KL Divergence')
    axes[2].legend()
    
    plt.tight_layout()
    plt.savefig('visualizations/training_curves.png', dpi=150)
    print("\n Training complete!")
    print("Saved: best_beta_vae.pth, beta_vae_final.pth, training_curves.png")

if __name__ == '__main__':
    main()