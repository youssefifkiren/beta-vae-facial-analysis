import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import pandas as pd
import numpy as np
from tqdm import tqdm
import os
import sys

# Import BetaVAE from training script
sys.path.append(os.path.dirname(__file__))
from train_beta_vae import BetaVAE, FaceDataset

# Check GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Disorder classifier
class DisorderClassifier(nn.Module):
    def __init__(self, latent_dim=128, num_classes=6):  # Fixed: 6 classes not 5
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )
    
    def forward(self, x):
        return self.fc(x)

# Dataset for latent representations
class LatentDataset(Dataset):
    def __init__(self, latents, labels):
        self.latents = latents
        self.labels = labels
    
    def __len__(self):
        return len(self.latents)
    
    def __getitem__(self, idx):
        return self.latents[idx], self.labels[idx]

# Extract latent representations from VAE
def extract_latents(vae_model, dataloader, device):
    """Extract latent codes from VAE encoder for all images"""
    vae_model.eval()
    all_latents = []
    all_labels = []
    
    print(f"Extracting latent codes from {len(dataloader.dataset)} samples...")
    with torch.no_grad():
        for images, labels in tqdm(dataloader, desc="Extracting latents"):
            images = images.to(device)
            # Get latent representation (mu from encoder)
            mu, _ = vae_model.encode(images)
            all_latents.append(mu.cpu())
            all_labels.append(labels)
    
    latents = torch.cat(all_latents, dim=0)
    labels = torch.cat(all_labels, dim=0)
    
    print(f"Extracted latents shape: {latents.shape}")
    print(f"Labels shape: {labels.shape}")
    
    return latents, labels

# Training function
def train_epoch(model, dataloader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    for latents, labels in tqdm(dataloader, desc="Training"):
        latents = latents.to(device)
        labels = labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(latents)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
    
    return total_loss / len(dataloader), 100. * correct / total

# Validation function
def validate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for latents, labels in tqdm(dataloader, desc="Validation"):
            latents = latents.to(device)
            labels = labels.to(device)
            
            outputs = model(latents)
            loss = criterion(outputs, labels)
            
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    
    return total_loss / len(dataloader), 100. * correct / total

# Main training script
def main():
    # Hyperparameters
    LATENT_DIM = 128
    NUM_CLASSES = 6
    BATCH_SIZE = 64  # Larger batch for latent vectors (they're small)
    NUM_EPOCHS = 100
    LEARNING_RATE = 1e-3
    PATIENCE = 15  # Early stopping patience
    
    # Data paths
    DATA_DIR = os.path.dirname(__file__)
    TRAIN_CSV = os.path.join(DATA_DIR, 'data/train_augmented.csv')
    VAL_CSV = os.path.join(DATA_DIR, 'data/val.csv')
    TEST_CSV = os.path.join(DATA_DIR, 'data/test.csv')
    VAE_MODEL_PATH = os.path.join(DATA_DIR, 'models/beta_vae_final.pth')
    
    # Check if VAE model exists
    if not os.path.exists(VAE_MODEL_PATH):
        print(f"ERROR: VAE model not found at {VAE_MODEL_PATH}")
        print("Please train the β-VAE first using: python train_beta_vae.py")
        return
    
    # Data transforms (same as VAE training)
    IMG_SIZE = 128
    transform = transforms.Compose([
        transforms.Resize(IMG_SIZE),
        transforms.CenterCrop(IMG_SIZE),
        transforms.ToTensor(),
    ])
    
    # Load datasets
    print("Loading datasets...")
    train_dataset = FaceDataset(TRAIN_CSV, transform=transform)
    val_dataset = FaceDataset(VAL_CSV, transform=transform)
    test_dataset = FaceDataset(TEST_CSV, transform=transform)
    
    # Create dataloaders for image loading
    train_loader_images = DataLoader(train_dataset, batch_size=32, shuffle=False, num_workers=4)
    val_loader_images = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=4)
    test_loader_images = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=4)
    
    # Load pre-trained VAE
    print(f"Loading VAE model from {VAE_MODEL_PATH}...")
    vae_model = BetaVAE(latent_dim=LATENT_DIM).to(device)
    vae_model.load_state_dict(torch.load(VAE_MODEL_PATH, map_location=device))
    print("VAE model loaded successfully!")
    
    # Extract latent representations
    print("\n" + "="*50)
    print("EXTRACTING LATENT REPRESENTATIONS")
    print("="*50)
    
    train_latents, train_labels = extract_latents(vae_model, train_loader_images, device)
    val_latents, val_labels = extract_latents(vae_model, val_loader_images, device)
    test_latents, test_labels = extract_latents(vae_model, test_loader_images, device)
    
    # Save latent representations for later use
    print("\nSaving latent representations...")
    torch.save({
        'train_latents': train_latents,
        'train_labels': train_labels,
        'val_latents': val_latents,
        'val_labels': val_labels,
        'test_latents': test_latents,
        'test_labels': test_labels
    }, os.path.join(DATA_DIR, 'models/latent_representations.pth'))
    print("Latent representations saved!")
    
    # Create latent datasets
    train_latent_dataset = LatentDataset(train_latents, train_labels)
    val_latent_dataset = LatentDataset(val_latents, val_labels)
    test_latent_dataset = LatentDataset(test_latents, test_labels)
    
    # Create dataloaders for latent training
    train_loader = DataLoader(train_latent_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_latent_dataset, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_latent_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    # Initialize classifier
    print("\n" + "="*50)
    print("TRAINING CLASSIFIER")
    print("="*50)
    
    model = DisorderClassifier(latent_dim=LATENT_DIM, num_classes=NUM_CLASSES).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.CrossEntropyLoss()
    
    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5
    )
    
    # Training loop
    best_val_acc = 0
    patience_counter = 0
    train_losses = []
    train_accs = []
    val_losses = []
    val_accs = []
    
    print(f"\nTraining for {NUM_EPOCHS} epochs...")
    print(f"Train samples: {len(train_latent_dataset)}")
    print(f"Val samples: {len(val_latent_dataset)}")
    print(f"Test samples: {len(test_latent_dataset)}\n")
    
    for epoch in range(NUM_EPOCHS):
        print(f"\nEpoch {epoch+1}/{NUM_EPOCHS}")
        print("-" * 50)
        
        # Train
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion, device)
        train_losses.append(train_loss)
        train_accs.append(train_acc)
        
        # Validate
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        val_losses.append(val_loss)
        val_accs.append(val_acc)
        
        # Update learning rate
        scheduler.step(val_loss)
        
        print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
        print(f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%")
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            torch.save(model.state_dict(), os.path.join(DATA_DIR, 'models/disorder_classifier_best.pth'))
            print(f" Best model saved! (Val Acc: {best_val_acc:.2f}%)")
        else:
            patience_counter += 1
            print(f"No improvement ({patience_counter}/{PATIENCE})")
        
        # Early stopping
        if patience_counter >= PATIENCE:
            print(f"\nEarly stopping triggered after {epoch+1} epochs")
            break
    
    # Save final model
    torch.save(model.state_dict(), os.path.join(DATA_DIR, 'models/disorder_classifier_final.pth'))
    
    # Load best model for testing
    print("\n" + "="*50)
    print("TESTING BEST MODEL")
    print("="*50)
    
    model.load_state_dict(torch.load(os.path.join(DATA_DIR, 'models/disorder_classifier_best.pth')))
    test_loss, test_acc = validate(model, test_loader, criterion, device)
    
    print(f"\nTest Loss: {test_loss:.4f}")
    print(f"Test Accuracy: {test_acc:.2f}%")
    print(f"Random Baseline: {100/NUM_CLASSES:.2f}%")
    
    # Save training history
    import matplotlib.pyplot as plt
    
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Training and Validation Loss')
    plt.grid(True)
    
    plt.subplot(1, 2, 2)
    plt.plot(train_accs, label='Train Acc')
    plt.plot(val_accs, label='Val Acc')
    plt.axhline(y=100/NUM_CLASSES, color='r', linestyle='--', label='Random Baseline')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.title('Training and Validation Accuracy')
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(DATA_DIR, 'visualizations/classifier_training_curves.png'), dpi=150, bbox_inches='tight')
    print(f"\nTraining curves saved to classifier_training_curves.png")

if __name__ == '__main__':
    main()