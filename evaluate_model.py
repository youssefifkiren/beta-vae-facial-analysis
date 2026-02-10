"""
Model Evaluation and Visualization Pipeline

This script generates comprehensive evaluation metrics and visualizations:
1. Confusion matrix
2. Classification report (precision, recall, F1)
3. t-SNE visualization of latent space
4. Reconstruction examples from β-VAE
"""

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.manifold import TSNE
from torchvision import transforms
from torch.utils.data import DataLoader
from PIL import Image
import os
import sys

# Import models
sys.path.append(os.path.dirname(__file__))
from train_beta_vae import BetaVAE, FaceDataset
from classifier_training import DisorderClassifier

# Device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def load_models(vae_path, classifier_path, latent_dim=128, num_classes=6):
    """Load trained VAE and classifier models"""
    vae = BetaVAE(latent_dim=latent_dim).to(device)
    vae.load_state_dict(torch.load(vae_path, map_location=device))
    vae.eval()
    
    classifier = DisorderClassifier(latent_dim=latent_dim, num_classes=num_classes).to(device)
    classifier.load_state_dict(torch.load(classifier_path, map_location=device))
    classifier.eval()
    
    return vae, classifier

def get_predictions(vae, classifier, dataloader):
    """Get predictions and ground truth labels"""
    all_preds = []
    all_labels = []
    all_latents = []
    
    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            
            # Get latent codes
            mu, _ = vae.encode(images)
            
            # Get predictions
            outputs = classifier(mu)
            _, preds = outputs.max(1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())
            all_latents.append(mu.cpu().numpy())
    
    return np.array(all_preds), np.array(all_labels), np.vstack(all_latents)

def plot_confusion_matrix(y_true, y_pred, class_names, output_path='visualizations/confusion_matrix.png'):
    """Generate and save confusion matrix"""
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names,
                cbar_kws={'label': 'Count'})
    plt.xlabel('Predicted Label', fontsize=12)
    plt.ylabel('True Label', fontsize=12)
    plt.title('Confusion Matrix - Disorder Classification', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f" Saved confusion matrix to {output_path}")
    plt.close()

def plot_tsne(latents, labels, class_names, output_path='visualizations/tsne_latent_space.png'):
    """Generate t-SNE visualization of latent space"""
    print("Computing t-SNE projection (this may take a minute)...")
    
    # Subsample if too many points (for speed)
    max_samples = 2000
    if len(latents) > max_samples:
        indices = np.random.choice(len(latents), max_samples, replace=False)
        latents = latents[indices]
        labels = labels[indices]
    
    tsne = TSNE(n_components=2, random_state=42, perplexity=30, max_iter=1000)
    latents_2d = tsne.fit_transform(latents)
    
    plt.figure(figsize=(12, 10))
    
    # Plot each class with different color
    colors = plt.cm.tab10(np.linspace(0, 1, len(class_names)))
    
    for i, class_name in enumerate(class_names):
        mask = labels == i
        plt.scatter(latents_2d[mask, 0], latents_2d[mask, 1], 
                   c=[colors[i]], label=class_name, alpha=0.6, s=50, edgecolors='k', linewidth=0.5)
    
    plt.xlabel('t-SNE Dimension 1', fontsize=12)
    plt.ylabel('t-SNE Dimension 2', fontsize=12)
    plt.title('t-SNE Visualization of β-VAE Latent Space', fontsize=14, fontweight='bold')
    plt.legend(loc='best', frameon=True, fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f" Saved t-SNE plot to {output_path}")
    plt.close()

def plot_reconstructions(vae, dataset, num_samples=10, output_path='visualizations/reconstruction_examples.png'):
    """Generate reconstruction examples"""
    indices = np.random.choice(len(dataset), num_samples, replace=False)
    
    fig, axes = plt.subplots(2, num_samples, figsize=(num_samples * 2, 4))
    
    with torch.no_grad():
        for idx, sample_idx in enumerate(indices):
            image, label = dataset[sample_idx]
            image_batch = image.unsqueeze(0).to(device)
            
            # Get reconstruction
            recon, _, _ = vae(image_batch)
            
            # Convert to numpy
            orig_np = image.permute(1, 2, 0).cpu().numpy()
            recon_np = recon[0].permute(1, 2, 0).cpu().numpy()
            
            # Plot original
            axes[0, idx].imshow(orig_np)
            axes[0, idx].axis('off')
            if idx == 0:
                axes[0, idx].set_title('Original', fontsize=10, fontweight='bold')
            
            # Plot reconstruction
            axes[1, idx].imshow(recon_np)
            axes[1, idx].axis('off')
            if idx == 0:
                axes[1, idx].set_title('Reconstructed', fontsize=10, fontweight='bold')
    
    plt.suptitle('β-VAE Reconstruction Examples', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f" Saved reconstruction examples to {output_path}")
    plt.close()

def save_classification_report(y_true, y_pred, class_names, output_path='outputs/classification_report.txt'):
    """Save detailed classification report"""
    report = classification_report(y_true, y_pred, target_names=class_names, digits=4)
    
    with open(output_path, 'w') as f:
        f.write("="*60 + "\n")
        f.write("CLASSIFICATION REPORT\n")
        f.write("="*60 + "\n\n")
        f.write(report)
        f.write("\n" + "="*60 + "\n")
    
    print(f" Saved classification report to {output_path}")
    print("\n" + report)

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Evaluate trained models')
    parser.add_argument('--vae-model', type=str, default='models/beta_vae_final.pth',
                      help='Path to VAE model')
    parser.add_argument('--classifier-model', type=str, default='models/disorder_classifier_best.pth',
                      help='Path to classifier model')
    parser.add_argument('--test-csv', type=str, default='data/test.csv',
                      help='Test dataset CSV')
    parser.add_argument('--output-dir', type=str, default='.',
                      help='Output directory for visualizations')
    
    args = parser.parse_args()
    
    print("="*60)
    print("MODEL EVALUATION PIPELINE")
    print("="*60)
    
    # Check if models exist
    if not os.path.exists(args.vae_model):
        print(f"ERROR: VAE model not found: {args.vae_model}")
        print("Please train the β-VAE first: python train_beta_vae.py")
        return
    
    if not os.path.exists(args.classifier_model):
        print(f"ERROR: Classifier model not found: {args.classifier_model}")
        print("Please train the classifier first: python classifier_training.py")
        return
    
    # Load models
    print(f"\nLoading models...")
    vae, classifier = load_models(args.vae_model, args.classifier_model)
    print(" Models loaded")
    
    # Load test dataset
    IMG_SIZE = 128
    transform = transforms.Compose([
        transforms.Resize(IMG_SIZE),
        transforms.CenterCrop(IMG_SIZE),
        transforms.ToTensor(),
    ])
    
    print(f"\nLoading test dataset from {args.test_csv}...")
    test_dataset = FaceDataset(args.test_csv, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    print(f" Test dataset loaded: {len(test_dataset)} samples")
    
    # Get class names
    class_names = sorted(pd.read_csv(args.test_csv)['label'].unique())
    print(f" Classes: {class_names}")
    
    # Get predictions
    print("\n" + "="*60)
    print("GENERATING PREDICTIONS")
    print("="*60)
    y_pred, y_true, latents = get_predictions(vae, classifier, test_loader)
    
    # Calculate accuracy
    accuracy = np.mean(y_pred == y_true) * 100
    print(f"\nTest Accuracy: {accuracy:.2f}%")
    print(f"Random Baseline: {100/len(class_names):.2f}%")
    
    # Generate visualizations
    print("\n" + "="*60)
    print("GENERATING VISUALIZATIONS")
    print("="*60)
    
    # Confusion matrix
    plot_confusion_matrix(
        y_true, y_pred, class_names,
        output_path=os.path.join(args.output_dir, 'visualizations/confusion_matrix.png')
    )
    
    # t-SNE
    plot_tsne(
        latents, y_true, class_names,
        output_path=os.path.join(args.output_dir, 'visualizations/tsne_latent_space.png')
    )
    
    # Reconstructions
    plot_reconstructions(
        vae, test_dataset, num_samples=10,
        output_path=os.path.join(args.output_dir, 'visualizations/reconstruction_examples.png')
    )
    
    # Classification report
    save_classification_report(
        y_true, y_pred, class_names,
        output_path=os.path.join(args.output_dir, 'outputs/classification_report.txt')
    )


if __name__ == '__main__':
    main()
