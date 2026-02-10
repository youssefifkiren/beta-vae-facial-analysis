"""
Enhanced Reconstruction Visualization

Shows β-VAE reconstruction quality with:
- Original vs Reconstructed side-by-side for each class
- Multiple examples per disorder
- Pixel-level differences highlighted
- Reconstruction error metrics

Usage:
    python reconstruction_visual.py
    python reconstruction_visual.py --samples-per-class 6
"""

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from torchvision import transforms
from PIL import Image
import os
import argparse

from train_beta_vae import BetaVAE, FaceDataset

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Human-readable disorder names
DISORDER_NAMES = {
    'facial_asymmetry': 'Facial Asymmetry',
    'hypertelorism': 'Hypertelorism',
    'hypotelorism': 'Hypotelorism',
    'long_lower_face': 'Long Lower Face',
    'normal': 'Normal',
    'short_lower_face': 'Short Lower Face'
}

CLASS_NAMES = list(DISORDER_NAMES.keys())

def load_vae(model_path='models/beta_vae_final.pth'):
    """Load trained VAE"""
    vae = BetaVAE(latent_dim=128).to(device)
    vae.load_state_dict(torch.load(model_path, map_location=device))
    vae.eval()
    return vae

def calculate_mse(original, reconstructed):
    """Calculate mean squared error"""
    return np.mean((original - reconstructed) ** 2)

def get_samples_per_class(csv_path, samples_per_class=6):
    """Get random samples from each disorder class"""
    df = pd.read_csv(csv_path)
    
    samples = {}
    for class_name in CLASS_NAMES:
        class_samples = df[df['label'] == class_name]
        if len(class_samples) >= samples_per_class:
            selected = class_samples.sample(n=samples_per_class, random_state=42)
        else:
            selected = class_samples
        samples[class_name] = selected
    
    return samples

def create_reconstruction_grid(vae, samples_dict, img_size=128):
    """Create comprehensive reconstruction visualization"""
    
    transform = transforms.Compose([
        transforms.Resize(img_size),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
    ])
    
    num_classes = len(CLASS_NAMES)
    samples_per_class = max(len(samples) for samples in samples_dict.values())
    
    # Create figure: 3 rows per class (original, reconstructed, difference)
    fig_height = num_classes * 3 * 1.5
    fig_width = samples_per_class * 2
    
    fig, axes = plt.subplots(num_classes * 3, samples_per_class, 
                            figsize=(fig_width, fig_height))
    
    if num_classes == 1:
        axes = axes.reshape(3, -1)
    
    print(f"\nGenerating reconstructions for {num_classes} classes × {samples_per_class} samples...")
    
    all_mse = []
    
    for class_idx, class_name in enumerate(CLASS_NAMES):
        samples = samples_dict[class_name]
        
        print(f"  Processing {class_name}...")
        
        for sample_idx in range(len(samples)):
            row = samples.iloc[sample_idx]
            image_path = row['image_path']
            
            # Load image
            image = Image.open(image_path).convert('RGB')
            image_tensor = transform(image).unsqueeze(0).to(device)
            
            # Get reconstruction
            with torch.no_grad():
                recon, mu, logvar = vae(image_tensor)
            
            # Convert to numpy
            original_np = image_tensor[0].permute(1, 2, 0).cpu().numpy()
            recon_np = recon[0].permute(1, 2, 0).cpu().numpy()
            
            # Calculate difference
            diff = np.abs(original_np - recon_np)
            mse = calculate_mse(original_np, recon_np)
            all_mse.append(mse)
            
            # Row indices for this class
            base_row = class_idx * 3
            
            # Plot original
            ax_orig = axes[base_row, sample_idx]
            ax_orig.imshow(original_np)
            ax_orig.axis('off')
            if sample_idx == 0:
                ax_orig.set_ylabel('Original', fontsize=10, fontweight='bold')
                # Add class label
                ax_orig.text(-0.3, 0.5, DISORDER_NAMES[class_name],
                           transform=ax_orig.transAxes,
                           ha='right', va='center',
                           fontsize=11, fontweight='bold',
                           rotation=0)
            
            # Plot reconstruction
            ax_recon = axes[base_row + 1, sample_idx]
            ax_recon.imshow(recon_np)
            ax_recon.axis('off')
            if sample_idx == 0:
                ax_recon.set_ylabel('Reconstructed', fontsize=10, fontweight='bold')
            
            # Add MSE below reconstruction
            ax_recon.text(0.5, -0.05, f'MSE: {mse:.4f}',
                         transform=ax_recon.transAxes,
                         ha='center', va='top',
                         fontsize=8, color='blue')
            
            # Plot difference (heatmap)
            ax_diff = axes[base_row + 2, sample_idx]
            diff_gray = np.mean(diff, axis=2)  # Average across RGB
            im = ax_diff.imshow(diff_gray, cmap='hot', vmin=0, vmax=0.5)
            ax_diff.axis('off')
            if sample_idx == 0:
                ax_diff.set_ylabel('Difference', fontsize=10, fontweight='bold')
    
    # Overall title
    avg_mse = np.mean(all_mse)
    fig.suptitle(f'β-VAE Reconstruction Quality Analysis\n'
                f'Average MSE: {avg_mse:.4f} | Total Samples: {len(all_mse)}',
                fontsize=16, fontweight='bold', y=0.995)
    
    plt.tight_layout(rect=[0, 0, 1, 0.99])
    
    return fig, avg_mse

def main():
    parser = argparse.ArgumentParser(description='Visualize VAE reconstructions')
    parser.add_argument('--vae-model', type=str, default='models/beta_vae_final.pth',
                       help='Path to VAE model')
    parser.add_argument('--test-csv', type=str, default='data/test.csv',
                       help='Test dataset CSV')
    parser.add_argument('--samples-per-class', type=int, default=6,
                       help='Number of samples per class')
    parser.add_argument('--output', type=str, default='visualizations/reconstruction_quality.png',
                       help='Output image path')
    
    args = parser.parse_args()
    
    print("="*60)
    print("RECONSTRUCTION QUALITY VISUALIZATION")
    print("="*60)
    print(f"Device: {device}")
    print(f"Samples per class: {args.samples_per_class}\n")
    
    # Load VAE
    print("Loading VAE model...")
    vae = load_vae(args.vae_model)
    print(" VAE loaded\n")
    
    # Get test samples
    print("Loading test samples...")
    samples_dict = get_samples_per_class(args.test_csv, args.samples_per_class)
    print(f" Loaded {sum(len(s) for s in samples_dict.values())} total samples\n")
    
    # Generate visualization
    fig, avg_mse = create_reconstruction_grid(vae, samples_dict)
    
    # Save
    fig.savefig(args.output, dpi=150, bbox_inches='tight')
    print(f"\n Saved visualization to: {args.output}")
    
    plt.show()
    
    print("\n" + "="*60)
    print("COMPLETE!")
    print("="*60)
    print(f"\nAverage reconstruction MSE: {avg_mse:.4f}")
    print("\nVisualization shows:")
    print("  • Row 1: Original images")
    print("  • Row 2: Reconstructed images (with MSE)")
    print("  • Row 3: Pixel differences (red = high error)")
    print(f"  • {args.samples_per_class} samples per disorder class")

if __name__ == '__main__':
    main()
