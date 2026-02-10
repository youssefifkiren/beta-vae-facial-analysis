"""
Enhanced Reconstruction Visualization - Individual Files Per Class

Generates separate reconstruction images for each disorder class.
Better for LaTeX report integration.

Usage:
    python reconstruction_individual.py
    python reconstruction_individual.py --samples-per-class 4
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

def get_samples_per_class(csv_path, samples_per_class=4):
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

def create_single_class_reconstruction(vae, class_name, samples, img_size=128):
    """Create reconstruction visualization for a single class"""
    
    transform = transforms.Compose([
        transforms.Resize(img_size),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
    ])
    
    num_samples = len(samples)
    
    # Create figure: 3 rows (original, reconstructed, difference)
    fig, axes = plt.subplots(3, num_samples, figsize=(num_samples * 2, 6))
    
    if num_samples == 1:
        axes = axes.reshape(3, 1)
    
    all_mse = []
    
    for sample_idx in range(num_samples):
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
        
        # Plot original
        ax_orig = axes[0, sample_idx]
        ax_orig.imshow(original_np)
        ax_orig.axis('off')
        if sample_idx == 0:
            ax_orig.set_ylabel('Original', fontsize=12, fontweight='bold')
        
        # Plot reconstruction
        ax_recon = axes[1, sample_idx]
        ax_recon.imshow(recon_np)
        ax_recon.axis('off')
        if sample_idx == 0:
            ax_recon.set_ylabel('Reconstructed', fontsize=12, fontweight='bold')
        
        # Add MSE below reconstruction
        ax_recon.text(0.5, -0.1, f'MSE: {mse:.4f}',
                     transform=ax_recon.transAxes,
                     ha='center', va='top',
                     fontsize=10, color='blue', fontweight='bold')
        
        # Plot difference (heatmap)
        ax_diff = axes[2, sample_idx]
        diff_gray = np.mean(diff, axis=2)  # Average across RGB
        im = ax_diff.imshow(diff_gray, cmap='hot', vmin=0, vmax=0.5)
        ax_diff.axis('off')
        if sample_idx == 0:
            ax_diff.set_ylabel('Difference', fontsize=12, fontweight='bold')
    
    # Overall title
    avg_mse = np.mean(all_mse)
    fig.suptitle(f'{DISORDER_NAMES[class_name]} - Reconstruction Quality\\n'
                f'Average MSE: {avg_mse:.4f}',
                fontsize=14, fontweight='bold', y=0.98)
    
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    
    return fig, avg_mse

def main():
    parser = argparse.ArgumentParser(description='Visualize VAE reconstructions per class')
    parser.add_argument('--vae-model', type=str, default='models/beta_vae_final.pth',
                       help='Path to VAE model')
    parser.add_argument('--test-csv', type=str, default='data/test.csv',
                       help='Test dataset CSV')
    parser.add_argument('--samples-per-class', type=int, default=4,
                       help='Number of samples per class')
    parser.add_argument('--output-dir', type=str, default='visualizations',
                       help='Output directory')
    
    args = parser.parse_args()
    
    print("="*60)
    print("RECONSTRUCTION QUALITY VISUALIZATION (Individual Files)")
    print("="*60)
    print(f"Device: {device}")
    print(f"Samples per class: {args.samples_per_class}")
    print(f"Output directory: {args.output_dir}\n")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load VAE
    print("Loading VAE model...")
    vae = load_vae(args.vae_model)
    print(" VAE loaded\n")
    
    # Get test samples
    print("Loading test samples...")
    samples_dict = get_samples_per_class(args.test_csv, args.samples_per_class)
    print(f" Loaded {sum(len(s) for s in samples_dict.values())} total samples\n")
    
    # Generate visualization for each class
    print("Generating individual reconstruction visualizations...\n")
    
    all_avg_mse = []
    
    for class_name in CLASS_NAMES:
        samples = samples_dict[class_name]
        
        print(f"  Processing {DISORDER_NAMES[class_name]}...")
        
        fig, avg_mse = create_single_class_reconstruction(vae, class_name, samples)
        all_avg_mse.append(avg_mse)
        
        # Save with clean filename
        output_path = os.path.join(args.output_dir, f'reconstruction_{class_name}.png')
        fig.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"     Saved to: {output_path} (MSE: {avg_mse:.4f})")
        
        plt.close(fig)
    
    # Calculate overall average
    overall_avg = np.mean(all_avg_mse)
    
    print("\n" + "="*60)
    print("COMPLETE!")
    print("="*60)
    print(f"\nOverall average reconstruction MSE: {overall_avg:.4f}")
    print(f"\nGenerated {len(CLASS_NAMES)} individual visualization files:")
    for class_name in CLASS_NAMES:
        print(f"  • reconstruction_{class_name}.png")
    print("\nEach visualization shows:")
    print("  • Row 1: Original images")
    print("  • Row 2: Reconstructed images (with MSE)")
    print("  • Row 3: Pixel differences (red = high error)")

if __name__ == '__main__':
    main()
