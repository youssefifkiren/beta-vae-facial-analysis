"""
Latent Space Interpolation - Visualize Facial Variation Transitions

This script demonstrates controlled generation of facial variations by:
1. Finding representative examples of each disorder in the test set
2. Interpolating between their latent codes
3. Generating smooth transitions showing how facial features change

Usage:
    python latent_interpolation.py
    python latent_interpolation.py --steps 10
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

CLASS_NAMES = [
    'facial_asymmetry',
    'hypertelorism', 
    'hypotelorism',
    'long_lower_face',
    'normal',
    'short_lower_face'
]

def load_vae(model_path='models/beta_vae_final.pth'):
    """Load trained VAE"""
    vae = BetaVAE(latent_dim=128).to(device)
    vae.load_state_dict(torch.load(model_path, map_location=device))
    vae.eval()
    return vae

def get_representative_samples(csv_path='data/test.csv', samples_per_class=1):
    """Get representative examples of each disorder"""
    df = pd.read_csv(csv_path)
    
    representatives = {}
    for class_name in CLASS_NAMES:
        class_samples = df[df['label'] == class_name]
        if len(class_samples) > 0:
            # Get first N samples
            representatives[class_name] = class_samples.head(samples_per_class)
    
    return representatives

def interpolate_latents(latent1, latent2, steps=10):
    """Linear interpolation between two latent codes"""
    alphas = np.linspace(0, 1, steps)
    interpolated = []
    
    for alpha in alphas:
        latent = (1 - alpha) * latent1 + alpha * latent2
        interpolated.append(latent)
    
    return torch.stack(interpolated)

def generate_from_latents(vae, latents):
    """Generate images from latent codes"""
    with torch.no_grad():
        latents = latents.to(device)
        images = vae.decode(latents)
    return images.cpu()

def create_transition_grid(vae, disorder_pairs, steps=10, img_size=128):
    """Create grid showing transitions between disorder pairs"""
    
    transform = transforms.Compose([
        transforms.Resize(img_size),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
    ])
    
    num_pairs = len(disorder_pairs)
    fig, axes = plt.subplots(num_pairs, steps, figsize=(steps * 2, num_pairs * 2))
    
    if num_pairs == 1:
        axes = axes.reshape(1, -1)
    
    for pair_idx, (class1, class2, image1_path, image2_path) in enumerate(disorder_pairs):
        print(f"\nGenerating transition: {class1} → {class2}")
        
        # Load images and get latent codes
        img1 = Image.open(image1_path).convert('RGB')
        img2 = Image.open(image2_path).convert('RGB')
        
        tensor1 = transform(img1).unsqueeze(0).to(device)
        tensor2 = transform(img2).unsqueeze(0).to(device)
        
        with torch.no_grad():
            latent1, _ = vae.encode(tensor1)
            latent2, _ = vae.encode(tensor2)
        
        # Interpolate
        interpolated_latents = interpolate_latents(latent1[0], latent2[0], steps)
        
        # Generate images
        generated_images = generate_from_latents(vae, interpolated_latents)
        
        # Plot
        for step_idx in range(steps):
            ax = axes[pair_idx, step_idx]
            img_array = generated_images[step_idx].permute(1, 2, 0).numpy()
            ax.imshow(img_array)
            ax.axis('off')
            
            # Add labels at edges
            if step_idx == 0:
                ax.set_title(class1.replace('_', '\n'), fontsize=8, fontweight='bold')
            elif step_idx == steps - 1:
                ax.set_title(class2.replace('_', '\n'), fontsize=8, fontweight='bold')
            else:
                alpha = step_idx / (steps - 1)
                ax.set_title(f'{alpha:.1f}', fontsize=7, color='gray')
    
    plt.suptitle('Latent Space Interpolation: Facial Variation Transitions', 
                 fontsize=14, fontweight='bold', y=0.98)
    plt.tight_layout()
    
    return fig

def main():
    parser = argparse.ArgumentParser(description='Generate latent space interpolations')
    parser.add_argument('--vae-model', type=str, default='models/beta_vae_final.pth',
                       help='Path to VAE model')
    parser.add_argument('--test-csv', type=str, default='data/test.csv',
                       help='Test dataset CSV')
    parser.add_argument('--steps', type=int, default=10,
                       help='Number of interpolation steps')
    parser.add_argument('--output', type=str, default='visualizations/latent_interpolations.png',
                       help='Output image path')
    
    args = parser.parse_args()
    
    print("="*60)
    print("LATENT SPACE INTERPOLATION")
    print("="*60)
    print(f"Device: {device}")
    print(f"Interpolation steps: {args.steps}\n")
    
    # Load VAE
    print("Loading VAE model...")
    vae = load_vae(args.vae_model)
    print(" VAE loaded\n")
    
    # Get representative samples
    print("Finding representative samples...")
    representatives = get_representative_samples(args.test_csv)
    
    # Define interesting transitions
    disorder_pairs = []
    
    # Normal to each disorder
    if 'normal' in representatives:
        normal_img = representatives['normal'].iloc[0]['image_path']
        
        for disorder in ['hypertelorism', 'hypotelorism', 'facial_asymmetry', 'long_lower_face']:
            if disorder in representatives:
                disorder_img = representatives[disorder].iloc[0]['image_path']
                disorder_pairs.append(('normal', disorder, normal_img, disorder_img))
    
    print(f" Found {len(disorder_pairs)} disorder pairs to interpolate\n")
    
    # Generate transitions
    print("Generating interpolations...")
    fig = create_transition_grid(vae, disorder_pairs, steps=args.steps)
    
    # Save
    fig.savefig(args.output, dpi=150, bbox_inches='tight')
    print(f"\n Saved interpolation grid to: {args.output}")
    
    plt.show()

if __name__ == '__main__':
    main()
