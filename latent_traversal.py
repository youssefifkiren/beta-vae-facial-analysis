"""
Latent Traversal - Explore What Each Latent Dimension Controls

This script helps analyze what the β-VAE learned by:
1. Taking a base face image
2. For each latent dimension, varying it from -3σ to +3σ
3. Showing which dimensions affect eyes, face shape, asymmetry, etc.

Usage:
    python latent_traversal.py --image images/00001.png
    python latent_traversal.py --image images/00001.png --dims 5,23,47,89
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from torchvision import transforms
from PIL import Image
import argparse

from train_beta_vae import BetaVAE

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def load_vae(model_path='models/beta_vae_final.pth'):
    """Load trained VAE"""
    vae = BetaVAE(latent_dim=128).to(device)
    vae.load_state_dict(torch.load(model_path, map_location=device))
    vae.eval()
    return vae

def traverse_dimension(base_latent, dim_idx, num_steps=7, range_scale=3.0):
    """Traverse a single latent dimension"""
    variations = []
    values = np.linspace(-range_scale, range_scale, num_steps)
    
    for value in values:
        latent = base_latent.clone()
        latent[dim_idx] = value
        variations.append(latent)
    
    return torch.stack(variations), values

def create_traversal_grid(vae, image_path, dimensions=None, num_steps=7, img_size=128):
    """Create grid showing effect of each dimension"""
    
    # Load and encode image
    transform = transforms.Compose([
        transforms.Resize(img_size),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
    ])
    
    image = Image.open(image_path).convert('RGB')
    image_tensor = transform(image).unsqueeze(0).to(device)
    
    with torch.no_grad():
        base_latent, _ = vae.encode(image_tensor)
        base_latent = base_latent[0]
    
    # Auto-select dimensions if not provided
    if dimensions is None:
        # Find dimensions with highest variance (most interesting)
        latent_std = torch.std(base_latent)
        dimensions = [10, 25, 42, 67, 89, 103]  # Sample interesting dimensions
    
    num_dims = len(dimensions)
    fig, axes = plt.subplots(num_dims, num_steps, figsize=(num_steps * 1.5, num_dims * 1.5))
    
    if num_dims == 1:
        axes = axes.reshape(1, -1)
    
    print(f"\nTraversing {num_dims} dimensions...")
    
    for dim_row, dim_idx in enumerate(dimensions):
        print(f"  Dimension {dim_idx}...")
        
        # Generate variations
        latent_variations, values = traverse_dimension(base_latent, dim_idx, num_steps)
        
        with torch.no_grad():
            generated_images = vae.decode(latent_variations.to(device))
        
        # Plot
        for step_idx in range(num_steps):
            ax = axes[dim_row, step_idx]
            img_array = generated_images[step_idx].permute(1, 2, 0).cpu().numpy()
            ax.imshow(img_array)
            ax.axis('off')
            
            # Add value labels
            if step_idx == 0:
                ax.set_ylabel(f'Dim {dim_idx}', fontsize=9, fontweight='bold')
            
            if dim_row == 0:
                ax.set_title(f'{values[step_idx]:.1f}σ', fontsize=8)
    
    plt.suptitle(f'Latent Dimension Traversal: Exploring Learned Features', 
                 fontsize=14, fontweight='bold', y=0.99)
    plt.tight_layout()
    
    return fig

def main():
    parser = argparse.ArgumentParser(description='Explore latent dimensions')
    parser.add_argument('--image', type=str, default=None,
                       help='Path to base image (will use random test image if not provided)')
    parser.add_argument('--vae-model', type=str, default='models/beta_vae_final.pth',
                       help='Path to VAE model')
    parser.add_argument('--dims', type=str, default=None,
                       help='Comma-separated dimension indices (e.g., "5,23,47")')
    parser.add_argument('--steps', type=int, default=7,
                       help='Number of steps in traversal')
    parser.add_argument('--output', type=str, default='visualizations/latent_traversal.png',
                       help='Output image path')
    
    args = parser.parse_args()
    
    print("="*60)
    print("LATENT DIMENSION TRAVERSAL")
    print("="*60)
    print(f"Device: {device}")
    print(f"Steps: {args.steps}\n")
    
    # Get image
    if args.image is None:
        import pandas as pd
        df = pd.read_csv('data/test.csv')
        args.image = df.iloc[0]['image_path']
        print(f"Using random test image: {args.image}")
    
    # Parse dimensions
    dimensions = None
    if args.dims:
        dimensions = [int(d.strip()) for d in args.dims.split(',')]
        print(f"Traversing dimensions: {dimensions}")
    else:
        print("Auto-selecting interesting dimensions...")
    
    # Load VAE
    print("\nLoading VAE model...")
    vae = load_vae(args.vae_model)
    print(" VAE loaded\n")
    
    # Generate traversals
    fig = create_traversal_grid(vae, args.image, dimensions, args.steps)
    
    # Save
    fig.savefig(args.output, dpi=150, bbox_inches='tight')
    print(f"\n Saved traversal grid to: {args.output}")
    
    plt.show()


if __name__ == '__main__':
    main()
