"""
StyleGAN2 Synthetic Face Generation Pipeline

This script:
1. Downloads pretrained StyleGAN2-FFHQ model
2. Generates high-quality synthetic face images
3. Saves them for annotation by the existing pipeline
"""

import os
import sys
import pickle
import numpy as np
import torch
import PIL.Image
from tqdm import tqdm

# Add StyleGAN2 repo to path
stylegan_path = os.path.join(os.path.dirname(__file__), 'stylegan2-ada-pytorch')
sys.path.insert(0, stylegan_path)

import dnnlib
import legacy

def generate_synthetic_faces(
    num_images=2000,
    output_dir='synthetic_faces',
    seed=42,
    truncation_psi=0.7,
    batch_size=4
):
    """
    Generate synthetic face images using pretrained StyleGAN2-FFHQ
    
    Args:
        num_images: Number of images to generate
        output_dir: Directory to save generated images
        seed: Random seed for reproducibility
        truncation_psi: Truncation trick parameter (0.7 = more realistic, 1.0 = more diverse)
        batch_size: Number of images to generate at once (adjust based on VRAM)
    """
    
    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    print(f"Output directory: {output_dir}")
    
    # Load pretrained StyleGAN2-FFHQ model
    model_url = 'https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/ffhq.pkl'
    print(f"\nLoading StyleGAN2-FFHQ model from NVIDIA servers...")
    print("This may take a few minutes on first run (downloading ~365MB)...")
    
    try:
        with dnnlib.util.open_url(model_url) as f:
            G = legacy.load_network_pkl(f)['G_ema'].to(device)
        print(" Model loaded successfully!")
    except Exception as e:
        print(f"ERROR: Failed to load model: {e}")
        print("\nTroubleshooting:")
        print("1. Check internet connection")
        print("2. Ensure you have ~2GB free disk space")
        print("3. Try downloading manually from:")
        print(f"   {model_url}")
        return
    
    print(f"\nModel info:")
    print(f"  - Latent dimension (z): {G.z_dim}")
    print(f"  - Output resolution: {G.img_resolution}x{G.img_resolution}")
    print(f"  - Conditional: {G.c_dim > 0}")
    
    # Generate images
    print(f"\nGenerating {num_images} images with truncation psi={truncation_psi}...")
    print("Lower psi = more realistic faces, Higher psi = more diverse faces")
    
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    # Prepare label (unconditional for FFHQ)
    label = torch.zeros([batch_size, G.c_dim], device=device)
    
    num_batches = (num_images + batch_size - 1) // batch_size
    
    for batch_idx in tqdm(range(num_batches), desc="Generating batches"):
        # Calculate actual batch size for last batch
        current_batch_size = min(batch_size, num_images - batch_idx * batch_size)
        
        # Generate random latent codes
        z = torch.randn([current_batch_size, G.z_dim], device=device)
        
        # Generate images
        with torch.no_grad():
            if current_batch_size < batch_size:
                # Adjust label size for last batch
                label_current = torch.zeros([current_batch_size, G.c_dim], device=device)
                img = G(z, label_current, truncation_psi=truncation_psi, noise_mode='const')
            else:
                img = G(z, label, truncation_psi=truncation_psi, noise_mode='const')
        
        # Convert to PIL images and resave at 512x512
        img = (img.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8)
        
        for i in range(current_batch_size):
            img_idx = batch_idx * batch_size + i
            img_pil = PIL.Image.fromarray(img[i].cpu().numpy(), 'RGB')
            
            # Resize to 512x512 to match original FFHQ dataset
            img_pil_resized = img_pil.resize((512, 512), PIL.Image.LANCZOS)
            
            # Save with same naming convention as FFHQ
            img_path = os.path.join(output_dir, f'synthetic_{img_idx:05d}.png')
            img_pil_resized.save(img_path)
    
    print(f"\n Successfully generated {num_images} images!")
    print(f"  Saved to: {output_dir}/")
    print(f"  Resolution: 512x512 pixels")
    print(f"  Format: PNG")
    
    return output_dir

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Generate synthetic faces using StyleGAN2-FFHQ')
    parser.add_argument('--num-images', type=int, default=2000,
                      help='Number of images to generate (default: 2000)')
    parser.add_argument('--output-dir', type=str, default='synthetic_faces',
                      help='Output directory (default: synthetic_faces)')
    parser.add_argument('--seed', type=int, default=42,
                      help='Random seed (default: 42)')
    parser.add_argument('--truncation', type=float, default=0.7,
                      help='Truncation psi, 0.7=realistic, 1.0=diverse (default: 0.7)')
    parser.add_argument('--batch-size', type=int, default=4,
                      help='Batch size, lower if CUDA OOM (default: 4)')
    
    args = parser.parse_args()
    
    print("="*60)
    print("StyleGAN2 Synthetic Face Generator")
    print("="*60)
    print(f"Configuration:")
    print(f"  - Images to generate: {args.num_images}")
    print(f"  - Output directory: {args.output_dir}")
    print(f"  - Random seed: {args.seed}")
    print(f"  - Truncation psi: {args.truncation}")
    print(f"  - Batch size: {args.batch_size}")
    print("="*60)
    
    output_dir = generate_synthetic_faces(
        num_images=args.num_images,
        output_dir=args.output_dir,
        seed=args.seed,
        truncation_psi=args.truncation,
        batch_size=args.batch_size
    )

if __name__ == '__main__':
    main()
