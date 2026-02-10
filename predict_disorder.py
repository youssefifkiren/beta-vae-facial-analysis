"""
Facial Disorder Prediction - Inference Script

Usage:
    python predict_disorder.py --image path/to/face.jpg
    python predict_disorder.py --image path/to/face.jpg --show-viz
"""

import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import argparse
import os
import sys
import matplotlib.pyplot as plt
import numpy as np

# Import models
from train_beta_vae import BetaVAE
from classifier_training import DisorderClassifier

# Device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Class names
CLASS_NAMES = [
    'facial_asymmetry',
    'hypertelorism',
    'hypotelorism',
    'long_lower_face',
    'normal',
    'short_lower_face'
]

def load_models(vae_path='models/beta_vae_final.pth', classifier_path='models/disorder_classifier_best.pth'):
    """Load trained models"""
    print("Loading models...")
    
    # Load VAE
    vae = BetaVAE(latent_dim=128).to(device)
    vae.load_state_dict(torch.load(vae_path, map_location=device))
    vae.eval()
    
    # Load classifier
    classifier = DisorderClassifier(latent_dim=128, num_classes=6).to(device)
    classifier.load_state_dict(torch.load(classifier_path, map_location=device))
    classifier.eval()
    
    print(" Models loaded successfully!\n")
    return vae, classifier

def preprocess_image(image_path, img_size=128):
    """Load and preprocess image"""
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")
    
    # Load image
    image = Image.open(image_path).convert('RGB')
    
    # Transform
    transform = transforms.Compose([
        transforms.Resize(img_size),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
    ])
    
    image_tensor = transform(image).unsqueeze(0)  # Add batch dimension
    return image_tensor, image

def predict(vae, classifier, image_tensor):
    """Make prediction with confidence scores"""
    with torch.no_grad():
        # Get latent representation
        image_tensor = image_tensor.to(device)
        mu, logvar = vae.encode(image_tensor)
        
        # Get classification
        outputs = classifier(mu)
        probabilities = torch.softmax(outputs, dim=1)
        
        # Get prediction
        confidence, predicted_class = probabilities.max(1)
        
        return predicted_class.item(), confidence.item(), probabilities[0].cpu().numpy()

def visualize_prediction(image, predicted_class, confidence, all_probs, save_path=None):
    """Create visualization of prediction"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Show image
    ax1.imshow(image)
    ax1.axis('off')
    ax1.set_title(f'Input Image\nPrediction: {CLASS_NAMES[predicted_class]}\nConfidence: {confidence:.1%}', 
                  fontsize=12, fontweight='bold')
    
    # Show probability distribution
    colors = ['red' if i == predicted_class else 'skyblue' for i in range(len(CLASS_NAMES))]
    bars = ax2.barh(CLASS_NAMES, all_probs * 100, color=colors)
    ax2.set_xlabel('Probability (%)', fontsize=11)
    ax2.set_title('Disorder Probabilities', fontsize=12, fontweight='bold')
    ax2.set_xlim(0, 100)
    
    # Add percentage labels
    for bar, prob in zip(bars, all_probs):
        width = bar.get_width()
        ax2.text(width + 2, bar.get_y() + bar.get_height()/2, 
                f'{prob*100:.1f}%', 
                ha='left', va='center', fontsize=9)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f" Visualization saved to: {save_path}")
    
    plt.show()

def main():
    parser = argparse.ArgumentParser(description='Predict facial disorder from image')
    parser.add_argument('--image', type=str, required=True, help='Path to face image')
    parser.add_argument('--vae-model', type=str, default='models/beta_vae_final.pth',
                       help='Path to VAE model')
    parser.add_argument('--classifier-model', type=str, default='models/disorder_classifier_best.pth',
                       help='Path to classifier model')
    parser.add_argument('--show-viz', action='store_true',
                       help='Show visualization of prediction')
    parser.add_argument('--save-viz', type=str, default=None,
                       help='Path to save visualization')
    
    args = parser.parse_args()
    
    print("="*60)
    print("FACIAL DISORDER PREDICTION")
    print("="*60)
    print(f"Device: {device}")
    print(f"Image: {args.image}\n")
    
    # Load models
    vae, classifier = load_models(args.vae_model, args.classifier_model)
    
    # Preprocess image
    print("Processing image...")
    image_tensor, original_image = preprocess_image(args.image)
    
    # Make prediction
    print("Making prediction...\n")
    predicted_class, confidence, all_probs = predict(vae, classifier, image_tensor)
    
    # Print results
    print("="*60)
    print("PREDICTION RESULTS")
    print("="*60)
    print(f"Predicted Disorder: {CLASS_NAMES[predicted_class].upper().replace('_', ' ')}")
    print(f"Confidence: {confidence:.2%}")
    print("\nAll Probabilities:")
    print("-"*60)
    
    # Sort by probability
    sorted_indices = np.argsort(all_probs)[::-1]
    for idx in sorted_indices:
        prob = all_probs[idx]
        marker = "" if idx == predicted_class else " "
        print(f"{marker} {CLASS_NAMES[idx]:20s} : {prob:6.2%}  {'' * int(prob * 50)}")
    
    print("="*60)
    
    # Visualization
    if args.show_viz or args.save_viz:
        save_path = args.save_viz if args.save_viz else f"prediction_{os.path.basename(args.image)}.png"
        visualize_prediction(original_image, predicted_class, confidence, all_probs, save_path)
    
    print("\n Prediction complete!")

if __name__ == '__main__':
    main()
