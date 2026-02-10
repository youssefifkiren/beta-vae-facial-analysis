"""
Enhanced Visual Evaluation - Beautiful Grid of Predictions

Creates a comprehensive visualization showing:
- Multiple test images per disorder class
- Ground truth vs predicted labels
- Color-coded correct/incorrect predictions
- Human-readable disorder names
- Confidence scores

Usage:
    python visual_evaluation.py
    python visual_evaluation.py --samples-per-class 8
"""

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from torchvision import transforms
from PIL import Image
import os
import argparse

from train_beta_vae import BetaVAE, FaceDataset
from classifier_training import DisorderClassifier

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Human-readable disorder names
DISORDER_NAMES = {
    'facial_asymmetry': 'Facial\nAsymmetry',
    'hypertelorism': 'Hypertelorism\n(Wide Eyes)',
    'hypotelorism': 'Hypotelorism\n(Close Eyes)',
    'long_lower_face': 'Long\nLower Face',
    'normal': 'Normal',
    'short_lower_face': 'Short\nLower Face'
}

CLASS_NAMES = list(DISORDER_NAMES.keys())

def load_models(vae_path='models/beta_vae_final.pth', classifier_path='models/disorder_classifier_best.pth'):
    """Load trained models"""
    vae = BetaVAE(latent_dim=128).to(device)
    vae.load_state_dict(torch.load(vae_path, map_location=device))
    vae.eval()
    
    classifier = DisorderClassifier(latent_dim=128, num_classes=6).to(device)
    classifier.load_state_dict(torch.load(classifier_path, map_location=device))
    classifier.eval()
    
    return vae, classifier

def predict_image(vae, classifier, image_tensor):
    """Make prediction on single image"""
    with torch.no_grad():
        image_tensor = image_tensor.to(device)
        mu, _ = vae.encode(image_tensor)
        outputs = classifier(mu)
        probabilities = torch.softmax(outputs, dim=1)
        confidence, predicted_class = probabilities.max(1)
        
        return predicted_class.item(), confidence.item()

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

def create_evaluation_grid(vae, classifier, samples_dict, img_size=128):
    """Create beautiful evaluation grid"""
    
    transform = transforms.Compose([
        transforms.Resize(img_size),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
    ])
    
    num_classes = len(CLASS_NAMES)
    samples_per_class = max(len(samples) for samples in samples_dict.values())
    
    # Create figure with extra space for labels
    fig = plt.figure(figsize=(samples_per_class * 2.5, num_classes * 2.5))
    gs = fig.add_gridspec(num_classes, samples_per_class, hspace=0.3, wspace=0.2)
    
    print(f"\nGenerating predictions for {num_classes} classes × {samples_per_class} samples...")
    
    correct_count = 0
    total_count = 0
    
    for class_idx, class_name in enumerate(CLASS_NAMES):
        samples = samples_dict[class_name]
        
        print(f"  Processing {class_name}...")
        
        for sample_idx in range(len(samples)):
            row = samples.iloc[sample_idx]
            image_path = row['image_path']
            true_label = row['label']
            
            # Load and predict
            image = Image.open(image_path).convert('RGB')
            image_tensor = transform(image).unsqueeze(0)
            predicted_class_idx, confidence = predict_image(vae, classifier, image_tensor)
            predicted_label = CLASS_NAMES[predicted_class_idx]
            
            # Track accuracy
            is_correct = (predicted_label == true_label)
            if is_correct:
                correct_count += 1
            total_count += 1
            
            # Plot
            ax = fig.add_subplot(gs[class_idx, sample_idx])
            ax.imshow(image)
            ax.axis('off')
            
            # Color code: green = correct, red = incorrect
            border_color = 'green' if is_correct else 'red'
            linewidth = 3 if is_correct else 2
            
            # Add colored border
            for spine in ax.spines.values():
                spine.set_edgecolor(border_color)
                spine.set_linewidth(linewidth)
                spine.set_visible(True)
            
            # Add prediction text
            if is_correct:
                pred_text = f" {confidence:.0%}"
                text_color = 'green'
            else:
                pred_short = predicted_label.replace('_', '\n')
                pred_text = f" {pred_short}\n{confidence:.0%}"
                text_color = 'red'
            
            ax.text(0.5, -0.1, pred_text, 
                   transform=ax.transAxes,
                   ha='center', va='top',
                   fontsize=8, 
                   fontweight='bold',
                   color=text_color,
                   bbox=dict(boxstyle='round,pad=0.3', 
                           facecolor='white', 
                           edgecolor=border_color,
                           alpha=0.8))
            
            # Add class label on first column
            if sample_idx == 0:
                ax.text(-0.15, 0.5, DISORDER_NAMES[class_name],
                       transform=ax.transAxes,
                       ha='right', va='center',
                       fontsize=11,
                       fontweight='bold',
                       color='black')
    
    # Overall title with accuracy
    accuracy = (correct_count / total_count) * 100
    fig.suptitle(f'Visual Evaluation: Predictions on Test Set\n'
                f'Overall Accuracy: {accuracy:.1f}% ({correct_count}/{total_count} correct)',
                fontsize=16, fontweight='bold', y=0.98)
    
    # Add legend
    correct_patch = mpatches.Patch(color='green', label='Correct Prediction')
    incorrect_patch = mpatches.Patch(color='red', label='Incorrect Prediction')
    plt.legend(handles=[correct_patch, incorrect_patch], 
              loc='upper center', 
              bbox_to_anchor=(0.5, -0.02),
              ncol=2,
              frameon=True,
              fontsize=10)
    
    return fig, accuracy

def main():
    parser = argparse.ArgumentParser(description='Visual evaluation with multiple samples')
    parser.add_argument('--vae-model', type=str, default='models/beta_vae_final.pth',
                       help='Path to VAE model')
    parser.add_argument('--classifier-model', type=str, default='models/disorder_classifier_best.pth',
                       help='Path to classifier model')
    parser.add_argument('--test-csv', type=str, default='data/test.csv',
                       help='Test dataset CSV')
    parser.add_argument('--samples-per-class', type=int, default=6,
                       help='Number of samples to show per class')
    parser.add_argument('--output', type=str, default='visualizations/visual_evaluation.png',
                       help='Output image path')
    
    args = parser.parse_args()
    
    print("="*60)
    print("VISUAL EVALUATION")
    print("="*60)
    print(f"Device: {device}")
    print(f"Samples per class: {args.samples_per_class}\n")
    
    # Load models
    print("Loading models...")
    vae, classifier = load_models(args.vae_model, args.classifier_model)
    print(" Models loaded\n")
    
    # Get test samples
    print("Loading test samples...")
    samples_dict = get_samples_per_class(args.test_csv, args.samples_per_class)
    print(f" Loaded {sum(len(s) for s in samples_dict.values())} total samples\n")
    
    # Generate visualization
    fig, accuracy = create_evaluation_grid(vae, classifier, samples_dict)
    
    # Save
    fig.savefig(args.output, dpi=150, bbox_inches='tight')
    print(f"\n Saved visualization to: {args.output}")
    
    plt.show()
    
    print("\n" + "="*60)
    print("COMPLETE!")
    print("="*60)
    print(f"\nThis visualization shows {args.samples_per_class} random samples")
    print("from each disorder class with:")
    print("  • Green borders = Correct predictions")
    print("  • Red borders = Incorrect predictions")
    print("  • Confidence scores shown below each image")
    print(f"  • Overall accuracy: {accuracy:.1f}%")

if __name__ == '__main__':
    main()
