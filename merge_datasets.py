"""
Merge Original and Synthetic Datasets

This script merges the original FFHQ annotations with synthetic StyleGAN2 annotations
to create an augmented training dataset.
"""

import pandas as pd
import argparse
from pathlib import Path

def merge_datasets(
    original_train='train.csv',
    synthetic_csv='synthetic_annotations.csv',
    output_csv='train_augmented.csv',
    balance_classes=True,
    target_samples_per_class=500
):
    """
    Merge original and synthetic training data
    
    Args:
        original_train: Path to original training CSV
        synthetic_csv: Path to synthetic annotations CSV
        output_csv: Output path for merged CSV
        balance_classes: Whether to balance class distribution
        target_samples_per_class: Target number of samples per class
    """
    
    print("="*60)
    print("Dataset Merger")
    print("="*60)
    
    # Load datasets
    print(f"\nLoading {original_train}...")
    original_df = pd.read_csv(original_train)
    print(f"  Original train samples: {len(original_df)}")
    print(f"  Classes: {original_df['label'].nunique()}")
    
    print(f"\nLoading {synthetic_csv}...")
    synthetic_df = pd.read_csv(synthetic_csv)
    print(f"  Synthetic samples: {len(synthetic_df)}")
    print(f"  Classes: {synthetic_df['label'].nunique()}")
    
    # Show original distribution
    print("\n" + "-"*60)
    print("ORIGINAL TRAINING DISTRIBUTION")
    print("-"*60)
    print(original_df['label'].value_counts().sort_index())
    
    print("\n" + "-"*60)
    print("SYNTHETIC DATA DISTRIBUTION")
    print("-"*60)
    print(synthetic_df['label'].value_counts().sort_index())
    
    if balance_classes:
        print("\n" + "-"*60)
        print(f"BALANCING TO {target_samples_per_class} SAMPLES PER CLASS")
        print("-"*60)
        
        balanced_dfs = []
        
        for label in original_df['label'].unique():
            # Get original samples for this class
            orig_samples = original_df[original_df['label'] == label]
            synth_samples = synthetic_df[synthetic_df['label'] == label]
            
            total_available = len(orig_samples) + len(synth_samples)
            
            # How many samples do we need?
            if total_available >= target_samples_per_class:
                # Use all original + augment with synthetic
                needed_synthetic = target_samples_per_class - len(orig_samples)
                
                if needed_synthetic > 0:
                    if len(synth_samples) >= needed_synthetic:
                        synth_subset = synth_samples.sample(n=needed_synthetic, random_state=42)
                    else:
                        synth_subset = synth_samples  # Use all available
                    
                    class_samples = pd.concat([orig_samples, synth_subset])
                else:
                    # Already have enough original samples
                    class_samples = orig_samples.sample(n=target_samples_per_class, random_state=42)
            else:
                # Use everything available
                class_samples = pd.concat([orig_samples, synth_samples])
            
            balanced_dfs.append(class_samples)
            
            print(f"{label:20s}: {len(orig_samples):4d} orig + {len(class_samples)-len(orig_samples):4d} synth = {len(class_samples):4d} total")
        
        merged_df = pd.concat(balanced_dfs, ignore_index=True)
    else:
        # Simple concatenation
        merged_df = pd.concat([original_df, synthetic_df], ignore_index=True)
    
    # Shuffle
    merged_df = merged_df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    # Save
    merged_df.to_csv(output_csv, index=False)
    
    print("\n" + "="*60)
    print("FINAL AUGMENTED DATASET")
    print("="*60)
    print(f"Total samples: {len(merged_df)}")
    print("\nClass distribution:")
    print(merged_df['label'].value_counts().sort_index())
    print(f"\n Saved to {output_csv}")
    
    return merged_df

def main():
    parser = argparse.ArgumentParser(description='Merge original and synthetic datasets')
    parser.add_argument('--original-train', type=str, default='train.csv',
                      help='Original training CSV')
    parser.add_argument('--synthetic-csv', type=str, default='synthetic_annotations.csv',
                      help='Synthetic annotations CSV')
    parser.add_argument('--output-csv', type=str, default='train_augmented.csv',
                      help='Output merged CSV')
    parser.add_argument('--no-balance', action='store_true',
                      help='Do not balance classes')
    parser.add_argument('--target-samples', type=int, default=500,
                      help='Target samples per class when balancing')
    
    args = parser.parse_args()
    
    merge_datasets(
        original_train=args.original_train,
        synthetic_csv=args.synthetic_csv,
        output_csv=args.output_csv,
        balance_classes=not args.no_balance,
        target_samples_per_class=args.target_samples
    )

if __name__ == '__main__':
    main()
