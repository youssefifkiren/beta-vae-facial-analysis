import pandas as pd
from sklearn.model_selection import train_test_split

# Load the full annotations
df = pd.read_csv('annotations_all.csv')

print("Original distribution:")
print(df['label'].value_counts())

# Strategy: Balance to ~500 samples per class
TARGET_SAMPLES = 500

balanced_dfs = []
for label in df['label'].unique():
    label_df = df[df['label'] == label]
    n_samples = len(label_df)
    
    if n_samples >= TARGET_SAMPLES:
        # Undersample large classes
        sampled = label_df.sample(n=TARGET_SAMPLES, random_state=42)
    else:
        # Oversample small classes (with replacement)
        sampled = label_df.sample(n=TARGET_SAMPLES, replace=True, random_state=42)
    
    balanced_dfs.append(sampled)

balanced_df = pd.concat(balanced_dfs, ignore_index=True)

print("\n" + "="*60)
print("NEW BALANCED DISTRIBUTION")
print("="*60)
print(balanced_df['label'].value_counts())
print(f"\nTotal: {len(balanced_df)} images")

# Save new balanced dataset
balanced_df.to_csv('data/annotations_balanced.csv', index=False)

# Create train/val/test splits (70/15/15)
train_df, temp_df = train_test_split(
    balanced_df, 
    test_size=0.3, 
    stratify=balanced_df['label'], 
    random_state=42
)

val_df, test_df = train_test_split(
    temp_df, 
    test_size=0.5, 
    stratify=temp_df['label'], 
    random_state=42
)

# Save splits
train_df.to_csv('data/train.csv', index=False)
val_df.to_csv('data/val.csv', index=False)
test_df.to_csv('data/test.csv', index=False)

print("\n" + "="*60)
print("DATASET SPLITS")
print("="*60)
print(f"Train: {len(train_df)} images ({len(train_df)/len(balanced_df)*100:.1f}%)")
print(f"Val:   {len(val_df)} images ({len(val_df)/len(balanced_df)*100:.1f}%)")
print(f"Test:  {len(test_df)} images ({len(test_df)/len(balanced_df)*100:.1f}%)")

print("\nTrain set distribution:")
print(train_df['label'].value_counts())
