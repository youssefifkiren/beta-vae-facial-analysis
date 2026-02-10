import cv2
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# Only use face and eye cascades (these always exist)
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

def extract_features(image_path):
    """Extract facial features using only face and eye detection"""
    try:
        img = cv2.imread(str(image_path))
        if img is None:
            return None
            
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Detect face
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        if len(faces) == 0:
            return None
        
        # Get largest face
        x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
        roi_gray = gray[y:y+h, x:x+w]
        
        # Detect eyes
        eyes = eye_cascade.detectMultiScale(roi_gray, 1.1, 5)
        if len(eyes) < 2:
            return None
        
        # Get two largest eyes and sort left to right
        eyes = sorted(eyes, key=lambda e: e[2] * e[3], reverse=True)[:2]
        eyes = sorted(eyes, key=lambda e: e[0])
        left_eye = eyes[0]
        right_eye = eyes[1]
        
        # Eye centers
        left_eye_center = np.array([left_eye[0] + left_eye[2]//2, left_eye[1] + left_eye[3]//2])
        right_eye_center = np.array([right_eye[0] + right_eye[2]//2, right_eye[1] + right_eye[3]//2])
        
        # Eye distance and ratio
        eye_distance = np.linalg.norm(left_eye_center - right_eye_center)
        eye_ratio = eye_distance / w
        
        # Face aspect ratio
        face_aspect_ratio = h / w
        
        # Eye vertical position (normalized)
        avg_eye_y = (left_eye_center[1] + right_eye_center[1]) / 2
        eye_position = avg_eye_y / h
        
        # Eye size ratio (relative to face)
        avg_eye_width = (left_eye[2] + right_eye[2]) / 2
        eye_size_ratio = avg_eye_width / w
        
        # Facial symmetry (eye height difference)
        eye_height_diff = abs(left_eye_center[1] - right_eye_center[1])
        asymmetry = eye_height_diff / h
        
        # Eye angle (tilt of line connecting eyes)
        eye_angle = np.abs(np.arctan2(
            right_eye_center[1] - left_eye_center[1],
            right_eye_center[0] - left_eye_center[0]
        ))
        
        # Upper face ratio (from eyes to top)
        upper_face_ratio = avg_eye_y / h
        
        # Lower face ratio (estimated from eyes to bottom)
        lower_face_ratio = (h - avg_eye_y) / h
        
        return {
            'eye_ratio': eye_ratio,
            'face_aspect_ratio': face_aspect_ratio,
            'eye_position': eye_position,
            'eye_size_ratio': eye_size_ratio,
            'asymmetry': asymmetry,
            'eye_angle': eye_angle,
            'upper_face_ratio': upper_face_ratio,
            'lower_face_ratio': lower_face_ratio,
            'face_width': w,
            'face_height': h
        }
    except Exception as e:
        return None

# Process images
image_dir = Path('ffhq_images')
data = []
failed = 0

print("Processing images with facial feature extraction...")
image_files = list(image_dir.glob('*.png'))[:10000]

for img_path in tqdm(image_files):
    features = extract_features(img_path)
    if features:
        features['image_path'] = str(img_path)
        data.append(features)
    else:
        failed += 1

df = pd.DataFrame(data)
print(f"\nSuccessfully processed: {len(df)} images")
print(f"Failed: {failed} images")

# Save raw features
df.to_csv('data/features_raw.csv', index=False)

# Analyze distributions
print("\n" + "="*60)
print("FEATURE STATISTICS")
print("="*60)

feature_cols = ['eye_ratio', 'face_aspect_ratio', 'eye_position', 
                'eye_size_ratio', 'asymmetry', 'lower_face_ratio']

print(df[feature_cols].describe())

# Calculate percentile thresholds
print("\n" + "="*60)
print("PERCENTILE THRESHOLDS")
print("="*60)

thresholds = {}
for feature in feature_cols:
    p20 = df[feature].quantile(0.20)
    p80 = df[feature].quantile(0.80)
    thresholds[f'{feature}_low'] = p20
    thresholds[f'{feature}_high'] = p80
    print(f"{feature:20s} | 20th: {p20:.4f} | 80th: {p80:.4f}")

# Assign labels
def assign_label(row):
    # Hypertelorism: Wide-set eyes
    if row['eye_ratio'] > thresholds['eye_ratio_high']:
        return 'hypertelorism'
    
    # Hypotelorism: Close-set eyes
    elif row['eye_ratio'] < thresholds['eye_ratio_low']:
        return 'hypotelorism'
    
    # Small lower face (micrognathia proxy)
    elif row['lower_face_ratio'] < thresholds['lower_face_ratio_low']:
        return 'short_lower_face'
    
    # Large lower face (prognathism)
    elif row['lower_face_ratio'] > thresholds['lower_face_ratio_high']:
        return 'long_lower_face'
    
    # Facial asymmetry
    elif row['asymmetry'] > thresholds['asymmetry_high']:
        return 'facial_asymmetry'
    
    # Normal
    else:
        return 'normal'

df['label'] = df.apply(assign_label, axis=1)

print("\n" + "="*60)
print("LABEL DISTRIBUTION")
print("="*60)
print(df['label'].value_counts())
print("\nPercentages:")
print(df['label'].value_counts(normalize=True) * 100)

# Save all annotations
df.to_csv('data/annotations_all.csv', index=False)

# Balance classes
min_samples = df['label'].value_counts().min()
print(f"\n" + "="*60)
print(f"BALANCING CLASSES (min samples: {min_samples})")
print("="*60)

if min_samples >= 50:
    balanced_df = df.groupby('label').sample(n=min_samples, random_state=42)
    balanced_df.to_csv('data/annotations_balanced.csv', index=False)
    print(f"Balanced dataset: {len(balanced_df)} images")
    print(balanced_df['label'].value_counts())
else:
    print("Using original distribution")
    df.to_csv('data/annotations_balanced.csv', index=False)
    balanced_df = df

# Train/val/test splits
train_df, temp_df = train_test_split(balanced_df, test_size=0.3, stratify=balanced_df['label'], random_state=42)
val_df, test_df = train_test_split(temp_df, test_size=0.5, stratify=temp_df['label'], random_state=42)

train_df.to_csv('data/train.csv', index=False)
val_df.to_csv('data/val.csv', index=False)
test_df.to_csv('data/test.csv', index=False)

print("\n" + "="*60)
print("DATASET SPLITS")
print("="*60)
print(f"Train: {len(train_df)} images")
print(f"Val:   {len(val_df)} images")
print(f"Test:  {len(test_df)} images")

print("\nTrain set label distribution:")
print(train_df['label'].value_counts())
