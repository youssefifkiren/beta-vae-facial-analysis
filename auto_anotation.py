import mediapipe as mp
import cv2
import numpy as np
import pandas as pd
from pathlib import Path

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True)

def extract_landmarks(image_path):
    image = cv2.imread(str(image_path))
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(image_rgb)
    
    if not results.multi_face_landmarks:
        return None
    
    landmarks = results.multi_face_landmarks[0]
    return [(lm.x, lm.y, lm.z) for lm in landmarks.landmark]

def calculate_features(landmarks):
    """Calculate measurable facial features"""
    landmarks = np.array(landmarks)
    
    # Eye distance (inter-pupillary distance)
    left_eye = landmarks[33]  # left eye center
    right_eye = landmarks[263]  # right eye center
    eye_distance = np.linalg.norm(left_eye - right_eye)
    
    # Face width (temple to temple)
    left_temple = landmarks[127]
    right_temple = landmarks[356]
    face_width = np.linalg.norm(left_temple - right_temple)
    
    # Eye distance ratio (hypertelorism indicator)
    eye_ratio = eye_distance / face_width
    
    # Nose bridge height
    nose_bridge = landmarks[168]
    nose_tip = landmarks[1]
    nose_height = abs(nose_bridge[1] - nose_tip[1])
    
    # Chin projection
    chin = landmarks[152]
    forehead = landmarks[10]
    chin_projection = abs(chin[2] - forehead[2])
    
    # Facial asymmetry
    left_points = landmarks[[127, 234, 93, 132]]
    right_points = landmarks[[356, 454, 323, 361]]
    asymmetry = np.mean([np.linalg.norm(l - r) for l, r in zip(left_points, right_points)])
    
    return {
        'eye_ratio': eye_ratio,
        'nose_height': nose_height,
        'chin_projection': chin_projection,
        'asymmetry': asymmetry
    }

def assign_synthetic_disorder(features):
    """Assign labels based on feature thresholds"""
    # These thresholds are synthetic - adjust based on your data distribution
    
    if features['eye_ratio'] > 0.55:
        return 'hypertelorism'  # wide-set eyes
    elif features['nose_height'] < 0.15:
        return 'flat_nasal_bridge'
    elif features['chin_projection'] < 0.02:
        return 'micrognathia'  # small chin
    elif features['asymmetry'] > 0.08:
        return 'facial_asymmetry'
    else:
        return 'normal'

# Process all images
image_dir = Path('ffhq_images')
data = []

for img_path in image_dir.glob('*.png'):
    landmarks = extract_landmarks(img_path)
    if landmarks:
        features = calculate_features(landmarks)
        label = assign_synthetic_disorder(features)
        data.append({
            'image_path': str(img_path),
            'label': label,
            **features
        })

df = pd.DataFrame(data)
df.to_csv('annotations.csv', index=False)
print(df['label'].value_counts())