# extract_keywords.py
import os
import cv2
import numpy as np
import mediapipe as mp

mp_pose = mp.solutions.pose

DATA_DIR = "data"
RAW_DIR = os.path.join(DATA_DIR, "raw")
CORRECT_DIR = os.path.join(RAW_DIR, "correct")
INCORRECT_DIR = os.path.join(RAW_DIR, "incorrect")

def extract_keypoints(image_path):
    """Extract 2D pose keypoints using MediaPipe"""
    image = cv2.imread(image_path)
    if image is None:
        print(f"Could not read {image_path}")
        return None

    with mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.5) as pose:
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = pose.process(image_rgb)
        if not results.pose_landmarks:
            print(f"No pose detected in {image_path}")
            return None
        
        keypoints = []
        for lm in results.pose_landmarks.landmark:
            keypoints.extend([lm.x, lm.y])  # only X, Y
        return np.array(keypoints)

def augment_keypoints(keypoints, num_aug=5, noise=0.02):
    """Generate augmented keypoints by adding small jitter"""
    augmented = []
    for _ in range(num_aug):
        jitter = np.random.normal(0, noise, size=keypoints.shape)
        aug = keypoints + jitter
        augmented.append(aug)
    return augmented

def process_folder(folder_path, label):
    keypoints_list = []
    labels_list = []

    for filename in os.listdir(folder_path):
        if filename.lower().endswith((".jpg", ".jpeg", ".png")):
            path = os.path.join(folder_path, filename)
            keypoints = extract_keypoints(path)
            if keypoints is not None:
                # original
                keypoints_list.append(keypoints)
                labels_list.append(label)

                # augmented
                for aug in augment_keypoints(keypoints, num_aug=10):
                    keypoints_list.append(aug)
                    labels_list.append(label)
            else:
                print(f"Skipping {filename} - no keypoints detected")

    return keypoints_list, labels_list

def main():
    os.makedirs(DATA_DIR, exist_ok=True)

    X_correct, y_correct = process_folder(CORRECT_DIR, label=1)
    X_incorrect, y_incorrect = process_folder(INCORRECT_DIR, label=0)

    X = np.array(X_correct + X_incorrect)
    y = np.array(y_correct + y_incorrect)

    np.save(os.path.join(DATA_DIR, "X.npy"), X)
    np.save(os.path.join(DATA_DIR, "y.npy"), y)

    print(f"Saved X.npy with shape {X.shape}")
    print(f"Saved y.npy with shape {y.shape}")

if __name__ == "__main__":
    main()
