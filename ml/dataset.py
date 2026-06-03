"""
dataset.py — keypoint extraction with provenance.

The key difference from the original src/extract_keypoints.py:
we keep a *group id* (the source image) for every sample. That lets the
trainer split by source image, so augmented copies of one photo can never
appear in both train and test. Without this, the model "memorises" jittered
duplicates and reports an inflated accuracy (the leakage the old pipeline had).
"""
import os
import numpy as np
import cv2
import mediapipe as mp

mp_pose = mp.solutions.pose

RAW_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "raw")
CLASSES = {"correct": 1, "incorrect": 0}


def extract_keypoints(image_path):
    """Return a (66,) array of normalized x,y for 33 landmarks, or None."""
    image = cv2.imread(image_path)
    if image is None:
        return None
    with mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.5) as pose:
        results = pose.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        if not results.pose_landmarks:
            return None
        kp = []
        for lm in results.pose_landmarks.landmark:
            kp.extend([lm.x, lm.y])
        return np.asarray(kp, dtype=np.float32)


def load_raw():
    """
    Extract keypoints from every raw image.

    Returns
    -------
    X      : (N, 66) keypoints, one row per source image
    y      : (N,)    labels (1 correct, 0 incorrect)
    groups : (N,)    integer source-image id (here, one per image)
    """
    X, y, groups = [], [], []
    gid = 0
    for cls, label in CLASSES.items():
        folder = os.path.join(RAW_DIR, cls)
        for fname in sorted(os.listdir(folder)):
            if not fname.lower().endswith((".png", ".jpg", ".jpeg")):
                continue
            kp = extract_keypoints(os.path.join(folder, fname))
            if kp is None:
                print(f"[skip] no pose detected: {cls}/{fname}")
                continue
            X.append(kp)
            y.append(label)
            groups.append(gid)
            gid += 1
    return np.array(X), np.array(y), np.array(groups)


def augment(X, y, groups, n_aug=12, noise=0.02, seed=0):
    """
    Add jittered copies of each row. Crucially this is called *after* the
    train/test split, on the training set only — so test rows stay pristine.
    Augmented rows inherit the group id of their parent (kept for bookkeeping).
    """
    rng = np.random.default_rng(seed)
    Xa, ya, ga = [X], [y], [groups]
    for _ in range(n_aug):
        Xa.append(X + rng.normal(0, noise, size=X.shape).astype(np.float32))
        ya.append(y)
        ga.append(groups)
    return np.vstack(Xa), np.concatenate(ya), np.concatenate(ga)


if __name__ == "__main__":
    X, y, g = load_raw()
    print(f"Extracted {X.shape[0]} source images, {X.shape[1]} features each")
    print(f"  correct={int((y==1).sum())}  incorrect={int((y==0).sum())}")
