# src/utils.py
import math
import numpy as np
import pandas as pd

def angle_between(p1, p2, p3):
    """Angle at p2 between p1-p2-p3 in degrees. p1,p2,p3 are (x,y)."""
    a = np.array(p1) - np.array(p2)
    b = np.array(p3) - np.array(p2)
    denom = (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8)
    cos = np.dot(a, b) / denom
    cos = np.clip(cos, -1.0, 1.0)
    return math.degrees(math.acos(cos))

def normalize_keypoints(kps):
    """kps: list of (x,y) or np.nan -> returns normalized flattened array"""
    arr = np.array(kps, dtype=float)  # shape (K,2)
    # replace nan with mean of visible coords
    mask = np.isnan(arr)
    if mask.any():
        mean_vals = np.nanmean(arr, axis=0)
        arr[mask] = mean_vals
    mins = np.min(arr, axis=0)
    maxs = np.max(arr, axis=0)
    rng = np.maximum(maxs - mins, 1e-6)
    norm = (arr - mins) / rng
    return norm.flatten()

def seq_to_dataset(seq_list, labels, seq_len=48):
    """
    seq_list: list of np.arrays shape (T,2K)
    labels: list of 0/1 labels per seq
    Pads/truncates each seq to seq_len frames.
    Returns X (N, seq_len, 2K), y (N,)
    """
    K2 = seq_list[0].shape[1]
    N = len(seq_list)
    X = np.zeros((N, seq_len, K2), dtype=float)
    for i, s in enumerate(seq_list):
        t = s.shape[0]
        if t >= seq_len:
            X[i] = s[:seq_len]
        else:
            # pad by repeating last frame
            pad = np.tile(s[-1:], (seq_len - t, 1))
            X[i] = np.vstack([s, pad])
    y = np.array(labels, dtype=int)
    return X, y
