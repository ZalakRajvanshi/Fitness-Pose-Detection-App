"""
groups.py — recover clip/person structure the flat CSV threw away.

The dataset is frames concatenated from many short clips. Within a clip,
consecutive frames are almost identical (median frame-to-frame distance ~0.004);
between clips there is a large jump. We detect those jumps to label contiguous
segments — an approximate "clip id" we can group on so frames from one clip
never straddle a train/test split.

This is an approximation of a true by-person split (the data has no person ids),
and we label it as such in the writeup. It is still far more honest than a random
frame split, which leaks near-duplicate neighbouring frames across the split.
"""
import numpy as np


def xy_features(df):
    return [c for c in df.columns if c.endswith("_x") or c.endswith("_y")]


def infer_segments(df, thresh=0.2):
    """Return an int array of segment (≈clip) ids, one per row."""
    feat = xy_features(df)
    X = df[feat].values
    dist = np.linalg.norm(np.diff(X, axis=0), axis=1)
    boundaries = dist > thresh                  # True where a new clip starts
    seg = np.zeros(len(df), dtype=int)
    seg[1:] = np.cumsum(boundaries)
    return seg


if __name__ == "__main__":
    import pandas as pd, os
    for ex in ("plank", "squat"):
        df = pd.read_csv(os.path.join(os.path.dirname(__file__), "data_ext", ex, "train.csv"))
        seg = infer_segments(df)
        sizes = np.bincount(seg)
        print(f"{ex}: {seg.max()+1} segments, "
              f"median {int(np.median(sizes))} frames, "
              f"min {sizes.min()}, max {sizes.max()}")
