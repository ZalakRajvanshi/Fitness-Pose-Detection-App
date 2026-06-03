"""
export_web.py — export a trained classifier to a compact JSON the browser runs.

We ship the logistic-regression model: on the robust x,y-only feature set it
matched or beat the neural net (plank 99.6% vs 97%, squat ~99%), and a linear
model is a single matmul + softmax — tiny, fast, and trivial to verify in JS.
No TensorFlow.js runtime needed in the browser.

Output: web/models/<exercise>.json with
  feature_plan : [[landmark_index, "x"|"y"], ...]   how to build the input vector
  scaler       : {mean, scale}                       StandardScaler params
  layers       : [{W, b, activation}]                here one softmax layer
  classes,names: label codes + human names

    python ml/export_web.py            # plank
    python ml/export_web.py squat
"""
import os
import sys
import json
import warnings

warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression

from train_real import load, CLASS_NAMES  # reuse identical feature selection

HERE = os.path.dirname(__file__)
WEB_MODELS = os.path.join(HERE, "..", "web", "models")
os.makedirs(WEB_MODELS, exist_ok=True)

# MediaPipe BlazePose 33-landmark names -> index.
LM = [
    "nose", "left_eye_inner", "left_eye", "left_eye_outer", "right_eye_inner",
    "right_eye", "right_eye_outer", "left_ear", "right_ear", "mouth_left",
    "mouth_right", "left_shoulder", "right_shoulder", "left_elbow",
    "right_elbow", "left_wrist", "right_wrist", "left_pinky", "right_pinky",
    "left_index", "right_index", "left_thumb", "right_thumb", "left_hip",
    "right_hip", "left_knee", "right_knee", "left_ankle", "right_ankle",
    "left_heel", "right_heel", "left_foot_index", "right_foot_index",
]
LM_INDEX = {name: i for i, name in enumerate(LM)}


def feature_plan(features):
    """Turn ['nose_x','nose_y',...] into [[0,'x'],[0,'y'],...] for the browser."""
    plan = []
    for f in features:
        axis = f[-1]                # 'x' or 'y'
        name = f[:-2]               # strip '_x'
        plan.append([LM_INDEX[name], axis])
    return plan


def main(exercise="plank"):
    tr, te, feat = load(exercise)
    le = LabelEncoder().fit(tr["label"])
    ytr = le.transform(tr["label"])
    classes = list(le.classes_)
    names = [CLASS_NAMES.get(c, c) for c in classes]

    scaler = StandardScaler().fit(tr[feat])
    Xtr = scaler.transform(tr[feat])
    clf = LogisticRegression(max_iter=2000).fit(Xtr, ytr)

    # Normalise to a softmax layer: W (n_feat x n_class), b (n_class).
    if len(classes) == 2:
        # binary sklearn gives one row; softmax([0, w·x+b]) == sigmoid(w·x+b).
        W = np.vstack([np.zeros_like(clf.coef_[0]), clf.coef_[0]]).T
        b = np.array([0.0, clf.intercept_[0]])
    else:
        W = clf.coef_.T
        b = clf.intercept_

    payload = {
        "exercise": exercise,
        "feature_plan": feature_plan(feat),
        "scaler": {"mean": scaler.mean_.tolist(), "scale": scaler.scale_.tolist()},
        "layers": [{"W": W.tolist(), "b": b.tolist(), "activation": "softmax"}],
        "classes": classes,
        "names": names,
    }
    out = os.path.join(WEB_MODELS, f"{exercise}.json")
    with open(out, "w") as f:
        json.dump(payload, f)
    size_kb = os.path.getsize(out) / 1024
    print(f"{exercise}: {len(feat)} features, {len(classes)} classes "
          f"-> {out} ({size_kb:.1f} KB)")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "plank")
