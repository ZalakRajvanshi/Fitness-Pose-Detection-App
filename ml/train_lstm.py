"""
train_lstm.py — does temporal context help? A sequence model vs the per-frame one.

An exercise is a *motion*, so the natural question is whether feeding a short
window of frames into an LSTM beats classifying each frame independently. We
build sequences honestly:

  - slide a window (length L) over frames WITHIN a segment (never across a clip
    boundary from groups.py), so a window is real contiguous motion;
  - the window label is the majority frame label;
  - split BY SEGMENT (a whole clip is train or test, never both).

We compare, at the window level and under the same by-segment split:
  - LSTM over the window
  - per-frame logistic regression, aggregated to a window by majority vote.

Honest expectation: on plank (a static hold) form is readable from a single
posture, so temporal context should give little. The value of this script is the
method — group-aware sequence construction + a fair comparison — and reporting
the result straight, whatever it is.

    python ml/train_lstm.py            # plank
"""
import os
import sys
import warnings

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import tensorflow as tf

from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import accuracy_score, classification_report

from groups import infer_segments, xy_features


def majority(arr):
    """Most frequent integer label in arr."""
    return np.bincount(np.asarray(arr).astype(int)).argmax()

tf.get_logger().setLevel("ERROR")
tf.random.set_seed(42)
np.random.seed(42)

HERE = os.path.dirname(__file__)
RESULTS = os.path.join(HERE, "results")
WIN = 32      # window length (frames)
STRIDE = 8


def build_windows(X, y, seg, win=WIN, stride=STRIDE):
    """Sliding windows within each segment. Returns Xw (n,win,F), yw, gw."""
    Xw, yw, gw = [], [], []
    for s in np.unique(seg):
        idx = np.where(seg == s)[0]
        if len(idx) < win:
            continue
        for start in range(0, len(idx) - win + 1, stride):
            w = idx[start:start + win]
            Xw.append(X[w])
            yw.append(majority(y[w]))
            gw.append(s)
    return np.array(Xw), np.array(yw), np.array(gw)


def main(exercise="plank"):
    df = pd.read_csv(os.path.join(HERE, "data_ext", exercise, "train.csv"))
    feat = xy_features(df)
    le = LabelEncoder().fit(df["label"])
    X = df[feat].values.astype("float32")
    y = le.transform(df["label"])
    seg = infer_segments(df)
    names = list(le.classes_)

    # standardise on the whole pool (stats only; split is by segment below)
    scaler = StandardScaler().fit(X)
    Xs = scaler.transform(X).astype("float32")

    Xw, yw, gw = build_windows(Xs, y, seg)
    print(f"{exercise}: {len(Xw)} windows of {WIN} frames, {Xw.shape[2]} features, "
          f"classes={names}")

    # by-segment split
    gss = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=42)
    tr, te = next(gss.split(Xw, yw, gw))
    Xtr, Xte, ytr, yte = Xw[tr], Xw[te], yw[tr], yw[te]

    # ---- LSTM ----
    lstm = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(WIN, Xw.shape[2])),
        tf.keras.layers.LSTM(64),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(32, activation="relu"),
        tf.keras.layers.Dense(len(names), activation="softmax"),
    ])
    lstm.compile(optimizer="adam", loss="sparse_categorical_crossentropy",
                 metrics=["accuracy"])
    es = tf.keras.callbacks.EarlyStopping(patience=6, restore_best_weights=True)
    lstm.fit(Xtr, ytr, validation_split=0.15, epochs=40, batch_size=32,
             verbose=0, callbacks=[es])
    lstm_pred = lstm.predict(Xte, verbose=0).argmax(1)
    lstm_acc = accuracy_score(yte, lstm_pred)

    # ---- per-frame baseline, aggregated to window by majority vote ----
    # train on individual frames of the training segments
    train_segments = set(gw[tr])
    frame_mask = np.array([s in train_segments for s in seg])
    base = LogisticRegression(max_iter=2000).fit(Xs[frame_mask], y[frame_mask])
    # predict each window's frames, majority vote
    base_pred = np.array([
        majority(base.predict(win)) for win in Xte
    ])
    base_acc = accuracy_score(yte, base_pred)

    print(f"  Per-frame LogReg (window majority vote): {base_acc:.3f}")
    print(f"  LSTM (temporal window):                  {lstm_acc:.3f}")

    rep = classification_report(yte, lstm_pred, target_names=names, digits=3)
    verdict = ("temporal context helped" if lstm_acc > base_acc + 0.005
               else "temporal context did not beat per-frame "
                    "(expected for a near-static posture)")
    with open(os.path.join(RESULTS, f"{exercise}_lstm.md"), "w", encoding="utf-8") as f:
        f.write(f"# {exercise.capitalize()} — temporal (LSTM) vs per-frame\n\n")
        f.write(f"By-segment split, {len(Xw)} windows of {WIN} frames.\n\n")
        f.write("| Model | Window accuracy |\n|---|---:|\n")
        f.write(f"| Per-frame LogReg (majority vote) | {base_acc:.1%} |\n")
        f.write(f"| LSTM (32-frame window) | {lstm_acc:.1%} |\n\n")
        f.write(f"**Finding:** {verdict}. The pipeline (group-aware sequence "
                f"construction + fair comparison) is the transferable part — on a "
                f"dynamic, motion-dependent error it is where temporal models pay "
                f"off.\n\n## LSTM per-class\n\n```\n" + rep + "```\n")
    print(f"  saved -> results/{exercise}_lstm.md")

    # plot
    plt.figure(figsize=(5, 4))
    bars = plt.bar(["Per-frame\nLogReg", "LSTM\n(temporal)"],
                   [base_acc, lstm_acc], color=["#94a3b8", "#5eead4"])
    for b, v in zip(bars, [base_acc, lstm_acc]):
        plt.text(b.get_x() + b.get_width() / 2, v + 0.01, f"{v:.1%}",
                 ha="center", fontweight="bold")
    plt.ylim(0, 1.06)
    plt.ylabel("Window accuracy (by-segment split)")
    plt.title(f"{exercise.capitalize()} — temporal vs per-frame")
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS, f"{exercise}_lstm.png"), dpi=130)
    plt.close()


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "plank")
