"""
train_real.py — form classifier trained on a real, public dataset.

Dataset: NgoQuocBao1010/Exercise-Correction (MIT) — MediaPipe pose keypoints
already extracted to CSV, with a held-out train/test split provided by the
authors. We respect that split (no leakage from our side) and evaluate honestly.

  plank : 3-class form quality  -> C (correct) / L (hips too low) / H (hips too high)
  squat : 2-class rep stage     -> up / down   (drives rep counting)

For each exercise we compare a logistic-regression baseline against a small
neural net under the same protocol, then save the model + artifacts so it can
be exported to TensorFlow.js for the web app.

Usage:
    python ml/train_real.py            # trains plank (the form classifier)
    python ml/train_real.py squat      # trains squat stage classifier
"""
import os
import sys
import json
import warnings

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf

from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

tf.get_logger().setLevel("ERROR")
tf.random.set_seed(42)
np.random.seed(42)

HERE = os.path.dirname(__file__)
DATA = os.path.join(HERE, "data_ext")
RESULTS = os.path.join(HERE, "results")
MODELS = os.path.join(HERE, "models")
os.makedirs(RESULTS, exist_ok=True)
os.makedirs(MODELS, exist_ok=True)

# Human-readable class names for plank's terse labels.
CLASS_NAMES = {
    "C": "correct",
    "L": "hips too low",
    "H": "hips too high",
    "up": "up",
    "down": "down",
}


# Train on x,y only. The z and visibility (v) channels differ between the
# MediaPipe build used to make this dataset (solutions.pose) and the one that
# runs in the browser (tasks-vision), and the near-constant v columns blow up
# under standardisation. x,y normalized image coords are the consistent subset,
# so the shipped model transfers cleanly to the live web app.
XY_ONLY = True


def load(exercise):
    tr = pd.read_csv(os.path.join(DATA, exercise, "train.csv"))
    te = pd.read_csv(os.path.join(DATA, exercise, "test.csv"))
    feat = [c for c in tr.columns if c != "label"]
    if XY_ONLY:
        feat = [c for c in feat if c.endswith("_x") or c.endswith("_y")]
    return tr, te, feat


def make_nn(n_features, n_classes):
    m = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(n_features,)),
        tf.keras.layers.Dense(128, activation="relu"),
        tf.keras.layers.Dropout(0.4),
        tf.keras.layers.Dense(64, activation="relu"),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(n_classes, activation="softmax"),
    ])
    m.compile(optimizer="adam",
              loss="sparse_categorical_crossentropy",
              metrics=["accuracy"])
    return m


def main(exercise="plank"):
    print(f"=== Training form classifier: {exercise} ===")
    tr, te, feat = load(exercise)

    le = LabelEncoder().fit(tr["label"])
    ytr = le.transform(tr["label"])
    yte = le.transform(te["label"])
    classes = list(le.classes_)
    names = [CLASS_NAMES.get(c, c) for c in classes]

    print(f"  train={len(tr)}  test={len(te)}  features={len(feat)}  "
          f"classes={classes}")

    scaler = StandardScaler().fit(tr[feat])
    Xtr = scaler.transform(tr[feat])
    Xte = scaler.transform(te[feat])

    # ---- baseline ----
    lr = LogisticRegression(max_iter=2000, multi_class="auto").fit(Xtr, ytr)
    lr_acc = accuracy_score(yte, lr.predict(Xte))
    print(f"  Logistic Regression test accuracy: {lr_acc:.3f}")

    # ---- neural net ----
    nn = make_nn(len(feat), len(classes))
    es = tf.keras.callbacks.EarlyStopping(patience=6, restore_best_weights=True)
    hist = nn.fit(Xtr, ytr, validation_split=0.15, epochs=60, batch_size=64,
                  verbose=0, callbacks=[es])
    nn_pred = nn.predict(Xte, verbose=0).argmax(1)
    nn_acc = accuracy_score(yte, nn_pred)
    print(f"  Neural Net          test accuracy: {nn_acc:.3f}")

    # ---- save artifacts ----
    nn.save(os.path.join(MODELS, f"{exercise}_nn.keras"))
    np.save(os.path.join(MODELS, f"{exercise}_scaler_mean.npy"), scaler.mean_)
    np.save(os.path.join(MODELS, f"{exercise}_scaler_scale.npy"), scaler.scale_)
    with open(os.path.join(MODELS, f"{exercise}_meta.json"), "w") as f:
        json.dump({"features": feat, "classes": classes, "names": names}, f, indent=2)

    # ---- report ----
    rep = classification_report(yte, nn_pred, target_names=names, digits=3)
    with open(os.path.join(RESULTS, f"{exercise}_report.md"), "w", encoding="utf-8") as f:
        f.write(f"# {exercise.capitalize()} form classifier — honest test results\n\n")
        f.write(f"Real public dataset, author-provided held-out test split "
                f"({len(te)} rows).\n\n")
        f.write("| Model | Test accuracy |\n|---|---:|\n")
        f.write(f"| Logistic Regression (baseline) | {lr_acc:.1%} |\n")
        f.write(f"| Neural Net | **{nn_acc:.1%}** |\n\n")
        f.write("## Per-class (neural net)\n\n```\n" + rep + "```\n")
    print("\n" + rep)

    # ---- plots ----
    cm = confusion_matrix(yte, nn_pred)
    plt.figure(figsize=(4.6, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="mako",
                xticklabels=names, yticklabels=names, cbar=False)
    plt.xlabel("Predicted"); plt.ylabel("True")
    plt.title(f"{exercise.capitalize()} — confusion matrix ({nn_acc:.0%})")
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS, f"{exercise}_confusion.png"), dpi=130)
    plt.close()

    plt.figure(figsize=(6, 4))
    plt.plot(hist.history["accuracy"], label="train")
    plt.plot(hist.history["val_accuracy"], label="val")
    plt.xlabel("epoch"); plt.ylabel("accuracy"); plt.legend()
    plt.title(f"{exercise.capitalize()} — learning curve")
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS, f"{exercise}_learning_curve.png"), dpi=130)
    plt.close()

    print(f"\nSaved model + artifacts to {MODELS}")
    print(f"Saved report + plots to {RESULTS}")
    return lr_acc, nn_acc


if __name__ == "__main__":
    ex = sys.argv[1] if len(sys.argv) > 1 else "plank"
    main(ex)
