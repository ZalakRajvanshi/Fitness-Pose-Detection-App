"""
train.py — honest training & evaluation for the pose form-classifier.

What this script demonstrates (the portfolio point):

  1. The OLD protocol (augment everything, then random split) leaks near-
     duplicate rows across the split and reports an inflated accuracy.
  2. The HONEST protocol (split by source image, augment train only) reports
     what the model would really do on an unseen person/photo.
  3. A logistic-regression baseline vs a small neural net under that honest
     protocol — so the model choice is justified by evidence, not vibes.

Outputs (ml/results/):
  comparison.md        side-by-side metrics table
  accuracy.png         bar chart of the four numbers
  confusion_matrix.png honest NN confusion matrix
"""
import os
import warnings

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
warnings.filterwarnings("ignore")

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold, train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

from dataset import load_raw, augment

tf.get_logger().setLevel("ERROR")
tf.random.set_seed(42)
np.random.seed(42)

RESULTS = os.path.join(os.path.dirname(__file__), "results")
os.makedirs(RESULTS, exist_ok=True)
N_AUG = 12


# ----------------------------------------------------------------------
# Models
# ----------------------------------------------------------------------
def make_nn(n_features):
    m = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(n_features,)),
        tf.keras.layers.Dense(64, activation="relu"),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(32, activation="relu"),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(1, activation="sigmoid"),
    ])
    m.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    return m


def fit_nn(Xtr, ytr, Xte, yte):
    m = make_nn(Xtr.shape[1])
    m.fit(Xtr, ytr, validation_data=(Xte, yte), epochs=40, batch_size=8, verbose=0)
    proba = m.predict(Xte, verbose=0).ravel()
    return m, (proba >= 0.5).astype(int)


# ----------------------------------------------------------------------
# 1. LEAKY protocol  (the old way) — for contrast
# ----------------------------------------------------------------------
def leaky_accuracy(X, y, groups):
    """Augment first, then random split. Duplicates straddle the split."""
    Xa, ya, _ = augment(X, y, groups, n_aug=N_AUG)
    scaler = StandardScaler()
    Xa = scaler.fit_transform(Xa)
    Xtr, Xte, ytr, yte = train_test_split(
        Xa, ya, test_size=0.3, random_state=42, stratify=ya
    )
    _, pred = fit_nn(Xtr, ytr, Xte, yte)
    return accuracy_score(yte, pred)


# ----------------------------------------------------------------------
# 2. HONEST protocol — group split, augment train only
# ----------------------------------------------------------------------
def honest_cv(X, y, groups, n_splits=5):
    """
    GroupKFold over source images. For each fold:
      - augment the TRAIN images only
      - fit the scaler on TRAIN only
      - evaluate on the pristine held-out images
    Returns per-model accuracy lists + pooled predictions for the NN.
    """
    gkf = GroupKFold(n_splits=n_splits)
    acc_lr, acc_nn = [], []
    pooled_true, pooled_pred = [], []

    for tr, te in gkf.split(X, y, groups):
        Xtr_raw, ytr_raw, gtr = X[tr], y[tr], groups[tr]
        Xte_raw, yte = X[te], y[te]

        # augment train only
        Xtr_aug, ytr_aug, _ = augment(Xtr_raw, ytr_raw, gtr, n_aug=N_AUG)

        scaler = StandardScaler().fit(Xtr_aug)
        Xtr = scaler.transform(Xtr_aug)
        Xte = scaler.transform(Xte_raw)

        # logistic-regression baseline
        lr = LogisticRegression(max_iter=1000).fit(Xtr, ytr_aug)
        acc_lr.append(accuracy_score(yte, lr.predict(Xte)))

        # neural net
        _, pred = fit_nn(Xtr, ytr_aug, Xte, yte)
        acc_nn.append(accuracy_score(yte, pred))
        pooled_true.extend(yte.tolist())
        pooled_pred.extend(pred.tolist())

    return (np.array(acc_lr), np.array(acc_nn),
            np.array(pooled_true), np.array(pooled_pred))


# ----------------------------------------------------------------------
# Reporting
# ----------------------------------------------------------------------
def write_report(leaky_nn, lr_acc, nn_acc, ptrue, ppred):
    lines = []
    lines.append("# Model evaluation — honest vs leaky\n")
    lines.append(f"- Held-out source images evaluated (pooled across folds): "
                 f"**{len(ptrue)}**\n")
    lines.append("\n## Headline\n")
    lines.append("| Protocol | Model | Accuracy |")
    lines.append("|---|---|---:|")
    lines.append(f"| Leaky (augment → random split) | Neural net | {leaky_nn:.1%} |")
    lines.append(f"| **Honest** (group split, augment train only) | Logistic Reg (baseline) | {lr_acc.mean():.1%} ± {lr_acc.std():.1%} |")
    lines.append(f"| **Honest** (group split, augment train only) | Neural net | {nn_acc.mean():.1%} ± {nn_acc.std():.1%} |")
    lines.append("")
    lines.append("> The leaky number is the one the original pipeline reported. "
                 "Under a source-image-aware split the real generalisation "
                 "accuracy is the honest row — that gap *is* the leakage.\n")
    lines.append("\n## Honest NN — per-class report\n")
    lines.append("```\n" + classification_report(ptrue, ppred, digits=3,
                 target_names=["incorrect", "correct"]) + "```\n")
    report = "\n".join(lines)
    with open(os.path.join(RESULTS, "comparison.md"), "w", encoding="utf-8") as f:
        f.write(report)
    # Plain-ASCII console summary (Windows consoles choke on unicode arrows).
    print("\n--- Results ---")
    print(f"Leaky NN accuracy (misleading): {leaky_nn:.1%}")
    print(f"Honest LogReg accuracy:         {lr_acc.mean():.1%} +/- {lr_acc.std():.1%}")
    print(f"Honest NN accuracy:             {nn_acc.mean():.1%} +/- {nn_acc.std():.1%}")


def plot_accuracy(leaky_nn, lr_acc, nn_acc):
    labels = ["Leaky NN\n(misleading)", "Honest\nLogReg", "Honest\nNN"]
    vals = [leaky_nn, lr_acc.mean(), nn_acc.mean()]
    errs = [0, lr_acc.std(), nn_acc.std()]
    colors = ["#fb7185", "#94a3b8", "#5eead4"]
    plt.figure(figsize=(6, 4))
    bars = plt.bar(labels, vals, yerr=errs, capsize=6, color=colors)
    for b, v in zip(bars, vals):
        plt.text(b.get_x() + b.get_width() / 2, v + 0.02, f"{v:.0%}",
                 ha="center", fontweight="bold")
    plt.ylim(0, 1.08)
    plt.ylabel("Accuracy")
    plt.title("Honest evaluation reveals the real accuracy")
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS, "accuracy.png"), dpi=130)
    plt.close()


def plot_confusion(ptrue, ppred):
    cm = confusion_matrix(ptrue, ppred)
    plt.figure(figsize=(4.4, 3.8))
    sns.heatmap(cm, annot=True, fmt="d", cmap="mako",
                xticklabels=["incorrect", "correct"],
                yticklabels=["incorrect", "correct"], cbar=False)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Honest NN — confusion matrix")
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS, "confusion_matrix.png"), dpi=130)
    plt.close()


def main():
    print("Extracting keypoints from raw images…")
    X, y, groups = load_raw()
    print(f"  {X.shape[0]} source images "
          f"(correct={int((y==1).sum())}, incorrect={int((y==0).sum())})\n")

    n_splits = min(5, int(min((y == 0).sum(), (y == 1).sum())))
    print(f"Leaky protocol (for contrast)…")
    leaky_nn = leaky_accuracy(X, y, groups)

    print(f"Honest {n_splits}-fold group CV…")
    lr_acc, nn_acc, ptrue, ppred = honest_cv(X, y, groups, n_splits=n_splits)

    write_report(leaky_nn, lr_acc, nn_acc, ptrue, ppred)
    plot_accuracy(leaky_nn, lr_acc, nn_acc)
    plot_confusion(ptrue, ppred)
    print(f"\nSaved report + plots to {RESULTS}")


if __name__ == "__main__":
    main()
