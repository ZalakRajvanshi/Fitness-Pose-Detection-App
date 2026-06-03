"""
robustness.py — how much does the evaluation protocol change the score?

Same model (the shipped logistic regression, x,y features), same data pool
(train.csv), three protocols:

  random frame split   frames from one clip land in both train & test -> leaky
  by-segment group CV   whole clips held out (≈ by-person)            -> honest
  provided test.csv     the authors' held-out split                  -> reference

The gap between the first two is the same leakage lesson from the 20-image demo,
now measured on the real dataset. Run:

    python ml/robustness.py            # plank
    python ml/robustness.py squat
"""
import os
import sys
import warnings

warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold, train_test_split
from sklearn.metrics import accuracy_score
from sklearn.pipeline import make_pipeline

from groups import infer_segments, xy_features

HERE = os.path.dirname(__file__)
RESULTS = os.path.join(HERE, "results")


def fit_eval(Xtr, ytr, Xte, yte):
    clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))
    clf.fit(Xtr, ytr)
    return accuracy_score(yte, clf.predict(Xte))


def main(exercise="plank"):
    tr = pd.read_csv(os.path.join(HERE, "data_ext", exercise, "train.csv"))
    te = pd.read_csv(os.path.join(HERE, "data_ext", exercise, "test.csv"))
    feat = xy_features(tr)
    X, y = tr[feat].values, tr["label"].values
    seg = infer_segments(tr)

    # 1) random frame split (leaky) — average a few seeds for stability
    rand = []
    for s in range(5):
        Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25,
                                              random_state=s, stratify=y)
        rand.append(fit_eval(Xtr, ytr, Xte, yte))
    rand = np.array(rand)

    # 2) by-segment group CV (honest, ≈ by-person)
    gkf = GroupKFold(n_splits=5)
    grp = [fit_eval(X[tr_i], y[tr_i], X[te_i], y[te_i])
           for tr_i, te_i in gkf.split(X, y, seg)]
    grp = np.array(grp)

    # 3) provided author split (reference)
    prov = fit_eval(X, y, te[feat].values, te["label"].values)

    rows = [
        ("Random frame split (leaky)", rand.mean(), rand.std()),
        ("By-segment group CV (honest, ~by-person)", grp.mean(), grp.std()),
        ("Provided test.csv (reference)", prov, 0.0),
    ]
    print(f"\n=== {exercise} robustness ===")
    for name, m, s in rows:
        print(f"  {name:42s} {m:.1%}" + (f" ± {s:.1%}" if s else ""))

    # report
    with open(os.path.join(RESULTS, f"{exercise}_robustness.md"), "w", encoding="utf-8") as f:
        f.write(f"# {exercise.capitalize()} — does the split change the score?\n\n")
        f.write(f"Same model (logistic regression, x,y features), "
                f"same data pool ({len(tr)} frames, {seg.max()+1} inferred clips).\n\n")
        f.write("| Protocol | Accuracy |\n|---|---:|\n")
        for name, m, s in rows:
            f.write(f"| {name} | {m:.1%}" + (f" ± {s:.1%}" if s else "") + " |\n")
        f.write("\n> The by-segment number is the honest one: whole clips are "
                "held out, approximating an unseen person. The gap to the random "
                "split is leakage from near-duplicate neighbouring frames.\n")

    # plot
    names = [r[0].split(" (")[0] for r in rows]
    vals = [r[1] for r in rows]
    errs = [r[2] for r in rows]
    colors = ["#fb7185", "#5eead4", "#94a3b8"]
    plt.figure(figsize=(6.4, 4))
    bars = plt.bar(names, vals, yerr=errs, capsize=6, color=colors)
    for b, v in zip(bars, vals):
        plt.text(b.get_x() + b.get_width() / 2, v + 0.01, f"{v:.0%}",
                 ha="center", fontweight="bold")
    plt.ylim(0, 1.06)
    plt.ylabel("Accuracy")
    plt.title(f"{exercise.capitalize()} — evaluation protocol vs reported accuracy")
    plt.xticks(rotation=12, ha="right", fontsize=8.5)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS, f"{exercise}_robustness.png"), dpi=130)
    plt.close()
    print(f"  saved -> results/{exercise}_robustness.md + .png")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "plank")
