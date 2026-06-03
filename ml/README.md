# ML pipeline — honest training & evaluation

This folder is the "model-training story" for FormCoach. It deliberately keeps
the original `../src/` pipeline untouched and rebuilds it correctly, so the repo
shows a clear **before → after**.

## The problem with the original pipeline

`../src/extract_keypoints.py` augments every image **10×** *before* training, and
`../src/train_model.py` then does a random train/test split. Because the
augmented rows are near-duplicates of their source image, copies of the same
photo land in **both** train and test. The model effectively sees the test set
during training, so the reported **97% accuracy is leakage**, not skill.

## The fix (this folder)

1. **Provenance tracking** — [dataset.py](dataset.py) keeps a *group id*
   (source image) for every sample.
2. **Group-aware split** — [train.py](train.py) uses `GroupKFold`, so all copies
   of one image stay on the same side of the split.
3. **Augment train only** — jitter is added *after* the split, never to test rows.
4. **Baseline first** — a logistic-regression baseline is evaluated under the
   exact same protocol as the neural net, so the model choice is evidence-based.

Run it:

```bash
python ml/train.py
```

Outputs land in [results/](results/): `comparison.md`, `accuracy.png`,
`confusion_matrix.png`.

## The result (and why it matters)

| Protocol | Model | Accuracy |
|---|---|---:|
| Leaky (augment → random split) | Neural net | **100%** |
| Honest (group split) | Logistic Reg | ~35% |
| Honest (group split) | Neural net | ~15% |

The honest accuracy is **near or below chance** — because the dataset is only
**20 images (10 per class)**. There simply isn't enough real signal to learn a
person-independent "correct vs incorrect" classifier.

**That is the finding.** The headline 97% was an illusion; once measured
honestly, the data — not the model — is the bottleneck. This is exactly the
judgement a good ML engineer is expected to show: *trust the protocol, not the
number.*

## Part 2 — a real model on a real dataset ([train_real.py](train_real.py))

Once the methodology was trustworthy, the only missing piece was data. So we
swapped the 20 images for a public dataset —
[NgoQuocBao1010/Exercise-Correction](https://github.com/NgoQuocBao1010/Exercise-Correction)
(MIT) — which ships MediaPipe keypoints already extracted to CSV, **with an
author-provided held-out train/test split**. No filming, no manual labelling.

Two classifiers are trained, each comparing a logistic-regression baseline to a
small neural net under the same protocol:

| Exercise | Task | Classes | Test rows | LogReg | Neural net |
|---|---|---|---:|---:|---:|
| **Plank** | form quality | correct / hips-too-low / hips-too-high | 710 | 99.6% | **99.7%** |
| **Squat** | rep stage | up / down | 853 | 99.5% | **99.5%** |

Artifacts land in [models/](models/) (`*_nn.keras`, scaler stats, and a
`*_meta.json` listing the exact feature order + class names) and reports/plots in
[results/](results/) (`plank_report.md`, `plank_confusion.png`, learning curves…).

```bash
python ml/train_real.py          # plank form classifier
python ml/train_real.py squat    # squat stage classifier
```

### Honesty note (what these numbers do and don't claim)

The plank labels (C / L / H) are visually very separable, the dataset is large
(~28k rows), and the baseline already nails it — so ~99% is believable *for this
dataset*. The one caveat we don't hide: the CSV rows are per-frame, and the
provided split is by clip, not guaranteed fully **person-independent**. The
right next robustness check is a by-person split; until then we report this as
"strong on the provided benchmark," not "solved." That distinction is the same
discipline that caught the original leakage.

## Part 3 — is the score real? by-person robustness ([robustness.py](robustness.py))

The CSVs are flat frames with no person id, but they are clips concatenated with
large jumps between them. [groups.py](groups.py) recovers those clip boundaries
(frame-to-frame distance threshold), and [robustness.py](robustness.py)
re-evaluates the shipped model under a **by-clip (≈ by-person) split**:

| Protocol | Plank | Squat |
|---|---:|---:|
| Random frame split (leaky) | 99.7% | 99.7% |
| **By-segment group CV (honest)** | **99.6% ± 0.2%** | **99.3% ± 0.8%** |
| Provided test split | 99.6% | 99.1% |

The accuracy survives the honest split — so unlike the 20-image case, this ~99% is
trustworthy. (The clip recovery is an approximation of a true by-person split,
since the data has no person ids — stated plainly rather than hidden.)

## Part 4 — does temporal context help? ([train_lstm.py](train_lstm.py))

An exercise is a motion, so we build 32-frame sequences **within** clips, split by
clip, and compare an LSTM to the per-frame model:

| Model | Plank window accuracy |
|---|---:|
| Per-frame LogReg (majority vote) | **99.9%** |
| LSTM (32-frame window) | 94.7% |

Temporal modeling does **not** beat per-frame for a static plank hold — the honest,
expected result. The reusable part is the group-aware sequence construction + fair
comparison; LSTMs pay off on dynamic, motion-dependent errors.

## Next step — into the app

Export `plank_nn.keras` to **TensorFlow.js** and run it in
[../web/](../web/) alongside the rule-based cues, so the UI shows a learned
form verdict (e.g. *"hips too low — 0.94"*) next to the explainable angle hints.
The `*_meta.json` files give the JS side the exact feature order to reproduce.
