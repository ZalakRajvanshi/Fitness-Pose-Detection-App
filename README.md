# FormCoach — real-time AI workout-form coaching

A pose-based fitness coach that watches you through a webcam, **counts reps**, and
judges your **form** in real time — for squats, push-ups, and planks. It runs
fully **on-device** (no video leaves the browser) and the form verdict comes from
a classifier **trained, evaluated honestly, and exported into the live app**.

This repo is built to double as an ML portfolio piece: the interesting part isn't
a single accuracy number, it's the *judgement* around it — catching a flawed
evaluation, sourcing real data, checking robustness, knowing when a fancier model
is not worth it, and shipping the result into a working product.

```
┌─────────────┐   train + evaluate    ┌──────────────┐   export 4 KB JSON   ┌────────────┐
│ public pose │ ───────────────────▶  │  classifier   │ ──────────────────▶ │  web app   │
│  dataset    │   honest protocol     │ (Python/ml/)  │   no TF.js runtime  │ (web/)     │
└─────────────┘                       └──────────────┘                      └────────────┘
```

---

## Quick start

**Run the app** (needs `localhost`, not `file://`, for ES modules + webcam):

```bash
python web/serve.py        # opens http://localhost:8000
```

Allow the camera, pick an exercise, press **Start**. Try **Plank** and shift your
hips up/down — the `AI` chip flips between *Correct / Hips too low / Hips too high*
with a live confidence.

**Reproduce the ML** (Python 3.10, `pip install -r requirements.txt`):

```bash
python ml/train_real.py        # train form classifier on the real dataset
python ml/robustness.py        # leaky vs honest (by-person) evaluation
python ml/train_lstm.py        # temporal model vs per-frame
python ml/export_web.py        # export model -> web/models/plank.json
```

---

## The ML story (what each step shows)

### 1. Catching a broken evaluation
The original pipeline (kept untouched in [src/](src/)) reported **97% accuracy**.
[ml/train.py](ml/train.py) shows that was **data leakage**: it augmented 20 images
10× *before* a random split, so near-duplicate copies of the same photo sat in
both train and test. Under a source-aware split, real accuracy collapses to
**~15–35%** — i.e. the 20-image dataset, not the model, was the ceiling.

> Lesson: trust the protocol, not the number.

### 2. A real model on a real dataset
[ml/train_real.py](ml/train_real.py) swaps in a public dataset —
[NgoQuocBao1010/Exercise-Correction](https://github.com/NgoQuocBao1010/Exercise-Correction)
(MIT), MediaPipe keypoints already extracted, with an author-provided test split.
A logistic-regression **baseline** is compared to a neural net under one protocol:

| Exercise | Task | Classes | Test rows | Baseline (LogReg) | Neural net |
|---|---|---|---:|---:|---:|
| **Plank** | form quality | correct / hips-too-low / hips-too-high | 710 | **99.6%** | 97.0% |
| **Squat** | rep stage | up / down | 853 | 99.1% | 99.2% |

On the robust **x,y-only** feature set the linear baseline matches or beats the
neural net — so that's what ships. (Why x,y only: the `z` and visibility channels
differ between the MediaPipe build that made the dataset and the one in the
browser; x,y normalized coordinates transfer cleanly.)

### 3. Is the score real? A by-person robustness check
The flat CSV has no person ids, but frames come from many short clips with large
jumps between them. [ml/groups.py](ml/groups.py) recovers those clip boundaries;
[ml/robustness.py](ml/robustness.py) then re-evaluates under a **by-clip
(≈ by-person) split**:

| Protocol | Plank | Squat |
|---|---:|---:|
| Random frame split (leaky) | 99.7% | 99.7% |
| **By-segment group CV (honest)** | **99.6% ± 0.2%** | **99.3% ± 0.8%** |
| Provided test split (reference) | 99.6% | 99.1% |

Unlike the 20-image case, the accuracy **survives** the honest split — so here the
~99% is trustworthy, not leakage. The point of the check is to *verify*, not to
assume one way or the other.

### 4. Does temporal context help? An honest LSTM
An exercise is a motion, so [ml/train_lstm.py](ml/train_lstm.py) builds **32-frame
sequences within clips** (never crossing a boundary), splits by clip, and compares
an **LSTM** to the per-frame model:

| Model | Plank window accuracy |
|---|---:|
| Per-frame LogReg (majority vote) | **99.9%** |
| LSTM (32-frame window) | 94.7% |

**Finding:** temporal modeling does **not** beat per-frame here — plank form is
readable from a single static posture. The transferable part is the method
(group-aware sequence construction + fair comparison); LSTMs earn their keep on
*dynamic*, motion-dependent errors, not static holds. Reporting this straight,
rather than forcing the fancy model to win, is the point.

### 5. Shipping the model into the product
[ml/export_web.py](ml/export_web.py) exports the model to a 4 KB JSON, and
[web/js/form-model.js](web/js/form-model.js) runs it in the browser as a plain
standardise + matmul + softmax — **no ML runtime**. The port was verified to
reproduce the Python model exactly: **99.6%** (plank) / **99.1%** (squat) on the
held-out test set.

---

## The product

- **On-device pose** via MediaPipe Tasks Vision (JS) — webcam never leaves the page.
- **Rep counting** — a smoothed two-state machine on the driving joint angle.
- **Form coaching** — the trained model's verdict *plus* transparent angle-based
  cues (*"knees caving in", "hips sagging", "go deeper"*).
- **Polished, minimal UI** — dark theme, live skeleton, big rep counter, AI chip,
  session timer.

See [web/README.md](web/README.md) for the front-end details.

---

## Repo layout

```
src/      original pipeline — left intact as the "before"
ml/       honest training + evaluation
  dataset.py / train.py      leakage demo on the original 20 images
  data_ext/                  downloaded public dataset (squat, plank CSVs)
  train_real.py              real classifier: baseline vs neural net
  groups.py / robustness.py  clip recovery + by-person robustness check
  train_lstm.py              temporal model vs per-frame
  export_web.py              model -> web/models/<exercise>.json
  models/ results/           trained models, reports, plots
web/      the on-device app (HTML/CSS/JS) + exported models
```

## Honesty notes & next steps

- The dataset has no true person ids; the by-clip split is an **approximation** of
  person-independence (documented in [ml/README.md](ml/README.md)).
- Plank labels are visually separable and the dataset is large, so ~99% is
  believable *for this benchmark* — not a claim of a solved real-world product.
- Natural extensions: more exercises, a genuinely dynamic form error to let the
  LSTM shine, and static deployment (GitHub Pages / Vercel — the app is all static).

## Tools & libraries

Python 3.10 · MediaPipe · TensorFlow / Keras · scikit-learn · OpenCV ·
NumPy / Pandas · Matplotlib / Seaborn · MediaPipe Tasks Vision (JS).

## References

- Cao, Z. et al., "OpenPose: Realtime Multi-Person 2D Pose Estimation Using Part Affinity Fields," CVPR, 2017.
- Lugaresi, C. et al., "MediaPipe: A Framework for Building Perception Pipelines," arXiv, 2019.
- Hochreiter, S., & Schmidhuber, J., "Long Short-Term Memory," Neural Computation, 1997.

## Credits

Dataset: [NgoQuocBao1010/Exercise-Correction](https://github.com/NgoQuocBao1010/Exercise-Correction)
(MIT). Pose estimation: [MediaPipe](https://developers.google.com/mediapipe).
