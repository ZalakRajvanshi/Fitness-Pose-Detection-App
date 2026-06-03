# FormCoach — web app

A polished, on-device AI workout-form tracker. It uses your webcam + MediaPipe
Pose to detect body landmarks in real time, count reps, and give live form
feedback for **squats, push-ups, and planks**.

Everything runs locally in the browser — no video ever leaves the device.

## Run it

The app uses ES modules and the webcam, so it must be served over
`http://localhost` (opening `index.html` directly with `file://` won't work).

```bash
python web/serve.py
```

This serves the app and opens `http://localhost:8000` in your browser. Allow
camera access, pick an exercise, and press **Start camera**.

> Any static server works, e.g. `npx serve web` or `python -m http.server 8000`
> from inside the `web/` folder.

## How it works

| Piece | File |
|-------|------|
| UI shell | [index.html](index.html) |
| Design system (dark, minimal) | [styles.css](styles.css) |
| App controller (camera, detection loop, rendering) | [js/main.js](js/main.js) |
| Geometry + rep state machine | [js/pose-math.js](js/pose-math.js) |
| Per-exercise rules & coaching cues | [js/exercises.js](js/exercises.js) |

### Rep counting
A two-state machine over a driving joint angle (knee for squats, elbow for
push-ups). A rep completes on each full down → up transition. The signal is
exponentially smoothed to avoid double-counting on jitter.

### Form feedback
Each exercise defines `checks` — small rules over joint angles and landmark
geometry (e.g. *"knees caving in"*, *"hips sagging"*, *"go deeper"*). Any
violated rule surfaces as a live coaching cue.

### Trained model (the "AI" verdict)
Plank form is judged by a **trained classifier**, not just rules. The model is
trained in Python (`../ml/`) on a real public dataset, then exported to a tiny
JSON ([models/plank.json](models/plank.json), 4 KB) and run directly in
[js/form-model.js](js/form-model.js) — a standardise + matmul + softmax, no ML
runtime in the browser. It predicts **correct / hips-too-low / hips-too-high**
with a confidence, shown as the `AI` chip in the form panel.

The squat model ([models/squat.json](models/squat.json)) reports the rep
**stage** (up/down). Push-ups currently use rules only.

> The JS port was verified to reproduce the Python model exactly: **99.6%**
> (plank) and **99.1%** (squat) on the held-out test set — identical to training.

## Pipeline

```
ml/train_real.py   → trains on real dataset, compares baseline vs neural net
ml/export_web.py   → exports the chosen model to web/models/<exercise>.json
web/js/form-model.js → runs that JSON live in the browser
```
