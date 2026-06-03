// main.js — app controller: camera, pose detection, UI wiring.
import {
  PoseLandmarker,
  FilesetResolver,
  DrawingUtils,
} from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.12";

import { EXERCISES } from "./exercises.js";
import { Smoother, RepCounter } from "./pose-math.js";
import { FormModel } from "./form-model.js";

// ---------- DOM ----------
const el = (id) => document.getElementById(id);
const video = el("video");
const canvas = el("overlay");
const ctx = canvas.getContext("2d");

const ui = {
  startBtn: el("startBtn"),
  stopBtn: el("stopBtn"),
  resetBtn: el("resetBtn"),
  overlay: el("videoOverlay"),
  overlayTitle: el("overlayTitle"),
  overlaySub: el("overlaySubtitle"),
  spinner: el("spinner"),
  statusPill: el("statusPill"),
  statusDot: el("statusDot"),
  statusText: el("statusText"),
  liveChips: el("liveChips"),
  repChipMini: el("repChipMini"),
  formChipMini: el("formChipMini"),
  primaryLabel: el("primaryLabel"),
  primaryValue: el("primaryValue"),
  primarySub: el("primarySub"),
  formCard: el("formCard"),
  formBadge: el("formBadge"),
  mlVerdict: el("mlVerdict"),
  mlVerdictText: el("mlVerdictText"),
  mlVerdictProb: el("mlVerdictProb"),
  cues: el("cues"),
  kneeVal: el("kneeVal"),
  backVal: el("backVal"),
  tempoVal: el("tempoVal"),
  timerVal: el("timerVal"),
  fpsHint: el("fpsHint"),
  exButtons: [...document.querySelectorAll(".ex-btn")],
};

// ---------- State ----------
let landmarker = null;
let running = false;
let rafId = null;
let exerciseKey = "squat";
let exercise = EXERCISES.squat;
let counter = new RepCounter(exercise.down ?? 95, exercise.up ?? 160);
let repSmoother = new Smoother(0.45);
let drawer = null;
let sessionStart = 0;
let holdMs = 0; // accumulated good-form hold time for plank
let lastTs = 0;
let lastVideoTime = -1;
let fpsEMA = 0;

// Trained classifiers (exported from the Python pipeline). Lazy-loaded + cached.
const formModels = {}; // exerciseKey -> FormModel | null (null = none/failed)
let formModel = null; // model for the current exercise, if any
// Exercises where the model decides correctness (vs. just reporting a stage).
const ML_FORM_JUDGE = { plank: { good: "C" } };

async function ensureModel(key) {
  if (!(key in formModels)) {
    try {
      formModels[key] = await FormModel.load(`./models/${key}.json`);
    } catch {
      formModels[key] = null;
    }
  }
  if (key === exerciseKey) formModel = formModels[key];
}

// ---------- Model bootstrap ----------
async function loadModel() {
  ui.spinner.hidden = false;
  ui.overlayTitle.textContent = "Loading pose model…";
  ui.overlaySub.textContent = "First load fetches a small model (~6 MB).";
  const vision = await FilesetResolver.forVisionTasks(
    "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.12/wasm"
  );
  landmarker = await PoseLandmarker.createFromOptions(vision, {
    baseOptions: {
      modelAssetPath:
        "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task",
      delegate: "GPU",
    },
    runningMode: "VIDEO",
    numPoses: 1,
  });
  drawer = new DrawingUtils(ctx);
}

// ---------- Camera ----------
async function startCamera() {
  const stream = await navigator.mediaDevices.getUserMedia({
    video: { width: { ideal: 1280 }, height: { ideal: 960 }, facingMode: "user" },
    audio: false,
  });
  video.srcObject = stream;
  await video.play();
  canvas.width = video.videoWidth;
  canvas.height = video.videoHeight;
}

function stopCamera() {
  const s = video.srcObject;
  if (s) s.getTracks().forEach((t) => t.stop());
  video.srcObject = null;
}

// ---------- Session lifecycle ----------
async function start() {
  ui.startBtn.disabled = true;
  try {
    if (!landmarker) await loadModel();
    await ensureModel(exerciseKey);
    await startCamera();
  } catch (err) {
    ui.spinner.hidden = true;
    ui.overlayTitle.textContent = "Couldn't start";
    ui.overlaySub.textContent =
      err?.name === "NotAllowedError"
        ? "Camera permission was denied. Allow it and try again."
        : "Error: " + (err?.message || err);
    ui.startBtn.disabled = false;
    return;
  }
  ui.spinner.hidden = true;
  ui.overlay.classList.add("is-hidden");
  ui.liveChips.hidden = false;
  ui.stopBtn.disabled = false;
  setStatus(true);
  resetSession();
  running = true;
  lastTs = performance.now();
  loop();
}

function stop() {
  running = false;
  if (rafId) cancelAnimationFrame(rafId);
  stopCamera();
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  ui.overlay.classList.remove("is-hidden");
  ui.overlayTitle.textContent = "Session paused";
  ui.overlaySub.textContent = "Press start to resume your workout.";
  ui.startBtn.disabled = false;
  ui.stopBtn.disabled = true;
  ui.liveChips.hidden = true;
  setStatus(false);
}

function resetSession() {
  counter = new RepCounter(exercise.down ?? 95, exercise.up ?? 160);
  repSmoother = new Smoother(0.45);
  holdMs = 0;
  sessionStart = performance.now();
  ui.primaryValue.textContent = exercise.isHold ? "0s" : "0";
  ui.primaryLabel.textContent = exercise.primaryLabel;
  ui.primarySub.textContent = `${exercise.name} session`;
  setCues([{ text: "Stand in frame to begin analysis.", kind: "muted" }]);
  setFormBadge("Waiting", "");
  ui.mlVerdict.hidden = true;
}

// ---------- Main loop ----------
function loop() {
  if (!running) return;
  const now = performance.now();

  if (video.currentTime !== lastVideoTime) {
    lastVideoTime = video.currentTime;
    const result = landmarker.detectForVideo(video, now);
    render(result, now);
  }

  // fps
  const dt = now - lastTs;
  lastTs = now;
  if (dt > 0) fpsEMA = fpsEMA ? 0.9 * fpsEMA + 0.1 * (1000 / dt) : 1000 / dt;
  ui.fpsHint.textContent = `${Math.round(fpsEMA)} fps`;

  // timer
  ui.timerVal.textContent = fmtClock(now - sessionStart);

  rafId = requestAnimationFrame(loop);
}

function render(result, now) {
  ctx.save();
  ctx.clearRect(0, 0, canvas.width, canvas.height);

  const lmList = result.landmarks?.[0];
  if (!lmList) {
    ctx.restore();
    return;
  }

  // Skeleton overlay (canvas is mirrored via CSS, so draw in raw coords).
  drawer.drawConnectors(lmList, PoseLandmarker.POSE_CONNECTIONS, {
    color: "rgba(94,234,212,0.55)",
    lineWidth: 3,
  });
  drawer.drawLandmarks(lmList, { color: "#5eead4", fillColor: "#0a0c10", radius: 4, lineWidth: 2 });
  ctx.restore();

  analyze(lmList, now);
}

// ---------- Analysis + coaching ----------
function analyze(lm, now) {
  const m = exercise.metrics(lm);

  // Live metrics panel
  ui.kneeVal.textContent = `${Math.round(m.knee)}°`;
  ui.backVal.textContent = `${Math.round(m.back)}°`;

  // Rule-based checks (transparent, always on)
  const phase = exercise.isHold ? "hold" : counter.phase;
  let violations = exercise.checks
    .map((fn) => fn({ lm, m, phase, depthAngle: m.knee }))
    .filter(Boolean);

  // Trained model (if one exists for this exercise)
  const ml = formModel ? formModel.predict(lm) : null;
  const judge = ML_FORM_JUDGE[exerciseKey];
  updateMlVerdict(ml, judge);

  // When the model is the form judge, its verdict decides correctness and
  // produces the cue; otherwise the rules decide and the model just reports.
  let formOk;
  if (ml && judge) {
    formOk = ml.label === judge.good;
    violations = formOk ? [] : [capitalize(ml.name)];
  } else {
    formOk = violations.length === 0;
  }

  if (exercise.isHold) {
    // Plank: accumulate good-form hold time.
    if (formOk) holdMs += now - (analyze._last || now);
    analyze._last = now;
    ui.primaryValue.textContent = `${Math.round(holdMs / 1000)}s`;
    ui.tempoVal.textContent = "—";
  } else {
    const ang = repSmoother.push(exercise.repAngle(lm));
    const count = counter.update(ang, now);
    ui.primaryValue.textContent = String(count);
    ui.repChipMini.textContent = `${count} reps`;
    ui.tempoVal.textContent = counter.tempoMs ? `${(counter.tempoMs / 1000).toFixed(1)}s` : "—";
  }

  // Form UI
  if (formOk) {
    setFormBadge("Good form", "good");
    setCues([{ text: "Clean rep — keep it up.", kind: "good" }]);
    ui.formChipMini.textContent = "Good form";
    ui.formChipMini.className = "chip chip-form is-good";
    ui.formCard.style.borderColor = "rgba(94,234,212,0.35)";
  } else {
    setFormBadge("Fix form", "bad");
    setCues(violations.map((t) => ({ text: t, kind: "bad" })));
    ui.formChipMini.textContent = violations[0];
    ui.formChipMini.className = "chip chip-form is-bad";
    ui.formCard.style.borderColor = "rgba(251,113,133,0.35)";
  }
}

// ---------- UI helpers ----------
function setStatus(live) {
  ui.statusPill.classList.toggle("is-live", live);
  ui.statusText.textContent = live ? "Live" : "Idle";
}
function capitalize(s) {
  return s.charAt(0).toUpperCase() + s.slice(1);
}
function updateMlVerdict(ml, judge) {
  if (!ml) {
    ui.mlVerdict.hidden = true;
    return;
  }
  ui.mlVerdict.hidden = false;
  const good = judge ? ml.label === judge.good : null;
  ui.mlVerdictText.textContent = judge ? capitalize(ml.name) : `Stage: ${ml.name}`;
  ui.mlVerdictText.className =
    "ml-text" + (good === true ? " is-good" : good === false ? " is-bad" : "");
  ui.mlVerdictProb.textContent = `${Math.round(ml.prob * 100)}%`;
}
function setFormBadge(text, kind) {
  ui.formBadge.textContent = text;
  ui.formBadge.className = "form-badge" + (kind ? ` is-${kind}` : "");
}
function setCues(items) {
  ui.cues.innerHTML = "";
  for (const it of items) {
    const li = document.createElement("li");
    li.className = "cue" + (it.kind === "muted" ? " cue-muted" : it.kind === "good" ? " cue-good" : "");
    li.textContent = it.text;
    ui.cues.appendChild(li);
  }
}
function fmtClock(ms) {
  const s = Math.floor(ms / 1000);
  return `${String(Math.floor(s / 60)).padStart(2, "0")}:${String(s % 60).padStart(2, "0")}`;
}

// ---------- Events ----------
ui.startBtn.addEventListener("click", start);
ui.stopBtn.addEventListener("click", stop);
ui.resetBtn.addEventListener("click", () => running && resetSession());

ui.exButtons.forEach((btn) => {
  btn.addEventListener("click", () => {
    ui.exButtons.forEach((b) => b.classList.remove("is-active"));
    btn.classList.add("is-active");
    exerciseKey = btn.dataset.ex;
    exercise = EXERCISES[exerciseKey];
    formModel = formModels[exerciseKey] ?? null;
    ensureModel(exerciseKey);
    if (running) resetSession();
    else {
      ui.primaryLabel.textContent = exercise.primaryLabel;
      ui.primarySub.textContent = `${exercise.name} session`;
    }
  });
});
