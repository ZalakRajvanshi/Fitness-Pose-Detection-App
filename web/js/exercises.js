// exercises.js
// Per-exercise configuration: which joints to track, rep thresholds,
// and the rule-based form checks that generate live coaching cues.
//
// MediaPipe Pose landmark indices used here:
//   11 L-shoulder 12 R-shoulder  13 L-elbow 14 R-elbow  15 L-wrist 16 R-wrist
//   23 L-hip      24 R-hip       25 L-knee  26 R-knee    27 L-ankle 28 R-ankle

import { angle } from "./pose-math.js";

export const EXERCISES = {
  squat: {
    name: "Squat",
    primaryLabel: "Reps",
    isHold: false,

    // The angle that drives rep counting (averaged left/right knee).
    repAngle: (lm) => {
      const l = angle(lm[23], lm[25], lm[27]); // hip-knee-ankle (L)
      const r = angle(lm[24], lm[26], lm[28]); // hip-knee-ankle (R)
      return (l + r) / 2;
    },
    // State machine: go "down" below `down`, complete rep when back above `up`.
    down: 95,
    up: 160,

    // Returns { knee, back } for the live-metrics panel.
    metrics: (lm) => ({
      knee: (angle(lm[23], lm[25], lm[27]) + angle(lm[24], lm[26], lm[28])) / 2,
      back: angle(lm[11], lm[23], lm[25]), // shoulder-hip-knee = torso lean
    }),

    // Form rules. Each returns a cue string when violated, else null.
    // `phase` is "down" | "up", `depthAngle` is the rep angle.
    checks: [
      ({ m, phase }) =>
        phase === "down" && m.knee > 110
          ? "Go deeper — aim for thighs near parallel"
          : null,
      ({ m }) =>
        m.back < 70 ? "Chest up — you're leaning too far forward" : null,
      ({ lm }) => {
        // Knees caving in: knees should stay at least as wide as ankles.
        const kneeW = Math.abs(lm[25].x - lm[26].x);
        const ankleW = Math.abs(lm[27].x - lm[28].x);
        return kneeW < ankleW * 0.7 ? "Push your knees out, don't let them cave" : null;
      },
    ],
  },

  pushup: {
    name: "Push-up",
    primaryLabel: "Reps",
    isHold: false,

    repAngle: (lm) => {
      const l = angle(lm[11], lm[13], lm[15]); // shoulder-elbow-wrist (L)
      const r = angle(lm[12], lm[14], lm[16]); // shoulder-elbow-wrist (R)
      return (l + r) / 2;
    },
    down: 95,
    up: 160,

    metrics: (lm) => ({
      knee: (angle(lm[11], lm[13], lm[15]) + angle(lm[12], lm[14], lm[16])) / 2, // elbow
      back: angle(lm[11], lm[23], lm[25]), // shoulder-hip-knee = body line
    }),

    checks: [
      ({ m }) =>
        m.back < 150 ? "Keep a straight line — don't let hips sag or pike" : null,
      ({ m, phase }) =>
        phase === "down" && m.knee > 110
          ? "Lower more — bend elbows toward 90°"
          : null,
    ],
  },

  plank: {
    name: "Plank",
    primaryLabel: "Hold",
    isHold: true, // timed hold instead of reps

    metrics: (lm) => ({
      knee: angle(lm[23], lm[25], lm[27]), // leg straightness
      back: angle(lm[11], lm[23], lm[25]), // body line
    }),

    // For a hold, "good form" = holding the line. checks decide good/bad.
    checks: [
      ({ m }) => (m.back < 155 ? "Hips too high — flatten your back" : null),
      ({ m }) => (m.back > 195 ? "Hips sagging — squeeze your glutes" : null),
      ({ m }) => (m.knee < 150 ? "Straighten your legs" : null),
    ],
  },
};
