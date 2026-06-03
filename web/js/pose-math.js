// pose-math.js — small geometry helpers shared across the app.

/**
 * Angle at point b formed by a-b-c, in degrees (0–180).
 * Each point is a MediaPipe landmark with normalized {x, y}.
 */
export function angle(a, b, c) {
  const abx = a.x - b.x, aby = a.y - b.y;
  const cbx = c.x - b.x, cby = c.y - b.y;
  const dot = abx * cbx + aby * cby;
  const magAB = Math.hypot(abx, aby);
  const magCB = Math.hypot(cbx, cby);
  const cos = dot / (magAB * magCB + 1e-8);
  return (Math.acos(Math.max(-1, Math.min(1, cos))) * 180) / Math.PI;
}

/** Exponential smoothing to reduce jitter on a scalar signal. */
export class Smoother {
  constructor(alpha = 0.4) {
    this.alpha = alpha;
    this.value = null;
  }
  push(x) {
    if (x == null || Number.isNaN(x)) return this.value;
    this.value = this.value == null ? x : this.alpha * x + (1 - this.alpha) * this.value;
    return this.value;
  }
}

/**
 * Rep counter as a two-state machine driven by an angle signal.
 * Counts a rep on each full down→up transition.
 */
export class RepCounter {
  constructor(downThresh, upThresh) {
    this.down = downThresh;
    this.up = upThresh;
    this.state = "up";
    this.count = 0;
    this.lastDownTime = null;
    this.tempoMs = null; // time of last full rep
  }
  update(angleVal, nowMs) {
    if (this.state === "up" && angleVal <= this.down) {
      this.state = "down";
      this.lastDownTime = nowMs;
    } else if (this.state === "down" && angleVal >= this.up) {
      this.state = "up";
      this.count += 1;
      if (this.lastDownTime != null) this.tempoMs = nowMs - this.lastDownTime;
    }
    return this.count;
  }
  get phase() {
    return this.state;
  }
}
