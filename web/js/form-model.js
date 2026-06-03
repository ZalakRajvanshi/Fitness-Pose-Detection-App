// form-model.js — runs the exported linear classifier in the browser.
// No ML runtime: the model is a scaler + one softmax layer, so the whole
// forward pass is a standardise + matmul + softmax in plain JS.

function softmax(v) {
  const max = Math.max(...v);
  const exps = v.map((x) => Math.exp(x - max));
  const sum = exps.reduce((a, b) => a + b, 0);
  return exps.map((e) => e / sum);
}

function relu(v) {
  return v.map((x) => Math.max(0, x));
}

// out[j] = sum_i a[i] * W[i][j] + b[j]
function dense(a, W, b) {
  const nOut = b.length;
  const out = new Array(nOut).fill(0);
  for (let i = 0; i < a.length; i++) {
    const wi = W[i];
    const ai = a[i];
    for (let j = 0; j < nOut; j++) out[j] += ai * wi[j];
  }
  for (let j = 0; j < nOut; j++) out[j] += b[j];
  return out;
}

export class FormModel {
  constructor(spec) {
    this.spec = spec;
  }

  static async load(url) {
    const res = await fetch(url);
    if (!res.ok) throw new Error(`model fetch failed: ${url}`);
    return new FormModel(await res.json());
  }

  /** Build the standardised input vector from MediaPipe landmarks. */
  vectorize(landmarks) {
    const { feature_plan, scaler } = this.spec;
    const z = new Array(feature_plan.length);
    for (let i = 0; i < feature_plan.length; i++) {
      const [idx, axis] = feature_plan[i];
      const lm = landmarks[idx];
      const raw = lm ? lm[axis] : scaler.mean[i]; // missing -> neutral (z=0)
      z[i] = (raw - scaler.mean[i]) / scaler.scale[i];
    }
    return z;
  }

  /**
   * @returns {{label:string, name:string, prob:number, probs:number[]}}
   */
  predict(landmarks) {
    let a = this.vectorize(landmarks);
    const layers = this.spec.layers;
    for (let l = 0; l < layers.length; l++) {
      const { W, b, activation } = layers[l];
      a = dense(a, W, b);
      a = activation === "softmax" ? softmax(a) : relu(a);
    }
    let best = 0;
    for (let j = 1; j < a.length; j++) if (a[j] > a[best]) best = j;
    return {
      label: this.spec.classes[best],
      name: this.spec.names[best],
      prob: a[best],
      probs: a,
    };
  }
}
