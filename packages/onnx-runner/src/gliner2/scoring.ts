/** Probabilities, utilities and Python-parity float helpers (gliner2/classification/scoring.py, joint_ie/candidates.py). */
import { pySum } from "./pyparity.js";
import type { TaskSpec } from "./types.js";

export type TaskLogits = ReadonlyMap<string, number>;
/** task -> label -> logit; both levels in declaration order. */
export type Scores = ReadonlyMap<string, TaskLogits>;

export function sigmoid(x: number): number {
  if (x >= 0) {
    const z = Math.exp(-x);
    return 1.0 / (1.0 + z);
  }
  const z = Math.exp(x);
  return z / (1.0 + z);
}

export function softmax(values: readonly number[]): number[] {
  let m = -Infinity;
  for (const v of values) if (v > m) m = v;
  const exps = values.map((v) => Math.exp(v - m));
  const total = pySum(exps);
  return exps.map((e) => e / total);
}

function probabilityToLogit(p: number): number {
  if (!(0.0 <= p && p <= 1.0)) throw new RangeError("probability must be between zero and one");
  if (p === 0.0) return -Infinity;
  if (p === 1.0) return Infinity;
  return Math.log(p) - Math.log1p(-p);
}

/** `center_logit(x, t) = x - logit(t)`. */
export function centerLogit(logit: number, threshold: number): number {
  return logit - probabilityToLogit(threshold);
}

/** Geometric mean with Python's compensated log-sum; `null` for an empty list. */
export function geometricMean(values: readonly number[]): number | null {
  if (values.length === 0) return null;
  if (values.some((v) => v < 0 || v > 1)) throw new RangeError("confidence components must be probabilities in [0, 1]");
  if (values.some((v) => v === 0)) return 0.0;
  return Math.exp(pySum(values.map((v) => Math.log(v))) / values.length);
}

/** Reported probability of every label of a task (`activation` + `temperature`), in declaration order. */
export function taskProbabilities(spec: TaskSpec, logits: TaskLogits): Map<string, number> {
  let activation = spec.activation;
  if (activation === "auto") activation = !spec.isExclusive ? "sigmoid" : "softmax";
  const T = spec.temperature;
  const out = new Map<string, number>();
  if (activation === "softmax") {
    const names = [...logits.keys()];
    const probs = softmax(names.map((n) => (logits.get(n) as number) / T));
    names.forEach((n, i) => out.set(n, probs[i] as number));
    return out;
  }
  for (const [label, logit] of logits) out.set(label, sigmoid(logit / T));
  return out;
}

/** Decision utility of a logit: `center_logit(logit / temperature, threshold)`. */
export function utility(spec: TaskSpec, logit: number): number {
  return centerLogit(logit / spec.temperature, spec.threshold);
}
