/** Candidate retention and local assignment enumeration (gliner2/classification/candidates.py). */
import { cmpStr, pySum, sortedStrs, union } from "./pyparity.js";
import { centerLogit, sigmoid } from "./scoring.js";
import type { TaskLogits } from "./scoring.js";
import type { TaskSpec } from "./types.js";

export interface LocalAssignment {
  readonly task: string;
  readonly labels: ReadonlySet<string>;
  readonly utility: number;
}

export interface RetainOptions {
  readonly candidateThreshold: number;
  readonly cap: number;
  readonly rescued: ReadonlySet<string>;
}

/** `retain()`: labels whose `sigmoid(logit / T)` reaches the retention floor (capped), plus constraint-rescued labels. */
export function retain(spec: TaskSpec, logits: TaskLogits, options: RetainOptions): Set<string> {
  const floor = spec.candidateThreshold !== null ? spec.candidateThreshold : options.candidateThreshold;
  const finite = new Map([...logits].filter(([, v]) => Number.isFinite(v)));
  const probs = new Map([...finite].map(([l, v]) => [l, sigmoid(v / spec.temperature)] as const));
  const utils = new Map([...finite].map(([l, v]) => [l, centerLogit(v / spec.temperature, spec.threshold)] as const));
  let keep = new Set([...probs].filter(([, p]) => p >= floor).map(([l]) => l));
  if (keep.size > options.cap) {
    keep = new Set(
      [...keep]
        .sort((a, b) => (utils.get(b) as number) - (utils.get(a) as number) || cmpStr(a, b))
        .slice(0, options.cap),
    );
  }
  if (keep.size < spec.minLabels) keep = new Set(finite.keys());
  return union(
    keep,
    [...options.rescued].filter((l) => finite.has(l)),
  );
}

const utilityOf = (labels: Iterable<string>, utils: ReadonlyMap<string, number>): number =>
  pySum([...labels].map((l) => utils.get(l) as number));

/** All subsets of `items`, ascending size (order irrelevant after the final sort). */
function* subsetsOf(items: Iterable<string>): Generator<Set<string>> {
  const arr = sortedStrs(items);
  for (let size = 0; size <= arr.length; size++) yield* combos(arr, size, 0, []);
}
function* combos(arr: readonly string[], k: number, start: number, cur: string[]): Generator<Set<string>> {
  if (cur.length === k) {
    yield new Set(cur);
    return;
  }
  for (let i = start; i < arr.length; i++) {
    cur.push(arr[i] as string);
    yield* combos(arr, k, i + 1, cur);
    cur.pop();
  }
}

function cmpLocals(a: LocalAssignment, b: LocalAssignment): number {
  const ua = -a.utility;
  const ub = -b.utility;
  if (ua < ub) return -1;
  if (ua > ub) return 1;
  const A = sortedStrs(a.labels);
  const B = sortedStrs(b.labels);
  for (let i = 0; i < Math.min(A.length, B.length); i++) {
    const c = cmpStr(A[i] as string, B[i] as string);
    if (c !== 0) return c;
  }
  return A.length - B.length;
}
const sortedLocals = (ls: LocalAssignment[]): LocalAssignment[] => ls.sort(cmpLocals);

export interface EnumerateOptions {
  readonly boundLabels: ReadonlySet<string>;
  readonly setCoupled: boolean;
  readonly countCoupled: boolean;
}

const rankFree = (labels: Iterable<string>, utils: ReadonlyMap<string, number>): string[] =>
  [...labels].sort((a, b) => (utils.get(b) as number) - (utils.get(a) as number) || cmpStr(a, b));

/** `enumerate_locals()`: candidate selections for one task, utility-descending then by sorted label tuple. */
export function enumerateLocals(
  spec: TaskSpec,
  retained: ReadonlySet<string>,
  utils: ReadonlyMap<string, number>,
  options: EnumerateOptions,
): LocalAssignment[] {
  const minl = spec.minLabels;
  const maxl = Math.min(spec.effectiveMax, retained.size);
  if (spec.isExclusive) {
    return sortedLocals(
      [...retained].map((l) => ({ task: spec.name, labels: new Set([l]), utility: utils.get(l) as number })),
    );
  }
  if (options.setCoupled) {
    const out: LocalAssignment[] = [];
    for (const base of subsetsOf(retained)) {
      if (minl <= base.size && base.size <= maxl) {
        out.push({ task: spec.name, labels: base, utility: utilityOf(base, utils) });
      }
    }
    return sortedLocals(out);
  }
  const bound = new Set([...retained].filter((l) => options.boundLabels.has(l)));
  const free = new Set([...retained].filter((l) => !bound.has(l)));
  const freeRanked = rankFree(free, utils);
  const out: LocalAssignment[] = [];
  for (const base of subsetsOf(bound)) {
    if (base.size > maxl) continue;
    if (options.countCoupled) {
      const lo = Math.max(minl, base.size);
      const hi = Math.min(maxl, base.size + free.size);
      for (let count = lo; count <= hi; count++) {
        const need = count - base.size;
        const selected = union(base, freeRanked.slice(0, need));
        if (selected.size === count) out.push({ task: spec.name, labels: selected, utility: utilityOf(selected, utils) });
      }
      continue;
    }
    const selected = new Set(base);
    for (const f of freeRanked) {
      if (selected.size >= maxl) break;
      if ((utils.get(f) as number) > 0) selected.add(f);
    }
    if (selected.size < minl) {
      for (const f of freeRanked) {
        if (selected.size >= minl) break;
        if (!selected.has(f)) selected.add(f);
      }
    }
    if (!(minl <= selected.size && selected.size <= maxl)) continue;
    out.push({ task: spec.name, labels: selected, utility: utilityOf(selected, utils) });
  }
  return sortedLocals(out);
}

/** `_fallback_locals` (used e.g. when `min_labels > len(labels)` with an unbounded task). */
export function fallbackLocals(
  spec: TaskSpec,
  retained: ReadonlySet<string>,
  utils: ReadonlyMap<string, number>,
): LocalAssignment[] {
  const ranked = rankFree(retained, utils);
  const maxl = Math.min(spec.effectiveMax, ranked.length);
  let count = Math.max(spec.minLabels, 1);
  count = maxl > 0 ? Math.min(count, maxl) : 0;
  const chosen = new Set(ranked.slice(0, count));
  return [{ task: spec.name, labels: chosen, utility: utilityOf(chosen, utils) }];
}
