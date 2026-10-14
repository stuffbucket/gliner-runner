/** Decoders and the infeasibility ladder (gliner2/classification/decoding/*). */
import type { Constraint } from "./constraints.js";
import type { LocalAssignment } from "./candidates.js";
import { BudgetExceeded, Gliner2ConfigError, Gliner2InfeasibleError } from "./errors.js";
import type { DecodeProblem } from "./problem.js";
import { cmpStr, pySum, sortedStrs } from "./pyparity.js";

export type DecoderName = "independent" | "exact" | "beam" | "min_violations";

export interface Solution {
  /** Insertion order matters only for the compensated sum order (exactly like the Python dict). */
  readonly assignments: ReadonlyMap<string, LocalAssignment>;
  readonly score: number;
  readonly violations: readonly Constraint[];
  readonly exact: boolean;
  readonly decoder: DecoderName;
}

export type DecoderRequest = "auto" | "independent" | "exact" | "beam";
export type InfeasibleMode = "relax" | "min_violations" | "raise";

/** Python `ClassificationConfig` fields that influence decoding. */
export interface ResolvedDecodeConfig {
  readonly decoder: DecoderRequest;
  readonly exactNodeBudget: number;
  readonly beamSize: number;
  readonly candidateThreshold: number;
  readonly maxCandidatesPerTask: number;
  readonly includeConfidence: boolean;
  readonly onInfeasible: InfeasibleMode;
}

const DECODERS: readonly string[] = ["auto", "independent", "exact", "beam"];
const INFEASIBLE_MODES: readonly string[] = ["relax", "min_violations", "raise"];

/** Public decode configuration. Only `candidateThreshold` and `includeConfidence` are set by the runner. */
export interface DecodeConfig {
  /** Retention floor on `sigmoid(logit / temperature)` (the request `options.threshold`), in [0, 1]. */
  readonly candidateThreshold: number;
  readonly includeConfidence: boolean;
  readonly decoder?: DecoderRequest;
  readonly exactNodeBudget?: number;
  readonly beamSize?: number;
  readonly maxCandidatesPerTask?: number;
  readonly onInfeasible?: InfeasibleMode;
}

/** Applies defaults (`exact_node_budget=200000, beam_size=16, max_candidates_per_task=64, decoder="auto", on_infeasible="relax"`) and validates. */
export function resolveConfig(config: DecodeConfig): ResolvedDecodeConfig {
  const resolved: ResolvedDecodeConfig = {
    decoder: config.decoder ?? "auto",
    exactNodeBudget: config.exactNodeBudget ?? 200_000,
    beamSize: config.beamSize ?? 16,
    candidateThreshold: config.candidateThreshold,
    maxCandidatesPerTask: config.maxCandidatesPerTask ?? 64,
    includeConfidence: config.includeConfidence,
    onInfeasible: config.onInfeasible ?? "relax",
  };
  if (!DECODERS.includes(resolved.decoder)) throw new Gliner2ConfigError(`decoder must be one of ${DECODERS.join(", ")}`);
  if (!INFEASIBLE_MODES.includes(resolved.onInfeasible)) {
    throw new Gliner2ConfigError(`on_infeasible must be one of ${INFEASIBLE_MODES.join(", ")}`);
  }
  if (!(resolved.exactNodeBudget > 0)) throw new Gliner2ConfigError("exact_node_budget must be positive");
  if (!(resolved.beamSize > 0)) throw new Gliner2ConfigError("beam_size must be positive");
  if (!(0 <= resolved.candidateThreshold && resolved.candidateThreshold <= 1)) {
    throw new Gliner2ConfigError("candidate_threshold must be in [0, 1]");
  }
  if (!(resolved.maxCandidatesPerTask > 0)) throw new Gliner2ConfigError("max_candidates_per_task must be positive");
  return resolved;
}

const orderTasks = (p: DecodeProblem): string[] =>
  [...p.taskOrder].sort(
    (a, b) => p.touching(b).length - p.touching(a).length || p.localsOf(a).length - p.localsOf(b).length || cmpStr(a, b),
  );

type Signature = ReadonlyArray<readonly [string, readonly string[]]>;
const signature = (chosen: ReadonlyMap<string, LocalAssignment>, order: readonly string[]): Signature =>
  order.filter((t) => chosen.has(t)).map((t) => [t, sortedStrs((chosen.get(t) as LocalAssignment).labels)] as const);

function cmpSig(A: Signature, B: Signature): number {
  for (let i = 0; i < Math.min(A.length, B.length); i++) {
    const a = A[i] as readonly [string, readonly string[]];
    const b = B[i] as readonly [string, readonly string[]];
    let c = cmpStr(a[0], b[0]);
    if (c !== 0) return c;
    const x = a[1];
    const y = b[1];
    for (let j = 0; j < Math.min(x.length, y.length); j++) {
      c = cmpStr(x[j] as string, y[j] as string);
      if (c !== 0) return c;
    }
    if (x.length !== y.length) return x.length - y.length;
  }
  return A.length - B.length;
}

function independentDecode(p: DecodeProblem): Solution {
  const chosen = new Map<string, LocalAssignment>();
  for (const task of p.taskOrder) {
    const touching = p.touching(task);
    let picked: LocalAssignment | null = null;
    for (const local of p.localsOf(task)) {
      const a = p.assignment(new Map([[task, local]]), [task]);
      if (touching.every((c) => c.evaluate(a) !== false)) {
        picked = local;
        break;
      }
    }
    chosen.set(task, picked ?? (p.localsOf(task)[0] as LocalAssignment));
  }
  const score = pySum([...chosen.values()].map((l) => l.utility));
  return { assignments: chosen, score, violations: p.violationsOf(chosen), exact: true, decoder: "independent" };
}

function exactDecode(p: DecodeProblem, budget: number): Solution | null {
  const order = orderTasks(p);
  const suffix = new Array<number>(order.length + 1).fill(0.0);
  for (let i = order.length - 1; i >= 0; i--) {
    const ls = p.localsOf(order[i] as string);
    let best = 0.0;
    if (ls.length > 0) {
      best = -Infinity;
      for (const l of ls) if (l.utility > best) best = l.utility;
    }
    suffix[i] = best + (suffix[i + 1] as number);
  }
  const best: { assign: Map<string, LocalAssignment> | null; score: number } = { assign: null, score: -Infinity };
  let nodes = 0;
  const chosen = new Map<string, LocalAssignment>();
  const dfs = (i: number, score: number): void => {
    nodes += 1;
    if (nodes > budget) throw new BudgetExceeded();
    if (score + (suffix[i] as number) <= best.score) return;
    if (i === order.length) {
      best.assign = new Map(chosen);
      best.score = score;
      return;
    }
    const task = order[i] as string;
    const touching = p.touching(task);
    for (const local of p.localsOf(task)) {
      if (score + local.utility + (suffix[i + 1] as number) <= best.score) break;
      chosen.set(task, local);
      const a = p.assignment(chosen, order.slice(0, i + 1));
      if (touching.every((c) => c.evaluate(a) !== false)) dfs(i + 1, score + local.utility);
      chosen.delete(task);
    }
  };
  dfs(0, 0.0);
  if (best.assign === null) return null;
  return { assignments: best.assign, score: best.score, violations: [], exact: true, decoder: "exact" };
}

function beamDecode(p: DecodeProblem, beamSize: number): Solution | null {
  const order = orderTasks(p);
  let beams: Array<[number, Map<string, LocalAssignment>]> = [[0.0, new Map()]];
  for (let i = 0; i < order.length; i++) {
    const task = order[i] as string;
    const touching = p.touching(task);
    const expanded: Array<[number, Map<string, LocalAssignment>]> = [];
    for (const [score, chosen] of beams) {
      for (const local of p.localsOf(task)) {
        const next = new Map(chosen);
        next.set(task, local);
        const a = p.assignment(next, order.slice(0, i + 1));
        if (touching.every((c) => c.evaluate(a) !== false)) expanded.push([score + local.utility, next]);
      }
    }
    const keyed = expanded.map((e) => ({ e, sig: signature(e[1], order) }));
    keyed.sort((x, y) => {
      if (-x.e[0] < -y.e[0]) return -1;
      if (-x.e[0] > -y.e[0]) return 1;
      return cmpSig(x.sig, y.sig);
    });
    beams = [];
    const seen = new Set<string>();
    for (const { e, sig } of keyed) {
      const key = JSON.stringify(sig);
      if (seen.has(key)) continue;
      seen.add(key);
      beams.push(e);
      if (beams.length >= beamSize) break;
    }
    if (beams.length === 0) return null;
  }
  const [score, chosen] = beams[0] as [number, Map<string, LocalAssignment>];
  return {
    assignments: new Map(chosen),
    score,
    violations: p.violationsOf(chosen),
    exact: false,
    decoder: "beam",
  };
}

function minViolationsDecode(p: DecodeProblem, budget: number): Solution {
  const order = orderTasks(p);
  const best: { assign: Map<string, LocalAssignment> | null; weight: number; score: number } = {
    assign: null,
    weight: Infinity,
    score: -Infinity,
  };
  let nodes = 0;
  const chosen = new Map<string, LocalAssignment>();
  const dfs = (i: number, score: number): void => {
    nodes += 1;
    if (nodes > budget) throw new BudgetExceeded();
    if (i === order.length) {
      const weight = p.violationsOf(chosen).length;
      // key = (weight, -score) tuple comparison
      if (weight < best.weight || (weight === best.weight && -score < -best.score)) {
        best.assign = new Map(chosen);
        best.weight = weight;
        best.score = score;
      }
      return;
    }
    const task = order[i] as string;
    for (const local of p.localsOf(task)) {
      chosen.set(task, local);
      dfs(i + 1, score + local.utility);
      chosen.delete(task);
    }
  };
  let exhausted = false;
  try {
    dfs(0, 0.0);
  } catch (error) {
    if (error instanceof BudgetExceeded) exhausted = true;
    else throw error;
  }
  const assign: Map<string, LocalAssignment> =
    best.assign ?? new Map(p.taskOrder.map((t) => [t, p.localsOf(t)[0] as LocalAssignment] as const));
  // dict(assign) insertion order == DFS `order` (tasks sorted), NOT declaration order; matters for the Neumaier sum order only.
  const score = pySum([...assign.values()].map((l) => l.utility));
  return { assignments: new Map(assign), score, violations: p.violationsOf(assign), exact: !exhausted, decoder: "min_violations" };
}

function selectDecoder(p: DecodeProblem, requested: DecoderRequest): Exclude<DecoderRequest, "auto"> {
  if (requested !== "auto") return requested;
  return p.constraints.some((c) => c.references().size > 1) ? "exact" : "independent";
}

function primary(p: DecodeProblem, config: ResolvedDecodeConfig): Solution | null {
  const decoder = selectDecoder(p, config.decoder);
  if (decoder === "independent") {
    const s = independentDecode(p);
    return s.violations.length === 0 ? s : null;
  }
  if (decoder === "beam") {
    const s = beamDecode(p, config.beamSize);
    return s !== null && s.violations.length === 0 ? s : null;
  }
  let sol: Solution | null;
  try {
    sol = exactDecode(p, config.exactNodeBudget);
  } catch (error) {
    if (error instanceof BudgetExceeded) sol = beamDecode(p, config.beamSize);
    else throw error;
  }
  return sol !== null && sol.violations.length === 0 ? sol : null;
}

/** `decode()` with the `relax` / `min_violations` / `raise` ladder. `widen` rebuilds the problem with full retention. */
export function decodeLadder(
  problem: DecodeProblem,
  config: ResolvedDecodeConfig,
  widen: () => DecodeProblem,
  describeViolation: (c: Constraint) => string,
): Solution {
  const sol = primary(problem, config);
  if (sol !== null) return sol;
  const mode = config.onInfeasible;
  let working = problem;
  if (mode === "relax") {
    const relaxed = widen();
    const recovered = primary(relaxed, config);
    if (recovered !== null) return recovered;
    working = relaxed;
  }
  if (mode === "relax" || mode === "min_violations") return minViolationsDecode(working, config.exactNodeBudget);
  const diagnosis = minViolationsDecode(working, config.exactNodeBudget);
  throw new Gliner2InfeasibleError(
    "no assignment satisfies the classification constraints",
    diagnosis.violations.map(describeViolation),
  );
}
