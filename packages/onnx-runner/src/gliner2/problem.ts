/** Decode problem construction (gliner2/classification/decoding/base.py). */
import type { Assignment, Constraint, Kleene } from "./constraints.js";
import { enumerateLocals, fallbackLocals, retain } from "./candidates.js";
import type { LocalAssignment } from "./candidates.js";
import { union } from "./pyparity.js";
import { utility } from "./scoring.js";
import type { Scores } from "./scoring.js";
import type { CompiledSchema } from "./types.js";

/** Assignment over local candidate sets: decided tasks use their chosen labels, others the always/possible sets. */
export class SearchAssignment implements Assignment {
  private readonly dec: Set<string>;
  constructor(
    private readonly problem: DecodeProblem,
    private readonly chosen: ReadonlyMap<string, LocalAssignment>,
    decided: Iterable<string>,
  ) {
    this.dec = new Set(decided);
  }
  isDecided(task: string): boolean {
    return this.dec.has(task);
  }
  selected(task: string): ReadonlySet<string> {
    return this.dec.has(task) ? (this.chosen.get(task) as LocalAssignment).labels : (this.problem.always.get(task) as ReadonlySet<string>);
  }
  domain(task: string): ReadonlySet<string> {
    return this.dec.has(task) ? (this.chosen.get(task) as LocalAssignment).labels : (this.problem.possible.get(task) as ReadonlySet<string>);
  }
  holds(task: string, label: string): Kleene {
    if (this.selected(task).has(label)) return true;
    if (this.domain(task).has(label)) return this.isDecided(task) ? false : null;
    return false;
  }
  levels(task: string): ReadonlySet<number> {
    const idx = this.problem.indexMap.get(task) as ReadonlyMap<string, number>;
    const out = new Set<number>();
    for (const l of this.domain(task)) {
      const i = idx.get(l);
      if (i !== undefined) out.add(i);
    }
    return out;
  }
  index(task: string, label: string): number {
    return (this.problem.indexMap.get(task) as ReadonlyMap<string, number>).get(label) ?? -1;
  }
  defaultLabel(task: string): string | null {
    return this.problem.schema.task(task).default;
  }
}

export class DecodeProblem {
  readonly taskOrder: readonly string[];
  readonly possible = new Map<string, ReadonlySet<string>>();
  readonly always = new Map<string, ReadonlySet<string>>();
  readonly indexMap = new Map<string, ReadonlyMap<string, number>>();
  private readonly touch = new Map<string, readonly Constraint[]>();

  constructor(
    readonly schema: CompiledSchema,
    taskOrder: readonly string[],
    readonly locals: ReadonlyMap<string, readonly LocalAssignment[]>,
    readonly constraints: readonly Constraint[],
  ) {
    this.taskOrder = [...taskOrder];
    for (const t of this.taskOrder) {
      const ls = locals.get(t) as readonly LocalAssignment[];
      const first = ls[0];
      this.possible.set(t, first === undefined ? new Set() : union(...ls.map((x) => x.labels)));
      this.always.set(
        t,
        first === undefined ? new Set() : new Set([...first.labels].filter((l) => ls.every((x) => x.labels.has(l)))),
      );
      this.indexMap.set(t, new Map(schema.task(t).labelNames.map((n, i) => [n, i] as const)));
    }
  }

  localsOf(task: string): readonly LocalAssignment[] {
    return this.locals.get(task) as readonly LocalAssignment[];
  }

  touching(task: string): readonly Constraint[] {
    let c = this.touch.get(task);
    if (c === undefined) {
      c = this.constraints.filter((k) => k.references().has(task));
      this.touch.set(task, c);
    }
    return c;
  }

  assignment(chosen: ReadonlyMap<string, LocalAssignment>, decided: Iterable<string>): SearchAssignment {
    return new SearchAssignment(this, chosen, decided);
  }

  /** Constraints (compiled order, incl. lowered defaults) that evaluate to `false` on a fully decided assignment. */
  violationsOf(chosen: ReadonlyMap<string, LocalAssignment>): Constraint[] {
    const a = new SearchAssignment(this, chosen, this.taskOrder);
    return this.constraints.filter((c) => c.evaluate(a) === false);
  }
}

export interface ProblemConfig {
  readonly candidateThreshold: number;
  readonly maxCandidatesPerTask: number;
}

/** `build_problem` (the runner never passes `active`, so every task is active). */
export function buildProblem(
  compiled: CompiledSchema,
  scores: Scores,
  config: ProblemConfig,
  fullRetentionTasks: Iterable<string> = [],
): DecodeProblem {
  const order = compiled.taskOrder;
  const activeSet = new Set(order);
  const full = new Set(fullRetentionTasks);
  const constraints = compiled.constraints.filter((c) => [...c.references()].every((t) => activeSet.has(t)));
  const labelRefs = new Map<string, Set<string>>();
  const setTasks = new Set<string>();
  const countTasks = new Set<string>();
  for (const c of constraints) {
    for (const [t, l] of c.labelReferences()) {
      let s = labelRefs.get(t);
      if (s === undefined) {
        s = new Set();
        labelRefs.set(t, s);
      }
      s.add(l);
    }
    for (const t of c.setReferences()) setTasks.add(t);
    for (const t of c.countReferences()) countTasks.add(t);
  }
  const locals = new Map<string, LocalAssignment[]>();
  for (const task of order) {
    const spec = compiled.task(task);
    const logits = scores.get(task);
    if (logits === undefined) throw new RangeError(`missing logits for task ${JSON.stringify(task)}`);
    const rescued = labelRefs.get(task) ?? new Set<string>();
    let retained: Set<string> = full.has(task)
      ? new Set(spec.labelNames)
      : retain(spec, logits, {
          candidateThreshold: config.candidateThreshold,
          cap: config.maxCandidatesPerTask,
          rescued,
        });
    if (retained.size < spec.minLabels) retained = new Set(spec.labelNames);
    const utils = new Map([...retained].map((l) => [l, utility(spec, logits.get(l) as number)] as const));
    const setCoupled = setTasks.has(task);
    const countCoupled = countTasks.has(task) && !setCoupled;
    let ls = enumerateLocals(spec, retained, utils, { boundLabels: rescued, setCoupled, countCoupled });
    if (ls.length === 0) ls = fallbackLocals(spec, retained, utils);
    locals.set(task, ls);
  }
  return new DecodeProblem(compiled, order, locals, constraints);
}
