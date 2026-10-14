/** Result construction and `to_dict` (gliner2/classification/result.py). */
import type { Constraint } from "./constraints.js";
import type { DecoderName, Solution } from "./decoders.js";
import { OrderedObject, serializeOrdered } from "./ordered-json.js";
import type { OrderedJson } from "./ordered-json.js";
import { sortedStrs } from "./pyparity.js";
import { geometricMean, taskProbabilities } from "./scoring.js";
import type { Scores } from "./scoring.js";
import type { CompiledSchema } from "./types.js";

/** Per-task outcome (`TaskResult`). */
export interface TaskOutcome {
  /** Selected labels in declaration order (not utility order). */
  readonly labels: readonly string[];
  /** Reported probabilities of every label, declaration order. */
  readonly probabilities: ReadonlyMap<string, number>;
  /** `null` when not computed (`includeConfidence=false`) or for an empty selection without a default. */
  readonly confidence: number | null;
  readonly exclusive: boolean;
}

/** Python `Violation.__str__`. */
export function violationString(c: Constraint): string {
  return `violated ${c.repr()} (tasks: ${sortedStrs(c.references()).join(", ")}; weight 1.0)`;
}

/** Decoded classification of one text (`ClassificationResult`). */
export class Gliner2Result {
  constructor(
    /** task -> outcome, declaration order. */
    readonly tasks: ReadonlyMap<string, TaskOutcome>,
    readonly feasible: boolean,
    readonly violations: readonly string[],
    readonly objective: number,
    readonly decoder: DecoderName,
    readonly exact: boolean,
    /** The flag the result was built with (decides whether confidences were computed). */
    readonly includeConfidence: boolean,
  ) {}

  /**
   * `to_dict(include_confidence)` as an order-preserving object. Without confidence the value is
   * `{task: value}` and `_meta` is present only when infeasible. `include` defaults to the build flag.
   */
  toDict(include: boolean = this.includeConfidence): OrderedObject {
    const out: Array<readonly [string, OrderedJson]> = [];
    for (const [name, tr] of this.tasks) {
      const value: OrderedJson = tr.exclusive ? (tr.labels[0] ?? null) : [...tr.labels];
      if (include) {
        out.push([
          name,
          new OrderedObject([
            ["value", value],
            ["confidence", tr.confidence],
            ["probabilities", new OrderedObject([...tr.probabilities])],
          ]),
        ]);
      } else {
        out.push([name, value]);
      }
    }
    if (include || !this.feasible) {
      out.push([
        "_meta",
        new OrderedObject([
          ["feasible", this.feasible],
          ["decoder", this.decoder],
          ["exact", this.exact],
          ["objective", this.objective],
          ["violations", [...this.violations]],
        ]),
      ]);
    }
    return new OrderedObject(out);
  }

  /** The response `output` object: `to_dict(include_confidence=<build flag>)`, exactly what the PyTorch backend emits. */
  get output(): OrderedObject {
    return this.toDict();
  }

  /** JSON text of {@link output} with Python key order (also for integer-like task or label names). */
  serialize(): string {
    return serializeOrdered(this.output);
  }
}

/** `ResultBuilder.build`. */
export function buildResult(
  compiled: CompiledSchema,
  scores: Scores,
  solution: Solution,
  order: readonly string[],
  includeConfidence: boolean,
): Gliner2Result {
  const tasks = new Map<string, TaskOutcome>();
  for (const task of order) {
    const spec = compiled.task(task);
    const selected = (solution.assignments.get(task) as { labels: ReadonlySet<string> }).labels;
    const labels = spec.labelNames.filter((l) => selected.has(l));
    const probs = taskProbabilities(spec, scores.get(task) as ReadonlyMap<string, number>);
    const probOf = (l: string): number => probs.get(l) as number;
    let confidence: number | null = null;
    if (includeConfidence) {
      if (labels.length === 0) confidence = spec.default !== null ? 1.0 : null;
      else if (spec.isExclusive) confidence = probOf(labels[0] as string);
      else {
        const sel = new Set(labels);
        confidence = geometricMean(spec.labelNames.map((l) => (sel.has(l) ? probOf(l) : 1.0 - probOf(l))));
      }
    }
    const ordered = new Map(spec.labelNames.map((l) => [l, probOf(l)] as const));
    tasks.set(task, { labels, probabilities: ordered, confidence, exclusive: spec.isExclusive });
  }
  return new Gliner2Result(
    tasks,
    solution.violations.length === 0,
    solution.violations.map(violationString),
    solution.score,
    solution.decoder,
    solution.exact,
    includeConfidence,
  );
}
