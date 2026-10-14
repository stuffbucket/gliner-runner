import type { Constraint } from "./constraints.js";

/** Label with its optional description (`LabelSpec`). */
export interface LabelSpec {
  readonly name: string;
  readonly description: string | null;
}

export type Activation = "auto" | "sigmoid" | "softmax";

/** Validated, normalized task (`TaskSpec`). Field semantics: dist/port-spec/gliner2-classification.md section 2.1. */
export interface TaskSpec {
  readonly name: string;
  readonly labels: readonly LabelSpec[];
  readonly labelNames: readonly string[];
  /** After the `default` adjustment (`default` forces at least 1). */
  readonly minLabels: number;
  /** `null` = unbounded. */
  readonly maxLabels: number | null;
  readonly ordered: boolean;
  readonly threshold: number;
  readonly candidateThreshold: number | null;
  readonly activation: Activation;
  readonly temperature: number;
  readonly default: string | null;
  readonly instruction: string | null;
  readonly examples: ReadonlyArray<readonly [string, string]>;
  /** `minLabels === 1 && maxLabels === 1`. */
  readonly isExclusive: boolean;
  /** `labels.length` when `maxLabels` is null. */
  readonly effectiveMax: number;
}

/** Anything with a name-indexed task table (a schema or a compiled schema). */
export interface TaskLookup {
  readonly tasks: ReadonlyMap<string, TaskSpec>;
}

/** Parsed `ClassificationSchema`; `tasks` is in declaration order. */
export interface ClassificationSchema extends TaskLookup {
  readonly tasks: ReadonlyMap<string, TaskSpec>;
  readonly constraints: readonly Constraint[];
}

/** One entry of the model schema (`classifications[i]` of the emitted model schema). */
export interface ModelTaskEntry {
  readonly task: string;
  readonly labels: readonly string[];
  readonly multiLabel: boolean;
  readonly clsThreshold: number;
  readonly classAct: Activation;
  /** The task instruction (only present when non-empty). */
  readonly prompt?: string;
  /** Ordered `[label, description]` pairs for labels that have a description. */
  readonly labelDescriptions?: ReadonlyArray<readonly [string, string]>;
  readonly examples?: ReadonlyArray<readonly [string, string]>;
}

/** Result of {@link compileSchema}. */
export interface CompiledSchema extends TaskLookup {
  /** Model-schema entries, one per task, in declaration order (what the prompt builder consumes). */
  readonly classifications: readonly ModelTaskEntry[];
  readonly specs: readonly TaskSpec[];
  /** Declared constraints followed by the lowered default rules. */
  readonly constraints: readonly Constraint[];
  readonly taskOrder: readonly string[];
  readonly tasks: ReadonlyMap<string, TaskSpec>;
  /** Total number of labels over all tasks (= logits per row). */
  readonly labelCount: number;
  /** Start offset of each task's logits in a flat row, parallel to `specs`. */
  readonly labelOffsets: readonly number[];
  /** Lookup that throws `Gliner2SchemaError("unknown task ...")` like Python's `task_spec`. */
  task(name: string): TaskSpec;
}
