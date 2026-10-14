/** Schema parsing and validation (gliner2/classification/schema.py, `ClassificationSchema.from_dict`). */
import { constraintFromDict } from "./constraints.js";
import { Gliner2MalformedSchemaError, Gliner2SchemaError } from "./errors.js";
import { pyIsBlank, pyRepr, pyReprValue, sortedStrs } from "./pyparity.js";
import type { Activation, ClassificationSchema, LabelSpec, TaskSpec } from "./types.js";

/** Strings that would corrupt the model prompt if they appeared inside a label/task/instruction. */
export const RESERVED_TOKENS: readonly string[] = [
  "[P]",
  "[L]",
  "[C]",
  "[E]",
  "[R]",
  "[DESCRIPTION]",
  "[EXAMPLE]",
  "[OUTPUT]",
  "(",
  ")",
];

const ACTIVATIONS: readonly string[] = ["auto", "sigmoid", "softmax"];

/** `_clean()`: non-empty (Python strip semantics) and free of reserved marker substrings. */
export function cleanPy(value: unknown, what: string): string {
  if (typeof value !== "string" || pyIsBlank(value)) {
    throw new Gliner2SchemaError(`${what} must be a non-empty string`);
  }
  for (const token of RESERVED_TOKENS) {
    if (value.includes(token)) {
      throw new Gliner2SchemaError(
        `${what} may not contain ${pyRepr(token)}: label and prompt strings are ` +
          `injected into the model prompt verbatim, and this would corrupt ` +
          `logit-to-label alignment`,
      );
    }
  }
  return value;
}

const isInt = (v: unknown): v is number => typeof v === "number" && Number.isInteger(v);
const isNum = (v: unknown): v is number => typeof v === "number";
const isMapping = (v: unknown): v is Record<string, unknown> =>
  v !== null && typeof v === "object" && !Array.isArray(v);

/** Label list input: names, or `{label: description|null}` as a plain object, a `Map`, or ordered pairs. */
export type LabelsInput =
  | readonly string[]
  | Readonly<Record<string, string | null>>
  | ReadonlyMap<string, string | null>;

/** `_coerce_labels`. Plain objects reorder integer-like keys (JS), so prefer a Map/array when labels can be "1","2". */
function coerceLabels(labels: unknown): LabelSpec[] {
  const make = (name: unknown, description: unknown): LabelSpec => {
    const cleanName = cleanPy(name, "label name");
    if (description !== null && description !== undefined) {
      cleanPy(description, `description of label ${pyRepr(cleanName)}`);
    }
    return { name: cleanName, description: (description as string | null | undefined) ?? null };
  };
  if (labels instanceof Map) {
    return [...labels.entries()].map(([n, d]) => make(n, d));
  }
  if (isMapping(labels)) {
    return Object.entries(labels).map(([n, d]) => make(n, d));
  }
  if (typeof labels === "string") {
    throw new Gliner2SchemaError("labels must be a list or a {label: description} mapping, not a str");
  }
  if (!Array.isArray(labels)) {
    throw new Gliner2MalformedSchemaError(`labels must be a list or a mapping, got ${pyReprValue(labels)}`);
  }
  return labels.map((item: unknown) => {
    if (typeof item === "string") return make(item, null);
    throw new Gliner2SchemaError(`invalid label entry ${pyReprValue(item)}`);
  });
}

/**
 * `TaskSpec` construction and `__post_init__` validation. `options` uses the Python/JSON snake_case keys.
 * Explicit `null` is treated like "not given", as the runner drops `None` values before calling the library.
 */
export function makeTaskSpec(name: string, o: Readonly<Record<string, unknown>>): TaskSpec {
  const minLabelsRaw = o["min_labels"] ?? 0;
  const maxLabelsRaw = o["max_labels"] ?? null;
  const ordered = o["ordered"] ?? false;
  const threshold = o["threshold"] ?? 0.5;
  const candidateThreshold = o["candidate_threshold"] ?? null;
  const activation = o["activation"] ?? "auto";
  const temperature = o["temperature"] ?? 1.0;
  const defaultLabel = o["default"] ?? null;
  const instruction = o["instruction"] ?? null;
  const rawExamples = o["examples"] ?? [];
  const q = pyRepr(name);

  // NB: in Python, _coerce_labels runs BEFORE TaskSpec.__post_init__ (argument evaluation order).
  const labels = coerceLabels(o["labels"]);
  cleanPy(name, "task name");
  if (labels.length === 0) throw new Gliner2SchemaError(`task ${q} must declare at least one label`);
  const names = labels.map((l) => l.name);
  if (new Set(names).size !== names.length) throw new Gliner2SchemaError(`task ${q} has duplicate label names`);
  if (!isInt(minLabelsRaw) || minLabelsRaw < 0) {
    throw new Gliner2SchemaError(`task ${q}: min_labels must be a non-negative int`);
  }
  let minLabels: number = minLabelsRaw;
  let maxLabels: number | null = null;
  if (maxLabelsRaw !== null) {
    if (!isInt(maxLabelsRaw) || maxLabelsRaw < 0) {
      throw new Gliner2SchemaError(`task ${q}: max_labels must be a non-negative int`);
    }
    if (maxLabelsRaw > labels.length) {
      throw new Gliner2SchemaError(
        `task ${q}: max_labels (${maxLabelsRaw}) exceeds the number of labels (${labels.length})`,
      );
    }
    maxLabels = maxLabelsRaw;
  }
  let def: string | null = null;
  if (defaultLabel !== null) {
    if (typeof defaultLabel !== "string" || !names.includes(defaultLabel)) {
      throw new Gliner2SchemaError(`task ${q}: default ${pyReprValue(defaultLabel)} is not one of its labels`);
    }
    def = defaultLabel;
    if (minLabels < 1) minLabels = 1;
  }
  if (maxLabels !== null && minLabels > maxLabels) {
    throw new Gliner2SchemaError(`task ${q}: min_labels (${minLabels}) exceeds max_labels (${maxLabels})`);
  }
  if (ordered === true && labels.length < 2) {
    throw new Gliner2SchemaError(`ordered task ${q} requires at least two labels`);
  }
  if (typeof ordered !== "boolean") {
    throw new Gliner2MalformedSchemaError(`task ${q}: ordered must be a bool, got ${pyReprValue(ordered)}`);
  }
  if (!isNum(threshold) || !(0 < threshold && threshold < 1)) {
    throw new Gliner2SchemaError(`task ${q}: threshold must be in (0, 1)`);
  }
  if (candidateThreshold !== null && !(isNum(candidateThreshold) && 0 <= candidateThreshold && candidateThreshold <= 1)) {
    throw new Gliner2SchemaError(`task ${q}: candidate_threshold must be in [0, 1]`);
  }
  if (typeof activation !== "string" || !ACTIVATIONS.includes(activation)) {
    throw new Gliner2SchemaError(`task ${q}: activation must be one of ('auto', 'sigmoid', 'softmax')`);
  }
  if (!isNum(temperature) || temperature <= 0) {
    throw new Gliner2SchemaError(`task ${q}: temperature must be positive`);
  }
  if (instruction !== null) cleanPy(instruction, `instruction of task ${q}`);
  if (!Array.isArray(rawExamples)) {
    throw new Gliner2MalformedSchemaError(`task ${q}: examples must be a list, got ${pyReprValue(rawExamples)}`);
  }
  const examples: Array<readonly [string, string]> = [];
  for (const pair of rawExamples as unknown[]) {
    if (!Array.isArray(pair)) {
      throw new Gliner2MalformedSchemaError(`task ${q}: each example must be a list, got ${pyReprValue(pair)}`);
    }
    if (pair.length !== 2) throw new Gliner2SchemaError(`task ${q}: each example must be a (text, label) pair`);
    const [text, label] = pair as [unknown, unknown];
    if (typeof label !== "string" || !names.includes(label)) {
      throw new Gliner2SchemaError(
        `task ${q}: example label ${pyReprValue(label)} is not one of its labels (the processor drops such examples silently)`,
      );
    }
    if (typeof text !== "string") {
      throw new Gliner2MalformedSchemaError(`task ${q}: example text must be a string, got ${pyReprValue(text)}`);
    }
    examples.push([text, label]);
  }
  return {
    name,
    labels,
    labelNames: names,
    minLabels,
    maxLabels,
    ordered,
    threshold,
    candidateThreshold: candidateThreshold as number | null,
    activation: activation as Activation,
    temperature,
    default: def,
    instruction: instruction as string | null,
    examples,
    isExclusive: minLabels === 1 && maxLabels === 1,
    effectiveMax: maxLabels === null ? labels.length : maxLabels,
  };
}

const TASK_KEYS = new Set([
  "labels",
  "min_labels",
  "max_labels",
  "ordered",
  "threshold",
  "candidate_threshold",
  "activation",
  "temperature",
  "default",
  "instruction",
  "examples",
]);

/** Task table input: plain object, `Map`, or ordered `[name, spec]` pairs (the only forms that keep integer-like names ordered). */
export type TasksInput =
  | Readonly<Record<string, unknown>>
  | ReadonlyMap<string, unknown>
  | ReadonlyArray<readonly [string, unknown]>;

export interface SchemaInput {
  readonly version?: unknown;
  readonly tasks?: TasksInput | undefined;
  readonly constraints?: readonly unknown[] | undefined;
}

function taskEntries(tasks: TasksInput | undefined): Array<[string, unknown]> {
  if (tasks === undefined || tasks === null) return [];
  if (tasks instanceof Map) return [...tasks.entries()];
  if (Array.isArray(tasks)) return tasks.map(([n, s]) => [n, s] as [string, unknown]);
  return Object.entries(tasks as Record<string, unknown>);
}

/**
 * `ClassificationSchema.from_dict`: validates every task and constraint in Python order.
 * Constraints may reference any declared task (forward references are fine).
 */
export function schemaFromDict(data: SchemaInput): ClassificationSchema {
  const tasks = new Map<string, TaskSpec>();
  for (const [name, specRaw] of taskEntries(data.tasks)) {
    if (!isMapping(specRaw)) {
      throw new Gliner2MalformedSchemaError(`task ${pyRepr(name)} must be a mapping, got ${pyReprValue(specRaw)}`);
    }
    if (!("labels" in specRaw)) throw new Gliner2MalformedSchemaError("'labels'", "KeyError");
    for (const k of Object.keys(specRaw)) {
      if (!TASK_KEYS.has(k)) {
        throw new Gliner2MalformedSchemaError(
          `ClassificationSchema.task() got an unexpected keyword argument '${k}'`,
        );
      }
    }
    if (tasks.has(name)) throw new Gliner2SchemaError(`task ${pyRepr(name)} is already defined`);
    tasks.set(name, makeTaskSpec(name, specRaw));
  }
  const constraints = [];
  for (const raw of data.constraints ?? []) {
    const c = constraintFromDict(raw);
    const missing = [...c.references()].filter((t) => !tasks.has(t));
    if (missing.length > 0) {
      const nm = sortedStrs(missing)[0] as string;
      throw new Gliner2SchemaError(
        `constraint references undeclared task ${pyRepr(nm)}; declare task ${pyRepr(nm)} before constraining it`,
      );
    }
    constraints.push(c);
  }
  return { tasks, constraints };
}
