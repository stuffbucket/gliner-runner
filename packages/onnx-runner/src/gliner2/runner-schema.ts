/**
 * Conversion from the runner request schema (`ClassificationSchema` of src/gliner_runner/contracts.py) into the
 * GLiNER2 `ClassificationSchema.from_dict` input, replicating `_oracle_task` / `_infer_sync`
 * (src/gliner_runner/backends/pytorch.py:94-160).
 */
import { compileSchema } from "./compile.js";
import { schemaFromDict } from "./schema.js";
import type { SchemaInput } from "./schema.js";
import type { CompiledSchema } from "./types.js";

/**
 * One runner classification task as produced by the request contract. Omitted fields take the pydantic defaults
 * (`min_labels=1`, `max_labels=1`, `ordered=false`, `threshold=0.5`, `activation="auto"`, `temperature=1`);
 * an explicit `null` is dropped (`exclude_none`), so `max_labels: null` means "unbounded".
 */
export interface RunnerClassificationTask {
  readonly labels: readonly string[];
  readonly label_descriptions?: Readonly<Record<string, string | null>> | ReadonlyMap<string, string | null> | null | undefined;
  readonly min_labels?: number | undefined;
  readonly max_labels?: number | null | undefined;
  readonly ordered?: boolean | undefined;
  readonly threshold?: number | undefined;
  readonly activation?: "auto" | "sigmoid" | "softmax" | undefined;
  readonly temperature?: number | undefined;
  readonly default?: string | null | undefined;
  readonly instruction?: string | null | undefined;
}

/**
 * The runner `ClassificationSchema`. `tasks` may be a plain object, a `Map`, or ordered `[name, task]` pairs. Task
 * order drives prompt order, logit order and output key order; a plain object reorders integer-like task names
 * (JS semantics), so an HTTP layer that must preserve such names should hand over a Map/pairs.
 */
export interface RunnerClassificationSchema {
  readonly kind?: "classification" | undefined;
  readonly tasks:
    | Readonly<Record<string, RunnerClassificationTask>>
    | ReadonlyMap<string, RunnerClassificationTask>
    | ReadonlyArray<readonly [string, RunnerClassificationTask]>;
  readonly constraints?: ReadonlyArray<Readonly<Record<string, unknown>>> | undefined;
}

/** `_oracle_task`: `task.model_dump(mode="json", exclude_none=True)` with described labels turned into a label map. */
export function oracleTask(task: RunnerClassificationTask): Record<string, unknown> {
  const payload: Record<string, unknown> = {
    labels: [...task.labels],
    min_labels: task.min_labels ?? 1,
    ordered: task.ordered ?? false,
    threshold: task.threshold ?? 0.5,
    activation: task.activation ?? "auto",
    temperature: task.temperature ?? 1.0,
  };
  // max_labels defaults to 1 when omitted; an explicit null is dropped (unbounded).
  if (task.max_labels !== null) payload["max_labels"] = task.max_labels ?? 1;
  if (task.default !== null && task.default !== undefined) payload["default"] = task.default;
  if (task.instruction !== null && task.instruction !== undefined) payload["instruction"] = task.instruction;
  const descriptions = task.label_descriptions;
  if (descriptions !== null && descriptions !== undefined) {
    const lookup = (label: string): string | null =>
      (descriptions instanceof Map
        ? (descriptions as ReadonlyMap<string, string | null>).get(label)
        : (Object.hasOwn(descriptions, label)
            ? (descriptions as Readonly<Record<string, string | null>>)[label]
            : undefined)) ?? null;
    payload["labels"] = new Map(task.labels.map((label) => [label, lookup(label)] as const));
  }
  return payload;
}

/** Runner schema -> input of {@link schemaFromDict}. */
export function runnerSchemaToGliner2(schema: RunnerClassificationSchema): SchemaInput {
  const entries: Array<readonly [string, RunnerClassificationTask]> =
    schema.tasks instanceof Map
      ? [...schema.tasks.entries()]
      : Array.isArray(schema.tasks)
        ? [...(schema.tasks as ReadonlyArray<readonly [string, RunnerClassificationTask]>)]
        : Object.entries(schema.tasks as Readonly<Record<string, RunnerClassificationTask>>);
  return {
    version: 3,
    tasks: new Map(entries.map(([name, task]) => [name, oracleTask(task)] as const)),
    constraints: [...(schema.constraints ?? [])],
  };
}

/**
 * Parses and compiles a runner schema. Throws `Gliner2SchemaError` (incl. `Gliner2MalformedSchemaError`) for anything
 * the PyTorch runner reports as InvalidSchemaError (and for the malformed-constraint cases it leaves unmapped).
 */
export function compileRunnerSchema(schema: RunnerClassificationSchema): CompiledSchema {
  return compileSchema(schemaFromDict(runnerSchemaToGliner2(schema)));
}
