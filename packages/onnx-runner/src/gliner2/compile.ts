/** `compile_schema` (gliner2/classification/compiler.py:187-216). */
import { DictAssignment, defaultRule } from "./constraints.js";
import type { Constraint } from "./constraints.js";
import { Gliner2SchemaError } from "./errors.js";
import { pyRepr } from "./pyparity.js";
import { RESERVED_TOKENS } from "./schema.js";
import type { ClassificationSchema, CompiledSchema, ModelTaskEntry, TaskSpec } from "./types.js";

/** Compiles a parsed schema: static checks, prompt-entry emission and default lowering. */
export function compileSchema(schema: ClassificationSchema): CompiledSchema {
  const specs = [...schema.tasks.values()];
  if (specs.length === 0) throw new Gliner2SchemaError("cannot compile a schema with no tasks");
  const taskOrder = specs.map((s) => s.name);
  for (const c of schema.constraints) c.checkSchema(schema);

  checkPrefixCollisions(taskOrder);
  if (schema.constraints.length > 0) staticFeasibility(schema, specs);

  const classifications = specs.map(emitModelEntry);
  assertModelSchema(classifications);

  // Lower defaults: Iff(IsDefault(t), Not(AnyOtherSelected(t))), appended in task order.
  const constraints: Constraint[] = [...schema.constraints];
  for (const spec of specs) if (spec.default !== null) constraints.push(defaultRule(spec.name));

  const offsets: number[] = [];
  let total = 0;
  for (const spec of specs) {
    offsets.push(total);
    total += spec.labelNames.length;
  }
  const tasks = new Map(specs.map((s) => [s.name, s] as const));
  return {
    classifications,
    specs,
    constraints,
    taskOrder,
    tasks,
    labelCount: total,
    labelOffsets: offsets,
    task(name: string): TaskSpec {
      const t = tasks.get(name);
      if (t === undefined) throw new Gliner2SchemaError(`unknown task ${pyRepr(name)}`);
      return t;
    },
  };
}

function checkPrefixCollisions(taskOrder: readonly string[]): void {
  for (const x of taskOrder) {
    for (const y of taskOrder) {
      if (x === y) continue;
      if (y.startsWith(x) && y.length > x.length && (y[x.length] === ":" || y[x.length] === " ")) {
        throw new Gliner2SchemaError(
          `task name ${pyRepr(x)} is a boundary-prefix of ${pyRepr(y)}; the prompt resolver could confuse them. Rename one.`,
        );
      }
    }
  }
}

function staticFeasibility(schema: ClassificationSchema, specs: readonly TaskSpec[]): void {
  const undetermined = new DictAssignment(schema, {}, []);
  for (const c of schema.constraints) {
    if (c.evaluate(undetermined) === false) {
      throw new Gliner2SchemaError("constraint set is unsatisfiable on the declared label sets");
    }
  }
  for (const spec of specs) {
    if (!spec.isExclusive) continue;
    const touching = schema.constraints.filter((c) => c.references().has(spec.name));
    if (touching.length === 0) continue;
    let reachable = false;
    for (const label of spec.labelNames) {
      const a = new DictAssignment(schema, { [spec.name]: [label] }, [spec.name]);
      if (touching.every((c) => c.evaluate(a) !== false)) {
        reachable = true;
        break;
      }
    }
    if (!reachable) {
      throw new Gliner2SchemaError(
        `no label of exclusive task ${pyRepr(spec.name)} satisfies its constraints; the task is pinned to the empty set`,
      );
    }
  }
}

function emitModelEntry(spec: TaskSpec): ModelTaskEntry {
  const descriptions = spec.labels
    .filter((l) => l.description !== null && l.description !== "")
    .map((l) => [l.name, l.description as string] as const);
  return {
    task: spec.name,
    labels: [...spec.labelNames],
    multiLabel: !spec.isExclusive,
    clsThreshold: spec.threshold,
    classAct: spec.activation,
    ...(spec.instruction ? { prompt: spec.instruction } : {}),
    ...(descriptions.length > 0 ? { labelDescriptions: descriptions } : {}),
    ...(spec.examples.length > 0 ? { examples: spec.examples.map((p) => [p[0], p[1]] as const) } : {}),
  };
}

/** Compile-time reserved-token scan over everything emitted (example text is first validated here). */
function assertModelSchema(classifications: readonly ModelTaskEntry[]): void {
  for (const e of classifications) {
    const strs: string[] = [e.task, ...e.labels];
    if (e.prompt !== undefined) strs.push(e.prompt);
    for (const [k, v] of e.labelDescriptions ?? []) strs.push(k, v);
    for (const p of e.examples ?? []) strs.push(p[0], p[1]);
    for (const s of strs) {
      if (RESERVED_TOKENS.some((t) => s.includes(t))) {
        throw new Gliner2SchemaError(
          `classification task ${pyRepr(e.task)} emitted a reserved marker token in ${pyRepr(s)}; this would corrupt logit-to-label alignment`,
        );
      }
    }
  }
}
