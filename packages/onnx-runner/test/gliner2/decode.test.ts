import assert from "node:assert/strict";
import { describe, it } from "node:test";
import {
  Gliner2InfeasibleError,
  Gliner2SchemaError,
  compileSchema,
  decode,
  decodeScores,
  schemaFromDict,
} from "../../src/gliner2/index.js";
import type { Scores } from "../../src/gliner2/index.js";
import { compareOrdered, loadFixture } from "./support.js";

interface DecodeCase {
  kind: "decode" | "schema_error";
  tasks: Record<string, Record<string, unknown>>;
  constraints: Array<Record<string, unknown>>;
  logits?: Record<string, Record<string, number>>;
  config?: { candidate_threshold: number; include_confidence: boolean; exact_node_budget?: number };
  expected?: {
    to_dict_conf: Record<string, unknown>;
    to_dict_plain: unknown;
    to_dict_default: unknown;
    requested_include_confidence: boolean;
  };
  error_class?: string;
  error?: string;
}

const cases = loadFixture<DecodeCase[]>("decode.json");
const decodeCases = cases.filter((c) => c.kind === "decode");
const schemaCases = cases.filter((c) => c.kind !== "decode");

/** The goldens drop `None`-valued task fields exactly like the runner does (`exclude_none`). */
function schemaOf(c: DecodeCase) {
  const tasks = Object.entries(c.tasks).map(([name, spec]) => [
    name,
    Object.fromEntries(Object.entries(spec).filter(([, v]) => v !== null)),
  ] as const);
  return schemaFromDict({ tasks, constraints: c.constraints });
}

describe("gliner2 decode goldens", () => {
  it("has coverage of every decoder path in the fixture subset", () => {
    const seen = new Set<string>();
    for (const c of decodeCases) {
      const meta = (c.expected?.to_dict_conf as { _meta: { decoder: string; feasible: boolean; exact: boolean } })._meta;
      seen.add(`${meta.decoder}/feasible=${meta.feasible}/exact=${meta.exact}`);
    }
    for (const key of [
      "independent/feasible=true/exact=true",
      "exact/feasible=true/exact=true",
      "beam/feasible=true/exact=false",
      "min_violations/feasible=false/exact=true",
      "min_violations/feasible=false/exact=false",
    ]) {
      assert.ok(seen.has(key), `missing golden path ${key}`);
    }
  });

  for (const [index, c] of decodeCases.entries()) {
    const meta = (c.expected?.to_dict_conf as { _meta: { decoder: string } })._meta;
    it(`decode case ${index} (${meta.decoder}, ${Object.keys(c.tasks).length} tasks)`, () => {
      const compiled = compileSchema(schemaOf(c));
      const logits = c.logits as Record<string, Record<string, number>>;
      // Declaration order of tasks and labels, independent of JS object key reordering.
      const scores: Scores = new Map(
        compiled.taskOrder.map((t) => [
          t,
          new Map(compiled.task(t).labelNames.map((l) => [l, (logits[t] as Record<string, number>)[l] as number])),
        ]),
      );
      const cfg = c.config as NonNullable<DecodeCase["config"]>;
      const result = decodeScores(compiled, scores, {
        candidateThreshold: cfg.candidate_threshold,
        includeConfidence: cfg.include_confidence,
        ...(cfg.exact_node_budget !== undefined ? { exactNodeBudget: cfg.exact_node_budget } : {}),
      });
      const expected = c.expected as NonNullable<DecodeCase["expected"]>;
      const diffs: string[] = [];
      compareOrdered(result.toDict(true), expected.to_dict_conf, "conf", diffs);
      compareOrdered(result.toDict(false), expected.to_dict_plain, "plain", diffs);
      compareOrdered(result.toDict(), expected.to_dict_default, "default", diffs);
      assert.deepEqual(diffs, []);
      // The runner contract: `output` == to_dict(include_confidence=<request flag>), serializable with Python key order.
      assert.equal(result.serialize(), JSON.stringify(result.output.toJSON()));
    });
  }

  for (const [index, c] of schemaCases.entries()) {
    it(`compile-time schema error ${index}: ${c.error}`, () => {
      assert.throws(
        () => compileSchema(schemaOf(c)),
        (error: unknown) =>
          error instanceof Gliner2SchemaError && error.pythonClass === c.error_class && error.message === c.error,
      );
    });
  }

  it("decode() accepts Float32Array rows and rejects wrongly sized rows", () => {
    const compiled = compileSchema(schemaFromDict({ tasks: { t: { labels: ["a", "b", "c"], min_labels: 1, max_labels: 1 } } }));
    const result = decode(compiled, new Float32Array([0.5, 2.5, -1]), { candidateThreshold: 0.5, includeConfidence: false });
    assert.equal(result.serialize(), '{"t":"b"}');
    assert.throws(() => decode(compiled, [1, 2], { candidateThreshold: 0.5, includeConfidence: false }), /logit\/label count mismatch/);
    assert.throws(() => decode(compiled, [1, 2, Number.NaN], { candidateThreshold: 0.5, includeConfidence: false }), /finite/);
  });

  it("onInfeasible=raise throws Gliner2InfeasibleError with Python violation strings", () => {
    const compiled = compileSchema(
      schemaFromDict({
        tasks: { t: { labels: ["a", "b"], min_labels: 1, max_labels: 2 }, u: { labels: ["x", "y"], min_labels: 1, max_labels: 2 } },
        constraints: [
          { type: "Implies", cond: { type: "LabelRef", task: "t", label: "a" }, then: { type: "LabelRef", task: "u", label: "x" } },
          { type: "Cardinality", task: "u", minimum: 0, maximum: 0 },
          { type: "LabelRef", task: "t", label: "a" },
        ],
      }),
    );
    const config = { candidateThreshold: 0.5, includeConfidence: true } as const;
    assert.throws(
      () => decode(compiled, [3, -3, 3, -3], { ...config, onInfeasible: "raise" }),
      (error: unknown) => error instanceof Gliner2InfeasibleError && error.violations.length > 0 && error.violations.every((v) => v.startsWith("violated ")),
    );
    const relaxed = decode(compiled, [3, -3, 3, -3], config);
    assert.equal(relaxed.feasible, false);
    assert.equal(relaxed.decoder, "min_violations");
  });

  it("explicit decoder choices and config validation", () => {
    const compiled = compileSchema(
      schemaFromDict({
        tasks: { a: { labels: ["x", "y", "z"], min_labels: 0, max_labels: 2 }, b: { labels: ["p", "q"], min_labels: 1, max_labels: 1 } },
        constraints: [{ type: "Implies", cond: { type: "LabelRef", task: "a", label: "x" }, then: { type: "LabelRef", task: "b", label: "q" } }],
      }),
    );
    const logits = [2.0, 1.0, -3.0, 0.6, 0.2];
    const base = { candidateThreshold: 0.1, includeConfidence: true } as const;
    for (const decoder of ["auto", "exact", "beam"] as const) {
      assert.equal(decode(compiled, logits, { ...base, decoder }).feasible, true, decoder);
    }
    assert.equal(decode(compiled, logits, { ...base, decoder: "beam" }).decoder, "beam");
    assert.equal(decode(compiled, logits, { ...base, decoder: "exact" }).decoder, "exact");
    // Independent decoding ignores cross-task coupling: the unconstrained argmax violates the implication, so the
    // ladder falls through to the min-violations decoder; with the constraint removed it stays independent.
    assert.equal(decode(compiled, logits, { ...base, decoder: "independent" }).decoder, "min_violations");
    const uncoupled = compileSchema(schemaFromDict({ tasks: { a: { labels: ["x", "y", "z"], max_labels: 2, min_labels: 0 } } }));
    assert.equal(decode(uncoupled, [2.0, 1.0, -3.0], { ...base, decoder: "independent" }).decoder, "independent");
    // A tiny node budget forces the exact search to fall back to the beam decoder.
    assert.equal(decode(compiled, logits, { ...base, decoder: "exact", exactNodeBudget: 1 }).decoder, "beam");
    assert.throws(() => decode(compiled, logits, { ...base, candidateThreshold: 1.5 }), /candidate_threshold/);
    assert.throws(() => decode(compiled, logits, { ...base, beamSize: 0 }), /beam_size/);
    assert.throws(() => decode(compiled, logits, { ...base, exactNodeBudget: 0 }), /exact_node_budget/);
    assert.throws(() => decode(compiled, logits, { ...base, maxCandidatesPerTask: 0 }), /max_candidates_per_task/);
    assert.throws(() => decode(compiled, logits, { ...base, decoder: "nope" as "auto" }), /decoder must be/);
    assert.throws(() => decode(compiled, logits, { ...base, onInfeasible: "nope" as "relax" }), /on_infeasible/);
  });
});
