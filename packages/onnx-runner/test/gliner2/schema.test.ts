import assert from "node:assert/strict";
import { describe, it } from "node:test";
import {
  Gliner2MalformedSchemaError,
  Gliner2SchemaError,
  compileRunnerSchema,
  compileSchema,
  isGliner2SchemaError,
  oracleTask,
  runnerSchemaToGliner2,
  schemaFromDict,
} from "../../src/gliner2/index.js";
import type { SchemaInput } from "../../src/gliner2/index.js";
import { loadFixture } from "./support.js";

interface ErrorCase {
  data: SchemaInput;
  result: { ok: boolean; n_constraints?: number; error_class?: string; error?: string };
}
const cases = loadFixture<ErrorCase[]>("schema-errors.json");

describe("gliner2 schema goldens", () => {
  for (const [index, c] of cases.entries()) {
    const label = c.result.ok ? "valid" : `${c.result.error_class}: ${c.result.error}`;
    it(`schema case ${index}: ${label}`.slice(0, 160), () => {
      if (c.result.ok) {
        const compiled = compileSchema(schemaFromDict(c.data));
        assert.equal(compiled.constraints.length, c.result.n_constraints);
        return;
      }
      assert.throws(
        () => compileSchema(schemaFromDict(c.data)),
        (error: unknown) => {
          assert.ok(isGliner2SchemaError(error), "every schema failure is a Gliner2SchemaError");
          assert.equal(error.pythonClass, c.result.error_class);
          assert.equal(error.message, c.result.error);
          // SchemaError -> Gliner2SchemaError; TypeError/KeyError -> Gliner2MalformedSchemaError.
          assert.equal(error instanceof Gliner2MalformedSchemaError, c.result.error_class !== "SchemaError");
          return true;
        },
      );
    });
  }

  it("reports additional malformed constraint shapes like Python", () => {
    const tasks = { t: { labels: ["a", "b", "c"] } };
    const thrown = (constraints: unknown[]): Error => {
      try {
        schemaFromDict({ tasks, constraints });
      } catch (error) {
        return error as Error;
      }
      throw new Error("expected an error");
    };
    assert.equal(
      thrown([{ type: "Implies", cond: { type: "LabelRef", task: "t", label: "a" } }]).message,
      "Implies.__init__() missing 1 required positional argument: 'then'",
    );
    assert.equal(thrown([{ type: "Implies" }]).message, "Implies.__init__() missing 2 required positional arguments: 'cond' and 'then'");
    assert.equal(thrown([{}]).message, "unknown constraint type None");
    assert.ok(thrown([{ type: "LabelRef", task: 5, label: "a" }]) instanceof Gliner2MalformedSchemaError);
    assert.ok(thrown([{ type: "Not", child: "x" }]) instanceof Gliner2MalformedSchemaError);
  });

  it("preserves task and label order for integer-like names when given pairs/maps", () => {
    const schema = schemaFromDict({
      tasks: [
        ["10", { labels: new Map([["2", "two"], ["1", null]]) }],
        ["2", { labels: ["b", "a"] }],
      ],
    });
    assert.deepEqual([...schema.tasks.keys()], ["10", "2"]);
    const compiled = compileSchema(schema);
    assert.deepEqual(compiled.task("10").labelNames, ["2", "1"]);
    assert.deepEqual(compiled.labelOffsets, [0, 2]);
    assert.equal(compiled.labelCount, 4);
  });
});

describe("runner schema conversion (_oracle_task)", () => {
  it("applies pydantic defaults and drops None", () => {
    assert.deepEqual(oracleTask({ labels: ["a", "b"] }), {
      labels: ["a", "b"],
      min_labels: 1,
      max_labels: 1,
      ordered: false,
      threshold: 0.5,
      activation: "auto",
      temperature: 1,
    });
    const unbounded = oracleTask({ labels: ["a"], max_labels: null, default: "a", instruction: null });
    assert.equal("max_labels" in unbounded, false);
    assert.equal(unbounded["default"], "a");
    assert.equal("instruction" in unbounded, false);
  });

  it("turns label_descriptions into a label map in label order with null for undescribed labels", () => {
    const task = oracleTask({ labels: ["b", "a", "c"], label_descriptions: { a: "alpha", c: null } });
    assert.ok(task["labels"] instanceof Map);
    assert.deepEqual([...(task["labels"] as Map<string, string | null>)], [["b", null], ["a", "alpha"], ["c", null]]);
    const empty = oracleTask({ labels: ["x"], label_descriptions: {} });
    assert.deepEqual([...(empty["labels"] as Map<string, string | null>)], [["x", null]]);
  });

  it("compiles a runner schema, carrying constraints through", () => {
    const compiled = compileRunnerSchema({
      kind: "classification",
      tasks: {
        intent: { labels: ["billing", "support"], label_descriptions: { billing: "money" }, instruction: "Pick one" },
        tags: { labels: ["refund", "bug"], min_labels: 0, max_labels: null, default: "bug" },
      },
      constraints: [{ type: "Implies", cond: { type: "LabelRef", task: "tags", label: "refund" }, then: { type: "LabelRef", task: "intent", label: "billing" } }],
    });
    assert.deepEqual(compiled.taskOrder, ["intent", "tags"]);
    assert.equal(compiled.task("intent").isExclusive, true);
    assert.equal(compiled.task("tags").minLabels, 1); // default forces min_labels >= 1
    assert.equal(compiled.constraints.length, 2); // declared + lowered default rule
    assert.deepEqual(compiled.classifications[0]?.labelDescriptions, [["billing", "money"]]);
    assert.equal(compiled.classifications[0]?.prompt, "Pick one");
    assert.deepEqual(runnerSchemaToGliner2({ tasks: new Map([["z", { labels: ["a"] }]]) }).constraints, []);
  });

  it("maps invalid content to Gliner2SchemaError", () => {
    assert.throws(
      () => compileRunnerSchema({ tasks: { "bad(name)": { labels: ["a", "b"] } } }),
      (error: unknown) => isGliner2SchemaError(error) && error instanceof Gliner2SchemaError && /may not contain '\('/.test(error.message),
    );
  });
});
