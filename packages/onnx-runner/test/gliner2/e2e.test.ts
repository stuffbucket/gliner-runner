import assert from "node:assert/strict";
import { describe, it } from "node:test";
import { compileSchema, decode, schemaFromDict } from "../../src/gliner2/index.js";
import { compareOrdered, loadFixture } from "./support.js";

interface Variant {
  schema: { tasks: Record<string, Record<string, unknown>>; constraints: Array<Record<string, unknown>> };
  input_token_counts: number[];
  logits: Array<Record<string, number[]>>;
  label_orders: Record<string, string[]>;
  results: Record<string, unknown[]>;
}
interface E2E {
  texts: string[];
  variants: Record<string, Variant>;
}
const golden = loadFixture<E2E>("e2e-small-model.json");

describe("gliner2 end-to-end (real small-model logits -> to_dict)", () => {
  let total = 0;
  for (const [name, variant] of Object.entries(golden.variants)) {
    // The golden schema is the GLiNER2 `from_dict` form that the runner builds (labels list or {label: description}).
    const compiled = compileSchema(
      schemaFromDict({ tasks: Object.entries(variant.schema.tasks), constraints: variant.schema.constraints }),
    );
    for (const [key, expectedRows] of Object.entries(variant.results)) {
      const threshold = Number(/threshold=([\d.]+)/.exec(key)?.[1]);
      const includeConfidence = /include_confidence=True/.test(key);
      total += golden.texts.length;
      it(`${name} / ${key}`, () => {
        assert.equal(expectedRows.length, golden.texts.length);
        golden.texts.forEach((_text, i) => {
          const rowLogits = compiled.taskOrder.flatMap((t) => (variant.logits[i] as Record<string, number[]>)[t] as number[]);
          // The ONNX runner hands over float32 logits.
          const result = decode(compiled, Float32Array.from(rowLogits), { candidateThreshold: threshold, includeConfidence });
          const diffs: string[] = [];
          compareOrdered(result.output, expectedRows[i], `row${i}`, diffs);
          assert.deepEqual(diffs, []);
          // Task order of the compiled schema equals the label order recorded from the model.
          assert.deepEqual(compiled.taskOrder.map((t) => compiled.task(t).labelNames), compiled.taskOrder.map((t) => variant.label_orders[t]));
        });
      });
    }
  }

  it("covers the 40 end-to-end cases of the golden", () => {
    assert.equal(total, 40);
  });
});
