import assert from "node:assert/strict";
import { test } from "node:test";

import { OrderedObject } from "../../src/gliner2/index.js";
import { NUMERIC_TOLERANCE, compareOutputs, parseOrdered } from "./parity-compare.js";

const output = (confidence: number, labels: string[] = ["a", "b"]): string =>
  `{"t":{"value":"a","confidence":${confidence},"probabilities":{${labels
    .map((label, i) => `"${label}":${0.5 + i * 0.1}`)
    .join(",")}}},"_meta":{"feasible":true,"decoder":"independent","exact":true,"objective":1.0,"violations":[]}}`;

test("parseOrdered preserves integer-like key order and nested structure", () => {
  const parsed = parseOrdered('{"10":1,"2":[{"b":null,"a":"x\\u00e9\\"q"}],"1":-1.5e-3}');
  assert.ok(parsed instanceof OrderedObject);
  assert.deepEqual(parsed.keys(), ["10", "2", "1"]);
  assert.equal(parsed.serialize(), '{"10":1,"2":[{"b":null,"a":"xé\\"q"}],"1":-0.0015}');
  assert.throws(() => parseOrdered('{"a":1,}'), SyntaxError);
  assert.throws(() => parseOrdered('{"a":1} x'), SyntaxError);
});

test("compareOutputs accepts differences below the tolerance and records raw diffs", () => {
  const result = compareOutputs(parseOrdered(output(0.80002)), parseOrdered(output(0.8)));
  assert.deepEqual(result.mismatches, []);
  assert.equal(result.numeric.compared, 4);
  assert.equal(result.numeric.exactZero, 3);
  assert.equal(result.numeric.syntheticZero, 1);
  assert.ok(result.numeric.maxAbsDiff > 1e-6 && result.numeric.maxAbsDiff < NUMERIC_TOLERANCE);
});

test("compareOutputs rejects numeric drift over the tolerance, key order and value changes", () => {
  const over = compareOutputs(parseOrdered(output(0.8002)), parseOrdered(output(0.8)));
  assert.equal(over.numeric.overTolerance, 1);
  assert.equal(over.mismatches.length, 1);

  const reordered = compareOutputs(parseOrdered(output(0.8, ["b", "a"])), parseOrdered(output(0.8, ["a", "b"])));
  assert.match(reordered.mismatches.join("\n"), /key sequence/);

  const relabelled = compareOutputs(
    parseOrdered(output(0.8).replace('"value":"a"', '"value":"b"')),
    parseOrdered(output(0.8)),
  );
  assert.match(relabelled.mismatches.join("\n"), /value/);

  const meta = compareOutputs(
    parseOrdered(output(0.8).replace('"exact":true', '"exact":false')),
    parseOrdered(output(0.8)),
  );
  assert.match(meta.mismatches.join("\n"), /exact/);

  const nulled = compareOutputs(parseOrdered(output(0.8).replace('"confidence":0.8', '"confidence":null')), parseOrdered(output(0.8)));
  assert.equal(nulled.mismatches.length, 1);
});
