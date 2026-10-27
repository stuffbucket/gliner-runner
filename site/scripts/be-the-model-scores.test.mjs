// Checks the shape of site/src/data/be-the-model.json, which the "be the model" page reads.
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { test } from "node:test";

import { LABELS, bands } from "./be-the-model.mjs";

const data = JSON.parse(readFileSync(new URL("../src/data/be-the-model.json", import.meta.url), "utf8"));
const names = data.labels.map((label) => label.name);

test("the file has the agreed shape", () => {
  assert.deepEqual(Object.keys(data), ["about", "model", "mode", "labels", "examples"]);
  assert.equal(data.model, "decide-340m");
  assert.equal(data.mode, "single");
  assert.deepEqual(data.labels, Object.entries(LABELS).map(([name, description]) => ({ name, description })));
  assert.ok(data.examples.length >= 100);
  const ids = new Set();
  for (const example of data.examples) {
    assert.deepEqual(Object.keys(example), ["id", "text", "ask", "age", "scores", "multi"]);
    assert.match(example.id, /^k\d{3}$/);
    assert.ok(!ids.has(example.id), `${example.id} is repeated`);
    ids.add(example.id);
    assert.ok(example.text.length > 0 && example.ask.length > 0);
    assert.ok(Number.isInteger(example.age) && example.age >= 5 && example.age <= 16);
    assert.deepEqual(Object.keys(example.scores), names);
    assert.deepEqual(Object.keys(example.multi), names);
  }
});

test("single-label scores sum to about 1, and every score is a probability with 3 decimals", () => {
  for (const example of data.examples) {
    const sum = Object.values(example.scores).reduce((a, b) => a + b, 0);
    assert.ok(Math.abs(sum - 1) < 0.01, `${example.id} sums to ${sum}`);
    for (const p of [...Object.values(example.scores), ...Object.values(example.multi)]) {
      assert.ok(p >= 0 && p <= 1 && Math.round(p * 1000) === p * 1000, `${example.id} has ${p}`);
    }
  }
});

test("the texts are only what the kid says, with no stage directions", () => {
  for (const example of data.examples) assert.doesNotMatch(example.text, /[()[\]*]|—|–/, example.id);
});

test("every label's slider walks through many examples, from near 0 to its top", () => {
  for (const [name, counts] of Object.entries(bands(data.examples))) {
    assert.ok(counts[0] > 0, `${name} has no example near 0`);
    assert.ok(counts.filter((n) => n > 0).length >= 6, `${name} covers ${counts}`);
    const top = counts.findLastIndex((n) => n > 0);
    assert.ok(counts.slice(0, top + 1).every((n) => n > 0), `${name} has an empty band below its top: ${counts}`);
  }
});
