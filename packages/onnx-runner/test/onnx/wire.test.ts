import assert from "node:assert/strict";
import { test } from "node:test";

import { InMemoryArtifactCatalog, createServer, defaultSettings } from "../../src/index.js";
import { OrderedObject, parseOrderedJson } from "../../src/gliner2/index.js";
import { recordTaskOrder, taskOrderOf } from "../../src/task-order.js";
import { stringifyWire } from "../../src/wire.js";
import { FakeBackend, PINNED_340M, wireRequest } from "../helpers.js";
import { InferenceRequestSchema, type BackendFactory, type BackendResult } from "../../src/index.js";

test("stringifyWire keeps ordered-object key order and otherwise matches JSON.stringify", () => {
  const ordered = new OrderedObject([
    ["10", 1],
    ["2", new OrderedObject([["b", [1, 2]], ["1", null]])],
    ["a", "x"],
  ]);
  assert.equal(stringifyWire({ output: ordered, n: 1 }), '{"output":{"10":1,"2":{"b":[1,2],"1":null},"a":"x"},"n":1}');
  assert.equal(stringifyWire({ a: undefined, b: [undefined, () => 1], c: Number.NaN }), '{"b":[null,null],"c":null}');
  assert.equal(stringifyWire({ toJSON: () => ({ z: 1 }) }), '{"z":1}');
  assert.equal(stringifyWire("é\n"), JSON.stringify("é\n"));
  for (const sample of [null, 0, true, [], {}, { a: [{ b: { c: [] } }] }]) {
    assert.equal(stringifyWire(sample), JSON.stringify(sample));
  }
});

test("the HTTP layer emits OrderedObject backend output with Python key order", async () => {
  const output = new OrderedObject([
    ["10", new OrderedObject([["value", "a"], ["probabilities", new OrderedObject([["3", 0.25], ["1", 0.75]])]])],
    ["2", new OrderedObject([["value", "b"]])],
    ["_meta", new OrderedObject([["feasible", true]])],
  ]);
  const backend = new FakeBackend();
  backend.inferBatch = async (requests): Promise<BackendResult[]> =>
    requests.map(() => ({ output, usage: { inputTokens: 3, outputTokens: 0 } }));
  const factory: BackendFactory = { name: "onnx", capabilities: () => backend.capabilities, create: () => backend };
  const catalog = new InMemoryArtifactCatalog({ installed: { "decide-340m": { [PINNED_340M]: "/tmp/x" } } });
  const settings = { ...defaultSettings(), host: "127.0.0.1", port: 0, batchWindowMs: 0 };
  const server = createServer({ catalog, backends: [factory], settings });
  const address = await server.listen();
  try {
    const response = await fetch(`http://127.0.0.1:${address.port}/v1/infer`, {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify(wireRequest()),
    });
    const text = await response.text();
    assert.equal(response.status, 200, text);
    const root = parseOrderedJson(text) as OrderedObject;
    assert.equal(
      (root.get("output") as OrderedObject).serialize(),
      '{"10":{"value":"a","probabilities":{"3":0.25,"1":0.75}},"2":{"value":"b"},"_meta":{"feasible":true}}',
    );
    const batch = await fetch(`http://127.0.0.1:${address.port}/v1/batch`, {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({ requests: [wireRequest(), wireRequest()] }),
    });
    assert.equal(((await batch.text()).match(/"10":\{"value":"a","probabilities":\{"3":0\.25,"1":0\.75\}\}/g) ?? []).length, 2);
  } finally {
    await server.close();
  }
});

test("recordTaskOrder remembers declared task order only for integer-like names", () => {
  const raw =
    '{"model":"decide-340m","backend":"onnx","precision":"fp32","operation":"classify","text":"t",' +
    '"schema":{"kind":"classification","tasks":{"10":{"labels":["a"]},"2":{"labels":["b"]},"x":{"labels":["c"]}}}}';
  const request = InferenceRequestSchema.parse(JSON.parse(raw));
  const tasks = (request.schema as { tasks: object }).tasks;
  assert.deepEqual(Object.keys(tasks), ["2", "10", "x"]);
  recordTaskOrder(raw, [request], false);
  assert.deepEqual(taskOrderOf(tasks), ["10", "2", "x"]);

  const plain = InferenceRequestSchema.parse(wireRequest());
  recordTaskOrder(JSON.stringify(wireRequest()), [plain], false);
  assert.deepEqual(taskOrderOf((plain.schema as { tasks: object }).tasks), ["priority"]);

  const batched = `{"requests":[${raw},${raw}]}`;
  const items = [InferenceRequestSchema.parse(JSON.parse(raw)), InferenceRequestSchema.parse(JSON.parse(raw))];
  recordTaskOrder(batched, items, true);
  for (const item of items) {
    assert.deepEqual(taskOrderOf((item.schema as { tasks: object }).tasks), ["10", "2", "x"]);
  }
  recordTaskOrder("{not json 1:", [request], false);
});

const ORDER_PREFIX =
  '{"model":"decide-340m","backend":"onnx","precision":"fp32","operation":"classify","text":"t",' +
  '"schema":{"kind":"classification","tasks":';

function declaredOrder(raw: string): string[] {
  const request = InferenceRequestSchema.parse(JSON.parse(raw));
  recordTaskOrder(raw, [request], false);
  return [...taskOrderOf((request.schema as { tasks: object }).tasks)];
}

test("declared task order survives escaped integer-like names", () => {
  const body = (key: string): string => `${ORDER_PREFIX}{"b":{"labels":["x"]},${key}:{"labels":["a"]}}}}`;
  assert.deepEqual(declaredOrder(body('"1"')), ["b", "1"]);
  assert.deepEqual(declaredOrder(body('"\\u0031"')), ["b", "1"]);
  assert.deepEqual(declaredOrder(body('"\\u0031\\u0030"')), ["b", "10"]);
  assert.deepEqual(declaredOrder(body('"a"')), ["b", "a"]);
});

test("declared task order follows Python for repeated names: first position, last value", () => {
  const raw = `${ORDER_PREFIX}{"b":{"labels":["x"]},"2":{"labels":["p"]},"b":{"labels":["last"]},"1":{"labels":["q"]}}}}`;
  const request = InferenceRequestSchema.parse(JSON.parse(raw));
  recordTaskOrder(raw, [request], false);
  const tasks = (request.schema as { tasks: Record<string, { labels: string[] }> }).tasks;
  assert.deepEqual([...taskOrderOf(tasks)], ["b", "2", "1"]);
  assert.deepEqual(tasks["b"]?.labels, ["last"]);
  assert.deepEqual((parseOrderedJson('{"b":1,"2":2,"b":3,"1":4}') as OrderedObject).entries, [
    ["b", 3],
    ["2", 2],
    ["1", 4],
  ]);
});

test("declared task order includes __proto__ and survives very long numeric literals", () => {
  const raw =
    `${ORDER_PREFIX}{"z":{"labels":["x"]},"__proto__":{"labels":["a"]},"3":{"labels":["c"]}}},` +
    `"options":{"threshold":0.5${"0".repeat(100)}1}}`;
  assert.deepEqual(declaredOrder(raw), ["z", "__proto__", "3"]);
});
