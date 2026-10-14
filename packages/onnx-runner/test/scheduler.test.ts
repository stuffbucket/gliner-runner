import assert from "node:assert/strict";
import { setTimeout as sleep } from "node:timers/promises";
import test from "node:test";

import {
  Metrics,
  ModelWorker,
  QueueFullError,
  RequestCancelledError,
  RunnerClosedError,
  type WorkerOptions,
} from "../src/index.js";
import { FakeBackend, Gate, makeRequest, schemaWith, until } from "./helpers.js";

async function startWorker(options: WorkerOptions = {}, backend = new FakeBackend()) {
  const metrics = new Metrics();
  const worker = new ModelWorker(backend, { batchWindowMs: 10, metrics, ...options });
  await worker.start();
  return { worker, backend, metrics };
}

test("rejects invalid scheduler configuration", () => {
  const backend = new FakeBackend();
  assert.throws(() => new ModelWorker(backend, { queueCapacity: 0 }), /invalid scheduler configuration/);
  assert.throws(() => new ModelWorker(backend, { maxBatchSize: 0 }), /invalid scheduler configuration/);
  assert.throws(() => new ModelWorker(backend, { batchWindowMs: -1 }), /invalid scheduler configuration/);
});

test("micro-batches concurrent compatible requests and preserves order", async () => {
  const { worker, backend, metrics } = await startWorker();
  const requests = ["a", "bb", "ccc"].map((text) => makeRequest({ text }));
  const responses = await Promise.all(requests.map((request) => worker.submit(request)));
  assert.deepEqual(backend.batchSizes, [3]);
  assert.deepEqual(
    responses.map((response) => response.request_id),
    requests.map((request) => request.request_id),
  );
  assert.deepEqual(
    responses.map((response) => response.usage),
    [1, 2, 3].map((inputTokens) => ({ inputTokens, outputTokens: 0 })),
  );
  assert.deepEqual(metrics.snapshot(), {
    batches_total: 1,
    last_batch_size: 3,
    queue_depth: 0,
    requests_completed_total: 3,
    requests_submitted_total: 3,
  });
  await worker.close();
  assert.equal(backend.closed, true);
});

test("default batch size is 16 and extra work forms later batches", async () => {
  const { worker, backend } = await startWorker({});
  await Promise.all(Array.from({ length: 20 }, () => worker.submit(makeRequest())));
  assert.deepEqual(backend.batchSizes, [16, 4]);
  await worker.close();
});

test("never batches different schemas, options or operations", async () => {
  const { worker, backend } = await startWorker();
  await Promise.all([
    worker.submit(makeRequest()),
    worker.submit(makeRequest({ schema: schemaWith(["x", "y"]) })),
    worker.submit(makeRequest({ options: { threshold: 0.9 } })),
  ]);
  assert.deepEqual(backend.batchSizes.sort(), [1, 1, 1]);
  await worker.close();
});

test("grouping ignores key order and default filling but not array order", async () => {
  const described = (descriptions: Record<string, string>, labels = ["a", "b"]) =>
    makeRequest({
      schema: { kind: "classification", tasks: { p: { labels, label_descriptions: descriptions } } },
    });
  const { worker, backend } = await startWorker();
  await Promise.all([
    worker.submit(described({ a: "x", b: "y" })),
    worker.submit(described({ b: "y", a: "x" })),
    worker.submit(
      makeRequest({
        schema: {
          kind: "classification",
          tasks: { p: { labels: ["a", "b"], label_descriptions: { a: "x", b: "y" }, min_labels: 1 } },
          constraints: [],
        },
        options: { include_confidence: true, threshold: 0.5, output_format: "native" },
      }),
    ),
    worker.submit(described({ a: "x", b: "y" }, ["b", "a"])),
  ]);
  assert.deepEqual(backend.batchSizes, [3, 1]);
  await worker.close();
});

test("rebatches interleaved compatible requests across incompatible ones", async () => {
  const { worker, backend } = await startWorker();
  const a = (text: string) => makeRequest({ text });
  const b = (text: string) => makeRequest({ text, schema: schemaWith(["x", "y"]) });
  await Promise.all([a("a1"), b("b1"), a("a2"), b("b2")].map((request) => worker.submit(request)));
  assert.deepEqual(backend.batchSizes, [2, 2]);
  assert.deepEqual(
    backend.batches.map((batch) => batch.map((request) => request.text)),
    [
      ["a1", "a2"],
      ["b1", "b2"],
    ],
  );
  await worker.close();
});

test("enforces the batch limit against both the backlog and the queue", async () => {
  const { worker, backend } = await startWorker({ maxBatchSize: 2 });
  const a = () => makeRequest();
  const b = () => makeRequest({ schema: schemaWith(["x", "y"]) });
  await Promise.all([a(), b(), a(), a(), a()].map((request) => worker.submit(request)));
  assert.deepEqual(backend.batchSizes, [2, 1, 2]);
  await worker.close();
});

test("a zero batch window still dispatches", async () => {
  const { worker, backend } = await startWorker({ batchWindowMs: 0 });
  await worker.submit(makeRequest());
  assert.deepEqual(backend.batchSizes, [1]);
  await worker.close();
});

test("close drains accepted work, including incompatible backlog", async () => {
  const { worker, backend } = await startWorker();
  const pending = [
    worker.submit(makeRequest()),
    worker.submit(makeRequest({ schema: schemaWith(["x", "y"]) })),
    worker.submit(makeRequest()),
  ];
  const closing = worker.close();
  const responses = await Promise.all(pending);
  await closing;
  assert.equal(responses.length, 3);
  assert.deepEqual(backend.batchSizes, [2, 1]);
  assert.equal(backend.closed, true);
});

test("work that outlives the worker loop fails with 'model worker closed'", async () => {
  const { worker, backend, metrics } = await startWorker();
  const poisoned = makeRequest();
  (poisoned.schema as unknown as Record<string, unknown>)["self"] = poisoned.schema;
  const results = await Promise.allSettled([worker.submit(poisoned), worker.submit(makeRequest())]);
  for (const result of results) {
    assert.equal(result.status, "rejected");
    assert.ok(result.status === "rejected" && result.reason instanceof RunnerClosedError);
    assert.equal(result.status === "rejected" && result.reason.message, "model worker closed");
  }
  assert.equal(metrics.snapshot()["queue_depth"], 0);
  await assert.rejects(worker.submit(makeRequest()), /model worker is not accepting requests/);
  await assert.rejects(worker.close(), RangeError);
  assert.equal(backend.closed, true);
});

test("cancelling before dispatch removes the request without running inference", async () => {
  const { worker, backend, metrics } = await startWorker();
  const controller = new AbortController();
  const pending = worker.submit(makeRequest(), controller.signal);
  controller.abort();
  await assert.rejects(pending, RequestCancelledError);
  await sleep(30);
  assert.deepEqual(backend.batches, []);
  assert.deepEqual(metrics.snapshot(), {
    requests_submitted_total: 1,
    requests_cancelled_total: 1,
    queue_depth: 0,
  });
  await worker.close();
});

test("a cancelled batch does not stop the worker", async () => {
  const { worker, backend } = await startWorker();
  const controller = new AbortController();
  const cancelled = worker.submit(makeRequest(), controller.signal);
  controller.abort();
  await assert.rejects(cancelled, RequestCancelledError);
  const response = await worker.submit(makeRequest({ text: "later" }));
  assert.equal(response.usage.inputTokens, 5);
  assert.deepEqual(backend.batchSizes, [1]);
  await worker.close();
});

test("a pre-aborted signal is rejected without queueing", async () => {
  const { worker, metrics } = await startWorker();
  await assert.rejects(worker.submit(makeRequest(), AbortSignal.abort()), RequestCancelledError);
  assert.deepEqual(metrics.snapshot(), {});
  await worker.close();
});

test("cancelling after dispatch discards the result but still counts", async () => {
  const backend = new FakeBackend();
  backend.gate = new Gate();
  const { worker, metrics } = await startWorker({}, backend);
  const controller = new AbortController();
  const pending = worker.submit(makeRequest(), controller.signal);
  await until(() => backend.batches.length === 1, "dispatch");
  controller.abort();
  await assert.rejects(pending, RequestCancelledError);
  backend.gate.open();
  await worker.close();
  assert.deepEqual(metrics.snapshot(), {
    batches_total: 1,
    last_batch_size: 1,
    queue_depth: 0,
    requests_cancelled_total: 1,
    requests_completed_total: 1,
    requests_submitted_total: 1,
  });
});

test("a full intake queue rejects with QueueFullError and exact metrics", async () => {
  const backend = new FakeBackend();
  backend.gate = new Gate();
  const { worker, metrics } = await startWorker({ queueCapacity: 1 }, backend);
  const first = worker.submit(makeRequest());
  await until(() => backend.batches.length === 1, "first dispatch");
  const second = worker.submit(makeRequest());
  await assert.rejects(worker.submit(makeRequest()), (error: unknown) => {
    assert.ok(error instanceof QueueFullError);
    assert.equal(error.message, "model worker queue is full");
    return true;
  });
  assert.equal(metrics.snapshot()["queue_depth"], 1);
  backend.gate.open();
  await Promise.all([first, second]);
  await worker.close();
  assert.deepEqual(metrics.snapshot(), {
    batches_total: 2,
    last_batch_size: 1,
    queue_depth: 0,
    requests_completed_total: 2,
    requests_rejected_total: 1,
    requests_submitted_total: 2,
  });
});

test("queue_depth counts backlog items", async () => {
  const backend = new FakeBackend();
  backend.gate = new Gate();
  const { worker, metrics } = await startWorker({ batchWindowMs: 5 }, backend);
  const pending = [
    worker.submit(makeRequest()),
    worker.submit(makeRequest({ schema: schemaWith(["x", "y"]) })),
    worker.submit(makeRequest({ schema: schemaWith(["p", "q"]) })),
  ];
  await until(() => backend.batches.length === 1, "first dispatch");
  assert.equal(metrics.snapshot()["queue_depth"], 2);
  backend.gate.open();
  await Promise.all(pending);
  await worker.close();
});

test("start is idempotent; closed or unstarted workers reject work", async () => {
  const backend = new FakeBackend();
  const worker = new ModelWorker(backend, { batchWindowMs: 1 });
  await assert.rejects(worker.submit(makeRequest()), /model worker is not accepting requests/);
  await worker.close();
  assert.deepEqual(backend.events, []);

  await Promise.all([worker.start(), worker.start()]);
  await worker.start();
  assert.deepEqual(backend.events, ["load:start", "load:end"]);
  assert.equal(worker.accepting, true);
  await worker.close();
  assert.equal(worker.accepting, false);
  await assert.rejects(worker.submit(makeRequest()), (error: unknown) => {
    assert.ok(error instanceof RunnerClosedError);
    assert.equal(error.message, "model worker is not accepting requests");
    return true;
  });
});

test("backend exceptions and wrong result counts reach every submitter", async () => {
  const wrongCount = new FakeBackend();
  wrongCount.resultCount = () => 1;
  const first = await startWorker({}, wrongCount);
  const results = await Promise.allSettled([
    first.worker.submit(makeRequest()),
    first.worker.submit(makeRequest()),
  ]);
  for (const result of results) {
    assert.ok(result.status === "rejected");
    assert.match(String(result.reason), /backend returned the wrong number of results/);
  }
  assert.deepEqual(first.metrics.snapshot(), {
    batch_failures_total: 1,
    batches_total: 1,
    last_batch_size: 2,
    queue_depth: 0,
    requests_submitted_total: 2,
  });
  await first.worker.close();

  const failing = new FakeBackend();
  failing.inferError = new Error("device lost");
  const second = await startWorker({}, failing);
  await assert.rejects(second.worker.submit(makeRequest()), /device lost/);
  assert.equal(second.metrics.snapshot()["batch_failures_total"], 1);
  failing.inferError = undefined;
  await second.worker.submit(makeRequest());
  await second.worker.close();
});

test("responses echo request identity and carry per-request usage and timings", async () => {
  const ticks = [10_000, 10_250, 10_750];
  const { worker } = await startWorker({ batchWindowMs: 0, clock: () => ticks.shift() as number });
  const request = makeRequest({ model: "alias-of-model", precision: "fp16", text: "four" });
  const response = await worker.submit(request);
  assert.deepEqual(response, {
    request_id: request.request_id,
    model: "alias-of-model",
    backend: "onnx",
    precision: "fp16",
    output: { label: "useful" },
    error: null,
    timing: { queue_ms: 250, inference_ms: 500 },
    usage: { inputTokens: 4, outputTokens: 0 },
  });
  await worker.close();
});

test("timings are clamped when the clock moves backwards", async () => {
  const ticks = [20_000, 10_000, 5_000];
  const { worker } = await startWorker({ batchWindowMs: 0, clock: () => ticks.shift() as number });
  const response = await worker.submit(makeRequest());
  assert.deepEqual(response.timing, { queue_ms: 0, inference_ms: 0 });
  await worker.close();
});

test("backlog counts toward the queue bound: total pending work per worker is capped", async () => {
  const backend = new FakeBackend();
  backend.gate = new Gate();
  const { worker, metrics } = await startWorker({ queueCapacity: 2, batchWindowMs: 30 }, backend);
  const accepted = [worker.submit(makeRequest())];
  await sleep(5); // the first request has left the intake queue and is collecting its batch
  accepted.push(
    worker.submit(makeRequest({ schema: schemaWith(["x", "y"]) })),
    worker.submit(makeRequest({ schema: schemaWith(["p", "q"]) })),
  );
  await until(() => backend.batches.length === 1, "first dispatch");
  // The two incompatible requests have moved from the intake queue into the backlog; the bound still applies.
  assert.equal(metrics.snapshot()["queue_depth"], 2);
  await assert.rejects(worker.submit(makeRequest({ schema: schemaWith(["m", "n"]) })), QueueFullError);
  await assert.rejects(worker.submit(makeRequest()), QueueFullError);
  assert.equal(metrics.snapshot()["queue_depth"], 2);
  assert.equal(metrics.snapshot()["requests_rejected_total"], 2);

  backend.gate.open();
  await Promise.all(accepted);
  // Capacity is available again once the backlog drains.
  await worker.submit(makeRequest());
  await worker.close();
  assert.equal(metrics.snapshot()["queue_depth"], 0);
});
