import assert from "node:assert/strict";
import test, { type TestContext } from "node:test";

import { GlinerApiError, GlinerClient, type InferenceRequest } from "@gliner-runner/client";
import { GlinerSidecar } from "@gliner-runner/client/sidecar";

import { CLI, FakeFactory, Gate, freePort, startServer, until, type TestServer } from "./helpers.js";

const request: InferenceRequest = {
  model: "decide-340m",
  backend: "onnx",
  precision: "fp32",
  operation: "classify",
  text: "An urgent invoice",
  schema: {
    kind: "classification",
    tasks: {
      priority: { labels: ["urgent", "routine"], label_descriptions: { urgent: "needs action" } },
    },
  },
};

async function connect(t: TestContext, options: Parameters<typeof startServer>[0] = {}) {
  const server: TestServer = await startServer(options);
  t.after(() => server.server.close());
  return { server, client: new GlinerClient({ baseUrl: server.base }) };
}

test("the TypeScript client parses every successful response", async (t) => {
  const { server, client } = await connect(t);

  const health = await server.get("/healthz");
  assert.deepEqual(await health.json(), { status: "ok", accepting_requests: true });

  const capabilities = await client.capabilities();
  assert.deepEqual(capabilities, [server.factory.capabilities()]);

  const models = await client.models();
  assert.equal(models.models.length, 3);
  assert.equal(models.memory_limit_bytes > 0, true);

  const response = await client.infer(request);
  assert.equal(response.backend, "onnx");
  assert.deepEqual(response.output, { label: "useful" });
  assert.equal(response.usage.outputTokens, 0);

  const batch = await client.batch([request, { ...request, text: "Another" }]);
  assert.equal(batch.responses.length, 2);
  assert.deepEqual(
    batch.responses.map((item) => item.usage.inputTokens),
    ["An urgent invoice".length, "Another".length],
  );

  const refreshed = await client.refreshModels();
  assert.equal(refreshed.models_checked, 3);

  const after = await client.models();
  assert.equal(after.models.find((model) => model.loaded)?.model_id, "decide-340m");
});

test("client errors expose the server's envelope", async (t) => {
  const { client } = await connect(t);

  await assert.rejects(client.infer({ ...request, backend: "pytorch" }), (error: unknown) => {
    assert.ok(error instanceof GlinerApiError);
    assert.equal(error.status, 409);
    assert.equal(error.code, "unsupported_configuration");
    assert.equal(error.retryable, false);
    return true;
  });

  await assert.rejects(client.infer({ ...request, model: "decide-multilingual" }), (error: unknown) => {
    assert.ok(error instanceof GlinerApiError);
    assert.equal(error.code, "model_download_required");
    const event = error.detail["event"] as Record<string, unknown>;
    assert.equal(event["download_endpoint"], "/v1/models/decide-multilingual/downloads");
    return true;
  });

  await assert.rejects(
    client.startModelDownload("decide-340m", { approved: true, destination: "/models" }),
    (error: unknown) =>
      error instanceof GlinerApiError && error.status === 409 && error.code === "unsupported_configuration",
  );

  await assert.rejects(
    client.modelDownload("3f2504e0-4f89-41d3-9a0c-0305e82c3301"),
    (error: unknown) =>
      error instanceof GlinerApiError && error.status === 404 && error.code === "download_job_not_found",
  );
});

test("validation arrays surface as invalid_error_response, and 429 is retryable", async (t) => {
  const factory = new FakeFactory();
  const gate = new Gate();
  factory.configure = (backend) => {
    backend.gate = gate;
  };
  const { server, client } = await connect(t, { factory, settings: { queueCapacity: 1 } });

  const invalid = await server.post("/v1/infer", { model: "x" });
  assert.equal(invalid.status, 422);
  const raw = new GlinerClient({
    baseUrl: server.base,
    fetch: async () => new Response(JSON.stringify(await invalid.clone().json()), { status: 422 }),
  });
  await assert.rejects(
    raw.infer(request),
    (error: unknown) => error instanceof GlinerApiError && error.code === "invalid_error_response",
  );

  const first = client.infer(request);
  await until(() => factory.backends[0]?.batches.length === 1, "dispatch");
  const second = client.infer(request);
  await until(() => server.server.runtime.metrics.snapshot()["requests_submitted_total"] === 2, "queued");
  await assert.rejects(client.infer(request), (error: unknown) => {
    assert.ok(error instanceof GlinerApiError);
    assert.equal(error.status, 429);
    assert.equal(error.code, "queue_full");
    assert.equal(error.retryable, true);
    return true;
  });
  gate.open();
  await Promise.all([first, second]);
});

test("the client's AbortSignal cancels queued work on the server", async (t) => {
  const factory = new FakeFactory();
  const gate = new Gate();
  factory.configure = (backend) => {
    backend.gate = gate;
  };
  const { server, client } = await connect(t, { factory });
  const first = client.infer(request);
  await until(() => factory.backends[0]?.batches.length === 1, "dispatch");
  const controller = new AbortController();
  const doomed = client.infer({ ...request, text: "abandoned" }, { signal: controller.signal });
  await until(() => server.server.runtime.metrics.snapshot()["requests_submitted_total"] === 2, "queued");
  controller.abort();
  await assert.rejects(doomed);
  await until(() => server.server.runtime.metrics.snapshot()["requests_cancelled_total"] === 1, "cancel");
  gate.open();
  await first;
  assert.equal(factory.backends[0]?.batches.length, 1);
});

test("the sidecar starts and stops the CLI", async () => {
  const port = await freePort();
  const sidecar = new GlinerSidecar({
    command: process.execPath,
    commandArguments: [CLI],
    port,
    startupTimeoutMs: 15_000,
    shutdownTimeoutMs: 5_000,
    runnerArguments: ["--max-batch-size", "4"],
    env: { PATH: process.env["PATH"] ?? "" },
  });
  await sidecar.start();
  try {
    assert.equal(sidecar.running, true);
    const models = await sidecar.client.models();
    assert.equal(models.device, "cpu");
    assert.equal(models.models.length, 3);
    await sidecar.client.capabilities();
  } finally {
    await sidecar.stop();
  }
  assert.equal(sidecar.running, false);
  await assert.rejects(fetch(`http://127.0.0.1:${port}/healthz`, { signal: AbortSignal.timeout(500) }));
});

test("the client surfaces 413 request_too_large as a non-retryable API error", async (t) => {
  const { client } = await connect(t, { settings: { maxBodyBytes: 1024 } });
  await assert.rejects(client.infer({ ...request, text: "x".repeat(4096) }), (error: unknown) => {
    assert.ok(error instanceof GlinerApiError);
    assert.equal(error.status, 413);
    assert.equal(error.code, "request_too_large");
    assert.equal(error.retryable, false);
    return true;
  });
  assert.equal((await client.infer(request)).output !== undefined, true);
});
