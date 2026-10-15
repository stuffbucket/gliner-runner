import assert from "node:assert/strict";
import { setTimeout as sleep } from "node:timers/promises";
import test, { type TestContext } from "node:test";

import { InvalidSchemaError, minimumMemoryBytes } from "../src/index.js";
import {
  FakeFactory,
  Gate,
  PINNED_340M,
  schemaWith,
  startServer,
  until,
  wireRequest,
  type TestServer,
} from "./helpers.js";

async function serve(t: TestContext, options: Parameters<typeof startServer>[0] = {}): Promise<TestServer> {
  const server = await startServer(options);
  t.after(() => server.server.close());
  return server;
}

async function json(response: Response): Promise<any> {
  return response.json();
}

test("health, infer, openapi and metrics follow the HTTP contract", async (t) => {
  const { get, post } = await serve(t);
  const health = await get("/healthz");
  assert.equal(health.status, 200);
  assert.deepEqual(await health.json(), { status: "ok", accepting_requests: true });

  const response = await post("/v1/infer", wireRequest({ request_id: "955B971B-4295-4E39-9E45-9B404E32BB6E" }));
  assert.equal(response.status, 200);
  assert.match(response.headers.get("content-type") ?? "", /^application\/json/);
  const body = await json(response);
  assert.deepEqual(Object.keys(body).sort(), [
    "backend",
    "error",
    "model",
    "output",
    "precision",
    "request_id",
    "timing",
    "usage",
  ]);
  assert.equal(body.request_id, "955b971b-4295-4e39-9e45-9b404e32bb6e");
  assert.deepEqual(body.output, { label: "useful" });
  assert.equal(body.error, null);
  assert.deepEqual(body.usage, { inputTokens: "An urgent invoice".length, outputTokens: 0 });
  assert.equal(typeof body.timing.queue_ms, "number");

  const openapi = await json(await get("/openapi.json"));
  assert.ok("/v1/infer" in openapi.paths);
  assert.equal(openapi.info.version, "1.0.0");

  const metrics = await get("/metrics");
  assert.equal(metrics.headers.get("content-type"), "text/plain; version=0.0.4; charset=utf-8");
  const text = await metrics.text();
  assert.match(text, /# TYPE gliner_runner_internal gauge\n/);
  assert.match(text, /gliner_runner_internal\{metric="requests_completed_total"\} 1\n/);
  assert.match(text, /gliner_runner_internal\{metric="queue_depth"\} 0\.0\n/);
});

test("a fresh server exposes only the metrics header", async (t) => {
  const { get } = await serve(t);
  assert.equal(
    await (await get("/metrics")).text(),
    "# HELP gliner_runner_internal Runtime scheduler metrics.\n# TYPE gliner_runner_internal gauge\n",
  );
});

test("capabilities lists registered backends only", async (t) => {
  const { get, factory } = await serve(t);
  const capabilities = await json(await get("/v1/capabilities"));
  assert.deepEqual(capabilities, [factory.capabilities()]);
});

test("backend invalid-schema errors become 422 invalid_schema", async (t) => {
  const factory = new FakeFactory();
  factory.configure = (backend) => {
    backend.inferError = new InvalidSchemaError("constraint set is infeasible");
  };
  const { post } = await serve(t, { factory });
  const response = await post("/v1/infer", wireRequest());
  assert.equal(response.status, 422);
  assert.match(response.headers.get("content-type") ?? "", /^application\/json/);
  assert.deepEqual(await response.json(), {
    detail: { code: "invalid_schema", message: "constraint set is infeasible", retryable: false },
  });
});

test("unexpected backend failures return a JSON 500 envelope", async (t) => {
  const factory = new FakeFactory();
  factory.configure = (backend) => {
    backend.inferError = new Error("secret internals");
  };
  const logs: string[] = [];
  const { post } = await serve(t, { factory, log: (message) => logs.push(message) });
  const response = await post("/v1/infer", wireRequest());
  assert.equal(response.status, 500);
  assert.match(response.headers.get("content-type") ?? "", /^application\/json/);
  const body = await json(response);
  assert.equal(body.detail.code, "internal_error");
  assert.equal(body.detail.retryable, false);
  assert.doesNotMatch(JSON.stringify(body), /secret internals/);
  assert.ok(logs.some((line) => line.includes("secret internals")));
  assert.ok(logs.every((line) => !line.includes("An urgent invoice")));
});

test("requests naming another backend are rejected without fallback", async (t) => {
  const { post, factory } = await serve(t);
  for (const backend of ["pytorch", "mlx"]) {
    const response = await post("/v1/infer", wireRequest({ backend }));
    assert.equal(response.status, 409);
    assert.deepEqual(await response.json(), {
      detail: {
        code: "unsupported_configuration",
        message: `backend ${backend} is not installed; no fallback was attempted`,
        retryable: false,
      },
    });
  }
  assert.equal(factory.created.length, 0);
});

test("unsupported operation and precision return 409 even for the resident model", async (t) => {
  const { post } = await serve(t);
  assert.equal((await post("/v1/infer", wireRequest())).status, 200);
  const entities = await post(
    "/v1/infer",
    wireRequest({ operation: "extract_entities", schema: { kind: "entities", labels: ["a"] } }),
  );
  assert.equal(entities.status, 409);
  assert.equal((await json(entities)).detail.code, "unsupported_configuration");
  const bf16 = await post("/v1/infer", wireRequest({ precision: "bf16" }));
  assert.equal(bf16.status, 409);
});

test("validation failures use the FastAPI 422 array shape", async (t) => {
  const { post } = await serve(t);
  const response = await post("/v1/infer", wireRequest({ text: "" }));
  assert.equal(response.status, 422);
  assert.deepEqual((await json(response)).detail, [
    {
      type: "string_too_short",
      loc: ["body", "text"],
      msg: "String should have at least 1 character",
      input: "",
      ctx: { min_length: 1 },
    },
  ]);
  const malformed = await post("/v1/infer", "{bad");
  assert.equal(malformed.status, 422);
  assert.equal((await json(malformed)).detail[0].type, "json_invalid");
});

test("routing errors keep FastAPI shapes", async (t) => {
  const { get, base } = await serve(t);
  const missing = await get("/nope");
  assert.equal(missing.status, 404);
  assert.deepEqual(await missing.json(), { detail: "Not Found" });
  const wrongMethod = await fetch(`${base}/v1/infer`);
  assert.equal(wrongMethod.status, 405);
  assert.equal(wrongMethod.headers.get("allow"), "POST");
  assert.deepEqual(await wrongMethod.json(), { detail: "Method Not Allowed" });
});

test("model inventory reports catalog state, profiles and guidance", async (t) => {
  const { get, post } = await serve(t);
  const before = await json(await get("/v1/models"));
  assert.equal(before.default_model, "decide-340m");
  assert.equal(before.device, "cpu");
  assert.equal(before.idle_ttl_seconds, 0);
  assert.equal(before.models.length, 3);
  assert.deepEqual(
    before.models.map((model: any) => [model.model_id, model.download_required, model.loaded]),
    [
      ["decide-340m", false, false],
      ["decide-multilingual", true, false],
      ["decide-1b", false, false],
    ],
  );
  const [first] = before.models;
  assert.equal(first.family, "GLiNER2.5-Decide");
  assert.deepEqual(first.local_revisions, [PINNED_340M]);
  assert.equal(first.download_destination, "/models");
  assert.deepEqual(first.supported_profiles, [
    { device: "cpu", precision: "fp16" },
    { device: "cpu", precision: "fp32" },
  ]);
  assert.deepEqual(first.default_profile, { device: "cpu", precision: "fp16" });
  assert.deepEqual(first.profile_guidance["cpu/fp32"], {
    estimated_minimum_memory_bytes: minimumMemoryBytes(486_444_053, "fp32"),
    memory_basis: first.profile_guidance["cpu/fp32"].memory_basis,
    baseline_profile: "cpu/fp32 direct official PyTorch oracle",
    observed_max_abs_score_drift: null,
    drift_order_of_magnitude: null,
    characterization_scope: "No published numeric characterization for this model/profile.",
  });

  await post("/v1/infer", wireRequest({ model: "fastino/GLiNER2.5-Decide", precision: "fp16" }));
  const after = await json(await get("/v1/models"));
  const loaded = after.models.filter((model: any) => model.loaded);
  assert.equal(loaded.length, 1);
  assert.equal(loaded[0].model_id, "decide-340m");
  assert.deepEqual(loaded[0].loaded_profile, { device: "cpu", precision: "fp16" });
});

test("uninstalled models answer 409 model_download_required with the download event", async (t) => {
  const { post } = await serve(t);
  const response = await post("/v1/infer", wireRequest({ model: "decide-multilingual" }));
  assert.equal(response.status, 409);
  const { detail } = await json(response);
  assert.equal(detail.code, "model_download_required");
  assert.equal(detail.retryable, false);
  assert.deepEqual(detail.event, {
    type: "model_download_required",
    model: "decide-multilingual",
    revision: "e173bca1f0c4217d7a55b3b0c705d5ece8a0218d",
    destination: "/models",
    download_endpoint: "/v1/models/decide-multilingual/downloads",
  });
  const unknown = await post("/v1/infer", wireRequest({ model: "mystery" }));
  assert.equal(unknown.status, 422);
  assert.equal((await json(unknown)).detail.code, "model_integrity");
});

test("download requests are validated, then declined because artifacts are prepared offline", async (t) => {
  const { post, get } = await serve(t);
  const body = { approved: true, destination: "/models" };
  const declined = await post("/v1/models/decide-340m/downloads", body);
  assert.equal(declined.status, 409);
  const detail = (await json(declined)).detail;
  assert.equal(detail.code, "unsupported_configuration");
  assert.equal(detail.retryable, false);
  assert.match(detail.message, /model preparation tool/);

  const unapproved = await post("/v1/models/decide-340m/downloads", { ...body, approved: false });
  assert.equal(unapproved.status, 422);
  assert.equal((await json(unapproved)).detail.code, "invalid_download_request");

  const unknown = await post("/v1/models/mystery/downloads", body);
  assert.equal(unknown.status, 422);
  assert.equal((await json(unknown)).detail.message, "unknown GLiNER2.5 model 'mystery'");

  const encoded = await post(`/v1/models/${encodeURIComponent("fastino/GLiNER2.5-Decide")}/downloads`, body);
  assert.equal(encoded.status, 409);

  const malformed = await post("/v1/models/decide-340m/downloads", { approved: "yes", destination: "" });
  assert.equal(malformed.status, 422);
  assert.ok(Array.isArray((await json(malformed)).detail));

  const jobId = "3f2504e0-4f89-41d3-9a0c-0305e82c3301";
  const missing = await get(`/v1/model-downloads/${jobId}`);
  assert.equal(missing.status, 404);
  assert.deepEqual(await missing.json(), {
    detail: {
      code: "download_job_not_found",
      message: `'unknown download job ${jobId}'`,
      retryable: false,
    },
  });
  const invalid = await get("/v1/model-downloads/not-a-uuid");
  assert.equal(invalid.status, 422);
  assert.deepEqual((await json(invalid)).detail[0].loc, ["path", "job_id"]);
});

test("refresh returns the catalog result and maps failures to 502", async (t) => {
  const { post, catalog } = await serve(t);
  const refreshed = await json(await post("/v1/models/refresh", ""));
  assert.deepEqual(Object.keys(refreshed).sort(), [
    "changed_models",
    "discovered_models",
    "models_checked",
    "refreshed_at",
  ]);
  assert.equal(refreshed.models_checked, 3);

  const failing = await serve(t, {
    catalog: Object.assign(Object.create(catalog), {
      refresh: async () => {
        throw new Error("registry offline");
      },
    }),
  });
  const response = await failing.post("/v1/models/refresh", "");
  assert.equal(response.status, 502);
  assert.deepEqual(await response.json(), {
    detail: { code: "catalog_refresh_failed", message: "registry offline", retryable: true },
  });
});

test("memory-limit failures are reported over HTTP with diagnostics", async (t) => {
  const { post, factory } = await serve(t, { settings: { memoryLimitBytes: 3_000_000_000 } });
  const response = await post("/v1/infer", wireRequest());
  assert.equal(response.status, 409);
  const { detail } = await json(response);
  assert.match(detail.message, /estimated minimum is 3650112851 bytes/);
  assert.deepEqual(detail, {
    code: "model_memory_limit",
    message: detail.message,
    retryable: false,
    model: "decide-340m",
    profile: "cpu/fp32",
    limit_bytes: 3_000_000_000,
    required_bytes: 3_650_112_851,
    observed_bytes: null,
    attempted_load_unloaded: false,
  });
  assert.equal(factory.created.length, 0);
});

test("batch returns responses in request order and micro-batches them together", async (t) => {
  const { post, factory } = await serve(t);
  const items = ["a", "bb", "ccc"].map((text) => wireRequest({ text }));
  const response = await post("/v1/batch", { requests: items });
  assert.equal(response.status, 200);
  const { responses } = await json(response);
  assert.deepEqual(
    responses.map((item: any) => item.usage.inputTokens),
    [1, 2, 3],
  );
  assert.equal(responses.length, 3);
  assert.deepEqual(factory.backends[0]?.batchSizes, [3]);
});

test("batch items for different schemas are executed in separate batches", async (t) => {
  const { post, factory } = await serve(t);
  const response = await post("/v1/batch", {
    requests: [wireRequest(), wireRequest({ schema: schemaWith(["x", "y"]) })],
  });
  assert.equal(response.status, 200);
  assert.deepEqual(factory.backends[0]?.batchSizes.sort(), [1, 1]);
});

test("batch mixing aliases of one model succeeds on the shared worker", async (t) => {
  const { post, factory } = await serve(t);
  const response = await post("/v1/batch", {
    requests: [wireRequest({ model: "decide-340m" }), wireRequest({ model: "fastino/GLiNER2.5-Decide" })],
  });
  assert.equal(response.status, 200);
  assert.equal(factory.created.length, 1);
  assert.deepEqual(factory.backends[0]?.batchSizes, [2]);
  const { responses } = await json(response);
  assert.deepEqual(
    responses.map((item: any) => item.model),
    ["decide-340m", "fastino/GLiNER2.5-Decide"],
  );
});

test("one failing batch item fails the whole call", async (t) => {
  const { post } = await serve(t);
  const response = await post("/v1/batch", {
    requests: [wireRequest(), wireRequest({ model: "decide-multilingual" })],
  });
  assert.equal(response.status, 409);
  assert.equal((await json(response)).detail.code, "model_download_required");
});

test("batch bounds are validated", async (t) => {
  const { post } = await serve(t);
  const empty = await post("/v1/batch", { requests: [] });
  assert.equal(empty.status, 422);
  assert.equal((await json(empty)).detail[0].type, "too_short");
  const invalidItem = await post("/v1/batch", { requests: [wireRequest({ text: "" })] });
  assert.deepEqual((await json(invalidItem)).detail[0].loc, ["body", "requests", 0, "text"]);
});

test("overlapping HTTP requests are micro-batched", async (t) => {
  const { post, factory } = await serve(t, { settings: { batchWindowMs: 25 } });
  const responses = await Promise.all(
    Array.from({ length: 5 }, (_, index) => post("/v1/infer", wireRequest({ text: "x".repeat(index + 1) }))),
  );
  assert.ok(responses.every((response) => response.status === 200));
  assert.deepEqual(factory.backends[0]?.batchSizes, [5]);
});

test("a full queue answers 429 with Retry-After", async (t) => {
  const factory = new FakeFactory();
  const gate = new Gate();
  factory.configure = (backend) => {
    backend.gate = gate;
  };
  const { post, server } = await serve(t, { factory, settings: { queueCapacity: 1 } });
  const first = post("/v1/infer", wireRequest());
  await until(() => factory.backends[0]?.batches.length === 1, "first dispatch");
  const second = post("/v1/infer", wireRequest());
  await until(() => server.runtime.metrics.snapshot()["requests_submitted_total"] === 2, "second queued");

  const rejected = await post("/v1/infer", wireRequest());
  assert.equal(rejected.status, 429);
  assert.equal(rejected.headers.get("retry-after"), "1");
  assert.deepEqual(await rejected.json(), {
    detail: { code: "queue_full", message: "model worker queue is full", retryable: true },
  });

  gate.open();
  assert.equal((await first).status, 200);
  assert.equal((await second).status, 200);
});

test("a closed runtime degrades health and answers 503 runtime_closed", async (t) => {
  const { get, post, server } = await serve(t);
  await post("/v1/infer", wireRequest());
  await server.runtime.close();
  const health = await get("/healthz");
  assert.equal(health.status, 200);
  assert.deepEqual(await health.json(), { status: "degraded", accepting_requests: false });
  const response = await post("/v1/infer", wireRequest());
  assert.equal(response.status, 503);
  assert.deepEqual(await response.json(), {
    detail: { code: "runtime_closed", message: "runtime is closed", retryable: true },
  });
});

test("a client disconnect cancels queued work before dispatch", async (t) => {
  const factory = new FakeFactory();
  const gate = new Gate();
  factory.configure = (backend) => {
    backend.gate = gate;
  };
  const { post, server } = await serve(t, { factory });
  const blocker = post("/v1/infer", wireRequest());
  await until(() => factory.backends[0]?.batches.length === 1, "first dispatch");

  const controller = new AbortController();
  const port = (server.http.address() as { port: number }).port;
  const doomed = fetch(`http://127.0.0.1:${port}/v1/infer`, {
    method: "POST",
    body: JSON.stringify(wireRequest({ text: "never runs" })),
    signal: controller.signal,
  });
  await until(() => server.runtime.metrics.snapshot()["requests_submitted_total"] === 2, "queued");
  controller.abort();
  await assert.rejects(doomed);
  await until(() => server.runtime.metrics.snapshot()["requests_cancelled_total"] === 1, "cancellation");
  assert.equal(server.runtime.metrics.snapshot()["queue_depth"], 0);

  gate.open();
  assert.equal((await blocker).status, 200);
  assert.deepEqual(
    factory.backends[0]?.batches.flat().map((request) => request.text),
    ["An urgent invoice"],
  );
});

test("aborting a batch cancels every queued item", async (t) => {
  const factory = new FakeFactory();
  const gate = new Gate();
  factory.configure = (backend) => {
    backend.gate = gate;
  };
  const { post, server } = await serve(t, { factory });
  const blocker = post("/v1/infer", wireRequest());
  await until(() => factory.backends[0]?.batches.length === 1, "first dispatch");

  const controller = new AbortController();
  const port = (server.http.address() as { port: number }).port;
  const doomed = fetch(`http://127.0.0.1:${port}/v1/batch`, {
    method: "POST",
    body: JSON.stringify({ requests: [wireRequest(), wireRequest(), wireRequest()] }),
    signal: controller.signal,
  });
  await until(() => server.runtime.metrics.snapshot()["requests_submitted_total"] === 4, "batch queued");
  controller.abort();
  await assert.rejects(doomed);
  await until(() => server.runtime.metrics.snapshot()["requests_cancelled_total"] === 3, "cancellations");
  gate.open();
  await blocker;
  assert.equal(factory.backends[0]?.batches.length, 1);
});

test("a failing batch item cancels its queued siblings", async (t) => {
  const factory = new FakeFactory();
  const gate = new Gate();
  factory.configure = (backend) => {
    backend.gate = gate;
  };
  const { post, server } = await serve(t, { factory, settings: { queueCapacity: 2 } });
  const blocker = post("/v1/infer", wireRequest());
  await until(() => factory.backends[0]?.batches.length === 1, "first dispatch");
  const response = await post("/v1/batch", {
    requests: Array.from({ length: 4 }, () => wireRequest()),
  });
  assert.equal(response.status, 429);
  await until(() => server.runtime.metrics.snapshot()["requests_cancelled_total"] === 2, "sibling cancellation");
  gate.open();
  await blocker;
  await sleep(20);
  assert.equal(factory.backends[0]?.batches.length, 1);
});

test("close drains in-flight requests before shutting down", async (t) => {
  const factory = new FakeFactory();
  const gate = new Gate();
  factory.configure = (backend) => {
    backend.gate = gate;
  };
  const { post, server, base } = await serve(t, { factory });
  const inflight = post("/v1/infer", wireRequest());
  await until(() => factory.backends[0]?.batches.length === 1, "dispatch");
  let closed = false;
  const closing = server.close().then(() => {
    closed = true;
  });
  await sleep(30);
  assert.equal(closed, false);
  await assert.rejects(fetch(`${base}/healthz`, { signal: AbortSignal.timeout(500) }));
  gate.open();
  assert.equal((await inflight).status, 200);
  await closing;
  assert.equal(factory.backends[0]?.closed, true);
});

test("bodies over the size limit get 413 request_too_large and the server keeps serving", async (t) => {
  const { post, base } = await serve(t, { settings: { maxBodyBytes: 2048 } });
  const huge = wireRequest({ text: "x".repeat(4096) });
  for (const path of ["/v1/infer", "/v1/batch"]) {
    const response = await post(path, path === "/v1/batch" ? { requests: [huge] } : huge);
    assert.equal(response.status, 413);
    assert.match(response.headers.get("content-type") ?? "", /^application\/json/);
    assert.deepEqual(await response.json(), {
      detail: {
        code: "request_too_large",
        message: "request body exceeds the limit of 2048 bytes",
        retryable: false,
        limit_bytes: 2048,
      },
    });
  }

  // No content-length: the limit applies while streaming.
  const chunks = Array.from({ length: 8 }, () => Buffer.from("x".repeat(512)));
  const streamed = await fetch(`${base}/v1/infer`, {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: new ReadableStream({
      start(controller) {
        for (const chunk of chunks) {
          controller.enqueue(chunk);
        }
        controller.close();
      },
    }),
    duplex: "half",
  } as RequestInit);
  assert.equal(streamed.status, 413);
  assert.equal(((await streamed.json()) as any).detail.code, "request_too_large");

  const ok = await post("/v1/infer", wireRequest());
  assert.equal(ok.status, 200);
});

test("a body of exactly the limit is accepted", async (t) => {
  const body = JSON.stringify(wireRequest());
  const { post } = await serve(t, { settings: { maxBodyBytes: Buffer.byteLength(body) } });
  assert.equal((await post("/v1/infer", body)).status, 200);
  const tooBig = await serve(t, { settings: { maxBodyBytes: Buffer.byteLength(body) - 1 } });
  assert.equal((await tooBig.post("/v1/infer", body)).status, 413);
});

test("CORS is off by default: an Origin header gets no CORS headers and no preflight answer", async (t) => {
  const { base } = await serve(t);
  const headers = { origin: "http://localhost:5173" };
  const response = await fetch(`${base}/healthz`, { headers });
  assert.equal(response.status, 200);
  assert.equal(response.headers.get("access-control-allow-origin"), null);
  assert.equal(response.headers.get("vary"), null);
  const preflight = await fetch(`${base}/v1/infer`, { method: "OPTIONS", headers });
  assert.equal(preflight.status, 405);
  assert.equal(preflight.headers.get("access-control-allow-origin"), null);
});

test("CORS: an exactly matching origin is echoed, answers preflight, and other origins get nothing", async (t) => {
  const { base, post } = await serve(t, { settings: { corsOrigins: ["http://localhost:5173", "https://app.example.com"] } });
  const allowed = { origin: "https://app.example.com" };

  const health = await fetch(`${base}/healthz`, { headers: allowed });
  assert.equal(health.status, 200);
  assert.equal(health.headers.get("access-control-allow-origin"), "https://app.example.com");
  assert.equal(health.headers.get("vary"), "Origin");

  const missing = await fetch(`${base}/nope`, { headers: allowed });
  assert.equal(missing.status, 404);
  assert.equal(missing.headers.get("access-control-allow-origin"), "https://app.example.com");

  const preflight = await fetch(`${base}/v1/infer`, {
    method: "OPTIONS",
    headers: { ...allowed, "access-control-request-method": "POST", "access-control-request-headers": "content-type" },
  });
  assert.equal(preflight.status, 204);
  assert.equal(await preflight.text(), "");
  assert.equal(preflight.headers.get("access-control-allow-origin"), "https://app.example.com");
  assert.equal(preflight.headers.get("vary"), "Origin");
  assert.equal(preflight.headers.get("access-control-allow-methods"), "GET, POST, OPTIONS");
  assert.equal(preflight.headers.get("access-control-allow-headers"), "content-type");
  assert.equal(preflight.headers.get("access-control-max-age"), "600");

  for (const origin of ["http://localhost:5174", "http://localhost:5173/", "https://app.example.com:443", "HTTPS://APP.EXAMPLE.COM", "null"]) {
    const other = await fetch(`${base}/healthz`, { headers: { origin } });
    assert.equal(other.headers.get("access-control-allow-origin"), null, origin);
    assert.equal(other.headers.get("vary"), null, origin);
    const denied = await fetch(`${base}/v1/infer`, { method: "OPTIONS", headers: { origin, "access-control-request-method": "POST" } });
    assert.equal(denied.status, 405, origin);
    assert.equal(denied.headers.get("access-control-allow-methods"), null, origin);
  }
  const noOrigin = await fetch(`${base}/healthz`);
  assert.equal(noOrigin.headers.get("access-control-allow-origin"), null);

  // Requests without an Origin still work as before.
  assert.equal((await post("/v1/infer", wireRequest())).status, 200);
});
