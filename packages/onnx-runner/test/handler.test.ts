import assert from "node:assert/strict";
import test from "node:test";

import { createRunnerHandler, defaultSettings, RequestCancelledError, type RunnerHandler } from "../src/index.js";
import { FakeFactory, Gate, installedCatalog, wireRequest } from "./helpers.js";

function handler(t: test.TestContext, maxBodyBytes = 4096): { runner: RunnerHandler; factory: FakeFactory } {
  const factory = new FakeFactory();
  const runner = createRunnerHandler({
    catalog: installedCatalog(),
    backends: [factory],
    settings: { ...defaultSettings(), maxBodyBytes },
  });
  t.after(() => runner.close());
  return { runner, factory };
}

test("the transport-free handler serves the API from method, path and body", async (t) => {
  const { runner } = handler(t);
  const health = await runner.handle({ method: "GET", path: "/healthz" });
  assert.equal(health.status, 200);
  assert.equal(health.contentType, "application/json");
  assert.deepEqual(JSON.parse(health.body), { status: "ok", accepting_requests: true });

  const inferred = await runner.handle({ method: "POST", path: "/v1/infer?ignored=1", body: JSON.stringify(wireRequest()) });
  assert.equal(inferred.status, 200, inferred.body);
  assert.equal((JSON.parse(inferred.body) as { model: string }).model, "decide-340m");

  const lazy = await runner.handle({ method: "POST", path: "/v1/infer", body: () => Promise.resolve(JSON.stringify(wireRequest())) });
  assert.equal(lazy.status, 200);
});

test("unknown paths, wrong methods, invalid and oversize bodies get the HTTP server's replies", async (t) => {
  const { runner } = handler(t, 64);
  assert.equal((await runner.handle({ method: "GET", path: "/nope" })).status, 404);
  const wrong = await runner.handle({ method: "GET", path: "/v1/infer" });
  assert.equal(wrong.status, 405);
  assert.equal(wrong.headers?.["allow"], "POST");
  assert.equal((await runner.handle({ method: "POST", path: "/v1/infer", body: "{" })).status, 422);
  const large = await runner.handle({ method: "POST", path: "/v1/infer", body: "é".repeat(40) });
  assert.equal(large.status, 413);
  assert.equal((JSON.parse(large.body) as { detail: { code: string } }).detail.code, "request_too_large");
});

test("a cancelled request rejects instead of producing a reply", async (t) => {
  const { runner, factory } = handler(t);
  const gate = new Gate();
  t.after(() => gate.open());
  factory.configure = (backend) => {
    backend.gate = gate;
  };
  // The first request holds the model's worker; the second waits in its queue and is cancelled there.
  const first = runner.handle({ method: "POST", path: "/v1/infer", body: JSON.stringify(wireRequest()) });
  const controller = new AbortController();
  const second = runner.handle({
    method: "POST",
    path: "/v1/infer",
    body: JSON.stringify(wireRequest()),
    signal: controller.signal,
  });
  setTimeout(() => controller.abort(), 20);
  await assert.rejects(second);
  gate.open();
  assert.equal((await first).status, 200);
  await assert.rejects(
    runner.handle({ method: "POST", path: "/v1/infer", body: () => Promise.reject(new RequestCancelledError("gone")) }),
    RequestCancelledError,
  );
});
