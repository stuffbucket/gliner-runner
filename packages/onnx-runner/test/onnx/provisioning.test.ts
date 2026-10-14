import assert from "node:assert/strict";
import { existsSync, mkdtempSync, readFileSync, readdirSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { setTimeout as sleep } from "node:timers/promises";
import test, { type TestContext } from "node:test";

import { GlinerClient, ModelDownloadDeniedError, ModelDownloadFailedError } from "@gliner-runner/client";

import { createServer, type RunnerServer } from "../../src/index.js";
import { OnnxArtifactCatalog } from "../../src/onnx/artifacts.js";
import { OnnxBackendFactory } from "../../src/onnx/backend.js";
import { HuggingFaceClient } from "../../src/onnx/huggingface.js";
import { OnnxModelProvisioner } from "../../src/onnx/install.js";
import { wireRequest } from "../helpers.js";
import { startMockHf, type Behavior, type MockHf } from "./hf-mock.js";
import { buildTemplateFixture, type TemplateFixture } from "./template-fixture.js";

const REPO = "fastino/GLiNER2.5-Decide";
const PIN = "9d1bfb848cd16d93ee56ccbc59cd92bf0f284d90";

interface Harness {
  fixture: TemplateFixture;
  hub: MockHf;
  modelsDir: string;
  server: RunnerServer;
  base: string;
  catalog: OnnxArtifactCatalog;
  post(path: string, body: unknown): Promise<Response>;
  get(path: string): Promise<Response>;
  infer(overrides?: Record<string, unknown>): Promise<Response>;
}

async function harness(
  t: TestContext,
  options: { autoDownload?: boolean; behavior?: (name: string, attempt: number) => Behavior; delayMs?: number; maxAttempts?: number } = {},
): Promise<Harness> {
  const fixture = buildTemplateFixture();
  const hub = await startMockHf({
    repository: REPO,
    revision: PIN,
    files: fixture.sources,
    ...(options.behavior === undefined ? {} : { behavior: options.behavior }),
    ...(options.delayMs === undefined ? {} : { delayMs: options.delayMs }),
  });
  const modelsDir = join(fixture.root, "models");
  const huggingface = new HuggingFaceClient({
    endpoint: hub.endpoint,
    token: undefined,
    retryBaseMs: 1,
    retryMaxMs: 2,
    maxAttempts: options.maxAttempts ?? 3,
    idleTimeoutMs: 5000,
  });
  const catalog = new OnnxArtifactCatalog(modelsDir, { remote: huggingface });
  const provisioner = new OnnxModelProvisioner({
    modelsDir,
    templatesDir: fixture.templatesDir,
    huggingface,
    env: { HOME: join(fixture.root, "home") },
  });
  const server = createServer({
    catalog,
    backends: [new OnnxBackendFactory(catalog)],
    provisioner,
    autoDownload: options.autoDownload ?? false,
    settings: { host: "127.0.0.1", port: 0, modelsDir },
  });
  const address = await server.listen();
  const base = `http://127.0.0.1:${address.port}`;
  t.after(async () => {
    await server.close();
    await hub.close();
  });
  const post = (path: string, body: unknown): Promise<Response> =>
    fetch(`${base}${path}`, { method: "POST", headers: { "content-type": "application/json" }, body: JSON.stringify(body) });
  return {
    fixture,
    hub,
    modelsDir,
    server,
    base,
    catalog,
    post,
    get: (path) => fetch(`${base}${path}`),
    infer: (overrides = {}) =>
      post(
        "/v1/infer",
        wireRequest({
          text: "my card was charged twice",
          schema: { kind: "classification", tasks: { intent: { labels: ["billing", "support"] } } },
          ...overrides,
        }),
      ),
  };
}

async function waitForJob(h: Harness, jobId: string, timeoutMs = 10_000): Promise<any> {
  const deadline = Date.now() + timeoutMs;
  for (;;) {
    const job = (await (await h.get(`/v1/model-downloads/${jobId}`)).json()) as any;
    if (job.state === "succeeded" || job.state === "failed") {
      return job;
    }
    assert.ok(Date.now() < deadline, `job stuck in ${job.state}`);
    await sleep(10);
  }
}

const stagingLeftovers = (h: Harness): string[] => {
  const parent = join(h.modelsDir, "onnx", "fastino--GLiNER2.5-Decide");
  return existsSync(parent) ? readdirSync(parent).filter((name) => name.startsWith(".")) : [];
};

test("missing model -> 409 event -> approved download -> succeeded -> inference works", async (t) => {
  const h = await harness(t);

  const models = (await (await h.get("/v1/models")).json()) as any;
  const model = models.models.find((entry: any) => entry.model_id === "decide-340m");
  assert.equal(model.download_destination, h.modelsDir);
  assert.equal(model.download_required, true);
  assert.equal(model.available_locally, false);
  assert.deepEqual(model.local_revisions, []);

  const missing = await h.infer();
  assert.equal(missing.status, 409);
  assert.deepEqual(await missing.json(), {
    detail: {
      code: "model_download_required",
      message: `model 'decide-340m' at revision ${PIN} is not installed; install it with the ONNX model preparation tool`,
      retryable: false,
      event: {
        type: "model_download_required",
        model: "decide-340m",
        revision: PIN,
        destination: h.modelsDir,
        download_endpoint: "/v1/models/decide-340m/downloads",
      },
    },
  });
  assert.equal(h.hub.requests.length, 0, "inference never downloads implicitly");

  const unapproved = await h.post("/v1/models/decide-340m/downloads", { approved: false, destination: h.modelsDir });
  assert.equal(unapproved.status, 422);
  assert.equal(((await unapproved.json()) as any).detail.code, "invalid_download_request");
  const unknown = await h.post("/v1/models/mystery/downloads", { approved: true, destination: h.modelsDir });
  assert.equal(unknown.status, 422);
  assert.equal(((await unknown.json()) as any).detail.message, "unknown GLiNER2.5 model 'mystery'");
  assert.equal(h.hub.requests.length, 0);

  const created = await h.post("/v1/models/decide-340m/downloads", { approved: true, destination: h.modelsDir });
  assert.equal(created.status, 202);
  const job = (await created.json()) as any;
  assert.deepEqual(Object.keys(job).sort(), ["destination", "error", "job_id", "model_id", "revision", "state"]);
  assert.match(job.job_id, /^[0-9a-f-]{36}$/);
  assert.deepEqual([job.model_id, job.revision, job.destination, job.state, job.error], ["decide-340m", PIN, h.modelsDir, "queued", null]);

  const done = await waitForJob(h, job.job_id);
  assert.deepEqual([done.state, done.error, done.model_id, done.revision], ["succeeded", null, "decide-340m", PIN]);
  assert.deepEqual(stagingLeftovers(h), []);

  const after = ((await (await h.get("/v1/models")).json()) as any).models.find((entry: any) => entry.model_id === "decide-340m");
  assert.deepEqual([after.download_required, after.available_locally, after.local_revisions], [false, true, [PIN]]);
  const capabilities = (await (await h.get("/v1/capabilities")).json()) as any[];
  assert.deepEqual(capabilities[0].validated_models, ["decide-340m"]);

  const inferred = await h.infer();
  assert.equal(inferred.status, 200);
  const body = (await inferred.json()) as any;
  assert.equal(body.error, null);
  assert.equal(body.model, "decide-340m");

  // The installed files are byte-identical to the reference artifact.
  const directory = join(h.modelsDir, "onnx", "fastino--GLiNER2.5-Decide", PIN);
  for (const [name, content] of Object.entries(h.fixture.artifact)) {
    assert.ok(readFileSync(join(directory, name)).equals(content), name);
  }
  assert.equal(readFileSync(join(directory, "manifest.json"), "utf8"), h.fixture.manifest);
  const requested = h.hub.requests.map((request) => request.url).sort();
  assert.deepEqual(requested, ["model.safetensors", "tokenizer.json", "tokenizer_config.json"].map((name) => `/${REPO}/resolve/${PIN}/${name}`));
});

test("repository aliases are canonicalised and an installed model is not downloaded again", async (t) => {
  const h = await harness(t);
  const created = await h.post(`/v1/models/${encodeURIComponent(REPO)}/downloads`, { approved: true, destination: h.modelsDir, revision: null });
  assert.equal(created.status, 202);
  const job = (await created.json()) as any;
  assert.equal(job.model_id, "decide-340m");
  assert.equal((await waitForJob(h, job.job_id)).state, "succeeded");

  const requests = h.hub.requests.length;
  const again = await h.post("/v1/models/decide-340m/downloads", { approved: true, destination: h.modelsDir });
  const second = await waitForJob(h, ((await again.json()) as any).job_id);
  assert.equal(second.state, "succeeded");
  assert.equal(h.hub.requests.length, requests, "a valid existing artifact is reused, never re-fetched or overwritten");
});

test("duplicate running jobs are refused; the second request names the model and revision", async (t) => {
  const h = await harness(t, { delayMs: 150 });
  const first = (await (await h.post("/v1/models/decide-340m/downloads", { approved: true, destination: h.modelsDir })).json()) as any;
  const duplicate = await h.post("/v1/models/fastino%2FGLiNER2.5-Decide/downloads", { approved: true, destination: h.modelsDir, revision: PIN });
  assert.equal(duplicate.status, 422);
  assert.deepEqual(((await duplicate.json()) as any).detail, {
    code: "invalid_download_request",
    message: `a download for decide-340m@${PIN} is already running`,
    retryable: false,
  });
  assert.equal((await waitForJob(h, first.job_id)).state, "succeeded");
});

test("a destination other than the configured models directory fails the job", async (t) => {
  const h = await harness(t);
  const created = await h.post("/v1/models/decide-340m/downloads", { approved: true, destination: join(h.fixture.root, "elsewhere") });
  assert.equal(created.status, 202);
  const job = await waitForJob(h, ((await created.json()) as any).job_id);
  assert.equal(job.state, "failed");
  assert.match(job.error, /^destination must equal the configured provider location /);
  assert.equal(h.hub.requests.length, 0);
});

test("only pinned revisions with a template can be prepared; others fail the job clearly", async (t) => {
  const h = await harness(t);
  const other = "1".repeat(40);
  const unpinned = await waitForJob(h, ((await (await h.post("/v1/models/decide-340m/downloads", { approved: true, destination: h.modelsDir, revision: other })).json()) as any).job_id);
  assert.equal(unpinned.state, "failed");
  assert.equal(unpinned.revision, other);
  assert.match(unpinned.error, /only the pinned revision 9d1bfb84.* has an ONNX template/);
  const untemplated = await waitForJob(h, ((await (await h.post("/v1/models/decide-1b/downloads", { approved: true, destination: h.modelsDir })).json()) as any).job_id);
  assert.equal(untemplated.state, "failed");
  assert.match(untemplated.error, /no ONNX template for fastino\/GLiNER2\.5-Decide-1B@/);
  assert.equal(h.hub.requests.length, 0);
});

test("download failures end the job as failed with the error and leave nothing behind", async (t) => {
  const h = await harness(t, { behavior: (name) => (name === "model.safetensors" ? { status: 500 } : undefined), maxAttempts: 2 });
  const created = (await (await h.post("/v1/models/decide-340m/downloads", { approved: true, destination: h.modelsDir })).json()) as any;
  const job = await waitForJob(h, created.job_id);
  assert.equal(job.state, "failed");
  assert.match(job.error, /failed to download model\.safetensors after 2 attempts: HTTP 500/);
  assert.deepEqual(stagingLeftovers(h), []);
  assert.equal(existsSync(join(h.modelsDir, "onnx", "fastino--GLiNER2.5-Decide", PIN)), false);
  assert.equal(((await (await h.get("/v1/models")).json()) as any).models[0].download_required, true);

  // A corrupted download (wrong hash) also fails closed.
  const corrupt = await harness(t, { behavior: (name) => (name === "tokenizer.json" ? "corrupt" : undefined) });
  const second = await waitForJob(corrupt, ((await (await corrupt.post("/v1/models/decide-340m/downloads", { approved: true, destination: corrupt.modelsDir })).json()) as any).job_id);
  assert.equal(second.state, "failed");
  assert.match(second.error, /tokenizer\.json failed verification/);
  assert.deepEqual(stagingLeftovers(corrupt), []);
});

test("unknown job ids are 404 and malformed ids 422", async (t) => {
  const h = await harness(t);
  const id = "3f2504e0-4f89-41d3-9a0c-0305e82c3301";
  const missing = await h.get(`/v1/model-downloads/${id.toUpperCase()}`);
  assert.equal(missing.status, 404);
  assert.deepEqual(await missing.json(), { detail: { code: "download_job_not_found", message: `'unknown download job ${id}'`, retryable: false } });
  assert.equal((await h.get("/v1/model-downloads/nope")).status, 422);
});

test("shutdown cancels a running download and cleans its staging directories", async (t) => {
  const h = await harness(t, { behavior: () => "stall" });
  await h.post("/v1/models/decide-340m/downloads", { approved: true, destination: h.modelsDir });
  const deadline = Date.now() + 5000;
  while ((h.hub.attempts.get("model.safetensors") ?? 0) === 0) {
    assert.ok(Date.now() < deadline, "download never started");
    await sleep(5);
  }
  assert.equal(stagingLeftovers(h).length, 1);
  const started = Date.now();
  await h.server.close();
  assert.ok(Date.now() - started < 3000, "shutdown does not wait for the stalled transfer");
  assert.deepEqual(stagingLeftovers(h), []);
  assert.equal(existsSync(join(h.modelsDir, "onnx", "fastino--GLiNER2.5-Decide", PIN)), false);
});

test("auto-download: concurrent inference requests share one job and then run", async (t) => {
  const h = await harness(t, { autoDownload: true, delayMs: 100 });
  const batch = h.post("/v1/batch", {
    requests: [
      wireRequest({ text: "my card", schema: { kind: "classification", tasks: { intent: { labels: ["billing", "support"] } } } }),
      wireRequest({ text: "urgent billing", schema: { kind: "classification", tasks: { intent: { labels: ["billing", "support"] } } } }),
    ],
  });
  const singles = [h.infer(), h.infer({ text: "billing support" }), h.infer({ model: "fastino/GLiNER2.5-Decide" })];
  const responses = await Promise.all([batch, ...singles]);
  assert.deepEqual(responses.map((response) => response.status), [200, 200, 200, 200]);
  assert.equal(((await responses[0]?.json()) as any).responses.length, 2);
  for (const name of ["model.safetensors", "tokenizer.json", "tokenizer_config.json"]) {
    assert.equal(h.hub.attempts.get(name), 1, `${name} was fetched exactly once for five waiting requests`);
  }
  assert.deepEqual(stagingLeftovers(h), []);
  // Later requests do not start another job.
  assert.equal((await h.infer()).status, 200);
  assert.equal(h.hub.attempts.get("model.safetensors"), 1);
});

test("auto-download: failures surface as 409 model_download_required with the job error", async (t) => {
  const h = await harness(t, { autoDownload: true, behavior: (name) => (name === "tokenizer.json" ? { status: 500 } : undefined), maxAttempts: 1 });
  const responses = await Promise.all([h.infer(), h.infer()]);
  for (const response of responses) {
    assert.equal(response.status, 409);
    const detail = ((await response.json()) as any).detail;
    assert.equal(detail.code, "model_download_required");
    assert.match(detail.message, /automatic installation failed: failed to download tokenizer\.json after 1 attempts: HTTP 500/);
    assert.deepEqual(detail.event, {
      type: "model_download_required",
      model: "decide-340m",
      revision: PIN,
      destination: h.modelsDir,
      download_endpoint: "/v1/models/decide-340m/downloads",
    });
  }
  assert.equal(h.hub.attempts.get("model.safetensors"), 1, "the two waiting requests shared one job");
});

test("auto-download only applies to curated models with a template at the pinned revision", async (t) => {
  const h = await harness(t, { autoDownload: true });
  const untemplated = await h.infer({ model: "decide-1b" });
  assert.equal(untemplated.status, 409);
  const detail = ((await untemplated.json()) as any).detail;
  assert.equal(detail.code, "model_download_required");
  assert.match(detail.message, /install it with the ONNX model preparation tool/);
  assert.equal(detail.event.download_endpoint, "/v1/models/decide-1b/downloads");
  const unpinned = await h.infer({ model: `decide-340m@${"2".repeat(40)}` });
  assert.equal(unpinned.status, 409);
  assert.equal(((await unpinned.json()) as any).detail.event.revision, "2".repeat(40));
  assert.equal(h.hub.requests.length, 0);
  assert.equal((await h.infer({ model: "mystery" })).status, 422);
});

test("a client that disconnects while waiting does not cancel the shared job", async (t) => {
  const h = await harness(t, { autoDownload: true, delayMs: 200 });
  const controller = new AbortController();
  const abandoned = fetch(`${h.base}/v1/infer`, {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify(wireRequest({ text: "a", schema: { kind: "classification", tasks: { intent: { labels: ["billing", "support"] } } } })),
    signal: controller.signal,
  });
  await sleep(50);
  controller.abort();
  await assert.rejects(abandoned);
  const kept = await h.infer();
  assert.equal(kept.status, 200);
  assert.equal(h.hub.attempts.get("model.safetensors"), 1);
});

test("refresh uses the Hub API and persists the Python-compatible cache", async (t) => {
  const fixture = buildTemplateFixture();
  const shas = {
    "fastino/GLiNER2.5-Decide": "a".repeat(40),
    "fastino/GLiNER2.5-multi-Decide": "b".repeat(40),
    "fastino/GLiNER2.5-Decide-1B": "c".repeat(40),
  };
  const listing = [...Object.keys(shas), "fastino/GLiNER2.5-Zeta-Decide", "fastino/GLiNER2.5-Alpha-DECIDE", "fastino/GLiNER2.5-Extract"];
  const hub = await startMockHf({ repository: REPO, revision: PIN, files: {}, shas, listing, pageSize: 3 });
  t.after(() => hub.close());
  const modelsDir = join(fixture.root, "models");
  const catalog = new OnnxArtifactCatalog(modelsDir, { remote: new HuggingFaceClient({ endpoint: hub.endpoint, token: undefined }) });
  const server = createServer({ catalog, backends: [new OnnxBackendFactory(catalog)], settings: { host: "127.0.0.1", port: 0, modelsDir } });
  const address = await server.listen();
  t.after(() => server.close());
  const post = (): Promise<Response> => fetch(`http://127.0.0.1:${address.port}/v1/models/refresh`, { method: "POST" });

  const first = (await (await post()).json()) as any;
  assert.deepEqual(first.changed_models, ["decide-340m", "decide-multilingual", "decide-1b"]);
  assert.deepEqual(first.discovered_models, ["fastino/GLiNER2.5-Alpha-DECIDE", "fastino/GLiNER2.5-Zeta-Decide"]);
  assert.equal(first.models_checked, 3);
  assert.equal(typeof first.refreshed_at, "number");

  const text = readFileSync(join(modelsDir, "catalog-cache.json"), "utf8");
  const cache = JSON.parse(text);
  assert.ok(text.endsWith("}\n"));
  assert.equal(text, `${JSON.stringify(sortKeys(cache), null, 2)}\n`);
  assert.equal(cache["decide-340m"].discovered_revision, "a".repeat(40));
  assert.deepEqual(cache["_discovered_models"].repositories, first.discovered_models);
  assert.equal(existsSync(join(modelsDir, "catalog-cache.tmp")), false);

  const second = (await (await post()).json()) as any;
  assert.deepEqual(second.changed_models, []);
  shas["fastino/GLiNER2.5-Decide-1B"] = "d".repeat(40);
  assert.deepEqual(((await (await post()).json()) as any).changed_models, ["decide-1b"]);

  // The cache is read when a catalog is constructed and feeds /v1/models.
  const reloaded = new OnnxArtifactCatalog(modelsDir);
  const inventory = await reloaded.inventory();
  assert.deepEqual(inventory.models.map((model) => model.discoveredRevision), ["a".repeat(40), "b".repeat(40), "d".repeat(40)]);
  assert.deepEqual(inventory.discoveredModels, first.discovered_models);
  const models = (await (await fetch(`http://127.0.0.1:${address.port}/v1/models`)).json()) as any;
  assert.deepEqual(models.discovered_models, first.discovered_models);
  assert.equal(models.models[2].discovered_revision, "d".repeat(40));
  assert.equal(models.models[2].pinned_revision, "67af949e17bebd20f05fe020a971079889e7a462");
});

test("refresh failures answer 502 catalog_refresh_failed and keep the previous cache", async (t) => {
  const fixture = buildTemplateFixture();
  const hub = await startMockHf({ repository: REPO, revision: PIN, files: {}, shas: { [REPO]: "a".repeat(40) } });
  t.after(() => hub.close());
  const modelsDir = join(fixture.root, "models");
  const catalog = new OnnxArtifactCatalog(modelsDir, { remote: new HuggingFaceClient({ endpoint: hub.endpoint, token: undefined }) });
  const server = createServer({ catalog, backends: [new OnnxBackendFactory(catalog)], settings: { host: "127.0.0.1", port: 0, modelsDir } });
  const address = await server.listen();
  t.after(() => server.close());
  const response = await fetch(`http://127.0.0.1:${address.port}/v1/models/refresh`, { method: "POST" });
  assert.equal(response.status, 502);
  const detail = ((await response.json()) as any).detail;
  assert.equal(detail.code, "catalog_refresh_failed");
  assert.equal(detail.retryable, true);
  assert.match(detail.message, /HTTP 404/);
  assert.equal(existsSync(join(modelsDir, "catalog-cache.json")), false);

  // No remote configured at all is the same failure, not a crash.
  const offline = createServer({ catalog: new OnnxArtifactCatalog(modelsDir), backends: [], settings: { host: "127.0.0.1", port: 0, modelsDir } });
  const offlineAddress = await offline.listen();
  t.after(() => offline.close());
  assert.equal((await fetch(`http://127.0.0.1:${offlineAddress.port}/v1/models/refresh`, { method: "POST" })).status, 502);
});

function sortKeys(value: unknown): unknown {
  if (Array.isArray(value)) {
    return value.map(sortKeys);
  }
  if (typeof value === "object" && value !== null) {
    return Object.fromEntries(Object.entries(value).sort(([a], [b]) => (a < b ? -1 : 1)).map(([k, v]) => [k, sortKeys(v)]));
  }
  return value;
}

test("the TypeScript client's ensureModel drives the whole flow against the real server", async (t) => {
  const h = await harness(t, { delayMs: 30 });
  const client = new GlinerClient({ baseUrl: h.base });

  await assert.rejects(client.ensureModel("decide-340m", { approve: () => false }), ModelDownloadDeniedError);
  assert.equal(h.hub.requests.length, 0);

  const asked: string[] = [];
  const model = await client.ensureModel("decide-340m", {
    approve: (item) => {
      asked.push(`${item.model_id}:${item.download_destination}`);
      return true;
    },
    pollIntervalMs: 10,
  });
  assert.deepEqual(asked, [`decide-340m:${h.modelsDir}`]);
  assert.deepEqual([model.download_required, model.available_locally, model.local_revisions], [false, true, [PIN]]);
  // Already installed: no question, no new download.
  const requests = h.hub.requests.length;
  await client.ensureModel("decide-340m", { approve: () => assert.fail("installed") });
  assert.equal(h.hub.requests.length, requests);

  const response = await client.infer({
    model: "decide-340m",
    backend: "onnx",
    precision: "fp32",
    operation: "classify",
    text: "my card was charged twice",
    schema: { kind: "classification", tasks: { intent: { labels: ["billing", "support"] } } },
  });
  assert.equal(response.error, null);

  const failing = await harness(t, { behavior: (name) => (name === "tokenizer.json" ? { status: 500 } : undefined), maxAttempts: 1 });
  await assert.rejects(
    new GlinerClient({ baseUrl: failing.base }).ensureModel("decide-340m", { approve: () => true, pollIntervalMs: 10 }),
    (error: unknown) => error instanceof ModelDownloadFailedError && /HTTP 500/.test(error.message),
  );
});

void mkdtempSync;
void tmpdir;
