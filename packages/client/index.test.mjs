import assert from "node:assert/strict";
import test from "node:test";

import {
  DownloadRequestSchema,
  GlinerClient,
  ClassificationTaskSchema,
  InferenceRequestSchema,
} from "./dist/index.js";

const request = {
  model: "decide-340m",
  backend: "pytorch",
  precision: "fp32",
  operation: "classify",
  text: "An urgent invoice",
  schema: {
    kind: "classification",
    tasks: { priority: { labels: ["urgent", "routine"] } },
  },
};

test("validates requests before transport", async () => {
  let called = false;
  const client = new GlinerClient({
    fetch: async () => {
      called = true;
      return new Response("{}");
    },
  });

  await assert.rejects(
    client.infer({ ...request, text: "" }),
    /text must not be blank/,
  );
  assert.equal(called, false);
});

test("validates successful server responses", async () => {
  const client = new GlinerClient({
    fetch: async () =>
      Response.json({
        request_id: "955b971b-4295-4e39-9e45-9b404e32bb6e",
        model: "decide-340m",
        backend: "pytorch",
        precision: "fp32",
        output: { priority: "urgent" },
        error: null,
        timing: { queue_ms: 1, inference_ms: 2 },
        usage: { inputTokens: 42, outputTokens: 0 },
      }),
  });

  const response = await client.infer(request);
  assert.deepEqual(response.output, { priority: "urgent" });
  assert.deepEqual(response.usage, { inputTokens: 42, outputTokens: 0 });
});

test("rejects malformed successful responses", async () => {
  const client = new GlinerClient({
    fetch: async () => Response.json({ output: "missing metadata" }),
  });

  await assert.rejects(client.infer(request), /Invalid input/);
});

test("requires exact usage on successful responses", async () => {
  const client = new GlinerClient({
    fetch: async () =>
      Response.json({
        request_id: "955b971b-4295-4e39-9e45-9b404e32bb6e",
        model: "decide-340m",
        backend: "pytorch",
        precision: "fp32",
        output: { priority: "urgent" },
        error: null,
        timing: { queue_ms: 1, inference_ms: 2 },
      }),
  });

  await assert.rejects(client.infer(request), /usage/);
});

test("exports schemas for Electron boundaries", () => {
  assert.equal(InferenceRequestSchema.parse(request).operation, "classify");
  assert.throws(
    () =>
      InferenceRequestSchema.parse({
        ...request,
        operation: "extract_entities",
      }),
    /requires a entities schema/,
  );
  assert.equal(
    DownloadRequestSchema.parse({
      approved: true,
      destination: "/models",
    }).approved,
    true,
  );
});

test("validates optional classification label descriptions", () => {
  assert.deepEqual(
    ClassificationTaskSchema.parse({
      labels: ["urgent", "routine", "unknown"],
      label_descriptions: {
        urgent: "Requires immediate handling",
        unknown: null,
      },
    }).label_descriptions,
    {
      urgent: "Requires immediate handling",
      unknown: null,
    },
  );
  assert.throws(
    () =>
      ClassificationTaskSchema.parse({
        labels: ["urgent", "routine"],
        label_descriptions: { missing: "Not declared" },
      }),
    /must be a declared label/,
  );
});

test("starts an explicitly approved model download", async () => {
  const client = new GlinerClient({
    fetch: async (_url, init) => {
      assert.equal(init.method, "POST");
      return Response.json({
        job_id: "955b971b-4295-4e39-9e45-9b404e32bb6e",
        model_id: "decide-340m",
        revision: "9d1bfb848cd16d93ee56ccbc59cd92bf0f284d90",
        destination: "/models",
        state: "queued",
        error: null,
      });
    },
  });

  const job = await client.startModelDownload("decide-340m", {
    approved: true,
    destination: "/models",
  });
  assert.equal(job.state, "queued");
});

const PINNED = "9d1bfb848cd16d93ee56ccbc59cd92bf0f284d90";
const JOB_ID = "955b971b-4295-4e39-9e45-9b404e32bb6e";

function inventoryItem(overrides = {}) {
  return {
    model_id: "decide-340m",
    repository: "fastino/GLiNER2.5-Decide",
    family: "GLiNER2.5-Decide",
    description: "English GLiNER2.5 classification model",
    pinned_revision: PINNED,
    discovered_revision: PINNED,
    parameter_count: 486444053,
    local_revisions: [],
    available_locally: false,
    loaded: false,
    loaded_profile: null,
    supported_profiles: [{ device: "cpu", precision: "fp32" }],
    default_profile: { device: "cpu", precision: "fp32" },
    profile_guidance: {},
    download_destination: "/models",
    download_required: true,
    ...overrides,
  };
}

function inventory(...items) {
  return {
    default_model: "decide-340m",
    device: "cpu",
    idle_ttl_seconds: 0,
    memory_limit_bytes: 1073741824,
    discovered_models: [],
    models: items,
  };
}

/** A scripted runner: serves inventories in order, then job states in order. */
function scriptedRunner({ inventories, jobs, startError }) {
  const calls = [];
  const fetch = async (url, init = {}) => {
    const path = new URL(url).pathname;
    calls.push({ method: init.method, path, body: init.body });
    if (path === "/v1/models") {
      return Response.json(inventories.shift());
    }
    if (path === "/v1/models/decide-340m/downloads") {
      if (startError !== undefined) {
        return Response.json({ detail: startError }, { status: 422 });
      }
      return Response.json(jobs.shift(), { status: 202 });
    }
    if (path === `/v1/model-downloads/${JOB_ID}`) {
      return Response.json(jobs.shift());
    }
    return new Response("{}", { status: 404 });
  };
  return { fetch, calls };
}

const job = (state, error = null) => ({
  job_id: JOB_ID,
  model_id: "decide-340m",
  revision: PINNED,
  destination: "/models",
  state,
  error,
});

test("ensureModel returns immediately when the model is installed, without asking", async () => {
  const runner = scriptedRunner({ inventories: [inventory(inventoryItem({ download_required: false, local_revisions: [PINNED], available_locally: true }))], jobs: [] });
  const client = new GlinerClient({ fetch: runner.fetch });
  const model = await client.ensureModel("decide-340m", {
    approve: () => assert.fail("must not ask for approval"),
  });
  assert.equal(model.download_required, false);
  assert.deepEqual(runner.calls.map((call) => call.path), ["/v1/models"]);
});

test("ensureModel asks for approval, downloads to the reported destination and polls to success", async () => {
  const installed = inventoryItem({ download_required: false, local_revisions: [PINNED], available_locally: true });
  const runner = scriptedRunner({
    inventories: [inventory(inventoryItem()), inventory(installed)],
    jobs: [job("queued"), job("running"), job("running"), job("succeeded")],
  });
  const client = new GlinerClient({ fetch: runner.fetch });
  const asked = [];
  const model = await client.ensureModel("fastino/GLiNER2.5-Decide", {
    approve: (item) => {
      asked.push(item.model_id);
      return true;
    },
    pollIntervalMs: 1,
  });
  assert.deepEqual(asked, ["decide-340m"]);
  assert.equal(model.download_required, false);
  const start = runner.calls.find((call) => call.method === "POST");
  assert.deepEqual(JSON.parse(start.body), { approved: true, destination: "/models" });
  assert.equal(runner.calls.filter((call) => call.path === `/v1/model-downloads/${JOB_ID}`).length, 3);
});

test("ensureModel never downloads when approval is denied (sync or async)", async () => {
  for (const approve of [() => false, async () => false]) {
    const runner = scriptedRunner({ inventories: [inventory(inventoryItem())], jobs: [] });
    const client = new GlinerClient({ fetch: runner.fetch });
    await assert.rejects(client.ensureModel("decide-340m", { approve }), (error) => {
      assert.equal(error.name, "ModelDownloadDeniedError");
      assert.equal(error.modelId, "decide-340m");
      return true;
    });
    assert.equal(runner.calls.some((call) => call.method === "POST"), false);
  }
});

test("ensureModel throws with the job error when the download fails", async () => {
  const runner = scriptedRunner({
    inventories: [inventory(inventoryItem())],
    jobs: [job("queued"), job("failed", "failed to download model.safetensors after 5 attempts: HTTP 503")],
  });
  const client = new GlinerClient({ fetch: runner.fetch });
  await assert.rejects(client.ensureModel("decide-340m", { approve: () => true, pollIntervalMs: 1 }), (error) => {
    assert.equal(error.name, "ModelDownloadFailedError");
    assert.match(error.message, /HTTP 503/);
    assert.equal(error.job.state, "failed");
    return true;
  });
});

test("ensureModel reports a succeeded job that did not make the model available", async () => {
  const runner = scriptedRunner({
    inventories: [inventory(inventoryItem()), inventory(inventoryItem())],
    jobs: [job("succeeded")],
  });
  const client = new GlinerClient({ fetch: runner.fetch });
  await assert.rejects(client.ensureModel("decide-340m", { approve: () => true }), /still not installed/);
});

test("ensureModel rejects unknown models, surfaces refusals, honours revisions and cancellation", async () => {
  const unknown = scriptedRunner({ inventories: [inventory(inventoryItem())], jobs: [] });
  await assert.rejects(new GlinerClient({ fetch: unknown.fetch }).ensureModel("mystery", { approve: () => true }), /unknown GLiNER2\.5 model 'mystery'/);

  const refused = scriptedRunner({
    inventories: [inventory(inventoryItem())],
    jobs: [],
    startError: { code: "invalid_download_request", message: `a download for decide-340m@${PINNED} is already running`, retryable: false },
  });
  await assert.rejects(new GlinerClient({ fetch: refused.fetch }).ensureModel("decide-340m", { approve: () => true }), (error) => {
    assert.equal(error.code, "invalid_download_request");
    return true;
  });

  // An explicit revision is checked against the installed revisions, and is sent with the download.
  const other = "1".repeat(40);
  const pinnedOnly = inventoryItem({ download_required: false, local_revisions: [PINNED], available_locally: true });
  const withRevision = scriptedRunner({
    inventories: [inventory(pinnedOnly), inventory(inventoryItem({ ...pinnedOnly, local_revisions: [PINNED, other] }))],
    jobs: [job("succeeded")],
  });
  await new GlinerClient({ fetch: withRevision.fetch }).ensureModel(`decide-340m@${other}`, { approve: () => true });
  assert.deepEqual(JSON.parse(withRevision.calls.find((call) => call.method === "POST").body), {
    approved: true,
    destination: "/models",
    revision: other,
  });
  const satisfied = scriptedRunner({ inventories: [inventory(pinnedOnly)], jobs: [] });
  await new GlinerClient({ fetch: satisfied.fetch }).ensureModel(`decide-340m@${PINNED}`, { approve: () => assert.fail("installed") });

  const slow = scriptedRunner({ inventories: [inventory(inventoryItem())], jobs: [job("queued"), job("running"), job("running")] });
  const controller = new AbortController();
  const pending = new GlinerClient({ fetch: slow.fetch }).ensureModel("decide-340m", {
    approve: () => true,
    pollIntervalMs: 20,
    signal: controller.signal,
  });
  setTimeout(() => controller.abort(new Error("stop")), 30);
  await assert.rejects(pending, /stop/);
});
