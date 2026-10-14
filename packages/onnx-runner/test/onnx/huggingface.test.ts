import assert from "node:assert/strict";
import { existsSync, mkdtempSync, readFileSync, readdirSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { test } from "node:test";

import { RequestCancelledError } from "../../src/index.js";
import { HuggingFaceClient, HuggingFaceError, huggingFaceOptionsFromEnv, type DownloadProgress } from "../../src/onnx/huggingface.js";
import { startMockHf, type MockHf, type MockHfOptions } from "./hf-mock.js";
import { sha256 } from "./template-fixture.js";

const REPO = "fastino/GLiNER2.5-Decide";
const REVISION = "9d1bfb848cd16d93ee56ccbc59cd92bf0f284d90";
const FILES: Record<string, Buffer> = {
  "model.safetensors": Buffer.from("w".repeat(1000)),
  "tokenizer.json": Buffer.from('{"tokenizer":true}'),
};
const described = (name: string) => ({ path: name, size: (FILES[name] as Buffer).length, sha256: sha256(FILES[name] as Buffer) });

async function withHub(
  options: Partial<MockHfOptions>,
  run: (hub: MockHf, directory: string) => Promise<void>,
): Promise<void> {
  const hub = await startMockHf({ repository: REPO, revision: REVISION, files: FILES, ...options });
  try {
    await run(hub, mkdtempSync(join(tmpdir(), "gliner-hf-")));
  } finally {
    await hub.close();
  }
}

const fast = { retryBaseMs: 1, retryMaxMs: 2 };

test("downloads files with streaming verification and progress", async () => {
  await withHub({}, async (hub, directory) => {
    const client = new HuggingFaceClient({ endpoint: hub.endpoint, token: undefined, ...fast });
    const progress: DownloadProgress[] = [];
    await client.downloadAll(REPO, REVISION, ["model.safetensors", "tokenizer.json"].map(described), directory, {
      onProgress: (entry) => progress.push(entry),
    });
    assert.deepEqual(readdirSync(directory).sort(), ["model.safetensors", "tokenizer.json"]);
    assert.ok(readFileSync(join(directory, "model.safetensors")).equals(FILES["model.safetensors"] as Buffer));
    assert.equal(progress.at(-1)?.bytes, 1000 + 18);
    assert.equal(progress.at(-1)?.total, 1000 + 18);
    assert.ok(progress.length >= 2);
    assert.equal(hub.requests[0]?.url, `/${REPO}/resolve/${REVISION}/model.safetensors`);
    assert.equal(hub.requests[0]?.headers.authorization, undefined);
  });
});

test("sends the bearer token to the hub and follows redirects without forwarding it to the CDN", async () => {
  await withHub({ token: "hf_secret", cdn: true }, async (hub, directory) => {
    const client = new HuggingFaceClient({ endpoint: hub.endpoint, token: "hf_secret", ...fast });
    await client.downloadFile(REPO, REVISION, described("tokenizer.json"), join(directory, "tokenizer.json"));
    assert.equal(readFileSync(join(directory, "tokenizer.json"), "utf8"), '{"tokenizer":true}');
    const hubRequest = hub.requests.find((request) => request.server === "hub");
    const cdnRequest = hub.requests.find((request) => request.server === "cdn");
    assert.equal(hubRequest?.headers.authorization, "Bearer hf_secret");
    assert.ok(cdnRequest, "the redirect was followed");
    assert.equal(cdnRequest.headers.authorization, undefined);

    const anonymous = new HuggingFaceClient({ endpoint: hub.endpoint, token: undefined, ...fast });
    await assert.rejects(
      anonymous.downloadFile(REPO, REVISION, described("tokenizer.json"), join(directory, "again.json")),
      (error: unknown) => error instanceof HuggingFaceError && error.status === 401 && /HF_TOKEN/.test(error.message),
    );
    // A 401 is final: one request, no retries.
    assert.equal(hub.requests.filter((request) => request.server === "hub" && request.headers.authorization === undefined).length, 1);
  });
});

test("endpoint and token come from HF_ENDPOINT and HF_TOKEN", () => {
  assert.deepEqual(huggingFaceOptionsFromEnv({ HF_ENDPOINT: "http://mirror", HF_TOKEN: "t" }), { endpoint: "http://mirror", token: "t" });
  assert.deepEqual(huggingFaceOptionsFromEnv({ HF_ENDPOINT: "", HF_TOKEN: "" }), { endpoint: undefined, token: undefined });
  assert.equal(new HuggingFaceClient({ endpoint: "http://mirror/", token: undefined }).endpoint, "http://mirror");
  assert.equal(
    new HuggingFaceClient({ endpoint: "http://mirror", token: undefined }).sourceUrl(REPO, REVISION, "a b/c.json"),
    `http://mirror/${REPO}/resolve/${REVISION}/a%20b/c.json`,
  );
});

test("retries transient failures with backoff and then succeeds", async () => {
  const plan = new Map<string, number>();
  await withHub(
    {
      behavior: (name, attempt) => {
        plan.set(name, attempt);
        if (name === "tokenizer.json") {
          return [{ status: 503 }, "reset", { status: 429 }, undefined][attempt - 1] as never;
        }
        return undefined;
      },
    },
    async (hub, directory) => {
      const client = new HuggingFaceClient({ endpoint: hub.endpoint, token: undefined, ...fast });
      await client.downloadFile(REPO, REVISION, described("tokenizer.json"), join(directory, "tokenizer.json"));
      assert.equal(plan.get("tokenizer.json"), 4);
      assert.equal(readFileSync(join(directory, "tokenizer.json"), "utf8"), '{"tokenizer":true}');
      assert.deepEqual(readdirSync(directory), ["tokenizer.json"], "no .partial file is left behind");
    },
  );
});

test("gives up after the configured attempts and removes the partial file", async () => {
  await withHub({ behavior: () => ({ status: 500 }) }, async (hub, directory) => {
    const client = new HuggingFaceClient({ endpoint: hub.endpoint, token: undefined, maxAttempts: 3, ...fast });
    await assert.rejects(
      client.downloadFile(REPO, REVISION, described("tokenizer.json"), join(directory, "tokenizer.json")),
      /failed to download tokenizer\.json after 3 attempts: HTTP 500/,
    );
    assert.equal(hub.attempts.get("tokenizer.json"), 3);
    assert.deepEqual(readdirSync(directory), []);
  });
});

test("a missing file is final (404 is not retried)", async () => {
  await withHub({}, async (hub, directory) => {
    const client = new HuggingFaceClient({ endpoint: hub.endpoint, token: undefined, ...fast });
    await assert.rejects(
      client.downloadFile(REPO, REVISION, { path: "absent.bin", size: 1, sha256: "0".repeat(64) }, join(directory, "absent.bin")),
      (error: unknown) => error instanceof HuggingFaceError && error.status === 404 && !error.retryable,
    );
    assert.equal(hub.requests.length, 1);
  });
});

test("size and hash mismatches fail closed without retrying", async () => {
  await withHub({ behavior: (name) => (name === "tokenizer.json" ? "corrupt" : undefined) }, async (hub, directory) => {
    const client = new HuggingFaceClient({ endpoint: hub.endpoint, token: undefined, ...fast });
    await assert.rejects(
      client.downloadFile(REPO, REVISION, described("tokenizer.json"), join(directory, "tokenizer.json")),
      /failed verification: sha256/,
    );
    assert.equal(hub.attempts.get("tokenizer.json"), 1);
    assert.deepEqual(readdirSync(directory), []);
  });

  await withHub({}, async (hub, directory) => {
    const client = new HuggingFaceClient({ endpoint: hub.endpoint, token: undefined, ...fast });
    // The pinned size is smaller than what the server sends (Content-Length mismatch) ...
    await assert.rejects(
      client.downloadFile(REPO, REVISION, { ...described("model.safetensors"), size: 999 }, join(directory, "a")),
      /1000 bytes on the server, the pinned revision records 999/,
    );
    // ... or larger.
    await assert.rejects(
      client.downloadFile(REPO, REVISION, { ...described("model.safetensors"), size: 1001 }, join(directory, "b")),
      /1000 bytes on the server, the pinned revision records 1001/,
    );
    assert.deepEqual(readdirSync(directory), []);
  });

  // Without a Content-Length the stream itself is measured: short and long bodies are both refused.
  await withHub({ behavior: (name) => (name === "tokenizer.json" ? "truncate" : undefined) }, async (hub, directory) => {
    const client = new HuggingFaceClient({ endpoint: hub.endpoint, token: undefined, ...fast });
    await assert.rejects(
      client.downloadFile(REPO, REVISION, described("tokenizer.json"), join(directory, "t")),
      /is 17 bytes, the pinned revision records 18/,
    );
    await assert.rejects(
      client.downloadFile(REPO, REVISION, { ...described("tokenizer.json"), size: 5 }, join(directory, "u")),
      /pinned revision records 5|exceeds the pinned size/,
    );
    assert.deepEqual(readdirSync(directory), []);
  });
});

test("a stalled transfer is aborted and retried", async () => {
  await withHub({ behavior: (name, attempt) => (name === "tokenizer.json" && attempt === 1 ? "stall" : undefined) }, async (hub, directory) => {
    const client = new HuggingFaceClient({ endpoint: hub.endpoint, token: undefined, idleTimeoutMs: 100, ...fast });
    await client.downloadFile(REPO, REVISION, described("tokenizer.json"), join(directory, "tokenizer.json"));
    assert.equal(hub.attempts.get("tokenizer.json"), 2);
  });
});

test("cancellation aborts an in-flight download and cleans up", async () => {
  await withHub({ behavior: () => "stall" }, async (hub, directory) => {
    const client = new HuggingFaceClient({ endpoint: hub.endpoint, token: undefined, idleTimeoutMs: 60_000, ...fast });
    const controller = new AbortController();
    const pending = client.downloadFile(REPO, REVISION, described("tokenizer.json"), join(directory, "tokenizer.json"), {
      signal: controller.signal,
    });
    setTimeout(() => controller.abort(), 50);
    await assert.rejects(pending, RequestCancelledError);
    assert.deepEqual(readdirSync(directory), []);
    await assert.rejects(
      client.downloadFile(REPO, REVISION, described("tokenizer.json"), join(directory, "x"), { signal: AbortSignal.abort() }),
      RequestCancelledError,
    );
    assert.equal(existsSync(join(directory, "x.partial")), false);
  });
});

test("catalog lookups: model sha and the paginated publisher listing", async () => {
  const listing = ["fastino/GLiNER2.5-Decide", "fastino/GLiNER2.5-New-Decide", "fastino/GLiNER2.5-Other", "fastino/GLiNER2.5-X-Decide"];
  await withHub({ sha: "a".repeat(40), listing, pageSize: 2 }, async (hub) => {
    const client = new HuggingFaceClient({ endpoint: hub.endpoint, token: "t", ...fast });
    assert.equal(await client.modelSha(REPO), "a".repeat(40));
    assert.deepEqual(await client.listModels("fastino", "GLiNER2.5"), listing);
    const list = hub.requests.filter((request) => request.url.startsWith("/api/models?"));
    assert.equal(list.length, 2);
    assert.match(list[0]?.url ?? "", /author=fastino&search=GLiNER2\.5/);
    assert.equal(list[0]?.headers.authorization, "Bearer t");
    await assert.rejects(client.modelSha("fastino/missing"), /HTTP 404/);
  });
  await withHub({ sha: undefined }, async (hub) => {
    await assert.rejects(new HuggingFaceClient({ endpoint: hub.endpoint, token: undefined }).modelSha(REPO), /HTTP 404/);
  });
});
