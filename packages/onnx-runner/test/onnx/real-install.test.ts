import assert from "node:assert/strict";
import { spawn } from "node:child_process";
import { existsSync, mkdtempSync, readFileSync, readdirSync, rmSync } from "node:fs";
import { homedir, tmpdir } from "node:os";
import { join } from "node:path";
import test from "node:test";

import { KNOWN_MODELS } from "../../src/index.js";
import { OnnxArtifactCatalog } from "../../src/onnx/artifacts.js";
import { sha256OfFile } from "../../src/onnx/assemble.js";
import { loadTemplate, resolveTemplatesDir } from "../../src/onnx/templates.js";
import { CLI } from "../helpers.js";
import { startMockHf } from "./hf-mock.js";

/**
 * Guarded real-model test. Runs only when GLINER_RUNNER_TEST_SNAPSHOTS_DIR points at a directory holding
 * `<owner>--<name>/<revision>/` snapshots and the real templates for that model exist. It serves the local
 * snapshot from a mock Hugging Face endpoint, runs `models install` into a temporary models directory and checks
 * that every output file's SHA-256 equals the template's recorded digest, which is the digest of the Python
 * canonical artifact: the Node preparation is byte-identical to it.
 *
 * GLINER_RUNNER_TEST_REAL_MODELS (comma-separated model ids) restricts the models; the default is all of them.
 */
const SNAPSHOTS = process.env["GLINER_RUNNER_TEST_SNAPSHOTS_DIR"];
const ONLY = (process.env["GLINER_RUNNER_TEST_REAL_MODELS"] ?? "").split(",").filter((id) => id !== "");
const templatesDir = resolveTemplatesDir();

function run(args: string[], env: NodeJS.ProcessEnv): Promise<{ code: number | null; stdout: string; stderr: string }> {
  return new Promise((resolve) => {
    const child = spawn(process.execPath, [CLI, ...args], {
      env: { PATH: process.env["PATH"] ?? "", HOME: mkdtempSync(join(tmpdir(), "gliner-home-")), ...env },
      stdio: ["ignore", "pipe", "pipe"],
    });
    let stdout = "";
    let stderr = "";
    child.stdout.on("data", (chunk: Buffer) => (stdout += chunk.toString()));
    child.stderr.on("data", (chunk: Buffer) => (stderr += chunk.toString()));
    child.once("exit", (code) => resolve({ code, stdout, stderr }));
  });
}

for (const model of KNOWN_MODELS) {
  const owner = model.repository.replace("/", "--");
  const snapshot = SNAPSHOTS === undefined ? undefined : join(SNAPSHOTS, owner, model.pinnedRevision);
  const templatePresent = existsSync(join(templatesDir, owner, model.pinnedRevision, "template.json"));
  const skip =
    SNAPSHOTS === undefined
      ? "set GLINER_RUNNER_TEST_SNAPSHOTS_DIR to run"
      : snapshot === undefined || !existsSync(join(snapshot, "model.safetensors"))
        ? `no snapshot for ${model.modelId} in GLINER_RUNNER_TEST_SNAPSHOTS_DIR`
        : !templatePresent
          ? `no template for ${model.modelId}`
          : ONLY.length > 0 && !ONLY.includes(model.modelId)
            ? "not selected by GLINER_RUNNER_TEST_REAL_MODELS"
            : false;

  test(`real ${model.modelId}: models install is byte-identical to the canonical artifact`, { skip, timeout: 30 * 60_000 }, async (t) => {
    const template = await loadTemplate(templatesDir, model, model.pinnedRevision);
    assert.ok(template);
    const files: Record<string, { path: string }> = {};
    for (const source of template.sourceFiles) {
      files[source.path] = { path: join(snapshot as string, source.path) };
    }
    const hub = await startMockHf({ repository: model.repository, revision: model.pinnedRevision, files });
    const modelsDir = mkdtempSync(join(tmpdir(), `gliner-real-${model.modelId}-`));
    t.after(async () => {
      await hub.close();
      rmSync(modelsDir, { recursive: true, force: true });
    });

    const started = Date.now();
    const result = await run(["models", "install", model.modelId, "--yes", "--models-dir", modelsDir], {
      HF_ENDPOINT: hub.endpoint,
    });
    assert.equal(result.code, 0, result.stderr);
    const { path } = JSON.parse(result.stdout.trim()) as { path: string };
    const elapsed = (Date.now() - started) / 1000;
    process.stderr.write(`# real ${model.modelId}: install took ${elapsed.toFixed(1)}s\n`);

    assert.deepEqual(
      readdirSync(path).sort(),
      [...template.outputs.map((output) => output.path), "manifest.json"].sort(),
    );
    for (const output of template.outputs) {
      const actual = await sha256OfFile(join(path, output.path));
      assert.deepEqual(
        { path: output.path, bytes: actual.bytes, sha256: actual.sha256 },
        { path: output.path, bytes: output.size, sha256: output.sha256 },
      );
    }
    const manifest = template.templateFiles.find((file) => file.output === "manifest.json");
    assert.ok(manifest);
    const written = await sha256OfFile(join(path, "manifest.json"));
    assert.equal(written.sha256, manifest.sha256);
    assert.equal(written.bytes, manifest.size);

    // The strict catalog accepts it too (manifest schema, sizes and SHA-256 of every listed file).
    const resolved = await new OnnxArtifactCatalog(modelsDir).resolve(model.modelId);
    assert.equal(resolved.path, path);
    assert.deepEqual(readdirSync(join(modelsDir, "onnx", owner)).sort(), [model.pinnedRevision, `${model.pinnedRevision}.gliner-runner.json`]);
  });

  test(`real ${model.modelId}: offline models prepare discovers the local snapshot byte-identically`, { skip, timeout: 30 * 60_000 }, async (t) => {
    const template = await loadTemplate(templatesDir, model, model.pinnedRevision);
    assert.ok(template);
    const modelsDir = mkdtempSync(join(tmpdir(), `gliner-real-prepare-${model.modelId}-`));
    t.after(() => rmSync(modelsDir, { recursive: true, force: true }));
    // The candidate directory layout `<root>/<owner>--<name>/<revision>` is found through GLINER_RUNNER_SNAPSHOT_DIRS;
    // the endpoint is unreachable, so nothing can use the network.
    const result = await run(["models", "prepare", model.modelId, "--models-dir", modelsDir], {
      GLINER_RUNNER_SNAPSHOT_DIRS: SNAPSHOTS as string,
      HF_ENDPOINT: "http://127.0.0.1:1",
    });
    assert.equal(result.code, 0, result.stderr);
    const { path } = JSON.parse(result.stdout.trim()) as { path: string };
    for (const output of template.outputs) {
      const actual = await sha256OfFile(join(path, output.path));
      assert.deepEqual({ path: output.path, sha256: actual.sha256 }, { path: output.path, sha256: output.sha256 });
    }
    const record = JSON.parse(readFileSync(join(modelsDir, "onnx", owner, `${model.pinnedRevision}.gliner-runner.json`), "utf8"));
    assert.equal(record.state, "prepared");
    assert.equal(record.source.location, snapshot);
    assert.ok(record.source.files.every((file: { verified: boolean }) => file.verified));
  });
}

// The machine's own Hugging Face cache, when it has this repository: reported, and used only if it is the pinned revision.
const hubCache = join(process.env["HF_HUB_CACHE"] ?? join(process.env["HF_HOME"] ?? join(homedir(), ".cache", "huggingface"), "hub"), "models--fastino--GLiNER2.5-Decide");
test(
  "real local Hugging Face cache: scan reports the cached revisions; the pinned one is prepared offline",
  { skip: existsSync(join(hubCache, "snapshots")) && existsSync(join(templatesDir, "fastino--GLiNER2.5-Decide")) ? false : "no local HF cache for fastino/GLiNER2.5-Decide or no template", timeout: 30 * 60_000 },
  async (t) => {
    const model = KNOWN_MODELS[0] as (typeof KNOWN_MODELS)[number];
    const modelsDir = mkdtempSync(join(tmpdir(), "gliner-real-hfcache-"));
    t.after(() => rmSync(modelsDir, { recursive: true, force: true }));
    const cached = readdirSync(join(hubCache, "snapshots"));
    const scan = await run(["models", "scan", "--model", "decide-340m", "--json", "--models-dir", modelsDir], {
      HOME: homedir(),
      ...(process.env["HF_HOME"] === undefined ? {} : { HF_HOME: process.env["HF_HOME"] }),
      ...(process.env["HF_HUB_CACHE"] === undefined ? {} : { HF_HUB_CACHE: process.env["HF_HUB_CACHE"] }),
      HF_ENDPOINT: "http://127.0.0.1:1",
    });
    assert.equal(scan.code, 0, scan.stderr);
    const entry = JSON.parse(scan.stdout).models[0];
    const found = entry.candidates.filter((candidate: { location: string }) => candidate.location.startsWith(hubCache));
    process.stderr.write(`# local HF cache revisions: ${cached.join(", ")}; pinned ${model.pinnedRevision}; candidates: ${JSON.stringify(found)}\n`);
    assert.equal(found.length, cached.length);
    if (cached.includes(model.pinnedRevision)) {
      assert.equal(found.find((candidate: { revision: string }) => candidate.revision === model.pinnedRevision).verified, true);
      const result = await run(["models", "prepare", "decide-340m", "--models-dir", modelsDir], {
        HOME: homedir(),
        ...(process.env["HF_HOME"] === undefined ? {} : { HF_HOME: process.env["HF_HOME"] }),
        HF_ENDPOINT: "http://127.0.0.1:1",
      });
      assert.equal(result.code, 0, result.stderr);
    } else {
      // Not the pinned revision: it must be refused, whatever its contents.
      assert.ok(found.every((candidate: { verified: boolean; reason: string }) => !candidate.verified && /is not the pinned revision/.test(candidate.reason)));
      assert.equal(entry.selected === null || !String(entry.selected).startsWith(hubCache), true);
    }
  },
);
