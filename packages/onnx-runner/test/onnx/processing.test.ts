import assert from "node:assert/strict";
import { spawn } from "node:child_process";
import { existsSync, lstatSync, mkdirSync, mkdtempSync, readFileSync, readdirSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { delimiter, join } from "node:path";
import test from "node:test";

import { KNOWN_MODELS, type KnownModel } from "../../src/index.js";
import { OnnxArtifactCatalog } from "../../src/onnx/artifacts.js";
import { HuggingFaceClient } from "../../src/onnx/huggingface.js";
import { OnnxModelProvisioner } from "../../src/onnx/install.js";
import {
  RECORD_KIND,
  buildRecord,
  readProcessingRecord,
  recordPath,
  runnerVersion,
  writeProcessingRecord,
} from "../../src/onnx/processing-record.js";
import { loadTemplate } from "../../src/onnx/templates.js";
import { CLI } from "../helpers.js";
import { startMockHf } from "./hf-mock.js";
import { buildTemplateFixture, sha256, writeHfCache, writeSnapshotDir, type TemplateFixture } from "./template-fixture.js";

const MODEL = KNOWN_MODELS[0] as KnownModel;
const PIN = MODEL.pinnedRevision;
const SLUG = "fastino--GLiNER2.5-Decide";
const logs: string[] = [];

function provisioner(fixture: TemplateFixture, env: NodeJS.ProcessEnv = {}, extra: Partial<ConstructorParameters<typeof OnnxModelProvisioner>[0]> = {}) {
  return new OnnxModelProvisioner({
    modelsDir: join(fixture.root, "models"),
    templatesDir: fixture.templatesDir,
    env: { HOME: join(fixture.root, "home"), ...env },
    huggingface: new HuggingFaceClient({ endpoint: "http://127.0.0.1:1", token: undefined, maxAttempts: 1 }),
    ...extra,
  });
}
const context = () => ({ signal: new AbortController().signal, log: (message: string) => logs.push(message) });
const modelsDir = (fixture: TemplateFixture): string => join(fixture.root, "models");
const parent = (fixture: TemplateFixture): string => join(modelsDir(fixture), "onnx", SLUG);

test("install uses a verified local snapshot (no network) and records what was applied", async () => {
  const fixture = buildTemplateFixture();
  const hf = join(fixture.root, "hf");
  writeHfCache(hf, MODEL.repository, PIN, fixture.sources);
  const hub = await startMockHf({ repository: MODEL.repository, revision: PIN, files: fixture.sources });
  try {
    const subject = provisioner(fixture, { HF_HUB_CACHE: hf }, {
      huggingface: new HuggingFaceClient({ endpoint: hub.endpoint, token: undefined }),
    });
    const path = await subject.install("decide-340m", PIN, context());
    assert.equal(hub.requests.length, 0, "a local snapshot means no download");
    for (const [name, content] of Object.entries(fixture.artifact)) {
      assert.ok(readFileSync(join(path, name)).equals(content), name);
    }

    const record = await readProcessingRecord(modelsDir(fixture), MODEL, PIN);
    assert.ok(record);
    assert.deepEqual(Object.keys(record).sort(), [
      "artifact",
      "format_version",
      "gliner_runner_version",
      "kind",
      "model_id",
      "processed",
      "repository",
      "revision",
      "source",
      "state",
      "steps",
      "template",
      "updated_at",
    ]);
    assert.deepEqual([record.format_version, record.kind, record.model_id, record.repository, record.revision], [1, RECORD_KIND, "decide-340m", MODEL.repository, PIN]);
    assert.equal(record.gliner_runner_version, runnerVersion());
    assert.match(runnerVersion(), /^\d+\.\d+\.\d+/);
    assert.ok(!Number.isNaN(Date.parse(record.updated_at)));
    assert.deepEqual([record.processed, record.state], [true, "prepared"]);
    assert.equal(record.source.location, join(hf, `models--${SLUG}`, "snapshots", PIN));
    assert.deepEqual(record.source.files.map((file) => [file.path, file.size, file.sha256, file.verified]), [
      ["model.safetensors", fixture.sources["model.safetensors"]?.length, sha256(fixture.sources["model.safetensors"] as Buffer), true],
      ["tokenizer.json", fixture.sources["tokenizer.json"]?.length, sha256(fixture.sources["tokenizer.json"] as Buffer), true],
      ["tokenizer_config.json", fixture.sources["tokenizer_config.json"]?.length, sha256(fixture.sources["tokenizer_config.json"] as Buffer), true],
    ]);
    assert.deepEqual(record.steps, [
      "discovered_local",
      "verified_source_sha256",
      "assembled_onnx_from_template:v1 (2 transpose, 3 identity of 5 tensors)",
      "verified_outputs_sha256",
      "published",
    ]);
    assert.equal(record.template.sha256, sha256(readFileSync(join(fixture.templateDir, "template.json"))));
    assert.equal(record.template.format_version, 1);
    assert.equal(record.artifact?.path, path);
    assert.equal(record.artifact?.manifest_sha256, sha256(fixture.manifest));
    assert.deepEqual(record.artifact?.files.map((file) => file.path), ["model.onnx", "model.onnx.data", "gliner2.json", "tokenizer.json", "tokenizer_config.json"].sort((a, b) => Object.keys(fixture.artifact).indexOf(a) - Object.keys(fixture.artifact).indexOf(b)));
    for (const file of record.artifact?.files ?? []) {
      assert.equal(file.sha256, sha256(fixture.artifact[file.path] as Buffer));
    }

    // The record lives next to the artifact, never inside it, and the catalog ignores it.
    assert.deepEqual(readdirSync(parent(fixture)).sort(), [PIN, `${PIN}.gliner-runner.json`]);
    assert.equal(recordPath(modelsDir(fixture), MODEL.repository, PIN), join(parent(fixture), `${PIN}.gliner-runner.json`));
    assert.ok(!readdirSync(path).some((name) => name.endsWith(".gliner-runner.json")));
    const catalog = new OnnxArtifactCatalog(modelsDir(fixture));
    assert.deepEqual((await catalog.inventory()).models[0]?.localRevisions, [PIN]);
    assert.equal((await catalog.resolve("decide-340m")).path, path);
    // No temporary files left behind.
    assert.ok(readdirSync(parent(fixture)).every((name) => !name.includes(".tmp-")));
  } finally {
    await hub.close();
  }
});

test("without a local snapshot the files are downloaded, and the record says so", async () => {
  const fixture = buildTemplateFixture();
  const hub = await startMockHf({ repository: MODEL.repository, revision: PIN, files: fixture.sources });
  try {
    const other = join(fixture.root, "hf");
    // A different revision in the cache is not used (and is reported as rejected by scan).
    writeHfCache(other, MODEL.repository, "5a7adf72a23b4d311abae6ce050d7f0012bb3416", fixture.sources);
    const subject = provisioner(fixture, { HF_HUB_CACHE: other }, { huggingface: new HuggingFaceClient({ endpoint: hub.endpoint, token: undefined }) });
    await subject.install("decide-340m", PIN, context());
    assert.equal(hub.requests.length, 3);
    const record = await readProcessingRecord(modelsDir(fixture), MODEL, PIN);
    assert.equal(record?.source.location, `huggingface:${hub.endpoint}`);
    assert.deepEqual(record?.steps.slice(0, 2), ["downloaded", "verified_source_sha256"]);
    assert.equal(record?.state, "prepared");
  } finally {
    await hub.close();
  }
});

test("failed attempts are recorded with the error; a later success replaces the record", async () => {
  const fixture = buildTemplateFixture();
  const files = { ...fixture.sources, "tokenizer.json": Buffer.from("tampered tokenizer") };
  const hub = await startMockHf({ repository: MODEL.repository, revision: PIN, files });
  try {
    const subject = provisioner(fixture, {}, { huggingFace: undefined, huggingface: new HuggingFaceClient({ endpoint: hub.endpoint, token: undefined, maxAttempts: 1 }) } as never);
    await assert.rejects(subject.install("decide-340m", PIN, context()), /tokenizer\.json/);
    const failed = await readProcessingRecord(modelsDir(fixture), MODEL, PIN);
    assert.deepEqual([failed?.state, failed?.processed, failed?.steps], ["failed", false, []]);
    assert.match(failed?.error ?? "", /tokenizer\.json/);
    assert.equal(failed?.artifact, undefined);
    assert.equal(failed?.source.files.every((file) => !file.verified), true);
    assert.equal(existsSync(join(parent(fixture), PIN)), false);
  } finally {
    await hub.close();
  }
  // Now a good local snapshot: the same record is replaced by a prepared one.
  const ok = writeSnapshotDir(join(fixture.root, "snap"), fixture.sources);
  await provisioner(fixture).prepare("decide-340m", { snapshot: ok, log: (message) => logs.push(message) });
  const record = await readProcessingRecord(modelsDir(fixture), MODEL, PIN);
  assert.deepEqual([record?.state, record?.processed, record?.error], ["prepared", true, undefined]);
  assert.equal(record?.source.location, ok);
});

test("prepare reports every rejected candidate, records the failure, and never overwrites", async () => {
  const fixture = buildTemplateFixture();
  const bad = writeSnapshotDir(join(fixture.root, "bad"), { ...fixture.sources, "tokenizer_config.json": Buffer.from("{}") });
  const subject = provisioner(fixture);
  await assert.rejects(
    subject.prepare("decide-340m", { snapshot: bad }),
    new RegExp(`snapshot ${bad.replace(/[.*+?^${}()|[\]\\]/g, "\\$&")} does not match fastino/GLiNER2\\.5-Decide@${PIN}: .*tokenizer_config\\.json has 2 bytes`),
  );
  const record = await readProcessingRecord(modelsDir(fixture), MODEL, PIN);
  assert.equal(record?.state, "failed");
  assert.deepEqual(record?.rejected_candidates?.map((entry) => entry.location), [bad]);

  await assert.rejects(subject.prepare("decide-340m"), /no verified local snapshot of fastino\/GLiNER2\.5-Decide@.*: no snapshot directories were found/);

  const good = writeSnapshotDir(join(fixture.root, "good"), fixture.sources);
  const path = await subject.prepare("decide-340m", { snapshot: good });
  await assert.rejects(subject.prepare("decide-340m", { snapshot: good }), /already exists; it is never overwritten/);
  // The failed second attempt did not downgrade the record of the existing artifact.
  assert.equal((await readProcessingRecord(modelsDir(fixture), MODEL, PIN))?.state, "prepared");
  assert.ok(existsSync(path));
});

test("prepare without --snapshot finds the snapshot in the standard locations", async () => {
  const fixture = buildTemplateFixture();
  const provider = join(fixture.root, "home", ".cache", "gliner-runner", "models", "providers", "huggingface", SLUG, PIN);
  writeSnapshotDir(provider, fixture.sources);
  // The Python provider's marker file is ignored content-wise: the files are verified by hash.
  writeFileSync(join(provider, ".gliner-runner.json"), JSON.stringify({ files: [], model_id: "decide-340m", repository: MODEL.repository, revision: PIN }));
  const subject = provisioner(fixture, {}, { platform: "linux" });
  const path = await subject.prepare("decide-340m");
  assert.ok(readFileSync(join(path, "model.onnx")).equals(fixture.artifact["model.onnx"] as Buffer));
  assert.equal((await readProcessingRecord(modelsDir(fixture), MODEL, PIN))?.source.location, provider);
  assert.ok(existsSync(join(provider, ".gliner-runner.json")), "the source directory is left untouched");
});

test("scan records verified_source or not_processed and never downgrades a prepared artifact", async () => {
  const fixture = buildTemplateFixture();
  const subject = provisioner(fixture);
  const empty = await subject.scan("decide-340m");
  assert.deepEqual([empty.candidates, empty.selected, empty.installed, empty.record?.state, empty.record?.processed], [[], undefined, false, "not_processed", false]);
  assert.equal(empty.record?.source.location, null);

  const hf = join(fixture.root, "hf");
  writeHfCache(hf, MODEL.repository, PIN, fixture.sources);
  writeHfCache(hf, MODEL.repository, "1".repeat(40), fixture.sources);
  const withCache = provisioner(fixture, { HF_HUB_CACHE: hf });
  const found = await withCache.scan("decide-340m");
  assert.equal(found.selected?.verified, true);
  assert.equal(found.candidates.length, 2);
  assert.equal(found.record?.state, "verified_source");
  assert.deepEqual(found.record?.steps, ["discovered_local", "verified_source_sha256"]);
  assert.equal(found.record?.rejected_candidates?.length, 1);
  assert.match(found.record?.rejected_candidates?.[0]?.reason ?? "", /is not the pinned revision/);

  await withCache.install("decide-340m", PIN, context());
  const after = await withCache.scan("decide-340m");
  assert.equal(after.installed, true);
  assert.equal(after.record?.state, "prepared");
  assert.equal(after.record?.artifact?.files.length, 5);

  // Unknown models and models without a template are reported, not crashed on.
  await assert.rejects(subject.scan("mystery"), /unknown GLiNER2\.5 model/);
  const untemplated = await subject.scan("decide-1b");
  assert.equal(untemplated.template, undefined);
  assert.deepEqual(untemplated.candidates, []);
  assert.equal(untemplated.record, undefined);
});

test("records are written atomically: concurrent writers never expose a partial file", async () => {
  const fixture = buildTemplateFixture();
  const template = await loadTemplate(fixture.templatesDir, MODEL, PIN);
  assert.ok(template);
  const dir = modelsDir(fixture);
  const path = recordPath(dir, MODEL.repository, PIN);
  const states = ["not_processed", "verified_source", "failed", "prepared"] as const;
  let corrupted = 0;
  let writing = true;
  const reader = (async () => {
    while (writing) {
      try {
        JSON.parse(readFileSync(path, "utf8"));
      } catch (error) {
        if ((error as { code?: string }).code !== "ENOENT") {
          corrupted += 1;
        }
      }
      await new Promise((resolve) => setImmediate(resolve));
    }
  })();
  await Promise.all(
    Array.from({ length: 60 }, (_, index) =>
      writeProcessingRecord(dir, buildRecord({
        template,
        state: states[index % 4] as (typeof states)[number],
        steps: Array.from({ length: 200 }, (__, step) => `step-${index}-${step}`),
        source: { location: "/x".repeat(500), verified: true },
        ...(index % 4 === 3 ? { artifactPath: "/a" } : {}),
      })),
    ),
  );
  writing = false;
  await reader;
  assert.equal(corrupted, 0);
  assert.ok(JSON.parse(readFileSync(path, "utf8")).kind === RECORD_KIND);
  assert.deepEqual(readdirSync(parent(fixture)), [`${PIN}.gliner-runner.json`]);
  assert.equal(lstatSync(path).isFile(), true);
  // Unreadable or foreign files are not records.
  writeFileSync(path, "{broken");
  assert.equal(await readProcessingRecord(dir, MODEL, PIN), undefined);
  writeFileSync(path, JSON.stringify({ kind: "other", format_version: 1 }));
  assert.equal(await readProcessingRecord(dir, MODEL, PIN), undefined);
  rmSync(path);
  assert.equal(await readProcessingRecord(dir, MODEL, PIN), undefined);
});

test("a record that cannot be written never fails the installation", async () => {
  const fixture = buildTemplateFixture();
  // A regular file where the models directory's `onnx` parent must go is the worst case for the record, so
  // use a read-only record target instead: replace the record path with a directory.
  const snap = writeSnapshotDir(join(fixture.root, "snap"), fixture.sources);
  mkdirSync(parent(fixture), { recursive: true });
  mkdirSync(recordPath(modelsDir(fixture), MODEL.repository, PIN));
  const messages: string[] = [];
  const path = await provisioner(fixture).prepare("decide-340m", { snapshot: snap, log: (message) => messages.push(message) });
  assert.ok(existsSync(join(path, "manifest.json")));
  assert.ok(messages.some((message) => message.startsWith("could not write the processing record")));
});

function cli(args: string[], env: NodeJS.ProcessEnv): Promise<{ code: number | null; stdout: string; stderr: string }> {
  return new Promise((resolve) => {
    const child = spawn(process.execPath, [CLI, ...args], { env: { PATH: process.env["PATH"] ?? "", ...env }, stdio: ["ignore", "pipe", "pipe"] });
    let stdout = "";
    let stderr = "";
    child.stdout.on("data", (chunk: Buffer) => (stdout += chunk.toString()));
    child.stderr.on("data", (chunk: Buffer) => (stderr += chunk.toString()));
    child.once("exit", (code) => resolve({ code, stdout, stderr }));
  });
}

test("models scan, prepare (discovering) and list show processed status", async () => {
  const fixture = buildTemplateFixture();
  const home = mkdtempSync(join(tmpdir(), "gliner-home-"));
  const dirs = join(fixture.root, "dirs");
  writeSnapshotDir(join(dirs, SLUG, PIN), fixture.sources);
  writeHfCache(join(fixture.root, "hub"), MODEL.repository, "2".repeat(40), fixture.sources);
  const env = {
    HOME: home,
    GLINER_RUNNER_SNAPSHOT_DIRS: [join(fixture.root, "missing"), dirs].join(delimiter),
    HF_HUB_CACHE: join(fixture.root, "hub"),
    HF_ENDPOINT: "http://127.0.0.1:1",
  };
  const args = ["--models-dir", modelsDir(fixture), "--templates-dir", fixture.templatesDir];

  const json = await cli(["models", "scan", "--model", "decide-340m", "--json", ...args], env);
  assert.equal(json.code, 0, json.stderr);
  const scan = JSON.parse(json.stdout);
  assert.equal(scan.models.length, 1);
  const entry = scan.models[0];
  assert.deepEqual([entry.model_id, entry.state, entry.processed, entry.installed, entry.template_available], ["decide-340m", "verified_source", false, false, true]);
  assert.deepEqual(entry.candidates.map((candidate: any) => [candidate.origin, candidate.verified]), [
    ["GLINER_RUNNER_SNAPSHOT_DIRS", true],
    ["HF_HUB_CACHE", false],
  ]);
  assert.equal(entry.selected, join(dirs, SLUG, PIN));

  const text = await cli(["models", "scan", ...args], env);
  assert.equal(text.code, 0);
  assert.match(text.stdout, /decide-340m \(fastino\/GLiNER2\.5-Decide@9d1bfb848cd1\)/);
  assert.match(text.stdout, /status: verified_source; artifact not installed; template available/);
  assert.match(text.stdout, /verified {2}.*\[GLINER_RUNNER_SNAPSHOT_DIRS\]/);
  assert.match(text.stdout, /rejected {2}.*\[HF_HUB_CACHE\] {2}revision 2{40} is not the pinned revision/);
  assert.match(text.stdout, /decide-1b .*\n {2}status: not_processed; artifact not installed; template missing/);
  assert.equal((await cli(["models", "scan", "--model", "mystery", ...args], env)).code, 2);

  // Install finds the local snapshot: no network (HF_ENDPOINT is unreachable) and no approval needed.
  const install = await cli(["models", "install", "decide-340m", ...args], env);
  assert.equal(install.code, 0, install.stderr);
  assert.match(install.stderr, /using the verified local snapshot/);

  const list = JSON.parse((await cli(["models", "list", ...args], env)).stdout);
  const first = list.models[0];
  assert.deepEqual([first.installed, first.processed, first.state], [true, true, "prepared"]);
  assert.equal(first.record.source.location, join(dirs, SLUG, PIN));
  assert.equal(first.record.artifact.manifest_sha256, sha256(fixture.manifest));
  assert.equal(list.models[1].processed, false);

  // The strict catalog sees only the artifact; the sibling record does not disturb it.
  assert.deepEqual((await new OnnxArtifactCatalog(modelsDir(fixture)).inventory()).models[0]?.localRevisions, [PIN]);
  const prepared = await cli(["models", "scan", "--json", ...args], env);
  assert.equal(JSON.parse(prepared.stdout).models[0].state, "prepared");
});

test("models prepare without --snapshot works offline from the discovered snapshot", async () => {
  const fixture = buildTemplateFixture();
  const home = mkdtempSync(join(tmpdir(), "gliner-home-"));
  writeHfCache(join(home, ".cache", "huggingface", "hub"), MODEL.repository, PIN, fixture.sources);
  const result = await cli(["models", "prepare", "decide-340m", "--models-dir", modelsDir(fixture), "--templates-dir", fixture.templatesDir], {
    HOME: home,
    HF_ENDPOINT: "http://127.0.0.1:1",
  });
  assert.equal(result.code, 0, result.stderr);
  assert.match(result.stderr, /preparing from .*huggingface\/hub\/models--fastino--GLiNER2\.5-Decide\/snapshots\/9d1bfb84/);
  const { path } = JSON.parse(result.stdout.trim());
  assert.ok(readFileSync(join(path, "model.onnx.data")).equals(fixture.artifact["model.onnx.data"] as Buffer));
});
