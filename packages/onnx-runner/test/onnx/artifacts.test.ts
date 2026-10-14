import assert from "node:assert/strict";
import {
  appendFileSync,
  copyFileSync,
  mkdirSync,
  readFileSync,
  renameSync,
  rmSync,
  statSync,
  symlinkSync,
  utimesSync,
  writeFileSync,
} from "node:fs";
import { join } from "node:path";
import { test } from "node:test";

import {
  ModelDownloadRequiredError,
  ModelIntegrityError,
  createServer,
  defaultSettings,
} from "../../src/index.js";
import { OnnxArtifactCatalog, artifactDirectory, driftOrderOfMagnitude } from "../../src/onnx/artifacts.js";
import { OnnxBackendFactory } from "../../src/onnx/backend.js";
import { createComponents } from "../../src/onnx/factory.js";
import { parseManifest } from "../../src/onnx/manifest.js";
import { wireRequest } from "../helpers.js";
import { MODEL, manifestFor, writeSyntheticArtifact } from "./synthetic.js";

const PINNED = MODEL.pinnedRevision;

function rewriteManifest(directory: string, edit: (manifest: Record<string, unknown>) => void): void {
  const path = join(directory, "manifest.json");
  const manifest = JSON.parse(readFileSync(path, "utf8")) as Record<string, unknown>;
  edit(manifest);
  writeFileSync(path, JSON.stringify(manifest));
}

test("artifact directory follows <models>/onnx/<owner>--<name>/<revision>", () => {
  assert.equal(
    artifactDirectory("/m", MODEL.repository, PINNED),
    `/m/onnx/fastino--GLiNER2.5-Decide/${PINNED}`,
  );
});

test("a valid artifact is listed, resolved, and characterized from its recorded parity", async () => {
  const { modelsDir, directory } = writeSyntheticArtifact();
  const catalog = new OnnxArtifactCatalog(modelsDir);
  const inventory = await catalog.inventory();
  assert.equal(inventory.destination, modelsDir);
  const entry = inventory.models.find((model) => model.modelId === MODEL.modelId);
  assert.deepEqual(entry?.localRevisions, [PINNED]);
  assert.deepEqual(
    inventory.models.filter((model) => model.localRevisions.length > 0).map((model) => model.modelId),
    ["decide-340m"],
  );
  assert.deepEqual(catalog.installedModelIds(), ["decide-340m"]);

  for (const reference of ["decide-340m", MODEL.repository, `decide-340m@${PINNED}`, `${MODEL.repository}@${PINNED}`]) {
    const resolved = await catalog.resolve(reference);
    assert.equal(resolved.path, directory);
    assert.equal(resolved.modelId, "decide-340m");
    assert.equal(resolved.revision, PINNED);
    assert.equal(resolved.parameterCount, MODEL.parameterCount);
  }

  const profile = catalog.characterize("decide-340m", { device: "cpu", precision: "fp32" });
  assert.equal(profile?.observedMaxAbsScoreDrift, 5.2e-6);
  assert.equal(profile?.driftOrderOfMagnitude, "1e-6");
  assert.match(profile?.scope ?? "", /19-row synthetic-v1/);
  assert.equal(catalog.characterize("decide-340m", { device: "cpu", precision: "fp16" }), undefined);
  assert.equal(catalog.characterize("decide-1b", { device: "cpu", precision: "fp32" }), undefined);
  assert.equal(driftOrderOfMagnitude(0), "0");
  assert.equal(driftOrderOfMagnitude(3.57e-7), "1e-7");
});

test("a missing artifact asks for installation; unknown models and revisions are integrity errors", async () => {
  const { modelsDir } = writeSyntheticArtifact();
  const catalog = new OnnxArtifactCatalog(modelsDir);
  await assert.rejects(catalog.resolve("decide-1b"), (error: unknown) => {
    assert.ok(error instanceof ModelDownloadRequiredError);
    assert.equal(error.destination, modelsDir);
    return true;
  });
  await assert.rejects(catalog.resolve("nope"), ModelIntegrityError);
  await assert.rejects(catalog.resolve("decide-340m@main"), /invalid revision/);
  // Another well-formed revision is not installed.
  await assert.rejects(catalog.resolve(`decide-340m@${"1".repeat(40)}`), ModelDownloadRequiredError);
  // A directory without a manifest is not an installation.
  const empty = writeSyntheticArtifact();
  rmSync(join(empty.directory, "manifest.json"));
  await assert.rejects(new OnnxArtifactCatalog(empty.modelsDir).resolve("decide-340m"), ModelDownloadRequiredError);
});

test("integrity failures: altered content, size, missing and unlisted files", async () => {
  const altered = writeSyntheticArtifact();
  const original = readFileSync(join(altered.directory, "model.onnx"));
  const flipped = Buffer.from(original);
  flipped[flipped.length - 1] = (flipped[flipped.length - 1] as number) ^ 0xff;
  writeFileSync(join(altered.directory, "model.onnx"), flipped);
  const catalog = new OnnxArtifactCatalog(altered.modelsDir);
  // Listing is a cheap size check and cannot see a same-size change; loading hashes every file.
  assert.deepEqual((await catalog.inventory()).models[0]?.localRevisions, [PINNED]);
  await assert.rejects(catalog.resolve("decide-340m"), /model\.onnx does not match the manifest/);
  assert.deepEqual(catalog.installedModelIds(), []);

  const resized = writeSyntheticArtifact();
  appendFileSync(join(resized.directory, "tokenizer.json"), " ");
  const resizedCatalog = new OnnxArtifactCatalog(resized.modelsDir);
  assert.deepEqual((await resizedCatalog.inventory()).models[0]?.localRevisions, []);
  await assert.rejects(resizedCatalog.resolve("decide-340m"), /tokenizer\.json has \d+ bytes, manifest records \d+/);

  const missing = writeSyntheticArtifact();
  rmSync(join(missing.directory, "tokenizer_config.json"));
  await assert.rejects(
    new OnnxArtifactCatalog(missing.modelsDir).resolve("decide-340m"),
    /listed file tokenizer_config\.json is missing/,
  );

  const unlisted = writeSyntheticArtifact();
  writeFileSync(join(unlisted.directory, "extra.bin"), "x");
  await assert.rejects(
    new OnnxArtifactCatalog(unlisted.modelsDir).resolve("decide-340m"),
    /unlisted files present: extra\.bin/,
  );
  const nested = writeSyntheticArtifact();
  mkdirSync(join(nested.directory, "sub"));
  await assert.rejects(new OnnxArtifactCatalog(nested.modelsDir).resolve("decide-340m"), /unlisted files present: sub/);
});

test("verification is cached per path, size and mtime for the process lifetime", async () => {
  const { modelsDir, directory } = writeSyntheticArtifact();
  const catalog = new OnnxArtifactCatalog(modelsDir);
  const target = join(directory, "tokenizer_config.json");
  // Whole-millisecond mtime so that restoring it with utimes is exact.
  utimesSync(target, new Date(1_700_000_000_000), new Date(1_700_000_000_000));
  await catalog.resolve("decide-340m");

  // Same size and mtime: the cached verification stands, so no re-hash happens.
  const before = statSync(target);
  const content = readFileSync(target);
  const changed = Buffer.from(content);
  changed[0] = changed[0] === 0x7b ? 0x20 : 0x7b;
  writeFileSync(target, changed);
  utimesSync(target, before.atime, before.mtime);
  assert.equal((await catalog.resolve("decide-340m")).path, directory);

  // A new mtime invalidates the cache and the content is verified again.
  utimesSync(target, before.atime, new Date(before.mtimeMs + 5000));
  await assert.rejects(catalog.resolve("decide-340m"), ModelIntegrityError);
  // The failure is cached too, and restoring the content under a new mtime recovers.
  await assert.rejects(catalog.resolve("decide-340m"), ModelIntegrityError);
  writeFileSync(target, content);
  utimesSync(target, before.atime, new Date(before.mtimeMs + 9000));
  assert.equal((await catalog.resolve("decide-340m")).path, directory);

  // Without a remote catalog, refresh fails (POST /v1/models/refresh then answers 502).
  await assert.rejects(catalog.refresh(), /no remote model catalog/);
});

test("concurrent first resolves share one verification", async () => {
  const { modelsDir } = writeSyntheticArtifact();
  const catalog = new OnnxArtifactCatalog(modelsDir);
  const results = await Promise.all(Array.from({ length: 5 }, () => catalog.resolve("decide-340m")));
  assert.equal(new Set(results.map((result) => result.path)).size, 1);
});

const model = MODEL;
const goodManifest = (): Record<string, unknown> =>
  JSON.parse(JSON.stringify(manifestFor(model, { "gliner2.json": "a", "model.onnx": "b", "tokenizer.json": "c", "tokenizer_config.json": "d" })));

const mutations: Array<[string, (manifest: Record<string, any>) => void, RegExp]> = [
  ["format_version", (m) => (m.format_version = 2), /format_version/],
  ["kind", (m) => (m.kind = "other"), /kind/],
  ["model_id", (m) => (m.model_id = "decide-1b"), /model_id/],
  ["repository", (m) => (m.repository = "evil/model"), /repository/],
  ["revision", (m) => (m.revision = "1".repeat(40)), /revision/],
  ["parameter_count", (m) => (m.parameter_count = 1), /parameter_count/],
  ["operations", (m) => (m.operations = ["classify", "extract_entities"]), /operations/],
  ["precision", (m) => (m.precision = "fp16"), /precision/],
  ["quantization", (m) => (m.quantization = "int8"), /quantization/],
  ["provider", (m) => (m.provider = "CoreMLExecutionProvider"), /provider/],
  ["opset", (m) => (m.opset = 17), /opset/],
  ["opset imports", (m) => (m.opset_imports = [{ domain: "ai.onnx", version: 17 }]), /opset_imports/],
  ["preprocessing", (m) => (m.preprocessing = "other.json"), /preprocessing/],
  ["graph path", (m) => (m.graph.path = "other.onnx"), /graph\.path/],
  ["graph inputs", (m) => m.graph.inputs.pop(), /graph\.inputs/],
  ["graph input type", (m) => (m.graph.inputs[0].type = "tensor(int32)"), /graph\.inputs/],
  ["graph outputs", (m) => (m.graph.outputs[0].name = "scores"), /graph\.outputs/],
  ["external data not listed", (m) => (m.graph.external_data = ["model.onnx.data"]), /external data model\.onnx\.data is not a listed file/],
  ["missing required file", (m) => (m.files = m.files.filter((f: any) => f.path !== "tokenizer.json")), /required file tokenizer\.json/],
  ["duplicate file", (m) => m.files.push({ ...m.files[0] }), /more than once/],
  ["path traversal", (m) => (m.files[0].path = "../x"), /unacceptable file path/],
  ["manifest listed", (m) => (m.files[0].path = "manifest.json"), /unacceptable file path/],
  ["bad digest", (m) => (m.files[0].sha256 = "XYZ"), /sha256/],
  ["bad size", (m) => (m.files[0].size = -1), /size/],
  ["file extra field", (m) => (m.files[0].mode = 0o777), /unexpected fields/],
  ["empty files", (m) => (m.files = []), /files must be a non-empty list/],
  ["no parity", (m) => delete m.parity, /parity evidence is missing/],
  ["parity over tolerance", (m) => (m.parity.max_abs_logit_difference = 0.5), /exceeds its absolute tolerance/],
];

test("parseManifest accepts the good manifest and tolerates unknown provenance fields", () => {
  const manifest = goodManifest();
  manifest["exporter"] = { torch: "x" };
  const parsed = parseManifest(manifest, model, PINNED);
  assert.equal(parsed.files.length, 4);
  assert.equal(parsed.parity.maxAbsLogitDifference, 5e-6);
  assert.throws(() => parseManifest("nope", model, PINNED), ModelIntegrityError);
  assert.throws(() => parseManifest(goodManifest(), model, "2".repeat(40)), /not the pinned revision/);
});

for (const [name, mutate, message] of mutations) {
  test(`parseManifest rejects ${name}`, () => {
    const manifest = goodManifest();
    mutate(manifest);
    assert.throws(
      () => parseManifest(manifest, model, PINNED),
      (error: unknown) => error instanceof ModelIntegrityError && message.test(error.message),
    );
  });
}

test("a directory named for a non-pinned revision fails validation even if its manifest is self-consistent", async () => {
  const { modelsDir, directory } = writeSyntheticArtifact();
  const other = "3".repeat(40);
  const moved = join(modelsDir, "onnx", "fastino--GLiNER2.5-Decide", other);
  const { cpSync } = await import("node:fs");
  cpSync(directory, moved, { recursive: true });
  const catalog = new OnnxArtifactCatalog(modelsDir);
  await assert.rejects(catalog.resolve(`decide-340m@${other}`), /not the pinned revision|revision must be/);
  assert.deepEqual((await catalog.inventory()).models[0]?.localRevisions, [PINNED]);
});

test("the factory advertises exactly the installed curated models and only cpu/fp32", async () => {
  const { modelsDir } = writeSyntheticArtifact();
  const components = await createComponents({ ...defaultSettings(), modelsDir });
  const [factory] = components.backends;
  assert.equal(factory?.name, "onnx");
  assert.deepEqual(factory?.capabilities(), {
    backend: "onnx",
    operations: ["classify"],
    precisions: ["fp32"],
    devices: ["cpu"],
    dynamic_batching: true,
    precision_profiles: [{ device: "cpu", precision: "fp32" }],
    validated_models: ["decide-340m"],
  });
  const resolved = await components.catalog.resolve("decide-340m");
  assert.throws(() => factory?.create(resolved, "mps", "fp32"), /only cpu\/fp32/);
  assert.throws(() => factory?.create(resolved, "cpu", "fp16"), /only cpu\/fp32/);
  const empty = new OnnxBackendFactory(new OnnxArtifactCatalog(writeSyntheticArtifact().modelsDir));
  assert.deepEqual(empty.capabilities().validated_models, []);
});

test("HTTP surface: 409 model_download_required, 422 model_integrity, and capabilities", async () => {
  const { modelsDir, directory } = writeSyntheticArtifact();
  const settings = { ...defaultSettings(), host: "127.0.0.1", port: 0, modelsDir };
  const server = createServer({ ...(await createComponents(settings)), settings });
  const address = await server.listen();
  const base = `http://127.0.0.1:${address.port}`;
  const post = async (body: unknown): Promise<{ status: number; body: any }> => {
    const response = await fetch(`${base}/v1/infer`, {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify(body),
    });
    return { status: response.status, body: await response.json() };
  };
  try {
    const missing = await post(wireRequest({ model: "decide-1b" }));
    assert.equal(missing.status, 409);
    assert.equal(missing.body.detail.code, "model_download_required");
    assert.equal(missing.body.detail.event.destination, modelsDir);

    const unsupported = await post(wireRequest({ precision: "fp16" }));
    assert.equal(unsupported.status, 409);
    assert.equal(unsupported.body.detail.code, "unsupported_configuration");

    const pytorch = await post(wireRequest({ backend: "pytorch" }));
    assert.equal(pytorch.status, 409);
    assert.match(pytorch.body.detail.message, /no fallback/);

    appendFileSync(join(directory, "model.onnx"), "x");
    const broken = await post(wireRequest());
    assert.equal(broken.status, 422);
    assert.equal(broken.body.detail.code, "model_integrity");

    const models = (await (await fetch(`${base}/v1/models`)).json()) as any;
    assert.equal(models.models[0].available_locally, false);
    const capabilities = (await (await fetch(`${base}/v1/capabilities`)).json()) as any[];
    assert.deepEqual(capabilities[0].validated_models, []);
  } finally {
    await server.close();
  }
});

test("GET /v1/models reports the recorded parity evidence for cpu/fp32", async () => {
  const { modelsDir } = writeSyntheticArtifact();
  const settings = { ...defaultSettings(), host: "127.0.0.1", port: 0, modelsDir };
  const server = createServer({ ...(await createComponents(settings)), settings });
  const address = await server.listen();
  try {
    const models = (await (await fetch(`http://127.0.0.1:${address.port}/v1/models`)).json()) as any;
    const entry = models.models.find((model: any) => model.model_id === "decide-340m");
    assert.equal(entry.available_locally, true);
    assert.equal(entry.download_required, false);
    assert.deepEqual(entry.supported_profiles, [{ device: "cpu", precision: "fp32" }]);
    const guidance = entry.profile_guidance["cpu/fp32"];
    assert.equal(guidance.observed_max_abs_score_drift, 5.2e-6);
    assert.equal(guidance.drift_order_of_magnitude, "1e-6");
    const other = models.models.find((model: any) => model.model_id === "decide-1b");
    assert.equal(other.profile_guidance["cpu/fp32"].observed_max_abs_score_drift, null);
  } finally {
    await server.close();
  }
});

test("symbolic links and other non-regular entries are refused in place of listed files", async () => {
  for (const name of ["tokenizer.json", "model.onnx", "manifest.json"]) {
    const { modelsDir, directory } = writeSyntheticArtifact();
    const target = join(modelsDir, `${name}.real`);
    // The link target has identical bytes, so only the file type distinguishes it from an intact artifact.
    renameSync(join(directory, name), target);
    symlinkSync(target, join(directory, name));
    const catalog = new OnnxArtifactCatalog(modelsDir);
    const expected = name === "manifest.json" ? /manifest\.json is not a regular file/ : new RegExp(`${name.replace(".", "\\.")} is not a regular file`);
    await assert.rejects(catalog.resolve("decide-340m"), (error: unknown) => {
      assert.ok(error instanceof ModelIntegrityError && !(error instanceof ModelDownloadRequiredError), String(error));
      assert.match(error.message, expected);
      return true;
    });
    const inventory = await catalog.inventory();
    assert.deepEqual(inventory.models.find((model) => model.modelId === "decide-340m")?.localRevisions, []);
    assert.deepEqual(catalog.installedModelIds(), []);
  }
});

test("a listed file replaced by a symlink after verification is caught on the next resolve", async () => {
  const { modelsDir, directory } = writeSyntheticArtifact();
  const catalog = new OnnxArtifactCatalog(modelsDir);
  await catalog.resolve("decide-340m");
  const copy = join(modelsDir, "tokenizer.copy");
  copyFileSync(join(directory, "tokenizer.json"), copy);
  rmSync(join(directory, "tokenizer.json"));
  symlinkSync(copy, join(directory, "tokenizer.json"));
  await assert.rejects(catalog.resolve("decide-340m"), /tokenizer\.json is not a regular file/);
});

test("a listed file that is a directory, and a symlinked artifact directory, are refused", async () => {
  const dir = writeSyntheticArtifact();
  rmSync(join(dir.directory, "tokenizer_config.json"));
  mkdirSync(join(dir.directory, "tokenizer_config.json"));
  await assert.rejects(
    new OnnxArtifactCatalog(dir.modelsDir).resolve("decide-340m"),
    /tokenizer_config\.json is not a regular file/,
  );

  const linked = writeSyntheticArtifact();
  const moved = join(linked.modelsDir, "moved");
  renameSync(linked.directory, moved);
  symlinkSync(moved, linked.directory);
  const catalog = new OnnxArtifactCatalog(linked.modelsDir);
  await assert.rejects(catalog.resolve("decide-340m"), /artifact directory is not a regular directory/);
  assert.deepEqual(
    (await catalog.inventory()).models.find((model) => model.modelId === "decide-340m")?.localRevisions,
    [],
  );
});

test("regular artifacts still verify and are cached after the lstat hardening", async () => {
  const { modelsDir } = writeSyntheticArtifact();
  const catalog = new OnnxArtifactCatalog(modelsDir);
  const first = await catalog.resolve("decide-340m");
  const second = await catalog.resolve(`decide-340m@${PINNED}`);
  assert.equal(first.path, second.path);
});
