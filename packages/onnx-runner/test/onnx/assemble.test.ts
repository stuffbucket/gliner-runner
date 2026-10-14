import assert from "node:assert/strict";
import { existsSync, readFileSync, readdirSync, statSync, writeFileSync, mkdirSync } from "node:fs";
import { join } from "node:path";
import { test } from "node:test";

import {
  ArtifactExistsError,
  AssembleError,
  InsufficientSpaceError,
  assembleArtifact,
  transposeWords,
  type Progress,
} from "../../src/onnx/assemble.js";
import { OnnxArtifactCatalog, artifactDirectory } from "../../src/onnx/artifacts.js";
import { loadTemplate, type OnnxTemplate } from "../../src/onnx/templates.js";
import { RequestCancelledError } from "../../src/index.js";
import { buildTemplateFixture, randomWords, safetensorsBytes, sha256, type FixtureTensor, type TemplateFixture } from "./template-fixture.js";

async function loaded(fixture: TemplateFixture): Promise<OnnxTemplate> {
  const template = await loadTemplate(fixture.templatesDir, fixture.model, fixture.model.pinnedRevision);
  assert.ok(template);
  return template;
}

const modelsDirOf = (fixture: TemplateFixture): string => join(fixture.root, "models");
const parentOf = (fixture: TemplateFixture): string => join(modelsDirOf(fixture), "onnx", "fastino--GLiNER2.5-Decide");

function assertArtifact(fixture: TemplateFixture, directory: string): void {
  const expected = { ...fixture.artifact, "manifest.json": Buffer.from(fixture.manifest) };
  assert.deepEqual(readdirSync(directory).sort(), Object.keys(expected).sort());
  for (const [name, content] of Object.entries(expected)) {
    assert.ok(readFileSync(join(directory, name)).equals(content), `${name} differs from the reference bytes`);
  }
}

const leftovers = (fixture: TemplateFixture): string[] =>
  existsSync(parentOf(fixture)) ? readdirSync(parentOf(fixture)).filter((name) => name.startsWith(".")) : [];

test("builds the canonical artifact byte for byte (identity, transpose, alignment gaps)", async () => {
  const fixture = buildTemplateFixture();
  const template = await loaded(fixture);
  const stages: string[] = [];
  const directory = await assembleArtifact({
    template,
    snapshotDir: fixture.snapshotDir,
    modelsDir: modelsDirOf(fixture),
    onProgress: (progress) => stages.push(progress.stage),
  });
  assert.equal(directory, artifactDirectory(modelsDirOf(fixture), fixture.model.repository, fixture.model.pinnedRevision));
  assertArtifact(fixture, directory);
  assert.deepEqual(leftovers(fixture), []);
  assert.equal(stages.at(-1), "finalize");
  assert.ok(stages.includes("verify-source") && stages.includes("weights") && stages.includes("verify-output"));

  // The artifact the assembler built is accepted by the strict catalog (manifest, sizes and digests).
  const resolved = await new OnnxArtifactCatalog(modelsDirOf(fixture)).resolve("decide-340m");
  assert.equal(resolved.path, directory);

  // Tensors start at their recorded, 64-aligned offsets and the padding between them is zero.
  const data = readFileSync(join(directory, "model.onnx.data"));
  let position = 0;
  for (const tensor of template.data.tensors) {
    assert.equal(tensor.offset % 64, 0);
    assert.ok(data.subarray(position, tensor.offset).every((byte) => byte === 0));
    position = tensor.offset + tensor.length;
  }
  assert.equal(position, data.length);
  assert.ok(template.data.tensors.some((tensor, index) => index > 0 && tensor.offset > (template.data.tensors[index - 1] as { offset: number; length: number }).offset + (template.data.tensors[index - 1] as { length: number }).length));
});

test("tiny copy blocks and transpose bands (many partial blocks) give identical bytes", async () => {
  const fixture = buildTemplateFixture();
  const directory = await assembleArtifact({
    template: await loaded(fixture),
    snapshotDir: fixture.snapshotDir,
    modelsDir: modelsDirOf(fixture),
    limits: { copyBlockBytes: 7, transposeBandBytes: 100 },
  });
  assertArtifact(fixture, directory);
});

for (const alignment of [1, 4, 256, 4096]) {
  test(`alignment ${alignment}`, async () => {
    const fixture = buildTemplateFixture({ alignment });
    const template = await loaded(fixture);
    const directory = await assembleArtifact({ template, snapshotDir: fixture.snapshotDir, modelsDir: modelsDirOf(fixture) });
    assertArtifact(fixture, directory);
    for (const tensor of template.data.tensors) {
      assert.equal(tensor.offset % alignment, 0);
    }
  });
}

test("non-square, single-row, single-column and empty transposes", async () => {
  const tensors: FixtureTensor[] = [
    { name: "row", shape: [1, 9], transform: "transpose" },
    { name: "col", shape: [9, 1], transform: "transpose" },
    { name: "empty", shape: [0, 5], transform: "transpose" },
    { name: "square", shape: [8, 8], transform: "transpose" },
    { name: "odd", shape: [33, 65], transform: "transpose" },
  ];
  const fixture = buildTemplateFixture({ tensors });
  for (const limits of [undefined, { transposeBandBytes: 1 }, { transposeBandBytes: 300 }]) {
    const modelsDir = join(fixture.root, `models-${JSON.stringify(limits)}`);
    const directory = await assembleArtifact({
      template: await loaded(fixture),
      snapshotDir: fixture.snapshotDir,
      modelsDir,
      limits,
    });
    assertArtifact(fixture, directory);
  }
});

test("transposeWords moves raw words", () => {
  const source = Uint32Array.from([1, 2, 3, 4, 5, 6]);
  const target = new Uint32Array(6);
  transposeWords(source, target, 2, 3);
  assert.deepEqual([...target], [1, 4, 2, 5, 3, 6]);
  const big = randomWords(100 * 37, 9);
  const out = new Uint32Array(big.length);
  transposeWords(big, out, 100, 37);
  assert.equal(out[5 * 100 + 7], big[7 * 37 + 5]);
});

test("a source that does not match the pinned revision is refused before anything is written", async () => {
  const fixture = buildTemplateFixture();
  writeFileSync(join(fixture.snapshotDir, "tokenizer.json"), "{}");
  await assert.rejects(
    assembleArtifact({ template: await loaded(fixture), snapshotDir: fixture.snapshotDir, modelsDir: modelsDirOf(fixture) }),
    /snapshot file tokenizer\.json does not match the pinned revision/,
  );
  assert.deepEqual(leftovers(fixture), []);
  assert.equal(existsSync(artifactDirectory(modelsDirOf(fixture), fixture.model.repository, fixture.model.pinnedRevision)), false);

  const missing = buildTemplateFixture();
  await assert.rejects(
    assembleArtifact({ template: await loaded(missing), snapshotDir: join(missing.root, "nowhere"), modelsDir: modelsDirOf(missing) }),
    /snapshot is missing model\.safetensors|snapshot is missing/,
  );
});

test("hash mismatch fails closed even when sources are trusted (outputs are re-verified)", async () => {
  const fixture = buildTemplateFixture();
  // Same size, one flipped bit in the last tensor: the data file's SHA-256 can no longer match.
  const weights = Buffer.from(fixture.sources["model.safetensors"] as Buffer);
  weights[weights.length - 1] = (weights[weights.length - 1] as number) ^ 1;
  writeFileSync(join(fixture.snapshotDir, "model.safetensors"), weights);
  await assert.rejects(
    assembleArtifact({
      template: await loaded(fixture),
      snapshotDir: fixture.snapshotDir,
      modelsDir: modelsDirOf(fixture),
      verifySources: false,
    }),
    /prepared model\.onnx\.data does not match the template/,
  );
  assert.deepEqual(leftovers(fixture), []);
  assert.equal(existsSync(join(parentOf(fixture), fixture.model.pinnedRevision)), false);

  // Same for a template file changed after the template was verified.
  const tampered = buildTemplateFixture();
  const template = await loaded(tampered);
  writeFileSync(join(tampered.templateDir, "graph.onnx"), "tampered");
  await assert.rejects(
    assembleArtifact({ template, snapshotDir: tampered.snapshotDir, modelsDir: modelsDirOf(tampered) }),
    /prepared model\.onnx does not match the template/,
  );
  assert.deepEqual(leftovers(tampered), []);
});

test("weights that do not have the recorded dtype or shape are refused up front", async () => {
  const fixture = buildTemplateFixture();
  const template = await loaded(fixture);
  const wrongShape = safetensorsBytes(
    template.data.tensors.map((tensor) => ({
      name: tensor.source,
      shape: tensor.source === "bias" ? [1, 7] : [...tensor.sourceShape],
      data: Buffer.alloc(tensor.length),
    })),
  );
  writeFileSync(join(fixture.snapshotDir, "model.safetensors"), wrongShape);
  await assert.rejects(
    assembleArtifact({ template, snapshotDir: fixture.snapshotDir, modelsDir: modelsDirOf(fixture), verifySources: false }),
    /the template requires F32\[7\]/,
  );
  const wrongType = safetensorsBytes(
    template.data.tensors.map((tensor) => ({ name: tensor.source, shape: [...tensor.sourceShape], data: Buffer.alloc(tensor.length / 2), dtype: "F16" })),
  );
  writeFileSync(join(fixture.snapshotDir, "model.safetensors"), wrongType);
  await assert.rejects(
    assembleArtifact({ template, snapshotDir: fixture.snapshotDir, modelsDir: modelsDirOf(fixture), verifySources: false }),
    AssembleError,
  );
  writeFileSync(join(fixture.snapshotDir, "model.safetensors"), "not safetensors at all");
  await assert.rejects(
    assembleArtifact({ template, snapshotDir: fixture.snapshotDir, modelsDir: modelsDirOf(fixture), verifySources: false }),
    /invalid model\.safetensors/,
  );
  assert.deepEqual(leftovers(fixture), []);
});

test("never overwrites an existing artifact", async () => {
  const fixture = buildTemplateFixture();
  const template = await loaded(fixture);
  const options = { template, snapshotDir: fixture.snapshotDir, modelsDir: modelsDirOf(fixture) };
  const directory = await assembleArtifact(options);
  const before = sha256(readFileSync(join(directory, "model.onnx.data")));
  writeFileSync(join(directory, "marker"), "keep");
  await assert.rejects(assembleArtifact(options), ArtifactExistsError);
  assert.equal(readFileSync(join(directory, "marker"), "utf8"), "keep");
  assert.equal(sha256(readFileSync(join(directory, "model.onnx.data"))), before);
  assert.deepEqual(leftovers(fixture), []);

  // Even a foreign or empty directory in the way is left alone.
  const other = buildTemplateFixture();
  const foreign = artifactDirectory(modelsDirOf(other), other.model.repository, other.model.pinnedRevision);
  mkdirSync(foreign, { recursive: true });
  await assert.rejects(
    assembleArtifact({ template: await loaded(other), snapshotDir: other.snapshotDir, modelsDir: modelsDirOf(other) }),
    ArtifactExistsError,
  );
  assert.deepEqual(readdirSync(foreign), []);
});

test("manifest.json is written last, after every other output was verified", async () => {
  const fixture = buildTemplateFixture();
  const seen: string[][] = [];
  const parent = parentOf(fixture);
  await assembleArtifact({
    template: await loaded(fixture),
    snapshotDir: fixture.snapshotDir,
    modelsDir: modelsDirOf(fixture),
    onProgress: (progress: Progress) => {
      const staging = existsSync(parent) ? readdirSync(parent).find((name) => name.startsWith(".staging-")) : undefined;
      if (staging !== undefined) {
        seen.push(readdirSync(join(parent, staging)));
      }
      if (progress.stage === "verify-output") {
        assert.ok(!seen.at(-1)?.includes("manifest.json"));
      }
    },
  });
  assert.ok(seen.length > 0);
  assert.ok(seen.every((names) => !names.includes("manifest.json")));
});

test("cancellation stops the build and removes the staging directory", async () => {
  const fixture = buildTemplateFixture();
  const controller = new AbortController();
  await assert.rejects(
    assembleArtifact({
      template: await loaded(fixture),
      snapshotDir: fixture.snapshotDir,
      modelsDir: modelsDirOf(fixture),
      signal: controller.signal,
      limits: { copyBlockBytes: 8 },
      onProgress: (progress) => {
        if (progress.stage === "weights") {
          controller.abort();
        }
      },
    }),
    RequestCancelledError,
  );
  assert.deepEqual(leftovers(fixture), []);
  assert.equal(existsSync(join(parentOf(fixture), fixture.model.pinnedRevision)), false);

  const already = AbortSignal.abort();
  await assert.rejects(
    assembleArtifact({ template: await loaded(fixture), snapshotDir: fixture.snapshotDir, modelsDir: modelsDirOf(fixture), signal: already }),
    RequestCancelledError,
  );
});

test("insufficient free disk space is reported before anything is written", async () => {
  const fixture = buildTemplateFixture();
  await assert.rejects(
    assembleArtifact({
      template: await loaded(fixture),
      snapshotDir: fixture.snapshotDir,
      modelsDir: modelsDirOf(fixture),
      freeBytes: async () => 1024,
    }),
    InsufficientSpaceError,
  );
  assert.deepEqual(leftovers(fixture), []);
  // An unknown amount of free space (statfs unavailable) does not block the build.
  const directory = await assembleArtifact({
    template: await loaded(fixture),
    snapshotDir: fixture.snapshotDir,
    modelsDir: modelsDirOf(fixture),
    freeBytes: async () => undefined,
  });
  assert.ok(statSync(directory).isDirectory());
});
