import assert from "node:assert/strict";
import { readFileSync, rmSync, writeFileSync } from "node:fs";
import { join } from "node:path";
import { test } from "node:test";

import { KNOWN_MODELS, type KnownModel } from "../../src/index.js";
import {
  TemplateError,
  defaultTemplatesDir,
  listTemplates,
  loadTemplate,
  resolveTemplatesDir,
} from "../../src/onnx/templates.js";
import { buildTemplateFixture, sha256 } from "./template-fixture.js";

const MODEL = KNOWN_MODELS[0] as KnownModel;

test("a consistent template loads with its manifest and strict structure", async () => {
  const fixture = buildTemplateFixture();
  const template = await loadTemplate(fixture.templatesDir, MODEL, MODEL.pinnedRevision);
  assert.ok(template);
  assert.equal(template.revision, MODEL.pinnedRevision);
  assert.equal(template.data.tensors.length, 5);
  assert.equal(template.manifest.modelId, "decide-340m");
  assert.deepEqual(template.outputs.map((output) => output.path).sort(), Object.keys(fixture.artifact).sort());
  assert.deepEqual((await listTemplates(fixture.templatesDir)).map((entry) => entry.model.modelId), ["decide-340m"]);
});

test("only the pinned revision with a template directory is preparable", async () => {
  const fixture = buildTemplateFixture();
  assert.equal(await loadTemplate(fixture.templatesDir, MODEL, "1".repeat(40)), undefined);
  assert.equal(await loadTemplate(join(fixture.root, "empty"), MODEL, MODEL.pinnedRevision), undefined);
  assert.equal(await loadTemplate(fixture.templatesDir, KNOWN_MODELS[1] as KnownModel, (KNOWN_MODELS[1] as KnownModel).pinnedRevision), undefined);
  assert.deepEqual(await listTemplates(join(fixture.root, "empty")), []);
});

const mutations: [string, (template: Record<string, any>) => void, RegExp][] = [
  ["format_version", (t) => (t.format_version = 2), /format_version/],
  ["kind", (t) => (t.kind = "other"), /kind/],
  ["model_id", (t) => (t.model_id = "decide-1b"), /model_id/],
  ["repository", (t) => (t.repository = "evil/model"), /repository/],
  ["revision", (t) => (t.revision = "2".repeat(40)), /revision/],
  ["source_files empty", (t) => (t.source_files = []), /source_files must be a non-empty list/],
  ["no weights", (t) => (t.source_files = t.source_files.filter((f: any) => f.path !== "model.safetensors")), /must include model\.safetensors/],
  ["source path traversal", (t) => (t.source_files[1].path = "../x"), /plain file name/],
  ["duplicate source", (t) => t.source_files.push({ ...t.source_files[1] }), /more than once/],
  ["bad source digest", (t) => (t.source_files[1].sha256 = "xyz"), /sha256 must be 64/],
  ["negative size", (t) => (t.source_files[1].size = -1), /size must be an integer/],
  ["missing graph", (t) => (t.template_files = t.template_files.filter((f: any) => f.output !== "model.onnx")), /must produce model\.onnx/],
  ["template path traversal", (t) => (t.template_files[0].path = "a/b"), /plain file name/],
  ["data alignment", (t) => (t.data.alignment = 0), /alignment/],
  ["tensor transform", (t) => (t.data.tensors[0].transform = "rotate"), /transform/],
  ["tensor dtype", (t) => (t.data.tensors[0].dtype = "float16"), /dtype/],
  ["tensor shape for transpose", (t) => (t.data.tensors[1].shape = [4, 6]), /transpose needs/],
  ["identity shape mismatch", (t) => (t.data.tensors[0].shape = [5, 3]), /identity shape/],
  ["tensor length", (t) => (t.data.tensors[0].length = 8), /length 8 does not match/],
  ["tensor offset not aligned", (t) => (t.data.tensors[1].offset = 60), /rounded up to 64/],
  ["tensor offset overlapping", (t) => (t.data.tensors[1].offset = 0), /rounded up/],
  ["duplicate initializer", (t) => (t.data.tensors[1].name = t.data.tensors[0].name), /repeats initializer/],
  ["data size with trailing padding", (t) => (t.data.size += 64), /no trailing padding/],
  ["data path is the manifest", (t) => (t.data.path = "manifest.json"), /data\.path/],
  ["output digest", (t) => (t.outputs[0].sha256 = "0".repeat(64)), /does not match/],
  ["output without an origin", (t) => t.outputs.push({ path: "extra.bin", size: 1, sha256: "0".repeat(64) }), /no unambiguous origin/],
  ["output lists the manifest", (t) => t.outputs.push({ path: "manifest.json", size: 1, sha256: "0".repeat(64) }), /must not list manifest\.json/],
  ["output missing tokenizer", (t) => (t.outputs = t.outputs.filter((o: any) => o.path !== "tokenizer.json")), /must include tokenizer\.json/],
  ["duplicate output", (t) => t.outputs.push({ ...t.outputs[0] }), /more than once/],
  ["template file digest", (t) => (t.template_files[0].sha256 = "0".repeat(64)), /does not match the recorded size\/sha256 of its origin/],
];
for (const [name, mutate, message] of mutations) {
  test(`rejects a template with ${name}`, async () => {
    const fixture = buildTemplateFixture({ mutate });
    await assert.rejects(loadTemplate(fixture.templatesDir, MODEL, MODEL.pinnedRevision), (error: unknown) => {
      assert.ok(error instanceof TemplateError, String(error));
      assert.match(error.message, message);
      return true;
    });
    const invalid: unknown[] = [];
    assert.deepEqual(await listTemplates(fixture.templatesDir, KNOWN_MODELS, (_, error) => invalid.push(error)), []);
    assert.equal(invalid.length, 1);
  });
}

test("rejects altered, missing and non-JSON template files and a disagreeing manifest", async () => {
  const altered = buildTemplateFixture();
  writeFileSync(join(altered.templateDir, "graph.onnx"), "changed");
  await assert.rejects(loadTemplate(altered.templatesDir, MODEL, MODEL.pinnedRevision), /graph\.onnx does not match/);

  const missing = buildTemplateFixture();
  rmSync(join(missing.templateDir, "gliner2.json"));
  await assert.rejects(loadTemplate(missing.templatesDir, MODEL, MODEL.pinnedRevision), /gliner2\.json is missing/);

  const garbage = buildTemplateFixture();
  writeFileSync(join(garbage.templateDir, "template.json"), "{nope");
  await assert.rejects(loadTemplate(garbage.templatesDir, MODEL, MODEL.pinnedRevision), /not valid JSON/);

  // A manifest whose file list disagrees with the outputs (still a valid manifest, correctly hashed).
  const disagree = buildTemplateFixture();
  const manifest = JSON.parse(disagree.manifest) as { files: { sha256: string }[] };
  manifest.files[0]!.sha256 = "f".repeat(64);
  const text = JSON.stringify(manifest, null, 2);
  writeFileSync(join(disagree.templateDir, "manifest.json"), text);
  const template = JSON.parse(readFileSync(join(disagree.templateDir, "template.json"), "utf8"));
  template.template_files.find((file: any) => file.output === "manifest.json").sha256 = sha256(text);
  template.template_files.find((file: any) => file.output === "manifest.json").size = Buffer.byteLength(text);
  writeFileSync(join(disagree.templateDir, "template.json"), JSON.stringify(template));
  await assert.rejects(loadTemplate(disagree.templatesDir, MODEL, MODEL.pinnedRevision), /manifest files do not equal the template outputs/);
});

test("template directory resolution: flag, then environment, then the bundled directory", () => {
  assert.equal(resolveTemplatesDir("/flag", { GLINER_RUNNER_ONNX_TEMPLATES_DIR: "/env" }), "/flag");
  assert.equal(resolveTemplatesDir(undefined, { GLINER_RUNNER_ONNX_TEMPLATES_DIR: "/env" }), "/env");
  assert.equal(resolveTemplatesDir("", { GLINER_RUNNER_ONNX_TEMPLATES_DIR: "" }), defaultTemplatesDir());
  assert.match(defaultTemplatesDir().replaceAll("\\", "/"), /packages\/onnx-runner\/templates$|onnx-runner\/templates$/);
});
