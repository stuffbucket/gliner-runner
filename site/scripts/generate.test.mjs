import assert from "node:assert/strict";
import { test } from "node:test";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import { mkdtempSync, readdirSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { copyRunnerTemplates, fmtBytes, fmtDrift, parseNativeEvidence, rewriteLinks } from "./generate.mjs";

const root = resolve(dirname(fileURLToPath(import.meta.url)), "../..");
const ctx = { root, docNames: new Set(["protocol", "operations"]) };

test("rewrites doc links to site pages and repo links to GitHub", () => {
  const out = rewriteLinks(
    "[a](protocol.md#x) [b](../src/gliner_runner) [c](../README.md) [d](https://x.dev) [e](#top)",
    ctx,
  );
  assert.equal(
    out,
    "[a](/gliner-runner/docs/protocol/#x) [b](https://github.com/stuffbucket/gliner-runner/tree/main/src/gliner_runner) " +
      "[c](https://github.com/stuffbucket/gliner-runner/blob/main/README.md) [d](https://x.dev) [e](#top)",
  );
});

test("leaves fenced code untouched", () => {
  const text = "```\n[a](protocol.md)\n```";
  assert.equal(rewriteLinks(text, ctx), text);
});

test("parses native PyTorch evidence", () => {
  const rows = parseNativeEvidence(
    `("decide-1b", "mps/fp16"): (\n 0.0015, "1e-3",\n "Four-example " "scope.",\n ),`,
  );
  assert.deepEqual(rows, [
    { modelId: "decide-1b", profile: "mps/fp16", drift: 0.0015, order: "1e-3", scope: "Four-example scope." },
  ]);
});

test("formats numbers", () => {
  assert.equal(fmtDrift(7.152557373046875e-6), "7.153e-6");
  assert.equal(fmtDrift(0), "0");
  assert.equal(fmtBytes(1744089152), "1.62 GiB");
});

test("copies only the in-tab model's template for the playground worker", () => {
  const root = resolve(dirname(fileURLToPath(import.meta.url)), "../..");
  const out = mkdtempSync(join(tmpdir(), "runner-templates-"));
  assert.equal(copyRunnerTemplates(root, out), 1);
  assert.deepEqual(readdirSync(out), ["fastino--GLiNER2.5-Decide"]);
  const [revision] = readdirSync(join(out, "fastino--GLiNER2.5-Decide"));
  assert.deepEqual(readdirSync(join(out, "fastino--GLiNER2.5-Decide", revision)).sort(), [
    "gliner2.json",
    "graph.onnx",
    "manifest.json",
    "template.json",
  ]);
});
