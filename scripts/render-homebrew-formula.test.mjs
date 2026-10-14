import assert from "node:assert/strict";
import { execFileSync } from "node:child_process";
import { mkdtempSync, readFileSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import test from "node:test";
import { fileURLToPath } from "node:url";

import { manifestDigests, parseSums, renderFormula } from "./render-homebrew-formula.mjs";

const template = (name) => readFileSync(fileURLToPath(new URL(`../packaging/homebrew/${name}`, import.meta.url)), "utf8");
const STABLE_TEMPLATE = template("gliner-runner.rb");
const RC_TEMPLATE = template("gliner-runner-rc.rb");
const hex = (character) => character.repeat(64);

const sumsFor = (version) =>
  [
    `${hex("a")}  gliner-runner-onnx-${version}-darwin-arm64.tar.gz`,
    `${hex("b")}  gliner-runner-onnx-${version}-linux-x64.tar.gz`,
    `${hex("c")}  gliner-runner-onnx-${version}-linux-arm64.tar.gz`,
    `${hex("d")}  gliner-runner-onnx-${version}-linux-arm64.tar.gz.sha256`,
    `${hex("e")}  release-manifest.json`,
    "",
  ].join("\n");

const CASES = [
  { channel: "stable", version: "1.2.3", template: STABLE_TEMPLATE, klass: "GlinerRunner", conflicts: "gliner-runner-rc" },
  { channel: "rc", version: "1.2.3-rc.4", template: RC_TEMPLATE, klass: "GlinerRunnerRc", conflicts: "gliner-runner" },
];

for (const { channel, version, template: source, klass, conflicts } of CASES) {
  const sums = sumsFor(version);

  test(`${channel}: renders version and the three platform digests`, () => {
    const formula = renderFormula({ template: source, version, digests: parseSums(sums) });
    assert.match(formula, new RegExp(`class ${klass} < Formula`));
    assert.ok(formula.includes(`/download/v${version}/gliner-runner-onnx-${version}-darwin-arm64.tar.gz"`));
    // Homebrew reads the version from the URL; its audit rejects an explicit `version` that repeats it.
    assert.doesNotMatch(formula, /^\s+version "/m);
    assert.match(formula, new RegExp(`conflicts_with "${conflicts}"`));
    assert.doesNotMatch(formula, /PLACEHOLDER/);
    assert.match(formula, new RegExp(`darwin-arm64\\.tar\\.gz"\\n\\s+sha256 "${hex("a")}"`));
    assert.match(formula, new RegExp(`linux-x64\\.tar\\.gz"\\n\\s+sha256 "${hex("b")}"`));
    assert.match(formula, new RegExp(`linux-arm64\\.tar\\.gz"\\n\\s+sha256 "${hex("c")}"`));
    assert.ok(!formula.includes(hex("d")), "the .sha256 file digest must not be used");
  });

  test(`${channel}: manifest and SHA256SUMS produce the same formula`, () => {
    const manifest = JSON.stringify({
      artifacts: [
        { role: "onnx-runtime", name: `gliner-runner-onnx-${version}-darwin-arm64.tar.gz`, sha256: hex("a") },
        { role: "onnx-runtime", name: `gliner-runner-onnx-${version}-linux-x64.tar.gz`, sha256: hex("b") },
        { role: "onnx-runtime", name: `gliner-runner-onnx-${version}-linux-arm64.tar.gz`, sha256: hex("c") },
        { role: "openapi", name: "openapi.json", sha256: hex("f") },
      ],
    });
    assert.equal(
      renderFormula({ template: source, version, digests: manifestDigests(manifest) }),
      renderFormula({ template: source, version, digests: parseSums(sums) }),
    );
  });

  test(`${channel}: missing platform digest is an error`, () => {
    const partial = sums.split("\n").filter((line) => !line.includes("linux-x64")).join("\n");
    assert.throws(() => renderFormula({ template: source, version, digests: parseSums(partial) }), /no sha256 for .*linux-x64/);
  });
}

test("stable rejects release candidates; rc requires one", () => {
  assert.throws(
    () => renderFormula({ template: STABLE_TEMPLATE, version: "1.2.3-rc.1", digests: parseSums(sumsFor("1.2.3-rc.1")), channel: "stable" }),
    /stable formula requires a X\.Y\.Z version/,
  );
  assert.throws(
    () => renderFormula({ template: RC_TEMPLATE, version: "1.2.3", digests: parseSums(sumsFor("1.2.3")), channel: "rc" }),
    /rc formula requires a X\.Y\.Z-rc\.N version/,
  );
  assert.throws(
    () => renderFormula({ template: STABLE_TEMPLATE, version: "1.2.3-rc.1", digests: parseSums(sumsFor("1.2.3-rc.1")) }),
    /template is not the rc formula/,
  );
  assert.throws(
    () => renderFormula({ template: RC_TEMPLATE, version: "1.2.3-beta.1", digests: parseSums(sumsFor("1.2.3-beta.1")) }),
    /requires/,
  );
});

test("CLI picks the template from the version and writes --output", () => {
  const dir = mkdtempSync(join(tmpdir(), "render-formula-"));
  const script = fileURLToPath(new URL("./render-homebrew-formula.mjs", import.meta.url));
  for (const { version, klass } of CASES) {
    const sums = join(dir, `SHA256SUMS-${version}`);
    const output = join(dir, `${version}.rb`);
    writeFileSync(sums, sumsFor(version));
    execFileSync(process.execPath, [script, "--version", version, "--sums", sums, "--output", output]);
    assert.match(readFileSync(output, "utf8"), new RegExp(`class ${klass} < Formula`));
  }
});

test("CLI fails on a channel/version mismatch", () => {
  const dir = mkdtempSync(join(tmpdir(), "render-formula-"));
  const sums = join(dir, "SHA256SUMS");
  writeFileSync(sums, sumsFor("1.2.3"));
  const script = fileURLToPath(new URL("./render-homebrew-formula.mjs", import.meta.url));
  assert.throws(
    () => execFileSync(process.execPath, [script, "--version", "1.2.3", "--channel", "rc", "--sums", sums], { stdio: "pipe" }),
    /rc formula requires/,
  );
});
