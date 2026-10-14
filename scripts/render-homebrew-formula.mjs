#!/usr/bin/env node
// Renders a Homebrew formula from packaging/homebrew/.
//
//   node scripts/render-homebrew-formula.mjs --version X.Y.Z[-rc.N] [--channel stable|rc]
//        (--sums FILE | --manifest release-manifest.json) [--template FILE] [--output FILE]
//
// stable (gliner-runner.rb) requires X.Y.Z; rc (gliner-runner-rc.rb) requires X.Y.Z-rc.N. The channel
// defaults from the version. Writes the formula to --output (default: stdout). Digests come from a
// sha256sum-format file (the release's SHA256SUMS, or the per-archive .sha256 files concatenated) or
// release-manifest.json; the three platform archives must all be present.
import { readFileSync, writeFileSync } from "node:fs";
import { dirname, join, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import { parseArgs } from "node:util";

const ROOT = resolve(dirname(fileURLToPath(import.meta.url)), "..");
export const CHANNELS = {
  stable: { pattern: /^\d+\.\d+\.\d+$/, template: "gliner-runner.rb", klass: "GlinerRunner", expects: "X.Y.Z" },
  rc: { pattern: /^\d+\.\d+\.\d+-rc\.\d+$/, template: "gliner-runner-rc.rb", klass: "GlinerRunnerRc", expects: "X.Y.Z-rc.N" },
};
export const channelOf = (version) => (/-/.test(version) ? "rc" : "stable");
export const PLATFORMS = [
  { target: "darwin-arm64", placeholder: "PLACEHOLDER_SHA256_DARWIN_ARM64" },
  { target: "linux-x64", placeholder: "PLACEHOLDER_SHA256_LINUX_X64" },
  { target: "linux-arm64", placeholder: "PLACEHOLDER_SHA256_LINUX_ARM64" },
];

const archiveName = (version, target) => `gliner-runner-onnx-${version}-${target}.tar.gz`;

/** name -> sha256 from `sha256sum` output. */
export function parseSums(text) {
  const digests = new Map();
  for (const line of text.split("\n")) {
    const match = /^([0-9a-f]{64}) [ *](.+)$/.exec(line.trim());
    if (match !== null) {
      digests.set(match[2], match[1]);
    }
  }
  return digests;
}

/** name -> sha256 from a release-manifest.json artifact list. */
export function manifestDigests(text) {
  const manifest = JSON.parse(text);
  return new Map(
    manifest.artifacts.filter((artifact) => artifact.role === "onnx-runtime").map((artifact) => [artifact.name, artifact.sha256]),
  );
}

export function renderFormula({ template, version, digests, channel = channelOf(version) }) {
  const spec = CHANNELS[channel];
  if (spec === undefined) {
    throw new Error(`unknown channel ${channel}; expected stable or rc`);
  }
  if (!spec.pattern.test(version)) {
    throw new Error(`the ${channel} formula requires a ${spec.expects} version, got ${version}`);
  }
  if (!new RegExp(`^class ${spec.klass} < Formula$`, "m").test(template)) {
    throw new Error(`template is not the ${channel} formula (expected class ${spec.klass})`);
  }
  let rendered = template.replaceAll("PLACEHOLDER_VERSION", version);
  for (const { target, placeholder } of PLATFORMS) {
    const name = archiveName(version, target);
    const digest = digests.get(name);
    if (digest === undefined) {
      throw new Error(`no sha256 for ${name}`);
    }
    rendered = rendered.replaceAll(placeholder, digest);
  }
  const leftover = /PLACEHOLDER_[A-Z0-9_]+/.exec(rendered);
  if (leftover !== null) {
    throw new Error(`unreplaced placeholder ${leftover[0]}`);
  }
  return rendered;
}

function main() {
  const { values } = parseArgs({
    options: {
      version: { type: "string" },
      sums: { type: "string" },
      manifest: { type: "string" },
      channel: { type: "string" },
      template: { type: "string" },
      output: { type: "string" },
    },
  });
  if (values.version === undefined || (values.sums === undefined) === (values.manifest === undefined)) {
    throw new Error("usage: --version X.Y.Z[-rc.N] [--channel stable|rc] (--sums FILE | --manifest release-manifest.json) [--template F] [--output F]");
  }
  const digests =
    values.sums !== undefined
      ? parseSums(readFileSync(values.sums, "utf8"))
      : manifestDigests(readFileSync(values.manifest, "utf8"));
  const channel = values.channel ?? channelOf(values.version);
  const templatePath = values.template ?? join(ROOT, "packaging", "homebrew", CHANNELS[channel]?.template ?? "");
  const formula = renderFormula({ template: readFileSync(templatePath, "utf8"), version: values.version, digests, channel });
  if (values.output === undefined) {
    process.stdout.write(formula);
  } else {
    writeFileSync(values.output, formula);
  }
}

if (process.argv[1] !== undefined && resolve(process.argv[1]) === fileURLToPath(import.meta.url)) {
  try {
    main();
  } catch (error) {
    process.stderr.write(`render-homebrew-formula: ${error.message}\n`);
    process.exit(1);
  }
}
