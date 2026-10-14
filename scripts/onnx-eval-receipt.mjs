#!/usr/bin/env node
// Evaluation receipt tooling for the Python-free ONNX runner.
//
//   node scripts/onnx-eval-receipt.mjs record    run the real parity evaluation locally and write the receipt
//   node scripts/onnx-eval-receipt.mjs check     model-free, network-free consistency check (what CI runs)
//   node scripts/onnx-eval-receipt.mjs identity [--list]   print the runner identity digest
//
// `record` needs GLINER_RUNNER_ONNX_MODELS_DIR with the real prepared artifacts. `check` never loads a model.
// See docs/onnx-runtime.md ("Evaluation receipt").
import { execFileSync, spawnSync } from "node:child_process";
import { randomUUID } from "node:crypto";
import { existsSync, mkdirSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { cpus } from "node:os";
import { dirname, join, resolve } from "node:path";
import { fileURLToPath } from "node:url";

import {
  BUILD_INFO_PATH,
  CORPUS_IDS,
  GOLDEN_DIR,
  LEDGER_PATH,
  PACKAGE_DIR,
  RECEIPT_FORMAT_VERSION,
  RECEIPT_KIND,
  RECEIPT_PATH,
  RECORD_COMMAND,
  TOLERANCE,
  checkReceipt,
  computeIdentity,
  corpusPath,
  fileSha256,
  goldenPath,
  packageVersion,
  readCatalogPins,
  renderBuildInfo,
  sha256Hex,
  templateDir,
  writeReceiptArtifacts,
} from "./onnx-receipt-lib.mjs";

const ROOT = resolve(dirname(fileURLToPath(import.meta.url)), "..");
const PACKAGE_NAME = "@gliner-runner/onnx-runner";

function die(message) {
  process.stderr.write(`onnx-eval-receipt: ${message}\n`);
  process.exit(1);
}

function build() {
  execFileSync("pnpm", ["--filter", `${PACKAGE_NAME}...`, "run", "build"], { cwd: ROOT, stdio: "inherit" });
}

function installedVersion(name) {
  try {
    return JSON.parse(readFileSync(join(ROOT, PACKAGE_DIR, "node_modules", name, "package.json"), "utf8")).version;
  } catch {
    return "unknown";
  }
}

function probeCli() {
  const output = execFileSync(process.execPath, [join(ROOT, PACKAGE_DIR, "dist", "src", "cli.js"), "--version", "--json"], {
    encoding: "utf8",
    env: { PATH: process.env.PATH ?? "" },
    stdio: ["ignore", "pipe", "pipe"],
  });
  return JSON.parse(output);
}

function runCheck() {
  if (!process.argv.includes("--no-build")) {
    build();
  }
  const { errors, notes } = checkReceipt({
    root: ROOT,
    baseRef: process.env.GLINER_RUNNER_RECEIPT_BASE_REF,
    probeCli,
  });
  for (const note of notes) {
    process.stdout.write(`note: ${note}\n`);
  }
  if (errors.length > 0) {
    for (const message of errors) {
      process.stderr.write(`FAIL: ${message}\n`);
    }
    process.stderr.write(`\n${errors.length} ONNX evaluation receipt problem(s). Fix: ${RECORD_COMMAND} (requires real model artifacts), then commit ${RECEIPT_PATH}, ${LEDGER_PATH} and ${BUILD_INFO_PATH}.\n`);
    return 1;
  }
  process.stdout.write(`OK: ONNX evaluation receipt is coherent (${RECEIPT_PATH})\n`);
  return 0;
}

function runIdentity() {
  const identity = computeIdentity(ROOT);
  if (process.argv.includes("--list")) {
    for (const entry of identity.entries) {
      process.stdout.write(`${entry.sha256}  ${entry.path}\n`);
    }
  }
  process.stdout.write(`${identity.sha256}\n`);
  return 0;
}

function runRecord() {
  const modelsDir = process.env.GLINER_RUNNER_ONNX_MODELS_DIR;
  if (modelsDir === undefined || modelsDir === "") {
    die("set GLINER_RUNNER_ONNX_MODELS_DIR to the directory holding the prepared ONNX artifacts (<dir>/onnx/<owner>--<name>/<revision>)");
  }
  const pins = readCatalogPins(ROOT);

  // Every curated model must be installed and must be the exact artifact the checked-in manifest describes.
  const models = pins.map((pin) => {
    const artifact = join(resolve(modelsDir), "onnx", pin.repository.replace("/", "--"), pin.revision, "manifest.json");
    if (!existsSync(artifact)) {
      die(`${pin.model_id}: artifact ${artifact} is missing; run \`mise run prepare:onnx\` or \`gliner-runner-onnx models install\``);
    }
    const manifestSha = fileSha256(ROOT, `${templateDir(pin)}/manifest.json`);
    if (sha256Hex(readFileSync(artifact)) !== manifestSha) {
      die(`${pin.model_id}: installed manifest ${artifact} differs from the checked-in template manifest`);
    }
    return {
      model_id: pin.model_id,
      repository: pin.repository,
      revision: pin.revision,
      template_sha256: fileSha256(ROOT, `${templateDir(pin)}/template.json`),
      manifest_sha256: manifestSha,
    };
  });
  const corpora = CORPUS_IDS.map((id) => ({ id, path: corpusPath(id), sha256: fileSha256(ROOT, corpusPath(id)) }));
  const goldens = pins.flatMap((pin) =>
    CORPUS_IDS.map((id) => ({ path: goldenPath(pin.model_id, id), sha256: fileSha256(ROOT, goldenPath(pin.model_id, id)) })),
  );

  const before = computeIdentity(ROOT);
  if (before.untracked.length > 0) {
    process.stderr.write(`warning: untracked files feed the runner identity; \`git add\` them or CI will compute a different identity:\n  ${before.untracked.join("\n  ")}\n`);
  }
  if (!existsSync(join(ROOT, BUILD_INFO_PATH))) {
    // Bootstrap: the package needs a build-info module to compile; the real one is written with the receipt.
    writeFileSync(
      join(ROOT, BUILD_INFO_PATH),
      renderBuildInfo({ version: packageVersion(ROOT), identitySha256: "unrecorded", evalReceiptSha256: "unrecorded", evalRunId: "unrecorded" }),
    );
  }
  build();

  const reportDir = join(ROOT, "dist", "onnx-parity-run");
  rmSync(reportDir, { recursive: true, force: true });
  mkdirSync(reportDir, { recursive: true });
  const env = { ...process.env };
  for (const key of ["GLINER_RUNNER_ONNX_PARITY_MODELS", "GLINER_RUNNER_ONNX_CORPUS_DIR"]) {
    delete env[key];
  }
  Object.assign(env, {
    GLINER_RUNNER_ONNX_MODELS_DIR: resolve(modelsDir),
    GLINER_RUNNER_ONNX_PARITY_DIR: join(ROOT, GOLDEN_DIR),
    GLINER_RUNNER_ONNX_PARITY_OUT_DIR: reportDir,
  });
  process.stdout.write(`running parity evaluation for ${pins.length} models x ${CORPUS_IDS.length} corpora\n`);
  const run = spawnSync(
    process.execPath,
    ["--test", join(ROOT, PACKAGE_DIR, "dist", "test", "onnx", "parity.test.js")],
    { cwd: ROOT, env, stdio: "inherit" },
  );
  if (run.status !== 0) {
    die("the parity evaluation failed; no receipt was written");
  }

  const results = [];
  for (const pin of pins) {
    for (const corpus of CORPUS_IDS) {
      const reportPath = join(reportDir, `node-vs-pytorch-${pin.model_id}-${corpus}.json`);
      if (!existsSync(reportPath)) {
        die(`no parity report for ${pin.model_id} x ${corpus} (test skipped or model missing); no receipt was written`);
      }
      const report = JSON.parse(readFileSync(reportPath, "utf8"));
      const corpusEntry = corpora.find((entry) => entry.id === corpus);
      const problems = [];
      if (report.model_id !== pin.model_id || report.revision !== pin.revision || report.repository !== pin.repository) {
        problems.push("report is for a different model revision");
      }
      if (report.corpus_id !== corpus || report.corpus_sha256 !== corpusEntry.sha256) {
        problems.push("report corpus does not match the checked-in corpus");
      }
      if (report.passed !== true || report.overall.exact_semantic_matches !== report.overall.requests || report.overall.requests === 0) {
        problems.push("report did not pass");
      }
      if (report.tolerance !== TOLERANCE || !(report.overall.max_raw_abs_diff <= TOLERANCE)) {
        problems.push(`max_raw_abs_diff ${report.overall.max_raw_abs_diff} exceeds ${TOLERANCE}`);
      }
      if (problems.length > 0) {
        die(`${pin.model_id} x ${corpus}: ${problems.join("; ")}; no receipt was written`);
      }
      results.push({
        model_id: pin.model_id,
        corpus,
        requests: report.overall.requests,
        exact_semantic_matches: report.overall.exact_semantic_matches,
        max_raw_abs_diff: report.overall.max_raw_abs_diff,
        synthetic_zero_diffs: report.overall.synthetic_zero_diffs,
        compared_values: report.overall.compared_values,
        passed: true,
      });
    }
  }

  const after = computeIdentity(ROOT);
  if (after.sha256 !== before.sha256) {
    die("runner inputs changed while the evaluation was running; no receipt was written");
  }

  const receiptPath = join(ROOT, RECEIPT_PATH);
  const receipt = {
    format_version: RECEIPT_FORMAT_VERSION,
    kind: RECEIPT_KIND,
    run_id: randomUUID(),
    created_at: new Date().toISOString(),
    previous_receipt_sha256: existsSync(receiptPath) ? sha256Hex(readFileSync(receiptPath)) : null,
    runner: { version: packageVersion(ROOT), identity_sha256: after.sha256 },
    models,
    corpora,
    goldens,
    results,
    tolerance: TOLERANCE,
    host: {
      platform: process.platform,
      arch: process.arch,
      cpu_model: cpus()[0]?.model ?? "unknown",
      cpu_cores: cpus().length,
      node: process.version,
      onnxruntime_node: installedVersion("onnxruntime-node"),
    },
    all_passed: true,
  };
  const sha = writeReceiptArtifacts(ROOT, receipt);
  process.stdout.write(`wrote ${RECEIPT_PATH} (sha256 ${sha}), ${BUILD_INFO_PATH} and appended ${LEDGER_PATH}\n`);

  // Rebuild so dist embeds the new receipt, then run the same checks CI will run (minus the base comparison).
  build();
  const { errors } = checkReceipt({ root: ROOT, baseRef: undefined, probeCli });
  if (errors.length > 0) {
    die(`the receipt just written is not coherent:\n  ${errors.join("\n  ")}`);
  }
  process.stdout.write(`commit ${RECEIPT_PATH}, ${LEDGER_PATH} and ${BUILD_INFO_PATH} with your change\n`);
  return 0;
}

const commands = { record: runRecord, check: runCheck, identity: runIdentity };
const command = commands[process.argv[2]];
if (command === undefined) {
  die("usage: onnx-eval-receipt.mjs record | check [--no-build] | identity [--list]");
}
process.exitCode = command();
