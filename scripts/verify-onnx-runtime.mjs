#!/usr/bin/env node
// Verifies a packaged ONNX runner runs with no Python, PyTorch, Transformers, GLiNER2, or repo
// node_modules available.
//
//   node scripts/verify-onnx-runtime.mjs [--archive PATH | --release-dir DIR]
//        [--models-dir DIR] [--allow-missing-backend]
//
// By default the .tar.gz produced by package-onnx-runtime.mjs is extracted into a fresh temporary
// directory outside the repository, every file is checked against release-manifest.json, and the
// launcher is started with a sanitized environment (PATH=/usr/bin:/bin, temporary HOME).
//
// Checks: both launchers (bin/gliner-runner and bin/gliner-runner-onnx) exist; --version reports the checked-in evaluation receipt (release/onnx-eval-receipt.json) and its runner
// identity, /healthz, /v1/capabilities (onnx/cpu/fp32/classify), one /v1/infer against decide-340m
// when real artifacts exist (GLINER_RUNNER_ONNX_MODELS_DIR or dist/onnx-models), and SIGTERM -> 0.
// When the packaged app has no ONNX backend (dist/src/onnx/factory.js absent), the capability and
// inference checks fail unless --allow-missing-backend is given, in which case they are SKIPPED.
import { spawn, execFileSync } from "node:child_process";
import { createServer } from "node:net";
import { existsSync, mkdtempSync, readFileSync, readdirSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { dirname, join, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import { parseArgs } from "node:util";

import { RECEIPT_PATH, sha256Hex } from "./onnx-receipt-lib.mjs";
import { describeTree } from "./package-onnx-runtime.mjs";

const ROOT = resolve(dirname(fileURLToPath(import.meta.url)), "..");
const results = [];
let failures = 0;

function record(status, name, detail = "") {
  results.push({ status, name, detail });
  if (status === "FAIL") {
    failures += 1;
  }
  process.stdout.write(`${status.padEnd(4)}  ${name}${detail === "" ? "" : `: ${detail}`}\n`);
}

function check(condition, name, detail = "") {
  record(condition ? "PASS" : "FAIL", name, condition ? "" : detail);
  return condition;
}

function freePort() {
  return new Promise((resolvePort, reject) => {
    const server = createServer();
    server.once("error", reject);
    server.listen(0, "127.0.0.1", () => {
      const { port } = server.address();
      server.close(() => resolvePort(port));
    });
  });
}

const sleep = (ms) => new Promise((done) => setTimeout(done, ms));

function defaultArchive() {
  const directory = join(ROOT, "dist", "release", "onnx-runtime");
  const suffix = `-${process.platform}-${process.arch}.tar.gz`;
  const found = existsSync(directory)
    ? readdirSync(directory).filter((name) => name.startsWith("gliner-runner-onnx-") && name.endsWith(suffix))
    : [];
  if (found.length !== 1) {
    throw new Error(`expected exactly one ${suffix} archive in ${directory}; run package:onnx-runtime`);
  }
  return join(directory, found[0]);
}

async function verifyManifest(releaseDir) {
  const manifest = JSON.parse(readFileSync(join(releaseDir, "release-manifest.json"), "utf8"));
  const actual = await describeTree(releaseDir, new Set(["release-manifest.json"]));
  const expected = new Map(manifest.files.map((entry) => [entry.path, entry]));
  const problems = [];
  for (const entry of actual) {
    const wanted = expected.get(entry.path);
    if (wanted === undefined) {
      problems.push(`unlisted ${entry.path}`);
    } else if (
      wanted.type !== entry.type ||
      wanted.sha256 !== entry.sha256 ||
      wanted.size !== entry.size ||
      wanted.target !== entry.target
    ) {
      problems.push(`mismatch ${entry.path}`);
    }
    expected.delete(entry.path);
  }
  for (const path of expected.keys()) {
    problems.push(`missing ${path}`);
  }
  check(problems.length === 0, `manifest lists all ${manifest.files.length} files with matching size/sha256`, problems.slice(0, 5).join("; "));
  return manifest;
}

function assertNoPython(releaseDir, manifest) {
  const banned = /(^|\/)(torch|transformers|gliner2|safetensors|numpy|site-packages|__pycache__)(\/|$)|\.py[co]?$|\.whl$/i;
  // app/dist/src/gliner2 is this project's own TypeScript port of the GLiNER2 schema logic.
  const hits = manifest.files
    .filter((entry) => !entry.path.startsWith("app/dist/src/") && banned.test(entry.path))
    .map((entry) => entry.path);
  check(hits.length === 0, "release contains no Python/PyTorch/Transformers/GLiNER2 files", hits.slice(0, 5).join(", "));
  check(manifest.pythonRequired === false && manifest.modelsIncluded === false, "manifest declares no Python and no bundled models");
  const hasNode = existsSync(join(releaseDir, manifest.node.path));
  check(hasNode, "bundled Node.js runtime is present", manifest.node.path);
}

/** The packaged binary must report the receipt and runner identity that are checked in. */
function verifyVersion(launcher, env, cwd, manifest) {
  const receiptBytes = readFileSync(join(ROOT, RECEIPT_PATH));
  const receipt = JSON.parse(receiptBytes.toString("utf8"));
  let reported;
  try {
    reported = JSON.parse(execFileSync(launcher, ["--version", "--json"], { env, cwd, encoding: "utf8", timeout: 60_000 }));
  } catch (error) {
    check(false, "packaged --version --json runs without models", error.message);
    return;
  }
  check(
    reported.eval_receipt_sha256 === sha256Hex(receiptBytes),
    "packaged --version reports the checked-in evaluation receipt sha256",
    `${reported.eval_receipt_sha256} != ${sha256Hex(receiptBytes)} (${RECEIPT_PATH}); rebuild after \`mise run eval:onnx-receipt\``,
  );
  check(
    reported.identity_sha256 === receipt.runner.identity_sha256,
    "packaged --version reports the receipt's runner identity sha256",
    `${reported.identity_sha256} != ${receipt.runner.identity_sha256}`,
  );
  check(reported.eval_run_id === receipt.run_id, "packaged --version reports the receipt run_id", `${reported.eval_run_id} != ${receipt.run_id}`);
  check(reported.version === manifest.version, "packaged --version reports the release version", `${reported.version} != ${manifest.version}`);
  record("INFO", "packaged version", JSON.stringify(reported));
}

async function getJson(url, init) {
  const response = await fetch(url, init);
  const text = await response.text();
  let body;
  try {
    body = JSON.parse(text);
  } catch {
    body = text;
  }
  return { status: response.status, body };
}

async function main() {
  const { values } = parseArgs({
    options: {
      archive: { type: "string" },
      "release-dir": { type: "string" },
      "models-dir": { type: "string" },
      "allow-missing-backend": { type: "boolean", default: false },
    },
  });
  const scratch = mkdtempSync(join(tmpdir(), "gliner-onnx-verify-"));
  let releaseDir;
  if (values["release-dir"] !== undefined) {
    releaseDir = resolve(values["release-dir"]);
  } else {
    const archive = resolve(values.archive ?? defaultArchive());
    const extractDir = join(scratch, "extract");
    execFileSync("mkdir", ["-p", extractDir]);
    execFileSync("tar", ["-xzf", archive, "-C", extractDir]);
    const [name] = readdirSync(extractDir);
    releaseDir = join(extractDir, name);
    record("INFO", "extracted archive", `${archive} -> ${releaseDir}`);
  }
  check(!releaseDir.startsWith(ROOT + "/"), "release runs from outside the repository tree (no repo node_modules)", releaseDir);

  const manifest = await verifyManifest(releaseDir);
  assertNoPython(releaseDir, manifest);

  const executables = manifest.executables ?? [];
  const extension = manifest.executable.endsWith(".cmd") ? ".cmd" : "";
  check(
    ["gliner-runner", "gliner-runner-onnx"].every((name) => executables.includes(`bin/${name}${extension}`)),
    "manifest lists the gliner-runner and gliner-runner-onnx launchers",
    JSON.stringify(executables),
  );
  const launcher = join(releaseDir, `bin/gliner-runner${extension}`);
  const legacyLauncher = join(releaseDir, manifest.executable);
  const home = join(scratch, "home");
  execFileSync("mkdir", ["-p", home]);
  const hasBackend = existsSync(join(releaseDir, "app", "dist", "src", "onnx", "factory.js"));

  const modelsDir = resolve(
    values["models-dir"] ?? process.env.GLINER_RUNNER_ONNX_MODELS_DIR ?? join(ROOT, "dist", "onnx-models"),
  );
  const decideDirectory = join(modelsDir, "onnx", "fastino--GLiNER2.5-Decide");
  const haveArtifacts = existsSync(join(decideDirectory, "9d1bfb848cd16d93ee56ccbc59cd92bf0f284d90", "manifest.json"));

  const env = { PATH: "/usr/bin:/bin", HOME: home, TMPDIR: scratch, LANG: "C", LC_ALL: "C" };
  verifyVersion(launcher, env, scratch, manifest);
  try {
    const user = execFileSync(launcher, ["--version"], { env, cwd: scratch, encoding: "utf8", timeout: 60_000 });
    const legacy = execFileSync(legacyLauncher, ["--version"], { env, cwd: scratch, encoding: "utf8", timeout: 60_000 });
    check(user === legacy, "bin/gliner-runner and bin/gliner-runner-onnx print the same --version", `${user} != ${legacy}`);
  } catch (error) {
    check(false, "bin/gliner-runner --version runs", error.message);
  }

  const port = await freePort();
  const args = ["serve", "--host", "127.0.0.1", "--port", String(port), "--device", "cpu"];
  args.push("--models-dir", haveArtifacts ? modelsDir : join(scratch, "models"));
  record("INFO", "launching", `${launcher} ${args.join(" ")} (env: ${JSON.stringify(env)})`);

  let log = "";
  let exited;
  const child = spawn(launcher, args, { env, cwd: scratch, stdio: ["ignore", "pipe", "pipe"] });
  const exit = new Promise((done) => {
    child.on("exit", (code, signal) => {
      exited = { code, signal };
      done(exited);
    });
    child.on("error", (error) => {
      log += `spawn error: ${error.message}\n`;
      exited = { code: -1, signal: null };
      done(exited);
    });
  });
  child.stdout.on("data", (chunk) => (log += chunk));
  child.stderr.on("data", (chunk) => (log += chunk));

  const base = `http://127.0.0.1:${port}`;
  try {
    let healthy;
    for (let attempt = 0; attempt < 150 && exited === undefined; attempt += 1) {
      try {
        healthy = await getJson(`${base}/healthz`);
        break;
      } catch {
        await sleep(200);
      }
    }
    if (!check(healthy?.status === 200 && healthy.body?.status === "ok", "/healthz responds ok", JSON.stringify(healthy ?? exited))) {
      return;
    }

    const capabilities = await getJson(`${base}/v1/capabilities`);
    const onnx = Array.isArray(capabilities.body)
      ? capabilities.body.find((entry) => entry.backend === "onnx")
      : undefined;
    if (!hasBackend && onnx === undefined && values["allow-missing-backend"]) {
      record("SKIP", "/v1/capabilities advertises onnx/cpu/fp32/classify", "packaged app has no dist/src/onnx/factory.js yet (--allow-missing-backend)");
      record("SKIP", "/v1/infer decide-340m", "ONNX backend is not wired into this package yet");
    } else {
      check(
        onnx?.devices?.includes("cpu") && onnx.precisions?.includes("fp32") && onnx.operations?.includes("classify"),
        "/v1/capabilities advertises onnx/cpu/fp32/classify",
        JSON.stringify(capabilities.body),
      );
      if (!haveArtifacts) {
        record("SKIP", "/v1/infer decide-340m", `no real artifacts under ${decideDirectory}`);
      } else {
        const request = {
          request_id: "5f0f7b6e-5b0a-4a74-9a37-0d1ef3d4e001",
          model: "decide-340m",
          backend: "onnx",
          precision: "fp32",
          operation: "classify",
          text: "The invoice needs urgent review before the payment deadline.",
          schema: {
            kind: "classification",
            tasks: { priority: { labels: ["urgent", "routine", "unknown"], min_labels: 1, max_labels: 1 } },
            constraints: [],
          },
          options: { threshold: 0.5, output_format: "native", include_confidence: true },
        };
        const started = Date.now();
        const inference = await getJson(`${base}/v1/infer`, {
          method: "POST",
          headers: { "content-type": "application/json" },
          body: JSON.stringify(request),
          signal: AbortSignal.timeout(300_000),
        });
        const ok =
          inference.status === 200 &&
          inference.body?.backend === "onnx" &&
          inference.body.model === "decide-340m" &&
          typeof inference.body.output?.priority !== "undefined" &&
          Number.isInteger(inference.body.usage?.inputTokens ?? inference.body.usage?.input_tokens);
        check(ok, "/v1/infer decide-340m returns a success response", `${inference.status} ${JSON.stringify(inference.body).slice(0, 500)}`);
        if (ok) {
          record("INFO", "infer response", `${Date.now() - started} ms: ${JSON.stringify(inference.body.output)}`);
        }
      }
    }
  } finally {
    if (exited === undefined) {
      child.kill("SIGTERM");
    }
    const timer = setTimeout(() => child.kill("SIGKILL"), 20_000);
    const final = await exit;
    clearTimeout(timer);
    check(final.code === 0 && final.signal === null, "SIGTERM exits 0", `code=${final.code} signal=${final.signal}\n${log}`);
    if (failures > 0) {
      process.stdout.write(`--- server log ---\n${log}\n`);
    }
    rmSync(scratch, { recursive: true, force: true });
  }
}

try {
  await main();
} catch (error) {
  record("FAIL", "verification aborted", error instanceof Error ? error.stack : String(error));
}
process.stdout.write(`\n${failures === 0 ? "OK" : "FAILED"}: ${results.filter((r) => r.status === "PASS").length} passed, ${results.filter((r) => r.status === "SKIP").length} skipped, ${failures} failed\n`);
process.exit(failures === 0 ? 0 : 1);
