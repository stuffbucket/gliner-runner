import assert from "node:assert/strict";
import { spawn, spawnSync, type ChildProcess } from "node:child_process";
import { mkdirSync, mkdtempSync, readFileSync, writeFileSync } from "node:fs";
import { createServer } from "node:net";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { setTimeout as sleep } from "node:timers/promises";
import test from "node:test";

import { BUILD_INFO } from "../src/build-info.js";
import { CLI, freePort } from "./helpers.js";

const scratchConfig = (): string => mkdtempSync(join(tmpdir(), "gliner-config-"));

const PACKAGE_VERSION = (
  JSON.parse(readFileSync(new URL("../../package.json", import.meta.url), "utf8")) as { version: string }
).version;

function cli(args: string[]): { status: number | null; stdout: string; stderr: string } {
  const result = spawnSync(process.execPath, [CLI, ...args], {
    env: { PATH: process.env["PATH"] ?? "" },
    encoding: "utf8",
  });
  return { status: result.status, stdout: result.stdout, stderr: result.stderr };
}

interface Running {
  child: ChildProcess;
  stderr: () => string;
  exit: Promise<{ code: number | null; signal: NodeJS.Signals | null }>;
}

function run(args: string[], env: NodeJS.ProcessEnv = {}): Running {
  const child = spawn(process.execPath, [CLI, ...args], {
    // A scratch XDG_CONFIG_HOME so the user's real settings.json is never read.
    env: { PATH: process.env["PATH"] ?? "", XDG_CONFIG_HOME: scratchConfig(), ...env },
    stdio: ["ignore", "pipe", "pipe"],
  });
  let stderr = "";
  child.stderr?.on("data", (chunk: Buffer) => (stderr += chunk.toString()));
  child.stdout?.resume();
  const exit = new Promise<{ code: number | null; signal: NodeJS.Signals | null }>((resolve) => {
    child.once("exit", (code, signal) => resolve({ code, signal }));
  });
  return { child, stderr: () => stderr, exit };
}

async function waitForHealth(port: number, running: Running): Promise<Response> {
  const deadline = Date.now() + 10_000;
  for (;;) {
    try {
      const response = await fetch(`http://127.0.0.1:${port}/healthz`, { signal: AbortSignal.timeout(500) });
      if (response.ok) {
        return response;
      }
    } catch {
      // not listening yet
    }
    if (running.child.exitCode !== null || Date.now() > deadline) {
      throw new Error(`server did not become healthy: ${running.stderr()}`);
    }
    await sleep(25);
  }
}

test("serve binds quickly, answers /healthz without a model and exits 0 on SIGTERM", async () => {
  const port = await freePort();
  const running = run(["serve", "--port", String(port)]);
  const started = Date.now();
  const health = await waitForHealth(port, running);
  assert.ok(Date.now() - started < 5000);
  assert.deepEqual(await health.json(), { status: "ok", accepting_requests: true });

  running.child.kill("SIGTERM");
  const { code, signal } = await running.exit;
  assert.equal(signal, null);
  assert.equal(code, 0);
  assert.match(running.stderr(), /Application shutdown complete/);
});

test("SIGINT also shuts down cleanly", async () => {
  const port = await freePort();
  const running = run(["serve", "--port", String(port)]);
  await waitForHealth(port, running);
  running.child.kill("SIGINT");
  assert.equal((await running.exit).code, 0);
  assert.match(running.stderr(), /Application shutdown complete/);
});

test("a bind failure exits non-zero promptly with a diagnostic", async () => {
  const blocker = createServer();
  await new Promise<void>((resolve) => blocker.listen(0, "127.0.0.1", resolve));
  const { port } = blocker.address() as { port: number };
  try {
    const running = run(["serve", "--port", String(port)]);
    const { code } = await running.exit;
    assert.notEqual(code, 0);
    assert.match(running.stderr(), /failed to listen on 127\.0\.0\.1:\d+: .*EADDRINUSE/);
  } finally {
    await new Promise<void>((resolve) => blocker.close(() => resolve()));
  }
});

test("invalid arguments fail fast with usage on stderr", async () => {
  for (const args of [["serve", "--bogus"], ["serve", "--port", "0"], ["serve", "extra"]]) {
    const running = run(args);
    const { code } = await running.exit;
    assert.equal(code, 2, args.join(" "));
    assert.match(running.stderr(), /^error: /);
    assert.match(running.stderr(), /Usage: gliner-runner serve/);
  }
});

test("an invalid environment value names the variable", async () => {
  const running = run(["serve"], { GLINER_RUNNER_PORT: "abc" });
  assert.equal((await running.exit).code, 2);
  assert.match(running.stderr(), /GLINER_RUNNER_PORT must be an integer/);
});

test("precedence is flag over environment over default", async () => {
  const envPort = await freePort();
  const fromEnv = run(["serve"], { GLINER_RUNNER_PORT: String(envPort), GLINER_RUNNER_DEVICE: "env-device" });
  await waitForHealth(envPort, fromEnv);
  const envModels = (await (await fetch(`http://127.0.0.1:${envPort}/v1/models`)).json()) as { device: string };
  assert.equal(envModels.device, "env-device");
  fromEnv.child.kill("SIGTERM");
  await fromEnv.exit;

  const flagPort = await freePort();
  const fromFlag = run(["serve", "--port", String(flagPort), "--device", "flag-device"], {
    GLINER_RUNNER_PORT: String(envPort),
    GLINER_RUNNER_DEVICE: "env-device",
  });
  await waitForHealth(flagPort, fromFlag);
  const flagModels = (await (await fetch(`http://127.0.0.1:${flagPort}/v1/models`)).json()) as { device: string };
  assert.equal(flagModels.device, "flag-device");
  fromFlag.child.kill("SIGTERM");
  await fromFlag.exit;
});

async function allowedOrigin(port: number, origin: string): Promise<string | null> {
  const response = await fetch(`http://127.0.0.1:${port}/healthz`, { headers: { origin } });
  return response.headers.get("access-control-allow-origin");
}

test("serve allows the default localhost origins until settings.json says otherwise", async () => {
  const port = await freePort();
  const running = run(["serve", "--port", String(port)]);
  await waitForHealth(port, running);
  assert.equal(await allowedOrigin(port, "http://localhost:5173"), "http://localhost:5173");
  assert.equal(await allowedOrigin(port, "https://stuffbucket.github.io"), null);
  assert.match(running.stderr(), /Browser origins allowed \(CORS\): http:\/\/localhost:3000, /);
  running.child.kill("SIGTERM");
  await running.exit;
});

test("serve reads saved origins from settings.json; --cors-origin replaces them for one run", async () => {
  const config = scratchConfig();
  mkdirSync(join(config, "gliner-runner"));
  writeFileSync(
    join(config, "gliner-runner", "settings.json"),
    JSON.stringify({ cors_origins: ["https://stuffbucket.github.io"] }),
  );
  const savedPort = await freePort();
  const saved = run(["serve", "--port", String(savedPort)], { XDG_CONFIG_HOME: config });
  await waitForHealth(savedPort, saved);
  assert.equal(await allowedOrigin(savedPort, "https://stuffbucket.github.io"), "https://stuffbucket.github.io");
  assert.equal(await allowedOrigin(savedPort, "http://localhost:5173"), null);
  saved.child.kill("SIGTERM");
  await saved.exit;

  const flagPort = await freePort();
  const flagged = run(["serve", "--port", String(flagPort), "--cors-origin", "http://localhost:9999"], {
    XDG_CONFIG_HOME: config,
  });
  await waitForHealth(flagPort, flagged);
  assert.equal(await allowedOrigin(flagPort, "http://localhost:9999"), "http://localhost:9999");
  assert.equal(await allowedOrigin(flagPort, "https://stuffbucket.github.io"), null);
  flagged.child.kill("SIGTERM");
  await flagged.exit;
});

test("serve refuses to start with a malformed settings.json and names the file", async () => {
  const config = scratchConfig();
  mkdirSync(join(config, "gliner-runner"));
  writeFileSync(join(config, "gliner-runner", "settings.json"), JSON.stringify({ cors_origins: ["http://x.test/"] }));
  const running = run(["serve"], { XDG_CONFIG_HOME: config });
  assert.equal((await running.exit).code, 2);
  assert.match(running.stderr(), /settings\.json: cors_origins must list origins like/);
});

for (const flag of ["--version", "-v", "--v"]) {
  test(`${flag} prints the version, identity and evaluation receipt SHAs without models`, () => {
    const result = cli([flag]);
    assert.equal(result.status, 0, result.stderr);
    assert.deepEqual(result.stdout.trimEnd().split("\n"), [
      `gliner-runner-onnx ${PACKAGE_VERSION}`,
      `identity sha256: ${BUILD_INFO.identitySha256}`,
      `eval receipt sha256: ${BUILD_INFO.evalReceiptSha256}`,
      `eval run id: ${BUILD_INFO.evalRunId}`,
    ]);
  });
}

test("--version --json reports the embedded receipt identity", () => {
  const result = cli(["--version", "--json"]);
  assert.equal(result.status, 0, result.stderr);
  const info = JSON.parse(result.stdout) as Record<string, string>;
  assert.deepEqual(Object.keys(info).sort(), [
    "eval_receipt_sha256",
    "eval_run_id",
    "identity_sha256",
    "node",
    "onnxruntime",
    "version",
  ]);
  assert.equal(info["version"], PACKAGE_VERSION);
  assert.equal(info["identity_sha256"], BUILD_INFO.identitySha256);
  assert.equal(info["eval_receipt_sha256"], BUILD_INFO.evalReceiptSha256);
  assert.equal(info["eval_run_id"], BUILD_INFO.evalRunId);
  assert.equal(info["node"], process.version);
  assert.notEqual(info["onnxruntime"], "unknown");
});

test("--version rejects unexpected arguments", () => {
  const result = cli(["--version", "--bogus"]);
  assert.equal(result.status, 2);
  assert.match(result.stderr, /only accepts --json/);
});
