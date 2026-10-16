import assert from "node:assert/strict";
import { spawn, type ChildProcess } from "node:child_process";
import { mkdtempSync, readFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import test from "node:test";

import { CLI, freePort } from "./helpers.js";
import { startMockHf } from "./onnx/hf-mock.js";
import { buildTemplateFixture } from "./onnx/template-fixture.js";

const REPO = "fastino/GLiNER2.5-Decide";
const PIN = "9d1bfb848cd16d93ee56ccbc59cd92bf0f284d90";
const SITE = "http://site.test/gliner-runner/playground/";

interface Child {
  process: ChildProcess;
  stdout: () => string;
  stderr: () => string;
  exit: Promise<number | null>;
}

function run(args: string[], env: NodeJS.ProcessEnv): Child {
  const child = spawn(process.execPath, [CLI, ...args], { env: { PATH: process.env["PATH"] ?? "", ...env }, stdio: ["ignore", "pipe", "pipe"] });
  let stdout = "";
  let stderr = "";
  child.stdout?.on("data", (chunk: Buffer) => (stdout += chunk.toString()));
  child.stderr?.on("data", (chunk: Buffer) => (stderr += chunk.toString()));
  const exit = new Promise<number | null>((resolve) => child.once("exit", (code) => resolve(code)));
  return { process: child, stdout: () => stdout, stderr: () => stderr, exit };
}

async function line(child: Child, pattern: RegExp): Promise<string> {
  const deadline = Date.now() + 60_000;
  for (;;) {
    const match = pattern.exec(child.stdout());
    if (match !== null) {
      return match[0];
    }
    if (child.process.exitCode !== null || Date.now() > deadline) {
      throw new Error(`no ${pattern} on stdout; stderr: ${child.stderr()}`);
    }
    await new Promise((resolve) => setTimeout(resolve, 25));
  }
}

async function environment(t: test.TestContext): Promise<NodeJS.ProcessEnv> {
  const fixture = buildTemplateFixture();
  const hub = await startMockHf({ repository: REPO, revision: PIN, files: fixture.sources });
  t.after(() => hub.close());
  return {
    HOME: mkdtempSync(join(tmpdir(), "gliner-home-")),
    XDG_CONFIG_HOME: mkdtempSync(join(tmpdir(), "gliner-config-")),
    GLINER_RUNNER_MODELS_DIR: join(fixture.root, "models"),
    GLINER_RUNNER_ONNX_TEMPLATES_DIR: fixture.templatesDir,
    HF_ENDPOINT: hub.endpoint,
  };
}

const playgroundUrl = (port: number): string =>
  `${SITE}?${new URLSearchParams({ mode: "local", runner: `http://127.0.0.1:${port}` }).toString()}`;

test("playground sets everything up once, then reuses the running runner", async (t) => {
  const env = await environment(t);
  const port = await freePort();
  const first = run(["playground", "--port", String(port), "--yes", "--no-open", "--site", SITE], env);
  t.after(() => first.process.kill("SIGKILL"));
  assert.equal(await line(first, /^http\S+$/m), playgroundUrl(port));
  assert.match(first.stderr(), /Allowed http:\/\/site\.test to call the runner/);
  assert.match(first.stderr(), /Started a runner on http:\/\/127\.0\.0\.1:\d+/);
  assert.match(first.stderr(), /decide-340m is installed/);
  const saved = JSON.parse(readFileSync(join(env["XDG_CONFIG_HOME"] as string, "gliner-runner", "settings.json"), "utf8")) as { cors_origins: string[] };
  assert.ok(saved.cors_origins.includes("http://site.test"));
  assert.ok(saved.cors_origins.includes("http://localhost:5173"), "the defaults are kept");

  const health = await fetch(`http://127.0.0.1:${port}/healthz`, { headers: { origin: "http://site.test" } });
  assert.equal(health.headers.get("access-control-allow-origin"), "http://site.test");

  const second = run(["playground", "--port", String(port), "--no-open", "--site", SITE], env);
  assert.equal(await second.exit, 0, second.stderr());
  assert.equal(second.stdout().trim(), playgroundUrl(port));
  assert.match(second.stderr(), /Using the runner already serving/);
  assert.doesNotMatch(second.stderr(), /Downloading/);

  first.process.kill("SIGTERM");
  assert.equal(await first.exit, 0, first.stderr());
});

test("a runner that does not allow the playground is left alone and another one is started", async (t) => {
  const env = await environment(t);
  const port = await freePort();
  const other = run(["serve", "--port", String(port), "--cors-origin", "http://other.test"], env);
  t.after(() => other.process.kill("SIGKILL"));
  for (let tries = 0; ; tries += 1) {
    if ((await fetch(`http://127.0.0.1:${port}/healthz`).catch(() => undefined))?.ok) {
      break;
    }
    assert.ok(tries < 400, other.stderr());
    await new Promise((resolve) => setTimeout(resolve, 25));
  }
  const playground = run(["playground", "--port", String(port), "--yes", "--no-open", "--site", SITE], env);
  t.after(() => playground.process.kill("SIGKILL"));
  const url = await line(playground, /^http\S+$/m);
  const runner = new URL(url).searchParams.get("runner") as string;
  assert.notEqual(runner, `http://127.0.0.1:${port}`);
  assert.match(playground.stderr(), /was started before http:\/\/site\.test was allowed/);
  playground.process.kill("SIGTERM");
  assert.equal(await playground.exit, 0);
  other.process.kill("SIGTERM");
  await other.exit;
});

test("without a terminal and without --yes nothing is downloaded", async (t) => {
  const env = await environment(t);
  const port = await freePort();
  const playground = run(["playground", "--port", String(port), "--no-open", "--site", SITE], env);
  assert.equal(await playground.exit, 1);
  assert.match(playground.stderr(), /decide-340m is not installed; pass --yes/);
  assert.equal(playground.stdout(), "");
});

test("playground rejects bad options with usage", async () => {
  for (const args of [["--port", "0"], ["--site", "ftp://x"], ["extra"]]) {
    const playground = run(["playground", ...args], {});
    assert.equal(await playground.exit, 2, args.join(" "));
    assert.match(playground.stderr(), /Usage: gliner-runner playground/);
  }
});
