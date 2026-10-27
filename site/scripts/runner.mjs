// Starts the built runner (packages/onnx-runner/dist) on a free port for scripts that need real model answers.
// The models come from GLINER_RUNNER_MODELS_DIR (or the user cache), as for `gliner-runner serve`.
import { spawn } from "node:child_process";
import { mkdtempSync } from "node:fs";
import { createServer } from "node:net";
import { tmpdir } from "node:os";
import { dirname, join, resolve } from "node:path";
import { fileURLToPath } from "node:url";

const root = resolve(dirname(fileURLToPath(import.meta.url)), "../..");

const freePort = () =>
  new Promise((done) => {
    const probe = createServer();
    probe.listen(0, "127.0.0.1", () => {
      const { port } = probe.address();
      probe.close(() => done(port));
    });
  });

/**
 * `cli` lets a caller use a runner built elsewhere (a worktree has no dist until it is built), and `args` adds
 * serve flags such as `--cors-origin`.
 */
export async function startRunner({ cli = join(root, "packages/onnx-runner/dist/src/cli.js"), args = [] } = {}) {
  const port = await freePort();
  const child = spawn(process.execPath, [cli, "serve", "--port", String(port), ...args], {
    env: { ...process.env, XDG_CONFIG_HOME: mkdtempSync(join(tmpdir(), "runner-config-")) },
    stdio: ["ignore", "ignore", "inherit"],
  });
  const base = `http://127.0.0.1:${port}`;
  for (let tries = 0; !(await fetch(`${base}/healthz`).then((r) => r.ok, () => false)); tries += 1) {
    if (tries > 200) throw new Error("the runner did not start");
    await new Promise((r) => setTimeout(r, 50));
  }
  return { base, stop: () => child.kill("SIGTERM") };
}

/** One classification, as the playground sends it. */
export async function classify(base, { model, text, labels, descriptions, several = false, threshold }) {
  const task = {
    labels,
    ...(descriptions ? { label_descriptions: descriptions } : {}),
    ...(several ? { min_labels: 0, max_labels: null } : {}),
    ...(threshold === undefined ? {} : { threshold }),
  };
  const reply = await fetch(`${base}/v1/infer`, {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify({
      model, backend: "onnx", precision: "fp32", operation: "classify", text,
      schema: { kind: "classification", tasks: { category: task } },
      ...(threshold === undefined ? {} : { options: { threshold } }),
    }),
  });
  const body = await reply.json();
  if (!reply.ok) throw new Error(JSON.stringify(body.detail));
  return body.output.category;
}
