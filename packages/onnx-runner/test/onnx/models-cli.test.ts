import assert from "node:assert/strict";
import { spawn } from "node:child_process";
import { cpSync, existsSync, mkdirSync, mkdtempSync, readFileSync, readdirSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import test from "node:test";

import { KNOWN_MODELS } from "../../src/index.js";
import { runModelsCommand, type ModelsIo } from "../../src/models-cli.js";
import { CLI } from "../helpers.js";
import { startMockHf, type MockHf } from "./hf-mock.js";
import { buildTemplateFixture, writeHfCache, type TemplateFixture } from "./template-fixture.js";

const REPO = "fastino/GLiNER2.5-Decide";
const PIN = "9d1bfb848cd16d93ee56ccbc59cd92bf0f284d90";

interface Result {
  code: number | null;
  stdout: string;
  stderr: string;
}

function cli(args: string[], env: NodeJS.ProcessEnv = {}): Promise<Result> {
  return new Promise((resolve) => {
    const child = spawn(process.execPath, [CLI, ...args], {
      // HOME is a scratch directory so no real Hugging Face or gliner-runner cache is ever found.
      env: { PATH: process.env["PATH"] ?? "", HOME: mkdtempSync(join(tmpdir(), "gliner-home-")), ...env },
      stdio: ["ignore", "pipe", "pipe"],
    });
    let stdout = "";
    let stderr = "";
    child.stdout.on("data", (chunk: Buffer) => (stdout += chunk.toString()));
    child.stderr.on("data", (chunk: Buffer) => (stderr += chunk.toString()));
    child.once("exit", (code) => resolve({ code, stdout, stderr }));
  });
}

async function setup(): Promise<{ fixture: TemplateFixture; hub: MockHf; modelsDir: string; args: string[] }> {
  const fixture = buildTemplateFixture();
  const hub = await startMockHf({ repository: REPO, revision: PIN, files: fixture.sources });
  const modelsDir = join(fixture.root, "models");
  return { fixture, hub, modelsDir, args: ["--models-dir", modelsDir, "--templates-dir", fixture.templatesDir] };
}

const artifactDir = (modelsDir: string): string => join(modelsDir, "onnx", "fastino--GLiNER2.5-Decide", PIN);

test("install without --yes refuses when stdin is not a terminal and downloads nothing", async (t) => {
  const { hub, modelsDir, args } = await setup();
  t.after(() => hub.close());
  const result = await cli(["models", "install", "decide-340m", ...args], { HF_ENDPOINT: hub.endpoint });
  assert.equal(result.code, 1);
  assert.match(result.stderr, /refusing to download decide-340m@9d1bfb848cd1/);
  assert.match(result.stderr, /stdin is not a terminal, pass --yes/);
  assert.equal(result.stdout, "");
  assert.equal(hub.requests.length, 0);
  assert.equal(existsSync(modelsDir), false);
});

test("install --yes downloads, prepares byte-identically and prints one JSON line", async (t) => {
  const { fixture, hub, modelsDir, args } = await setup();
  t.after(() => hub.close());
  const result = await cli(["models", "install", "decide-340m", "--yes", ...args], { HF_ENDPOINT: hub.endpoint });
  assert.equal(result.code, 0, result.stderr);
  const lines = result.stdout.trim().split("\n");
  assert.equal(lines.length, 1);
  assert.deepEqual(JSON.parse(lines[0] as string), { model_id: "decide-340m", path: artifactDir(modelsDir), revision: PIN });
  assert.deepEqual(Object.keys(JSON.parse(lines[0] as string)), ["model_id", "path", "revision"]);
  assert.match(result.stderr, /decide-340m: downloading 3 files/);
  assert.match(result.stderr, /decide-340m: installed /);
  for (const [name, content] of Object.entries(fixture.artifact)) {
    assert.ok(readFileSync(join(artifactDir(modelsDir), name)).equals(content), name);
  }
  assert.deepEqual(readdirSync(join(modelsDir, "onnx", "fastino--GLiNER2.5-Decide")), [PIN, `${PIN}.gliner-runner.json`]);

  const requests = hub.requests.length;
  const listed = await cli(["models", "list", ...args]);
  assert.equal(listed.code, 0);
  const inventory = JSON.parse(listed.stdout);
  assert.equal(inventory.models_dir, modelsDir);
  const [first, second, third] = inventory.models;
  assert.deepEqual([first.model_id, first.installed, first.installed_revisions, first.template_available], ["decide-340m", true, [PIN], true]);
  assert.deepEqual([second.model_id, second.installed, second.template_available], ["decide-multilingual", false, false]);
  assert.deepEqual([third.model_id, third.installed], ["decide-1b", false]);

  // Installing again needs no approval and no network: the verified artifact is reused.
  const again = await cli(["models", "install", "decide-340m", ...args], { HF_ENDPOINT: hub.endpoint });
  assert.equal(again.code, 0, again.stderr);
  assert.equal(JSON.parse(again.stdout.trim()).path, artifactDir(modelsDir));
  assert.equal(hub.requests.length, requests);
});

test("install failures exit 1 and leave no staging behind; usage errors exit 2", async (t) => {
  const fixture = buildTemplateFixture();
  const hub = await startMockHf({ repository: REPO, revision: PIN, files: { ...fixture.sources, "tokenizer.json": Buffer.from("tampered") } });
  t.after(() => hub.close());
  const modelsDir = join(fixture.root, "models");
  const args = ["--models-dir", modelsDir, "--templates-dir", fixture.templatesDir];
  const failed = await cli(["models", "install", "decide-340m", "--yes", ...args], { HF_ENDPOINT: hub.endpoint });
  assert.equal(failed.code, 1);
  assert.match(failed.stderr, /error: .*(failed verification|bytes on the server)/);
  assert.equal(failed.stdout, "");
  // Only the processing record of the failed attempt remains; no staging, no partial artifact.
  assert.deepEqual(readdirSync(join(modelsDir, "onnx", "fastino--GLiNER2.5-Decide")), [`${PIN}.gliner-runner.json`]);
  assert.equal(JSON.parse(readFileSync(join(modelsDir, "onnx", "fastino--GLiNER2.5-Decide", `${PIN}.gliner-runner.json`), "utf8")).state, "failed");

  assert.equal((await cli(["models", "install", "mystery", "--yes", ...args])).code, 2);
  assert.equal((await cli(["models", "install", ...args])).code, 2);
  assert.equal((await cli(["models", "frobnicate"])).code, 2);
  assert.equal((await cli(["models", "install", "decide-340m", "--bogus"])).code, 2);
  const noTemplate = await cli(["models", "install", "decide-1b", "--yes", ...args], { HF_ENDPOINT: hub.endpoint });
  assert.equal(noTemplate.code, 1);
  assert.match(noTemplate.stderr, /no ONNX template for fastino\/GLiNER2\.5-Decide-1B@/);
  const help = await cli(["models", "--help"]);
  assert.equal(help.code, 0);
  assert.match(help.stdout, /install <model-id>/);
});

test("prepare --snapshot builds the artifact offline from a local snapshot", async () => {
  const fixture = buildTemplateFixture();
  const modelsDir = join(fixture.root, "models");
  // The endpoint is unreachable: nothing may use the network.
  const result = await cli(
    ["models", "prepare", "decide-340m", "--snapshot", fixture.snapshotDir, "--models-dir", modelsDir, "--templates-dir", fixture.templatesDir],
    { HF_ENDPOINT: "http://127.0.0.1:1" },
  );
  assert.equal(result.code, 0, result.stderr);
  assert.deepEqual(JSON.parse(result.stdout.trim()), { model_id: "decide-340m", path: artifactDir(modelsDir), revision: PIN });
  for (const [name, content] of Object.entries(fixture.artifact)) {
    assert.ok(readFileSync(join(artifactDir(modelsDir), name)).equals(content), name);
  }
  // Never overwrites.
  const again = await cli(
    ["models", "prepare", "decide-340m", "--snapshot", fixture.snapshotDir, "--models-dir", modelsDir, "--templates-dir", fixture.templatesDir],
  );
  assert.equal(again.code, 1);
  assert.match(again.stderr, /already exists; it is never overwritten/);
  // Without --snapshot the standard locations are searched; none holds the pinned snapshot here.
  const none = await cli(["models", "prepare", "decide-340m", "--models-dir", join(fixture.root, "other-models"), "--templates-dir", fixture.templatesDir]);
  assert.equal(none.code, 1);
  assert.match(none.stderr, /no verified local snapshot of fastino\/GLiNER2\.5-Decide@9d1bfb84/);
  const wrong = await cli(["models", "prepare", "decide-340m", "--snapshot", join(fixture.root, "models"), "--models-dir", join(fixture.root, "other-models"), "--templates-dir", fixture.templatesDir]);
  assert.equal(wrong.code, 1);
  assert.match(wrong.stderr, /does not match fastino\/GLiNER2\.5-Decide@/);
});

function fakeIo(options: {
  interactive: boolean;
  answer?: boolean;
  env?: NodeJS.ProcessEnv;
  freeBytes?: ModelsIo["freeBytes"];
}): ModelsIo & { out: string[]; err: string[]; questions: string[] } {
  const out: string[] = [];
  const err: string[] = [];
  const questions: string[] = [];
  return {
    env: { HOME: mkdtempSync(join(tmpdir(), "gliner-home-")), ...options.env },
    interactive: options.interactive,
    freeBytes: options.freeBytes,
    stdout: (text) => out.push(text),
    stderr: (text) => err.push(text),
    confirm: async (question) => {
      questions.push(question);
      return options.answer ?? false;
    },
    out,
    err,
    questions,
  };
}

test("an interactive terminal is asked for approval (y/N) before any download", async (t) => {
  const { fixture, hub, modelsDir, args } = await setup();
  t.after(() => hub.close());
  const env = { HF_ENDPOINT: hub.endpoint };

  const denied = fakeIo({ interactive: true, answer: false, env });
  assert.equal(await runModelsCommand(["install", "decide-340m", ...args], denied), 1);
  assert.equal(denied.questions.length, 1);
  assert.match(denied.questions[0] as string, /Download decide-340m@9d1bfb848cd1 \(.* GiB\) from http:\/\/127\.0\.0\.1:\d+ and prepare it .* in .*\? \[y\/N\] $/);
  assert.match(denied.err.join(""), /download not approved/);
  assert.equal(hub.requests.length, 0);
  assert.equal(existsSync(modelsDir), false);

  const approved = fakeIo({ interactive: true, answer: true, env });
  assert.equal(await runModelsCommand(["install", "decide-340m", ...args], approved), 0);
  assert.equal(approved.questions.length, 1);
  assert.deepEqual(JSON.parse(approved.out.join("").trim()), { model_id: "decide-340m", path: artifactDir(modelsDir), revision: PIN });
  assert.ok(readFileSync(join(artifactDir(modelsDir), "model.onnx")).equals(fixture.artifact["model.onnx"] as Buffer));

  // --yes skips the prompt entirely; non-interactive without --yes refuses.
  const other = await setup();
  t.after(() => other.hub.close());
  const yes = fakeIo({ interactive: true, answer: false, env: { HF_ENDPOINT: other.hub.endpoint } });
  assert.equal(await runModelsCommand(["install", "decide-340m", "-y", ...other.args], yes), 0);
  assert.deepEqual(yes.questions, []);
  const refused = fakeIo({ interactive: false, env: { HF_ENDPOINT: other.hub.endpoint } });
  const third = await setup();
  t.after(() => third.hub.close());
  assert.equal(await runModelsCommand(["install", "decide-340m", ...third.args], refused), 1);
  assert.deepEqual(refused.questions, []);
});

test("the models directory and templates directory may come from the environment", async (t) => {
  const { fixture, hub, modelsDir } = await setup();
  t.after(() => hub.close());
  const result = await cli(["models", "install", "decide-340m", "--yes"], {
    HF_ENDPOINT: hub.endpoint,
    GLINER_RUNNER_MODELS_DIR: modelsDir,
    GLINER_RUNNER_ONNX_TEMPLATES_DIR: fixture.templatesDir,
  });
  assert.equal(result.code, 0, result.stderr);
  assert.equal(JSON.parse(result.stdout.trim()).path, artifactDir(modelsDir));
});

const [M340, MULTI, M1B] = KNOWN_MODELS;
const GB = 1e9;

/**
 * Templates for all three curated models in one directory (synthetic files, the pinned revisions), and a mock
 * Hugging Face hub serving the decide-340m files. `big` records 2.5 GB of weights in each template so the plan
 * shows whole GB; those templates only plan, they cannot be downloaded.
 */
async function setupAll(t: { after(fn: () => Promise<void>): void }, big = false) {
  const root = mkdtempSync(join(tmpdir(), "gliner-all-"));
  const templatesDir = join(root, "templates");
  const fixtures = KNOWN_MODELS.map((model) =>
    buildTemplateFixture({
      model,
      ...(big ? { mutate: (template: Record<string, any>) => void (template["source_files"][0].size = 2.5 * GB) } : {}),
    }),
  );
  for (const fixture of fixtures) {
    const target = join(templatesDir, fixture.model.repository.replace("/", "--"), fixture.model.pinnedRevision);
    mkdirSync(target, { recursive: true });
    cpSync(fixture.templateDir, target, { recursive: true });
  }
  const hub = await startMockHf({ repository: REPO, revision: PIN, files: (fixtures[0] as TemplateFixture).sources });
  t.after(() => hub.close());
  const modelsDir = join(root, "models");
  return { root, fixtures, hub, modelsDir, args: ["--models-dir", modelsDir, "--templates-dir", templatesDir] };
}

test("install --all plans one line per model (installed, found locally, download), then installs in turn", async (t) => {
  const { root, fixtures, hub, modelsDir, args } = await setupAll(t);
  const [first, second, third] = fixtures as [TemplateFixture, TemplateFixture, TemplateFixture];
  // decide-multilingual is installed already; decide-1b has a verified snapshot in a Hugging Face cache.
  const prepared = await runModelsCommand(["prepare", "decide-multilingual", "--snapshot", second.snapshotDir, ...args], fakeIo({ interactive: false }));
  assert.equal(prepared, 0);
  const hubDir = join(root, "hf-hub");
  const snapshot = writeHfCache(hubDir, (M1B as { repository: string }).repository, (M1B as { pinnedRevision: string }).pinnedRevision, third.sources);
  const env = { HF_ENDPOINT: hub.endpoint, HF_HUB_CACHE: hubDir };

  const io = fakeIo({ interactive: false, env });
  assert.equal(await runModelsCommand(["install", "--all", "--yes", ...args], io), 0);
  const err = io.err.join("");
  const multiPath = join(modelsDir, "onnx", "fastino--GLiNER2.5-multi-Decide", (MULTI as { pinnedRevision: string }).pinnedRevision);
  const plan = err.split("\n").slice(0, 4);
  assert.match(plan[0] as string, /^decide-340m: will download \d+\.\d GB \(needs about \d+\.\d GB disk\)$/);
  assert.equal(plan[1], "decide-multilingual: already installed");
  assert.equal(plan[2], `decide-1b: found on this machine at ${snapshot} \u2014 no download needed`);
  assert.match(plan[3] as string, /^Total: will download \d+\.\d GB \(needs about \d+\.\d GB disk\)$/);
  assert.match(err, new RegExp(`^decide-multilingual: already installed \\(${multiPath}\\)$`, "m"));
  assert.match(err, /decide-340m: downloading 3 files/);
  assert.match(err, /decide-1b: using the verified local snapshot/);
  assert.deepEqual(io.questions, []);
  const printed = io.out.map((text) => JSON.parse(text));
  assert.deepEqual(printed.map((entry) => entry.model_id), ["decide-340m", "decide-multilingual", "decide-1b"]);
  assert.equal(printed[1].path, multiPath);
  assert.ok(readFileSync(join(printed[0].path, "model.onnx")).equals(first.artifact["model.onnx"] as Buffer));
  assert.ok(existsSync(join(printed[2].path, "manifest.json")));
  // Only decide-340m touched the network.
  assert.ok(hub.requests.length > 0);
  assert.ok(hub.requests.every((request) => request.url.includes("GLiNER2.5-Decide/resolve") || request.url.includes("GLiNER2.5-Decide%2F")));

  // Everything is installed now: nothing to approve, no network, even without a terminal or --yes.
  const requests = hub.requests.length;
  const again = fakeIo({ interactive: false, env });
  assert.equal(await runModelsCommand(["install", "--all", ...args], again), 0);
  assert.match(again.err.join(""), /^decide-340m: already installed\ndecide-multilingual: already installed\ndecide-1b: already installed\nTotal: nothing to download\n/);
  assert.match(again.err.join(""), /^decide-340m: already installed \(/m);
  assert.equal(again.out.length, 3);
  assert.equal(hub.requests.length, requests);
});

test("install accepts several ids, installs each once, and rejects bad combinations", async (t) => {
  const { hub, modelsDir, fixtures, args } = await setupAll(t);
  const env = { HF_ENDPOINT: hub.endpoint, HF_HUB_CACHE: join(modelsDir, "none") };
  await runModelsCommand(["prepare", "decide-1b", "--snapshot", (fixtures[2] as TemplateFixture).snapshotDir, ...args], fakeIo({ interactive: false }));
  const io = fakeIo({ interactive: false, env });
  assert.equal(await runModelsCommand(["install", "decide-1b", "decide-340m", "decide-1b", "-y", ...args], io), 0);
  assert.deepEqual(io.out.map((text) => JSON.parse(text).model_id), ["decide-1b", "decide-340m"]);
  assert.match(io.err.join(""), /^decide-1b: already installed\ndecide-340m: will download /);
  assert.doesNotMatch(io.err.join(""), /decide-multilingual/);

  for (const argv of [
    ["install", "--all", "decide-340m", ...args],
    ["install", "decide-340m", "mystery", ...args],
    ["install", "decide-340m", "decide-1b", "--revision", PIN, ...args],
  ]) {
    const bad = fakeIo({ interactive: false, env });
    assert.equal(await runModelsCommand(argv, bad), 2, argv.join(" "));
    assert.deepEqual(bad.out, []);
  }
  assert.equal(hub.requests.length > 0, true); // the valid install above downloaded decide-340m only
});

test("install --all without a terminal and without --yes refuses after the plan, before any download", async (t) => {
  const { hub, modelsDir, args } = await setupAll(t, true);
  const io = fakeIo({ interactive: false, env: { HF_ENDPOINT: hub.endpoint, HF_HUB_CACHE: join(modelsDir, "none") } });
  assert.equal(await runModelsCommand(["install", "--all", ...args], io), 1);
  // Each template records 2.5 GB of weights; the prepared outputs are a few KB. Downloads are removed one by one,
  // so the total needs room for the largest download only.
  assert.match(io.err.join(""), /^decide-340m: will download 2\.5 GB \(needs about 2\.5 GB disk\)\n/);
  assert.match(io.err.join(""), /\ndecide-1b: will download 2\.5 GB \(needs about 2\.5 GB disk\)\n/);
  assert.match(io.err.join(""), /\nTotal: will download 7\.5 GB \(needs about 2\.5 GB disk\)\n/);
  assert.match(io.err.join(""), /error: refusing to download 7\.5 GB without approval; stdin is not a terminal, rerun with --yes\n$/);
  assert.deepEqual(io.out, []);
  assert.deepEqual(io.questions, []);
  assert.equal(hub.requests.length, 0);
  assert.equal(existsSync(modelsDir), false);
});

test("install --all asks once in a terminal: no downloads nothing, yes installs", async (t) => {
  const { hub, modelsDir, args } = await setupAll(t, true);
  const env = { HF_ENDPOINT: hub.endpoint, HF_HUB_CACHE: join(modelsDir, "none") };
  const denied = fakeIo({ interactive: true, answer: false, env });
  assert.equal(await runModelsCommand(["install", "--all", ...args], denied), 1);
  assert.deepEqual(denied.questions, ["Download 7.5 GB now? [y/N] "]);
  assert.match(denied.err.join(""), /error: download not approved\n$/);
  assert.deepEqual(denied.out, []);
  assert.equal(hub.requests.length, 0);
  assert.equal(existsSync(modelsDir), false);

  // Answering yes starts the downloads (the scaled-up template cannot verify, so decide-340m fails after approval).
  const approved = fakeIo({ interactive: true, answer: true, env });
  assert.equal(await runModelsCommand(["install", "--all", ...args], approved), 1);
  assert.equal(approved.questions.length, 1);
  assert.ok(hub.requests.length > 0);
});

test("install --all with an approved prompt installs every model", async (t) => {
  const { hub, root, fixtures, args } = await setupAll(t);
  const hubDir = join(root, "hf-hub");
  for (const fixture of [fixtures[1], fixtures[2]] as TemplateFixture[]) {
    writeHfCache(hubDir, fixture.model.repository, fixture.model.pinnedRevision, fixture.sources);
  }
  const io = fakeIo({ interactive: true, answer: true, env: { HF_ENDPOINT: hub.endpoint, HF_HUB_CACHE: hubDir } });
  assert.equal(await runModelsCommand(["install", "--all", ...args], io), 0);
  assert.equal(io.questions.length, 1);
  assert.match(io.questions[0] as string, /^Download \d+\.\d GB now\? \[y\/N\] $/);
  assert.equal(io.out.length, 3);
});

test("install --all refuses when the free disk space is not enough, before any prompt or download", async (t) => {
  const { hub, modelsDir, args } = await setupAll(t, true);
  const env = { HF_ENDPOINT: hub.endpoint, HF_HUB_CACHE: join(modelsDir, "none") };
  const asked: string[] = [];
  const io = fakeIo({
    interactive: true,
    answer: true,
    env,
    freeBytes: async (directory) => {
      asked.push(directory);
      return 1 * GB;
    },
  });
  assert.equal(await runModelsCommand(["install", "--all", "--yes", ...args], io), 1);
  assert.match(io.err.join(""), /error: not enough free disk space in .*: need about 2\.6 GB, 1\.0 GB available\n$/);
  assert.equal(asked.length, 1);
  assert.notEqual(asked[0], modelsDir); // the models directory does not exist yet: its nearest existing parent is checked
  assert.ok(existsSync(asked[0] as string));
  assert.deepEqual(io.questions, []);
  assert.deepEqual(io.out, []);
  assert.equal(hub.requests.length, 0);

  // Enough space: the same plan proceeds to the prompt.
  const enough = fakeIo({ interactive: true, answer: false, env, freeBytes: async () => 5 * GB });
  assert.equal(await runModelsCommand(["install", "--all", ...args], enough), 1);
  assert.equal(enough.questions.length, 1);
});
