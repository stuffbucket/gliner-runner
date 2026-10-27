// promptfoo `javascript` assert for the hard tier: takes the script out of the answer, runs it in a
// sandbox against a real runner, and scores what it prints. config: { task: <id in scorers.cjs> }.
//
// Sandbox (macOS sandbox-exec): a fresh temp dir as cwd and HOME, writes only there and in the user temp
// dir, no network except 127.0.0.1:<runner port>, a wall-clock timeout, and a minimal environment. Scripts
// run one at a time (a lock directory) so the runner's /metrics counters belong to one script.
"use strict";
const { spawn } = require("node:child_process");
const fs = require("node:fs");
const os = require("node:os");
const path = require("node:path");
const { TASKS } = require("./scorers.cjs");

const RUNNER = process.env.RUNNER_URL || "http://127.0.0.1:8099";
const PORT = new URL(RUNNER).port || "80";
const ARTIFACTS = process.env.HARD_ARTIFACTS || path.join(os.tmpdir(), "gr-hard-artifacts");
const LOCK = process.env.HARD_LOCK || path.join(os.tmpdir(), "gr-hard-runner.lock");
const DEFAULT_TIMEOUT_MS = 120000;

const JS = new Set(["js", "javascript", "mjs", "cjs", "node", "nodejs", "ts", "typescript"]);
const SH = new Set(["sh", "bash", "shell", "zsh"]);

/** The longest fenced code block in a language we can run. */
function extractScript(output) {
  const blocks = [];
  const re = /```([\w+-]*)[^\n]*\n([\s\S]*?)```/g;
  for (let m = re.exec(output); m; m = re.exec(output)) {
    let lang = m[1].toLowerCase();
    const code = m[2];
    if (!lang) lang = /^#!.*\b(ba|z)?sh\b/.test(code) || /^\s*curl\s/m.test(code) ? "sh" : "js";
    if (JS.has(lang)) blocks.push({ language: "node", ts: lang === "ts" || lang === "typescript", code });
    else if (SH.has(lang)) blocks.push({ language: "shell", code });
  }
  return blocks.sort((a, b) => b.code.length - a.code.length)[0] || null;
}

// The user temp dir stays writable because macOS `mktemp -d` uses it whatever TMPDIR says.
const SYSTEM_TMP = fs.realpathSync(os.tmpdir());

function profile(dir) {
  const q = (s) => JSON.stringify(s);
  return `(version 1)
(allow default)
(deny network*)
(allow network-outbound (remote ip ${q(`localhost:${PORT}`)}))
(allow network* (local unix-socket))
(deny file-write*)
(allow file-write* (subpath ${q(dir)}) (subpath ${q(SYSTEM_TMP)}) (literal "/dev/null") (literal "/dev/stdout") (literal "/dev/stderr") (regex #"^/dev/fd/") (regex #"^/dev/tty"))
`;
}

const alive = (pid) => {
  try {
    process.kill(pid, 0);
    return true;
  } catch (error) {
    return error.code === "EPERM";
  }
};

/** A lock directory holding the owner's pid; a lock whose owner has exited is taken over. */
async function withLock(fn) {
  for (;;) {
    try {
      fs.mkdirSync(LOCK);
      fs.writeFileSync(path.join(LOCK, "pid"), String(process.pid));
      break;
    } catch {
      try {
        const owner = Number(fs.readFileSync(path.join(LOCK, "pid"), "utf8"));
        const age = Date.now() - fs.statSync(LOCK).mtimeMs;
        if ((owner && !alive(owner)) || (!owner && age > 5000)) fs.rmSync(LOCK, { recursive: true, force: true });
      } catch {}
      await new Promise((r) => setTimeout(r, 250));
    }
  }
  try {
    return await fn();
  } finally {
    fs.rmSync(LOCK, { recursive: true, force: true });
  }
}

async function metrics() {
  const text = await (await fetch(`${RUNNER}/metrics`)).text();
  const get = (name) => {
    const m = text.match(new RegExp(`metric="${name}"\\} (\\d+)`));
    return m ? Number(m[1]) : 0;
  };
  return { rejected: get("requests_rejected_total"), submitted: get("requests_submitted_total") };
}

/**
 * Other traffic for tasks that set `busy`: keeps `concurrency` requests in flight on decide-340m so the
 * queue (256) stays full and the script's requests get 429 queue_full. Returns a stop function.
 */
function startLoad(concurrency) {
  let stopped = false;
  const text = "Other job: ".concat("a shared server is busy with requests from several clients at the same time. ".repeat(6));
  const body = JSON.stringify({
    model: "decide-340m", backend: "onnx", precision: "fp32", operation: "classify", text,
    schema: { kind: "classification", tasks: { topic: { labels: ["news", "sport", "weather", "music", "travel"] } } },
  });
  const loop = async () => {
    while (!stopped) {
      try {
        const r = await fetch(`${RUNNER}/v1/infer`, { method: "POST", headers: { "content-type": "application/json" }, body });
        await r.arrayBuffer();
      } catch {
        await new Promise((ok) => setTimeout(ok, 20));
      }
    }
  };
  const loops = Array.from({ length: concurrency }, loop);
  return async () => {
    stopped = true;
    await Promise.all(loops);
  };
}

async function waitForRejections(base, ms) {
  const end = Date.now() + ms;
  while (Date.now() < end) {
    if ((await metrics()).rejected > base) return true;
    await new Promise((ok) => setTimeout(ok, 50));
  }
  return false;
}

function run(cmd, args, options, timeoutMs) {
  return new Promise((resolve) => {
    const child = spawn(cmd, args, { ...options, stdio: ["ignore", "pipe", "pipe"], detached: true });
    let stdout = "";
    let stderr = "";
    let timedOut = false;
    child.stdout.on("data", (d) => (stdout += d));
    child.stderr.on("data", (d) => (stderr += d));
    const timer = setTimeout(() => {
      timedOut = true;
      try {
        process.kill(-child.pid, "SIGKILL");
      } catch {}
    }, timeoutMs);
    child.on("close", (code) => {
      clearTimeout(timer);
      resolve({ code, stdout, stderr, timedOut });
    });
  });
}

/** The JSON a script printed: the whole stdout, else the last line or block that parses. */
function parseStdout(stdout) {
  const text = stdout.trim();
  try {
    return JSON.parse(text);
  } catch {}
  for (let i = text.length - 1; i >= 0; i--) {
    if (text[i] !== "{" && text[i] !== "[") continue;
    try {
      return JSON.parse(text.slice(i));
    } catch {}
  }
  for (const start of ["[", "{"]) {
    const i = text.indexOf(start);
    if (i >= 0) {
      try {
        return JSON.parse(text.slice(i));
      } catch {}
    }
  }
  return undefined;
}

async function check(output, context) {
  const id = context && context.config && context.config.task;
  const task = TASKS[id];
  if (!task) return { pass: false, score: 0, reason: `unknown hard task ${id}` };
  const script = extractScript(output);
  if (!script) return { pass: false, score: 0, reason: "no runnable code block (js or sh) in the answer" };

  const dir = fs.realpathSync(fs.mkdtempSync(path.join(os.tmpdir(), "gr-hard-")));
  for (const [name, content] of Object.entries(task.files())) fs.writeFileSync(path.join(dir, name), content);
  const file = script.language === "shell" ? "script.sh" : script.ts ? "script.ts" : "script.js";
  fs.writeFileSync(path.join(dir, file), script.code);
  fs.writeFileSync(path.join(dir, ".sandbox.sb"), profile(dir));
  const cmd = script.language === "shell" ? "/bin/bash" : process.execPath;
  const env = { PATH: "/usr/bin:/bin:/usr/sbin:/sbin", HOME: dir, TMPDIR: dir, LANG: "en_US.UTF-8" };

  const result = await withLock(async () => {
    const before = await metrics();
    let stopLoad = null;
    if (task.busy) {
      stopLoad = startLoad(task.busy.concurrency);
      await waitForRejections(before.rejected, 10000);
      setTimeout(() => stopLoad && stopLoad(), task.busy.ms);
    }
    const ran = await run("/usr/bin/sandbox-exec", ["-f", path.join(dir, ".sandbox.sb"), cmd, file], { cwd: dir, env }, task.timeoutMs || DEFAULT_TIMEOUT_MS);
    if (stopLoad) await stopLoad();
    const after = await metrics();
    return { ...ran, metrics: { rejected: after.rejected - before.rejected, submitted: after.submitted - before.submitted } };
  });

  // Keep the script and its output for reading failures (tuning tasks only; see README).
  try {
    const keep = path.join(ARTIFACTS, `${id}-${Date.now()}-${Math.random().toString(36).slice(2, 6)}`);
    fs.mkdirSync(keep, { recursive: true });
    fs.writeFileSync(path.join(keep, file), script.code);
    fs.writeFileSync(path.join(keep, "stdout.txt"), result.stdout);
    fs.writeFileSync(path.join(keep, "stderr.txt"), result.stderr);
  } catch {}
  fs.rmSync(dir, { recursive: true, force: true });

  const tail = (s) => s.trim().split("\n").slice(-3).join(" | ").slice(0, 300);
  if (result.timedOut) return { pass: false, score: 0, reason: `timed out; stderr: ${tail(result.stderr)}` };
  const parsed = parseStdout(result.stdout);
  if (parsed === undefined && id !== "shell-tsv") {
    return { pass: false, score: 0, reason: `exit ${result.code}, no JSON on stdout; stderr: ${tail(result.stderr)} stdout: ${tail(result.stdout)}` };
  }
  try {
    const v = await task.score(parsed, { ...result, runner: RUNNER, language: script.language });
    const reason = `${v.reason}${result.code ? ` (exit ${result.code}; stderr: ${tail(result.stderr)})` : ""}`;
    return { pass: v.pass, score: v.pass ? 1 : 0, reason: `[${script.language}, ${result.metrics.submitted} submitted] ${reason}` };
  } catch (error) {
    return { pass: false, score: 0, reason: `scoring failed: ${error.message}` };
  }
}

module.exports = { check, extractScript, parseStdout, startLoad };
