// Deterministic asserts for the gliner-runner skill eval (promptfoo `javascript` assertions).
// Each export takes (output, context) and returns a promptfoo GradingResult. Options come from the
// assertion's `config`. Request bodies found in an answer are sent to a real runner at RUNNER_URL.
"use strict";
const vm = require("node:vm");

const RUNNER = process.env.RUNNER_URL || "http://127.0.0.1:8097";
const NO_MATCH = /^(none|none of (these|the above)|other|no match|no team|nothing|irrelevant|not applicable|n\/a|unknown|chit[- ]?chat|small talk|no action|general)/i;

const result = (pass, reason) => ({ pass, score: pass ? 1 : 0, reason });

/**
 * Every balanced {...} or [...] in the text, in order of appearance. `quotes` are the string delimiters to
 * respect: JSON needs only `"`; JS and Python literals also use `'` and backticks.
 */
function balancedSpans(text, open, close, quotes) {
  const spans = [];
  for (let start = text.indexOf(open); start !== -1; start = text.indexOf(open, start + 1)) {
    let depth = 0;
    let quote = null;
    for (let i = start; i < text.length; i++) {
      const ch = text[i];
      if (quote) {
        if (ch === "\\") i++;
        else if (ch === quote) quote = null;
        continue;
      }
      if (quotes.includes(ch)) quote = ch;
      else if (ch === open) depth++;
      else if (ch === close && --depth === 0) {
        spans.push({ start, src: text.slice(start, i + 1) });
        break;
      }
    }
  }
  return spans;
}

const objectSpans = (text) => [
  ...new Set([...balancedSpans(text, "{", "}", '"'), ...balancedSpans(text, "{", "}", "\"'`")].map((s) => s.src)),
];

/**
 * Simple constants the answer defines before its request object, such as `LABELS = [...]` in Python or
 * `const LABELS = [...]` in JS, so a body that refers to them can still be sent.
 */
function constants(text) {
  const out = Object.create(null);
  const re = /(?:^|\n)[ \t]*(?:const |let |var )?([A-Za-z_]\w*)\s*(?::[^=\n]+)?=\s*(?=[[{"'])/g;
  for (let m = re.exec(text); m; m = re.exec(text)) {
    const at = m.index + m[0].length;
    const ch = text[at];
    let src = null;
    if (ch === "[" || ch === "{") {
      const span = balancedSpans(text.slice(at), ch, ch === "[" ? "]" : "}", "\"'`")[0];
      if (span && span.start === 0) src = span.src;
    } else {
      // A string constant only when the literal is the whole right-hand side (not `" ".join(...)`).
      const end = text.indexOf(ch, at + 1);
      const rest = end > at ? text.slice(end + 1, text.indexOf("\n", end) === -1 ? undefined : text.indexOf("\n", end)) : "";
      if (end > at && /^\s*;?\s*(\/\/.*|#.*)?$/.test(rest)) src = text.slice(at, end + 1);
    }
    if (src === null) continue;
    const value = evaluate(src, out);
    if (value !== undefined && value !== null) out[m[1]] = value;
  }
  return out;
}

const PYTHON = { True: true, False: false, None: null };
const SAMPLE = "sample text";

/** Stands in for unknown names: property access and calls give itself, and it serializes as a sample string. */
const placeholder = new Proxy(function () {}, {
  get: (_, key) => (key === "toJSON" ? () => SAMPLE : key === Symbol.toPrimitive ? () => SAMPLE : placeholder),
  apply: () => placeholder,
});

/** Evaluates a literal in a context where known constants resolve and any other name is a placeholder. */
function evaluate(src, known = Object.create(null)) {
  try {
    const sandbox = vm.createContext(
      new Proxy(Object.create(null), {
        has: () => true,
        get: (_, key) => {
          if (key === Symbol.unscopables) return undefined;
          if (key in known) return known[key];
          if (key in PYTHON) return PYTHON[key];
          return placeholder;
        },
      }),
    );
    const value = vm.runInContext(`(${src})`, sandbox, { timeout: 50 });
    return JSON.parse(JSON.stringify(value));
  } catch {
    return undefined;
  }
}

function parseObject(src, known) {
  try {
    return JSON.parse(src);
  } catch {}
  // JS or Python dict literals (unquoted keys, trailing commas, variables).
  const value = evaluate(src, known);
  return value === undefined ? null : value;
}

/** The infer requests (or batches) found in an answer, in order of appearance. */
function findRequests(rawOutput) {
  // Shell-escaped single quotes inside curl -d '...' ('"'"' and '\'') stand for one apostrophe.
  const output = rawOutput.replace(/'"'"'/g, "'").replace(/'\\''/g, "'");
  const known = constants(output);
  const found = [];
  for (const span of objectSpans(output)) {
    const value = parseObject(span, known);
    if (!value || typeof value !== "object") continue;
    if (Array.isArray(value.requests)) found.push({ batch: true, body: value, size: span.length });
    else if (value.schema && typeof value.schema === "object") found.push({ batch: false, body: value, size: span.length });
  }
  return found;
}

async function post(path, body) {
  const response = await fetch(RUNNER + path, {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify(body),
  });
  let json = null;
  try {
    json = await response.json();
  } catch {}
  return { status: response.status, json };
}

const isMulti = (task) => (task.min_labels ?? 1) !== 1 || (task.max_labels === undefined ? 1 : task.max_labels) !== 1;

/**
 * Finds the request body in the answer, sends it to the runner, and checks it.
 * config: { mode: "single"|"multi", requireDescriptions, labelsInclude: [], mustSelect: [], maxThreshold,
 *           minTasks, noMatch, model }
 */
async function postBody(output, context) {
  const cfg = (context && context.config) || {};
  const requests = findRequests(output);
  if (requests.length === 0) return result(false, "no JSON request body with a `schema` found in the answer");
  // Try each candidate, keep the first that the server accepts; report the last failure otherwise.
  let lastFail = "";
  for (const candidate of requests) {
    let req = candidate.batch ? candidate.body.requests[0] : candidate.body;
    // fillDefaults: the task asks only for a fix to the schema, so a schema-only fragment is a fair answer.
    if (cfg.fillDefaults && !candidate.batch) {
      req = { model: "decide-340m", backend: "onnx", precision: "fp32", operation: "classify", text: "sample text", ...req };
    }
    // A request_id built by code (str(uuid.uuid4()), crypto.randomUUID()) cannot be evaluated here; drop it.
    if (!candidate.batch && req.request_id === SAMPLE) {
      req = { ...req };
      delete req.request_id;
    }
    const { status, json } = candidate.batch ? await post("/v1/batch", candidate.body) : await post("/v1/infer", req);
    if (status !== 200) {
      lastFail = `runner answered ${status}: ${JSON.stringify(json).slice(0, 300)}`;
      continue;
    }
    const response = candidate.batch ? json.responses[0] : json;
    const tasks = req.schema.tasks || {};
    const names = Object.keys(tasks);
    const problems = [];
    if (cfg.minTasks && names.length < cfg.minTasks) problems.push(`expected at least ${cfg.minTasks} tasks, got ${names.length}`);
    if (cfg.model && req.model !== cfg.model) problems.push(`expected model ${cfg.model}, got ${req.model}`);
    const main = names.map((n) => [n, tasks[n]]).sort((a, b) => b[1].labels.length - a[1].labels.length)[0];
    if (main) {
      const [name, task] = main;
      const out = response.output[name];
      if (cfg.mode === "single" && isMulti(task)) problems.push("expected a single-label task (min_labels 1, max_labels 1)");
      if (cfg.mode === "multi" && !isMulti(task)) problems.push("expected a several-labels task (min_labels 0, max_labels > 1 or null)");
      if (cfg.mode === "multi" && (task.min_labels ?? 1) !== 0) problems.push("several-labels task should allow zero labels (min_labels 0)");
      if (cfg.requireDescriptions && !(task.label_descriptions && Object.keys(task.label_descriptions).length >= 2)) {
        problems.push("expected label_descriptions for the labels");
      }
      for (const label of cfg.labelsInclude || []) {
        if (!task.labels.some((l) => l.toLowerCase() === label.toLowerCase())) problems.push(`label ${label} missing`);
      }
      if (cfg.maxThreshold !== undefined) {
        const t = task.threshold ?? 0.5;
        const o = (req.options && req.options.threshold) ?? 0.5;
        if (t > cfg.maxThreshold + 1e-9) problems.push(`task threshold ${t} is above ${cfg.maxThreshold}`);
        if (o > cfg.maxThreshold + 1e-9) problems.push(`options.threshold ${o} is above ${cfg.maxThreshold}, so labels below it are dropped`);
      }
      const selected = out ? [out.value].flat().filter((v) => v !== null) : [];
      for (const label of cfg.mustSelect || []) {
        if (!selected.some((v) => String(v).toLowerCase() === label.toLowerCase())) {
          problems.push(`expected ${label} in value, got ${JSON.stringify(out && out.value)}`);
        }
      }
      if (cfg.noMatch) {
        const hasNoMatchLabel = task.labels.some((l) => NO_MATCH.test(l.trim()));
        const allowsEmpty = (task.min_labels ?? 1) === 0;
        if (!hasNoMatchLabel && !allowsEmpty) problems.push("no way to say 'nothing fits': no none/other label and min_labels is not 0");
      }
    }
    if (problems.length === 0) return result(true, `runner 200; ${names.length} task(s)`);
    lastFail = `runner 200 but: ${problems.join("; ")}`;
  }
  return result(false, lastFail);
}

/**
 * Regex checks on the answer. config: { all: [re], any: [[re, ...], ...] (each inner list needs one hit), none: [re] }.
 * Patterns are case-insensitive.
 */
function regex(output, context) {
  const cfg = (context && context.config) || {};
  const re = (s) => new RegExp(s, "i");
  const missing = (cfg.all || []).filter((p) => !re(p).test(output));
  const anyMissing = (cfg.any || []).filter((group) => !group.some((p) => re(p).test(output)));
  const forbidden = (cfg.none || []).filter((p) => re(p).test(output));
  const problems = [
    ...missing.map((p) => `missing /${p}/`),
    ...anyMissing.map((g) => `none of ${g.map((p) => `/${p}/`).join(" ")}`),
    ...forbidden.map((p) => `forbidden /${p}/`),
  ];
  return result(problems.length === 0, problems.length ? problems.join("; ") : "all patterns matched");
}

module.exports = { postBody, regex, findRequests, objectSpans };
