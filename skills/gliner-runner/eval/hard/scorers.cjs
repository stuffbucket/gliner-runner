// Hard-tier tasks: the input files each script gets, and how its stdout is scored.
// Answers stay here and are never copied into the script's directory.
"use strict";
const { readFileSync } = require("node:fs");
const { join } = require("node:path");

const DATA = join(__dirname, "data");
const load = (name) => JSON.parse(readFileSync(join(DATA, name), "utf8"));
const strip = (rows) => rows.map(({ id, text }) => ({ id, text }));
const TOL = 1e-3;

/** Sends infer requests to the runner as /v1/batch calls of at most 256, in order. */
async function inferAll(runner, requests) {
  const out = [];
  for (let i = 0; i < requests.length; i += 256) {
    const response = await fetch(`${runner}/v1/batch`, {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({ requests: requests.slice(i, i + 256) }),
    });
    const json = await response.json();
    if (response.status !== 200) throw new Error(`canonical request failed: ${response.status} ${JSON.stringify(json).slice(0, 300)}`);
    out.push(...json.responses);
  }
  return out;
}

const request = (text, tasks, model = "decide-340m") => ({
  model,
  backend: "onnx",
  precision: "fp32",
  operation: "classify",
  text,
  schema: { kind: "classification", tasks },
});

/**
 * Accepts the shapes a script may reasonably print for "one result per id": an object keyed by id, an
 * array of objects with an `id` field, or an object holding such an array. Returns Map(id -> entry).
 */
function byId(parsed, field) {
  const map = new Map();
  if (Array.isArray(parsed)) {
    for (const row of parsed) if (row && typeof row === "object" && "id" in row) map.set(String(row.id), row);
    return map;
  }
  if (parsed && typeof parsed === "object") {
    const arrays = Object.values(parsed).filter((v) => Array.isArray(v) && v.every((r) => r && typeof r === "object" && "id" in r));
    if (arrays.length === 1 && Object.keys(parsed).length <= 3) return byId(arrays[0], field);
    for (const [id, value] of Object.entries(parsed)) map.set(id, value);
  }
  return map;
}

/** The value of `field` in an entry, or the entry itself when the script printed bare values. */
const pick = (entry, field) => (entry && typeof entry === "object" && !Array.isArray(entry) && field in entry ? entry[field] : entry);
const norm = (v) => (typeof v === "string" ? v.trim().toLowerCase() : v);
const sameSet = (a, b) => Array.isArray(a) && Array.isArray(b) && JSON.stringify([...a].map(norm).sort()) === JSON.stringify([...b].map(norm).sort());
const close = (a, b) => typeof a === "number" && typeof b === "number" && Math.abs(a - b) <= TOL;

function accuracy(rows, map, field, compare = (got, want) => norm(got) === norm(want)) {
  let ok = 0;
  const misses = [];
  for (const row of rows) {
    const got = pick(map.get(row.id), field);
    if (compare(got, row.answer)) ok++;
    else misses.push(`${row.id}: want ${JSON.stringify(row.answer)} got ${JSON.stringify(got)}`);
  }
  return { ok, misses };
}

const verdict = (pass, reason) => ({ pass, reason });
const missingIds = (rows, map) => rows.filter((r) => !map.has(r.id)).map((r) => r.id);

// ---------------------------------------------------------------------------------------------- tasks

const TASKS = {
  "route-tickets": {
    files: () => ({ "tickets.json": JSON.stringify(strip(load("tickets.json")), null, 1) }),
    async score(parsed) {
      const rows = load("tickets.json");
      const map = byId(parsed, "team");
      const missing = missingIds(rows, map);
      if (missing.length) return verdict(false, `missing ids: ${missing.slice(0, 5).join(", ")}`);
      const { ok, misses } = accuracy(rows, map, "team");
      const other = accuracy(rows.filter((r) => r.answer === "other"), map, "team").ok;
      const pass = ok >= 30 && other >= 5;
      return verdict(pass, `accuracy ${ok}/40 (need 30), off-topic to other ${other}/8 (need 5); ${misses.slice(0, 6).join("; ")}`);
    },
  },

  "tag-reviews": {
    files: () => ({ "reviews.json": JSON.stringify(strip(load("reviews.json")), null, 1) }),
    async score(parsed) {
      const rows = load("reviews.json");
      const map = byId(parsed, "tags");
      const missing = missingIds(rows, map);
      if (missing.length) return verdict(false, `missing ids: ${missing.slice(0, 5).join(", ")}`);
      const { ok, misses } = accuracy(rows, map, "tags", sameSet);
      const off = rows.filter((r) => r.answer.length === 0);
      const empty = off.filter((r) => sameSet(pick(map.get(r.id), "tags"), [])).length;
      const pass = ok >= 25 && empty >= 9;
      return verdict(pass, `exact tag sets ${ok}/30 (need 25), off-topic with no tag ${empty}/10 (need 9); ${misses.slice(0, 6).join("; ")}`);
    },
  },

  "bulk-600": {
    timeoutMs: 180000,
    files: () => ({ "feedback.json": JSON.stringify(strip(load("feedback600.json"))) }),
    async score(parsed, run) {
      const rows = load("feedback600.json");
      const map = byId(parsed, "label");
      const missing = missingIds(rows, map);
      const { ok } = accuracy(rows, map, "label");
      const problems = [];
      if (missing.length) problems.push(`${missing.length} ids missing`);
      if (ok < 570) problems.push(`accuracy ${ok}/600 (need 570)`);
      if (run.metrics.rejected > 0) problems.push(`server rejected ${run.metrics.rejected} requests with 429 queue_full`);
      return verdict(problems.length === 0, problems.length ? problems.join("; ") : `accuracy ${ok}/600, no 429`);
    },
  },

  "close-labels": {
    files: () => ({ "messages.json": JSON.stringify(strip(load("messages.json")), null, 1) }),
    async score(parsed) {
      const rows = load("messages.json");
      const map = byId(parsed, "type");
      const missing = missingIds(rows, map);
      if (missing.length) return verdict(false, `missing ids: ${missing.slice(0, 5).join(", ")}`);
      const { ok, misses } = accuracy(rows, map, "type");
      return verdict(ok >= 12, `accuracy ${ok}/20 (need 12; label names alone give 8); ${misses.slice(0, 6).join("; ")}`);
    },
  },

  "flag-bands": {
    files: () => ({ "comments.json": JSON.stringify(load("comments.json"), null, 1) }),
    async score(parsed, run) {
      const rows = load("comments.json");
      const labels = ["spam", "insult", "personal information", "off-topic"];
      const responses = await inferAll(
        run.runner,
        rows.map((r) => request(r.text, { flags: { labels, min_labels: 0, max_labels: null } })),
      );
      const map = byId(parsed);
      const misses = [];
      rows.forEach((row, k) => {
        const p = responses[k].output.flags.probabilities;
        const yes = labels.filter((l) => p[l] >= 0.8);
        const unsure = labels.filter((l) => p[l] >= 0.3 && p[l] < 0.8);
        const got = map.get(row.id);
        if (!got || !sameSet(got.yes, yes) || !sameSet(got.unsure, unsure)) {
          misses.push(`${row.id}: want yes=${JSON.stringify(yes)} unsure=${JSON.stringify(unsure)} got ${JSON.stringify(got)}`);
        }
      });
      return verdict(misses.length === 0, misses.length ? `${misses.length}/16 wrong; ${misses.slice(0, 4).join("; ")}` : "16/16 match");
    },
  },

  "parse-modes": {
    files: () => ({ "reviews.json": JSON.stringify(strip(parseRows()), null, 1) }),
    async score(parsed, run) {
      const rows = parseRows();
      const tasks = {
        sentiment: { labels: ["positive", "negative", "neutral"] },
        topics: { labels: ["battery", "screen", "price", "delivery"], min_labels: 0, max_labels: null },
      };
      const responses = await inferAll(run.runner, rows.map((r) => request(r.text, tasks)));
      const map = byId(parsed);
      const misses = [];
      rows.forEach((row, k) => {
        const { sentiment: s, topics: t } = responses[k].output;
        const got = map.get(row.id);
        const why = [];
        if (!got || typeof got !== "object") why.push("missing");
        else {
          if (norm(got.sentiment) !== s.value) why.push(`sentiment ${JSON.stringify(got.sentiment)} != ${s.value}`);
          if (!close(got.sentiment_confidence, s.confidence)) why.push(`sentiment_confidence ${got.sentiment_confidence} != ${s.confidence.toFixed(4)}`);
          if (!sameSet(got.topics, t.value)) why.push(`topics ${JSON.stringify(got.topics)} != ${JSON.stringify(t.value)}`);
          const scores = got.topic_scores || {};
          for (const l of tasks.topics.labels) {
            if (!close(scores[l], t.probabilities[l])) why.push(`topic_scores.${l} ${scores[l]} != ${t.probabilities[l].toFixed(4)}`);
          }
          if (t.confidence === null ? got.topics_confidence !== null : !close(got.topics_confidence, t.confidence)) {
            why.push(`topics_confidence ${got.topics_confidence} != ${t.confidence === null ? "null" : t.confidence.toFixed(4)}`);
          }
        }
        if (why.length) misses.push(`${row.id}: ${why.slice(0, 3).join(", ")}`);
      });
      return verdict(misses.length === 0, misses.length ? `${misses.length}/${rows.length} wrong; ${misses.slice(0, 4).join("; ")}` : "all match");
    },
  },

  "fix-broken": {
    files: () => ({ "tickets.json": JSON.stringify(strip(fixRows()), null, 1) }),
    async score(parsed, run) {
      const rows = fixRows();
      const responses = await inferAll(
        run.runner,
        rows.map((r) => request(r.text, { priority: { labels: ["urgent", "normal", "low"] } })),
      );
      return compareSingle(rows, responses, "priority", byId(parsed), "label", "confidence");
    },
  },

  "retry-429": {
    // Other traffic keeps the queue full for the first 15 s of the run.
    busy: { concurrency: 450, ms: 15000 },
    files: () => ({ "feedback.json": JSON.stringify(strip(busyRows())) }),
    async score(parsed, run) {
      const rows = busyRows();
      const map = byId(parsed, "label");
      const labelled = rows.filter((r) => ["positive", "negative"].includes(norm(pick(map.get(r.id), "label")))).length;
      const { ok } = accuracy(rows, map, "label");
      const pass = labelled === rows.length && ok >= 114;
      return verdict(pass, `${labelled}/120 labelled (need all), accuracy ${ok}/120 (need 114); ${run.metrics.rejected} rejections during the run`);
    },
  },

  "fix-threshold": {
    files: () => ({ "comments.json": JSON.stringify(load("comments.json"), null, 1) }),
    async score(parsed, run) {
      const rows = load("comments.json");
      const labels = ["spam", "insult", "personal information", "off-topic"];
      const responses = await inferAll(
        run.runner,
        rows.map((r) => request(r.text, { flags: { labels, min_labels: 0, max_labels: null } })),
      );
      const map = byId(parsed);
      const misses = [];
      rows.forEach((row, k) => {
        const p = responses[k].output.flags.probabilities;
        const want = labels.filter((l) => p[l] > 0.4);
        const got = map.get(row.id);
        const keys = got && typeof got === "object" ? Object.keys(got) : null;
        const ok = keys && sameSet(keys, want) && want.every((l) => close(got[l], p[l]));
        if (!ok) misses.push(`${row.id}: want ${JSON.stringify(Object.fromEntries(want.map((l) => [l, +p[l].toFixed(4)])))} got ${JSON.stringify(got)}`);
      });
      return verdict(misses.length === 0, misses.length ? `${misses.length}/16 wrong; ${misses.slice(0, 4).join("; ")}` : "16/16 match");
    },
  },

  // ------------------------------------------------------------------------------------- held out

  "zero-or-one": {
    files: () => ({ "chat.json": JSON.stringify(strip(load("chat.json")), null, 1) }),
    async score(parsed) {
      const rows = load("chat.json");
      const map = byId(parsed, "team");
      const missing = missingIds(rows, map);
      if (missing.length) return verdict(false, `missing ids: ${missing.slice(0, 5).join(", ")}`);
      const { ok, misses } = accuracy(rows, map, "team", (got, want) => (want === null ? got === null : norm(got) === want));
      const none = rows.filter((r) => r.answer === null && pick(map.get(r.id), "team") === null).length;
      return verdict(ok >= 16 && none >= 4, `accuracy ${ok}/20 (need 16), null for small talk ${none}/5 (need 4); ${misses.slice(0, 5).join("; ")}`);
    },
  },

  "shell-tsv": {
    files: () => ({ "lines.txt": readFileSync(join(DATA, "lines.txt"), "utf8") }),
    async score(_parsed, run) {
      const lines = readFileSync(join(DATA, "lines.txt"), "utf8").split("\n").filter((l) => l.length > 0);
      const responses = await inferAll(
        run.runner,
        lines.map((l) => request(l, { topic: { labels: ["food", "travel", "money", "health"] } })),
      );
      const out = run.stdout.split("\n").map((l) => l.replace(/\r$/, "")).filter((l) => l.trim().length > 0);
      const misses = [];
      if (run.language !== "shell") misses.push(`expected a shell script, got ${run.language}`);
      lines.forEach((_line, k) => {
        const want = responses[k].output.topic;
        const row = out.find((l) => l.split("\t")[0].trim() === String(k + 1));
        const cells = row ? row.split("\t") : [];
        if (cells.length !== 3 || cells[1].trim() !== want.value || Math.abs(Number(cells[2]) - want.confidence) > 0.002) {
          misses.push(`line ${k + 1}: want ${k + 1}\\t${want.value}\\t${want.confidence.toFixed(3)} got ${JSON.stringify(row)}`);
        }
      });
      return verdict(misses.length === 0, misses.length ? `${misses.length} wrong; ${misses.slice(0, 4).join("; ")}` : "15/15 lines match");
    },
  },

  "route-bands": {
    files: () => ({ "tickets.json": JSON.stringify(strip(bandRows()), null, 1) }),
    async score(parsed, run) {
      const rows = bandRows();
      const labels = ["billing", "technical", "shipping", "account", "other"];
      const responses = await inferAll(run.runner, rows.map((r) => request(r.text, { team: { labels } })));
      const map = byId(parsed);
      const misses = [];
      rows.forEach((row, k) => {
        const { value, confidence } = responses[k].output.team;
        const action = value === "other" || confidence < 0.5 ? "fallback" : confidence < 0.8 ? "review" : "route";
        const got = map.get(row.id);
        const team = action === "fallback" ? null : value;
        if (!got || got.action !== action || (team !== null && norm(got.team) !== team) || (team === null && got.team != null)) {
          misses.push(`${row.id}: want ${action}/${team} (${confidence.toFixed(3)} ${value}) got ${JSON.stringify(got)}`);
        }
      });
      return verdict(misses.length === 0, misses.length ? `${misses.length}/${rows.length} wrong; ${misses.slice(0, 4).join("; ")}` : "all match");
    },
  },

  "top-two": {
    files: () => ({ "messages.json": JSON.stringify(strip(load("asks.json")), null, 1) }),
    async score(parsed, run) {
      const rows = load("asks.json");
      const labels = ["question", "request", "complaint"];
      const responses = await inferAll(run.runner, rows.map((r) => request(r.text, { intent: { labels } })));
      const map = byId(parsed);
      const misses = [];
      rows.forEach((row, k) => {
        const p = responses[k].output.intent.probabilities;
        const [first, second] = [...labels].sort((a, b) => p[b] - p[a]);
        const ambiguous = p[first] - p[second] < 0.15;
        const got = map.get(row.id);
        const ok =
          got && norm(got.first) === first && norm(got.second) === second && close(got.p1, p[first]) && close(got.p2, p[second]) && got.ambiguous === ambiguous;
        if (!ok) misses.push(`${row.id}: want ${first} ${p[first].toFixed(3)} / ${second} ${p[second].toFixed(3)} ambiguous=${ambiguous} got ${JSON.stringify(got)}`);
      });
      return verdict(misses.length === 0, misses.length ? `${misses.length}/${rows.length} wrong; ${misses.slice(0, 4).join("; ")}` : "all match");
    },
  },
};

function parseRows() {
  const ids = new Set(["R01", "R02", "R03", "R04", "R05", "R06", "R13", "R17", "R21", "R22", "R29"]);
  return load("reviews.json").filter((r) => ids.has(r.id));
}
const fixRows = () => load("tickets.json").slice(0, 12);
const bandRows = () => load("tickets.json").slice(12);
const busyRows = () => load("feedback600.json").filter((_, k) => k % 5 === 0);

function compareSingle(rows, responses, task, map, labelField, confField) {
  const misses = [];
  rows.forEach((row, k) => {
    const want = responses[k].output[task];
    const got = map.get(row.id);
    if (!got || norm(got[labelField]) !== want.value || !close(got[confField], want.confidence)) {
      misses.push(`${row.id}: want ${want.value} ${want.confidence.toFixed(4)} got ${JSON.stringify(got)}`);
    }
  });
  return verdict(misses.length === 0, misses.length ? `${misses.length}/${rows.length} wrong; ${misses.slice(0, 4).join("; ")}` : "all match");
}

module.exports = { TASKS };
