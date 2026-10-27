// Checks the hard-tier harness without a model: reference answers that must pass and wrong answers that must
// fail. Needs the runner on RUNNER_URL (default http://127.0.0.1:8099). Usage: node selftest.mjs [task ...]
import { createRequire } from "node:module";
const require = createRequire(import.meta.url);
const { check } = require("./run.cjs");

const fence = (lang, code) => "Here is the script.\n\n```" + lang + "\n" + code + "\n```\n";
const common = `
import { readFileSync } from "node:fs";
const URL = "http://127.0.0.1:8099";
const req = (text, tasks, model = "decide-340m") => ({ model, backend: "onnx", precision: "fp32", operation: "classify", text, schema: { kind: "classification", tasks } });
async function batch(requests) {
  const out = [];
  for (let i = 0; i < requests.length; i += 256) {
    const r = await fetch(URL + "/v1/batch", { method: "POST", headers: { "content-type": "application/json" }, body: JSON.stringify({ requests: requests.slice(i, i + 256) }) });
    if (!r.ok) throw new Error(r.status + " " + await r.text());
    out.push(...(await r.json()).responses);
  }
  return out;
}
`;
const REF = {
  "route-tickets": common + `
const rows = JSON.parse(readFileSync("tickets.json", "utf8"));
const res = await batch(rows.map((r) => req(r.text, { team: { labels: ["billing", "technical", "shipping", "account", "other"], label_descriptions: { other: "Small talk, thanks, greetings, or anything unrelated to our product" } } })));
console.log(JSON.stringify(rows.map((r, k) => ({ id: r.id, team: res[k].output.team.value }))));`,
  "tag-reviews": common + `
const rows = JSON.parse(readFileSync("reviews.json", "utf8"));
const res = await batch(rows.map((r) => req(r.text, { topics: { labels: ["battery", "screen", "price", "delivery"], min_labels: 0, max_labels: null } })));
console.log(JSON.stringify(Object.fromEntries(rows.map((r, k) => [r.id, res[k].output.topics.value]))));`,
  "bulk-600": common + `
const rows = JSON.parse(readFileSync("feedback.json", "utf8"));
const res = await batch(rows.map((r) => req(r.text, { sentiment: { labels: ["positive", "negative"] } })));
console.log(JSON.stringify(rows.map((r, k) => ({ id: r.id, label: res[k].output.sentiment.value }))));`,
  "close-labels": common + `
const rows = JSON.parse(readFileSync("messages.json", "utf8"));
const res = await batch(rows.map((r) => req(r.text, { category: { labels: ["complaint", "feedback"], label_descriptions: { complaint: "The customer asks us to fix, replace, refund or correct something, whatever their tone", feedback: "The customer gives an opinion, positive or negative, and asks for nothing to be done" } } })));
console.log(JSON.stringify(Object.fromEntries(rows.map((r, k) => [r.id, res[k].output.category.value]))));`,
  "flag-bands": common + `
const YES = 0.8, UNSURE = 0.3;
const labels = ["spam", "insult", "personal information", "off-topic"];
const rows = JSON.parse(readFileSync("comments.json", "utf8"));
const res = await batch(rows.map((r) => req(r.text, { flags: { labels, min_labels: 0, max_labels: null } })));
console.log(JSON.stringify(Object.fromEntries(rows.map((r, k) => { const p = res[k].output.flags.probabilities; return [r.id, { yes: labels.filter((l) => p[l] >= YES), unsure: labels.filter((l) => p[l] >= UNSURE && p[l] < YES) }]; }))));`,
  "parse-modes": common + `
const rows = JSON.parse(readFileSync("reviews.json", "utf8"));
const res = await batch(rows.map((r) => req(r.text, { sentiment: { labels: ["positive", "negative", "neutral"] }, topics: { labels: ["battery", "screen", "price", "delivery"], min_labels: 0, max_labels: null } })));
console.log(JSON.stringify(Object.fromEntries(rows.map((r, k) => { const { sentiment: s, topics: t } = res[k].output; return [r.id, { sentiment: s.value, sentiment_confidence: s.confidence, topics: t.value, topic_scores: t.probabilities, topics_confidence: t.confidence }]; }))));`,
  "retry-429": common.replace("if (!r.ok) throw", "if (r.status === 429) { await new Promise((ok) => setTimeout(ok, 1000 * Number(r.headers.get('retry-after') || 1))); i -= 256; continue; }\n    if (!r.ok) throw") + `
const rows = JSON.parse(readFileSync("feedback.json", "utf8"));
const res = await batch(rows.map((r) => req(r.text, { sentiment: { labels: ["positive", "negative"] } })));
console.log(JSON.stringify(rows.map((r, k) => ({ id: r.id, label: res[k].output.sentiment.value }))));`,
  "fix-threshold": common + `
const rows = JSON.parse(readFileSync("comments.json", "utf8"));
const res = await batch(rows.map((r) => ({ ...req(r.text, { flags: { labels: ["spam", "insult", "personal information", "off-topic"], min_labels: 0, max_labels: null, threshold: 0.4 } }), options: { threshold: 0.4 } })));
console.log(JSON.stringify(Object.fromEntries(rows.map((r, k) => { const f = res[k].output.flags; return [r.id, Object.fromEntries(f.value.map((l) => [l, f.probabilities[l]]))]; }))));`,
  "zero-or-one": common + `
const rows = JSON.parse(readFileSync("chat.json", "utf8"));
const res = await batch(rows.map((r) => req(r.text, { department: { labels: ["sales", "support", "legal"], min_labels: 0, max_labels: 1 } })));
console.log(JSON.stringify(Object.fromEntries(rows.map((r, k) => [r.id, res[k].output.department.value[0] ?? null]))));`,
  "fix-broken": common + `
const rows = JSON.parse(readFileSync("tickets.json", "utf8"));
const res = await batch(rows.map((r) => req(r.text, { priority: { labels: ["urgent", "normal", "low"] } })));
console.log(JSON.stringify(Object.fromEntries(rows.map((r, k) => [r.id, { label: res[k].output.priority.value, confidence: res[k].output.priority.confidence }]))));`,
  "route-bands": common + `
const ACT = 0.8, REVIEW = 0.5;
const rows = JSON.parse(readFileSync("tickets.json", "utf8"));
const res = await batch(rows.map((r) => req(r.text, { team: { labels: ["billing", "technical", "shipping", "account", "other"] } })));
console.log(JSON.stringify(Object.fromEntries(rows.map((r, k) => { const { value, confidence } = res[k].output.team; const action = value === "other" || confidence < REVIEW ? "fallback" : confidence < ACT ? "review" : "route"; return [r.id, { action, team: action === "fallback" ? null : value }]; }))));`,
  "top-two": common + `
const labels = ["question", "request", "complaint"];
const rows = JSON.parse(readFileSync("messages.json", "utf8"));
const res = await batch(rows.map((r) => req(r.text, { intent: { labels } })));
console.log(JSON.stringify(Object.fromEntries(rows.map((r, k) => { const p = res[k].output.intent.probabilities; const [a, b] = [...labels].sort((x, y) => p[y] - p[x]); return [r.id, { first: a, p1: p[a], second: b, p2: p[b], ambiguous: p[a] - p[b] < 0.15 }]; }))));`,
};
const SHELL = `#!/bin/bash
set -euo pipefail
n=0
while IFS= read -r line || [ -n "$line" ]; do
  n=$((n+1))
  body=$(jq -n --arg t "$line" '{model:"decide-340m",backend:"onnx",precision:"fp32",operation:"classify",text:$t,schema:{kind:"classification",tasks:{topic:{labels:["food","travel","money","health"]}}}}')
  curl -s -X POST http://127.0.0.1:8099/v1/infer -H 'content-type: application/json' -d "$body" | jq -r --arg n "$n" '[$n, .output.topic.value, (.output.topic.confidence*1000|round/1000|tostring)] | @tsv'
done < lines.txt`;

const which = process.argv.slice(2);
const all = { ...Object.fromEntries(Object.entries(REF).map(([k, v]) => [k, fence("js", v)])), "shell-tsv": fence("bash", SHELL) };
let bad = 0;
for (const [task, answer] of Object.entries(all)) {
  if (which.length && !which.includes(task)) continue;
  const t0 = Date.now();
  const r = await check(answer, { config: { task } });
  if (!r.pass) bad++;
  console.log(r.pass ? "PASS" : "FAIL", task, `${Date.now() - t0}ms`, r.reason.slice(0, 400));
}

// Wrong answers: each must fail.
const one = `const req = (text, tasks) => ({ model: "decide-340m", backend: "onnx", precision: "fp32", operation: "classify", text, schema: { kind: "classification", tasks } });
const post = (path, body) => fetch("http://127.0.0.1:8099" + path, { method: "POST", headers: { "content-type": "application/json" }, body: JSON.stringify(body) });`;
const WRONG = {
  "bulk-600: 600 concurrent /v1/infer calls (429s)": ["bulk-600", `import { readFileSync } from "node:fs"; ${one}
const rows = JSON.parse(readFileSync("feedback.json", "utf8"));
const res = await Promise.all(rows.map(async (r) => { for (;;) { const x = await post("/v1/infer", req(r.text, { s: { labels: ["positive", "negative"] } })); if (x.status === 429) { await new Promise((ok) => setTimeout(ok, 1000)); continue; } return (await x.json()).output.s.value; } }));
console.log(JSON.stringify(rows.map((r, k) => ({ id: r.id, label: res[k] }))));`],
  "bulk-600: one batch of 600 (422)": ["bulk-600", `import { readFileSync } from "node:fs"; ${one}
const rows = JSON.parse(readFileSync("feedback.json", "utf8"));
const j = await (await post("/v1/batch", { requests: rows.map((r) => req(r.text, { s: { labels: ["positive", "negative"] } })) })).json();
console.log(JSON.stringify(rows.map((r, k) => ({ id: r.id, label: j.responses?.[k]?.output.s.value }))));`],
  "retry-429: one batch, no retry": ["retry-429", `import { readFileSync } from "node:fs"; ${one}
const rows = JSON.parse(readFileSync("feedback.json", "utf8"));
const j = await (await post("/v1/batch", { requests: rows.map((r) => req(r.text, { s: { labels: ["positive", "negative"] } })) })).json();
console.log(JSON.stringify(rows.map((r, k) => ({ id: r.id, label: j.responses?.[k]?.output.s.value }))));`],
  "tag-reviews: single-label mode": ["tag-reviews", `import { readFileSync } from "node:fs"; ${one}
const rows = JSON.parse(readFileSync("reviews.json", "utf8")); const out = {};
for (const r of rows) { const x = await (await post("/v1/infer", req(r.text, { topics: { labels: ["battery", "screen", "price", "delivery"] } }))).json(); out[r.id] = [x.output.topics.value]; }
console.log(JSON.stringify(out));`],
  "tag-reviews: network outside the runner": ["tag-reviews", `const r = await fetch("https://example.com"); console.log(await r.text());`],
};
for (const [name, [task, code]] of Object.entries(WRONG)) {
  if (which.length && !which.includes(task)) continue;
  const r = await check(fence("js", code), { config: { task } });
  if (r.pass) bad++;
  console.log(r.pass ? "WRONG ANSWER PASSED" : "fails as it should", name, "-", r.reason.slice(0, 200));
}
process.exitCode = bad ? 1 : 0;
