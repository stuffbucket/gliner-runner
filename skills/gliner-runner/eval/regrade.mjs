// Re-grades a promptfoo result file with the current asserts.cjs, without calling any model: javascript
// assertions are recomputed, other assertions (llm-rubric, regex) keep their recorded result.
// Usage: node regrade.mjs in.json out.json [tasks.json]
//   tasks.json (optional): the current tasks as JSON (npx -y js-yaml@4.1.0 tasks.yaml > tasks.json), so that
//   javascript assertion configs come from the current tasks rather than from the recorded run.
// Needs the runner at RUNNER_URL (default http://127.0.0.1:8097).
import { readFileSync, writeFileSync } from "node:fs";
import { createRequire } from "node:module";

const require = createRequire(import.meta.url);
const asserts = require("./asserts.cjs");
const [input, output, tasksFile] = process.argv.slice(2);
const data = JSON.parse(readFileSync(input, "utf8"));
const current = new Map(
  tasksFile ? JSON.parse(readFileSync(tasksFile, "utf8")).map((t) => [t.description, t.assert]) : [],
);
let changed = 0;
for (const row of data.results.results) {
  const text = row.response?.output ?? "";
  const parts = row.gradingResult?.componentResults ?? [];
  const recorded = row.testCase.assert ?? [];
  const now = current.get(row.testCase.description) ?? recorded;
  const results = [];
  for (const [i, assertion] of recorded.entries()) {
    const config = (now[i] && now[i].type === assertion.type ? now[i].config : assertion.config) ?? {};
    const fn = assertion.type === "javascript" && /asserts\.cjs:(\w+)$/.exec(assertion.value)?.[1];
    results.push(fn ? await asserts[fn](text, { config, vars: row.vars }) : parts[i]);
  }
  const pass = results.every((r) => r && r.pass);
  if (pass !== row.success) changed += 1;
  row.success = pass;
  row.gradingResult = {
    ...row.gradingResult,
    pass,
    reason: results.filter((r) => r && !r.pass).map((r) => r.reason).join(" | ") || "All assertions passed",
    componentResults: results,
  };
}
writeFileSync(output, JSON.stringify(data));
console.error(`${changed} results changed`);
