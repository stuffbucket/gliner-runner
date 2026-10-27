// Re-runs the hard-tier check on saved answers, without calling any model, and rewrites the result file.
// Usage: node regrade.mjs in.json out.json [--reason <regex>] [--task <id>]
//   --reason: only rows whose recorded failure reason matches (for example a sandbox fix: "Operation not permitted")
//   --task:   only rows of this task
// Other rows keep their recorded result. Needs the runner at RUNNER_URL (default http://127.0.0.1:8099).
import { readFileSync, writeFileSync } from "node:fs";
import { createRequire } from "node:module";

const require = createRequire(import.meta.url);
const { check } = require("./run.cjs");
const args = process.argv.slice(2);
const flag = (name) => (args.includes(name) ? args[args.indexOf(name) + 1] : undefined);
const [input, output] = args.filter((a, i) => !a.startsWith("--") && !args[i - 1]?.startsWith("--"));
const reason = flag("--reason") && new RegExp(flag("--reason"), "i");
const only = flag("--task");

const data = JSON.parse(readFileSync(input, "utf8"));
let rerun = 0;
let changed = 0;
for (const row of data.results.results) {
  const task = row.testCase.description;
  if (only && task !== only) continue;
  if (reason && !reason.test(row.gradingResult?.reason ?? "")) continue;
  const config = row.testCase.assert?.[0]?.config;
  const result = await check(row.response?.output ?? "", { config });
  rerun++;
  if (result.pass !== row.success) changed++;
  console.error(`${row.prompt?.label} ${row.provider?.label} ${task}: ${row.success ? "pass" : "fail"} -> ${result.pass ? "pass" : "fail"}`);
  row.success = result.pass;
  row.gradingResult = { ...row.gradingResult, pass: result.pass, score: result.score, reason: result.reason, componentResults: [result] };
}
writeFileSync(output, JSON.stringify(data));
console.error(`${rerun} re-run, ${changed} changed`);
