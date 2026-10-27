// Summarizes promptfoo JSON results (promptfoo eval -o results.json) as pass rates per
// variant x model x split, and per category. Several files (repeated runs) are summed.
// Usage: node report.mjs results.json [more.json ...] [--failures]
import { readFileSync } from "node:fs";

const args = process.argv.slice(2);
const flag = args.find((a) => a.startsWith("--"));
const rows = args
  .filter((a) => !a.startsWith("--"))
  .flatMap((file) => {
    const data = JSON.parse(readFileSync(file, "utf8"));
    return data.results.results ?? data.results;
  });

const cell = new Map();
const add = (key, pass) => {
  const c = cell.get(key) ?? { pass: 0, n: 0 };
  c.n += 1;
  c.pass += pass ? 1 : 0;
  cell.set(key, c);
};
for (const r of rows) {
  const variant = r.prompt?.label ?? "?";
  const model = r.provider?.label ?? r.provider?.id ?? "?";
  const meta = r.testCase?.metadata ?? {};
  const pass = r.success === true;
  add(`${variant}|${model}|${meta.split}`, pass);
  add(`${variant}|${model}|all`, pass);
  add(`cat|${variant}|${meta.category}`, pass);
  if (flag === "--failures" && !pass && meta.split !== "heldout") {
    console.log(`FAIL ${variant} ${model} ${r.testCase?.description}: ${(r.gradingResult?.reason ?? r.error ?? "").slice(0, 400)}`);
  }
}
const pct = (c) => (c ? `${c.pass}/${c.n} (${Math.round((100 * c.pass) / c.n)}%)` : "-");
const variants = [...new Set(rows.map((r) => r.prompt?.label))];
const models = [...new Set(rows.map((r) => r.provider?.label ?? r.provider?.id))];
console.log("| Variant | Model | Tuning | Held-out | All |\n| --- | --- | --- | --- | --- |");
for (const v of variants) for (const m of models) {
  console.log(`| ${v} | ${m} | ${pct(cell.get(`${v}|${m}|tuning`))} | ${pct(cell.get(`${v}|${m}|heldout`))} | ${pct(cell.get(`${v}|${m}|all`))} |`);
}
const cats = [...new Set(rows.map((r) => r.testCase?.metadata?.category))].sort();
console.log(`\n| Category | ${variants.join(" | ")} |\n| --- | ${variants.map(() => "---").join(" | ")} |`);
for (const c of cats) console.log(`| ${c} | ${variants.map((v) => pct(cell.get(`cat|${v}|${c}`))).join(" | ")} |`);
