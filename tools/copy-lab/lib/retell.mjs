// The retell stage: runs the promptfoo suite in retell/ over every candidate that needs it and turns the results
// into a score per candidate. Rows are cached by candidate text, facts, persona, run and prompt version, so a
// re-run sends promptfoo only what it has not read before.
import { spawn } from "node:child_process";
import { existsSync, mkdirSync, readFileSync, writeFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

import { hash, parseJson } from "./model.mjs";
import { PROMPT_VERSION, rubric } from "./prompts.mjs";

const here = dirname(fileURLToPath(import.meta.url));
const config = join(here, "../retell/promptfooconfig.mjs");
export const PROMPTFOO = JSON.parse(readFileSync(join(here, "../package.json"), "utf8")).config.promptfoo;

/** Reads `picks=yes nowrite=no …` from the start of the grader's reason. */
export function factFlags(reason, facts) {
  const flags = {};
  for (const [, id, yes] of reason.matchAll(/\b([\w-]+)\s*=\s*(yes|no)\b/gi)) {
    if (facts.some((f) => f.id === id) && !(id in flags)) flags[id] = yes.toLowerCase() === "yes";
  }
  return flags;
}

/** One row per promptfoo result: who read it, what they said, which facts came through, and keep reading. */
export function readRows(output, facts) {
  const required = facts.filter((f) => !f.optional);
  return output.results.results.map((r) => {
    const parts = r.gradingResult?.componentResults ?? [];
    const rubricPart = parts.find((c) => c.assertion?.type === "llm-rubric");
    const reason = rubricPart?.reason ?? "";
    const flags = factFlags(reason, facts);
    const parsed = Object.keys(flags).length >= required.length;
    let answer = {};
    try {
      answer = parseJson(r.response?.output ?? "");
    } catch {
      answer = { raw: r.response?.output ?? "" };
    }
    return {
      id: r.vars.id,
      run: Number(r.vars.run),
      persona: r.provider?.label ?? r.provider?.id,
      answer,
      flags,
      factScore: parsed ? required.filter((f) => flags[f.id]).length / required.length : (rubricPart?.score ?? 0),
      keep: /^\s*yes/i.test(String(answer.keep_reading ?? "")),
      reason,
      // failureReason 2 is an error (the call failed); 1 is only a failed assertion, which is a normal result.
      error: r.failureReason === 2 ? r.error || "error" : rubricPart ? null : "no grade",
    };
  });
}

const rowKey = (job, text, persona, run) => hash({ v: PROMPT_VERSION, text, rubric: job.rubric, grader: job.grader, persona, run });

export async function runRetell({ slot, candidates, dir, api, cacheDir, concurrency, log }) {
  const facts = slot.facts;
  const job = { slot: slot.id, api, personas: slot.personas, grader: slot.grader, rubric: rubric(facts), tests: [] };
  const rowsDir = join(cacheDir, "retell");
  mkdirSync(rowsDir, { recursive: true });
  const cached = [];
  for (const c of candidates) {
    for (let run = 1; run <= slot.runs; run += 1) {
      const files = slot.personas.map((p) => join(rowsDir, `${rowKey(job, c.text, p, run)}.json`));
      if (files.every(existsSync)) cached.push(...files.map((f) => ({ ...JSON.parse(readFileSync(f, "utf8")), id: c.id })));
      else job.tests.push({ id: c.id, run, text: c.text });
    }
  }
  const stats = { cachedRows: cached.length, rows: 0, calls: 0, ms: 0 };
  let fresh = [];
  if (job.tests.length > 0) {
    const jobFile = join(dir, "retell-job.json");
    const output = join(dir, "retell-promptfoo.json");
    const home = join(dir, ".promptfoo");
    mkdirSync(home, { recursive: true });
    writeFileSync(jobFile, JSON.stringify(job, null, 2));
    const args = ["-y", `promptfoo@${PROMPTFOO}`, "eval", "-c", config, "--no-cache", "--no-write", "--no-progress-bar", "-j", String(concurrency), "-o", output];
    log(`retell: ${job.tests.length} tests x ${slot.personas.length} personas through promptfoo (${cached.length} rows cached)`);
    const started = Date.now();
    const code = await new Promise((done) => {
      const child = spawn("npx", args, {
        cwd: dir,
        env: { ...process.env, COPY_LAB_RETELL: jobFile, PROMPTFOO_CONFIG_DIR: home, PROMPTFOO_DISABLE_TELEMETRY: "1", PROMPTFOO_DISABLE_UPDATE: "1" },
        stdio: ["ignore", "ignore", "pipe"],
      });
      let errors = "";
      child.stderr.on("data", (chunk) => (errors += chunk));
      child.on("close", (status) => {
        if (status !== 0 && status !== 100) log(errors.slice(-1500));
        done(status);
      });
    });
    stats.ms = Date.now() - started;
    // promptfoo exits with 100 when some tests fail, which is a normal result here.
    if (!existsSync(output)) throw new Error(`promptfoo exited with ${code} and wrote no results`);
    const data = JSON.parse(readFileSync(output, "utf8"));
    const usage = data.results.stats?.tokenUsage ?? {};
    stats.calls = (usage.numRequests ?? 0) + (usage.assertions?.numRequests ?? 0);
    fresh = readRows(data, facts);
    const textOf = new Map(candidates.map((c) => [c.id, c.text]));
    for (const row of fresh) {
      if (!row.error) writeFileSync(join(rowsDir, `${rowKey(job, textOf.get(row.id), row.persona, row.run)}.json`), JSON.stringify(row));
    }
  }
  const rows = [...cached, ...fresh];
  stats.rows = rows.length;
  return { rows, stats };
}

/** Folds the rows into each candidate: fact rate, keep-reading rate, optional facts, and the retell score (0 to 2). */
export function scoreRetell(candidates, rows, facts) {
  const optional = facts.filter((f) => f.optional);
  for (const c of candidates) {
    const mine = rows.filter((r) => r.id === c.id && !r.error);
    if (mine.length === 0) continue;
    const mean = (xs) => xs.reduce((a, b) => a + b, 0) / xs.length;
    const factRate = mean(mine.map((r) => r.factScore));
    const keepRate = mean(mine.map((r) => (r.keep ? 1 : 0)));
    c.retell = {
      score: Number((factRate + keepRate).toFixed(3)),
      factRate: Number(factRate.toFixed(3)),
      keepRate: Number(keepRate.toFixed(3)),
      perFact: Object.fromEntries(facts.map((f) => [f.id, Number(mean(mine.map((r) => (r.flags[f.id] ? 1 : 0))).toFixed(3))])),
      optional: optional.length > 0 ? Number(mean(mine.flatMap((r) => optional.map((f) => (r.flags[f.id] ? 1 : 0)))).toFixed(3)) : null,
      reads: mine.length,
      sample: mine.slice(0, 2).map((r) => ({ persona: r.persona, what: r.answer.what, like_chatgpt: r.answer.like_chatgpt, keep: r.answer.keep_reading })),
    };
  }
}
