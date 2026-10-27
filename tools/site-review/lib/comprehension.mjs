// Runs the promptfoo comprehension suite (comprehension/promptfooconfig.mjs) against the page text the checks stage
// saved, and reads back one result per question.
import { spawn } from "node:child_process";
import { existsSync, mkdirSync, readFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

const here = dirname(fileURLToPath(import.meta.url));
const config = join(here, "../comprehension/promptfooconfig.mjs");
export const PROMPTFOO = JSON.parse(readFileSync(join(here, "../package.json"), "utf8")).config.promptfoo;

/** The pages the questions ask about, as slugs (`home`, `start`, …). */
export const questionPages = () =>
  [...new Set(JSON.parse(readFileSync(join(here, "../comprehension/questions.json"), "utf8")).questions.map((q) => q.page))];

/** One row per question from promptfoo's JSON output. */
export function readResults(output) {
  return output.results.results.map((r) => {
    const parts = r.gradingResult?.componentResults ?? [];
    const failed = parts.filter((c) => !c.pass);
    return {
      id: r.testCase.description,
      page: r.vars.page,
      persona: r.vars.persona,
      question: r.vars.question,
      answer: r.response?.output ?? "",
      pass: r.success,
      reason: (failed.length > 0 ? failed : parts).map((c) => c.reason).join(" ") || r.error || "",
      latencyMs: r.latencyMs,
    };
  });
}

export async function runComprehension({ outDir, pagesDir, api, model, concurrency, log }) {
  const output = join(outDir, "comprehension.json");
  const home = join(outDir, ".promptfoo");
  mkdirSync(home, { recursive: true });
  const args = ["-y", `promptfoo@${PROMPTFOO}`, "eval", "-c", config, "--no-cache", "--no-write", "--no-progress-bar", "-j", String(concurrency), "-o", output];
  log(`npx ${args.join(" ")}`);
  const code = await new Promise((done) => {
    const child = spawn("npx", args, {
      cwd: outDir,
      // promptfoo keeps its database and logs under PROMPTFOO_CONFIG_DIR; keep them with this run.
      env: { ...process.env, SITE_REVIEW_PAGES: pagesDir, SITE_REVIEW_API: api, SITE_REVIEW_MODEL: model,
        PROMPTFOO_CONFIG_DIR: home, PROMPTFOO_DISABLE_TELEMETRY: "1", PROMPTFOO_DISABLE_UPDATE: "1" },
      stdio: ["ignore", "ignore", "pipe"],
    });
    let errors = "";
    child.stderr.on("data", (chunk) => (errors += chunk));
    child.on("close", (status) => {
      if (status !== 0 && status !== 100) log(errors.slice(-1500));
      done(status);
    });
  });
  // promptfoo exits with 100 when some tests fail, which is a normal result here.
  if (!existsSync(output)) return { tests: [], error: `promptfoo exited with ${code} and wrote no results` };
  const data = JSON.parse(readFileSync(output, "utf8"));
  const stats = data.results.stats;
  return { tests: readResults(data), stats: { durationMs: stats.durationMs, calls: (stats.tokenUsage?.numRequests ?? 0) + (stats.tokenUsage?.assertions?.numRequests ?? 0), tokens: stats.tokenUsage } };
}
