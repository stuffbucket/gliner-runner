#!/usr/bin/env node
// Reviews the documentation site the way a few real readers would, and writes a report of what to improve.
//
//   node tools/site-review/review.mjs --site http://127.0.0.1:4399/gliner-runner/ [--out DIR] [--model claude-sonnet-5]
//
// Stages: walk (each persona reads and clicks through the site), hands-on (each persona tries the playground
// against a local runner), checks (axe, speed, phone layout, links), comprehension (promptfoo questions),
// classify (gliner-runner sorts the findings into categories) and suggest (one model call per category).
// See README.md for what each stage costs.
import { mkdirSync, readFileSync, writeFileSync } from "node:fs";
import { dirname, join, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import { parseArgs } from "node:util";

import { classify, startRunner } from "../../site/scripts/runner.mjs";
import { checkLinks, checkPages } from "./lib/checks.mjs";
import { PROMPTFOO, questionPages, runComprehension } from "./lib/comprehension.mjs";
import { DEFAULT_DEPS_DIR, cachedChromium, loadBrowserTools } from "./lib/deps.mjs";
import { CATEGORIES, agreement, categorize, classifierText, collectFindings, groupFindings } from "./lib/findings.mjs";
import { handsOn } from "./lib/handson.mjs";
import { createModel, pool } from "./lib/model.mjs";
import { shortUrl } from "./lib/page.mjs";
import { renderReport } from "./lib/report.mjs";
import { suggest } from "./lib/suggest.mjs";
import { walkPersona } from "./lib/walk.mjs";

const here = dirname(fileURLToPath(import.meta.url));
const repo = resolve(here, "../..");
const STAGES = ["walk", "hands-on", "checks", "comprehension", "classify", "suggest"];
// The pages every run checks and asks questions about, whatever the personas click.
const KEY_PAGES = ["", "start/", "how-it-works/", "playground/", "use-in-react/", "examples/", "examples/support-tickets/", "models/", "evaluation/", "more-ideas/"];

const { values: flags } = parseArgs({
  options: {
    site: { type: "string", default: "https://stuffbucket.github.io/gliner-runner/" },
    out: { type: "string" },
    api: { type: "string", default: "http://localhost:4141/v1" },
    model: { type: "string", default: "claude-sonnet-5" },
    personas: { type: "string", default: join(here, "personas.json") },
    only: { type: "string" },
    steps: { type: "string", default: "8" },
    concurrency: { type: "string", default: "4" },
    stages: { type: "string", default: STAGES.join(",") },
    "runner-cli": { type: "string", default: join(repo, "packages/onnx-runner/dist/src/cli.js") },
    "models-dir": { type: "string" },
    "classifier-model": { type: "string", default: "decide-340m" },
    chrome: { type: "string" },
    "deps-dir": { type: "string", default: DEFAULT_DEPS_DIR },
    "external-links": { type: "boolean", default: false },
    headed: { type: "boolean", default: false },
    help: { type: "boolean", default: false },
  },
});
if (flags.help) {
  console.log(readFileSync(fileURLToPath(import.meta.url), "utf8").split("\n").slice(1, 9).map((l) => l.replace(/^\/\/ ?/, "")).join("\n"));
  process.exit(0);
}

const siteRoot = flags.site.replace(/\/?$/, "/");
const stamp = new Date().toISOString().replace(/[:.]/g, "-").slice(0, 19);
const outDir = resolve(flags.out ?? `/tmp/gr-work/site-review/${stamp}`);
const shotsDir = join(outDir, "screenshots");
const pagesDir = join(outDir, "pages");
for (const dir of [shotsDir, pagesDir]) mkdirSync(dir, { recursive: true });
const want = new Set(flags.stages.split(",").map((s) => s.trim()));
const steps = Number(flags.steps);
const concurrency = Number(flags.concurrency);
const allPersonas = JSON.parse(readFileSync(flags.personas, "utf8")).personas;
const personas = flags.only ? allPersonas.filter((p) => flags.only.split(",").includes(p.id)) : allPersonas;

const began = Date.now();
const log = (message) => console.log(`[${((Date.now() - began) / 1000).toFixed(0).padStart(4)}s] ${message}`);
const model = createModel({ base: flags.api, model: flags.model });
const timings = {};
const timed = async (name, work) => {
  const started = Date.now();
  try {
    return await work();
  } finally {
    timings[name] = Date.now() - started;
  }
};

let runner = null;
let browser = null;
const stop = async () => {
  runner?.stop();
  runner = null;
  await browser?.close().catch(() => {});
  browser = null;
};
process.on("SIGINT", () => stop().then(() => process.exit(130)));

const data = { meta: { site: siteRoot, model: flags.model, steps, started: new Date(began).toISOString() }, personas };
try {
  log(`reviewing ${siteRoot} into ${outDir}`);
  if (want.has("hands-on") || want.has("classify")) {
    // The playground calls the runner from the site's origin, so the runner must allow exactly that origin.
    const args = ["--cors-origin", new URL(siteRoot).origin, ...(flags["models-dir"] ? ["--models-dir", flags["models-dir"]] : [])];
    runner = await startRunner({ cli: flags["runner-cli"], args });
    log(`runner at ${runner.base}`);
  }
  const { chromium, AxeBuilder } = await loadBrowserTools(flags["deps-dir"]);
  browser = await chromium.launch({ headless: !flags.headed, executablePath: flags.chrome ?? cachedChromium() });

  // Two chains run at once: the personas (walk, then hands-on) and the checks of the key pages, then
  // comprehension, which reads the page text the checks saved.
  const people = (async () => {
    data.walks = want.has("walk")
      ? await timed("walk", () => pool(personas, concurrency, (persona) => walkPersona({ browser, persona, siteRoot, model, steps, shotsDir, log })))
      : [];
    const tasks = personas.filter((p) => p.handsOn);
    data.handsOn = want.has("hands-on") && runner
      ? await timed("hands-on", () => pool(tasks, concurrency, (persona) => handsOn({ browser, persona, siteRoot, runnerBase: runner.base, model, shotsDir, log })))
      : [];
  })();
  const pages = (async () => {
    data.checks = want.has("checks") || want.has("comprehension")
      ? await timed("checks", () => checkPages({ browser, AxeBuilder, siteRoot, urls: KEY_PAGES.map((p) => siteRoot + p), pagesDir, shotsDir, log }))
      : { pages: [], links: [] };
    const missing = questionPages().filter((slug) => !data.checks.pages.some((p) => p.slug === slug));
    if (want.has("comprehension") && missing.length > 0) log(`no page text for ${missing.join(", ")}; those questions will fail`);
    data.comprehension = want.has("comprehension")
      ? await timed("comprehension", () => runComprehension({ outDir, pagesDir, api: flags.api, model: flags.model, concurrency, log }))
      : { tests: [] };
    const path = Object.fromEntries(data.checks.pages.map((p) => [p.slug, p.path]));
    for (const test of data.comprehension.tests) test.pagePath = path[test.page] ?? `/${test.page}/`;
    log(`comprehension: ${data.comprehension.tests.filter((t) => t.pass).length} of ${data.comprehension.tests.length} passed`);
  })();
  await Promise.all([people, pages]);

  // Pages the personas reached that are not key pages get the same checks, then every link seen is requested.
  const extra = [...new Set(data.walks.flatMap((w) => w.path.map((s) => s.url.split("#")[0])))]
    .filter((url) => url.startsWith(siteRoot) && !KEY_PAGES.some((p) => siteRoot + p === url));
  if (want.has("checks")) {
    if (extra.length > 0) {
      const more = await timed("checks (walked pages)", () => checkPages({ browser, AxeBuilder, siteRoot, urls: extra, pagesDir, shotsDir, log }));
      data.checks.pages.push(...more.pages);
      data.checks.links.push(...more.links.filter((l) => !data.checks.links.some((k) => k.href === l.href)));
    }
    data.broken = await timed("links", () => checkLinks(data.checks.links, { external: flags["external-links"] }));
    log(`links: ${data.broken.length} broken of ${data.checks.links.length}`);
  } else {
    data.broken = [];
  }
  await browser.close();
  browser = null;

  const findings = collectFindings({
    walks: data.walks, handsOn: data.handsOn, comprehension: data.comprehension.tests,
    checks: { pages: want.has("checks") ? data.checks.pages : [] }, broken: data.broken,
  });
  log(`${findings.length} findings`);
  if (want.has("classify") && runner) {
    const labels = Object.keys(CATEGORIES);
    data.findings = await timed("classify", () => pool(findings, concurrency, async (finding) => {
      try {
        const output = await classify(runner.base, { model: flags["classifier-model"], text: classifierText(finding), labels, descriptions: CATEGORIES, several: true });
        return categorize(finding, output);
      } catch (failure) {
        log(`could not classify ${finding.id}: ${failure.message}`);
        return { ...finding, scores: {}, labels: [], primary: finding.known ?? "uncategorized" };
      }
    }));
  } else {
    // Without the classifier, a finding goes in its known category or "uncategorized".
    data.findings = findings.map((f) => ({ ...f, scores: {}, labels: [], primary: f.known ?? "uncategorized" }));
  }
  runner?.stop();
  runner = null;
  data.groups = groupFindings(data.findings);
  data.agreement = want.has("classify") ? agreement(data.findings) : { checked: 0, agreed: 0 };
  data.suggestions = want.has("suggest") ? await timed("suggest", () => suggest({ groups: data.groups, personas, model, concurrency })) : [];

  data.usage = model.usage;
  const promptfooCalls = data.comprehension.stats?.calls ?? 0;
  const callsOf = (stage) => model.usage[stage]?.calls ?? 0;
  data.stages = [
    { name: "walk", ms: timings.walk ?? 0, calls: callsOf("walk"), note: `${data.walks.reduce((s, w) => s + w.path.length, 0)} pages read` },
    { name: "hands-on", ms: timings["hands-on"] ?? 0, calls: callsOf("hands-on"), note: `${data.handsOn.length} playground tasks` },
    { name: "checks", ms: (timings.checks ?? 0) + (timings["checks (walked pages)"] ?? 0) + (timings.links ?? 0), calls: 0, note: `${data.checks.pages.length} pages, ${data.checks.links.length} links` },
    { name: "comprehension (promptfoo)", ms: timings.comprehension ?? 0, calls: promptfooCalls, note: `${data.comprehension.tests.length} questions, answer + grade each` },
    { name: "classify (gliner-runner)", ms: timings.classify ?? 0, calls: 0, note: `${data.findings.length} classifications` },
    { name: "suggest", ms: timings.suggest ?? 0, calls: callsOf("suggest"), note: `${data.groups.length} categories` },
  ];
  data.usage.comprehension = { calls: promptfooCalls };
  data.meta.ms = Date.now() - began;
  data.meta.finished = new Date().toISOString();
  data.meta.versions = { promptfoo: PROMPTFOO, ...JSON.parse(readFileSync(join(here, "package.json"), "utf8")).dependencies, classifier: flags["classifier-model"] };
  data.checks.links = data.checks.links.length;
  for (const walk of data.walks) for (const step of walk.path) step.chose &&= { ...step.chose, path: shortUrl(step.chose.href, siteRoot) };

  writeFileSync(join(outDir, "data.json"), JSON.stringify(data, null, 2));
  writeFileSync(join(outDir, "report.md"), renderReport(data));
  log(`done: ${join(outDir, "report.md")}`);
} finally {
  await stop();
}
