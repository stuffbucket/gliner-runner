#!/usr/bin/env node
// Writes, checks and tests short website copy for one slot, and ranks the candidates.
//
//   node tools/copy-lab/lab.mjs --slot fifteen-seconds [--slots slots.json] [--out /tmp/gr-work/copy-lab] [--concurrency 12]
//
// Stages: generate (several drafting prompts x several models), check (word count, sentence length, reading
// grade, stall words, meta; failures dropped with the reason), retell (promptfoo: a disinterested friend reads only
// the candidate and retells it; graded against the slot's facts), judge (the top candidates pairwise, both orders,
// by several models; Bradley-Terry). A slot with `show` also scores one-sentence lead-ins for an embedded demo.
// Model replies and retell rows are cached, so a re-run only pays for what changed. See README.md.
import { mkdirSync, readFileSync, writeFileSync } from "node:fs";
import { dirname, join, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import { parseArgs } from "node:util";

import { check } from "./lib/constraints.mjs";
import { candidateId, generate } from "./lib/generate.mjs";
import { runJudging } from "./lib/judge.mjs";
import { createClient } from "./lib/model.mjs";
import { renderReport } from "./lib/report.mjs";
import { runRetell, scoreRetell } from "./lib/retell.mjs";

const here = dirname(fileURLToPath(import.meta.url));

const { values: flags } = parseArgs({
  options: {
    slot: { type: "string" },
    slots: { type: "string", default: join(here, "slots.json") },
    out: { type: "string", default: "/tmp/gr-work/copy-lab" },
    api: { type: "string", default: "http://localhost:4141/v1" },
    concurrency: { type: "string", default: "12" },
    help: { type: "boolean", default: false },
  },
});
const config = JSON.parse(readFileSync(flags.slots, "utf8"));
if (flags.help || !flags.slot) {
  console.log(readFileSync(fileURLToPath(import.meta.url), "utf8").split("\n").slice(1, 10).map((l) => l.replace(/^\/\/ ?/, "")).join("\n"));
  console.log(`\nSlots in ${flags.slots}: ${config.slots.map((s) => s.id).join(", ")} (--slot all runs every one)`);
  process.exit(flags.help ? 0 : 1);
}

const outRoot = resolve(flags.out);
const concurrency = Number(flags.concurrency);
const began = Date.now();
const log = (message) => console.log(`[${((Date.now() - began) / 1000).toFixed(0).padStart(4)}s] ${message}`);
const client = createClient({ base: flags.api, cacheDir: join(outRoot, ".cache") });
const ids = flags.slot === "all" ? config.slots.map((s) => s.id) : flags.slot.split(",");

/** Ranks by retell score, then the optional facts, then the shorter text. */
const byRetell = (a, b) =>
  (b.retell?.score ?? -1) - (a.retell?.score ?? -1) || (b.retell?.optional ?? 0) - (a.retell?.optional ?? 0) || a.checks.words - b.checks.words;

/**
 * At most `limit` survivors, taken in turn from each source (drafting model and angle) so that a cap on the retell
 * stage keeps every source in play rather than the first few replies.
 */
function spread(survivors, limit) {
  const groups = Map.groupBy(survivors, (c) => `${c.source.model}/${c.source.angle}`);
  const queues = [...groups.values()];
  const picked = [];
  while (picked.length < Math.min(limit, survivors.length)) {
    for (const q of queues) if (q.length > 0 && picked.length < limit) picked.push(q.shift());
  }
  return picked;
}

/** Generate, check, retell and (when `judge`) judge one pool of candidates for one slot. */
async function runPool(slot, dir, { judge, timings }) {
  mkdirSync(dir, { recursive: true });
  const timed = async (stage, work) => {
    const started = Date.now();
    try {
      return await work();
    } finally {
      timings[stage] = (timings[stage] ?? 0) + Date.now() - started;
    }
  };
  const drafts = await timed("generate", () => generate({ slot, client, concurrency, log }));
  const baselines = (slot.baselines ?? []).map((b) => ({
    id: candidateId("b", b.text),
    text: b.text,
    source: { kind: "baseline", name: b.name },
    reference: Boolean(b.reference),
  }));
  for (const c of [...drafts, ...baselines]) c.checks = check(c.text, slot);
  const drops = drafts.filter((c) => c.checks.reasons.length > 0);
  const survivors = drafts.filter((c) => c.checks.reasons.length === 0);
  const retold = spread(survivors, slot.maxRetell ?? Infinity);
  log(`${slot.id}: ${drafts.length} drafted, ${drops.length} dropped, ${survivors.length} survive, ${retold.length} retold`);

  const { rows, stats } = await timed("retell", () =>
    runRetell({ slot, candidates: [...retold, ...baselines], dir, api: flags.api, cacheDir: join(outRoot, ".cache"), concurrency, log }));
  scoreRetell([...retold, ...baselines], rows, slot.facts);
  const ranked = [...survivors].sort(byRetell);

  let top = ranked;
  let judged = [];
  let judging = { games: [], positionBias: null };
  if (judge) {
    const entrants = [...ranked.slice(0, slot.judgeTop), ...baselines.filter((b) => b.reference)];
    judging = await timed("judge", () => runJudging({ slot, entrants, client, concurrency, log }));
    judged = [...entrants].sort((a, b) => b.judge.strength - a.judge.strength);
    top = judged.filter((e) => e.source.kind !== "baseline");
  }

  writeFileSync(join(dir, "candidates.json"), JSON.stringify({ slot: slot.id, candidates: [...ranked, ...baselines] }, null, 2));
  writeFileSync(join(dir, "drops.json"), JSON.stringify(drops.map((d) => ({ id: d.id, text: d.text, reasons: d.checks.reasons, source: d.source })), null, 2));
  writeFileSync(join(dir, "retell.json"), JSON.stringify(rows, null, 2));
  if (judge) writeFileSync(join(dir, "judge.json"), JSON.stringify(judging, null, 2));
  return { generated: drafts.length, drops, survivors, retold, baselines, ranked, top, judged, positionBias: judging.positionBias, judgeFailures: judging.failures ?? 0, promptfoo: stats };
}

const resolveSlot = (slot) => ({ ...config.defaults, ...slot });

const summary = [];
for (const id of ids) {
  const raw = config.slots.find((s) => s.id === id);
  if (!raw) throw new Error(`no slot "${id}" in ${flags.slots}`);
  const slot = resolveSlot(raw);
  const dir = join(outRoot, id);
  const before = JSON.parse(JSON.stringify(client.usage));
  const timings = {};
  const promptfooCalls = [];
  log(`slot ${id} into ${dir}`);

  const result = await runPool(slot, dir, { judge: true, timings });
  promptfooCalls.push(result.promptfoo);
  let show = null;
  if (slot.show) {
    const showSlot = { ...slot, ...slot.show, id: `${id}-show`, baselines: [] };
    show = await runPool(showSlot, join(dir, "show"), { judge: false, timings });
    promptfooCalls.push(show.promptfoo);
  }

  const delta = (stage, field) => (client.usage[stage]?.[field] ?? 0) - (before[stage]?.[field] ?? 0);
  const cost = [
    { stage: "generate", calls: delta("generate", "calls"), cached: delta("generate", "cached"), ms: timings.generate ?? 0 },
    { stage: "retell (promptfoo)", calls: promptfooCalls.reduce((s, p) => s + p.calls, 0), cached: promptfooCalls.reduce((s, p) => s + p.cachedRows * 2, 0), ms: timings.retell ?? 0 },
    { stage: "judge", calls: delta("judge", "calls"), cached: delta("judge", "cached"), ms: timings.judge ?? 0 },
  ];
  cost.push({ stage: "total", calls: cost.reduce((s, c) => s + c.calls, 0), cached: cost.reduce((s, c) => s + c.cached, 0), ms: cost.reduce((s, c) => s + c.ms, 0) });
  writeFileSync(join(dir, "report.md"), renderReport({ slot, result, show, cost }));
  writeFileSync(join(dir, "run.json"), JSON.stringify({ slot: id, cost, usage: client.usage, finished: new Date().toISOString() }, null, 2));
  summary.push({ id, cost: cost.at(-1), top: result.top.slice(0, 3).map((c) => c.text) });
  log(`slot ${id}: report at ${join(dir, "report.md")}`);
}

log(`done in ${((Date.now() - began) / 1000).toFixed(0)} s`);
for (const s of summary) console.log(`${s.id}: ${s.cost.calls} live calls, ${s.cost.cached} cached, ${(s.cost.ms / 1000).toFixed(0)} s`);
