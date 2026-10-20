// Plays "20 questions" for every answer in answers.json against the real model, with the playground's own game
// logic (src/components/twentyQuestions.js), answering each guess honestly. Use it to tune the tree:
//
//   GLINER_RUNNER_MODELS_DIR=dist/onnx-models node site/scripts/twenty-questions/simulate.mjs [--set clue|alt|both] \
//     [--staged] [--model decide-340m] [--only animal] [--report /tmp/report.json]
//
// `clue` is the tuning set and `alt` the held-out set: tune against `clue`, then check `alt` did not get worse.
// With --staged each game starts from `alt` and adds `clue` after the first wrong guess, the way a player adds a
// clue in the playground: it should do about as well as `--set both` (`alt` and `clue` together from the start).
//
// To see why one description does or does not reach one thing (the playground's "Why?"), trace it:
//
//   node site/scripts/twenty-questions/simulate.mjs --trace airplane --text "It moves fast and really high up."
//
// Without --text it traces the thing's own `clue` (or `alt` with --set alt).

import { createHash } from "node:crypto";
import { existsSync, readFileSync, writeFileSync } from "node:fs";
import { parseArgs } from "node:util";

import { MAX_QUESTIONS, afterClassification, afterReply, levelOf, newGame, nextStep, startGuess } from "../../src/components/twentyQuestions.js";
import { levelsFor, pathsTo, trace } from "../../src/components/twentyQuestionsTrace.js";
import { classify, startRunner } from "../runner.mjs";
import { buildTree, loadAnswers } from "./tree.mjs";

const { values } = parseArgs({
  options: {
    set: { type: "string", default: "clue" },
    model: { type: "string", default: "decide-340m" },
    only: { type: "string" },
    report: { type: "string" },
    concurrency: { type: "string", default: "8" },
    cache: { type: "string", default: "/tmp/twenty-questions-cache.json" },
    diagnose: { type: "boolean" },
    staged: { type: "boolean" },
    trace: { type: "string" },
    text: { type: "string" },
  },
});

// Classifications are deterministic, so they are cached: a change to one group only recomputes that group.
const cache = existsSync(values.cache) ? new Map(Object.entries(JSON.parse(readFileSync(values.cache, "utf8")))) : new Map();
let base;
async function scores(text, labels, descriptions) {
  const id = createHash("sha256").update(JSON.stringify([values.model, text, labels, descriptions ?? null])).digest("hex");
  if (!cache.has(id)) {
    cache.set(id, (await classify(base, { model: values.model, text, labels, descriptions })).probabilities);
  }
  return cache.get(id);
}
/** The clues for one thing in --set: `clue`, `alt`, or `both` (`alt` then `clue`, as --staged ends up with). */
const textOf = (item) => (values.set === "both" ? `${item.alt} ${item.clue}` : item[values.set]);
const top = (probabilities) => Object.entries(probabilities).sort((a, b) => b[1] - a[1])[0][0];

/**
 * Plays one game; returns the number of questions put to the player, the model calls made, whether it won, and the
 * path it took. Any `later` clue is added after the first wrong guess, as a player would in the playground.
 */
async function play(tree, item, clues, later) {
  let game = newGame(tree, clues);
  let calls = 0;
  const trail = [];
  while (game.phase === "asking") {
    const step = nextStep(game);
    if (step.kind === "guess") {
      game = startGuess(game, step);
      const yes = step.label === item.name;
      trail.push(yes ? `✓${step.label}` : `✗${step.label}`);
      game = afterReply(game, yes);
      if (!yes && later !== undefined && game.clues === clues) {
        game = { ...game, clues: `${clues} ${later}` };
        trail.push("+clue");
      }
      continue;
    }
    const probabilities = await scores(step.clues, step.labels, step.descriptions);
    calls += 1;
    trail.push(`[${step.again ? "again: " : ""}${top(probabilities)}]`);
    game = afterClassification(game, step, probabilities);
  }
  return { won: game.phase === "won", asked: game.asked, calls, trail };
}

async function pool(items, size, work) {
  const results = new Array(items.length);
  let next = 0;
  await Promise.all(
    Array.from({ length: size }, async () => {
      while (next < items.length) {
        const index = next++;
        results[index] = await work(items[index]);
      }
    }),
  );
  return results;
}

const answers = loadAnswers();
const { tree } = buildTree(answers);
/** Scores the true path of every clue: the category, the group within it, and the item's rank within its group. */
async function diagnose(games) {
  const rows = await pool(games, Number(values.concurrency), async ({ category, item }) => {
    const text = textOf(item);
    const categoryNode = tree.children.find((c) => c.label === category);
    const groupNode = categoryNode.children.find((g) => g.label === item.group);
    const root = levelOf(tree);
    const atRoot = await scores(text, root.labels, root.descriptions);
    const inCategory = levelOf(categoryNode);
    const atCategory = await scores(text, inCategory.labels, inCategory.descriptions);
    const inGroup = levelOf(groupNode);
    const atGroup = await scores(text, inGroup.labels, inGroup.descriptions);
    const rank = Object.entries(atGroup).sort((a, b) => b[1] - a[1]).findIndex(([label]) => label === item.name) + 1;
    return { category, group: item.group, name: item.name, rootPick: top(atRoot), groupPick: top(atCategory), groupProb: atCategory[item.group], itemPick: top(atGroup), rank, size: inGroup.labels.length };
  });
  const pct = (n, d) => `${((100 * n) / Math.max(d, 1)).toFixed(1)}%`;
  for (const category of [...new Set(rows.map((r) => r.category))]) {
    const mine = rows.filter((r) => r.category === category);
    console.log(`\n${category}: category right ${pct(mine.filter((r) => r.rootPick === category).length, mine.length)}, ` +
      `group right ${pct(mine.filter((r) => r.groupPick === r.group).length, mine.length)}, ` +
      `item top-1 ${pct(mine.filter((r) => r.rank === 1).length, mine.length)}, top-3 ${pct(mine.filter((r) => r.rank <= 3).length, mine.length)}, top-5 ${pct(mine.filter((r) => r.rank <= 5).length, mine.length)}`);
    const wrongRoot = new Map();
    for (const r of mine.filter((r) => r.rootPick !== category)) wrongRoot.set(r.name, r.rootPick);
    if (wrongRoot.size) console.log(`  first question wrong: ${[...wrongRoot].map(([n, p]) => `${n}→${p}`).join(", ")}`);
    for (const group of [...new Set(mine.map((r) => r.group))]) {
      const g = mine.filter((r) => r.group === group);
      const confusions = new Map();
      for (const r of g.filter((r) => r.groupPick !== group)) confusions.set(r.groupPick, (confusions.get(r.groupPick) ?? 0) + 1);
      console.log(`  ${group.padEnd(32)} group ${pct(g.filter((r) => r.groupPick === group).length, g.length).padStart(6)}  item top-1 ${pct(g.filter((r) => r.rank === 1).length, g.length).padStart(6)} top-5 ${pct(g.filter((r) => r.rank <= 5).length, g.length).padStart(6)}  ` +
        `→ ${[...confusions].sort((a, b) => b[1] - a[1]).map(([p, n]) => `${p} ${n}`).join(", ")}`);
      const missed = g.filter((r) => r.rank > 5).map((r) => `${r.name}(#${r.rank}→${r.itemPick})`);
      if (missed.length) console.log(`      items ranked below 5: ${missed.join(", ")}`);
    }
  }
}

/** Prints the trace of one thing for one description: every step on the way to it, and where it was won or lost. */
async function printTrace(name) {
  const paths = pathsTo(tree, name);
  const item = answers.categories.flatMap((c) => c.items).find((i) => i.name === name);
  const text = values.text ?? (item && textOf(item));
  if (paths.length === 0 || text === undefined) {
    console.log(paths.length === 0 ? `“${name}” is not in answers.json.` : `Give a description with --text.`);
    return;
  }
  const levels = new Map();
  for (const level of levelsFor(tree, paths)) levels.set(level.nodeKey, await scores(text, level.labels, level.descriptions));
  const result = trace(tree, name, levels);
  const mark = { ok: "✓", close: "~", lost: "✗" };
  console.log(`“${text}”\n`);
  (function show(step, depth) {
    const pad = "  ".repeat(depth);
    console.log(`${pad}${step.question}  (${step.n} options, sureness ${step.sureness.toFixed(2)})`);
    for (const option of step.options.filter((o) => o.onPath || o.rank <= 3)) {
      const where = option.rank === 1 ? `ahead by ${option.lead.toFixed(2)}` : `${option.behind.toFixed(1)}× behind`;
      const note = option.onPath ? `  ${mark[option.status]} #${option.rank} of ${step.n}, ${where}` : "";
      console.log(`${pad}  ${option.onPath ? "»" : " "} ${option.label.padEnd(32)} ${option.p.toFixed(3)}${note}`);
      if (option.child) show(option.child, depth + 1);
    }
  })(result.root, 0);
  for (const path of result.paths) {
    console.log(`\n${path.path.join(" › ")}: path chance ${path.pathP.toFixed(3)}, weakest step ${path.weakest.option.p.toFixed(3)} ` +
      `(${path.weakest.step.question}), typical step ${path.geoMean.toFixed(3)}`);
  }
  console.log(`\n${result.verdict}`);
}

const runner = await startRunner();
base = runner.base;
try {
  if (values.trace) {
    await printTrace(values.trace);
  } else if (values.diagnose) {
    const games = answers.categories
      .filter((category) => values.only === undefined || category.label === values.only)
      .flatMap((category) => category.items.map((item) => ({ category: category.label, item })));
    await diagnose(games);
  } else {
  const games = answers.categories
    .filter((category) => values.only === undefined || category.label === values.only)
    .flatMap((category) => category.items.map((item) => ({ category: category.label, item })));
  const started = Date.now();
  const results = await pool(games, Number(values.concurrency), async ({ category, item }) => {
    const outcome = values.staged ? await play(tree, item, item.alt, item.clue) : await play(tree, item, textOf(item));
    return {
      category,
      group: item.group,
      name: item.name,
      ...outcome,
      firstCategory: outcome.trail[0] === `[${category}]`,
      firstGroup: outcome.trail[0] === `[${category}]` && outcome.trail[1] === `[${item.group}]`,
    };
  });

  const percent = (n, d) => `${((100 * n) / Math.max(d, 1)).toFixed(1)}%`;
  const summarize = (rows) => {
    const won = rows.filter((r) => r.won);
    const asked = won.map((r) => r.asked).sort((a, b) => a - b);
    const calls = rows.map((r) => r.calls).sort((a, b) => a - b);
    return {
      games: rows.length,
      won: won.length,
      winRate: percent(won.length, rows.length),
      meanQuestions: asked.length ? (asked.reduce((a, b) => a + b, 0) / asked.length).toFixed(2) : "-",
      p90Questions: asked.length ? asked[Math.floor(asked.length * 0.9)] : "-",
      meanModelCalls: (calls.reduce((a, b) => a + b, 0) / Math.max(calls.length, 1)).toFixed(1),
      p90ModelCalls: calls.length ? calls[Math.floor(calls.length * 0.9)] : "-",
      firstQuestionRight: percent(rows.filter((r) => r.firstCategory).length, rows.length),
      groupRightFirstTime: percent(rows.filter((r) => r.firstGroup).length, rows.length),
    };
  };
  console.log(`set=${values.staged ? "alt then clue" : values.set} model=${values.model} games=${results.length} in ${((Date.now() - started) / 1000).toFixed(0)} s`);
  console.table(Object.fromEntries([...new Set(results.map((r) => r.category))].map((c) => [c, summarize(results.filter((r) => r.category === c))])));
  console.log("all:", summarize(results));
  const groups = new Map();
  for (const row of results) {
    const entry = groups.get(row.group) ?? { group: row.group, games: 0, won: 0, groupRight: 0 };
    entry.games += 1;
    entry.won += row.won ? 1 : 0;
    entry.groupRight += row.firstGroup ? 1 : 0;
    groups.set(row.group, entry);
  }
  console.log("groups (win rate, group picked first time):");
  for (const entry of [...groups.values()].sort((a, b) => a.won / a.games - b.won / b.games)) {
    console.log(`  ${entry.group.padEnd(32)} ${percent(entry.won, entry.games).padStart(6)}  ${percent(entry.groupRight, entry.games).padStart(6)}  (${entry.games})`);
  }
  const lost = results.filter((r) => !r.won);
  console.log(`lost ${lost.length} (more than ${MAX_QUESTIONS} questions):`);
  for (const row of lost) console.log(`  ${row.category}/${row.group}/${row.name}: ${row.trail.join(" › ")}`);
  if (values.report) writeFileSync(values.report, `${JSON.stringify(results, null, 2)}\n`);
  }
} finally {
  runner.stop();
  writeFileSync(values.cache, JSON.stringify(Object.fromEntries(cache)));
}
