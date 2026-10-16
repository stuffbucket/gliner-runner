// Plays "20 questions" for every answer in answers.json against the real model, with the playground's own game
// logic (src/components/twentyQuestions.js), answering each guess honestly. Use it to tune the tree:
//
//   GLINER_RUNNER_MODELS_DIR=dist/onnx-models node site/scripts/twenty-questions/simulate.mjs [--set clue|alt] \
//     [--model decide-340m] [--only animal] [--report /tmp/report.json]
//
// `clue` is the tuning set and `alt` the held-out set: tune against `clue`, then check `alt` did not get worse.

import { createHash } from "node:crypto";
import { existsSync, readFileSync, writeFileSync } from "node:fs";
import { parseArgs } from "node:util";

import { MAX_QUESTIONS, afterClassification, afterReply, newGame, nextStep, startGuess } from "../../src/components/twentyQuestions.js";
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
const top = (probabilities) => Object.entries(probabilities).sort((a, b) => b[1] - a[1])[0][0];

/** Plays one game; returns the number of questions asked, whether it won, and the path it took. */
async function play(tree, item, clues) {
  let game = newGame(tree, clues);
  const trail = [];
  while (game.phase === "asking") {
    const step = nextStep(game);
    if (step.kind === "guess") {
      game = startGuess(game, step);
      const yes = step.label === item.name;
      trail.push(yes ? `✓${step.label}` : `✗${step.label}`);
      game = afterReply(game, yes);
      continue;
    }
    const probabilities = await scores(clues, step.labels, step.descriptions);
    trail.push(`[${top(probabilities)}]`);
    game = afterClassification(game, step, probabilities);
  }
  return { won: game.phase === "won", asked: game.asked, trail };
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
  const level = (node) => ({
    labels: node.children.map((c) => c.label),
    descriptions: node.children.some((c) => c.description) ? Object.fromEntries(node.children.filter((c) => c.description).map((c) => [c.label, c.description])) : undefined,
  });
  const rows = await pool(games, Number(values.concurrency), async ({ category, item }) => {
    const text = item[values.set];
    const categoryNode = tree.children.find((c) => c.label === category);
    const groupNode = categoryNode.children.find((g) => g.label === item.group);
    const root = level(tree);
    const atRoot = await scores(text, root.labels, root.descriptions);
    const inCategory = level(categoryNode);
    const atCategory = await scores(text, inCategory.labels, inCategory.descriptions);
    const inGroup = level(groupNode);
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

const runner = await startRunner();
base = runner.base;
try {
  if (values.diagnose) {
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
    const outcome = await play(tree, item, item[values.set]);
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
    return {
      games: rows.length,
      won: won.length,
      winRate: percent(won.length, rows.length),
      meanQuestions: asked.length ? (asked.reduce((a, b) => a + b, 0) / asked.length).toFixed(2) : "-",
      p90Questions: asked.length ? asked[Math.floor(asked.length * 0.9)] : "-",
      firstQuestionRight: percent(rows.filter((r) => r.firstCategory).length, rows.length),
      groupRightFirstTime: percent(rows.filter((r) => r.firstGroup).length, rows.length),
    };
  };
  console.log(`set=${values.set} model=${values.model} games=${results.length} in ${((Date.now() - started) / 1000).toFixed(0)} s`);
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
