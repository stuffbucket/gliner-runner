// Tunes the 20 questions search settings (TUNING in src/components/twentyQuestions.js). It classifies every clue
// against every level of the tree once (cached, see simulate.mjs), then replays every game offline for a grid of
// settings, which is instant because the engine never classifies the same level twice.
//
//   GLINER_RUNNER_MODELS_DIR=dist/onnx-models node site/scripts/twenty-questions/tune.mjs [--set clue|alt]

import { createHash } from "node:crypto";
import { existsSync, readFileSync, writeFileSync } from "node:fs";
import { parseArgs } from "node:util";

import { afterClassification, afterReply, newGame, nextStep, startGuess } from "../../src/components/twentyQuestions.js";
import { classify, startRunner } from "../runner.mjs";
import { buildTree, loadAnswers } from "./tree.mjs";

const { values } = parseArgs({
  options: {
    set: { type: "string", default: "clue" },
    model: { type: "string", default: "decide-340m" },
    cache: { type: "string", default: "/tmp/twenty-questions-cache.json" },
    concurrency: { type: "string", default: "8" },
  },
});

const cache = existsSync(values.cache) ? new Map(Object.entries(JSON.parse(readFileSync(values.cache, "utf8")))) : new Map();
const id = (text, labels, descriptions) =>
  createHash("sha256").update(JSON.stringify([values.model, text, labels, descriptions ?? null])).digest("hex");

const answers = loadAnswers();
const { tree } = buildTree(answers);
const levels = [];
(function collect(node) {
  if (node.children === undefined) return;
  const descriptions = node.children.some((c) => c.description)
    ? Object.fromEntries(node.children.filter((c) => c.description).map((c) => [c.label, c.description]))
    : undefined;
  levels.push({ labels: node.children.map((c) => c.label), descriptions });
  node.children.forEach(collect);
})(tree);
const games = answers.categories.flatMap((c) => c.items.map((item) => ({ name: item.name, text: item[values.set] })));

// Fill the cache for every (clue, level) pair.
const missing = games.flatMap((game) => levels.map((level) => ({ text: game.text, ...level }))).filter((q) => !cache.has(id(q.text, q.labels, q.descriptions)));
if (missing.length > 0) {
  console.log(`classifying ${missing.length} (clue, level) pairs…`);
  const runner = await startRunner();
  try {
    let done = 0;
    await Promise.all(
      Array.from({ length: Number(values.concurrency) }, async () => {
        while (missing.length > 0) {
          const q = missing.pop();
          const out = await classify(runner.base, { model: values.model, text: q.text, labels: q.labels, descriptions: q.descriptions });
          cache.set(id(q.text, q.labels, q.descriptions), out.probabilities);
          if (++done % 2000 === 0) console.log(`  ${done}`);
        }
      }),
    );
  } finally {
    runner.stop();
    writeFileSync(values.cache, JSON.stringify(Object.fromEntries(cache)));
  }
}

function play({ name, text }, tuning) {
  let game = newGame(tree, text);
  while (game.phase === "asking") {
    const step = nextStep(game, tuning);
    if (step.kind === "guess") {
      game = afterReply(startGuess(game, step), step.label === name);
      continue;
    }
    game = afterClassification(game, step, cache.get(id(text, step.labels, step.descriptions)), tuning);
  }
  return game;
}

const results = [];
for (const discount of [1, 0.8, 0.6, 0.45, 0.35, 0.25, 0.18, 0.12]) {
  for (const temperature of [0.7, 1, 1.4, 2, 3]) {
    const outcomes = games.map((game) => play(game, { discount, temperature }));
    const won = outcomes.filter((g) => g.phase === "won");
    results.push({
      discount,
      temperature,
      winRate: won.length / games.length,
      meanAsked: won.reduce((sum, g) => sum + g.asked, 0) / Math.max(won.length, 1),
    });
  }
}
results.sort((a, b) => b.winRate - a.winRate || a.meanAsked - b.meanAsked);
console.log(`set=${values.set} games=${games.length}`);
for (const r of results.slice(0, 12)) {
  console.log(`  discount ${String(r.discount).padEnd(5)} temperature ${String(r.temperature).padEnd(4)} won ${(100 * r.winRate).toFixed(1)}%  mean questions ${r.meanAsked.toFixed(2)}`);
}
const baseline = results.find((r) => r.discount === 1 && r.temperature === 1);
console.log(`  (no tuning: won ${(100 * baseline.winRate).toFixed(1)}%, mean ${baseline.meanAsked.toFixed(2)})`);
