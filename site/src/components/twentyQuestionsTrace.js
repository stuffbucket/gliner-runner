// The trace behind "Why?" in 20 questions: for one thing (say, airplane), every step from the root down to it,
// with what the model gave each option at that step, so it is clear where the path was won or lost. A name can sit
// in more than one place (a kiwi is a bird and a fruit), so the trace is every path that ends in that name: still a
// tree, the part of the game's tree that leads to it, with each step's other options alongside.
// Like twentyQuestions.js the functions are pure: the playground, simulate.mjs --trace and the tests share them.

import { TUNING, isGuess, key, levelOf, levelQuestion, nodeAt, temper } from "./twentyQuestions.js";

/** Above this lead (in probability) a first place is clear; below it, or within CLOSE_RATIO, it is a close call. */
export const CLEAR_LEAD = 0.2;
export const CLOSE_RATIO = 2;

/** Every label path from the root to a thing called `name`. */
export function pathsTo(tree, name) {
  const found = [];
  (function walk(node, path) {
    if (isGuess(node)) {
      if (node.label === name) found.push(path);
      return;
    }
    for (const child of node.children) walk(child, [...path, child.label]);
  })(tree, []);
  return found;
}

/** The names of every thing in the tree, once each, sorted. */
export function thingNames(tree) {
  const names = new Set();
  (function walk(node) {
    if (isGuess(node)) names.add(node.label);
    else node.children.forEach(walk);
  })(tree);
  return [...names].sort((a, b) => a.localeCompare(b));
}

const level = (tree, path) => {
  const node = nodeAt(tree, path);
  return { nodeKey: key(path), path, question: levelQuestion(path, node.children), ...levelOf(node) };
};

/** The classifications a trace of `paths` needs: the root and every group on the way, each once. */
export function levelsFor(tree, paths) {
  const byKey = new Map();
  for (const path of paths) {
    for (let depth = 0; depth < path.length; depth += 1) {
      const parent = path.slice(0, depth);
      if (!byKey.has(key(parent))) byKey.set(key(parent), level(tree, parent));
    }
  }
  return [...byKey.values()];
}

/** Every classification in the tree (for the overall rank of every thing). */
export function allLevels(tree) {
  const found = [];
  (function walk(node, path) {
    if (isGuess(node)) return;
    found.push(level(tree, path));
    node.children.forEach((child) => walk(child, [...path, child.label]));
  })(tree, []);
  return found;
}

/**
 * How sure the model was at one step, from 0 (every option equally likely) to 1 (all on one option):
 * 1 minus the entropy of the probabilities over its largest possible value, ln n.
 */
export function sureness(values) {
  const total = values.reduce((sum, p) => sum + p, 0) || 1;
  if (values.length < 2) return 1;
  const entropy = values.reduce((sum, p) => (p > 0 ? sum - (p / total) * Math.log(p / total) : sum), 0);
  return 1 - entropy / Math.log(values.length);
}

/** ok: first by a clear lead; close: first by a little, or behind by less than CLOSE_RATIO; lost: further behind. */
export function status(option) {
  if (option.rank === 1) return option.lead >= CLEAR_LEAD ? "ok" : "close";
  return option.behind < CLOSE_RATIO ? "close" : "lost";
}

/**
 * The trace for `name`. `levels` maps a level's nodeKey to the probabilities its classification returned (missing
 * levels are left unscored). `ruledOut` holds the nodeKeys of wrong guesses in the game, if any.
 *
 * Returns `{ name, paths, root, best, verdict }`: `root` is the first step; every step has `question`, `n`,
 * `sureness` and `options` (best first), and an option on the way to `name` has its `child` step (or `isTarget` at
 * the end). Each option has `p` (its probability at this step), `rank`, `lead` (how far ahead of the next best) or
 * `behind` (how many times less likely than the best), and `status`. `paths` sums up each path: `pathP` (the
 * product of its steps, tempered as the game does: what the game ranks things by), the `weakest` step (lowest p),
 * `geoMean` (the typical p per step) and the step it was `lostAt`, if any.
 */
export function trace(tree, name, levels, { ruledOut = new Set(), tuning = TUNING } = {}) {
  const paths = pathsTo(tree, name);
  if (paths.length === 0) return { name, paths: [], root: undefined, best: undefined, verdict: `“${name}” is not one of the things the game knows.` };
  const onPath = new Set(paths.flatMap((path) => path.map((_, depth) => key(path.slice(0, depth + 1)))));

  function step(path) {
    const node = nodeAt(tree, path);
    const nodeKey = key(path);
    const probabilities = levels.get(nodeKey);
    const base = { nodeKey, path, question: levelQuestion(path, node.children), n: node.children.length };
    if (probabilities === undefined) return { ...base, scored: false, options: [] };
    const labels = node.children.map((child) => child.label);
    const shares = temper(labels, probabilities, tuning);
    const ranked = node.children
      .map((child, index) => ({ label: child.label, nodeKey: key([...path, child.label]), p: probabilities[child.label] ?? 0, share: shares[index] }))
      .sort((a, b) => b.p - a.p);
    const options = ranked.map((option, index) => {
      const best = index === 0 ? ranked[1] : ranked[0];
      const scored = {
        ...option,
        rank: index + 1,
        lead: index === 0 ? option.p - (best?.p ?? 0) : 0,
        behind: index === 0 ? 1 : ranked[0].p / Math.max(option.p, Number.EPSILON),
        isWinner: index === 0,
        onPath: onPath.has(option.nodeKey),
        isTarget: onPath.has(option.nodeKey) && option.label === name && isGuess(nodeAt(tree, [...path, option.label])),
        ruledOut: ruledOut.has(option.nodeKey),
      };
      scored.status = status(scored);
      if (scored.onPath && !scored.isTarget) scored.child = step([...path, option.label]);
      return scored;
    });
    return { ...base, scored: true, sureness: sureness(options.map((o) => o.p)), options };
  }

  const root = step([]);
  const summaries = paths.map((path) => {
    const steps = [];
    let current = root;
    for (const label of path) {
      const option = current?.options.find((o) => o.label === label);
      if (option === undefined) break;
      steps.push({ step: current, option });
      current = option.child;
    }
    const complete = steps.length === path.length;
    const pathP = complete ? steps.reduce((product, s) => product * s.option.share, 1) : undefined;
    const weakest = steps.reduce((low, s) => (low === undefined || s.option.p < low.option.p ? s : low), undefined);
    const geoMean = complete ? Math.exp(steps.reduce((sum, s) => sum + Math.log(Math.max(s.option.p, Number.EPSILON)), 0) / steps.length) : undefined;
    const lostAt = steps.find((s) => s.option.status === "lost") ?? steps.find((s) => s.option.status === "close");
    return { path, steps, complete, pathP, weakest, geoMean, lostAt };
  });
  const best = [...summaries].sort((a, b) => (b.pathP ?? -1) - (a.pathP ?? -1))[0];
  return { name, paths: summaries, root, best, verdict: verdict(name, best, summaries.length) };
}

const two = (p) => p.toFixed(2);
const times = (ratio) => `${ratio < 10 ? ratio.toFixed(1) : Math.round(ratio)}×`;

/** One sentence on the most likely path: where it was lost, where it was a close call, or that it was clear. */
function verdict(name, best, count) {
  if (best === undefined || !best.complete) return `Not every step on the way to ${name} has been scored yet.`;
  const via = count > 1 ? ` (by way of ${best.path.slice(0, -1).join(" › ")})` : "";
  const at = best.lostAt;
  if (at === undefined) return `On track all the way${via}: ${name} came first at every step, with a path chance of ${two(best.pathP)}.`;
  const { option, step } = at;
  const winner = step.options[0];
  const runnerUp = step.options[1];
  if (option.status === "lost") {
    return `Lost at “${step.question}”${via}: ${winner.label} (${two(winner.p)}) was ${times(option.behind)} more likely than ${option.label} (${two(option.p)}).`;
  }
  return option.rank === 1
    ? `A close call at “${step.question}”${via}: ${option.label} (${two(option.p)}) only just beat ${runnerUp.label} (${two(runnerUp.p)}).`
    : `A close call at “${step.question}”${via}: ${winner.label} (${two(winner.p)}) edged out ${option.label} (${two(option.p)}).`;
}

/**
 * Every thing whose path is fully scored, with its path chance (tempered as the game does), most likely first.
 * With every level scored (allLevels) this ranks all the things the game knows.
 */
export function leafRanking(tree, levels, tuning = TUNING) {
  const ranked = [];
  (function walk(node, path, chance) {
    if (isGuess(node)) {
      ranked.push({ label: node.label, nodeKey: key(path), p: chance });
      return;
    }
    const probabilities = levels.get(key(path));
    if (probabilities === undefined) return;
    const labels = node.children.map((child) => child.label);
    const shares = temper(labels, probabilities, tuning);
    node.children.forEach((child, index) => walk(child, [...path, child.label], chance * shares[index]));
  })(tree, [], 1);
  return ranked.sort((a, b) => b.p - a.p);
}

/**
 * The options of a step to draw for the probability range [low, high]: those inside it, and every option on the way
 * to the thing being traced whatever its probability, plus how many were left out above and below the range.
 */
export function inRange(options, [low, high]) {
  const shown = options.filter((o) => o.onPath || (o.p >= low && o.p <= high));
  return {
    shown,
    above: options.filter((o) => !o.onPath && o.p > high).length,
    below: options.filter((o) => !o.onPath && o.p < low).length,
  };
}
