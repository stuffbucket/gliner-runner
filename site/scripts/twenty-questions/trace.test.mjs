// Tests for the 20 questions trace (src/components/twentyQuestionsTrace.js) on a small hand-made tree.
import assert from "node:assert/strict";
import { test } from "node:test";

import { afterClassification, newGame, nextStep } from "../../src/components/twentyQuestions.js";
import { allLevels, inRange, leafRanking, levelsFor, likeliest, pathsTo, sureness, thingNames, trace } from "../../src/components/twentyQuestionsTrace.js";

const tree = {
  label: "anything",
  children: [
    {
      label: "animal",
      children: [
        { label: "bird", children: [{ label: "kiwi" }, { label: "owl" }] },
        { label: "fish", children: [{ label: "trout" }] },
      ],
    },
    {
      label: "vegetable",
      children: [{ label: "fruit", children: [{ label: "kiwi" }, { label: "apple" }, { label: "pear" }] }],
    },
  ],
};

const levels = new Map([
  ["", { animal: 0.7, vegetable: 0.3 }],
  ["animal", { bird: 0.9, fish: 0.1 }],
  ["animal › bird", { kiwi: 0.2, owl: 0.8 }],
  ["vegetable", { fruit: 1 }],
  ["vegetable › fruit", { kiwi: 0.5, apple: 0.4, pear: 0.1 }],
]);
// Path chances below are worked out by hand from these shares as they are, untempered.
const plain = { temperature: 1 };

test("a name in two places has two paths, and their levels are each classified once", () => {
  assert.deepEqual(pathsTo(tree, "kiwi"), [["animal", "bird", "kiwi"], ["vegetable", "fruit", "kiwi"]]);
  assert.deepEqual(levelsFor(tree, pathsTo(tree, "kiwi")).map((l) => l.nodeKey), ["", "animal", "animal › bird", "vegetable", "vegetable › fruit"]);
  assert.equal(allLevels(tree).length, 6);
  assert.deepEqual(thingNames(tree), ["apple", "kiwi", "owl", "pear", "trout"]);
});

test("each step on the way gets its place, its lead or how far behind, and a status", () => {
  const result = trace(tree, "kiwi", levels, { ruledOut: new Set(["owl"]) });
  const animal = result.root.options.find((o) => o.label === "animal");
  assert.equal(animal.rank, 1);
  assert.ok(Math.abs(animal.lead - 0.4) < 1e-9);
  assert.equal(animal.status, "ok");
  const bird = animal.child.options.find((o) => o.label === "bird").child.options;
  assert.equal(bird[0].label, "owl");
  assert.equal(bird[0].ruledOut, true);
  const kiwi = bird.find((o) => o.label === "kiwi");
  assert.equal(kiwi.isTarget, true);
  assert.equal(kiwi.rank, 2);
  assert.ok(Math.abs(kiwi.behind - 4) < 1e-9);
  assert.equal(kiwi.status, "lost");
  const fruit = result.root.options.find((o) => o.label === "vegetable").child.options[0].child.options[0];
  assert.equal(fruit.label, "kiwi");
  assert.equal(fruit.status, "close"); // first, but only 0.1 ahead
});

test("each path gets its chance, weakest step and typical step, and the verdict follows the likelier path", () => {
  const result = trace(tree, "kiwi", levels, { tuning: plain });
  const [viaBird, viaFruit] = result.paths;
  assert.ok(Math.abs(viaBird.pathP - 0.7 * 0.9 * 0.2) < 1e-9);
  assert.ok(Math.abs(viaFruit.pathP - 0.3 * 1 * 0.5) < 1e-9);
  assert.equal(viaBird.weakest.option.label, "kiwi");
  assert.ok(Math.abs(viaBird.geoMean - Math.cbrt(0.7 * 0.9 * 0.2)) < 1e-9);
  assert.equal(viaBird.lostAt.step.question, "Which bird is it?");
  assert.equal(result.best, viaFruit);
  assert.match(result.verdict, /^Lost at “Is it an animal or a vegetable\?” \(by way of vegetable › fruit\): animal \(0\.70\) was 2\.3× more likely than vegetable \(0\.30\)\.$/);
});

test("unknown names and unscored steps say so", () => {
  assert.match(trace(tree, "zebra", levels).verdict, /not one of the things/);
  const partial = trace(tree, "trout", new Map([["", { animal: 1, vegetable: 0 }]]));
  assert.equal(partial.paths[0].complete, false);
  assert.match(partial.verdict, /Not every step/);
});

test("sureness runs from 0 for an even split to 1 for a certain answer", () => {
  assert.ok(Math.abs(sureness([0.25, 0.25, 0.25, 0.25])) < 1e-9);
  assert.ok(Math.abs(sureness([1, 0, 0]) - 1) < 1e-9);
});

test("the overall ranking multiplies every step and leaves out unscored branches", () => {
  const ranked = leafRanking(tree, levels, plain);
  assert.deepEqual(ranked.map((r) => r.label), ["owl", "kiwi", "kiwi", "apple", "pear"]);
  assert.ok(Math.abs(ranked[0].p - 0.7 * 0.9 * 0.8) < 1e-9);
});

test("a probability range keeps every option on the way and counts the rest", () => {
  const result = trace(tree, "kiwi", levels);
  const fruit = result.root.options.find((o) => o.label === "vegetable").child.options[0].child;
  const { shown, above, below } = inRange(fruit.options, [0.45, 1]);
  assert.deepEqual(shown.map((o) => o.label), ["kiwi"]);
  assert.equal(above, 0);
  assert.equal(below, 2);
});

test("the game records what each classification returned, and for which clues, for the trace to reuse", () => {
  const game = newGame(tree, "small and brown");
  const step = nextStep(game);
  const after = { ...afterClassification(game, step, { animal: 0.6, vegetable: 0.4 }), clues: "small and brown, with feathers" };
  assert.deepEqual(after.levels.get(""), { clues: "small and brown", probabilities: { animal: 0.6, vegetable: 0.4 } });
  assert.equal(game.levels.size, 0);
});

test("the likeliest so far use only classifications for the current clues, without wrong guesses, each name once", () => {
  const game = {
    ...newGame(tree, "small and brown"),
    levels: new Map([...levels].map(([nodeKey, probabilities]) => [nodeKey, { clues: "small and brown", probabilities }])),
    ruledOut: new Set(["owl"]),
  };
  const top = likeliest(game, 3, plain);
  assert.deepEqual(top.map((t) => t.label), ["kiwi", "apple", "pear"]);
  assert.equal(top[0].nodeKey, "vegetable › fruit › kiwi"); // 0.15 by way of fruit beats 0.126 by way of bird
  game.levels.set("vegetable", { clues: "older clues", probabilities: { fruit: 1 } });
  assert.deepEqual(likeliest(game, 3, plain).map((t) => t.label), ["kiwi"]);
});
