// Tests for the 20 questions game (src/components/twentyQuestions.js) on a small hand-made tree, with a made-up
// classifier in place of the model.
import assert from "node:assert/strict";
import { test } from "node:test";

import { MAX_QUESTIONS, afterClassification, afterReply, newGame, nextStep, startGuess } from "../../src/components/twentyQuestions.js";
import { leafRanking } from "../../src/components/twentyQuestionsTrace.js";

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

// What the "model" says at each level, for each set of clues.
const answers = {
  "brown and fuzzy": new Map([
    ["", { animal: 0.3, vegetable: 0.7 }],
    ["animal", { bird: 0.9, fish: 0.1 }],
    ["animal › bird", { kiwi: 0.6, owl: 0.4 }],
    ["animal › fish", { trout: 1 }],
    ["vegetable", { fruit: 1 }],
    ["vegetable › fruit", { kiwi: 0.8, apple: 0.15, pear: 0.05 }],
  ]),
  "brown and fuzzy. It hoots at night.": new Map([
    ["", { animal: 0.9, vegetable: 0.1 }],
    ["animal", { bird: 0.95, fish: 0.05 }],
    ["animal › bird", { kiwi: 0.1, owl: 0.9 }],
    ["animal › fish", { trout: 1 }],
    ["vegetable", { fruit: 1 }],
    ["vegetable › fruit", { kiwi: 0.5, apple: 0.3, pear: 0.2 }],
  ]),
};
const classify = (step) => answers[step.clues].get(step.nodeKey);
// The guesses below are worked out by hand from these shares as they are, untempered.
const plain = { temperature: 1 };

/** Runs the game until it has a guess (or nothing left); returns the game, that guess and the classifications made. */
function untilGuess(game) {
  const classified = [];
  for (;;) {
    const step = nextStep(game, plain);
    if (step?.kind !== "guess") {
      if (step === undefined) return { game, step, classified };
      classified.push(step);
      game = afterClassification(game, step, classify(step));
      continue;
    }
    return { game, step, classified };
  }
}

/** Says no to `count` guesses in a row; returns the game and the names guessed. */
function sayNo(game, count) {
  const guessed = [];
  for (let i = 0; i < count && game.phase === "asking"; i += 1) {
    const { game: ready, step } = untilGuess(game);
    if (step === undefined) break;
    guessed.push(step.label);
    game = afterReply(startGuess(ready, step), false);
  }
  return { game, guessed };
}

test("classifying costs no question; each guess costs one", () => {
  const { game, step, classified } = untilGuess(newGame(tree, "brown and fuzzy"));
  assert.ok(classified.length > 0);
  assert.equal(game.asked, 0);
  assert.equal(step.number, 1);
  assert.equal(startGuess(game, step).asked, 1);
});

test("the guess is the likeliest thing of all, without classifying every level", () => {
  const { game, step, classified } = untilGuess(newGame(tree, "brown and fuzzy"));
  const all = leafRanking(tree, answers["brown and fuzzy"], plain);
  assert.equal(step.label, all[0].label);
  assert.equal(step.nodeKey, all[0].nodeKey);
  assert.ok(Math.abs(step.p - all[0].p) < 1e-9);
  assert.ok(classified.length < answers["brown and fuzzy"].size, "the fish level is never needed");
  assert.equal(game.levels.has("animal › fish"), false);
});

test("a wrong guess rules out the name everywhere: a kiwi is guessed once", () => {
  const { guessed } = sayNo(newGame(tree, "brown and fuzzy"), 10);
  assert.deepEqual(guessed, ["kiwi", "owl", "apple", "pear", "trout"]);
});

test("a new clue asks every level again, from the root, before the next guess", () => {
  const { game } = sayNo(newGame(tree, "brown and fuzzy"), 1);
  const later = { ...game, clues: "brown and fuzzy. It hoots at night." };
  const step = nextStep(later);
  assert.equal(step.kind, "classify");
  assert.equal(step.nodeKey, "");
  assert.equal(step.again, true);
  assert.match(step.text, /^With your new clue · /);
  assert.equal(untilGuess(later).step.label, "owl");
});

test("what it guesses depends on the clues, not the order they came in", () => {
  // A clue added after a wrong guess…
  const { game: first } = sayNo(newGame(tree, "brown and fuzzy"), 1);
  const late = sayNo({ ...first, clues: "brown and fuzzy. It hoots at night." }, 10);
  // …and the same clues from the start, with the same wrong guess, guess the same things in the same order.
  const fromStart = newGame(tree, "brown and fuzzy. It hoots at night.");
  const early = sayNo({ ...fromStart, ruledOut: new Set(first.ruledOut), asked: first.asked }, 10);
  assert.deepEqual(late.guessed, early.guessed);
  assert.deepEqual(late.guessed, ["owl", "trout", "apple", "pear"]);
});

test("a classification made while the clues changed is out of date at once", () => {
  const game = newGame(tree, "brown and fuzzy");
  const step = nextStep(game);
  const after = { ...afterClassification(game, step, classify(step)), clues: "brown and fuzzy. It hoots at night." };
  assert.deepEqual(after.levels.get(""), { clues: "brown and fuzzy", probabilities: classify(step) });
  assert.equal(nextStep(after).again, true);
});

test(`${MAX_QUESTIONS} wrong guesses lose, and so does running out of things`, () => {
  const many = {
    label: "anything",
    children: [{ label: "letters", children: Array.from({ length: 26 }, (_, i) => ({ label: String.fromCharCode(97 + i) })) }],
  };
  const even = Object.fromEntries(many.children[0].children.map((c, i) => [c.label, 26 - i]));
  let game = newGame(many, "a letter");
  game = afterClassification(game, nextStep(game), { letters: 1 });
  game = afterClassification(game, nextStep(game), even);
  for (let i = 0; i < MAX_QUESTIONS; i += 1) game = afterReply(startGuess(game, nextStep(game)), false);
  assert.equal(game.asked, MAX_QUESTIONS);
  assert.equal(game.phase, "lost");

  const { game: out } = sayNo(newGame(tree, "brown and fuzzy"), 10);
  assert.equal(out.phase, "lost");
  assert.equal(out.asked, 5);
});
