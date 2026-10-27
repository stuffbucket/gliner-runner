import assert from "node:assert/strict";
import { test } from "node:test";

import { check, grade, sentences, syllables, words } from "./constraints.mjs";
import { bradleyTerry } from "./judge.mjs";

test("words keep contractions and hyphens together", () => {
  assert.deepEqual(words("It doesn't write; it picks at-a-glance."), ["It", "doesn't", "write", "it", "picks", "at-a-glance"]);
});

test("sentences split after full stops, question and exclamation marks", () => {
  assert.deepEqual(sentences("You glance. You pick! Done? Yes"), ["You glance.", "You pick!", "Done?", "Yes"]);
});

test("syllables are right for common words", () => {
  const cases = { the: 1, mail: 1, picks: 1, phone: 1, table: 2, message: 2, junk: 1, answer: 2, decide: 2, glance: 1, every: 3, quickly: 2, wanted: 2, computer: 3, probability: 5 };
  for (const [word, n] of Object.entries(cases)) assert.equal(syllables(word), n, word);
});

test("grade is low for short plain sentences and high for long technical ones", () => {
  assert.ok(grade("You toss the junk mail. You keep the bills.") < 2);
  assert.ok(grade("The classifier computes a calibrated probability distribution over every candidate label simultaneously.") > 12);
});

test("check records each reason a text fails", () => {
  const limits = { maxWords: 10, maxSentenceWords: 5, maxGrade: 5, stallWords: ["label", "model"], meta: ["this page"] };
  const ok = check("It picks one. You choose the options.", limits);
  assert.deepEqual(ok.reasons, []);
  const bad = check("This page says the model reads your labels and gives each a probability.", limits);
  assert.equal(bad.reasons.length, 5);
  assert.match(bad.reasons.join(" | "), /13 words.*sentence of 13.*grade.*stall word: label, model.*meta: this page/);
});

test("Bradley-Terry ranks a player who wins more above one who wins less", () => {
  const games = [
    { a: "x", b: "y", score: 1 }, { a: "x", b: "y", score: 1 }, { a: "y", b: "x", score: 0 },
    { a: "y", b: "z", score: 1 }, { a: "x", b: "z", score: 1 }, { a: "z", b: "y", score: 0.5 },
  ];
  const strength = bradleyTerry(["x", "y", "z"], games);
  assert.ok(strength.x > strength.y && strength.y > strength.z);
});
