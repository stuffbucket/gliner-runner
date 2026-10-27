// Scores the "be the model" examples with the real model, so the page can show actual answers without running one.
// Run after changing the candidates, the labels or a model pin; the output is committed:
//
//   GLINER_RUNNER_MODELS_DIR=dist/onnx-models node site/scripts/be-the-model.mjs [--base http://127.0.0.1:PORT]
//
// It starts the built runner (packages/onnx-runner/dist), or uses the one at --base, and classifies every candidate in
// site/scripts/be-the-model/candidates.json against LABELS twice: single label (the scores sum to 1) and several
// labels (each label scored on its own). It then picks EXAMPLES examples for nearest-neighbour lookup: for each label,
// STOPS candidates drafted for that label, spread evenly from its lowest to its highest score on that label, one more
// for any 0.1 band still empty, then farthest-point sampling over the single-label scores for the rest; near ties go
// to the ask and the family (who the line names, see family()) the examples have least of. It writes
// site/src/data/be-the-model.json and prints how well the model tells the labels apart, how many examples fall in each
// 0.1 band of each label, and who the examples name.
//
// The candidates were drafted with a chat model (claude-sonnet-5) and checked by hand: near-duplicates removed, and
// any line that calls a grown-up by a role ("Stepdad, can I ...", which no kid says) or carries a stage direction
// dropped. The `draft` field says which prompt and approach each came from. The prompts, with APPROACH, N, STRENGTH and
// ASKS filled in, are below. The first drafts named mostly moms, dads, grandmas and aunts, so a second round (the
// candidates with a `who` field) set the family for each line, dealt from a weighted deck: a stepdad, a stepmom, a
// foster parent, two moms, two dads, one parent, a grandparent, an uncle or aunt, an older sibling, a babysitter, or
// no one named. That round's prompts add after RULES "Who, for each one in order:" and one line per text, and replace
// the family sentences of RULES with:
//
//     Who the kid is talking to, or who comes up, is set for each one below. The words the kid says must make that
//     plain to a stranger reading only this line, the way a real kid in that family would say it: a stepparent,
//     foster parent or babysitter comes up by role in passing ("my stepdad said", "can you ask my foster mom", "Sam,
//     my babysitter, said"), or the kid talks to one parent about the other. Kids call a grown-up what they really
//     call them (Mom, Mama, Papa, Grandpa, Uncle Dev, a first name), never by a role such as "Stepdad," "Foster mom,"
//     or "Babysitter,"; a role can come up in passing ("my stepdad said"). Keep the family ordinary: it is never the
//     point of the line, the joke or the source of the guilt. No jokes that depend on gender stereotypes.
//
// Of that round, lines that did not show their family in the words (most of the ones that used only a first name)
// were dropped, and by hand those that made the family the point or the grievance ("my stepdad doesn't care", "it's
// just us so money's tight"), echoed a wicked-stepmother trope, or read awkwardly.
//
//   RULES:
//     Each one: 1 to 3 sentences, plain words, how a real kid of that age actually talks. No made-up flourishes.
//     The kid is talking to a parent or the grown-up who looks after them. Vary who that is and who else gets named,
//     the way real families vary: mom, dad, a parent, grandma, an aunt, a stepparent, a foster parent, a babysitter;
//     some kids have two moms, two dads or one parent. Keep it natural, not a checklist. Kids call a grown-up what
//     they really call them (Mom, Dad, Mama, Papa, Grandma, Aunt Rosa, a first name), never by a role such as
//     "Stepdad," or "Babysitter,"; a role can come up in passing ("my stepdad said"). No jokes that depend on gender
//     stereotypes.
//     Only the words the kid says out loud: no stage directions, actions, sounds or narration in brackets,
//     asterisks or after dashes.
//
//   One approach at one strength (draft "<label>|high", "|middle" or "|low"; N is 6, 5 or 5), and mixes of two or
//   three approaches (draft "<label>+<label>", N is 3, APPROACH lists them and STRENGTH says all are clearly there):
//     Write N different short things a kid says to ask for something.
//     Approach: APPROACH
//     STRENGTH (high: The approach is the whole message, strong and unmistakable. / middle: The approach is clearly
//     there but mild, mixed into an ordinary ask. / low: Mostly a plain ask; the approach is only a small hint.)
//     RULES
//     Spread the ages from 5 to 16 and use different asks from this list: ASKS.
//     Return only a JSON array of objects: {"text": "...", "ask": "<one item from the list>", "age": <5-16>}.
//
//   A ladder (draft "<label>|ladder1" to "|ladder12"):
//     Write 12 different short things a kid says to ask for something, all using one approach, from faintest to
//     strongest.
//     Approach: APPROACH
//     Number 1 is a plain ask with only the faintest trace of the approach. Number 12 is nothing but the approach,
//     as strong and over the top as a real kid would get. The ones between rise step by step.
//     RULES
//     Spread the ages from 5 to 16 and use these asks: ASKS.
//     Return only a JSON array of 12 objects in order: {"text": "...", "ask": "<one item from the list>", "age": <5-16>}.
//
//   Pure (draft "<label>|pure"), for the labels whose top end was thin:
//     Write 10 different short things a kid says to ask for something, using nothing but one approach, as pure and
//     strong as a real kid would get.
//     Approach: APPROACH
//     Mix in no other approach: no reasons or plans, no deals, no compliments, no guilt, no begging, and no mention
//     of other kids or other grown-ups, unless the approach itself is about them.
//     RULES
//     Spread the ages from 5 to 16 and use these asks: ASKS.
//     Return only a JSON array of objects: {"text": "...", "ask": "<one item from the list>", "age": <5-16>}.
//
//   APPROACH, per label:
//     begging: begging or whining: please, please, pretty please, I really really want it
//     bargaining: bargaining: offering a deal, a chore or giving something up in return
//     flattery: flattery: complimenting the grown-up to get a yes
//     guilt trip: guilt trip: saying the grown-up does not care, never lets them do anything, or is being unfair to them
//     everyone else has one: everyone else has one: pointing to friends, classmates or siblings who already have it
//       or are allowed
//     reasoned case: a reasoned case: explaining why it is fine (homework is done, saved the money, knows the rules,
//       has a plan)
//     polite ask: a sincere, polite ask: a short, respectful request with no pressure
//     asking someone else: asking someone else: saying another grown-up (another parent, a grandparent, a
//       stepparent, a babysitter, an older sibling) already said yes, or going to ask them instead
//
//   ASKS: later bedtime, a dog, a phone, candy at the checkout, a sleepover, skipping a chore, more screen time,
//     a video game, staying home alone, going to a party, a later curfew, new sneakers, ice cream, a hamster,
//     getting ears pierced, walking to school alone, dyeing hair, a trampoline

import { readFileSync, writeFileSync } from "node:fs";
import { dirname, join, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import { parseArgs } from "node:util";

import { classify, startRunner } from "./runner.mjs";

const here = dirname(fileURLToPath(import.meta.url));
const CANDIDATES = join(here, "be-the-model/candidates.json");
const OUTPUT = join(here, "../src/data/be-the-model.json");
const MODEL = "decide-340m";
const EXAMPLES = 100;
const STOPS = 12;
// How much a crowded family counts against a candidate, per example that already names it: in score when picking a
// stop, and in distance when filling.
const FAMILY_TIE = 0.004;
const FAMILY_GAP = 0.005;

/** The labels and their descriptions, in the order the page shows them. Chosen by measuring confusion on clear examples. */
export const LABELS = {
  begging: "Pleads and whines, saying please over and over",
  bargaining: "Offers a deal: I'll clean my room, do the dishes or give up dessert if you say yes",
  flattery: "Compliments the grown-up, calling them the best, to get a yes",
  "guilt trip": "Says the grown-up does not care, never lets them do anything, or is unfair to them",
  "everyone else has one": "Points to friends or siblings who already have it or are allowed",
  "reasoned case": "Explains why it is fine: homework is done, they saved the money, they know the rules, they have a plan",
  "polite ask": "A short polite request, nothing more",
  "asking someone else": "Says another grown-up already said yes, or goes to ask them instead",
};
const NAMES = Object.keys(LABELS);

const round = (p) => Math.round(p * 1000) / 1000;
const roundAll = (scores) => Object.fromEntries(NAMES.map((name) => [name, round(scores[name])]));
const draftedFor = (candidate) => candidate.draft.split("|")[0].split("+");
const top = (scores) => NAMES.reduce((best, name) => (scores[name] > scores[best] ? name : best));
const distance = (a, b) => Math.hypot(...NAMES.map((name) => a.scores[name] - b.scores[name]));

// Who a line names, by role, to keep a mix of families in the examples. A line can name several, or no one.
// Mom or my stepdad, but not Mia's mom, his stepdad or Pop-pop.
const ours = (words) => new RegExp(String.raw`(?<!\b(?:\w+'s|your|his|her|their|other|friends'|the)\s)(?<![\w-])(${words})(?![\w-])`, "i");
const MOM = ours("mom|mommy|mum|mummy|mother|mama|ma");
const DAD = ours("dad|daddy|father|papa|pop|pops|baba|pa");
const ROLES = {
  stepparent: ours("step-?(?:mom|mum|dad|mother|father|parent|mommy|daddy)s?"),
  "foster parent": ours("foster"),
  "two moms": (t) => /\b(my other mom|my moms|both my moms)\b/i.test(t) || (ours("mom|mommy|mum|mother").test(t) && ours("mama|ma").test(t)),
  "two dads": (t) => /\b(my other dad|my dads|both my dads)\b/i.test(t) || (ours("dad|daddy|father").test(t) && ours("papa|pop|pops|baba|pa").test(t)),
  "one parent": /\b(just (you and me|me and you|the two of us|us two)|it'?s just us)\b/i,
  grandparent: /\b(grandma|grandpa|nana|nan|granny|gran|grandad|granddad|grandpop|pop-pop|abuela|abuelo|nonna|nonno|grandmother|grandfather|grandparents?)\b/i,
  "aunt or uncle": /\b(aunt|auntie|aunty|uncle|t[ií]a|t[ií]o)\b/i,
  "older sibling": /\b(big|older) (sister|brother|sis|bro)\b/i,
  babysitter: /\b(babysitter|sitter|nanny)\b/i,
  mom: MOM,
  dad: DAD,
};
export const FAMILY = [...Object.keys(ROLES), "no one"];

/** The roles a line names: "my foster mom" or Mom and Mama count once, as foster parent or two moms, not also as mom. */
export function family(text) {
  const found = Object.keys(ROLES).filter((role) => (typeof ROLES[role] === "function" ? ROLES[role](text) : ROLES[role].test(text)));
  const has = (role) => found.includes(role);
  const plain = (role, other) => !["foster parent", "one parent", other].some(has) || has(role === "mom" ? "two dads" : "two moms");
  const roles = found.filter((role) => (role === "mom" ? plain("mom", "two moms") : role === "dad" ? plain("dad", "two dads") : true));
  return roles.length > 0 ? roles : ["no one"];
}

/**
 * For each label, STOPS candidates drafted for it, nearest to evenly spaced targets between that label's lowest and
 * highest score, one more for any 0.1 band some candidate reaches but no example covers, then farthest-point
 * sampling over all the scores up to `total`. Near ties go to the ask, and the family, the examples have least of.
 */
export function select(candidates, total = EXAMPLES, stops = STOPS) {
  const chosen = [];
  const asks = new Map();
  const roles = new Map();
  const take = (candidate) => {
    if (chosen.includes(candidate)) return;
    chosen.push(candidate);
    asks.set(candidate.ask, (asks.get(candidate.ask) ?? 0) + 1);
    for (const role of family(candidate.text)) roles.set(role, (roles.get(role) ?? 0) + 1);
  };
  // How many examples already name the most common of this candidate's roles.
  const crowd = (candidate) => Math.max(...family(candidate.text).map((role) => roles.get(role) ?? 0));
  for (const name of NAMES) {
    const own = candidates.filter((c) => draftedFor(c).includes(name));
    const values = own.map((c) => c.scores[name]);
    const [low, high] = [Math.min(...values), Math.max(...values)];
    const picked = [];
    for (let i = 0; i < stops; i += 1) {
      const target = low + ((high - low) * i) / (stops - 1);
      // Nearest to the target; among near ties, the ask and the family the examples have least of.
      const cost = (c) => Math.abs(c.scores[name] - target) + 0.005 * (asks.get(c.ask) ?? 0) + FAMILY_TIE * crowd(c);
      const best = own.filter((c) => !picked.includes(c)).sort((a, b) => cost(a) - cost(b))[0];
      picked.push(best);
      take(best);
    }
  }
  // Fill any 0.1 band a label's score reaches but no example covers yet.
  for (const name of NAMES) {
    for (const [band, count] of bands(chosen)[name].entries()) {
      if (count > 0) continue;
      const inBand = candidates.filter((c) => Math.min(9, Math.floor(c.scores[name] * 10)) === band);
      const best = inBand.find((c) => draftedFor(c).includes(name)) ?? inBand[0];
      if (best) take(best);
    }
  }
  while (chosen.length < total) {
    let best;
    let bestGap = -Infinity;
    for (const c of candidates) {
      if (chosen.includes(c)) continue;
      const gap = Math.min(...chosen.map((d) => distance(c, d))) - FAMILY_GAP * crowd(c);
      if (gap > bestGap) [best, bestGap] = [c, gap];
    }
    take(best);
  }
  return chosen;
}

/** Counts per 0.1 band of each label's own score, for the coverage report and the test. */
export function bands(examples) {
  return Object.fromEntries(
    NAMES.map((name) => {
      const counts = Array(10).fill(0);
      for (const e of examples) counts[Math.min(9, Math.floor(e.scores[name] * 10))] += 1;
      return [name, counts];
    }),
  );
}

function report(candidates, examples) {
  const clear = candidates.filter((c) => /\|(high|pure|ladder1[012])$/.test(c.draft) && !c.draft.includes("+"));
  const confusion = Object.fromEntries(NAMES.map((name) => [name, Object.fromEntries(NAMES.map((n) => [n, 0]))]));
  for (const c of clear) confusion[draftedFor(c)[0]][top(c.scores)] += 1;
  const right = NAMES.reduce((sum, name) => sum + confusion[name][name], 0);
  const short = (name) => name.split(" ").map((w) => w[0].toUpperCase()).join("").padStart(4);
  console.log(`\nconfusion on ${clear.length} clear single-approach candidates (row: drafted for, column: top label): ${right} right`);
  console.log(" ".repeat(22) + NAMES.map(short).join(""));
  for (const name of NAMES) console.log(name.padEnd(22) + NAMES.map((n) => String(confusion[name][n]).padStart(4)).join(""));
  console.log(`\n${examples.length} examples per 0.1 band of each label's score, and distinct values among those drafted for it`);
  const counts = bands(examples);
  for (const name of NAMES) {
    const own = new Set(examples.filter((e) => draftedFor(e).includes(name)).map((e) => round(e.scores[name])));
    console.log(name.padEnd(22) + counts[name].map((n) => String(n).padStart(4)).join("") + `   ${own.size} own`);
  }
  const named = examples.filter((e) => !family(e.text).includes("no one")).length;
  const who = FAMILY.map((role) => `${role} ${examples.filter((e) => family(e.text).includes(role)).length}`);
  console.log(`\nwho the examples name (${named} name someone; a line can name several): ${who.join(", ")}`);
}

async function main() {
  const { values } = parseArgs({ options: { base: { type: "string" }, concurrency: { type: "string", default: "4" } } });
  const started = Date.now();
  const runner = values.base ? { base: values.base, stop() {} } : await startRunner();
  try {
    const candidates = JSON.parse(readFileSync(CANDIDATES, "utf8"));
    let next = 0;
    const worker = async () => {
      while (next < candidates.length) {
        const c = candidates[next++];
        const request = { model: MODEL, text: c.text, labels: NAMES, descriptions: LABELS };
        c.scores = (await classify(runner.base, request)).probabilities;
        c.multi = (await classify(runner.base, { ...request, several: true })).probabilities;
      }
    };
    await Promise.all(Array.from({ length: Number(values.concurrency) }, worker));
    const examples = select(candidates).sort(
      (a, b) => NAMES.indexOf(top(a.scores)) - NAMES.indexOf(top(b.scores)) || b.scores[top(b.scores)] - a.scores[top(a.scores)],
    );
    report(candidates, examples);
    const data = {
      about:
        `Things a kid might say to get a yes, each scored by ${MODEL} against ${NAMES.length} labels. "scores" is single-label ` +
        `mode: the labels compete and the scores sum to 1. "multi" is several-label mode: each label is scored on its own. ` +
        `Generated by site/scripts/be-the-model.mjs from site/scripts/be-the-model/candidates.json.`,
      model: MODEL,
      mode: "single",
      labels: NAMES.map((name) => ({ name, description: LABELS[name] })),
      examples: examples.map((e, i) => ({
        id: `k${String(i + 1).padStart(3, "0")}`,
        text: e.text,
        ask: e.ask,
        age: e.age,
        scores: roundAll(e.scores),
        multi: roundAll(e.multi),
      })),
    };
    writeFileSync(OUTPUT, `${JSON.stringify(data, null, 2)}\n`);
    console.log(`\nscored ${candidates.length} candidates and wrote ${examples.length} examples to ${OUTPUT} in ${((Date.now() - started) / 1000).toFixed(0)} s`);
  } finally {
    runner.stop();
  }
}

if (process.argv[1] && resolve(process.argv[1]) === fileURLToPath(import.meta.url)) {
  await main();
}
