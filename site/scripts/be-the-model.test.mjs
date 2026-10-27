// Tests for "Be the model" (src/components/beTheModel.js): which example a slider shows, stepping with the keys,
// and the data files.
import assert from "node:assert/strict";
import { existsSync, readFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { test } from "node:test";
import { fileURLToPath } from "node:url";

import {
  TIE,
  alongLabel,
  checkData,
  chooseData,
  inferRequest,
  keySteps,
  nearestAlong,
  nearestExample,
  scoresFromReply,
  stepAlong,
  stepExample,
  stopsPerLabel,
  topLabel,
} from "../src/components/beTheModel.js";
import { MOCK, REAL, beTheModelData, beTheModelPlugin, pageData } from "./be-the-model-data.mjs";
import { beTheModelText } from "./markdown-copies.mjs";

const site = join(dirname(fileURLToPath(import.meta.url)), "..");
const readData = (path) => JSON.parse(readFileSync(join(site, path), "utf8"));
const hasReal = existsSync(join(site, REAL));

// Three labels, scores adding up to 1.
const examples = [
  { id: "a", scores: { beg: 0.8, deal: 0.1, nice: 0.1 } },
  { id: "b", scores: { beg: 0.5, deal: 0.4, nice: 0.1 } },
  { id: "c", scores: { beg: 0.5, deal: 0.1, nice: 0.4 } },
  { id: "d", scores: { beg: 0.2, deal: 0.7, nice: 0.1 } },
  { id: "e", scores: { beg: 0.21, deal: 0.09, nice: 0.7 } },
  { id: "f", scores: { beg: 0.05, deal: 0.05, nice: 0.9 } },
];
const byId = (id) => examples.find((example) => example.id === id);

test("the slider shows the example whose score for that label is nearest", () => {
  assert.equal(nearestExample(examples, "beg", 0.79, byId("a").scores).id, "a");
  assert.equal(nearestExample(examples, "beg", 0.0, byId("a").scores).id, "f");
  assert.equal(nearestExample(examples, "beg", 1, byId("f").scores).id, "a");
  assert.equal(nearestExample(examples, "deal", 0.66, byId("a").scores).id, "d");
});

test("a tie goes to the example nearest the scores on screen", () => {
  // b and c both score 0.5 for beg. From d (deal 0.7) the text moves to b, which also leans on deal; from f
  // (nice 0.9) it moves to c, which also leans on nice.
  assert.equal(nearestExample(examples, "beg", 0.5, byId("d").scores).id, "b");
  assert.equal(nearestExample(examples, "beg", 0.5, byId("f").scores).id, "c");
});

test("scores within the tie distance count as a tie, so a small nudge keeps the text on screen", () => {
  // d (0.2) is nearest 0.2, but e (0.21) is within TIE and is the one on screen, so it stays.
  assert.ok(0.21 - 0.2 <= TIE);
  assert.equal(nearestExample(examples, "beg", 0.2, byId("e").scores).id, "e");
  assert.equal(nearestExample(examples, "beg", 0.2, byId("d").scores).id, "d");
  // With no tie distance, the nearest score wins outright.
  assert.equal(nearestExample(examples, "beg", 0.2, byId("e").scores, 0).id, "d");
  // Beyond the tie distance the nearer score wins, whatever is on screen.
  assert.equal(nearestExample(examples, "beg", 0.05, byId("e").scores).id, "f");
});

test("a full tie falls back to the id, so the list's order does not matter", () => {
  const twins = [
    { id: "y", scores: { beg: 0.5, deal: 0.5 } },
    { id: "x", scores: { beg: 0.5, deal: 0.5 } },
  ];
  assert.equal(nearestExample(twins, "beg", 0.5, { beg: 0, deal: 0 }).id, "x");
  assert.equal(nearestExample([...twins].reverse(), "beg", 0.5, { beg: 0, deal: 0 }).id, "x");
});

test("the arrow keys step to the next or previous example along a label's scores", () => {
  assert.deepEqual(
    alongLabel(examples, "beg").map((example) => example.id),
    ["f", "d", "e", "b", "c", "a"],
  );
  assert.equal(stepExample(examples, "beg", "e", 1).id, "b");
  assert.equal(stepExample(examples, "beg", "e", -1).id, "d");
  assert.equal(stepExample(examples, "beg", "a", 1).id, "a", "stops at the top");
  assert.equal(stepExample(examples, "beg", "f", -1).id, "f", "stops at the bottom");
  assert.equal(stepExample(examples, "beg", "d", keySteps("End")).id, "a");
  assert.equal(stepExample(examples, "beg", "d", keySteps("Home")).id, "f");
  assert.equal(stepExample(examples, "beg", "f", keySteps("PageUp")).id, "a");
  assert.throws(() => stepExample(examples, "beg", "zz", 1), /no example zz/);
});

test("keys: right and up go higher, left and down lower; other keys are left alone", () => {
  assert.equal(keySteps("ArrowRight"), 1);
  assert.equal(keySteps("ArrowUp"), 1);
  assert.equal(keySteps("ArrowLeft"), -1);
  assert.equal(keySteps("ArrowDown"), -1);
  assert.equal(keySteps("Tab"), undefined);
});

test("the binary search finds the same example as a look at every example", () => {
  // Made-up data with many near ties: scores on a coarse grid.
  let seed = 7;
  const random = () => ((seed = (seed * 16807) % 2147483647) / 2147483647);
  const names = ["p", "q", "r", "s"];
  const many = Array.from({ length: 120 }, (_, i) => {
    const raw = names.map(() => Math.round(random() * 20) + 1);
    const total = raw.reduce((a, b) => a + b, 0);
    return { id: `x${String(i).padStart(3, "0")}`, scores: Object.fromEntries(names.map((n, j) => [n, raw[j] / total])) };
  });
  const everyOne = (label, value, current, tie) => {
    const gap = (e) => Math.abs(e.scores[label] - value);
    const best = Math.min(...many.map(gap));
    const others = (e) => names.filter((n) => n !== label).reduce((t, n) => t + (e.scores[n] - current[n]) ** 2, 0);
    return many
      .filter((e) => gap(e) <= best + tie + 1e-12)
      .sort((a, b) => others(a) - others(b) || (a.id < b.id ? -1 : 1))[0];
  };
  for (const label of names) {
    const order = alongLabel(many, label);
    for (let k = 0; k <= 200; k++) {
      const value = k / 200;
      const current = many[k % many.length].scores;
      for (const tie of [0, TIE, 0.05]) {
        assert.equal(nearestAlong(order, label, value, current, tie).id, everyOne(label, value, current, tie).id);
      }
    }
  }
});

test("stepping with a ready-made order matches stepping with the list", () => {
  const order = alongLabel(examples, "deal");
  for (const { id } of examples) {
    for (const steps of [-10, -1, 1, 10]) assert.equal(stepAlong(order, id, steps).id, stepExample(examples, "deal", id, steps).id);
  }
});

test("an edited line is sent with the data's labels and descriptions, picking one label", () => {
  const data = readData(MOCK);
  const request = inferRequest(data, "Can I please stay up?");
  const task = request.schema.tasks.category;
  assert.equal(request.model, data.model);
  assert.equal(request.text, "Can I please stay up?");
  assert.deepEqual(task.labels, data.labels.map((label) => label.name));
  assert.equal(task.label_descriptions[data.labels[0].name], data.labels[0].description);
  assert.equal(task.min_labels, undefined, "one label is the default");
});

test("the scores come out of the model's reply, or a plain message when they cannot", () => {
  const data = { labels: [{ name: "beg" }, { name: "deal" }] };
  const reply = (probabilities) => JSON.stringify({ output: { category: { value: "beg", probabilities } } });
  assert.deepEqual(scoresFromReply(data, 200, reply({ beg: 0.75, deal: 0.25 })), { scores: { beg: 0.75, deal: 0.25 } });
  assert.match(scoresFromReply(data, 200, reply({ beg: 1 })).error, /no score for "deal"/);
  assert.match(scoresFromReply(data, 422, JSON.stringify({ detail: "text is empty" })).error, /text is empty/);
  assert.match(scoresFromReply(data, 200, "not json").error, /could not be read/);
});

test("the model's answer is the label with the highest score", () => {
  assert.equal(topLabel(byId("d").scores), "deal");
  assert.equal(topLabel(byId("f").scores), "nice");
});

test("checkData catches data the page cannot show", () => {
  const good = { mode: "single", labels: [{ name: "beg" }, { name: "deal" }], examples: [{ id: "a", text: "Please?", scores: { beg: 0.7, deal: 0.3 } }] };
  assert.equal(checkData(good), good);
  const bad = (change) => () => checkData({ ...good, examples: [{ ...good.examples[0], ...change }] });
  assert.throws(bad({ scores: { beg: 0.7, deal: 0.2 } }), /add up to 0.900/);
  assert.throws(bad({ scores: { beg: 0.7 } }), /no score from 0 to 1 for "deal"/);
  assert.throws(bad({ scores: { beg: 0.7, deal: 0.3, nice: 0 } }), /not in the list/);
  assert.throws(bad({ text: "" }), /no text/);
  assert.throws(() => checkData({ ...good, examples: [good.examples[0], good.examples[0]] }), /repeats/);
});

test("the real data is used when it exists, otherwise the mock", () => {
  const mock = readData(MOCK);
  assert.equal(chooseData(undefined, mock).mock, true);
  assert.equal(chooseData(mock, mock).mock, false);
  assert.throws(() => chooseData(undefined, undefined), /add src\/data\/be-the-model.json/);
});

test("the mock has the shape of the real data, says it is a mock, and its scores add up to 1", () => {
  const mock = readData(MOCK);
  checkData(mock, "mock");
  assert.match(mock.about, /^MOCK DATA/);
  assert.equal(mock.mode, "single");
  assert.ok(mock.examples.length >= 8 && mock.examples.length <= 10);
  for (const example of mock.examples) {
    assert.deepEqual(Object.keys(example).sort(), ["age", "ask", "id", "multi", "scores", "text"]);
    const sum = Object.values(example.scores).reduce((a, b) => a + b, 0);
    assert.ok(Math.abs(sum - 1) < 1e-9, `${example.id} adds up to ${sum}`);
  }
});

test("the real data, when it is there, is fit to show, with at least 10 stops on every slider", { skip: !hasReal }, () => {
  const data = checkData(readData(REAL), REAL);
  for (const [label, stops] of Object.entries(stopsPerLabel(data))) {
    assert.ok(stops >= 10, `"${label}" has only ${stops} different scores to stop at`);
  }
});

test("the page gets the real data when it exists, and only what it uses", async () => {
  const data = beTheModelData(site);
  assert.equal(data.mock, !hasReal);
  const plugin = beTheModelPlugin(site);
  const id = plugin.resolveId("virtual:be-the-model");
  assert.equal(plugin.resolveId("react"), undefined);
  const source = plugin.load.call({ addWatchFile() {} }, id);
  const shipped = JSON.parse(source.replace(/^export default /, "").replace(/;$/, ""));
  assert.deepEqual(shipped, pageData(data));
  assert.ok(shipped.examples.every((example) => !("multi" in example)));
});

test("the page's words and the labels assume no family shape", () => {
  const family = /\b(mum|mom|mommy|dad|daddy|mother|father|parents?)\b/i;
  for (const path of ["src/pages/be-the-model.astro", "src/components/BeTheModel.jsx"]) {
    assert.doesNotMatch(readFileSync(join(site, path), "utf8"), family, path);
  }
  const mock = readData(MOCK);
  for (const example of mock.examples) assert.doesNotMatch(example.text, family, example.id);
  for (const data of [mock, beTheModelData(site)]) {
    for (const { name, description } of data.labels) assert.doesNotMatch(`${name} ${description}`, family, name);
  }
});

test("the Markdown copy lists the text with the highest score for each label", () => {
  const data = beTheModelData(site);
  const text = beTheModelText(data);
  assert.match(text, /^On the web page this is a set of sliders/);
  const rows = text.split("\n").filter((line) => line.startsWith("| ") && !line.startsWith("| Text") && !line.startsWith("| ---"));
  assert.ok(rows.length >= 2 && rows.length <= data.labels.length);
  for (const row of rows) assert.equal(row.split(" | ").length, data.labels.length + 1);
});
