// Tests for the pure parts of the review: no browser, model or runner.
import assert from "node:assert/strict";
import { test } from "node:test";

import { CATEGORIES, agreement, categorize, classifierText, collectFindings, groupFindings, similarity } from "./findings.mjs";
import { guessMatches } from "./handson.mjs";
import { parseJson, pool } from "./model.mjs";
import { quoteOnPage, shortUrl } from "./page.mjs";
import { renderReport, topFindings } from "./report.mjs";

const scores = (top) => Object.fromEntries(Object.keys(CATEGORIES).map((c) => [c, c === top ? 0.9 : 0.1]));

const walk = {
  persona: "support-dev",
  goalReached: false,
  ending: "gave up",
  failures: [],
  path: [{
    step: 1, page: "/", url: "http://x/gliner-runner/", chose: null,
    reply: {
      understood: "It sorts text.",
      confusions: [{ quote: "decide-340m", why: "What is decide-340m?", onPage: true }],
      missing: ["A Node.js example"],
      visual: [],
      goal_reached: false,
      next: null,
    },
  }],
};
const checks = {
  pages: [{
    path: "/", slug: "home", status: 200, phoneOverflow: [], timing: { loadMs: 100, firstContentfulPaintMs: 50, requests: 3, transferKb: 10 },
    axe: [{ id: "color-contrast", impact: "serious", help: "Contrast", helpUrl: "u", nodes: 2, targets: [".output"] }],
  }],
};

test("parseJson reads a fenced reply and one with prose around it", () => {
  assert.deepEqual(parseJson('```json\n{"a": 1}\n```'), { a: 1 });
  assert.deepEqual(parseJson('Sure. {"a": {"b": 2}} Done.'), { a: { b: 2 } });
  assert.throws(() => parseJson("no json here"));
});

test("pool keeps the order of results and the limit", async () => {
  let running = 0;
  let most = 0;
  const out = await pool([3, 1, 2], 2, async (n) => {
    most = Math.max(most, ++running);
    await new Promise((r) => setTimeout(r, n * 5));
    running -= 1;
    return n * 10;
  });
  assert.deepEqual(out, [30, 10, 20]);
  assert.equal(most, 2);
});

test("quotes are found despite case, curly quotes and spacing; made-up ones are not", () => {
  const page = "Your text never leaves your device.\nIt’s   free.";
  assert.ok(quoteOnPage("your text never leaves  your device", page));
  assert.ok(quoteOnPage("\"It's free.\"", page));
  assert.ok(!quoteOnPage("Your data is encrypted", page));
  assert.ok(!quoteOnPage("", page));
});

test("shortUrl strips the site root and keeps other links", () => {
  assert.equal(shortUrl("http://h/gliner-runner/start/", "http://h/gliner-runner/"), "/start/");
  assert.equal(shortUrl("https://github.com/x", "http://h/gliner-runner/"), "https://github.com/x");
});

test("a guess matches only when it names the thing", () => {
  assert.ok(guessMatches("Is it a zebra?", "zebra"));
  assert.ok(guessMatches("Is it an apple tree?", "apple trees"));
  assert.ok(!guessMatches("Is it a horse?", "zebra"));
  assert.ok(!guessMatches("Is it a tree?", "apple tree"));
});

test("collectFindings turns replies and checks into findings, with known categories for checks", () => {
  const comprehension = [{ id: "q", pass: false, persona: "ml-engineer", pagePath: "/evaluation/", question: "How accurate?", answer: "The page does not say.", reason: "No accuracy." }];
  const broken = [{ from: "/", text: "Docs", href: "http://x/gone/", status: 404 }];
  const findings = collectFindings({ walks: [walk], handsOn: [{ persona: "react-dev", task: "classify", error: "timeout" }], comprehension, checks, broken });
  const kinds = findings.map((f) => `${f.source}:${f.kind}`);
  assert.deepEqual(kinds, [
    "walk:confusion", "walk:missing", "journey:goal not reached", "hands-on:task failed", "comprehension:not learned", "axe:color-contrast", "link:broken link",
  ]);
  assert.equal(findings.find((f) => f.source === "axe").known, "accessibility");
  assert.equal(findings.find((f) => f.source === "link").known, "navigation");
  assert.equal(new Set(findings.map((f) => f.id)).size, findings.length);
  assert.equal(classifierText(findings[1]), "A Node.js example");
});

test("categorize prefers a known category, then the highest passing label", () => {
  const output = { value: ["terminology", "concept clarity"], probabilities: { ...scores("terminology"), "concept clarity": 0.7 } };
  assert.equal(categorize({ known: null }, output).primary, "terminology");
  assert.equal(categorize({ known: "accessibility" }, output).primary, "accessibility");
  assert.equal(categorize({ known: null }, { value: [], probabilities: scores("navigation") }).primary, "navigation");
});

test("groupFindings merges near-duplicates on the same page and keeps their evidence", () => {
  const base = { primary: "missing example", page: "/use-in-react/", quote: "", severity: 0, source: "walk" };
  const findings = [
    { ...base, id: "f1", persona: "a", text: "Expected but did not find: a Node.js example calling /v1/infer" },
    { ...base, id: "f2", persona: "b", text: "Expected but did not find: an example calling /v1/infer from Node.js" },
    { ...base, id: "f3", persona: "c", text: "Expected but did not find: a Python client", page: "/start/" },
    { ...base, id: "f4", persona: "a", text: "Looks wrong: low contrast", primary: "visual design" },
  ];
  const groups = groupFindings(findings);
  const examples = groups.find((g) => g.category === "missing example");
  assert.equal(examples.items.length, 2);
  assert.deepEqual(examples.items[0].evidence.map((e) => e.id), ["f1", "f2"]);
  assert.equal(groups[0].category, "missing example");
  assert.ok(similarity("a Node.js example", "an example in Node.js") > 0.5);
  assert.equal(similarity("", "x"), 0);
});

test("agreement counts how often the classifier's top label is the known one", () => {
  const findings = [
    { known: "accessibility", scores: scores("accessibility") },
    { known: "navigation", scores: scores("terminology") },
    { known: null, scores: scores("terminology") },
    { known: "navigation", scores: {} },
  ];
  assert.deepEqual(agreement(findings), { checked: 2, agreed: 1 });
});

test("the report renders every section from a small run", () => {
  const findings = collectFindings({ walks: [walk], checks }).map((f) => ({ ...f, scores: scores("terminology"), labels: [], primary: f.known ?? "terminology" }));
  const groups = groupFindings(findings);
  const md = renderReport({
    meta: { site: "http://x/gliner-runner/", model: "m", steps: 8, started: "t", ms: 1000 },
    stages: [{ name: "walk", ms: 1000, calls: 1 }],
    usage: { walk: { calls: 1 } },
    personas: [{ id: "support-dev", name: "Dana", goal: "Tag tickets" }],
    walks: [walk], handsOn: [], comprehension: { tests: [] }, checks: { ...checks, links: 3 }, broken: [],
    findings, groups, agreement: agreement(findings),
    suggestions: [{ category: "terminology", suggestions: [{ page: "/", change: "Explain decide-340m", why: "Unclear", findings: ["f1"], effort: "small" }] }],
  });
  for (const heading of ["## Top findings", "## Suggested improvements", "## Personas", "## Accessibility (axe)", "## Links"]) assert.ok(md.includes(heading), heading);
  assert.ok(md.includes("Evidence f1: Dana on /, quoting \"decide-340m\""));
  assert.equal(topFindings(groups, 2).length, 2);
});
