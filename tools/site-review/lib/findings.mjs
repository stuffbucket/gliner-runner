// Turns everything the review saw into findings, groups them by category and merges near-duplicates. Pure
// functions only, so they can be tested without a browser, a model or a runner.

/** The categories gliner-runner chooses from. Short names, with a description that says what each one covers. */
export const CATEGORIES = {
  "concept clarity": "The explanation of what the program is, what it does or how it works is unclear or hard to follow",
  terminology: "A word, name or phrase is jargon, is not explained, or means something else to this reader",
  navigation: "The reader cannot find the right page or link, or the path through the pages to the goal is unclear",
  "missing example": "The reader wants a code sample, a worked example or a concrete case that is not there",
  "setup and install": "Installing the program, downloading models, starting the server or connecting a page to it",
  "visual design": "Layout, spacing, sizes, contrast, images or the look of the page get in the way",
  accessibility: "Screen readers, keyboard use, colour contrast, missing labels or ARIA problems",
  "trust and privacy": "Whether the text stays private, and whether the project is safe, maintained and trustworthy",
  "accuracy and claims": "How accurate the answers are, how it was evaluated, benchmarks and evidence for claims",
  performance: "Speed, load time, download size or slowness of the site or the program",
};

const IMPACT = { critical: 3, serious: 2, moderate: 1, minor: 0 };
const clip = (text, n) => (text.length > n ? `${text.slice(0, n - 1)}…` : text);

/**
 * Every confusion, missing item and visual note from the walks and the hands-on tasks, every failed comprehension
 * question, every axe rule, broken link, phone overflow and slow page, as one list of findings. Checks that need no
 * judgement carry a `known` category; the classifier still scores them, which shows how far it can be trusted.
 */
export function collectFindings({ walks = [], handsOn = [], comprehension = [], checks = { pages: [] }, broken = [], slowMs = 3000, heavyKb = 3000 }) {
  const out = [];
  const add = (finding) => out.push({ id: `f${out.length + 1}`, quote: "", quoteOnPage: null, known: null, severity: 0, ...finding });

  const fromReply = (reply, base) => {
    for (const c of reply.confusions ?? []) {
      add({ ...base, kind: "confusion", quote: c.quote ?? "", quoteOnPage: c.onPage ?? null, text: clip(`${c.why} Quote: "${c.quote}"`, 400) });
    }
    for (const m of reply.missing ?? []) add({ ...base, kind: "missing", text: clip(`Expected but did not find: ${m}`, 400) });
    for (const v of reply.visual ?? []) add({ ...base, kind: "visual", text: clip(`Looks wrong: ${v}`, 400) });
  };
  for (const walk of walks) {
    for (const step of walk.path) fromReply(step.reply, { source: "walk", persona: walk.persona, page: step.page });
    if (!walk.goalReached) {
      const last = walk.path.at(-1);
      add({ source: "journey", kind: "goal not reached", persona: walk.persona, page: last?.page ?? "/", known: "navigation", severity: 1,
        text: clip(`Did not reach the goal: ${walk.ending}. Last thought: ${last?.reply.understood ?? "none"}`, 400) });
    }
  }
  for (const task of handsOn) {
    if (task.error) {
      add({ source: "hands-on", kind: "task failed", persona: task.persona, page: "/playground/", severity: 2, text: `The hands-on ${task.task} task failed: ${task.error}` });
    } else if (task.reply) {
      fromReply(task.reply, { source: "hands-on", persona: task.persona, page: "/playground/" });
      if (task.reply.result_made_sense === false) {
        add({ source: "hands-on", kind: "result did not make sense", persona: task.persona, page: "/playground/", severity: 1,
          text: clip(`The playground result did not make sense to me: ${task.reply.understood}`, 400) });
      }
    }
  }
  for (const test of comprehension.filter((t) => !t.pass)) {
    add({ source: "comprehension", kind: "not learned", persona: test.persona, page: test.pagePath, severity: 1,
      text: clip(`A reader of this page could not answer "${test.question}". Their answer: ${test.answer} Grader: ${test.reason}`, 500) });
  }
  const rules = new Map();
  for (const page of checks.pages) {
    for (const v of page.axe) {
      const rule = rules.get(v.id) ?? { ...v, pages: [], nodes: 0 };
      rule.pages.push(page.path);
      rule.nodes += v.nodes;
      rules.set(v.id, rule);
    }
    if (page.phoneOverflow.length > 0) {
      add({ source: "layout", kind: "phone overflow", page: page.path, known: "visual design", severity: 1,
        text: `On a 390px wide phone the page scrolls sideways. Too wide: ${page.phoneOverflow.join(", ")}` });
    }
    if (page.timing.loadMs > slowMs || page.timing.transferKb > heavyKb) {
      add({ source: "performance", kind: "slow page", page: page.path, known: "performance", severity: 1,
        text: `The page took ${page.timing.loadMs} ms to load and transferred ${page.timing.transferKb} KB in ${page.timing.requests} requests.` });
    }
  }
  for (const rule of rules.values()) {
    add({ source: "axe", kind: rule.id, page: rule.pages[0], pages: rule.pages, known: "accessibility", severity: IMPACT[rule.impact] ?? 0,
      text: clip(`Accessibility check failed (${rule.impact}): ${rule.help}. ${rule.nodes} elements on ${rule.pages.length} pages, for example ${rule.targets[0] ?? ""}.`, 400) });
  }
  for (const link of broken) {
    add({ source: "link", kind: "broken link", page: link.from, known: "navigation", severity: 2,
      text: `The link "${link.text}" goes to ${link.href}, which answers ${link.status}.` });
  }
  return out;
}

/** The text the classifier reads: the finding without the framing prefix, which says nothing about the topic. */
export const classifierText = (finding) => finding.text.replace(/^(Expected but did not find|Looks wrong): /, "");

/** Classifier scores for a finding: the labels that passed the threshold, and the primary category. */
export function categorize(finding, output) {
  const scores = output.probabilities;
  const picked = [output.value].flat().filter(Boolean);
  const top = Object.entries(scores).sort((a, b) => b[1] - a[1])[0][0];
  return { ...finding, scores, labels: picked, primary: finding.known ?? picked.sort((a, b) => scores[b] - scores[a])[0] ?? top };
}

const STOP = new Set("a an and are as at be but by for from has have i in is it its me my not of on or so that the this to was what when which with you your did find expected looks quote wrong".split(" "));
const words = (text) => new Set(text.toLowerCase().replace(/[^a-z0-9 ]/g, " ").split(/\s+/).filter((w) => w.length > 2 && !STOP.has(w)));

/** Jaccard overlap of the content words of two texts, from 0 (none) to 1 (the same words). */
export function similarity(a, b) {
  const x = words(a);
  const y = words(b);
  if (x.size === 0 || y.size === 0) return 0;
  let both = 0;
  for (const w of x) if (y.has(w)) both += 1;
  return both / (x.size + y.size - both);
}

const sameQuote = (a, b) => a.quote.length > 12 && a.quote.trim().toLowerCase() === b.quote.trim().toLowerCase();

/**
 * Merges findings in the same category about the same page that quote the same words or say nearly the same
 * thing. Each merged finding keeps all its evidence. Returns groups by category, most evidence first.
 */
export function groupFindings(findings, { threshold = 0.4 } = {}) {
  const groups = new Map();
  for (const finding of findings) {
    const list = groups.get(finding.primary) ?? [];
    const match = list.find((m) => m.page === finding.page && (sameQuote(m, finding) || similarity(m.text, finding.text) >= threshold));
    const evidence = { id: finding.id, persona: finding.persona ?? null, page: finding.page, source: finding.source, quote: finding.quote, text: finding.text };
    if (match) {
      match.evidence.push(evidence);
      match.severity = Math.max(match.severity, finding.severity);
    } else {
      list.push({ ...finding, evidence: [evidence] });
    }
    groups.set(finding.primary, list);
  }
  const rank = (m) => m.evidence.length + new Set(m.evidence.map((e) => e.persona).filter(Boolean)).size + m.severity + (m.pages?.length ?? 1) - 1;
  return [...groups.entries()]
    .map(([category, items]) => ({ category, items: items.map((m) => ({ ...m, rank: rank(m) })).sort((a, b) => b.rank - a.rank) }))
    .sort((a, b) => b.items.reduce((s, m) => s + m.rank, 0) - a.items.reduce((s, m) => s + m.rank, 0));
}

/** How often the classifier's top label agrees with the category a check already knows. */
export function agreement(findings) {
  const known = findings.filter((f) => f.known && Object.keys(f.scores ?? {}).length > 0);
  const top = (f) => Object.entries(f.scores).sort((a, b) => b[1] - a[1])[0][0];
  return { checked: known.length, agreed: known.filter((f) => top(f) === f.known).length };
}
