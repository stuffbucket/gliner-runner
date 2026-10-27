// The Markdown report for one slot: the top 3, the baselines, the full ranking, the show-don't-tell option, the
// drop reasons and the cost.
const f2 = (n) => (n == null ? "–" : Number(n).toFixed(2));
const pct = (n) => (n == null ? "–" : `${Math.round(n * 100)}%`);
const cell = (s) => String(s).replace(/\|/g, "\\|").replace(/\n+/g, " ");
const source = (c) =>
  c.source.kind === "baseline" ? `baseline: ${c.source.name}` : `${c.source.model}, ${c.source.angle} angle${c.also?.length ? ` (also ${c.also.length} more)` : ""}`;

function topBlock(c, rank, facts) {
  const r = c.retell ?? {};
  const optional = facts.filter((f) => f.optional).map((f) => f.id);
  return [
    `### ${rank}. ${c.id}`,
    "",
    ...c.text.split("\n").map((l) => `> ${l}`),
    "",
    `| words | grade | longest sentence | retell score (0-2) | facts retold | keep reading | judge win rate | Bradley-Terry | source |`,
    `| --- | --- | --- | --- | --- | --- | --- | --- | --- |`,
    `| ${c.checks.words} | ${f2(c.checks.grade)} | ${c.checks.longestSentence} | ${f2(r.score)} | ${pct(r.factRate)} | ${pct(r.keepRate)} | ${pct(c.judge?.winRate)} | ${f2(c.judge?.strength)} | ${cell(source(c))} |`,
    "",
    `Per fact: ${facts.map((fa) => `${fa.id} ${pct(r.perFact?.[fa.id])}`).join(", ")}${optional.length ? ` (${optional.join(", ")} optional)` : ""}.`,
    ...(r.sample?.length ? ["", `A reader (${r.sample[0].persona}) said: "${cell(r.sample[0].what ?? "")}" Like ChatGPT? "${cell(r.sample[0].like_chatgpt ?? "")}"`] : []),
    "",
  ].join("\n");
}

function table(rows, facts) {
  return [
    `| # | id | text | words | grade | retell | facts | keep | ${facts.map((f) => f.id).join(" | ")} | win rate | source |`,
    `| --- | --- | --- | --- | --- | --- | --- | --- | ${facts.map(() => "---").join(" | ")} | --- | --- |`,
    ...rows.map((c, i) =>
      `| ${i + 1} | ${c.id} | ${cell(c.text.length > 220 ? `${c.text.slice(0, 217)}…` : c.text)} | ${c.checks.words} | ${f2(c.checks.grade)} | ${f2(c.retell?.score)} | ${pct(c.retell?.factRate)} | ${pct(c.retell?.keepRate)} | ${facts.map((f) => pct(c.retell?.perFact?.[f.id])).join(" | ")} | ${pct(c.judge?.winRate)} | ${cell(source(c))} |`),
    "",
  ].join("\n");
}

function dropSummary(drops) {
  const kinds = {};
  for (const d of drops) {
    for (const reason of d.checks.reasons) {
      const kind = /^\d+ words/.test(reason) ? "too many words"
        : /^\d+ sentences/.test(reason) ? "too many sentences"
        : /^a sentence/.test(reason) ? "a sentence too long"
        : /^grade/.test(reason) ? "grade too high"
        : reason.replace(/:.*/, "");
      kinds[kind] = (kinds[kind] ?? 0) + 1;
    }
  }
  return Object.entries(kinds).sort((a, b) => b[1] - a[1]).map(([k, n]) => `${n} × ${k}`).join("; ");
}

export function renderReport({ slot, result, show, cost }) {
  const facts = slot.facts;
  const lines = [
    `# Copy lab: ${slot.id}`,
    "",
    `${slot.where}. ${slot.purpose}`,
    "",
    `Limits: at most ${slot.maxWords} words${slot.maxSentences ? `, ${slot.maxSentences} sentence(s)` : ""}, no sentence over ${slot.maxSentenceWords} words, Flesch-Kincaid grade ${slot.maxGrade} or lower, none of: ${slot.stallWords.join(", ")}; no meta (${slot.meta.join(", ")}).`,
    "",
    `Pipeline: ${result.generated} drafted (${slot.angles.length} angles x ${slot.drafters.length} models), ${result.drops.length} dropped by the checks, ${result.survivors.length} survivors, of which ${result.retold.length} (taken in turn from each model and angle) plus ${slot.baselines.length} baselines were retold by ${slot.personas.join(" and ")} (${slot.runs} runs each, graded by ${slot.grader}); the top ${slot.judgeTop} plus the reference baseline judged pairwise in both orders by ${slot.judges.join(", ")}. Ranked by Bradley-Terry strength.`,
    "",
    "The retell score is the share of required facts the reader retold plus the share of reads that would keep reading (0 to 2). The win rate is the share of pairwise judgements won (each judgement is half \"understood faster and remembered\", half \"sounds like a person\").",
    "",
    "## Top 3",
    "",
    ...result.top.slice(0, 3).map((c, i) => topBlock(c, i + 1, facts)),
    "## Best by retell alone",
    "",
    "The judges rank the top 3 above. These are the 3 that readers retold best, whatever the judges thought:",
    "",
    ...result.ranked.slice(0, 3).map((c) => `- ${c.id} (retell ${f2(c.retell?.score)}, win rate ${pct(c.judge?.winRate)}): ${cell(c.text)}`),
    "",
    "## Baselines",
    "",
    "Baselines are retold and (the reference) judged even when they break a limit.",
    "",
    table(result.baselines, facts),
    ...result.baselines.filter((b) => b.checks.reasons.length).map((b) => `- ${b.source.name}: ${b.checks.reasons.join("; ")}`),
    "",
    "## Judged entrants",
    "",
    `Share of games won by the text shown first: ${pct(result.positionBias)} (both orders are played, so this cancels).${result.judgeFailures ? ` ${result.judgeFailures} judgement(s) failed to parse after retries and were left out.` : ""}`,
    "",
    table(result.judged, facts),
  ];
  if (show) {
    lines.push(
      "## Show, don't tell (layout option)",
      "",
      `${slot.show.about} Layout: the sentence, then \`<BeTheModel client:load />\` from \`${slot.show.embed}\` (it is a client-side React island; the page would import it as \`be-the-model.astro\` does).`,
      "",
      `${show.generated} sentences drafted, ${show.drops.length} dropped, ${show.survivors.length} survived, ${show.retold.length} retold. Top 3 by retell score (no pairwise judging):`,
      "",
      ...show.ranked.slice(0, 3).map((c, i) => topBlock(c, i + 1, facts)),
    );
  }
  lines.push(
    "## Every survivor by retell score",
    "",
    table(result.ranked, facts),
    "## Drops",
    "",
    `${result.drops.length} of ${result.generated} drafts broke a limit: ${dropSummary(result.drops)}. Each one with its reasons is in \`drops.json\`.`,
    "",
    "## Cost",
    "",
    "| stage | live calls | cached | time |",
    "| --- | --- | --- | --- |",
    ...cost.map((c) => `| ${c.stage} | ${c.calls} | ${c.cached} | ${(c.ms / 1000).toFixed(0)} s |`),
    "",
  );
  return lines.join("\n");
}
