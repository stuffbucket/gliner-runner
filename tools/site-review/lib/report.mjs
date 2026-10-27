// Renders report.md from the run's data. Pure: data in, Markdown out.

const cell = (text) => String(text ?? "").replace(/\|/g, "\\|").replace(/\s+/g, " ").trim();
const clip = (text, n) => {
  const s = cell(text);
  return s.length > n ? `${s.slice(0, n - 1)}…` : s;
};
const table = (head, rows) => [`| ${head.join(" | ")} |`, `| ${head.map(() => "---").join(" | ")} |`, ...rows.map((r) => `| ${r.map(cell).join(" | ")} |`)].join("\n");
const seconds = (ms) => `${(ms / 1000).toFixed(0)} s`;

/** The ten findings with the most evidence across all categories. */
export function topFindings(groups, n = 10) {
  return groups
    .flatMap((g) => g.items.map((item) => ({ ...item, category: g.category })))
    .sort((a, b) => b.rank - a.rank)
    .slice(0, n);
}

export function renderReport(data) {
  const { meta, stages, walks, handsOn, comprehension, checks, broken, groups, suggestions, agreement, personas } = data;
  const name = Object.fromEntries(personas.map((p) => [p.id, p.name]));
  const byId = new Map(groups.flatMap((g) => g.items.flatMap((item) => [[item.id, item], ...item.evidence.map((e) => [e.id, item])])));
  const calls = Object.values(data.usage).reduce((s, u) => s + u.calls, 0);
  const out = [];

  out.push(`# Site review: ${meta.site}`, "");
  out.push(`Run ${meta.started} with \`${meta.model}\`. ${walks.length} personas, at most ${meta.steps} pages each.`, "");
  out.push(table(["Stage", "Time", "Model calls", "Other work"], stages.map((s) => [s.name, seconds(s.ms), s.calls, s.note ?? ""])));
  out.push("", `Total: ${seconds(meta.ms)}, ${calls} model calls.`, "");

  out.push("## Summary", "");
  out.push(`- Goals reached: ${walks.filter((w) => w.goalReached).length} of ${walks.length} personas.`);
  if (comprehension.tests.length > 0) out.push(`- Comprehension: ${comprehension.tests.filter((t) => t.pass).length} of ${comprehension.tests.length} questions answered from the page.`);
  out.push(`- Accessibility: ${new Set(checks.pages.flatMap((p) => p.axe.map((v) => v.id))).size} axe rules fail, on ${checks.pages.filter((p) => p.axe.length > 0).length} of ${checks.pages.length} pages.`);
  out.push(`- Links: ${broken.length} broken internal links; ${walks.reduce((s, w) => s + w.failures.length, 0)} failed requests while walking.`);
  out.push(`- Hands-on: ${handsOn.filter((t) => t.reply).length} of ${handsOn.length} playground tasks completed.`);
  out.push(`- Findings: ${data.findings.length}, merged into ${groups.reduce((s, g) => s + g.items.length, 0)} in ${groups.length} categories.`, "");

  out.push("## Top findings", "");
  out.push(table(["#", "Category", "Page", "Finding", "Evidence"], topFindings(groups).map((f, i) => [
    i + 1, f.category, f.page, clip(f.text, 180), `${f.evidence.length} (${[...new Set(f.evidence.map((e) => e.persona ?? e.source))].join(", ")})`,
  ])));
  out.push("");

  out.push("## Suggested improvements", "");
  for (const block of suggestions) {
    out.push(`### ${block.category}`, "");
    if (block.error) out.push(`The model call failed: ${block.error}`, "");
    for (const s of block.suggestions) {
      out.push(`- **${cell(s.page)}** (${s.effort ?? "?"}): ${cell(s.change)}`);
      out.push(`  - Why: ${cell(s.why)}`);
      const evidence = (s.findings ?? []).map((id) => byId.get(id)).filter(Boolean);
      for (const item of [...new Set(evidence)].slice(0, 4)) {
        const e = item.evidence[0];
        out.push(`  - Evidence ${item.id}: ${e.persona ? name[e.persona] : e.source} on ${e.page}${e.quote ? `, quoting "${clip(e.quote, 120)}"` : ""}: ${clip(e.text, 160)}`);
      }
    }
    out.push("");
  }

  out.push("## Findings by category", "");
  for (const group of groups) {
    out.push(`### ${group.category} (${group.items.length})`, "");
    out.push(table(["Id", "Page", "Source", "Finding", "Evidence"], group.items.map((f) => [
      f.id, f.page, f.source, clip(f.text, 220), f.evidence.length,
    ])));
    out.push("");
  }
  out.push(`The classifier's top label matched the known category for ${agreement.agreed} of ${agreement.checked} findings from checks (axe, links, layout, speed).`, "");

  out.push("## Personas", "");
  for (const walk of walks) {
    const persona = personas.find((p) => p.id === walk.persona);
    out.push(`### ${persona.name}`, "", `Goal: ${persona.goal}`, "", `Result: ${walk.ending}.`, "");
    out.push(table(["Step", "Page", "Understood", "Confusions", "Next"], walk.path.map((s) => [
      s.step, s.page, clip(s.reply.understood, 200), (s.reply.confusions ?? []).length,
      s.chose ? `${s.chose.text} (${clip(s.reply.next?.why, 100)})` : s.reply.goal_reached ? "goal reached" : "stop",
    ])));
    out.push("");
  }

  if (handsOn.length > 0) {
    out.push("## Hands-on in the playground", "");
    for (const task of handsOn) {
      out.push(`### ${name[task.persona]}: ${task.task}`, "");
      if (task.error) {
        out.push(`Failed: ${task.error}`, "");
        continue;
      }
      out.push(`${task.did} It took ${task.seconds?.toFixed(1)} s.`, "");
      if (task.turns) out.push(`Guesses: ${task.turns.map((t) => `${t.guess} (${t.answer})`).join("; ")}. ${task.verdict ?? ""}`, "");
      out.push(`Made sense: ${task.reply.result_made_sense ? "yes" : "no"}. ${cell(task.reply.understood)}`, "");
    }
  }

  if (comprehension.tests.length > 0 || comprehension.error) {
    out.push("## Comprehension (promptfoo)", "");
    if (comprehension.error) out.push(`promptfoo failed: ${comprehension.error}`, "");
    out.push(table(["Question", "Page", "Persona", "Result", "Grader"], comprehension.tests.map((t) => [
      t.id, t.pagePath, t.persona, t.pass ? "pass" : "FAIL", clip(t.reason, 200),
    ])));
    out.push("");
  }

  out.push("## Accessibility (axe)", "");
  const rules = new Map();
  for (const page of checks.pages) for (const v of page.axe) rules.set(v.id, { ...v, pages: [...(rules.get(v.id)?.pages ?? []), page.path] });
  out.push(table(["Rule", "Impact", "Pages", "What", "Example"], [...rules.values()].map((r) => [
    `[${r.id}](${r.helpUrl})`, r.impact, r.pages.join(", "), r.help, clip(r.targets[0], 80),
  ])));
  out.push("");

  out.push("## Pages: speed and phone layout", "");
  out.push(table(["Page", "Status", "Load", "First paint", "Requests", "KB", "Phone (390px)"], checks.pages.map((p) => [
    p.path, p.status, `${p.timing.loadMs} ms`, `${p.timing.firstContentfulPaintMs} ms`, p.timing.requests, p.timing.transferKb,
    p.phoneOverflow.length ? `scrolls sideways: ${p.phoneOverflow.join(", ")}` : "fits",
  ])));
  out.push("");

  out.push("## Links", "");
  if (broken.length === 0) out.push(`All ${checks.links} internal links answered 200.`);
  else out.push(table(["From", "Link", "Target", "Status"], broken.map((b) => [b.from, b.text, b.href, b.status])));
  const failed = walks.flatMap((w) => w.failures.map((f) => [w.persona, f.url, f.status]));
  if (failed.length > 0) out.push("", "Failed requests while walking:", "", table(["Persona", "URL", "Status"], failed));
  out.push("", "## Files", "", "- `data.json`: every page read, every model reply, classifier scores, promptfoo results.",
    "- `screenshots/`: one per walk step (`<persona>-<step>.jpg`), hands-on before and after, and each key page at phone width.",
    "- `pages/`: the text of each key page, as the comprehension stage read it.", "");
  return out.join("\n");
}
