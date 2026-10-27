// One model call per category: turn that category's findings into concrete, page-specific changes, each tied to
// the findings it rests on.
import { pool } from "./model.mjs";

const MAX_FINDINGS = 30;

export function suggestPrompt(group, personas) {
  const who = Object.fromEntries(personas.map((p) => [p.id, `${p.name} (${p.goal})`]));
  const lines = group.items.slice(0, MAX_FINDINGS).map((item) => {
    const evidence = item.evidence
      .map((e) => `    - ${e.persona ? who[e.persona] ?? e.persona : e.source} on ${e.page}${e.quote ? `, quoting "${e.quote}"` : ""}: ${e.text}`)
      .join("\n");
    return `${item.id} (page ${item.page}${item.pages ? `; also ${item.pages.slice(1, 6).join(", ")}` : ""}):\n${evidence}`;
  });
  return [
    `You are improving the documentation site of GLiNER Runner, a local zero-shot text classifier with a CLI, an`,
    `HTTP server, a browser build and an Astro + Starlight site. Readers walked the site; below are the findings in`,
    `the category "${group.category}". Each finding has an id and its evidence.`,
    "",
    lines.join("\n\n"),
    "",
    "Suggest concrete changes to specific pages that would fix these. Write each change so an editor can make it",
    "without reading the findings: say what to add, remove, move or rewrite, and where. Merge findings that one",
    "change fixes. Skip findings that are wrong about the page or that no change to the site would fix.",
    "Answer with JSON only, in this shape, most valuable change first, at most 6 changes:",
    `{"suggestions": [{"page": "/start/", "change": "what to change", "why": "the reader problem it fixes", "findings": ["f12", "f30"], "effort": "small|medium|large"}]}`,
  ].join("\n");
}

export async function suggest({ groups, personas, model, concurrency }) {
  return pool(groups, concurrency, async (group) => {
    try {
      const reply = await model.json("suggest", [{ role: "user", content: suggestPrompt(group, personas) }]);
      return { category: group.category, suggestions: reply.suggestions ?? [] };
    } catch (failure) {
      return { category: group.category, suggestions: [], error: failure.message.slice(0, 300) };
    }
  });
}
