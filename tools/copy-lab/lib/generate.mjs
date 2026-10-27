// Drafting: each angle is sent to each drafting model, which returns several candidates. Duplicates (same words,
// ignoring case and punctuation) are kept once, crediting the first source.
import { hash, pool } from "./model.mjs";
import { draftingMessages } from "./prompts.mjs";

const normal = (text) => text.toLowerCase().replace(/[^\p{L}\p{N}]+/gu, " ").trim();

export const candidateId = (prefix, text) => `${prefix}${hash(text).slice(0, 8)}`;

export async function generate({ slot, client, concurrency, log }) {
  const jobs = slot.angles.flatMap((angle) => slot.drafters.map((model) => ({ angle, model })));
  log(`generate: ${jobs.length} drafting calls (${slot.angles.length} angles x ${slot.drafters.length} models, ${slot.perCall} each)`);
  const replies = await pool(jobs, concurrency, async ({ angle, model }) => {
    try {
      const reply = await client.json({
        stage: "generate",
        model,
        messages: draftingMessages({ slot, angle, count: slot.perCall }),
        temperature: 0.9,
        maxTokens: 8000,
      });
      return { angle, model, texts: (reply.candidates ?? []).filter((t) => typeof t === "string") };
    } catch (failure) {
      log(`generate: ${model} on ${angle} failed: ${failure.message}`);
      return { angle, model, texts: [] };
    }
  });
  const seen = new Map();
  for (const { angle, model, texts } of replies) {
    for (const raw of texts) {
      // A reply sometimes wraps a whole candidate in quotes; quotes inside a candidate stay.
      const trimmed = raw.trim();
      const text = /^["“][^"“”]*["”]$/.test(trimmed) ? trimmed.slice(1, -1).trim() : trimmed;
      const key = normal(text);
      if (!key) continue;
      if (seen.has(key)) {
        seen.get(key).also.push(`${model} / ${angle}`);
        continue;
      }
      seen.set(key, { id: candidateId("g", text), text, source: { kind: "generated", model, angle }, also: [] });
    }
  }
  return [...seen.values()];
}
