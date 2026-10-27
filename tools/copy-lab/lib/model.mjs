// A chat client for an OpenAI-compatible API with a disk cache. Every reply is stored under a hash of the model,
// the messages, the temperature and a `salt` (the run number, for calls that are sampled on purpose), so a
// re-run only pays for what changed. Calls are counted per stage, live and cached apart.
import { createHash } from "node:crypto";
import { existsSync, mkdirSync, readFileSync, writeFileSync } from "node:fs";
import { join } from "node:path";

import { parseJson } from "../../site-review/lib/model.mjs";

export { parseJson, pool } from "../../site-review/lib/model.mjs";

export const hash = (value) => createHash("sha256").update(JSON.stringify(value)).digest("hex").slice(0, 24);

export function createClient({ base, cacheDir }) {
  mkdirSync(cacheDir, { recursive: true });
  const usage = {};
  const count = (stage, field, n = 1) => {
    usage[stage] ??= { calls: 0, cached: 0, failures: 0, promptTokens: 0, completionTokens: 0, ms: 0 };
    usage[stage][field] += n;
  };

  async function chat({ stage, model, messages, temperature = 0, maxTokens = 4000, salt = 0 }) {
    const key = hash({ model, messages, temperature, salt });
    const file = join(cacheDir, `${key}.json`);
    if (existsSync(file)) {
      count(stage, "cached");
      return JSON.parse(readFileSync(file, "utf8")).content;
    }
    const started = Date.now();
    const reply = await fetch(`${base}/chat/completions`, {
      method: "POST",
      headers: { "content-type": "application/json", authorization: "Bearer local" },
      body: JSON.stringify({ model, messages, max_tokens: maxTokens, temperature }),
      signal: AbortSignal.timeout(240_000),
    });
    count(stage, "calls");
    count(stage, "ms", Date.now() - started);
    const body = await reply.json().catch(() => ({}));
    if (!reply.ok) throw new Error(`${model} answered ${reply.status}: ${JSON.stringify(body).slice(0, 300)}`);
    count(stage, "promptTokens", body.usage?.prompt_tokens ?? 0);
    count(stage, "completionTokens", body.usage?.completion_tokens ?? 0);
    const content = body.choices?.[0]?.message?.content ?? "";
    writeFileSync(file, JSON.stringify({ model, content }));
    return content;
  }

  /** A call that must return JSON. A failed request or a reply that does not parse is tried again, up to 3 times. */
  async function json(options) {
    let last;
    for (let attempt = 0; attempt < 3; attempt += 1) {
      try {
        return parseJson(await chat({ ...options, salt: `${options.salt ?? 0}/${attempt}` }));
      } catch (failure) {
        last = failure;
        count(options.stage, "failures");
      }
    }
    throw last;
  }

  return { usage, chat, json };
}
