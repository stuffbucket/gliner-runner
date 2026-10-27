// A small client for an OpenAI-compatible chat API. Every call is counted per stage so the report can say what
// each stage cost.

/** Pulls the first JSON object out of a reply, which may be wrapped in a ```json fence or have prose around it. */
export function parseJson(text) {
  const fenced = text.match(/```(?:json)?\s*([\s\S]*?)```/);
  const body = fenced ? fenced[1] : text;
  const start = body.indexOf("{");
  const end = body.lastIndexOf("}");
  if (start < 0 || end < start) throw new Error(`no JSON object in reply: ${text.slice(0, 200)}`);
  return JSON.parse(body.slice(start, end + 1));
}

export function createModel({ base, model, maxTokens = 2000 }) {
  const usage = {};
  const count = (stage, field, n) => {
    usage[stage] ??= { calls: 0, failures: 0, promptTokens: 0, completionTokens: 0, ms: 0 };
    usage[stage][field] += n;
  };

  async function chat(stage, messages) {
    const started = Date.now();
    const reply = await fetch(`${base}/chat/completions`, {
      method: "POST",
      // The local API takes any key; a key is sent because some OpenAI-compatible servers require the header.
      headers: { "content-type": "application/json", authorization: "Bearer local" },
      body: JSON.stringify({ model, messages, max_tokens: maxTokens, temperature: 0.2 }),
      signal: AbortSignal.timeout(180_000),
    });
    count(stage, "calls", 1);
    count(stage, "ms", Date.now() - started);
    const body = await reply.json().catch(() => ({}));
    if (!reply.ok) throw new Error(`${model} answered ${reply.status}: ${JSON.stringify(body).slice(0, 300)}`);
    count(stage, "promptTokens", body.usage?.prompt_tokens ?? 0);
    count(stage, "completionTokens", body.usage?.completion_tokens ?? 0);
    return body.choices?.[0]?.message?.content ?? "";
  }

  /** One call that must return JSON. A reply that does not parse, or a failed request, is retried once. */
  async function json(stage, messages) {
    let last;
    for (let attempt = 0; attempt < 2; attempt += 1) {
      try {
        return parseJson(await chat(stage, messages));
      } catch (failure) {
        last = failure;
        count(stage, "failures", 1);
      }
    }
    throw last;
  }

  return { model, usage, chat, json };
}

/** Text plus optional JPEG screenshots as one user message, in the OpenAI content-part format. */
export function userMessage(text, images = []) {
  if (images.length === 0) return { role: "user", content: text };
  return {
    role: "user",
    content: [
      { type: "text", text },
      ...images.map((jpeg) => ({ type: "image_url", image_url: { url: `data:image/jpeg;base64,${jpeg.toString("base64")}` } })),
    ],
  };
}

/** Runs `work` over `items` with at most `limit` in flight, keeping the order of the results. */
export async function pool(items, limit, work) {
  const results = new Array(items.length);
  let next = 0;
  await Promise.all(
    Array.from({ length: Math.min(limit, items.length) }, async () => {
      while (next < items.length) {
        const index = next++;
        results[index] = await work(items[index], index);
      }
    }),
  );
  return results;
}
