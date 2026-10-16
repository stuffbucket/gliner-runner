// Precomputes the landing page's example gallery with the real models, so the page shows actual answers without
// running a model. Run after changing the entries or a model pin; the output is committed:
//
//   GLINER_RUNNER_MODELS_DIR=dist/onnx-models node site/scripts/gallery.mjs
//
// It starts the built runner (packages/onnx-runner/dist), classifies every entry exactly as the matching
// `gliner-runner classify` command on its example page does, and writes site/src/data/gallery.json.

import { spawn } from "node:child_process";
import { mkdtempSync, writeFileSync } from "node:fs";
import { createServer } from "node:net";
import { tmpdir } from "node:os";
import { dirname, join, resolve } from "node:path";
import { fileURLToPath } from "node:url";

const here = dirname(fileURLToPath(import.meta.url));
const root = resolve(here, "../..");
const OUTPUT = join(here, "../src/data/gallery.json");

const SUPPORT = {
  billing: "Charges, invoices, refunds, or payment problems",
  technical: "Something does not work, crashes, or shows an error",
  shipping: "Delivery, tracking, or a damaged or missing package",
  account: "Logging in, passwords, or changing account details",
  cancellation: "The customer wants to stop or cancel the service",
};
const MESSAGES = {
  urgent: "Needs a reply or action within the hour",
  "can wait": "Useful, but nothing is needed today",
  spam: "Unwanted advertising, prizes, or scam attempts",
  personal: "A friendly message from family or a friend",
};
const MODERATION = {
  harassment: "Insults or attacks aimed at a person",
  spam: "Repeated or unwanted ads, scam links, or junk",
  "self-promotion": "Promotes the poster's own product, channel, or business",
  "off-topic": "About sports, phones, politics, or anything other than cooking",
  ok: "A normal post about cooking that follows the rules",
};
const PERSONAL = {
  "phone number": "A telephone number",
  "email address": "An email address",
  "home address": "A street address where someone lives or receives mail",
  "payment card": "A credit or debit card number, expiry date, or security code",
  none: "The message contains no personal information",
};
const MULTILINGUAL = {
  billing: "Charges, invoices, refunds, or payment problems",
  technical: "Something does not work or shows an error",
  shipping: "Delivery, tracking, or a missing package",
  account: "Logging in, passwords, or account details",
};
const labelsOf = (descriptions) => Object.keys(descriptions);

/**
 * Each entry repeats the command on its example page (`example` under /examples/): the same model, labels,
 * descriptions, `--multi` and `--threshold` (default 0.5).
 */
export const ENTRIES = [
  { example: "support-tickets", model: "decide-340m", text: "I was charged twice for my March order.", labels: labelsOf(SUPPORT), descriptions: SUPPORT },
  { example: "text-messages", model: "decide-340m", text: "The basement is flooding. Can you call me right now?", labels: labelsOf(MESSAGES), descriptions: MESSAGES },
  { example: "text-messages", model: "decide-340m", text: "Congratulations! You won a $500 gift card. Tap the link to claim it now.", labels: labelsOf(MESSAGES), descriptions: MESSAGES },
  {
    example: "urgent-or-can-wait", model: "decide-340m",
    text: "The fire alarm in building B is going off. Please leave the building now.",
    labels: ["urgent", "can wait"],
    descriptions: { urgent: "Needs a reply or action today", "can wait": "Nothing is needed today; it can be read later this week" },
  },
  { example: "reviews", model: "decide-340m", text: "The pasta was excellent and the waiter was very kind, but it was far too expensive.", labels: ["positive", "mixed", "negative"] },
  { example: "reviews", model: "decide-340m", text: "Too loud to talk, and the soup was cold.", labels: ["food", "service", "price", "cleanliness", "noise"], multi: true, threshold: 0.3 },
  { example: "moderation", model: "decide-340m", text: "BUY CHEAP WATCHES NOW!!! Click http://example.com/deal for 90% off.", labels: labelsOf(MODERATION), descriptions: MODERATION, multi: true, threshold: 0.4 },
  { example: "personal-information", model: "decide-340m", text: "Reach me at jordan.lee@example.com or on 415-555-0132.", labels: labelsOf(PERSONAL), descriptions: PERSONAL, multi: true, threshold: 0.4 },
  { example: "support-tickets", model: "decide-1b", text: "Please cancel my subscription before the next renewal.", labels: labelsOf(SUPPORT), descriptions: SUPPORT },
  { example: "other-languages", model: "decide-multilingual", text: "Ich kann mich nicht mehr einloggen, das Passwort wird nicht akzeptiert.", labels: labelsOf(MULTILINGUAL), descriptions: MULTILINGUAL },
  { example: "other-languages", model: "decide-multilingual", text: "Me cobraron dos veces el mismo pedido y quiero que me devuelvan el dinero.", labels: labelsOf(MULTILINGUAL), descriptions: MULTILINGUAL },
  { example: "which-language", model: "decide-340m", text: "¿Dónde está la estación de tren más cercana?", labels: ["English", "Spanish", "French", "German", "Italian"] },
  {
    example: "keep-or-discard", model: "decide-340m",
    text: "Signed lease agreement for the apartment, valid until 2028.",
    labels: ["important to keep", "safe to throw away"],
    descriptions: {
      "important to keep": "A legal, tax, money, or warranty record that I may need later",
      "safe to throw away": "Advertising, expired offers, or things that are over",
    },
  },
  {
    example: "email-lines", model: "decide-340m",
    text: "Could you send me the updated budget numbers?",
    labels: ["action request", "question", "deadline", "information only", "greeting or sign-off"],
    descriptions: {
      "action request": "Asks the reader to do something",
      question: "Asks something the reader should answer",
      deadline: "Says something must happen by a certain date or time",
      "information only": "Shares a fact and asks for nothing",
      "greeting or sign-off": "A hello, thanks, or closing line",
    },
  },
];

const freePort = () =>
  new Promise((done) => {
    const probe = createServer();
    probe.listen(0, "127.0.0.1", () => {
      const { port } = probe.address();
      probe.close(() => done(port));
    });
  });

async function main() {
  const port = await freePort();
  const runner = spawn(process.execPath, [join(root, "packages/onnx-runner/dist/src/cli.js"), "serve", "--port", String(port)], {
    env: { ...process.env, XDG_CONFIG_HOME: mkdtempSync(join(tmpdir(), "gallery-config-")) },
    stdio: ["ignore", "ignore", "inherit"],
  });
  const base = `http://127.0.0.1:${port}`;
  try {
    for (let tries = 0; !(await fetch(`${base}/healthz`).then((r) => r.ok, () => false)); tries += 1) {
      if (tries > 200) throw new Error("the runner did not start");
      await new Promise((r) => setTimeout(r, 50));
    }
    const versions = new Map((await (await fetch(`${base}/v1/models`)).json()).models.map((m) => [m.model_id, m.pinned_revision]));
    const entries = [];
    for (const entry of ENTRIES) {
      const threshold = entry.threshold ?? 0.5;
      const task = {
        labels: entry.labels,
        ...(entry.descriptions === undefined ? {} : { label_descriptions: entry.descriptions }),
        ...(entry.multi ? { min_labels: 0, max_labels: null } : {}),
        threshold,
      };
      const reply = await fetch(`${base}/v1/infer`, {
        method: "POST",
        headers: { "content-type": "application/json" },
        body: JSON.stringify({
          model: entry.model, backend: "onnx", precision: "fp32", operation: "classify",
          text: entry.text, schema: { kind: "classification", tasks: { category: task } }, options: { threshold },
        }),
      });
      const body = await reply.json();
      if (!reply.ok) throw new Error(`${entry.example}: ${JSON.stringify(body.detail)}`);
      const { value, probabilities } = body.output.category;
      entries.push({
        ...entry,
        multi: entry.multi === true,
        threshold,
        picked: [value].flat().filter(Boolean),
        scores: Object.fromEntries(Object.entries(probabilities).map(([label, p]) => [label, Math.round(p * 1000) / 1000])),
        revision: versions.get(entry.model),
      });
      console.log(`${entry.model.padEnd(20)} ${[value].flat().join(", ") || "(none)"}  <- ${entry.text}`);
    }
    writeFileSync(OUTPUT, `${JSON.stringify({ generated_by: "site/scripts/gallery.mjs", entries }, null, 2)}\n`);
    console.log(`wrote ${entries.length} entries to ${OUTPUT}`);
  } finally {
    runner.kill("SIGTERM");
  }
}

if (process.argv[1] && resolve(process.argv[1]) === fileURLToPath(import.meta.url)) {
  await main();
}
