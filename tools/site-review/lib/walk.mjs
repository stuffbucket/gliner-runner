// Walks the site as one persona: read a page, ask the model what this person makes of it and where they go next,
// follow that link, and stop at the goal, when they give up, or after `steps` pages.
import { writeFileSync } from "node:fs";
import { join } from "node:path";

import { userMessage } from "./model.mjs";
import { open, quoteOnPage, readPage, shortUrl } from "./page.mjs";

const PAGE_TEXT_LIMIT = 15_000;
const LINK_LIMIT = 120;

export function personaSystem(persona) {
  return [
    `You are ${persona.name}. ${persona.background}`,
    `What you know: ${persona.knows}`,
    `Your goal: ${persona.goal}`,
    "You are reading this project's website for the first time. React as this person would, from what is on the page",
    "and in the screenshot, not from what you know about the project from elsewhere. Be honest and specific; do not",
    "invent problems to seem thorough, and do not praise. Answer with one JSON object and nothing else.",
  ].join("\n");
}

const REPLY_SHAPE = `{
  "understood": "one or two sentences: what you now understand",
  "confusions": [{"quote": "exact words copied from the page", "why": "why this confused you"}],
  "missing": ["something you expected on this page, at this point, and did not find"],
  "visual": ["a problem with the look or layout in the screenshot that got in your way"],
  "goal_reached": false,
  "next": {"link": 3, "why": "why you would click this"}
}`;

export function stepPrompt({ step, steps, page, history, siteRoot }) {
  const links = page.links.slice(0, LINK_LIMIT).map((link, i) => `[${i + 1}] ${link.text} -> ${shortUrl(link.href, siteRoot)}`);
  const text = page.text.length > PAGE_TEXT_LIMIT ? `${page.text.slice(0, PAGE_TEXT_LIMIT)}\n[…the page goes on]` : page.text;
  return [
    `Page ${step} of at most ${steps}.`,
    history.length > 0 ? `Pages you have read so far:\n${history.map((h, i) => `${i + 1}. ${h.page}: ${h.understood}`).join("\n")}` : "This is the first page you see.",
    "",
    `This page: ${shortUrl(page.url, siteRoot)}`,
    `Title: ${page.title}`,
    `Headings:\n${page.headings.join("\n")}`,
    `Text of the page:\n"""\n${text}\n"""`,
    `Links on the page, by number:\n${links.join("\n")}`,
    "A screenshot of the top of the page is attached.",
    "",
    `Answer in this shape:\n${REPLY_SHAPE}`,
    "List at most 4 confusions, 3 missing items and 3 visual problems, and only real ones; [] is fine.",
    `Set "next" to null when you have reached your goal or would give up here. Otherwise pick one link number.`,
  ].join("\n");
}

/** Checks each confusion quote against the page, so made-up quotes can be told apart in the report. */
export const checkQuotes = (reply, pageText) =>
  (reply.confusions ?? []).map((c) => ({ ...c, onPage: quoteOnPage(c.quote, pageText) }));

export async function walkPersona({ browser, persona, siteRoot, model, steps, shotsDir, log }) {
  const context = await browser.newContext({ viewport: { width: 1280, height: 800 } });
  const failures = [];
  context.on("response", (response) => {
    if (response.status() >= 400 && response.url().startsWith(new URL(siteRoot).origin)) {
      failures.push({ url: response.url(), status: response.status(), from: response.request().frame()?.url() ?? "" });
    }
  });
  const page = await context.newPage();
  const path = [];
  let url = siteRoot;
  let ending = `stopped after ${steps} pages`;
  try {
    for (let step = 1; step <= steps; step += 1) {
      const status = await open(page, url);
      const seen = await readPage(page, siteRoot);
      const shot = `${persona.id}-${String(step).padStart(2, "0")}.jpg`;
      const image = await page.screenshot({ type: "jpeg", quality: 60 });
      writeFileSync(join(shotsDir, shot), image);
      const history = path.map((p) => ({ page: shortUrl(p.url, siteRoot), understood: p.reply.understood }));
      let reply;
      try {
        reply = await model.json("walk", [
          { role: "system", content: personaSystem(persona) },
          userMessage(stepPrompt({ step, steps, page: seen, history, siteRoot }), [image]),
        ]);
      } catch (failure) {
        // Keep what the walk found so far; one bad reply should not lose the persona.
        ending = `the model failed: ${failure.message.slice(0, 200)}`;
        break;
      }
      reply.confusions = checkQuotes(reply, seen.allText);
      const choice = reply.next && Number.isInteger(reply.next.link) ? seen.links[reply.next.link - 1] : null;
      path.push({ step, url: seen.url, page: shortUrl(seen.url, siteRoot), status, title: seen.title, screenshot: shot, reply, chose: choice ?? null });
      log(`${persona.id} ${step}: ${shortUrl(seen.url, siteRoot)} -> ${choice ? shortUrl(choice.href, siteRoot) : reply.goal_reached ? "goal reached" : "stop"}`);
      if (reply.goal_reached) {
        ending = "reached the goal";
        break;
      }
      if (!reply.next) {
        ending = "gave up";
        break;
      }
      if (!choice) {
        ending = `chose link ${reply.next.link}, which is not on the page`;
        break;
      }
      if (!choice.internal) {
        ending = `left the site for ${choice.href}`;
        break;
      }
      url = choice.href;
    }
  } finally {
    await context.close();
  }
  return { persona: persona.id, ending, goalReached: ending === "reached the goal", path, failures };
}
