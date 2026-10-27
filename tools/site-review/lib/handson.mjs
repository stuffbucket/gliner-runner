// One hands-on task in the playground, connected to a local runner: the persona decides what to try, the script
// does it with real clicks and typing, and the persona says what they made of the result.
import { writeFileSync } from "node:fs";
import { join } from "node:path";

import { userMessage } from "./model.mjs";
import { open, quoteOnPage, readPage } from "./page.mjs";
import { personaSystem } from "./walk.mjs";

// The review connects the playground to a runner on this computer, so the persona does not wait for the 1.95 GB
// in-browser download. Saying so keeps that choice from showing up as a finding about the site.
const SETUP = "For this test, the playground was already connected to GLiNER Runner on your computer (its \"Run it on my computer\" option), so you did not download the model into the browser.";

const PLANS = {
  classify: `${SETUP} Decide what you would try first, for your own goal. Answer in this shape:
{"text": "the text you would paste", "labels": ["label", "label"], "several": false, "expect": "what you expect to see"}`,
  "twenty-questions": `${SETUP} You want to try the "20 questions" example. Think of an animal, a plant or a
thing, and describe it in one or two sentences without naming it. Answer in this shape:
{"thing": "what you are thinking of, one or two words", "clues": "your description", "expect": "what you expect to happen"}`,
};

const REFLECT_SHAPE = `{
  "understood": "what you understood from the result",
  "result_made_sense": true,
  "confusions": [{"quote": "exact words copied from the page", "why": "why this confused you"}],
  "missing": ["something you expected to see or be able to do and could not"],
  "visual": ["a problem with the look or layout in the screenshot"]
}`;

/** Whether a guess such as "Is it a zebra?" names the thing: the persona answers honestly. */
export function guessMatches(guess, thing) {
  const words = (s) => s.toLowerCase().replace(/[^a-z ]/g, " ").split(/\s+/).filter((w) => w.length > 2 && !["is", "the", "it"].includes(w));
  const wanted = words(thing).map((w) => w.replace(/s$/, ""));
  const said = new Set(words(guess).map((w) => w.replace(/s$/, "")));
  return wanted.length > 0 && wanted.every((w) => said.has(w));
}

async function waitReady(page) {
  // The status pill says "Ready · your computer" once the runner has loaded a model.
  await page.getByRole("status").filter({ hasText: /^Ready/ }).waitFor({ timeout: 120_000 });
}

async function runClassify(page, plan) {
  const before = await page.locator(".pg-answer").count();
  await page.locator("#pg-text").fill(plan.text);
  await page.locator(".pg-labels input").fill([plan.labels].flat().join(", "));
  await page.getByRole("switch").setChecked(Boolean(plan.several));
  await page.locator(".pg-classify").click();
  await page.waitForFunction((n) => document.querySelectorAll(".pg-answer").length > n, before, { timeout: 60_000 });
  return { did: `Classified "${plan.text}" with labels ${[plan.labels].flat().join(", ")}${plan.several ? ", several labels allowed" : ""}.` };
}

async function runGame(page, plan) {
  await page.getByRole("button", { name: "20 questions", exact: true }).click();
  await page.locator("#pg-text").fill(plan.clues);
  await page.locator(".pg-classify").click();
  const turns = [];
  for (let asked = 0; asked < 20; asked += 1) {
    const yes = page.getByRole("button", { name: "Yes, that's it" });
    const again = page.getByRole("button", { name: "Play again" });
    await yes.or(again).first().waitFor({ timeout: 120_000 });
    if (await again.isVisible()) break;
    const guess = (await page.locator(".pg-turn-guess").innerText()).trim();
    const right = guessMatches(guess, plan.thing);
    turns.push({ guess, answer: right ? "yes" : "no" });
    await (right ? yes : page.getByRole("button", { name: "No", exact: true })).click();
    if (right) break;
  }
  await page.getByRole("button", { name: "Play again" }).waitFor({ timeout: 30_000 }).catch(() => {});
  const verdict = (await page.locator(".pg-turn").innerText().catch(() => "")).trim();
  return { did: `Played 20 questions thinking of "${plan.thing}" with the clues "${plan.clues}".`, turns, verdict };
}

export async function handsOn({ browser, persona, siteRoot, runnerBase, model, shotsDir, log }) {
  const context = await browser.newContext({ viewport: { width: 1280, height: 900 } });
  const page = await context.newPage();
  const url = `${siteRoot}playground/?mode=local&runner=${encodeURIComponent(runnerBase)}`;
  const record = { persona: persona.id, task: persona.handsOn, url };
  try {
    await open(page, url);
    await waitReady(page);
    const before = await readPage(page, siteRoot);
    const first = await page.screenshot({ type: "jpeg", quality: 60 });
    writeFileSync(join(shotsDir, `${persona.id}-handson-1.jpg`), first);
    record.plan = await model.json("hands-on", [
      { role: "system", content: personaSystem(persona) },
      userMessage(`${PLANS[persona.handsOn]}\n\nWhat the playground shows:\n"""\n${before.text}\n"""`, [first]),
    ]);
    const started = Date.now();
    Object.assign(record, persona.handsOn === "classify" ? await runClassify(page, record.plan) : await runGame(page, record.plan));
    record.seconds = (Date.now() - started) / 1000;
    const after = await readPage(page, siteRoot);
    const shot = await page.screenshot({ type: "jpeg", quality: 60 });
    writeFileSync(join(shotsDir, `${persona.id}-handson-2.jpg`), shot);
    record.screenshots = [`${persona.id}-handson-1.jpg`, `${persona.id}-handson-2.jpg`];
    const turns = record.turns ? `\nThe guesses, and your answers:\n${record.turns.map((t) => `${t.guess} ${t.answer}`).join("\n")}` : "";
    record.reply = await model.json("hands-on", [
      { role: "system", content: personaSystem(persona) },
      userMessage(
        [`You tried the playground. ${SETUP} ${record.did} You expected: ${record.plan.expect}`, turns,
          `It took ${record.seconds.toFixed(1)} seconds.`, `What the page shows now:\n"""\n${after.text}\n"""`,
          `A screenshot is attached. Answer in this shape:\n${REFLECT_SHAPE}`].join("\n"),
        [shot],
      ),
    ]);
    record.reply.confusions = (record.reply.confusions ?? []).map((c) => ({ ...c, onPage: quoteOnPage(c.quote, after.allText) }));
    log(`${persona.id} hands-on: ${record.did}`);
  } catch (failure) {
    record.error = failure.message.split("\n")[0];
    await page.screenshot({ type: "jpeg", quality: 60, path: join(shotsDir, `${persona.id}-handson-error.jpg`) }).catch(() => {});
    log(`${persona.id} hands-on failed: ${record.error}`);
  } finally {
    await context.close();
  }
  return record;
}
