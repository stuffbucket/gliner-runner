// The comprehension prompt: one persona, one page, one question. promptfoo calls this for every test. The page
// text comes from the snapshot the checks stage saved (SITE_REVIEW_PAGES), so the eval grades the site as built.
import { readFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

const here = dirname(fileURLToPath(import.meta.url));
const personas = JSON.parse(readFileSync(join(here, "../personas.json"), "utf8")).personas;

export default function prompt({ vars }) {
  const persona = personas.find((p) => p.id === vars.persona);
  const page = readFileSync(join(process.env.SITE_REVIEW_PAGES, `${vars.page}.txt`), "utf8");
  return [
    {
      role: "system",
      content: [
        `You are ${persona.name}. ${persona.background} What you know: ${persona.knows}`,
        "You have just read one page of a project's website. Answer your own question using only what that page says,",
        "in two to five sentences, in your own words. Do not use anything you know about the project from elsewhere.",
        "If the page does not answer the question, reply exactly: The page does not say.",
      ].join(" "),
    },
    { role: "user", content: `The page:\n"""\n${page}\n"""\n\nMy question: ${vars.question}` },
  ];
}
