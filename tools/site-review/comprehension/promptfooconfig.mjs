// promptfoo config for the comprehension stage. review.mjs runs it with:
//
//   SITE_REVIEW_PAGES=<out>/pages SITE_REVIEW_MODEL=claude-sonnet-5 SITE_REVIEW_API=http://localhost:4141/v1 \
//     npx -y promptfoo@0.123.1 eval -c tools/site-review/comprehension/promptfooconfig.mjs --no-cache --no-write -o out.json
//
// The answering model and the grader are the same OpenAI-compatible endpoint. A test passes when the answer
// matches the rubric (written from the page and the code) and contains the facts in `mustMatch`.
import { readFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

const here = dirname(fileURLToPath(import.meta.url));
const { questions } = JSON.parse(readFileSync(join(here, "questions.json"), "utf8"));

const api = process.env.SITE_REVIEW_API ?? "http://localhost:4141/v1";
const model = process.env.SITE_REVIEW_MODEL ?? "claude-sonnet-5";
const provider = (id, maxTokens) => ({
  id: `openai:chat:${id}`,
  label: id,
  // The local API takes any key.
  config: { apiBaseUrl: api, apiKey: "local", temperature: 0, max_tokens: maxTokens },
});

// The test is whether the page taught the reader the facts that matter, not whether they recited all of it.
const GRADING = "Pass the answer if it states every required fact correctly. Do not fail it for leaving out optional details or for adding correct ones. Fail it if it states something the page contradicts.";

export default {
  description: "Can a reader learn what each key page should teach?",
  prompts: [{ id: `file://${join(here, "prompt.mjs")}`, label: "persona-reads-page" }],
  providers: [provider(model, 600)],
  defaultTest: { options: { provider: provider(process.env.SITE_REVIEW_GRADER ?? model, 800) } },
  tests: questions.map((q) => ({
    description: q.id,
    vars: { page: q.page, persona: q.persona, question: q.question },
    metadata: { page: q.page, persona: q.persona },
    assert: [
      { type: "llm-rubric", value: `${GRADING} ${q.rubric}`, metric: "rubric" },
      ...q.mustMatch.map((pattern) => ({ type: "javascript", value: `new RegExp(${JSON.stringify(pattern)}, "i").test(output)`, metric: "facts" })),
    ],
  })),
};
