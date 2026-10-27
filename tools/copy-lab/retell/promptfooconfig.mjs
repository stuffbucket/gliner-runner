// promptfoo config for the retell test. lab.mjs writes a job file and runs:
//
//   COPY_LAB_RETELL=<out>/<slot>/retell-job.json npx -y promptfoo@0.123.1 eval -c tools/copy-lab/retell/promptfooconfig.mjs --no-cache --no-write -o out.json
//
// Each test is one candidate and one run number; each persona model is a provider, so every candidate is read
// (personas x runs) times. The grader checks the retelling against the slot's facts with one llm-rubric call.
import { readFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

const here = dirname(fileURLToPath(import.meta.url));
const job = JSON.parse(readFileSync(process.env.COPY_LAB_RETELL, "utf8"));

const provider = (id, temperature, maxTokens) => ({
  id: `openai:chat:${id}`,
  label: id,
  // The local API takes any key.
  config: { apiBaseUrl: job.api, apiKey: "local", temperature, max_tokens: maxTokens },
});

export default {
  description: `Retell test for the ${job.slot} slot`,
  prompts: [{ id: `file://${join(here, "prompt.mjs")}`, label: "disinterested-friend" }],
  // Sampled, so that the runs differ: the score is how often a reader comes away with the facts.
  providers: job.personas.map((id) => provider(id, 0.8, 1500)),
  defaultTest: { options: { provider: provider(job.grader, 0, 1500) } },
  tests: job.tests.map((t) => ({
    description: `${t.id} run ${t.run}`,
    vars: { id: t.id, run: t.run, text: t.text },
    assert: [
      { type: "llm-rubric", value: job.rubric, metric: "facts" },
      { type: "javascript", value: "/\"keep_reading\"\\s*:\\s*\"yes/i.test(output)", metric: "keep" },
    ],
  })),
};
