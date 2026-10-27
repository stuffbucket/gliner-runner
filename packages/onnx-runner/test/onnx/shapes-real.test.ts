import assert from "node:assert/strict";
import test from "node:test";

import {
  ClassifyTextRequestSchema,
  QuestionsRequestSchema,
  choiceConfidence,
  planClassifyText,
  planQuestions,
  scoreConfidence,
  scoreValue,
} from "../../src/index.js";
import { parseOrderedJson } from "../../src/gliner2/ordered-json.js";

/**
 * End-to-end test of the translated shapes against a running runner with real models. Runs only when
 * GLINER_RUNNER_SHAPES_URL names the server, for example:
 *
 *   GLINER_RUNNER_MODELS_DIR=dist/onnx-models node packages/onnx-runner/dist/src/cli.js serve --port 8104
 *   GLINER_RUNNER_SHAPES_URL=http://127.0.0.1:8104 node --test packages/onnx-runner/dist/test/onnx/shapes-real.test.js
 *
 * Every answer must equal, number for number, what the server returns for the translated `/v1/infer` requests sent
 * one by one. The examples are the ones in the jev documentation (https://docs.typesafe.ai/primitives/noul.md,
 * choice.md and score.md) and the GLiNER2.5-Decide model card (fastino/GLiNER2.5-Decide at
 * 9d1bfb848cd16d93ee56ccbc59cd92bf0f284d90).
 */
const BASE = process.env["GLINER_RUNNER_SHAPES_URL"];
const MODEL = process.env["GLINER_RUNNER_SHAPES_MODEL"] ?? "decide-340m";
const skip = BASE === undefined ? "set GLINER_RUNNER_SHAPES_URL to a running runner to run" : false;

async function post(path: string, body: unknown): Promise<{ status: number; body: any }> {
  const response = await fetch(`${BASE}${path}`, {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify(body),
  });
  return { status: response.status, body: await response.json() };
}

const explicit = { model: MODEL, backend: "onnx", precision: "fp32" };

function questions(state: string): Record<string, unknown> {
  return {
    ...explicit,
    state,
    questions: {
      is_human_escalation: { type: "noul", instructions: "Is the customer asking for a human agent?" },
      is_repeat_contact: {
        type: "noul",
        instructions: "Has the customer contacted support about this before?",
        criteria: {
          true: "Mentions a prior attempt, ticket, or that they have asked before",
          false: "No sign of any previous contact",
        },
      },
      department: {
        type: "choice",
        instructions: "Which team should handle this?",
        criteria: {
          returns: "Exchanges, wrong or damaged items",
          shipping: "Delivery status, delays, lost packages",
          billing: "Charges, invoices, payment problems",
        },
      },
      bug_severity: {
        type: "score",
        instructions: "How severe is the reported issue?",
        criteria: [
          "Cosmetic; no impact to functionality",
          "Broken or degraded feature, but workaround exists",
          "Blocking issue; no workaround exists",
        ],
      },
      topics: {
        type: "multi_select",
        instructions: "Which topics does the message raise?",
        criteria: { refund: null, "a human agent": "The customer wants to talk to a person", delivery: null },
        threshold: 0.4,
      },
    },
  };
}

test("real runner: /v1/questions equals the translated /v1/infer requests", { skip, timeout: 5 * 60_000 }, async (t) => {
  const probe = await post("/v1/questions", questions("Thanks, that fixed it!"));
  if (probe.status === 409) {
    t.skip(`${MODEL} is not installed: ${JSON.stringify(probe.body)}`);
    return;
  }
  for (const state of [
    "Thanks, that fixed it!",
    "I have asked three times now. Can I please just talk to a real person?",
    "My running shoes arrived in the wrong size. Can I swap them for a size 10?",
    "Nobody on our team can log in since this morning. We get a 500 error on every attempt.",
  ]) {
    const request = questions(state);
    const reply = await post("/v1/questions", request);
    assert.equal(reply.status, 200, JSON.stringify(reply.body));
    const raw = JSON.stringify(request);
    const parsed = QuestionsRequestSchema.parse(JSON.parse(raw));
    const planned = planQuestions(parsed, parseOrderedJson(raw));
    let tokens = 0;
    for (const question of planned) {
      const byHand = await post("/v1/infer", question.request);
      assert.equal(byHand.status, 200, JSON.stringify(byHand.body));
      tokens += byHand.body.usage.inputTokens;
      const task = byHand.body.output[question.id];
      const answer = reply.body.answers[question.id];
      const probabilities = question.labels.map((label) => task.probabilities[label] as number);
      switch (question.type) {
        case "noul":
          assert.equal(answer.noul, task.probabilities.yes);
          break;
        case "choice":
          assert.equal(answer.choice, task.value);
          assert.equal(answer.confidence, choiceConfidence(probabilities));
          assert.deepEqual(answer.probabilities, task.probabilities);
          break;
        case "multi_select":
          assert.deepEqual(answer.selected, task.value);
          assert.deepEqual(answer.probabilities, task.probabilities);
          break;
        case "score":
          assert.deepEqual(Object.values(answer.probabilities), probabilities);
          assert.equal(answer.score, scoreValue(probabilities));
          assert.equal(answer.confidence, scoreConfidence(probabilities));
          assert.deepEqual(Object.values(answer.legend), question.labels);
          break;
      }
    }
    assert.deepEqual(reply.body.usage, { input_tokens: tokens, output_tokens: 0 });
  }

  // What the answers mean, on the jev documentation's clearest examples. Only decide-340m is held to them: the
  // translation is checked above for every model, but decide-multilingual answers 0.31 for "talk to a real person".
  if (MODEL !== "decide-340m") return;
  const thanks = await post("/v1/questions", questions("Thanks, that fixed it!"));
  const person = await post(
    "/v1/questions",
    questions("I have asked three times now. Can I please just talk to a real person?"),
  );
  assert.ok(thanks.body.answers.is_human_escalation.noul < 0.5);
  assert.ok(person.body.answers.is_human_escalation.noul > 0.5);
  const shoes = await post(
    "/v1/questions",
    questions("My running shoes arrived in the wrong size. Can I swap them for a size 10?"),
  );
  assert.equal(shoes.body.answers.department.choice, "returns");
});

test("real runner: /v1/classify_text equals the translated /v1/infer request", { skip, timeout: 5 * 60_000 }, async (t) => {
  const request = {
    ...explicit,
    text: "The treaty was signed in Paris in 1992. It entered into force the following year, after the last signatory ratified it.",
    tasks: {
      answer: { labels: ["yes", "no"], prompt: "Did the treaty enter into force in 1992?" },
      topic: ["law", "sports", "cooking"],
      aspects: {
        labels: { history: "Past events", travel: null, politics: "Governments and treaties", food: null },
        multi_label: true,
        cls_threshold: 0.4,
        top_k: 2,
      },
    },
    include_confidence: true,
  };
  const reply = await post("/v1/classify_text", request);
  if (reply.status === 409) {
    t.skip(`${MODEL} is not installed: ${JSON.stringify(reply.body)}`);
    return;
  }
  assert.equal(reply.status, 200, JSON.stringify(reply.body));
  const raw = JSON.stringify(request);
  const plan = planClassifyText(ClassifyTextRequestSchema.parse(JSON.parse(raw)), parseOrderedJson(raw));
  const byHand = await post("/v1/infer", plan.requests[0]);
  assert.equal(byHand.status, 200, JSON.stringify(byHand.body));
  const output = byHand.body.output;
  assert.deepEqual(reply.body.result.answer, {
    label: output.answer.value,
    confidence: output.answer.probabilities[output.answer.value],
  });
  assert.deepEqual(reply.body.result.topic, {
    label: output.topic.value,
    confidence: output.topic.probabilities[output.topic.value],
  });
  const selected = [...(output.aspects.value as string[])].sort(
    (a, b) => output.aspects.probabilities[b] - output.aspects.probabilities[a],
  );
  assert.deepEqual(
    reply.body.result.aspects,
    selected.map((label) => ({ label, confidence: output.aspects.probabilities[label] })),
  );
  assert.ok(reply.body.result.aspects.length <= 2);
  assert.deepEqual(reply.body.usage, byHand.body.usage);

  const batch = await post("/v1/classify_text", {
    ...explicit,
    texts: ["I was charged twice for my last order.", "Where is my parcel? It is two weeks late."],
    tasks: { department: ["billing", "support", "shipping"] },
  });
  assert.equal(batch.status, 200, JSON.stringify(batch.body));
  assert.deepEqual(batch.body.results, [{ department: "billing" }, { department: "shipping" }]);
});

test("real runner: untranslatable requests are 422 with the reason", { skip }, async () => {
  const box = await post("/v1/questions", {
    ...explicit,
    state: "A kitchen",
    questions: { mug: { type: "box", instructions: "the coffee mug in `photo`" } },
  });
  assert.equal(box.status, 422);
  assert.match(box.body.detail[0].msg, /read text only/);
  const examples = await post("/v1/classify_text", {
    ...explicit,
    text: "x",
    tasks: { t: { labels: ["a", "b"], examples: [["text", "a"]] } },
  });
  assert.equal(examples.status, 422);
  assert.match(examples.body.detail[0].msg, /few-shot examples/);
});
