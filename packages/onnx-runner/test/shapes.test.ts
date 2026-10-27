import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test, { type TestContext } from "node:test";

import {
  ClassifyTextRequestSchema,
  InferenceRequestSchema,
  QuestionsRequestSchema,
  SHAPE_MESSAGES,
  choiceConfidence,
  classifyTextResponse,
  planClassifyText,
  planQuestions,
  questionsResponse,
  scoreConfidence,
  scoreValue,
  taskOrderOf,
  type BackendResult,
  type ClassificationTask,
  type InferenceRequest,
  type InferenceResponse,
} from "../src/index.js";
import { OrderedObject, parseOrderedJson, serializeOrdered, type OrderedJson } from "../src/gliner2/ordered-json.js";
import { parseJsonBody } from "../src/validation.js";
import { FakeFactory, startServer, type TestServer } from "./helpers.js";

/** Cases shared with tests/test_shapes.py; written by scripts/api_shapes_fixture.py. */
const FIXTURE = parseOrderedJson(
  readFileSync(new URL("../../../../tests/fixtures/api-shapes/cases.json", import.meta.url), "utf8"),
) as OrderedObject;

function cases(section: string): OrderedObject[] {
  return FIXTURE.get(section) as OrderedObject[];
}

const plain = (value: OrderedJson | undefined): unknown => JSON.parse(serializeOrdered(value as OrderedJson));

function inferenceResponse(entry: OrderedObject): InferenceResponse {
  return {
    request_id: "955b971b-4295-4e39-9e45-9b404e32bb6e",
    model: "decide-340m",
    backend: "onnx",
    precision: "fp32",
    output: entry.get("output") as OrderedObject,
    error: null,
    timing: { queue_ms: 0, inference_ms: 0 },
    usage: { inputTokens: entry.get("inputTokens") as number, outputTokens: 0 },
  };
}

for (const fixture of cases("questions")) {
  test(`questions translate like the shared fixture: ${fixture.get("name") as string}`, () => {
    const raw = serializeOrdered(fixture.get("request") as OrderedJson);
    const parsed = parseJsonBody(QuestionsRequestSchema, raw);
    assert.ok(parsed.success, JSON.stringify(parsed));
    const planned = planQuestions(parsed.data, parseOrderedJson(raw));
    assert.deepEqual(
      planned.map((question) => question.request),
      plain(fixture.get("requests")),
    );
    for (const question of planned) InferenceRequestSchema.parse(question.request);
    const responses = (fixture.get("outputs") as OrderedObject[]).map(inferenceResponse);
    // Same keys in the same order and the same numbers as the Python translation.
    assert.equal(
      serializeOrdered(questionsResponse(parsed.data, planned, responses)),
      serializeOrdered(fixture.get("response") as OrderedJson),
    );
  });
}

for (const fixture of cases("classify_text")) {
  test(`classify_text translates like the shared fixture: ${fixture.get("name") as string}`, () => {
    const raw = serializeOrdered(fixture.get("request") as OrderedJson);
    const parsed = parseJsonBody(ClassifyTextRequestSchema, raw);
    assert.ok(parsed.success, JSON.stringify(parsed));
    const plan = planClassifyText(parsed.data, parseOrderedJson(raw));
    assert.deepEqual(plan.requests, plain(fixture.get("requests")));
    const expectedOrder = ((fixture.get("requests") as OrderedObject[])[0]?.get("schema") as OrderedObject).get("tasks");
    assert.deepEqual(
      plan.tasks.map((task) => task.name),
      (expectedOrder as OrderedObject).keys(),
    );
    const responses = (fixture.get("outputs") as OrderedObject[]).map(inferenceResponse);
    assert.equal(
      serializeOrdered(classifyTextResponse(parsed.data, plan, responses)),
      serializeOrdered(fixture.get("response") as OrderedJson),
    );
  });
}

test("untranslatable requests fail validation with the shared messages", () => {
  for (const fixture of cases("invalid")) {
    const body = serializeOrdered(fixture.get("body") as OrderedJson);
    const parsed =
      fixture.get("endpoint") === "questions"
        ? parseJsonBody(QuestionsRequestSchema, body)
        : parseJsonBody(ClassifyTextRequestSchema, body);
    assert.equal(parsed.success, false);
    const found = parsed.success ? [] : parsed.errors.map((error) => JSON.stringify([error.loc, error.msg]));
    assert.ok(
      found.includes(JSON.stringify([plain(fixture.get("loc")), fixture.get("msg")])),
      `${fixture.get("msg") as string} not in ${found.join("\n")}`,
    );
  }
});

test("the confidence formulas match the jev documentation", () => {
  // https://docs.typesafe.ai/confidence.md and https://docs.typesafe.ai/primitives/score.md
  const close = (actual: number, expected: number): void => assert.ok(Math.abs(actual - expected) < 1e-12, `${actual}`);
  close(choiceConfidence([0.6, 0.3, 0.1]), 0.4);
  close(choiceConfidence([0.6, 0.2, 0.2]), 0.4);
  assert.equal(choiceConfidence([0.5, 0.5]), 0);
  assert.equal(choiceConfidence([1, 0, 0]), 1);
  close(scoreConfidence([0, 0.5, 0.5]), 0.25);
  assert.equal(scoreConfidence([0.5, 0, 0.5]), 0);
  close(scoreConfidence([0, 0.57, 0.43]), 0.355);
  assert.equal(scoreConfidence([0, 0, 1]), 1);
  close(scoreValue([0, 0.57, 0.43]), 1.43);
});

test("classify_text keeps integer-like task names in the order they were sent", () => {
  const raw =
    '{"model":"decide-340m","backend":"onnx","precision":"fp32","text":"x",' +
    '"tasks":{"b":["p","q"],"10":["p","q"],"2":["p","q"]}}';
  const parsed = parseJsonBody(ClassifyTextRequestSchema, raw);
  assert.ok(parsed.success);
  const plan = planClassifyText(parsed.data, parseOrderedJson(raw));
  assert.deepEqual(plan.tasks.map((task) => task.name), ["b", "10", "2"]);
});

/** Probabilities that depend on the text and each task's labels, so a wrong translation shows up as other numbers. */
function scripted(requests: readonly InferenceRequest[]): BackendResult[] {
  return requests.map((request) => {
    const schema = request.schema;
    if (schema.kind !== "classification") throw new Error("expected a classification request");
    const entries: Array<readonly [string, OrderedJson]> = [];
    for (const name of taskOrderOf(schema.tasks)) {
      const task: ClassificationTask | undefined = schema.tasks[name];
      assert.ok(task !== undefined);
      let raw = task.labels.map((_, index) => (((index * 37 + request.text.length + name.length) % 10) + 0.5) / 10.5);
      const exclusive = task.min_labels === 1 && task.max_labels === 1;
      if (exclusive) {
        let total = 0;
        for (const p of raw) total += p;
        raw = raw.map((p) => p / total);
      }
      const probability = (label: string): number => raw[task.labels.indexOf(label)] as number;
      const best = task.labels.reduce((a, b) => (probability(b) > probability(a) ? b : a));
      let value: OrderedJson;
      if (exclusive) {
        value = best;
      } else {
        let selected = task.labels.filter((label) => probability(label) > task.threshold);
        if (selected.length < task.min_labels) selected = [best];
        if (task.max_labels !== null) {
          const kept = [...selected].sort((a, b) => probability(b) - probability(a)).slice(0, task.max_labels);
          selected = task.labels.filter((label) => kept.includes(label));
        }
        value = selected;
      }
      entries.push([
        name,
        new OrderedObject([
          ["value", value],
          ["confidence", 0.5],
          ["probabilities", new OrderedObject(task.labels.map((label) => [label, probability(label)] as const))],
        ]),
      ]);
    }
    return { output: new OrderedObject(entries), usage: { inputTokens: request.text.length, outputTokens: 0 as const } };
  });
}

async function scriptedServer(t: TestContext): Promise<TestServer> {
  const factory = new FakeFactory();
  factory.configure = (backend) => {
    backend.inferBatch = async (requests) => scripted(requests);
  };
  const server = await startServer({ factory });
  t.after(() => server.server.close());
  return server;
}

const QUESTIONS = {
  model: "decide-340m",
  backend: "onnx",
  precision: "fp32",
  state: "I have asked three times now. Can I please just talk to a real person?",
  questions: {
    wants_human: { type: "noul", instructions: "Is the customer asking for a person?" },
    team: {
      type: "choice",
      instructions: "Which team should handle this?",
      criteria: { returns: "Exchanges", shipping: null, billing: "Charges" },
    },
    severity: { type: "score", criteria: ["Calm", "Frustrated", "Very angry"] },
    topics: { type: "multi_select", criteria: { refund: null, person: null, delivery: null }, threshold: 0.3 },
  },
};

test("POST /v1/questions answers exactly like the classification requests sent by hand", async (t) => {
  const { post } = await scriptedServer(t);
  const response = await post("/v1/questions", QUESTIONS);
  assert.equal(response.status, 200);
  const body = (await response.json()) as Record<string, any>;
  const raw = JSON.stringify(QUESTIONS);
  const parsed = parseJsonBody(QuestionsRequestSchema, raw);
  assert.ok(parsed.success);
  const planned = planQuestions(parsed.data, parseOrderedJson(raw));
  const byHand: InferenceResponse[] = [];
  for (const question of planned) {
    const reply = await post("/v1/infer", question.request);
    assert.equal(reply.status, 200);
    byHand.push((await reply.json()) as InferenceResponse);
  }
  const expected = JSON.parse(serializeOrdered(questionsResponse(parsed.data, planned, byHand))) as Record<string, any>;
  assert.deepEqual(body["answers"], expected["answers"]);
  assert.deepEqual(Object.keys(body["answers"]), ["wants_human", "team", "severity", "topics"]);
  assert.deepEqual(body["usage"], { input_tokens: 4 * QUESTIONS.state.length, output_tokens: 0 });
  const first = byHand[0]?.output as Record<string, any>;
  assert.equal(body["answers"]["wants_human"]["noul"], first["wants_human"]["probabilities"]["yes"]);
  assert.deepEqual(Object.keys(body).sort(), ["answers", "backend", "model", "precision", "request_id", "usage"]);
});

test("POST /v1/classify_text answers exactly like the classification requests sent by hand", async (t) => {
  const { post } = await scriptedServer(t);
  const payload = {
    model: "decide-340m",
    backend: "onnx",
    precision: "fp32",
    texts: ["The battery is weak.", "The delivery was slow and the price too high."],
    tasks: {
      sentiment: ["positive", "negative", "neutral"],
      aspects: { labels: ["battery", "price", "delivery"], multi_label: true, cls_threshold: 0.4, top_k: 2 },
    },
    include_confidence: true,
  };
  const response = await post("/v1/classify_text", payload);
  assert.equal(response.status, 200);
  const body = (await response.json()) as Record<string, any>;
  const raw = JSON.stringify(payload);
  const parsed = parseJsonBody(ClassifyTextRequestSchema, raw);
  assert.ok(parsed.success);
  const plan = planClassifyText(parsed.data, parseOrderedJson(raw));
  const byHand = (await (await post("/v1/batch", { requests: plan.requests })).json()) as { responses: InferenceResponse[] };
  const expected = JSON.parse(serializeOrdered(classifyTextResponse(parsed.data, plan, byHand.responses))) as Record<string, any>;
  assert.deepEqual(body["results"], expected["results"]);
  assert.equal("result" in body, false);
  assert.deepEqual(body["usage"], { inputTokens: payload.texts.join("").length, outputTokens: 0 });
});

test("an untranslatable question is a 422 that says what to do instead", async (t) => {
  const { post } = await scriptedServer(t);
  const response = await post("/v1/questions", { ...QUESTIONS, questions: { mug: { type: "box", instructions: "the mug" } } });
  assert.equal(response.status, 422);
  const detail = ((await response.json()) as { detail: Array<{ loc: unknown; msg: string }> }).detail;
  assert.deepEqual(detail[0]?.loc, ["body", "questions", "mug"]);
  assert.equal(detail[0]?.msg, `Value error, ${SHAPE_MESSAGES.box}`);
});

test("openapi lists the translated shapes at API version 1.1.0", async (t) => {
  const { get } = await scriptedServer(t);
  const openapi = (await (await get("/openapi.json")).json()) as { info: { version: string }; paths: Record<string, unknown> };
  assert.equal(openapi.info.version, "1.1.0");
  assert.ok("/v1/questions" in openapi.paths);
  assert.ok("/v1/classify_text" in openapi.paths);
});
