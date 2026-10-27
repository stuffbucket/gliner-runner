/**
 * Request shapes of other decision APIs, translated into ordinary classification requests.
 *
 * - `POST /v1/questions` takes the `state` and typed `questions` of TypeSafe's jev
 *   (https://docs.typesafe.ai/api/system-one.md, OpenAPI 0.2.0) and Microsoft-Decision-1
 *   (https://learn.microsoft.com/en-us/azure/foundry/foundry-models/how-to/use-foundry-models-microsoft-decision,
 *   2026-10-09). Each question becomes one `/v1/infer` request with one task, so questions cannot affect each other.
 * - `POST /v1/classify_text` takes the arguments of GLiNER2's `classify_text` and `batch_classify_text`
 *   (fastino-ai/GLiNER2 v2.1.0, gliner2/inference/runtime.py). All tasks go into one `/v1/infer` request per text,
 *   as GLiNER2 scores them in one pass.
 *
 * The answers are computed only from the probabilities the classification returns, so a client that sends the
 * translated requests itself gets the same numbers. src/gliner_runner/shapes.py is the Python twin of this file and
 * must stay in step with it: tests/fixtures/api-shapes holds cases that both test suites run.
 */
import type { ClassifyTextRequest, InferenceResponse, QuestionsRequest } from "./contracts.js";
import { OrderedObject, type OrderedJson } from "./gliner2/ordered-json.js";

/** A `/v1/infer` request body before validation, in wire form. */
export type WireRequest = Record<string, unknown>;

/** Member of a JSON object that may be an {@link OrderedObject} (backend output) or a plain object. */
function member(value: unknown, key: string): unknown {
  if (value instanceof OrderedObject) return value.get(key);
  if (typeof value === "object" && value !== null && Object.hasOwn(value, key)) {
    return (value as Record<string, unknown>)[key];
  }
  return undefined;
}

/** Keys in their original order (an {@link OrderedObject} keeps integer-like keys where they were sent). */
function keysOf(value: unknown): string[] {
  if (value instanceof OrderedObject) return value.keys();
  return Object.keys(value as Record<string, unknown>);
}

function describedLabels(criteria: unknown, labels: readonly string[]): Record<string, string> | undefined {
  const described: Array<[string, string]> = [];
  for (const label of labels) {
    const description = member(criteria, label);
    if (typeof description === "string") described.push([label, description]);
  }
  return described.length === 0 ? undefined : Object.fromEntries(described);
}

// --- jev and Microsoft-Decision-1 questions --------------------------------------------------------------------

/** The labels a noul (yes/no) question is asked with. Fastino's model card asks yes/no questions the same way. */
export const NOUL_LABELS = ["yes", "no"] as const;

export interface PlannedQuestion {
  /** Question id: the key in `answers`, and the name of the one task in `request`. */
  readonly id: string;
  readonly type: "noul" | "choice" | "score" | "multi_select";
  /** Labels in the order they were sent: option names, or score level descriptions. */
  readonly labels: readonly string[];
  /** The `/v1/infer` request that answers this question. */
  readonly request: WireRequest;
}

/**
 * Plans one classification request per question. `ordered` is the request as an order-preserving parse
 * (`parseOrderedJson`), so question ids and option names keep the order they were sent in; without it the order of
 * the validated object is used.
 */
export function planQuestions(request: QuestionsRequest, ordered?: OrderedJson): PlannedQuestion[] {
  const orderedQuestions = ordered === undefined ? undefined : member(ordered, "questions");
  const ids = orderedQuestions === undefined ? Object.keys(request.questions) : keysOf(orderedQuestions);
  return ids.map((id) => {
    const question = request.questions[id];
    if (question === undefined) throw new Error(`question '${id}' is missing`);
    const criteria = orderedQuestions === undefined ? question.criteria : member(member(orderedQuestions, id), "criteria");
    const type = question.type as PlannedQuestion["type"];
    let labels: string[];
    const task: Record<string, unknown> = {};
    let options: Record<string, unknown> | undefined;
    if (type === "noul") {
      labels = [...NOUL_LABELS];
      task["labels"] = labels;
      const descriptions = criteria === undefined || criteria === null
        ? undefined
        : describedLabels(
            new OrderedObject([
              ["yes", (member(criteria, "true") ?? null) as OrderedJson],
              ["no", (member(criteria, "false") ?? null) as OrderedJson],
            ]),
            labels,
          );
      if (descriptions !== undefined) task["label_descriptions"] = descriptions;
    } else if (type === "score") {
      labels = [...(criteria as string[])];
      task["labels"] = labels;
      task["ordered"] = true;
    } else {
      labels = keysOf(criteria);
      task["labels"] = labels;
      const descriptions = describedLabels(criteria, labels);
      if (descriptions !== undefined) task["label_descriptions"] = descriptions;
      if (type === "multi_select") {
        task["min_labels"] = 0;
        task["max_labels"] = null;
        if (typeof question.threshold === "number") {
          task["threshold"] = question.threshold;
          // The request threshold is the floor for keeping a label as a candidate; it must not drop a label the
          // task threshold would select.
          options = { threshold: question.threshold };
        }
      }
    }
    if (typeof question.instructions === "string") task["instruction"] = question.instructions;
    return {
      id,
      type,
      labels,
      request: {
        model: request.model,
        backend: request.backend,
        precision: request.precision,
        operation: "classify",
        text: request.state,
        schema: { kind: "classification", tasks: { [id]: task } },
        ...(options === undefined ? {} : { options }),
      },
    };
  });
}

/**
 * jev's Choice confidence: how far the top probability is above an even split, on a scale where an even split
 * is 0 and certainty is 1 (https://docs.typesafe.ai/confidence.md, "Choice"). Floored at 0 against rounding.
 */
export function choiceConfidence(probabilities: readonly number[]): number {
  const n = probabilities.length;
  let top = probabilities[0] as number;
  for (const p of probabilities) if (p > top) top = p;
  return Math.max(0, (top - 1 / n) / (1 - 1 / n));
}

/** jev's Score: each level index times its probability, added up in level order. */
export function scoreValue(probabilities: readonly number[]): number {
  let total = 0;
  probabilities.forEach((p, index) => {
    total += index * p;
  });
  return total;
}

/**
 * jev's Score confidence: 1 minus the probability-weighted distance from the most likely level, divided by the
 * same distance for an even spread measured from the middle level; floored at 0
 * (https://docs.typesafe.ai/confidence.md, "Score").
 */
export function scoreConfidence(probabilities: readonly number[]): number {
  const n = probabilities.length;
  let peak = 0;
  probabilities.forEach((p, index) => {
    if (p > (probabilities[peak] as number)) peak = index;
  });
  let spread = 0;
  probabilities.forEach((p, index) => {
    spread += p * Math.abs(index - peak);
  });
  let even = 0;
  for (let index = 0; index < n; index += 1) even += Math.abs(index - (n - 1) / 2);
  even /= n;
  return Math.max(0, 1 - spread / even);
}

/** The selected value and the probability of each of `labels`, from the native output of one task. */
function taskOutput(
  response: InferenceResponse,
  task: string,
  labels: readonly string[],
): { value: unknown; probabilities: number[] } {
  const output = member(response.output, task);
  const probabilities = member(output, "probabilities");
  const values = labels.map((label) => member(probabilities, label));
  if (!values.every((p) => typeof p === "number")) {
    throw new Error(`classification output has no probabilities for task '${task}'`);
  }
  return { value: member(output, "value"), probabilities: values as number[] };
}

/** One jev-shaped answer, computed only from the classification output of the question's task. */
export function questionAnswer(planned: PlannedQuestion, response: InferenceResponse): OrderedObject {
  const { value, probabilities } = taskOutput(response, planned.id, planned.labels);
  const byLabel = new OrderedObject(planned.labels.map((label, index) => [label, probabilities[index] as number]));
  switch (planned.type) {
    case "noul":
      return new OrderedObject([
        ["type", "noul"],
        ["noul", probabilities[0] as number],
      ]);
    case "choice":
      return new OrderedObject([
        ["type", "choice"],
        ["choice", value as string],
        ["confidence", choiceConfidence(probabilities)],
        ["probabilities", byLabel],
      ]);
    case "multi_select":
      return new OrderedObject([
        ["type", "multi_select"],
        ["selected", [...(value as string[])]],
        ["probabilities", byLabel],
      ]);
    case "score":
      return new OrderedObject([
        ["type", "score"],
        ["score", scoreValue(probabilities)],
        ["confidence", scoreConfidence(probabilities)],
        ["legend", new OrderedObject(planned.labels.map((label, index) => [String(index), label]))],
        ["probabilities", new OrderedObject(probabilities.map((p, index) => [String(index), p]))],
      ]);
  }
}

/** The `POST /v1/questions` response body. */
export function questionsResponse(
  request: QuestionsRequest,
  planned: readonly PlannedQuestion[],
  responses: readonly InferenceResponse[],
): OrderedObject {
  let inputTokens = 0;
  for (const response of responses) inputTokens += response.usage.inputTokens;
  return new OrderedObject([
    ["request_id", request.request_id],
    ["model", request.model],
    ["backend", request.backend],
    ["precision", request.precision],
    ["answers", new OrderedObject(planned.map((question, index) => [question.id, questionAnswer(question, responses[index] as InferenceResponse)]))],
    ["usage", new OrderedObject([["input_tokens", inputTokens], ["output_tokens", 0]])],
  ]);
}

// --- GLiNER2 classify_text -------------------------------------------------------------------------------------

export interface PlannedTask {
  readonly name: string;
  readonly multi: boolean;
  /** GLiNER2 sorts multi-label results by probability when `top_k` is set. */
  readonly topK: number | null;
}

export interface ClassifyTextPlan {
  readonly tasks: readonly PlannedTask[];
  /** One `/v1/infer` request per text, in order. */
  readonly requests: readonly WireRequest[];
}

/**
 * Plans GLiNER2 `classify_text` (one `text`) or `batch_classify_text` (`texts`). Every task goes into one
 * classification request per text. `ordered` works as in {@link planQuestions}.
 *
 * - A list of labels, or `labels` as `{name: description}`, become `labels` and `label_descriptions`.
 * - `prompt` becomes `instruction`, and `class_act` becomes `activation`.
 * - `multi_label: true` becomes `min_labels: 1, max_labels: top_k or unbounded` with `threshold: cls_threshold`:
 *   GLiNER2 keeps every label at or above `cls_threshold` and returns the best label alone when none is.
 * - A single-label task keeps the runner defaults (exactly one label, the most probable). GLiNER2 ignores
 *   `cls_threshold` and `top_k` there too.
 * - The top-level `threshold` is GLiNER2's span threshold and has no effect on classification there either.
 */
export function planClassifyText(request: ClassifyTextRequest, ordered?: OrderedJson): ClassifyTextPlan {
  const orderedTasks = ordered === undefined ? undefined : member(ordered, "tasks");
  const names = orderedTasks === undefined ? Object.keys(request.tasks) : keysOf(orderedTasks);
  const planned: PlannedTask[] = [];
  const tasks: Array<[string, Record<string, unknown>]> = [];
  let floor = 0.5;
  for (const name of names) {
    const source = orderedTasks === undefined ? request.tasks[name] : member(orderedTasks, name);
    const config = Array.isArray(source) ? { labels: source } : source;
    const rawLabels = member(config, "labels");
    const task: Record<string, unknown> = {};
    if (Array.isArray(rawLabels)) {
      task["labels"] = [...(rawLabels as string[])];
    } else {
      const labels = keysOf(rawLabels);
      task["labels"] = labels;
      const descriptions = describedLabels(rawLabels, labels);
      if (descriptions !== undefined) task["label_descriptions"] = descriptions;
    }
    const multi = member(config, "multi_label") === true;
    const topK = (member(config, "top_k") ?? null) as number | null;
    if (multi) {
      const labelCount = (task["labels"] as string[]).length;
      task["min_labels"] = 1;
      task["max_labels"] = topK === null ? null : Math.min(topK, labelCount);
      const threshold = member(config, "cls_threshold");
      if (typeof threshold === "number") {
        task["threshold"] = threshold;
        floor = Math.min(floor, threshold);
      }
    }
    const activation = member(config, "class_act");
    if (activation === "sigmoid" || activation === "softmax") task["activation"] = activation;
    const prompt = member(config, "prompt");
    if (typeof prompt === "string") task["instruction"] = prompt;
    planned.push({ name, multi, topK: multi ? topK : null });
    tasks.push([name, task]);
  }
  const texts = request.texts ?? [request.text as string];
  const requests = texts.map((text) => ({
    model: request.model,
    backend: request.backend,
    precision: request.precision,
    operation: "classify",
    text,
    schema: { kind: "classification", tasks: Object.fromEntries(tasks) },
    // The request threshold is the floor for keeping a label as a candidate; it must not drop a label that a task's
    // cls_threshold would select.
    ...(floor < 0.5 ? { options: { threshold: floor } } : {}),
  }));
  return { tasks: planned, requests };
}

/** One GLiNER2-shaped result (`format_results=True`), computed from the classification output. */
export function classifyTextResult(
  plan: ClassifyTextPlan,
  response: InferenceResponse,
  includeConfidence: boolean,
): OrderedObject {
  return new OrderedObject(
    plan.tasks.map((task): readonly [string, OrderedJson] => {
      const output = member(response.output, task.name);
      const probabilities = member(output, "probabilities");
      const probability = (label: string): number => {
        const p = member(probabilities, label);
        if (typeof p !== "number") throw new Error(`classification output has no probabilities for task '${task.name}'`);
        return p;
      };
      const value = member(output, "value");
      if (!task.multi) {
        const label = value as string;
        return [
          task.name,
          includeConfidence ? new OrderedObject([["label", label], ["confidence", probability(label)]]) : label,
        ];
      }
      let labels = [...(value as string[])];
      if (task.topK !== null) {
        // Python's sorted() is stable: equal probabilities keep the declared order.
        labels = labels
          .map((label, index) => ({ label, index, p: probability(label) }))
          .sort((a, b) => b.p - a.p || a.index - b.index)
          .map((entry) => entry.label);
      }
      return [
        task.name,
        includeConfidence
          ? labels.map((label) => new OrderedObject([["label", label], ["confidence", probability(label)]]))
          : labels,
      ];
    }),
  );
}

/** The `POST /v1/classify_text` response body: `result` for `text`, `results` for `texts`. */
export function classifyTextResponse(
  request: ClassifyTextRequest,
  plan: ClassifyTextPlan,
  responses: readonly InferenceResponse[],
): OrderedObject {
  let inputTokens = 0;
  for (const response of responses) inputTokens += response.usage.inputTokens;
  const results = responses.map((response) => classifyTextResult(plan, response, request.include_confidence));
  return new OrderedObject([
    ["request_id", request.request_id],
    ["model", request.model],
    ["backend", request.backend],
    ["precision", request.precision],
    (request.texts ?? null) === null ? ["result", results[0] as OrderedObject] : ["results", results],
    ["usage", new OrderedObject([["inputTokens", inputTokens], ["outputTokens", 0]])],
  ]);
}
