
import { z } from "zod";

import type { OrderedObject } from "./gliner2/ordered-json.js";

export type JsonValue =
  | string
  | number
  | boolean
  | null
  | JsonValue[]
  | { [key: string]: JsonValue };

/** A backend result value; ordered objects keep Python key order on the wire (see `stringifyWire`). */
export type BackendOutput = JsonValue | OrderedObject;

/**
 * Prefix that keeps `__proto__` representable while zod builds a record. `z.record` skips a `__proto__` key (a plain
 * object cannot hold one through assignment), but Python accepts it as an ordinary string key. The key is escaped
 * before the record is parsed, so it is validated like any other key, and restored into a null-prototype object
 * afterwards. Escaping is injective: only `__proto__` and keys that already start with the prefix get it.
 */
const KEY_ESCAPE = "\u0001";

const needsEscape = (key: string): boolean => key === "__proto__" || key.startsWith(KEY_ESCAPE);

/** Key as it appears in the request body (inverse of the escaping applied before record validation). */
export const unescapeRecordKey = (key: string): string =>
  key.startsWith(KEY_ESCAPE) ? key.slice(KEY_ESCAPE.length) : key;

function isPlainJsonObject(value: unknown): value is Record<string, unknown> {
  if (typeof value !== "object" || value === null || Array.isArray(value)) {
    return false;
  }
  const prototype: unknown = Object.getPrototypeOf(value);
  return prototype === Object.prototype || prototype === null;
}

function escapeRecordKeys(input: unknown): unknown {
  if (!isPlainJsonObject(input) || !Object.keys(input).some(needsEscape)) {
    return input;
  }
  const escaped: Record<string, unknown> = Object.create(null) as Record<string, unknown>;
  for (const key of Object.keys(input)) {
    escaped[needsEscape(key) ? KEY_ESCAPE + key : key] = input[key];
  }
  return escaped;
}

function restoreRecordKeys<T>(record: Record<string, T>): Record<string, T> {
  const restored: Record<string, T> = {};
  for (const escaped of Object.keys(record)) {
    const key = unescapeRecordKey(escaped);
    const value = record[escaped] as T;
    if (key === "__proto__") {
      // Plain assignment would replace the prototype; define an own property instead.
      Object.defineProperty(restored, key, { value, enumerable: true, writable: true, configurable: true });
    } else {
      restored[key] = value;
    }
  }
  return restored;
}

/**
 * `z.record(z.string(), value)` that keeps `__proto__` keys. The output is a plain object in which `__proto__` is
 * an ordinary own data property (`Object.keys`, spreading and `JSON.stringify` see it). The JSON Schema of the
 * input is that of the plain record.
 */
function dictionary<T extends z.ZodType>(value: T) {
  return z
    .preprocess(escapeRecordKeys, z.record(z.string(), value))
    .transform((record) => restoreRecordKeys(record));
}

export const JsonValueSchema: z.ZodType<JsonValue> = z.lazy(() =>
  z.union([
    z.string(),
    z.number(),
    z.boolean(),
    z.null(),
    z.array(JsonValueSchema),
    dictionary(JsonValueSchema),
  ]),
);

export const BackendNameSchema = z.enum(["pytorch", "mlx", "onnx"]);
export type BackendName = z.infer<typeof BackendNameSchema>;

export const PrecisionSchema = z.enum(["fp32", "fp16", "bf16", "int8", "int4"]);
export type Precision = z.infer<typeof PrecisionSchema>;

export const OperationSchema = z.enum([
  "classify",
  "extract_entities",
  "extract_relations",
  "extract_structured",
]);
export type Operation = z.infer<typeof OperationSchema>;

const isWhitespaceOnly = (value: string): boolean => /^\p{White_Space}*$/u.test(value);

const nonBlankString = z
  .string()
  .min(1)
  .refine((value) => value.length === 0 || !isWhitespaceOnly(value), { error: "must not be blank" });

function uniqueLabels(message: string) {
  return z
    .array(z.string())
    .min(1)
    .refine((labels) => new Set(labels).size === labels.length, { error: message });
}

export const ClassificationTaskSchema = z
  .strictObject({
    labels: uniqueLabels("classification labels must be unique"),
    label_descriptions: dictionary(z.string().nullable()).nullable().default(null),
    min_labels: z.number().int().min(0).default(1),
    max_labels: z.number().int().min(0).nullable().default(1),
    ordered: z.boolean().default(false),
    threshold: z.number().gt(0).lt(1).default(0.5),
    activation: z.enum(["auto", "sigmoid", "softmax"]).default("auto"),
    temperature: z.number().gt(0).default(1),
    default: z.string().nullable().default(null),
    instruction: z.string().nullable().default(null),
  })
  .superRefine((task, context) => {
    if (task.label_descriptions === null) {
      return;
    }
    const entries = Object.entries(task.label_descriptions);
    const unknown = entries.map(([label]) => label).filter((label) => !task.labels.includes(label));
    if (unknown.length > 0) {
      context.addIssue({
        code: "custom",
        message: `label description keys must be declared labels: ${unknown.sort().join(", ")}`,
      });
      return;
    }
    const blank = entries
      .filter(([, description]) => description !== null && isWhitespaceOnly(description))
      .map(([label]) => label);
    if (blank.length > 0) {
      context.addIssue({
        code: "custom",
        message: `label descriptions must be non-blank strings or null: ${blank.sort().join(", ")}`,
      });
    }
  });
export type ClassificationTask = z.output<typeof ClassificationTaskSchema>;

const nonEmptyRecord = <T extends z.ZodType>(value: T) =>
  dictionary(value).check((context) => {
    if (Object.keys(context.value).length === 0) {
      context.issues.push({
        code: "too_small",
        origin: "record",
        minimum: 1,
        inclusive: true,
        input: context.value,
      });
    }
  });

export const ClassificationSchema = z.strictObject({
  kind: z.literal("classification"),
  tasks: nonEmptyRecord(ClassificationTaskSchema),
  constraints: z.array(dictionary(JsonValueSchema)).default([]),
});

export const EntitySchema = z.strictObject({
  kind: z.literal("entities"),
  labels: uniqueLabels("entity labels must be unique"),
});

export const RelationSchema = z.strictObject({
  kind: z.literal("relations"),
  relation_types: z.array(z.string()).min(1),
});

export const StructuredSchema = z.strictObject({
  kind: z.literal("structured"),
  schema: nonEmptyRecord(JsonValueSchema),
});

export const InferenceSchema = z.discriminatedUnion("kind", [
  ClassificationSchema,
  EntitySchema,
  RelationSchema,
  StructuredSchema,
]);
export type InferenceSchema = z.output<typeof InferenceSchema>;
export type ClassificationRequestSchema = z.output<typeof ClassificationSchema>;

export const InferenceOptionsSchema = z.strictObject({
  threshold: z.number().min(0).max(1).default(0.5),
  output_format: z.enum(["native"]).default("native"),
  include_confidence: z.boolean().default(true),
});
export type InferenceOptions = z.output<typeof InferenceOptionsSchema>;

const expectedSchemaKind = {
  classify: "classification",
  extract_entities: "entities",
  extract_relations: "relations",
  extract_structured: "structured",
} as const satisfies Record<Operation, InferenceSchema["kind"]>;

export const InferenceRequestSchema = z
  .strictObject({
    request_id: z
      .guid()
      .transform((value) => value.toLowerCase())
      .default(() => globalThis.crypto.randomUUID()),
    model: nonBlankString,
    backend: BackendNameSchema,
    precision: PrecisionSchema,
    operation: OperationSchema,
    text: nonBlankString,
    schema: InferenceSchema,
    options: InferenceOptionsSchema.prefault({}),
  })
  .superRefine((request, context) => {
    const expected = expectedSchemaKind[request.operation];
    if (request.schema.kind !== expected) {
      context.addIssue({
        code: "custom",
        path: ["schema"],
        message: `${request.operation} requires a ${expected} schema`,
      });
    }
  });
export type InferenceRequest = z.output<typeof InferenceRequestSchema>;

export const BatchRequestSchema = z.strictObject({
  requests: z.array(InferenceRequestSchema).min(1).max(256),
});

/**
 * `POST /v1/questions`: the request shape of TypeSafe's jev and Microsoft-Decision-1 (`state` plus typed
 * `questions`), with this runner's explicit model, backend and precision. `shapes.ts` turns each question into one
 * ordinary classification request. The checks live in `questionProblem` so that the Python server reports the same
 * messages.
 */
export const QuestionSchema = z
  .strictObject({
    type: z.string(),
    instructions: JsonValueSchema.optional(),
    criteria: JsonValueSchema.optional(),
    threshold: z.number().gt(0).lt(1).nullable().optional(),
  })
  .superRefine((question, context) => {
    const problem = questionProblem(question);
    if (problem !== null) {
      context.addIssue({ code: "custom", message: problem });
    }
  });
export type Question = z.output<typeof QuestionSchema>;

export const QuestionsRequestSchema = z
  .strictObject({
    request_id: z
      .guid()
      .transform((value) => value.toLowerCase())
      .default(() => globalThis.crypto.randomUUID()),
    model: nonBlankString,
    backend: BackendNameSchema,
    precision: PrecisionSchema,
    state: JsonValueSchema,
    questions: nonEmptyRecord(QuestionSchema),
    assets: JsonValueSchema.optional(),
  })
  .superRefine((request, context) => {
    const state = stateProblem(request.state);
    if (state !== null) {
      context.addIssue({ code: "custom", path: ["state"], message: state });
    }
    if (request.assets !== undefined) {
      context.addIssue({ code: "custom", path: ["assets"], message: SHAPE_MESSAGES.assets });
    }
  });
export type QuestionsRequest = z.output<typeof QuestionsRequestSchema>;

/**
 * `POST /v1/classify_text`: the arguments of GLiNER2's `classify_text` and `batch_classify_text`
 * (fastino-ai/GLiNER2 v2.1.0, gliner2/inference/runtime.py), with this runner's explicit model, backend and
 * precision. Send `text` for one result or `texts` for a list.
 */
export const ClassifyTextRequestSchema = z
  .strictObject({
    request_id: z
      .guid()
      .transform((value) => value.toLowerCase())
      .default(() => globalThis.crypto.randomUUID()),
    model: nonBlankString,
    backend: BackendNameSchema,
    precision: PrecisionSchema,
    text: nonBlankString.nullable().optional(),
    texts: z.array(nonBlankString).min(1).max(256).nullable().optional(),
    tasks: nonEmptyRecord(JsonValueSchema),
    threshold: z.number().min(0).max(1).default(0.5),
    include_confidence: z.boolean().default(false),
  })
  .superRefine((request, context) => {
    if ((request.text ?? null) === null === ((request.texts ?? null) === null)) {
      context.addIssue({ code: "custom", message: SHAPE_MESSAGES.textOrTexts });
    }
    for (const [name, task] of Object.entries(request.tasks)) {
      const problem = gliner2TaskProblem(task);
      if (problem !== null) {
        context.addIssue({ code: "custom", path: ["tasks", name], message: problem });
      }
    }
  });
export type ClassifyTextRequest = z.output<typeof ClassifyTextRequestSchema>;

/** Validation messages of the translated shapes; src/gliner_runner/shapes.py uses the same text. */
export const SHAPE_MESSAGES = {
  questionType: "unknown question type; use noul, choice, score or multi_select",
  box:
    "box questions find things in images, and the GLiNER2.5 Decide models read text only; " +
    "use an image model for box questions",
  assets: "images are not supported: the GLiNER2.5 Decide models read text only; send the text as state",
  state:
    "state must be one string here: the GLiNER2.5 Decide models read plain text; " +
    'write structured state out as text, for example "subject: ...\\nmessage: ..."',
  blankState: "state must not be blank",
  instructions: "instructions must be one non-blank string or null; write structured instructions out as text",
  threshold: "threshold applies only to multi_select questions",
  noulCriteria: "noul criteria must be an object with only 'true' and 'false', each a non-blank string or null",
  choiceCriteria: "criteria must map each option name to a non-blank description string or null",
  choiceCount: "a choice needs at least two options",
  multiCount: "a multi_select needs at least one option",
  scoreCriteria: "score criteria must be a list of at least two level descriptions, each a non-blank string",
  scoreUnique: "score levels must be unique",
  textOrTexts: "send exactly one of text or texts",
  gliner2Task:
    "a task must be a list of label names, or an object with labels and any of multi_label, " +
    "cls_threshold, top_k, class_act and prompt",
  gliner2Labels:
    "labels must be a non-empty list of unique names, or an object mapping each name to a description or null",
  gliner2Examples:
    "few-shot examples are not part of this runner's protocol; leave examples out " +
    "and describe the labels instead",
  gliner2Threshold:
    "cls_threshold must be greater than 0 and less than 1 here; " +
    "to keep the best labels whatever their score, use top_k",
  gliner2TopK: "top_k must be an integer of at least 1",
  gliner2Activation: "class_act must be auto, sigmoid or softmax",
  gliner2SoftmaxMulti:
    "multi_label with class_act softmax picks labels differently in this runner; use class_act auto or sigmoid",
  gliner2MultiLabel: "multi_label must be true or false",
  gliner2Prompt: "prompt must be a non-blank string",
} as const;

const isJsonObject = (value: unknown): value is Record<string, unknown> =>
  typeof value === "object" && value !== null && !Array.isArray(value);

const isNonBlank = (value: unknown): value is string => typeof value === "string" && !isWhitespaceOnly(value);

const isDescription = (value: unknown): boolean => value === null || isNonBlank(value);

/** The reason `state` cannot be translated, or `null`. */
export function stateProblem(state: unknown): string | null {
  if (typeof state !== "string") return SHAPE_MESSAGES.state;
  return isWhitespaceOnly(state) ? SHAPE_MESSAGES.blankState : null;
}

/** The reason a jev or Decision-1 question cannot be translated, or `null`. */
export function questionProblem(question: {
  type: string;
  instructions?: unknown;
  criteria?: unknown;
  threshold?: number | null | undefined;
}): string | null {
  const { type, criteria } = question;
  if (type === "box") return SHAPE_MESSAGES.box;
  if (!["noul", "choice", "score", "multi_select"].includes(type)) return SHAPE_MESSAGES.questionType;
  const instructions = question.instructions ?? null;
  if (instructions !== null && !isNonBlank(instructions)) return SHAPE_MESSAGES.instructions;
  if ((question.threshold ?? null) !== null && type !== "multi_select") return SHAPE_MESSAGES.threshold;
  if (type === "noul") {
    if (criteria === undefined || criteria === null) return null;
    if (!isJsonObject(criteria)) return SHAPE_MESSAGES.noulCriteria;
    const keys = Object.keys(criteria);
    const valid = keys.every((key) => (key === "true" || key === "false") && isDescription(criteria[key]));
    return valid ? null : SHAPE_MESSAGES.noulCriteria;
  }
  if (type === "score") {
    if (!Array.isArray(criteria) || criteria.length < 2 || !criteria.every(isNonBlank)) {
      return SHAPE_MESSAGES.scoreCriteria;
    }
    return new Set(criteria).size === criteria.length ? null : SHAPE_MESSAGES.scoreUnique;
  }
  if (!isJsonObject(criteria) || !Object.values(criteria).every(isDescription)) {
    return SHAPE_MESSAGES.choiceCriteria;
  }
  const count = Object.keys(criteria).length;
  if (type === "choice" && count < 2) return SHAPE_MESSAGES.choiceCount;
  if (type === "multi_select" && count < 1) return SHAPE_MESSAGES.multiCount;
  return null;
}

const GLINER2_TASK_KEYS = ["labels", "multi_label", "cls_threshold", "top_k", "class_act", "prompt"];

const isLabelList = (labels: readonly unknown[]): boolean =>
  labels.length > 0 && labels.every((label) => typeof label === "string") && new Set(labels).size === labels.length;

/** The reason a GLiNER2 `classify_text` task cannot be translated, or `null`. */
export function gliner2TaskProblem(task: unknown): string | null {
  if (Array.isArray(task)) {
    return isLabelList(task) ? null : SHAPE_MESSAGES.gliner2Labels;
  }
  if (!isJsonObject(task) || !Object.hasOwn(task, "labels")) return SHAPE_MESSAGES.gliner2Task;
  if (Object.hasOwn(task, "examples")) return SHAPE_MESSAGES.gliner2Examples;
  if (!Object.keys(task).every((key) => GLINER2_TASK_KEYS.includes(key))) return SHAPE_MESSAGES.gliner2Task;
  const labels = task["labels"];
  const labelsValid = Array.isArray(labels)
    ? isLabelList(labels)
    : isJsonObject(labels) && Object.keys(labels).length > 0 && Object.values(labels).every(isDescription);
  if (!labelsValid) return SHAPE_MESSAGES.gliner2Labels;
  const multi = task["multi_label"] ?? false;
  if (typeof multi !== "boolean") return SHAPE_MESSAGES.gliner2MultiLabel;
  const threshold = task["cls_threshold"];
  if (threshold !== undefined && !(typeof threshold === "number" && threshold > 0 && threshold < 1)) {
    return SHAPE_MESSAGES.gliner2Threshold;
  }
  const topK = task["top_k"];
  if (topK !== undefined && topK !== null && !(typeof topK === "number" && Number.isInteger(topK) && topK >= 1)) {
    return SHAPE_MESSAGES.gliner2TopK;
  }
  const activation = task["class_act"] ?? "auto";
  if (typeof activation !== "string" || !["auto", "sigmoid", "softmax"].includes(activation)) {
    return SHAPE_MESSAGES.gliner2Activation;
  }
  if (multi && activation === "softmax") return SHAPE_MESSAGES.gliner2SoftmaxMulti;
  const prompt = task["prompt"];
  if (prompt !== undefined && prompt !== null && !isNonBlank(prompt)) return SHAPE_MESSAGES.gliner2Prompt;
  return null;
}

export const DownloadRequestSchema = z.strictObject({
  approved: z.boolean(),
  destination: z.string().min(1),
  revision: z
    .string()
    .regex(/^[a-f0-9]{40}$/)
    .nullable()
    .default(null),
});
export type DownloadRequest = z.output<typeof DownloadRequestSchema>;

export interface PrecisionProfile {
  device: string;
  precision: Precision;
}

/** Wire shape of `GET /v1/capabilities` elements; sets are serialized as arrays. */
export interface BackendCapabilities {
  backend: BackendName;
  operations: readonly Operation[];
  precisions: readonly Precision[];
  devices: readonly string[];
  dynamic_batching: boolean;
  precision_profiles: readonly PrecisionProfile[];
  validated_models: readonly string[];
}

export function supports(
  capabilities: BackendCapabilities,
  request: Pick<InferenceRequest, "operation" | "precision" | "model">,
  device: string,
): boolean {
  return (
    capabilities.operations.includes(request.operation) &&
    capabilities.precisions.includes(request.precision) &&
    capabilities.devices.includes(device) &&
    (capabilities.precision_profiles.length === 0 ||
      capabilities.precision_profiles.some(
        (profile) => profile.device === device && profile.precision === request.precision,
      )) &&
    (capabilities.validated_models.length === 0 ||
      capabilities.validated_models.includes(request.model))
  );
}

export interface Timing {
  queue_ms: number;
  inference_ms: number;
}

/** The only camelCase keys on the wire. */
export interface InferenceUsage {
  inputTokens: number;
  outputTokens: 0;
}

export interface InferenceResponse {
  request_id: string;
  model: string;
  backend: BackendName;
  precision: Precision;
  output: BackendOutput | null;
  error: { code: string; message: string; retryable: boolean } | null;
  timing: Timing;
  usage: InferenceUsage;
}

export interface HealthResponse {
  status: "ok" | "degraded";
  accepting_requests: boolean;
}

export interface ProfileGuidance {
  estimated_minimum_memory_bytes: number;
  memory_basis: string;
  baseline_profile: string;
  observed_max_abs_score_drift: number | null;
  drift_order_of_magnitude: string | null;
  characterization_scope: string;
}

export interface ModelInventoryItem {
  model_id: string;
  repository: string;
  family: "GLiNER2.5-Decide";
  description: string;
  pinned_revision: string;
  discovered_revision: string;
  parameter_count: number;
  local_revisions: readonly string[];
  available_locally: boolean;
  loaded: boolean;
  loaded_profile: PrecisionProfile | null;
  supported_profiles: readonly PrecisionProfile[];
  default_profile: PrecisionProfile;
  profile_guidance: Record<string, ProfileGuidance>;
  download_destination: string;
  download_required: boolean;
}

export interface ModelInventoryResponse {
  default_model: string;
  device: string;
  idle_ttl_seconds: number;
  memory_limit_bytes: number;
  discovered_models: readonly string[];
  models: readonly ModelInventoryItem[];
}

export interface RefreshResult {
  refreshed_at: number;
  models_checked: number;
  changed_models: readonly string[];
  discovered_models: readonly string[];
}

export interface DownloadJob {
  job_id: string;
  model_id: string;
  revision: string;
  destination: string;
  state: "queued" | "running" | "succeeded" | "failed";
  error: string | null;
}
