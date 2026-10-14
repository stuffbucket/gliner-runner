import { randomUUID } from "node:crypto";

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
      .default(() => randomUUID()),
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
