import { z } from "zod";

export const BackendSchema = z.enum(["pytorch", "mlx", "onnx"]);
export type Backend = z.infer<typeof BackendSchema>;

export const PrecisionSchema = z.enum(["fp32", "fp16", "bf16", "int8", "int4"]);
export type Precision = z.infer<typeof PrecisionSchema>;

export const OperationSchema = z.enum([
  "classify",
  "extract_entities",
  "extract_relations",
  "extract_structured",
]);
export type Operation = z.infer<typeof OperationSchema>;

export type JsonValue =
  | string
  | number
  | boolean
  | null
  | JsonValue[]
  | { [key: string]: JsonValue };

export const JsonValueSchema: z.ZodType<JsonValue> = z.lazy(() =>
  z.union([
    z.string(),
    z.number(),
    z.boolean(),
    z.null(),
    z.array(JsonValueSchema),
    z.record(z.string(), JsonValueSchema),
  ]),
);

export const ClassificationTaskSchema = z
  .object({
    labels: z
      .array(z.string())
      .min(1)
      .refine((labels) => new Set(labels).size === labels.length, {
        message: "classification labels must be unique",
      }),
    min_labels: z.number().int().nonnegative().optional(),
    max_labels: z.number().int().nonnegative().nullable().optional(),
    ordered: z.boolean().optional(),
    threshold: z.number().gt(0).lt(1).optional(),
    activation: z.enum(["auto", "sigmoid", "softmax"]).optional(),
    temperature: z.number().positive().optional(),
    default: z.string().nullable().optional(),
    instruction: z.string().nullable().optional(),
  })
  .strict();
export type ClassificationTask = z.infer<typeof ClassificationTaskSchema>;

export const ClassificationSchema = z
  .object({
    kind: z.literal("classification"),
    tasks: z
      .record(z.string(), ClassificationTaskSchema)
      .refine((tasks) => Object.keys(tasks).length > 0, {
        message: "at least one classification task is required",
      }),
    constraints: z.array(z.record(z.string(), JsonValueSchema)).optional(),
  })
  .strict();

export const EntitySchema = z
  .object({
    kind: z.literal("entities"),
    labels: z
      .array(z.string())
      .min(1)
      .refine((labels) => new Set(labels).size === labels.length, {
        message: "entity labels must be unique",
      }),
  })
  .strict();

export const RelationSchema = z
  .object({
    kind: z.literal("relations"),
    relation_types: z.array(z.string()).min(1),
  })
  .strict();

export const StructuredSchema = z
  .object({
    kind: z.literal("structured"),
    schema: z
      .record(z.string(), JsonValueSchema)
      .refine((schema) => Object.keys(schema).length > 0, {
        message: "structured schema must not be empty",
      }),
  })
  .strict();

export const InferenceSchema = z.discriminatedUnion("kind", [
  ClassificationSchema,
  EntitySchema,
  RelationSchema,
  StructuredSchema,
]);
export type InferenceSchema = z.infer<typeof InferenceSchema>;

export const InferenceOptionsSchema = z
  .object({
    threshold: z.number().min(0).max(1).optional(),
    output_format: z.literal("native").optional(),
    include_confidence: z.boolean().optional(),
  })
  .strict();
export type InferenceOptions = z.infer<typeof InferenceOptionsSchema>;

export const InferenceRequestSchema = z
  .object({
    request_id: z.string().uuid().optional(),
    model: z.string().refine((value) => value.trim().length > 0, {
      message: "model must not be blank",
    }),
    backend: BackendSchema,
    precision: PrecisionSchema,
    operation: OperationSchema,
    text: z.string().refine((value) => value.trim().length > 0, {
      message: "text must not be blank",
    }),
    schema: InferenceSchema,
    options: InferenceOptionsSchema.optional(),
  })
  .strict()
  .superRefine((request, context) => {
    const expectedKinds: Record<Operation, InferenceSchema["kind"]> = {
      classify: "classification",
      extract_entities: "entities",
      extract_relations: "relations",
      extract_structured: "structured",
    };
    if (request.schema.kind !== expectedKinds[request.operation]) {
      context.addIssue({
        code: "custom",
        path: ["schema", "kind"],
        message: `${request.operation} requires a ${expectedKinds[request.operation]} schema`,
      });
    }
  });
export type InferenceRequest = z.infer<typeof InferenceRequestSchema>;

export const TimingSchema = z
  .object({
    queue_ms: z.number().nonnegative(),
    inference_ms: z.number().nonnegative(),
  })
  .strict();
export type Timing = z.infer<typeof TimingSchema>;

export const InferenceErrorSchema = z
  .object({
    code: z.string(),
    message: z.string(),
    retryable: z.boolean(),
  })
  .strict();

export const InferenceResponseSchema = z
  .object({
    request_id: z.string().uuid(),
    model: z.string(),
    backend: BackendSchema,
    precision: PrecisionSchema,
    output: JsonValueSchema.nullable(),
    error: InferenceErrorSchema.nullable(),
    timing: TimingSchema,
  })
  .strict()
  .refine((value) => (value.output === null) !== (value.error === null), {
    message: "exactly one of output or error must be present",
  });
export type InferenceResponse = z.infer<typeof InferenceResponseSchema>;

export const PrecisionProfileSchema = z
  .object({
    device: z.string(),
    precision: PrecisionSchema,
  })
  .strict();
export type PrecisionProfile = z.infer<typeof PrecisionProfileSchema>;

export const BackendCapabilitiesSchema = z
  .object({
    backend: BackendSchema,
    operations: z.array(OperationSchema),
    precisions: z.array(PrecisionSchema),
    devices: z.array(z.string()),
    dynamic_batching: z.boolean(),
    precision_profiles: z.array(PrecisionProfileSchema),
    validated_models: z.array(z.string()),
  })
  .strict();
export type BackendCapabilities = z.infer<typeof BackendCapabilitiesSchema>;

const BatchResponseSchema = z
  .object({ responses: z.array(InferenceResponseSchema) })
  .strict();

const ApiErrorDetailSchema = z
  .object({
    code: z.string(),
    message: z.string(),
    retryable: z.boolean(),
  })
  .passthrough();

export interface GlinerClientOptions {
  baseUrl?: string;
  fetch?: typeof globalThis.fetch;
  headers?: HeadersInit;
}

export class GlinerApiError extends Error {
  readonly status: number;
  readonly code: string;
  readonly retryable: boolean;

  constructor(status: number, code: string, message: string, retryable = false) {
    super(message);
    this.name = "GlinerApiError";
    this.status = status;
    this.code = code;
    this.retryable = retryable;
  }
}

export class GlinerClient {
  readonly baseUrl: string;
  readonly headers: HeadersInit;
  private readonly fetchImpl: typeof globalThis.fetch;

  constructor(options: GlinerClientOptions = {}) {
    this.baseUrl = (options.baseUrl ?? "http://127.0.0.1:8090").replace(/\/+$/, "");
    this.fetchImpl = options.fetch ?? globalThis.fetch;
    this.headers = options.headers ?? {};
    if (typeof this.fetchImpl !== "function") {
      throw new TypeError("A Fetch API implementation is required");
    }
  }

  async capabilities(options: { signal?: AbortSignal } = {}): Promise<BackendCapabilities[]> {
    const body = await this.request("/v1/capabilities", {
      method: "GET",
      ...(options.signal === undefined ? {} : { signal: options.signal }),
    });
    return z.array(BackendCapabilitiesSchema).parse(body);
  }

  async infer(
    request: InferenceRequest,
    options: { signal?: AbortSignal } = {},
  ): Promise<InferenceResponse> {
    const validated = InferenceRequestSchema.parse(request);
    const body = await this.request("/v1/infer", {
      method: "POST",
      body: JSON.stringify(validated),
      ...(options.signal === undefined ? {} : { signal: options.signal }),
    });
    return InferenceResponseSchema.parse(body);
  }

  async batch(
    requests: InferenceRequest[],
    options: { signal?: AbortSignal } = {},
  ): Promise<{ responses: InferenceResponse[] }> {
    const validated = z.array(InferenceRequestSchema).min(1).max(256).parse(requests);
    const body = await this.request("/v1/batch", {
      method: "POST",
      body: JSON.stringify({ requests: validated }),
      ...(options.signal === undefined ? {} : { signal: options.signal }),
    });
    return BatchResponseSchema.parse(body);
  }

  private async request(path: string, init: RequestInit): Promise<unknown> {
    const headers = new Headers(this.headers);
    headers.set("Accept", "application/json");
    if (init.body !== undefined) {
      headers.set("Content-Type", "application/json");
    }
    const response = await this.fetchImpl(`${this.baseUrl}${path}`, {
      ...init,
      headers,
    });
    const body: unknown = await response.json().catch(() => undefined);
    if (!response.ok) {
      throw errorFromResponse(response.status, body);
    }
    if (body === undefined) {
      throw new GlinerApiError(
        response.status,
        "invalid_response",
        "The server returned a non-JSON response",
      );
    }
    return body;
  }
}

function errorFromResponse(status: number, body: unknown): GlinerApiError {
  if (typeof body === "object" && body !== null && "detail" in body) {
    const parsed = ApiErrorDetailSchema.safeParse(body.detail);
    if (parsed.success) {
      return new GlinerApiError(
        status,
        parsed.data.code,
        parsed.data.message,
        parsed.data.retryable,
      );
    }
  }
  return new GlinerApiError(
    status,
    "invalid_error_response",
    `The server returned HTTP ${status} without a valid error envelope`,
  );
}
