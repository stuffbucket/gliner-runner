export type Backend = "pytorch" | "mlx" | "onnx";
export type Precision = "fp32" | "fp16" | "bf16" | "int8" | "int4";
export type Operation =
  | "classify"
  | "extract_entities"
  | "extract_relations"
  | "extract_structured";

export interface ClassificationTask {
  labels: string[];
  min_labels?: number;
  max_labels?: number | null;
  ordered?: boolean;
  threshold?: number;
  activation?: "auto" | "sigmoid" | "softmax";
  temperature?: number;
  default?: string | null;
  instruction?: string | null;
}

export interface ClassificationSchema {
  kind: "classification";
  tasks: Record<string, ClassificationTask>;
  constraints?: Array<Record<string, JsonValue>>;
}

export type JsonValue =
  | string
  | number
  | boolean
  | null
  | JsonValue[]
  | { [key: string]: JsonValue };

export interface InferenceOptions {
  threshold?: number;
  output_format?: "native";
  include_confidence?: boolean;
}

export interface InferenceRequest {
  request_id?: string;
  model: string;
  backend: Backend;
  precision: Precision;
  operation: Operation;
  text: string;
  schema: ClassificationSchema | Record<string, JsonValue>;
  options?: InferenceOptions;
}

export interface Timing {
  queue_ms: number;
  inference_ms: number;
}

export interface InferenceResponse<TResult extends JsonValue = JsonValue> {
  request_id: string;
  model: string;
  backend: Backend;
  precision: Precision;
  output: TResult | null;
  error: { code: string; message: string; retryable: boolean } | null;
  timing: Timing;
}

export interface PrecisionProfile {
  device: string;
  precision: Precision;
}

export interface BackendCapabilities {
  backend: Backend;
  operations: Operation[];
  precisions: Precision[];
  devices: string[];
  dynamic_batching: boolean;
  precision_profiles: PrecisionProfile[];
  validated_models: string[];
}

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
    return this.request<BackendCapabilities[]>("/v1/capabilities", {
      method: "GET",
      ...(options.signal === undefined ? {} : { signal: options.signal }),
    });
  }

  async infer<TResult extends JsonValue = JsonValue>(
    request: InferenceRequest,
    options: { signal?: AbortSignal } = {},
  ): Promise<InferenceResponse<TResult>> {
    return this.request<InferenceResponse<TResult>>("/v1/infer", {
      method: "POST",
      body: JSON.stringify(request),
      ...(options.signal === undefined ? {} : { signal: options.signal }),
    });
  }

  async batch<TResult extends JsonValue = JsonValue>(
    requests: InferenceRequest[],
    options: { signal?: AbortSignal } = {},
  ): Promise<{ responses: Array<InferenceResponse<TResult>> }> {
    return this.request<{ responses: Array<InferenceResponse<TResult>> }>("/v1/batch", {
      method: "POST",
      body: JSON.stringify({ requests }),
      ...(options.signal === undefined ? {} : { signal: options.signal }),
    });
  }

  private async request<T>(path: string, init: RequestInit): Promise<T> {
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
    return body as T;
  }
}

function errorFromResponse(status: number, body: unknown): GlinerApiError {
  if (typeof body === "object" && body !== null && "detail" in body) {
    const detail = body.detail;
    if (
      typeof detail === "object" &&
      detail !== null &&
      "code" in detail &&
      typeof detail.code === "string" &&
      "message" in detail &&
      typeof detail.message === "string"
    ) {
      return new GlinerApiError(
        status,
        detail.code,
        detail.message,
        "retryable" in detail && detail.retryable === true,
      );
    }
  }
  return new GlinerApiError(
    status,
    "invalid_error_response",
    `The server returned HTTP ${status} without a valid error envelope`,
  );
}
