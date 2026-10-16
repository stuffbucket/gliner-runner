import { z } from "zod";

import { BackendRegistry, type BackendFactory } from "./backend.js";
import type { ArtifactCatalog } from "./catalog.js";
import {
  BatchRequestSchema,
  DownloadRequestSchema,
  InferenceRequestSchema,
  type HealthResponse,
  type InferenceResponse,
  type RefreshResult,
} from "./contracts.js";
import { DownloadJobManager, type ModelProvisioner } from "./downloads.js";
import {
  ModelDownloadRequiredError,
  RequestCancelledError,
  RequestTooLargeError,
  catalogRefreshFailed,
  domainHttpError,
  envelope,
  internalError,
  type HttpError,
} from "./errors.js";
import { buildModelsResponse } from "./inventory.js";
import { buildOpenApiDocument } from "./openapi.js";
import { Runtime } from "./runtime.js";
import type { Settings } from "./settings.js";
import { recordTaskOrder } from "./task-order.js";
import { stringifyWire } from "./wire.js";
import { parseJsonBody, parseValue, type ValidationErrorItem } from "./validation.js";

/** The settings the request handler uses; the HTTP server adds host, port, CORS and shutdown timing. */
export type HandlerSettings = Pick<
  Settings,
  | "device"
  | "queueCapacity"
  | "maxBatchSize"
  | "batchWindowMs"
  | "modelIdleTtlSeconds"
  | "memoryLimitBytes"
  | "maxBodyBytes"
  | "autoDownload"
>;

export interface RunnerHandlerOptions {
  catalog: ArtifactCatalog;
  /** Registered backends; a request naming any other backend is rejected without fallback. */
  backends: readonly BackendFactory[];
  settings: HandlerSettings;
  /**
   * Fetches and prepares models for `POST /v1/models/{id}/downloads`. Without one, download requests are validated
   * and then declined with 409 `unsupported_configuration`.
   */
  provisioner?: ModelProvisioner | undefined;
  /** Overrides `settings.autoDownload`: inference for a missing, installable model waits for a shared download job. */
  autoDownload?: boolean | undefined;
  /** Diagnostic sink; never receives request text, schemas or outputs. */
  log?: ((message: string) => void) | undefined;
}

export interface RunnerRequest {
  method: string;
  /** Request target: a path, optionally with a query string. */
  path: string;
  /**
   * The body as text, or a reader that may reject with `RequestTooLargeError` (the HTTP server streams it and
   * stops early). A text body larger than `settings.maxBodyBytes` UTF-8 bytes is answered with 413.
   */
  body?: string | (() => Promise<string>) | undefined;
  /** Aborted when the client goes away; the request is then cancelled. */
  signal?: AbortSignal | undefined;
}

export interface RunnerReply {
  status: number;
  body: string;
  contentType: string;
  headers?: Record<string, string> | undefined;
}

/**
 * The runner's `/v1` API, `/healthz`, `/metrics` and `/openapi.json` without a transport: the Node HTTP server
 * (`server.ts`) and the browser Web Worker both delegate to it.
 */
export interface RunnerHandler {
  readonly runtime: Runtime;
  /** Rejects only when the request was cancelled (its signal aborted, or a `RequestCancelledError`). */
  handle(request: RunnerRequest): Promise<RunnerReply>;
  /** Cancels running download jobs, so requests waiting on them fail fast. */
  abortJobs(): void;
  /** Waits for download jobs to stop, then unloads the model. */
  close(): Promise<void>;
}

interface Context {
  params: string[];
  signal: AbortSignal;
  readBody(): Promise<string>;
}

type Handler = (context: Context) => Promise<RunnerReply> | RunnerReply;

interface Route {
  method: string;
  pattern: RegExp;
  handler: Handler;
}

const json = (status: number, value: unknown, headers?: Record<string, string>): RunnerReply => ({
  status,
  body: stringifyWire(value),
  contentType: "application/json",
  headers,
});

const failure = (error: HttpError): RunnerReply => json(error.status, { detail: error.detail }, error.headers);

const validationFailure = (errors: readonly ValidationErrorItem[]): RunnerReply =>
  json(422, { detail: errors });

export function createRunnerHandler(options: RunnerHandlerOptions): RunnerHandler {
  const { settings } = options;
  const log = options.log ?? (() => undefined);
  const catalog = options.catalog;
  const runtime = new Runtime({
    catalog,
    backends: new BackendRegistry(options.backends),
    config: {
      device: settings.device,
      queueCapacity: settings.queueCapacity,
      maxBatchSize: settings.maxBatchSize,
      batchWindowMs: settings.batchWindowMs,
      modelIdleTtlSeconds: settings.modelIdleTtlSeconds,
      memoryLimitBytes: settings.memoryLimitBytes,
    },
  });
  const openApi = JSON.stringify(buildOpenApiDocument());
  const jobs =
    options.provisioner === undefined
      ? undefined
      : new DownloadJobManager({ catalog, provisioner: options.provisioner, log });
  const autoDownload = options.autoDownload ?? settings.autoDownload;

  /** Inference never downloads implicitly unless the operator enabled auto-download. */
  async function infer(request: z.output<typeof InferenceRequestSchema>, signal: AbortSignal): Promise<InferenceResponse> {
    try {
      return await runtime.infer(request, signal);
    } catch (error) {
      if (jobs === undefined || !autoDownload || !(error instanceof ModelDownloadRequiredError)) {
        throw error;
      }
      if (!(await jobs.installOnDemand(error.model, error.revision, signal))) {
        throw error;
      }
      return runtime.infer(request, signal);
    }
  }

  const routes: Route[] = [
    route("GET", "/healthz", (): RunnerReply => {
      const health: HealthResponse = {
        status: runtime.accepting ? "ok" : "degraded",
        accepting_requests: runtime.accepting,
      };
      return json(200, health);
    }),
    route("GET", "/v1/capabilities", () => json(200, runtime.capabilities())),
    route("GET", "/v1/models", async () => json(200, await buildModelsResponse(runtime, catalog))),
    route("POST", "/v1/models/refresh", async () => {
      try {
        const result = await catalog.refresh();
        const body: RefreshResult = {
          refreshed_at: result.refreshedAt,
          models_checked: result.modelsChecked,
          changed_models: result.changedModels,
          discovered_models: result.discoveredModels,
        };
        return json(200, body);
      } catch (error) {
        return failure(catalogRefreshFailed(error));
      }
    }),
    route("POST", "/v1/models/:model_id/downloads", async ({ params, readBody }) => {
      const parsed = parseJsonBody(DownloadRequestSchema, await readBody());
      if (!parsed.success) {
        return validationFailure(parsed.errors);
      }
      const modelId = params[0] as string;
      if (jobs !== undefined) {
        return json(202, await jobs.create(modelId, parsed.data));
      }
      if (!parsed.data.approved) {
        return failure(
          envelope(422, "invalid_download_request", "approved must be true; downloads never start implicitly", false),
        );
      }
      if (catalog.canonicalModelId(modelId) === undefined) {
        return failure(
          envelope(422, "invalid_download_request", `unknown GLiNER2.5 model '${modelId}'`, false),
        );
      }
      return failure(
        envelope(
          409,
          "unsupported_configuration",
          "ONNX model artifacts are installed with the model preparation tool; " +
            "this server does not download models",
          false,
        ),
      );
    }),
    route("GET", "/v1/model-downloads/:job_id", ({ params }) => {
      const parsed = parseValue(
        z.strictObject({ job_id: z.guid() }),
        { job_id: params[0] },
        "path",
      );
      if (!parsed.success) {
        return validationFailure(parsed.errors);
      }
      if (jobs !== undefined) {
        return json(200, jobs.get(parsed.data.job_id));
      }
      return failure(
        envelope(
          404,
          "download_job_not_found",
          `'unknown download job ${parsed.data.job_id.toLowerCase()}'`,
          false,
        ),
      );
    }),
    route("POST", "/v1/infer", async ({ readBody, signal }) => {
      const body = await readBody();
      const parsed = parseJsonBody(InferenceRequestSchema, body);
      if (!parsed.success) {
        return validationFailure(parsed.errors);
      }
      recordTaskOrder(body, [parsed.data], false);
      return json(200, await infer(parsed.data, signal));
    }),
    route("POST", "/v1/batch", async ({ readBody, signal }) => {
      const body = await readBody();
      const parsed = parseJsonBody(BatchRequestSchema, body);
      if (!parsed.success) {
        return validationFailure(parsed.errors);
      }
      recordTaskOrder(body, parsed.data.requests, true);
      const responses = await inferAll(parsed.data.requests, signal, infer);
      return json(200, { responses });
    }),
    route("GET", "/metrics", () => ({
      status: 200,
      body: runtime.metrics.prometheus(),
      contentType: "text/plain; version=0.0.4; charset=utf-8",
    })),
    route("GET", "/openapi.json", () => ({ status: 200, body: openApi, contentType: "application/json" })),
  ];

  function readBody(body: RunnerRequest["body"]): Promise<string> {
    if (typeof body === "function") {
      return body();
    }
    const text = body ?? "";
    if (new TextEncoder().encode(text).length > settings.maxBodyBytes) {
      return Promise.reject(new RequestTooLargeError(settings.maxBodyBytes));
    }
    return Promise.resolve(text);
  }

  async function dispatch(request: RunnerRequest, signal: AbortSignal): Promise<RunnerReply> {
    const pathname = new URL(request.path, "http://localhost").pathname;
    const method = request.method;
    const allowed = new Set<string>();
    for (const candidate of routes) {
      const match = candidate.pattern.exec(pathname);
      if (match === null) {
        continue;
      }
      if (candidate.method !== method) {
        allowed.add(candidate.method);
        continue;
      }
      const params = match.slice(1).map(decodeSegment);
      return candidate.handler({ params, signal, readBody: () => readBody(request.body) });
    }
    return allowed.size > 0
      ? json(405, { detail: "Method Not Allowed" }, { allow: [...allowed].sort().join(", ") })
      : json(404, { detail: "Not Found" });
  }

  return {
    runtime,
    async handle(request) {
      const signal = request.signal ?? new AbortController().signal;
      try {
        return await dispatch(request, signal);
      } catch (error) {
        if (signal.aborted || error instanceof RequestCancelledError) {
          throw error;
        }
        const mapped = domainHttpError(error);
        if (mapped === undefined) {
          log(`unhandled error: ${error instanceof Error ? (error.stack ?? error.message) : String(error)}`);
        }
        return failure(mapped ?? internalError());
      }
    },
    abortJobs: () => jobs?.abort(),
    async close() {
      await jobs?.close();
      await runtime.close();
    },
  };
}

function route(method: string, template: string, handler: Handler): Route {
  const source = template
    .split("/")
    .map((segment) => (segment.startsWith(":") ? "([^/]+)" : segment.replace(/[.*+?^${}()|[\]\\]/g, "\\$&")))
    .join("/");
  return { method, pattern: new RegExp(`^${source}$`), handler };
}

function decodeSegment(segment: string): string {
  try {
    return decodeURIComponent(segment);
  } catch {
    return segment;
  }
}

/**
 * Runs every item concurrently so they micro-batch together. The first failure fails the whole
 * batch and cancels siblings that have not been dispatched yet.
 */
async function inferAll(
  requests: readonly z.output<typeof InferenceRequestSchema>[],
  signal: AbortSignal,
  infer: (request: z.output<typeof InferenceRequestSchema>, signal: AbortSignal) => Promise<InferenceResponse>,
): Promise<InferenceResponse[]> {
  const siblings = new AbortController();
  const combined = AbortSignal.any([signal, siblings.signal]);
  try {
    return await Promise.all(requests.map((request) => infer(request, combined)));
  } catch (error) {
    siblings.abort();
    throw error;
  }
}
