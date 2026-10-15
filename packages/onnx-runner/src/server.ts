import {
  createServer as createHttpServer,
  type IncomingMessage,
  type Server,
  type ServerResponse,
} from "node:http";
import type { AddressInfo } from "node:net";
import { setTimeout as sleep } from "node:timers/promises";

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
import { resolveSettings, type Settings } from "./settings.js";
import { recordTaskOrder } from "./task-order.js";
import { stringifyWire } from "./wire.js";
import { parseJsonBody, parseValue, type ValidationErrorItem } from "./validation.js";

export interface ServerOptions {
  catalog: ArtifactCatalog;
  /** Registered backends; a request naming any other backend is rejected without fallback. */
  backends: readonly BackendFactory[];
  settings?: Partial<Settings>;
  /**
   * Fetches and prepares models for `POST /v1/models/{id}/downloads`. Without one the server validates download
   * requests and then declines them with 409 `unsupported_configuration`.
   */
  provisioner?: ModelProvisioner;
  /** Overrides `settings.autoDownload`: inference for a missing, installable model waits for a shared download job. */
  autoDownload?: boolean;
  /** Diagnostic sink; never receives request text, schemas or outputs. */
  log?: (message: string) => void;
}

export interface RunnerServer {
  readonly runtime: Runtime;
  readonly http: Server;
  listen(): Promise<AddressInfo>;
  /** Stops listening, lets in-flight requests finish, then unloads the model. */
  close(): Promise<void>;
}

interface Reply {
  status: number;
  body: string;
  contentType: string;
  headers?: Record<string, string> | undefined;
}

interface Context {
  request: IncomingMessage;
  params: string[];
  signal: AbortSignal;
  readBody(): Promise<string>;
}

type Handler = (context: Context) => Promise<Reply> | Reply;

interface Route {
  method: string;
  pattern: RegExp;
  handler: Handler;
}

const json = (status: number, value: unknown, headers?: Record<string, string>): Reply => ({
  status,
  body: stringifyWire(value),
  contentType: "application/json",
  headers,
});

const failure = (error: HttpError): Reply => json(error.status, { detail: error.detail }, error.headers);

const validationFailure = (errors: readonly ValidationErrorItem[]): Reply =>
  json(422, { detail: errors });

export function createServer(options: ServerOptions): RunnerServer {
  const settings = resolveSettings(options.settings);
  if (!Number.isSafeInteger(settings.maxBodyBytes) || settings.maxBodyBytes < 1) {
    throw new RangeError(`maxBodyBytes must be a positive integer, got ${settings.maxBodyBytes}`);
  }
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
    route("GET", "/healthz", (): Reply => {
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

  const exchanges = new Set<Promise<void>>();
  const http = createHttpServer((request, response) => {
    void serve(request, response);
  });

  async function serve(request: IncomingMessage, response: ServerResponse): Promise<void> {
    const controller = new AbortController();
    const finished = new Promise<void>((resolve) => {
      response.once("close", () => {
        if (!response.writableFinished) {
          controller.abort();
        }
        exchanges.delete(finished);
        resolve();
      });
    });
    exchanges.add(finished);
    const origin = request.headers.origin;
    const cors = origin !== undefined && settings.corsOrigins.includes(origin);
    let reply: Reply;
    try {
      reply =
        cors && request.method === "OPTIONS"
          ? {
              status: 204,
              body: "",
              contentType: "text/plain",
              headers: {
                "access-control-allow-methods": "GET, POST, OPTIONS",
                "access-control-allow-headers": "content-type",
                "access-control-max-age": "600",
              },
            }
          : await dispatch(request, controller.signal);
    } catch (error) {
      if (controller.signal.aborted || error instanceof RequestCancelledError) {
        response.destroy();
        return;
      }
      const mapped = domainHttpError(error);
      if (mapped === undefined) {
        log(`unhandled error: ${error instanceof Error ? (error.stack ?? error.message) : String(error)}`);
      }
      reply = failure(mapped ?? internalError());
    }
    if (response.destroyed) {
      return;
    }
    const body = Buffer.from(reply.body);
    response.writeHead(reply.status, {
      "content-type": reply.contentType,
      "content-length": body.length,
      ...reply.headers,
      ...(cors ? { "access-control-allow-origin": origin, vary: "Origin" } : {}),
    });
    response.end(body);
  }

  async function dispatch(request: IncomingMessage, signal: AbortSignal): Promise<Reply> {
    const pathname = new URL(request.url ?? "/", "http://localhost").pathname;
    const method = request.method ?? "GET";
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
      return candidate.handler({ request, params, signal, readBody: () => readBody(request, settings.maxBodyBytes) });
    }
    return allowed.size > 0
      ? json(405, { detail: "Method Not Allowed" }, { allow: [...allowed].sort().join(", ") })
      : json(404, { detail: "Not Found" });
  }

  let closing: Promise<void> | undefined;
  return {
    runtime,
    http,
    listen: () =>
      new Promise<AddressInfo>((resolve, reject) => {
        http.once("error", reject);
        http.listen({ host: settings.host, port: settings.port }, () => {
          http.off("error", reject);
          resolve(http.address() as AddressInfo);
        });
      }),
    close: () => (closing ??= shutdown()),
  };

  async function shutdown(): Promise<void> {
    // Waiting inference requests fail fast instead of holding shutdown open for a download that is being cancelled.
    jobs?.abort();
    if (http.listening) {
      const stopped = new Promise<void>((resolve) => http.close(() => resolve()));
      const drained = (async () => {
        while (exchanges.size > 0) {
          await Promise.allSettled([...exchanges]);
        }
      })();
      await Promise.race([drained, sleep(settings.shutdownTimeoutMs / 2, undefined, { ref: false })]);
      http.closeAllConnections();
      await stopped;
    }
    await jobs?.close();
    await runtime.close();
  }
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
 * Reads the whole body, failing with `RequestTooLargeError` as soon as it is known to exceed `limit` bytes (from
 * `content-length` or while streaming). The rest of an oversize body is discarded without being buffered; the
 * stream is not destroyed, because that would also tear down the socket the 413 reply must still be written to.
 */
function readBody(request: IncomingMessage, limit: number): Promise<string> {
  return new Promise<string>((resolve, reject) => {
    const declared = Number(request.headers["content-length"]);
    if (Number.isFinite(declared) && declared > limit) {
      request.resume();
      reject(new RequestTooLargeError(limit));
      return;
    }
    const chunks: Buffer[] = [];
    let size = 0;
    let settled = false;
    const fail = (error: unknown): void => {
      if (!settled) {
        settled = true;
        chunks.length = 0;
        reject(error);
      }
    };
    request.on("data", (chunk: Buffer) => {
      if (settled) {
        return;
      }
      size += chunk.length;
      if (size > limit) {
        fail(new RequestTooLargeError(limit));
      } else {
        chunks.push(chunk);
      }
    });
    request.once("end", () => {
      if (!settled) {
        settled = true;
        resolve(Buffer.concat(chunks).toString("utf8"));
      }
    });
    request.once("error", fail);
    request.once("close", () => {
      if (!request.complete) {
        fail(new RequestCancelledError("request body was not fully received"));
      }
    });
  });
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
